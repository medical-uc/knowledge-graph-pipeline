"""Single-best-answer MCQ generation from an asserted graph.

Reads a Stage-6 IR (`graph.ir.json`), emits items with a stem, four options, the
key, and the source sentence as a verified rationale.

WHY THE IR AND NOT SPARQL
-------------------------
The graph has the facts; the IR has the facts PLUS what you need to generate
safely: `polarity` to exclude everything the guards demoted, `evidence_start` /
`evidence_end` into the normalized markdown, span `label` and `uri` for pooling
distractors by semantic type, and — critically — the uncertain relations
themselves, which are a BLOCKLIST rather than a distractor source.

THE OPEN-WORLD PROBLEM, WHICH IS THE WHOLE DIFFICULTY
-----------------------------------------------------
A knowledge graph records what was extracted, not what is true. The absence of
`hypothyroidism caused_by pituitary failure` does not make it false — it means
nothing extracted it. Draw distractors from "concepts not connected to the head"
and you will eventually ship a question with two correct answers.

Three defences, all in `safe_distractors`:

1. **Blocklist across every graph.** A distractor is rejected if (head, pred,
   distractor) appears asserted, inferred, OR flagged. `uncertain` means "not
   verified", never "verified false" — a demoted relation is the single most
   dangerous thing to offer as a wrong answer, because it is exactly the sort of
   claim the document nearly supports.
2. **No other edge to the head.** `hyperthyroidism treated_with iodine` exists,
   so iodine is unusable as a distractor in ANY hyperthyroidism question, under
   any predicate — a defensible answer is a broken question.
3. **Stems say "a cause", never "the cause".** This is what makes multiple true
   answers in the graph harmless: with defence 1 only one of them can ever reach
   the options, and an indefinite stem stays true when it does.

NEGATIVE STEMS ARE NOT SUPPORTED AND WILL NOT BE
------------------------------------------------
"All of the following EXCEPT" requires knowing that three statements are FALSE.
Under an open-world assumption the graph cannot establish that about anything.
`_NEGATIVE_RE` enforces this on the model's output, not just in the prompt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .ir import Document
from . import config
from . import stage4_postcoord as s4
from . import guards


# --- which relations make a crisp question -------------------------------- #
# Every template asks for the TAIL, so the answer position is uniform.
# Indefinite articles throughout: see defence 3 above.
STEM_TEMPLATES = {
    "caused_by":       "Which of the following is a recognised cause of {head}?",
    "complication_of": "{head_cap} can develop as a complication of which condition?",
    "treated_with":    "Which agent is used in the management of {head}?",
    "secretes":        "Which substance is secreted by {head}?",
    "converts_to":     "{head_cap} is converted into which substance?",
    "catalyzes":       "{head_cap} acts on which substrate?",
    "stores":          "Which substance is stored in {head}?",
    "located_in":      "{head_cap} is found within which structure?",
    "transports":      "Which substance is transported by {head}?",
    "part_of":         "{head_cap} is part of which structure?",
    "synthesizes":     "Which substance is synthesised by {head}?",
    "requires":        "{head_cap} requires which substance?",
    "risk_factor_for": "{head_cap} is a risk factor for which condition?",
}

# `regulates` is deliberately absent: "what does TRH regulate?" has no crisp
# single answer, and the ontology treats it as the unsigned parent of
# stimulates/inhibits anyway. Vague relations make vague questions.

MIN_OPTIONS = 4          # 1 key + 3 distractors; fewer -> skip the item entirely

_NEGATIVE_RE = re.compile(
    r"\b(not|except|never|least|incorrect|false|untrue|EXCEPT)\b", re.IGNORECASE)


@dataclass
class Item:
    item_id: str
    stem: str
    options: list[str]
    answer: str
    rationale: str                      # the source sentence, verbatim
    citation: str
    relation: str
    confidence: Optional[float] = None
    stem_source: str = "template"       # or "llm"
    distractor_basis: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def render_label(doc: Document, span) -> str:
    """Human-readable option text. 'iodine [deficiency]' -> 'iodine deficiency'.

    The bracket form is a graph-internal marker that a node is post-coordinated.
    Printed in an answer option it reads as a bug, so it is unwrapped here —
    while keeping the deviation, which is the entire point of the node: "iodine"
    and "iodine deficiency" are different answers to a causation question.
    """
    quals = [s4.value_name(m) for m in s4.role_changing(span)]
    return f"{span.text} {' '.join(quals)}".strip() if quals else span.text


def _concept(span) -> str:
    return guards.concept_of(span)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def eligible(doc: Document) -> list:
    """Affirmed relations of a questionable type with two linked endpoints."""
    spans = {s.span_id: s for s in doc.spans}
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.type not in STEM_TEMPLATES:
            continue
        head, tail = spans.get(rel.head), spans.get(rel.tail or "")
        if head is None or tail is None:
            continue
        if rel.evidence_start is None or rel.evidence_end is None:
            continue                     # no rationale -> no item
        out.append((rel, head, tail))
    return out


def blocklist(doc: Document) -> dict[str, set]:
    """head concept -> every concept it touches, under ANY predicate, in ANY state.

    Deliberately coarse. A finer index keyed on (head, predicate, tail) would let
    `iodine` be a distractor for a hyperthyroidism causation question even though
    `hyperthyroidism treated_with iodine` is asserted — technically a different
    predicate, and still a question a knowledgeable candidate can argue with.
    """
    spans = {s.span_id: s for s in doc.spans}
    out: dict[str, set] = {}
    for rel in doc.relations:            # every polarity, including uncertain
        head, tail = spans.get(rel.head), spans.get(rel.tail or "")
        if head is None or tail is None:
            continue
        out.setdefault(_concept(head), set()).add(_concept(tail))
        out.setdefault(_concept(tail), set()).add(_concept(head))
    return out


def safe_distractors(doc: Document, rel, head, tail, blocked: dict,
                     limit: int = 3) -> tuple[list, list]:
    """(spans, basis-labels). Ranked by how instructive the confusion is.

    Tier 1, same-role siblings: other tails of the SAME predicate elsewhere in
    the graph, matching the answer's span label. Same semantic role and register,
    so the options do not give themselves away by shape.

    Tier 2, the confusions this pipeline's own guards exist to catch — the
    antonym pairs (hyper-/hypo-) and the deviation nodes (iodine vs iodine
    deficiency). These are the best distractors in the file because they are the
    mistakes a real learner makes, and the graph knows about them explicitly.

    Tier 3, any concept with the answer's span label. Weakest, but keeps the
    options type-consistent.
    """
    spans = {s.span_id: s for s in doc.spans}
    head_c, answer_c = _concept(head), _concept(tail)
    forbidden = {head_c, answer_c} | blocked.get(head_c, set())

    seen: dict[str, object] = {}
    basis: dict[str, str] = {}

    def offer(span, why):
        c = _concept(span)
        if c in forbidden or c in seen:
            return
        if span.label != tail.label:               # keep the option set uniform
            return
        if render_label(doc, span).strip().lower() == render_label(doc, tail).strip().lower():
            return
        seen[c] = span
        basis[c] = why

    # tier 1
    for other in doc.relations:
        if other.polarity != "affirmed" or other.type != rel.type:
            continue
        cand = spans.get(other.tail or "")
        if cand is not None:
            offer(cand, "same-role sibling")
    # tier 2
    for sp in doc.spans:
        if guards.are_antonyms(sp.text, tail.text):
            offer(sp, "antonym of the answer")
        elif (_concept(sp) == answer_c
              and bool(s4.role_changing(sp)) != bool(s4.role_changing(tail))):
            offer(sp, "same concept, different deviation")
    # tier 3
    for sp in doc.spans:
        offer(sp, "same semantic type")

    ordered = sorted(seen, key=lambda c: ["same-role sibling",
                                          "antonym of the answer",
                                          "same concept, different deviation",
                                          "same semantic type"].index(basis[c]))
    picked = ordered[:limit]
    return [seen[c] for c in picked], [basis[c] for c in picked]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def evidence_text(doc: Document, rel, head) -> str:
    chunk = doc.chunk_by_id(head.chunk_id)
    if chunk is None:
        return ""
    a = rel.evidence_start - chunk.char_start
    b = rel.evidence_end - chunk.char_start
    if a < 0 or b > len(chunk.text) or a >= b:
        return ""
    return chunk.text[a:b].strip()


def template_stem(rel, head_label: str) -> str:
    t = STEM_TEMPLATES[rel.type]
    return t.format(head=head_label,
                    head_cap=head_label[:1].upper() + head_label[1:])


def build_items(doc: Document) -> list[Item]:
    """Everything except phrasing. Pure — no LLM, no network."""
    blocked = blocklist(doc)
    chunks = {c.chunk_id: c for c in doc.chunks}
    items, n = [], 0
    for rel, head, tail in eligible(doc):
        picks, basis = safe_distractors(doc, rel, head, tail, blocked)
        if len(picks) < MIN_OPTIONS - 1:
            continue                     # pad an option set and you ship a giveaway
        quote = evidence_text(doc, rel, head)
        if not quote:
            continue
        answer = render_label(doc, tail)
        options = sorted([answer] + [render_label(doc, s) for s in picks],
                         key=str.lower)
        chunk = chunks.get(head.chunk_id)
        section = " > ".join(getattr(chunk, "section_path", None) or []) or "document"
        n += 1
        items.append(Item(
            item_id=f"q{n:03d}",
            stem=template_stem(rel, render_label(doc, head)),
            options=options,
            answer=answer,
            rationale=quote,
            citation=f"{doc.source.source_id} — {section}",
            relation=rel.type,
            confidence=rel.score,
            distractor_basis=basis,
        ))
    return items


# --------------------------------------------------------------------------- #
# LLM phrasing — the only stage that touches a model
# --------------------------------------------------------------------------- #

_PHRASE_SYSTEM = (
    "You rewrite exam question stems so they read like a well-written medical "
    "school single-best-answer item. Return ONLY a JSON array of "
    "{id, stem} objects.\n\n"
    "RULES, all mandatory:\n"
    "1. SINGLE BEST ANSWER ONLY. Never write a negative stem. The words 'not', "
    "'except', 'least' and 'incorrect' are forbidden.\n"
    "2. NEVER name the answer, or any of the other options, anywhere in the "
    "stem.\n"
    "3. Keep the indefinite article: ask for 'a recognised cause', never 'the "
    "cause'. The source may not list every cause, so a definite stem can be "
    "false even when the keyed answer is right.\n"
    "4. Ask exactly what the original stem asks. You are improving the prose, "
    "not the question. Do not add clinical detail, patient vignettes, ages, or "
    "any fact not present in the provided sentence.\n"
    "5. End with a question mark."
)


def phrase_prompt(items: list[Item]) -> str:
    payload = [{"id": i.item_id, "current_stem": i.stem,
                "source_sentence": i.rationale} for i in items]
    return ("Rewrite each stem. The source sentence is given only so the stem "
            "stays faithful to it; do not quote it.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def accept_stem(new: str, item: Item) -> tuple[bool, str]:
    """Guard the model's rewrite. (ok, reason).

    Same shape as Stage 2's evidence-substring rule: the model may improve
    wording, and anything it changes that it should not have costs us the
    rewrite, not the item. Every rejection falls back to the template stem, so a
    misbehaving model degrades the prose and never the correctness.
    """
    if not new or not new.strip():
        return False, "empty"
    new = new.strip()
    if not new.endswith("?"):
        return False, "not a question"
    if _NEGATIVE_RE.search(new):
        return False, "negative stem"
    low = new.lower()
    for opt in item.options:                       # answer AND distractors
        if opt.strip().lower() in low:
            return False, f"names an option ({opt!r})"
    if len(new) > 300:
        return False, "too long"
    return True, ""


def phrase_items(items: list[Item], call_llm, batch: int = 10) -> list[dict]:
    """Rewrite stems in batches. Returns the rejections, for reporting."""
    rejected = []
    by_id = {i.item_id: i for i in items}
    for start in range(0, len(items), batch):
        group = items[start:start + batch]
        raw = call_llm(_PHRASE_SYSTEM, phrase_prompt(group), None)
        for entry in _parse(raw):
            item = by_id.get(entry.get("id"))
            if item is None:
                continue
            ok, why = accept_stem(entry.get("stem", ""), item)
            if ok:
                item.stem = entry["stem"].strip()
                item.stem_source = "llm"
            else:
                rejected.append({"item_id": item.item_id, "reason": why,
                                 "proposed": entry.get("stem", "")})
    return rejected


def _parse(text: str) -> list[dict]:
    cleaned = re.sub(r"```(?:json)?|```", "", text or "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def generate(doc: Document, call_llm=None) -> tuple[list[Item], list[dict]]:
    items = build_items(doc)
    rejected = phrase_items(items, call_llm) if call_llm and items else []
    return items, rejected


def to_json(items: list[Item]) -> str:
    return json.dumps([asdict(i) for i in items], indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Single-best-answer MCQs from an asserted medkg graph.")
    ap.add_argument("input", help="graph.ir.json")
    ap.add_argument("--out", default="questions.json")
    ap.add_argument("--llm", default=None,
                    help="backend profile for stem phrasing; omit for templates only")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    doc = Document.from_json(args.input)
    call = None
    if args.llm:
        from .llm_backends import get_backend
        call, _ = get_backend(args.llm)

    items, rejected = generate(doc, call)
    if args.limit:
        items = items[:args.limit]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(to_json(items))

    asserted = sum(1 for r in doc.relations if r.polarity == "affirmed")
    print(f"{len(items)} items from {asserted} asserted relations "
          f"({len(eligible(doc))} of a questionable type)")
    if call:
        n = sum(1 for i in items if i.stem_source == "llm")
        print(f"stems: {n} rephrased, {len(items) - n} template "
              f"({len(rejected)} rewrites rejected)")
        for r in rejected[:5]:
            print(f"  rejected {r['item_id']}: {r['reason']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
