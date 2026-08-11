"""Single-best-answer MCQ generation from an asserted graph.

Reads a Stage-6 IR (`graph.ir.json`), emits items with a stem, five options, the
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

ONE QUESTION PER (HEAD, PREDICATE), WHICH DEFENCE 3 DOES NOT COVER
------------------------------------------------------------------
Defence 3 secures an item; it does not secure a paper. Three relations sharing
a head and a predicate produce three items, each hiding the other two tails via
defence 1 — so all three are individually defensible and the paper still asks
"a recognised cause of osteoporosis?" three times with three different keys. No
per-item check can see this, so `one_per_question` collapses the group before
any item is built.

NEGATIVE STEMS ARE NOT SUPPORTED AND WILL NOT BE
------------------------------------------------
"All of the following EXCEPT" requires knowing that four statements are FALSE.
Under an open-world assumption the graph cannot establish that about anything.
`_NEGATIVE_RE` enforces this on the model's output, not just in the prompt.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .ir import Document
from . import config
from . import console
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
    "converts_to":     "{head_cap} {copula} converted into which substance?",
    "catalyzes":       "{head_cap} acts on which substrate?",
    "stores":          "Which substance is stored in {head}?",
    "located_in":      "{head_cap} {copula} found within which structure?",
    "transports":      "Which substance is transported by {head}?",
    "part_of":         "{head_cap} {copula} part of which structure?",
    "synthesizes":     "Which substance is synthesised by {head}?",
    "requires":        "{head_cap} requires which substance?",
    "risk_factor_for": "{head_cap} {copula} a risk factor for which condition?",
}

# `regulates` is deliberately absent: "what does TRH regulate?" has no crisp
# single answer, and the ontology treats it as the unsigned parent of
# stimulates/inhibits anyway. Vague relations make vague questions.

# Predicates the ontology separates and a candidate cannot. Containment and
# membership both come out as "which structure", so a head carrying one of each
# would otherwise be asked twice.
_QUESTION_CLASSES = {
    "located_in": "containment",
    "part_of":    "containment",
}

MIN_OPTIONS = 5          # 1 key + 4 distractors; fewer -> skip the item entirely

# Distractors are picked before the stem-leak filter runs, so ask for a few
# spare candidates rather than losing an otherwise sound item to the trim.
DISTRACTOR_OVERSHOOT = 3

# How much of a paper one answer may key. A hub concept collects containment
# edges from everything around it, and every edge is an item, so `pelvis` keyed
# 8 of 69 on a pelvic floor document. What that buys a candidate is a guessing
# strategy: answer the hub whenever unsure and score at the hub's share. Held
# under a fifth of the paper, that strategy pays worse than picking at random
# from five options, which is the point of the number.
MAX_KEY_SHARE = 0.10
MIN_KEY_REPEATS = 3      # a floor, so a short paper is not capped to nothing

_NEGATIVE_RE = re.compile(
    r"\b(not|except|never|least|incorrect|false|untrue|EXCEPT)\b", re.IGNORECASE)

# --- what cannot be an option --------------------------------------------- #
# Direction and state adjectives link cleanly to UMLS and carry an anatomical
# or substance label, so nothing upstream rejects them — but "posterior" is not
# a structure any question can key, and "cellular" is not a place.
_NON_ANSWER_TERMS = {
    "anterior", "posterior", "superior", "inferior", "medial", "lateral",
    "proximal", "distal", "dorsal", "ventral", "superficial", "deep",
    "cellular", "intracellular", "extracellular", "systemic", "local",
    "acute", "chronic", "normal", "abnormal", "primary", "secondary",
    "tertiary", "active", "inactive",
}

# A section heading survives linking because its head noun links cleanly.
# "Disorders of the parathyroid glands" is a chapter title, not an answer.
_HEADING_RE = re.compile(
    r"^(disorders?|conditions?|diseases?|types?|forms?|issues?|examples?|"
    r"structures?|factors?|causes?|symptoms?|functions?|roles?)\b"
    r".*\b(of|like|such as|including)\b", re.IGNORECASE)

MAX_OPTION_WORDS = 6     # beyond this an option is a clause, not a term

# Words a stem template contributes itself. An option sharing one of these with
# the stem is not leaking the key, it just uses the same ordinary vocabulary.
_STEM_FRAME_WORDS = {
    "which", "what", "following", "recognised", "recognized", "cause",
    "causes", "caused", "condition", "conditions", "structure", "structures",
    "substance", "substances", "agent", "agents", "used", "management",
    "develop", "develops", "complication", "within", "found", "part",
    "acts", "acted", "upon", "requires", "required", "risk", "factor",
    "into", "converted", "converts", "secreted", "secretes", "stored",
    "stores", "transported", "transports", "synthesised", "synthesized",
    "substrate", "associated", "increased", "body", "levels", "form",
}


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


def tidy_label(label: str) -> str:
    """Undo what a span inherited from its position in a sentence.

    Two artefacts, both of which single an option out from its neighbours. A
    mention that opened a sentence carries a capital, and one that closed a
    sentence carries the full stop; either way the option looks unlike the four
    beside it, which hands over the key without any recall at all.

    Acronyms and internal capitals survive: a first word is only lowered when
    the rest of it is already lowercase, so `Calcium` folds while `PTH`, `CaSR`
    and `Vitamin D3` do not.

    Dangling brackets go too. A span boundary landing mid-parenthesis leaves
    `Duodenum (`, which reads as a typo wherever it is printed.
    """
    label = label.strip().strip("([{-–— \t").strip(".,;:)]}").strip()
    if not label:
        return label
    first = label.split(" ", 1)[0]
    if len(first) > 1 and not first[1:].islower():
        return label
    return label[:1].lower() + label[1:]


def _label_key(span) -> tuple:
    """Identity for surface-form agreement: concept plus its deviations.

    Keyed on the deviations as well as the concept because `iodine` and `iodine
    deficiency` share a CUI and must keep separate option text — collapsing
    them here would undo what post-coordination exists to record.
    """
    quals = tuple(sorted(s4.value_name(m) for m in s4.role_changing(span)))
    return (_concept(span), quals)


def is_abbreviation(label: str) -> bool:
    """Whether a label reads as an initialism rather than a term.

    Judged on the first word, so `ADH dysfunction` counts alongside `ADH` — a
    deviation does not turn an abbreviation back into prose. A short word of
    two or more capitals qualifies (`PTH`, `TSH`, `DHEA`), as does a capital
    followed by a digit (`T3`, `T4`), which the capital count alone misses.
    `Calcium` has one capital and no digit, so it does not.
    """
    words = label.strip().split()
    if not words:
        return False
    first = words[0]
    if len(first) > 6:
        return False
    return (sum(c.isupper() for c in first) >= 2
            or (first[:1].isupper() and any(c.isdigit() for c in first)))


def canonical_labels(doc: Document) -> dict[tuple, str]:
    """One agreed surface form per concept, so a paper does not say both.

    A document that writes `PTH` in one sentence and `parathyroid hormone` in
    the next yields two spans of one concept, and two items whose options
    disagree about what to call it.

    The expanded form wins whenever the document offers one, ahead of how often
    each spelling occurs. Frequency alone kept `GH` in a set beside `insulin`
    and `glucagon`, and an option shaped unlike its four neighbours is the key
    given away — the same tell as the sentence-initial capital, one level up.
    Only then does the most frequent spelling win, ties going to the longer.
    """
    counts: dict[tuple, dict[str, int]] = {}
    for span in doc.spans:
        surface = tidy_label(render_label(doc, span))
        if not surface:
            continue
        tally = counts.setdefault(_label_key(span), {})
        tally[surface] = tally.get(surface, 0) + 1
    return {key: max(tally, key=lambda s: (not is_abbreviation(s),
                                           tally[s], len(s)))
            for key, tally in counts.items()}


def option_label(doc: Document, span, canonical: dict) -> str:
    """The text a span gets as an option: its canonical form when one exists."""
    return canonical.get(_label_key(span)) or tidy_label(
        render_label(doc, span))


def usable_option(label: str) -> bool:
    """Whether a label can stand as an option at all, key or distractor.

    Rejects what is grammatically a span but never an answer: bare direction
    and state adjectives, and section headings that linked through their head
    noun. Applied to the key as well, because an item keyed `cellular` is not
    salvageable by swapping its distractors.
    """
    label = label.strip()
    if not label:
        return False
    if label.lower() in _NON_ANSWER_TERMS:
        return False
    if _HEADING_RE.match(label):
        return False
    if label.count("(") != label.count(")"):
        return False               # a span boundary landed inside a bracket
    return len(label.split()) <= MAX_OPTION_WORDS


def too_close(label: str, answer: str) -> bool:
    """Whether an option is a near-restatement of the key rather than a rival.

    Compared against the KEY alone, and only at the ends of the string. Offering
    `blood` beside a keyed `bloodstream`, or `inositol trisphosphate` beside a
    keyed `phosphate`, gives an item two defensible answers. Testing prefixes
    and suffixes rather than containment anywhere is what keeps `thyroid gland`
    usable against a keyed `parathyroid glands` — a genuine learner confusion
    that a plain substring test would throw away.
    """
    a, b = label.strip().lower(), answer.strip().lower()
    if not a or not b or a == b:
        return a == b
    short, long = sorted((a, b), key=len)
    if len(short) < 4:
        return False
    return long.startswith(short) or long.endswith(short)


def _content_words(text: str) -> set[str]:
    """Tokens carrying enough meaning that repeating one is a tell."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) >= 4 and t not in _STEM_FRAME_WORDS}


def leaked_option(stem: str, options: list[str]) -> Optional[str]:
    """The first option the stem gives away, or None.

    Every template names the head, so an option that shares a content word with
    the stem is one that restates the head — `bone` against "Cortical bone is
    part of which structure?", `hyperparathyroidism` against "a recognised
    cause of Tertiary hyperparathyroidism?". Word-level and symmetric, which is
    what a substring test in one direction misses when the key is the longer
    string of the two.
    """
    frame = _content_words(stem)
    for option in options:
        if _content_words(option) & frame:
            return option
    return None


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
                     canonical: Optional[dict] = None,
                     limit: int = MIN_OPTIONS - 1) -> tuple[list, list]:
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
    canonical = canonical if canonical is not None else {}
    spans = {s.span_id: s for s in doc.spans}
    head_c, answer_c = _concept(head), _concept(tail)
    forbidden = {head_c, answer_c} | blocked.get(head_c, set())
    answer_label = option_label(doc, tail, canonical)

    seen: dict[str, object] = {}
    basis: dict[str, str] = {}

    def offer(span, why):
        c = _concept(span)
        if c in forbidden or c in seen:
            return
        if span.label != tail.label:               # keep the option set uniform
            return
        label = option_label(doc, span, canonical)
        if not usable_option(label):
            return
        if too_close(label, answer_label):
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
    """The source quote behind a relation, grown out to its whole sentence.

    The stored offsets clip to what the extractor matched, which on a bulleted
    list is one cell — a rationale reading `Aorta` in full. Widening to the
    enclosing sentence costs nothing when the offsets already covered one and
    recovers a readable justification when they did not. Bounded by the chunk,
    so nothing is quoted that the citation does not cover.
    """
    chunk = doc.chunk_by_id(head.chunk_id)
    if chunk is None:
        return ""
    a = rel.evidence_start - chunk.char_start
    b = rel.evidence_end - chunk.char_start
    if a < 0 or b > len(chunk.text) or a >= b:
        return ""
    start = max((chunk.text.rfind(mark, 0, a) for mark in (". ", "\n")),
                default=-1)
    end = min((e for e in (chunk.text.find(mark, b) for mark in (". ", "\n"))
               if e != -1), default=-1)
    a = 0 if start == -1 else start + 1
    b = len(chunk.text) if end == -1 else end + 1
    return chunk.text[a:b].strip(" \t\n-*•").strip()


_FINITE_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|does|do|can|may|produces?|produced|"
    r"secretes?|secreted|causes?|caused|forms?|formed|contains?|contained|"
    r"leads?|led|acts?|acted|helps?|helped|stores?|stored|converts?|"
    r"converted|requires?|required|regulates?|regulated|releases?|released|"
    r"stimulates?|stimulated|synthesises?|synthesizes?|enhances?|absorbs?|"
    r"transports?|develops?|arises?|results?|consists?|includes?|involves?)\b",
    re.IGNORECASE)


def supports_item(quote: str, answer: str, head_label: str) -> bool:
    """Whether a quote can stand as the rationale for its item.

    A rationale is this module's one claim to being verified, so it has to be a
    statement mentioning what the item asks about. A cell lifted from a mnemonic
    list — `Pancreas (except the tail)` under a heading of retroperitoneal
    organs — records a true fact and justifies nothing to the candidate
    reading it, which is worse than having no item.

    Requires a finite verb and at least one endpoint by name. Naming one rather
    than both is deliberate: a sentence often carries the head only as a pronoun
    or an earlier subject, and demanding both would drop sound items over
    anaphora this module has no business resolving.
    """
    quote = quote.strip()
    if len(quote.split()) < 5 or not _FINITE_VERB_RE.search(quote):
        return False
    low = quote.lower()
    return answer.strip().lower() in low or head_label.strip().lower() in low


# Latin and Greek plurals that end in a vowel, which no trailing -s test finds.
_PLURAL_HEADS = {
    "viscera", "adnexa", "cristae", "rami", "sulci", "gyri", "septa", "ostia",
    "fasciae", "lamellae", "papillae", "villi", "alveoli", "bronchi", "nuclei",
    "glomeruli", "canaliculi", "mitochondria", "cristae", "data", "criteria",
    "phenomena", "ganglia", "cilia", "stomata", "foramina", "cornua",
}

# Singular nouns ending in -s that the -is/-us rule below does not already
# cover.
_SINGULAR_HEADS = {"pancreas", "diabetes", "ascites", "series", "species",
                   "faeces", "feces", "herpes"}


def is_plural_head(head_label: str) -> bool:
    """Whether a head label takes a plural verb.

    The last word decides. Anatomy is full of classical forms where a trailing
    `-s` means nothing, so `-is` and `-us` are read as singular — `symphysis`,
    `pelvis`, `hiatus`, `plexus`, `hypothalamus` — and the vowel plurals that
    no `-s` test can find are listed outright. `ischiopubic rami` is plural,
    `sacrococcygeal symphysis` is not.
    """
    words = head_label.strip().lower().split()
    last = words[-1] if words else ""
    if last in _PLURAL_HEADS:
        return True
    if last in _SINGULAR_HEADS:
        return False
    if last.endswith("is") or last.endswith("us"):
        return False               # symphysis, pelvis, hiatus, corpus
    return last.endswith("s") and not last.endswith("ss")


def template_stem(rel, head_label: str) -> str:
    """Fill a stem template, agreeing the copula with the head.

    `{copula}` is what keeps "Blood vessels is found within which structure?"
    out of a template-only run. The rewrite pass repairs most of these when it
    is enabled, which is exactly why the templates cannot rely on it.
    """
    t = STEM_TEMPLATES[rel.type]
    return t.format(head=head_label,
                    head_cap=head_label[:1].upper() + head_label[1:],
                    copula="are" if is_plural_head(head_label) else "is")


def question_class(relation_type: str) -> str:
    """The question a predicate asks, which is coarser than the predicate.

    `located_in` and `part_of` are distinct in the ontology and identical at
    the point of asking: both templates say "which structure", and once the
    rewrite pass has been over them "Which structure includes the pelvic
    inlet?" and "In which structure is the pelvic inlet located?" are one
    question with one key. Grouping on the predicate leaves both standing.
    """
    return _QUESTION_CLASSES.get(relation_type, relation_type)


def one_per_question(doc: Document, candidates: list) -> list:
    """Collapse relations that would ask the same question to their best one.

    `osteoporosis caused_by calcium`, `... caused_by vitamin D` and `...
    caused_by bone` are three relations and one question, and each becomes an
    item whose blocklist correctly hides the other two — so every item is
    defensible alone while the paper asks "a recognised cause of osteoporosis?"
    three times with three different keys. Nothing inside a single item can see
    that, which is why the grouping happens here rather than in
    `safe_distractors`.

    Grouped on the question asked rather than the predicate stored, so a head
    carrying both `located_in` and `part_of` yields one item and not two.

    Keeps the best-attested relation per (head concept, question class): highest
    score, then the longest evidence span, then the tail text so a rerun over
    an unchanged graph produces an unchanged paper.
    """
    groups: dict[tuple, list] = {}
    for rel, head, tail in candidates:
        groups.setdefault(
            (_concept(head), question_class(rel.type)), []).append(
                (rel, head, tail))
    return [max(group,
                key=lambda t: (t[0].score or 0.0,
                               (t[0].evidence_end or 0) -
                               (t[0].evidence_start or 0),
                               t[2].text))
            for group in groups.values()]


def key_cap(item_count: int) -> int:
    """How often one answer may be the key in a paper of this size."""
    return max(MIN_KEY_REPEATS, math.ceil(MAX_KEY_SHARE * item_count))


def cap_key_repeats(entries: list) -> list:
    """Thin out answers that key too much of the paper.

    Takes (key identity, Item) pairs and returns the survivors in their original
    order. Keyed on the concept and its deviations rather than the printed text,
    so `iodine` and `iodine deficiency` are counted apart — they are genuinely
    different answers — while two spellings of one concept are counted together.

    Within an over-represented answer the best-attested items stay: highest
    confidence, then the longest rationale, then the stem, which decides nothing
    of substance and keeps a rerun over an unchanged graph stable.

    The cap comes from the count before any of this runs, and is applied once.
    Recomputing it against the survivors would ratchet — each drop shrinks the
    paper, which tightens the cap, which drops more — and converge somewhere
    unrelated to what the graph actually supports.
    """
    cap = key_cap(len(entries))
    ranked: dict[tuple, list] = {}
    for position, (identity, item) in enumerate(entries):
        ranked.setdefault(identity, []).append((position, item))
    keep = set()
    for group in ranked.values():
        group.sort(key=lambda p: (-(p[1].confidence or 0.0),
                                  -len(p[1].rationale), p[1].stem))
        keep.update(position for position, _ in group[:cap])
    return [entries[p] for p in sorted(keep)]


def build_items(doc: Document) -> list[Item]:
    """Everything except phrasing. Pure — no LLM, no network."""
    blocked = blocklist(doc)
    canonical = canonical_labels(doc)
    chunks = {c.chunk_id: c for c in doc.chunks}
    candidates = eligible(doc)
    merged = one_per_question(doc, candidates)
    if len(merged) < len(candidates):
        console.announce_detail(
            f"{len(candidates) - len(merged)} relation(s) folded into an "
            f"existing question, leaving {len(merged)}")

    entries, leaked, unsupported = [], 0, 0
    for rel, head, tail in merged:
        answer = option_label(doc, tail, canonical)
        if not usable_option(answer):
            continue                     # a key nothing can rescue
        head_label = option_label(doc, head, canonical)
        if not usable_option(head_label):
            continue                     # nor one the stem cannot name
        stem = template_stem(rel, head_label)
        if leaked_option(stem, [answer]):
            leaked += 1
            continue                     # the stem names its own key
        quote = evidence_text(doc, rel, head)
        if not quote:
            continue
        if not supports_item(quote, answer, head_label):
            unsupported += 1
            continue                     # a rationale that justifies nothing
        picks, basis = safe_distractors(
            doc, rel, head, tail, blocked, canonical,
            limit=MIN_OPTIONS - 1 + DISTRACTOR_OVERSHOOT)

        # Distractors the stem gives away go, but the item survives on the
        # spares — only a leaking key is fatal.
        kept = [(s, b) for s, b in zip(picks, basis)
                if not leaked_option(stem, [option_label(doc, s, canonical)])]
        kept = kept[:MIN_OPTIONS - 1]
        if len(kept) < MIN_OPTIONS - 1:
            continue                     # pad an option set and you ship a giveaway
        options = sorted(
            [answer] + [option_label(doc, s, canonical) for s, _ in kept],
            key=str.lower)
        chunk = chunks.get(head.chunk_id)
        section = " > ".join(getattr(chunk, "section_path", None) or []) or "document"
        entries.append((_label_key(tail), Item(
            item_id="",                  # assigned once the paper is final
            stem=stem,
            options=options,
            answer=answer,
            rationale=quote,
            citation=f"{doc.source.source_id} — {section}",
            relation=rel.type,
            confidence=rel.score,
            distractor_basis=[b for _, b in kept],
        )))
    if leaked:
        console.announce_detail(
            f"{leaked} item(s) dropped: the template stem named its own key")
    if unsupported:
        console.announce_detail(
            f"{unsupported} item(s) dropped: the evidence does not state the "
            f"fact being asked")

    survivors = cap_key_repeats(entries)
    if len(survivors) < len(entries):
        console.announce_detail(
            f"{len(entries) - len(survivors)} item(s) dropped: an answer keyed "
            f"more than {key_cap(len(entries))} of {len(entries)} items")
    items = [item for _, item in survivors]
    for n, item in enumerate(items, 1):
        item.item_id = f"q{n:03d}"
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
    opt = leaked_option(new, item.options)         # answer AND distractors
    if opt is not None:
        return False, f"names an option ({opt!r})"
    if len(new) > 300:
        return False, "too long"
    return True, ""


def resolve_stem_collisions(items: list[Item],
                            templates: dict[str, str]) -> list[Item]:
    """Undo rewrites that made two items ask the same question.

    `accept_stem` judges a rewrite against its own item and has no way to see
    the rest of the paper, so the model can turn "Sacrum is found within which
    structure?" and "Sacrum is part of which structure?" into one sentence
    asked twice with two keys. Batching makes it likelier still: the model sees
    ten related stems at once and regularises them towards each other.

    A colliding item reverts to its template, which is distinct by construction.
    Items still colliding after that are a genuine duplicate question rather
    than a phrasing accident, so all but the best-attested one are dropped —
    highest confidence, then the earlier id, so the choice does not depend on
    dictionary order.
    """
    seen: dict[str, Item] = {}
    for item in items:
        key = item.stem.strip().lower()
        if key in seen and item.stem_source == "llm":
            item.stem = templates.get(item.item_id, item.stem)
            item.stem_source = "template"
        seen.setdefault(item.stem.strip().lower(), item)

    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(item.stem.strip().lower(), []).append(item)
    keep = {max(g, key=lambda i: (i.confidence or 0.0, [-ord(c) for c in
                                                       i.item_id])).item_id
            for g in groups.values()}
    return [i for i in items if i.item_id in keep]


def phrase_items(items: list[Item], call_llm, batch: int = 10) -> list[dict]:
    """Rewrite stems in batches. Returns the rejections, for reporting.

    Mutates `items` in place, so a caller wanting the collision pass applied
    must use what `generate` returns rather than the list it passed in.
    """
    rejected = []
    by_id = {i.item_id: i for i in items}
    starts = list(range(0, len(items), batch))
    console.announce_step(
        f"rephrasing {console.format_count(len(items), 'stem')} in "
        f"{len(starts)} LLM call(s) of up to {batch}")
    for start in console.with_progress(starts, "batches rephrased"):
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
    """Build items, then optionally rephrase their stems with `call_llm`.

    Returns (items, rejected_rewrites). Without a backend the templated stems
    stand as written and nothing is rejected.
    """
    console.announce_step(
        f"building items from {len(doc.relations)} relation(s)")
    items = build_items(doc)
    console.announce_detail(
        f"{console.format_count(len(items), 'item')} survived the "
        f"distractor-safety rules")
    if not (call_llm and items):
        return items, []

    templates = {i.item_id: i.stem for i in items}
    rejected = phrase_items(items, call_llm)
    kept = resolve_stem_collisions(items, templates)
    if len(kept) < len(items):
        console.announce_detail(
            f"{len(items) - len(kept)} item(s) dropped: a rewritten stem "
            f"duplicated another")
    return kept, rejected


def to_json(items: list[Item]) -> str:
    return json.dumps([asdict(i) for i in items], indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Single-best-answer MCQs from an asserted medkg graph.")
    ap.add_argument("input", help="graph.ir.json")
    ap.add_argument("--out", default="questions.json")
    ap.add_argument("--artifacts", default=None,
                    help=f"directory for generated files "
                         f"(default: {config.ARTIFACTS_DIR}/)")
    ap.add_argument("--llm", default=None,
                    help="backend profile for stem phrasing; omit for templates only")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.artifacts:
        config.ARTIFACTS_DIR = args.artifacts
    args.out = config.artifact_path(args.out)
    print(f"generating MCQs from {args.input}")
    doc = Document.from_json(args.input)
    console.announce_step(
        f"{len(doc.relations)} relation(s), {len(doc.spans)} span(s) read")
    call = None
    if args.llm:
        console.announce_step(
            f"loading the {args.llm} backend for stem phrasing")
        from .llm_backends import get_backend
        call, _ = get_backend(args.llm)
    else:
        console.announce_step("no --llm: stems stay templated")

    items, rejected = generate(doc, call)
    if args.limit:
        items = items[:args.limit]
        console.announce_step(f"keeping the first {args.limit} item(s)")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(to_json(items))

    asserted = sum(1 for r in doc.relations if r.polarity == "affirmed")
    print()
    print(f"{len(items)} items from {asserted} asserted relations "
          f"({len(eligible(doc))} of a questionable type)")
    if call:
        n = sum(1 for i in items if i.stem_source == "llm")
        print(f"stems: {n} rephrased, {len(items) - n} template "
              f"({len(rejected)} rewrites rejected)")
        for r in rejected[:5]:
            print(f"  rejected {r['item_id']}: {r['reason']}")
    print(f"questions -> {args.out}")


if __name__ == "__main__":
    main()
