"""Stage 2 — NER + abbreviations + negation, plus LLM relation AND modifier
extraction.

NER + negation come from scispaCy; relations and modifiers both come from an
injectable LLM pass (`call_llm`), so they handle titles/bullets, not just prose.
Prompts inject the section heading and a structured-content flag so the LLM can
read scope off the layout (e.g. a "Contraindications" heading negating its
bullets). Grounding guards (evidence/text must be an exact substring) still apply
to both passes, so a weaker backend routes more items to needs_review rather than
corrupting the graph.

Stage 2 only ever reads chunks whose `kind` is in
`config.EXTRACTABLE_CHUNK_KINDS` — prose, definitions, key points, clinical
notes and the closing summary. Captions and figure descriptions are tagged by
Stage 1 and belong to Stage 5 (prose about how a picture looks must never become
a clinical fact); learning objectives, references and deduplicated repeats are
excluded outright.

`looks_structured()` still matters. Stage 1 no longer restructures anything —
the upstream rewriter owns that — so a <con> block the rewriter returned as a
list still arrives one item per line, and those items carry anaphora ("They are
crucial for...") that the flag warns the RE prompt about.

Heavy libs (spacy, scispacy, negspacy, anthropic) are imported LAZILY inside the
functions that need them, so the pure helpers below (classification, prompt
building, validation, grounding) can be imported and unit-tested with only the
standard library.
"""
from __future__ import annotations

import itertools
import json
import os
import re
from typing import Optional

from .ir import Document, Span, Relation, Modifier
from . import config
from . import guards
from . import ontology

# ---------------------------------------------------------------------------
# NER + modifiers + negation
# ---------------------------------------------------------------------------

def load_nlp(model: str = None):
    """Build one scispaCy pipeline with abbreviation + negation components."""
    import spacy
    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401 (registers factory)
    from negspacy.negation import Negex  # noqa: F401 (registers factory)

    nlp = spacy.load(model or config.NER_MODELS[0])
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("negex", config={"ent_types": []})  # run over all entity types
    return nlp


def load_pipelines(models=None) -> list[tuple[str, object]]:
    """One pipeline per configured NER model.

    A single model bounds what the whole graph can express: `en_ner_bc5cdr_md`
    emits DISEASE and CHEMICAL and nothing else, so with it alone no relation
    about anatomy, cells or proteins can ever fire — `secretes`, `part_of` and
    `converts_to` are unreachable no matter how the ontology is worded. Adding
    `en_ner_bionlp13cg_md` (ORGAN, TISSUE, CELL, GENE_OR_GENE_PRODUCT,
    SIMPLE_CHEMICAL, ...) is what makes physiology extractable at all.

        MEDKG_NER_MODELS=en_ner_bc5cdr_md,en_ner_bionlp13cg_md
    """
    return [(m, load_nlp(m)) for m in (models or config.NER_MODELS)]


def map_label(model: str, raw_label: str) -> str:
    """Raw NER label -> ontology span label, per the ontology's `ner_models`."""
    mapping = config.NER_LABEL_MAPS.get(model, {})
    return mapping.get(raw_label, raw_label.replace("_", " ").title().replace(" ", ""))


def merge_entities(found: list[tuple]) -> list[tuple]:
    """Merge entities from several models over one chunk.

    `found` is (start, end, text, label, negated, model_index). Overlaps are
    resolved by keeping the LONGEST span, then the earliest-configured model —
    two models will both tag "thyroid hormone", and asserting it twice would
    double-count every relation it takes part in.
    """
    ordered = sorted(found, key=lambda e: (-(e[1] - e[0]), e[5], e[0]))
    kept: list[tuple] = []
    for ent in ordered:
        if any(not (ent[1] <= k[0] or ent[0] >= k[1]) for k in kept):
            continue                      # overlaps something already kept
        kept.append(ent)
    return sorted(kept, key=lambda e: e[0])


# Modifier types the LLM may emit — kept aligned with config.SNOMED_ATTR so
# Stage 4 has an attribute-property mapping for each.
def allowed_modifier_types() -> tuple:
    """Read live from the loaded ontology -- freezing this at import meant a
    swapped --ontology silently kept the old modifier set."""
    return tuple(config.SNOMED_ATTR.keys())


ALLOWED_MODIFIER_TYPES = allowed_modifier_types()

# JSON schemas for the two LLM passes. Canonical here (the stage owns its output
# shapes); guided backends like Meditron enforce them via guided_json.
RE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "head_id":    {"type": "string"},
            "tail_id":    {"type": "string"},
            "type":       {"type": "string"},
            "polarity":   {"type": "string", "enum": ["affirmed", "negated", "uncertain"]},
            "evidence":   {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["head_id", "tail_id", "type", "polarity", "evidence", "confidence"],
    },
}

MODIFIER_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "span_id": {"type": "string"},
            "type":    {"type": "string", "enum": list(ALLOWED_MODIFIER_TYPES)},
            "text":    {"type": "string"},
        },
        "required": ["span_id", "type", "text"],
    },
}

def modifier_menu() -> str:
    """The allowed modifier types WITH their glosses, read live from the ontology.

    This used to be a hand-written list of five types inside `_MOD_SYSTEM`. The
    ontology had grown to nine, so `temporality`, `quantity`, `frequency` and
    `route` were accepted by `validate_modifier` and never once requested from
    the model — a silent recall hole that no test could see, because both ends
    of it were consistent with themselves. Deriving the prompt from the same
    dict the validator reads makes that class of drift impossible.
    """
    lines = []
    for name in allowed_modifier_types():
        gloss = (config.MODIFIER_META.get(name, {}) or {}).get("gloss", "")
        lines.append(f"- {name}" + (f": {gloss}" if gloss else ""))
    return "\n".join(lines)


_MOD_SYSTEM = (
    "You are a biomedical modifier extractor. For each listed entity, extract any "
    "qualifiers the text states about it. Return ONLY a JSON array; each item "
    "{span_id, type, text}. `text` must be an exact substring of the provided "
    "text. Return [] if there are no modifiers.\n\n"
    "`deviation` matters more than the rest put together. Medical text names a "
    "substance and states an abnormal amount of it separately -- 'a deficiency "
    "in dietary iodine', 'an excess of thyroid hormones', 'insufficient hormone "
    "levels'. The entity list will only contain the substance. If you do not "
    "return the deviation, the graph ends up asserting that iodine causes "
    "goiter, when the source said its absence does. Whenever an entity is "
    "described as deficient, insufficient, lacking, excessive, elevated, over- "
    "or underproduced, emit a `deviation` modifier for it."
)


def looks_structured(text: str) -> bool:
    """Heuristic: is this chunk titles/bullets rather than prose? Drives the
    'structured' prompt flag (headings may carry scope). Pure/stdlib."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) == 1:                                 # a bare title / fragment
        ln = lines[0]
        return not ln.endswith((".", "!", "?")) and len(ln.split()) <= 8
    bullets = sum(1 for ln in lines if re.match(r"^([-*•–·◦]|\d+[.)])\s+", ln))
    ends_punct = sum(1 for ln in lines if ln.endswith((".", "!", "?")))
    return bullets / len(lines) >= 0.4 or ends_punct / len(lines) < 0.5


def _context_header(section_path=None, structured: bool = False) -> str:
    """Shared prompt preamble: injects the section heading (so a bullet keeps the
    meaning its heading gives it) and, for structured chunks, a scope hint."""
    parts = []
    if section_path:
        parts.append(f"Section: {' > '.join(section_path)}")
    if structured:
        parts.append("This text is structured (titles/bullets); the section heading "
                     "may carry scope — e.g. a 'Contraindications' or 'Ruled out' "
                     "heading negates the items beneath it.")
    return ("\n".join(parts) + "\n") if parts else ""


def extract_entities(doc: Document, nlp=None, pipelines=None) -> Document:
    """NER over every extractable chunk, across all configured models."""
    if pipelines is None:
        pipelines = [("", nlp)] if nlp is not None else load_pipelines()
    counter = itertools.count(1)
    for chunk in doc.extractable_chunks(config.EXTRACTABLE_CHUNK_KINDS):
        found = []
        for idx, (model, pipe) in enumerate(pipelines):
            for ent in pipe(chunk.text).ents:
                found.append((ent.start_char, ent.end_char, ent.text,
                              map_label(model, ent.label_),
                              bool(getattr(ent._, "negex", False)), idx))
        for start, end, text, label, negated, _ in merge_entities(found):
            doc.spans.append(Span(
                span_id=f"s{next(counter):03d}",
                chunk_id=chunk.chunk_id,
                text=text,
                char_start=chunk.char_start + start,
                char_end=chunk.char_start + end,
                label=label,
                negated=negated,
                modifiers=[],                 # filled by extract_modifiers_llm
            ))
    return doc


# ---------------------------------------------------------------------------
# LLM relation extraction  (pure helpers are stdlib-only)
# ---------------------------------------------------------------------------

def candidate_pairs(spans: list[Span]) -> list[tuple[Span, Span]]:
    """All ordered pairs whose (head.label, tail.label) is permitted by at least
    one relation type in the ontology. Shrinks the LLM's job dramatically."""
    pairs = []
    for a, b in itertools.permutations(spans, 2):
        for spec in config.RELATION_ONTOLOGY.values():
            if spec["tail"] is None:                       # unary relation -> not a pair
                continue
            head_ok = spec["head"] == config.ANY or a.label in spec["head"]
            tail_ok = spec["tail"] == config.ANY or b.label in spec["tail"]
            if head_ok and tail_ok:
                pairs.append((a, b))
                break
    return pairs


_SYSTEM = (
    "You are a biomedical relation extractor. Given a sentence and a list of "
    "entities with ids and types, return ONLY the relations that the sentence "
    "explicitly supports, drawn strictly from the allowed relation types. "
    "Respond with a JSON array and nothing else (no prose, no markdown fences). "
    "Each item: {head_id, tail_id, type, polarity, evidence, confidence}. "
    "polarity is one of affirmed|negated|uncertain. evidence must be an exact "
    "substring of the sentence. If no relation holds, return [].\n\n"
    "DIRECTION IS NOT OPTIONAL. Each allowed type below states what its HEAD "
    "and its TAIL mean. Read that before choosing head_id and tail_id, and do "
    "not infer direction from the word order of the sentence -- several types "
    "are named for the reverse of the way an English sentence usually runs. "
    "'Hyperthyroidism can lead to osteoporosis' is "
    "{head: osteoporosis, type: caused_by, tail: hyperthyroidism}, because "
    "caused_by points from the effect to its cause. The same rule holds when "
    "causation is phrased as a NOUN rather than a verb, which is where it is "
    "usually got wrong: 'Graves' disease is a frequent cause of "
    "hyperthyroidism' is {head: hyperthyroidism, type: caused_by, tail: "
    "Graves' disease}, and so is 'the most common cause is Hashimoto's "
    "thyroiditis'. If you cannot tell which entity is which, omit the relation "
    "rather than guessing.\n\n"
    "IF YOU NEED THE OPPOSITE OF A LISTED TYPE, use the listed type and swap "
    "head_id and tail_id. Do not invent a name for the reverse direction: there "
    "is no `produced_by`, only `secretes` with the arguments the other way "
    "round.\n\n"
    "THE EVIDENCE MUST CONTAIN BOTH ENTITIES. Quote the span of text that "
    "mentions the head and the tail and states the link between them. A quote "
    "from a neighbouring clause is not evidence, and a relation you can only "
    "support that way should be omitted.\n\n"
    "AN ENTITY IS ONLY WHAT IT NAMES. The entity list holds bare concepts, but "
    "sentences attribute effects to phrases built around them: 'a deficiency "
    "in dietary iodine', 'an excess of thyroid hormones', 'autoimmune "
    "reactions against thyroid peroxidase', 'inhibiting bone resorption', "
    "'removing iodine atoms'. Iodine does not cause goiter -- its absence "
    "does; thyroid peroxidase does not cause Hashimoto's -- the immune "
    "reaction against it does; calcitonin does not inhibit bone -- it inhibits "
    "resorption. If the entity you would use is only part of the phrase the "
    "sentence is really talking about, omit the relation."
)


def relation_menu(spans: list[Span] = None) -> str:
    """The allowed relation types WITH their direction glosses.

    Sending bare type names was a real defect: `caused_by` cannot be applied
    correctly by a model that has not been told whether the head is the cause or
    the effect, and the resulting graph asserted both directions of the same
    fact. The gloss is the contract, so it goes in the prompt.

    When `spans` is given, types whose head/tail labels are absent from this
    sentence are omitted -- a shorter, sharper menu, and it stops the model
    reaching for a type that could not survive `validate_relation` anyway.
    """
    labels = {s.label for s in spans} if spans else None
    lines = []
    for name in sorted(config.RELATION_ONTOLOGY):
        spec = config.RELATION_ONTOLOGY[name]
        if labels is not None:
            if spec["head"] and not (spec["head"] & labels):
                continue
            if spec["tail"] is not None and not (spec["tail"] & labels):
                continue
        head = "/".join(sorted(spec["head"])) if spec["head"] else "any"
        tail = "/".join(sorted(spec["tail"])) if spec["tail"] else "(none: unary)"
        gloss = spec.get("gloss") or ""
        lines.append(f"- {name} [head: {head} -> tail: {tail}] {gloss}")
    return "\n".join(lines)


def build_re_prompt(text: str, spans: list[Span], section_path=None,
                    structured: bool = False) -> str:
    ents = [{"id": s.span_id, "text": s.text, "type": s.label} for s in spans]
    return (
        _context_header(section_path, structured)
        + "Allowed relation types (HEAD and TAIL roles are defined here -- "
          "follow them exactly):\n"
        + relation_menu(spans) + "\n\n"
        f"Text: {text!r}\n"
        f"Entities: {json.dumps(ents)}\n"
        "Return the JSON array now."
    )


def parse_llm_json(text: str) -> list[dict]:
    """Strip accidental fences and parse a JSON array; never raises."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def resolve_type(rtype):
    """Map a proposed relation name onto (canonical type, swap head/tail).

    Returns (None, False) if the name is unknown. Extractors reach for the
    inverse of a type when the menu offers only one direction -- a real run
    produced `produced_by`, `stored_in`, `required_for` and `affected_by`, all
    discarded as unknown types even though each names a relation the ontology
    already has, read from the other end. Accepting them by swapping the
    arguments recovers the fact without widening the vocabulary.
    """
    if rtype in config.RELATION_ONTOLOGY:
        return rtype, False
    return config.RELATION_ALIASES.get(rtype, (None, False))


def _validate(item: dict, spans_by_id: dict[str, Span], sentence: str,
              sentence_offset: int):
    """(Relation, None) on success, (None, (code, reason)) on rejection.

    The reason string is the point. This bucket used to record only the raw item
    and its chunk, so 144 rejections in a real run were indistinguishable from
    each other and diagnosing them meant re-implementing these checks by hand
    against the IR. 88% turned out to share one cause -- a head or tail label the
    type would not accept -- which a reason code would have shown in seconds.
    """
    proposed = item.get("type")
    rtype, swap = resolve_type(proposed)
    if rtype is None:
        return None, ("unknown-type", f"unknown relation type {proposed!r}")
    head = spans_by_id.get(item.get("head_id"))
    tail = spans_by_id.get(item.get("tail_id"))
    if swap:
        head, tail = tail, head
    if head is None:
        return None, ("head-span-missing", "head span id not found"
                      + (" after inverting to %s" % rtype if swap else ""))

    spec = config.RELATION_ONTOLOGY[rtype]
    hier = config.LABEL_HIERARCHY
    if not ontology.label_satisfies(head.label, spec["head"], hier):
        return None, (f"head-label:{rtype}",
                      f"{rtype}: head {head.text!r} is labelled {head.label} "
                      f"but the type requires {sorted(spec['head'])}")
    if spec["tail"] is None:
        tail = None                                   # unary relation: drop any tail
    else:
        if tail is None:
            return None, ("tail-span-missing", f"{rtype}: tail span id not found")
        if not ontology.label_satisfies(tail.label, spec["tail"], hier):
            return None, (f"tail-label:{rtype}",
                          f"{rtype}: tail {tail.text!r} is labelled {tail.label} "
                          f"but the type requires {sorted(spec['tail'])}")

    conf = float(item.get("confidence", 0.0) or 0.0)
    if conf < config.RE_CONFIDENCE_FLOOR:
        return None, ("below-floor", f"confidence {conf} below the RE floor "
                                     f"{config.RE_CONFIDENCE_FLOOR}")
    evidence = item.get("evidence", "")
    idx = sentence.find(evidence) if evidence else -1
    if idx < 0:                       # ungrounded evidence -> reject (anti-hallucination)
        return None, ("ungrounded-evidence",
                      "evidence is not a substring of the chunk"
                      if evidence else "no evidence quoted")
    return Relation(
        rel_id="",  # assigned by caller
        head=head.span_id,
        tail=tail.span_id if tail else None,
        type=rtype,
        polarity=item.get("polarity", "affirmed"),
        evidence_start=sentence_offset + idx,
        evidence_end=sentence_offset + idx + len(evidence),
        score=conf,
    ), None


def validate_relation(item: dict, spans_by_id: dict[str, Span], sentence: str,
                      sentence_offset: int) -> Optional[Relation]:
    """Reject anything malformed, off-ontology, ungrounded, or below the floor.
    Returns a Relation with document-absolute evidence offsets, or None."""
    rel, _ = _validate(item, spans_by_id, sentence, sentence_offset)
    return rel


def _call_anthropic(system: str, user: str, schema=None) -> str:
    # `schema` is accepted for signature-compatibility with guided backends
    # (e.g. Meditron's guided_json); Claude follows the JSON instructions in the
    # prompt, so it's ignored here.
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=config.LLM_MODEL, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def extract_relations(doc: Document, call_llm=_call_anthropic,
                      progress=None) -> Document:
    """One LLM call per chunk (treated as one sentence context here for brevity;
    split into sentences for long chunks). `call_llm` is injectable for testing."""
    counter = itertools.count(1)
    spans_by_id = {s.span_id: s for s in doc.spans}
    chunks = doc.extractable_chunks(config.EXTRACTABLE_CHUNK_KINDS)
    for i, chunk in enumerate(chunks):
        if progress:
            progress("relations", f"{i + 1}/{len(chunks)}")
        spans = [s for s in doc.spans if s.chunk_id == chunk.chunk_id]
        if len(spans) < 1:
            continue
        pairs = candidate_pairs(spans)
        involved = {s.span_id for pair in pairs for s in pair}
        subset = [s for s in spans if s.span_id in involved]
        if not subset:
            continue
        prompt = build_re_prompt(chunk.text, subset, chunk.section_path,
                                 looks_structured(chunk.text))
        raw = call_llm(_SYSTEM, prompt, schema=RE_SCHEMA)
        if not raw:
            doc.needs_review.append({"stage": "re-no-response",
                                     "chunk": chunk.chunk_id,
                                     "reason": "backend returned no usable output"})
            continue
        for item in parse_llm_json(raw):
            rel, reason = _validate(item, spans_by_id, chunk.text, chunk.char_start)
            if rel is not None:
                rel.rel_id = f"r{next(counter):03d}"
                doc.relations.append(rel)
            else:
                code, text = reason
                doc.needs_review.append({"stage": "re", "code": code,
                                         "reason": text, "item": item,
                                         "chunk": chunk.chunk_id})
    flag_direction_conflicts(doc)
    guards.run_all(doc)
    return doc


def flag_direction_conflicts(doc) -> list[dict]:
    """Find asymmetric relations asserted in BOTH directions.

    `A caused_by B` together with `B caused_by A` cannot both be true, so their
    coexistence proves the extractor guessed at least one. That is worth
    detecting mechanically rather than trusting the prompt, because it is the
    one extraction error a reader is least likely to notice: each triple is
    individually plausible, and only the pair is absurd.

    Both are demoted to `uncertain` rather than dropped -- one of them is
    probably right, and silently deleting a correct fact is not an improvement
    on flagging two. Stage 6 keeps `uncertain` out of the asserted graph.
    """
    by_concept: dict[str, str] = {}
    for sp in doc.spans:
        by_concept[sp.span_id] = sp.uri or sp.text.lower()
    seen: dict[tuple, list] = {}
    for rel in doc.relations:
        if rel.tail is None:
            continue
        spec = config.RELATION_ONTOLOGY.get(rel.type, {})
        if spec.get("symmetric"):
            continue
        h, t = by_concept.get(rel.head), by_concept.get(rel.tail)
        if h is None or t is None or h == t:
            continue
        seen.setdefault((rel.type, h, t), []).append(rel)

    conflicts = []
    for (rtype, h, t), rels in seen.items():
        reverse = seen.get((rtype, t, h))
        if not reverse or h > t:            # report each pair once
            continue
        for rel in rels + reverse:
            rel.polarity = "uncertain"
        entry = {"stage": "re-direction", "relation": rtype,
                 "reason": "asserted in both directions; at least one is wrong",
                 "rel_ids": [r.rel_id for r in rels + reverse]}
        conflicts.append(entry)
        doc.needs_review.append(entry)
    return conflicts


# ---------------------------------------------------------------------------
# LLM modifier extraction (replaces the old dependency-parse heuristic)
# ---------------------------------------------------------------------------

def build_modifier_prompt(text: str, spans: list[Span], section_path=None,
                          structured: bool = False) -> str:
    ents = [{"id": s.span_id, "text": s.text, "type": s.label} for s in spans]
    return (
        _context_header(section_path, structured)
        + "Modifier types (use these names exactly):\n" + modifier_menu() + "\n\n"
        f"Text: {text!r}\n"
        f"Entities: {json.dumps(ents)}\n"
        "Return the JSON array now."
    )


def validate_modifier(item: dict, spans_by_id: dict[str, Span], text: str,
                      text_offset: int):
    """Reject unknown span/type or ungrounded text; else return (span_id, Modifier)
    with document-absolute offsets. Grounding is case-insensitive substring match."""
    span = spans_by_id.get(item.get("span_id"))
    mtype = item.get("type")
    # live, not the import-time snapshot: `--ontology` must be able to add a type
    if span is None or mtype not in allowed_modifier_types():
        return None
    mtext = item.get("text", "")
    idx = text.lower().find(mtext.lower()) if mtext else -1
    if idx < 0:                                    # ungrounded -> reject
        return None
    return span.span_id, Modifier(
        type=mtype, text=text[idx:idx + len(mtext)],
        char_start=text_offset + idx, char_end=text_offset + idx + len(mtext))


def extract_modifiers_llm(doc: Document, call_llm=_call_anthropic,
                          progress=None) -> Document:
    """One LLM call per chunk; attaches grounded modifiers to their spans. Handles
    titles/bullets (the section heading is injected) far better than a parser."""
    spans_by_id = {s.span_id: s for s in doc.spans}
    chunks = doc.extractable_chunks(config.EXTRACTABLE_CHUNK_KINDS)
    for i, chunk in enumerate(chunks):
        if progress:
            progress("modifiers", f"{i + 1}/{len(chunks)}")
        spans = [s for s in doc.spans if s.chunk_id == chunk.chunk_id]
        if not spans:
            continue
        structured = looks_structured(chunk.text)
        prompt = build_modifier_prompt(chunk.text, spans, chunk.section_path, structured)
        raw = call_llm(_MOD_SYSTEM, prompt, schema=MODIFIER_SCHEMA)
        if not raw:
            # The backend returned nothing (timeout, unparseable reply). Record
            # WHICH chunk, so this is re-runnable rather than an unexplained gap.
            doc.needs_review.append({"stage": "modifier-no-response",
                                     "chunk": chunk.chunk_id,
                                     "reason": "backend returned no usable output"})
            continue
        for item in parse_llm_json(raw):
            res = validate_modifier(item, spans_by_id, chunk.text, chunk.char_start)
            if res is None:
                doc.needs_review.append({"stage": "modifier", "item": item, "chunk": chunk.chunk_id})
                continue
            span_id, mod = res
            spans_by_id[span_id].modifiers.append(mod)
    return doc


def run(doc: Document, nlp=None, call_llm=_call_anthropic, progress=None) -> Document:
    doc = extract_entities(doc, nlp=nlp)                      # NER + negation
    if progress:
        progress("entities", len(doc.spans))
    doc = extract_modifiers_llm(doc, call_llm=call_llm, progress=progress)
    doc = extract_relations(doc, call_llm=call_llm, progress=progress)
    return doc
