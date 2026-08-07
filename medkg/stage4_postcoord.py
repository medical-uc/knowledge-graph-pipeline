"""Stage 4 — post-coordination.

For each linked span that carries modifiers, mint a deterministic INSTANCE node
that instantiates the concept, and emit one attribute triple per modifier
(attribute_property_uri -> value_uri). Pure stdlib — fully testable offline.

Post-coordination is not only a fidelity nicety. Some modifiers change what the
span DENOTES rather than merely qualifying it: "a deficiency in dietary iodine"
is not a kind of iodine, and a relation whose tail is the bare iodine concept
asserts that iodine causes goiter — the exact inverse of the source, and the
opposite of the clinical advice. `config.ROLE_CHANGING_MODIFIERS` names those
types, and Stage 6 redirects relation endpoints carrying one onto the instance
minted here. Everything else (severity, laterality, ...) keeps pointing at the
shared concept, because routing every modified mention through an instance would
shatter the graph into per-occurrence nodes and defeat entity linking.
"""
from __future__ import annotations

import hashlib
import re
import time

from .ir import Document, Instance
from . import config
from . import console


def det_id(*parts) -> str:
    """Deterministic short id from stable inputs -> idempotent re-runs."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:12]


def _value_uri(mod) -> tuple[str, bool]:
    """Resolve a modifier value to a concept URI. Returns (uri, needs_review)."""
    if mod.uri:                                   # linked in Stage 3
        return mod.uri, False
    key = mod.text.strip().lower()
    if key in config.MODIFIER_VALUE_LEXICON:      # config fallback
        return config.MODIFIER_VALUE_LEXICON[key], False
    # last resort: keep the surface form, flag for review
    return config.ONT + "value/" + det_id(key), True


def _all_role_changing(span) -> list:
    return [m for m in span.modifiers if m.type in config.ROLE_CHANGING_MODIFIERS]


def role_changing(span) -> list:
    """Modifiers that change what this span denotes, not just how it looks.

    Empty unless the span's label can actually TAKE a deviation
    (`config.DEVIATION_APPLIES_TO`): "iodine [deficiency]" is a real concept,
    "hyperthyroidism [excess]" is a category error that mints a duplicate
    subject node for a fact already asserted about the disease itself.
    """
    if span.label not in config.DEVIATION_APPLIES_TO:
        return []
    return _all_role_changing(span)


def misapplied_role_changing(span) -> list:
    """Role-changing modifiers the span's type cannot carry. Reported, not silent."""
    if span.label in config.DEVIATION_APPLIES_TO:
        return []
    return _all_role_changing(span)


def _last_segment(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1] if uri else ""


def value_name(mod) -> str:
    """A READABLE name for a modifier value: 'deficiency', never '260372006'.

    The lexicon is consulted BEFORE `mod.uri`, which inverts the priority used
    for the attribute triple in `_value_uri`, and the inversion is the point. A
    value linked by Stage 3 resolves to a terminology code, so composing the
    label from the URI produced `iodine [260372006]` in a real run — technically
    correct, unreadable, and worse than the wrong triple it replaced. The code
    still goes on the instance as the attribute VALUE; only the human-facing
    label prefers a word.
    """
    key = mod.text.strip().lower()
    uri = config.MODIFIER_VALUE_LEXICON.get(key)
    if not uri:
        # surface forms carry articles and prepositions: "a deficiency in"
        for k in sorted(config.MODIFIER_VALUE_LEXICON, key=len, reverse=True):
            if re.search(rf"\b{re.escape(k)}\b", key):
                uri = config.MODIFIER_VALUE_LEXICON[k]
                break
    name = _last_segment(uri or "")
    if not name:
        linked = _last_segment(mod.uri or "")
        name = linked if linked and not linked.isdigit() else key
    return name.lower()


def compose_label(span):
    """'iodine' + deviation 'A deficiency in' -> 'iodine [deficiency]'.

    The bracket form is deliberate: it keeps the linked concept's own surface
    form intact and readable at the front, so `goiter caused_by iodine
    [deficiency]` reads correctly in the `relations` query while still being
    obviously a composed node rather than a term from a terminology.

    None when nothing role-changing applies, so Stage 6 emits no label and the
    instance does not shadow its own concept in the `concepts` query.
    """
    quals = [value_name(m) for m in role_changing(span)]
    return f"{span.text} [{', '.join(quals)}]" if quals else None


def instance_by_span(doc: Document) -> dict[str, str]:
    """span_id -> instance URI, for spans whose instance REPLACES the concept as
    a relation endpoint. Stage 6 reads this; keeping the rule here means the two
    stages cannot disagree about which endpoints were rewritten."""
    spans = {s.span_id: s for s in doc.spans}
    out = {}
    for inst in doc.instances:
        span = spans.get(inst.source_span)
        if span is not None and role_changing(span):
            out[inst.source_span] = inst.inst_id
    return out


def postcoordinate(doc: Document) -> Document:
    """Mint one instance node per modified, linked, affirmed span.

    Returns the same Document with `instances` extended and any unresolvable
    modifier value or misapplied deviation recorded in `needs_review`.
    Idempotent: a span that already has an instance is skipped, so a resumed
    corpus run does not build a second parallel set of nodes.
    """
    started_at = time.time()
    console.announce_stage(4, "post-coordinate",
                           "modified spans -> instance nodes")
    # The ids are deterministic, so a second set of nodes would be invisible in
    # the graph and would steadily inflate both the IR and the review queue.
    already = {inst.source_span for inst in doc.instances}
    candidates = [span for span in doc.spans
                  if span.modifiers and span.uri and not span.negated]
    console.announce_step(
        f"{console.format_count(len(candidates), 'linked, affirmed span')} "
        f"carry modifiers; {len(already)} already have an instance")
    minted_before = len(doc.instances)
    attribute_count, redirected = 0, 0
    for span in doc.spans:
        if not span.modifiers or not span.uri or span.negated:
            continue
        if span.span_id in already:
            continue
        inst_id = config.INST + det_id(doc.source.source_id, span.char_start, span.cui)
        attrs: list[tuple[str, str]] = []
        for mod in span.modifiers:
            prop = config.SNOMED_ATTR.get(mod.type, config.ONT + "hasModifier")
            val_uri, review = _value_uri(mod)
            attrs.append((prop, val_uri))
            if review:
                doc.needs_review.append(
                    {"stage": "postcoord", "span_id": span.span_id,
                     "modifier": mod.text, "type": mod.type})
        for mod in misapplied_role_changing(span):
            doc.needs_review.append(
                {"stage": "postcoord-deviation-type", "span_id": span.span_id,
                 "modifier": mod.text, "span": span.text, "label": span.label,
                 "reason": f"a {mod.type} modifier was attached to a "
                           f"{span.label} span; only "
                           f"{list(config.DEVIATION_APPLIES_TO)} can carry one, "
                           f"so the relation endpoint was left on the concept"})
        doc.instances.append(
            Instance(inst_id=inst_id, type_uri=span.uri, source_span=span.span_id,
                     attributes=attrs, label=compose_label(span)))
        attribute_count += len(attrs)
        if role_changing(span):
            redirected += 1
    minted = len(doc.instances) - minted_before
    console.announce_detail(
        f"{console.format_count(minted, 'instance')} minted, "
        f"{console.format_count(attribute_count, 'attribute triple')}")
    console.announce_detail(
        f"{redirected} relation endpoint(s) redirected onto an instance "
        f"by a role-changing modifier")
    console.announce_finished("stage 4", started_at)
    return doc
