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


def lexicon_uri(text: str) -> str:
    """Resolve a modifier surface form against `MODIFIER_VALUE_LEXICON`, or ''.

    Exact match first, then the longest lexicon key appearing as a whole word:
    surface forms arrive carrying articles and prepositions ("a deficiency in",
    "excess of thyroid hormone").
    """
    key = text.strip().lower()
    if key in config.MODIFIER_VALUE_LEXICON:
        return config.MODIFIER_VALUE_LEXICON[key]
    for candidate in sorted(config.MODIFIER_VALUE_LEXICON, key=len,
                            reverse=True):
        if re.search(rf"\b{re.escape(candidate)}\b", key):
            return config.MODIFIER_VALUE_LEXICON[candidate]
    return ""


def _value_uri(mod) -> tuple[str, bool]:
    """Resolve a modifier value to a concept URI. Returns (uri, needs_review).

    The lexicon is consulted BEFORE the Stage-3 link, and for a role-changing
    modifier the link is not consulted at all. A deviation value is a closed
    vocabulary of about thirty surface forms, and the linker is being asked to
    resolve a bare qualifier word with no useful context — which in a
    ten-document run sent `hasDeviation` to `sct:36976004` (Hypoparathyroidism,
    a disease) for "not enough PTH" and to `sct:88323005` (Adequate) for
    "adequate". `ont:Deficiency`, the value the ontology defines for exactly
    this, was never once used. Off-lexicon deviations now go to review instead
    of to a plausible-looking wrong code.
    """
    uri = lexicon_uri(mod.text)
    if uri:
        return uri, False
    if mod.uri and mod.type not in config.ROLE_CHANGING_MODIFIERS:
        return mod.uri, False
    # last resort: keep the surface form, flag for review
    return config.ONT + "value/" + det_id(mod.text.strip().lower()), True


def _all_role_changing(span) -> list:
    return [m for m in span.modifiers
            if m.type in config.ROLE_CHANGING_MODIFIERS]


def contradictory_deviations(span) -> list:
    """Role-changing modifiers on this span that resolve to opposite values.

    One span carrying both `deficiency` and `excess` describes nothing: the
    instance would assert `hasDeviation Deficiency` and `hasDeviation Excess` of
    the same node. It happened six times in a ten-document run, and no guard saw
    it because each modifier is individually well-formed.

    Only pairs named in `config.MODIFIER_VALUE_ANTONYMS` count. "Two different
    values" is the wrong test: insulin deficiency and insulin resistance are
    both true of the same hormone, and rejecting that pair would lose two sound
    modifiers to catch one bad one.
    """
    mods = _all_role_changing(span)
    values = {lexicon_uri(m.text) for m in mods}
    values.discard("")
    if any(pair <= values for pair in config.MODIFIER_VALUE_ANTONYMS):
        return mods
    return []


def redundant_deviations(span) -> list:
    """Role-changing modifiers the span's own text already states.

    "Iodine deficiency", "ADH deficiency" and "hormone deficiency" are spans
    that already denote the deviation, and Stage 3 links them to a concept that
    does too. Attaching `deviation: deficiency` on top mints a second node for
    the same thing and asserts it against the plain concept — which is where
    `ADH deficiency caused_by ADH` came from.
    """
    text = span.text.strip().lower()
    out = []
    for mod in _all_role_changing(span):
        value = lexicon_uri(mod.text)
        if not value:
            continue
        synonyms = [key for key, uri in config.MODIFIER_VALUE_LEXICON.items()
                    if uri == value]
        if any(re.search(rf"\b{re.escape(key)}\b", text) for key in synonyms):
            out.append(mod)
    return out


def role_changing(span) -> list:
    """Modifiers that change what this span denotes, not just how it looks.

    Empty unless the span's label can actually TAKE a deviation
    (`config.DEVIATION_APPLIES_TO`): "iodine [deficiency]" is a real concept,
    "hyperthyroidism [excess]" is a category error that mints a duplicate
    subject node for a fact already asserted about the disease itself. Also
    empty when the deviations contradict each other or merely restate the span,
    both of which `postcoordinate` files for review.
    """
    if span.label not in config.DEVIATION_APPLIES_TO:
        return []
    if contradictory_deviations(span):
        return []
    redundant = {id(m) for m in redundant_deviations(span)}
    return [m for m in _all_role_changing(span) if id(m) not in redundant]


def misapplied_role_changing(span) -> list:
    """Role-changing modifiers the span's type cannot carry.

    Reported by `postcoordinate` and modelled nowhere, never silently dropped.
    """
    if span.label in config.DEVIATION_APPLIES_TO:
        return []
    return _all_role_changing(span)


def modelled_modifiers(span) -> list:
    """The modifiers that become attribute triples on this span's instance.

    Everything except role-changing modifiers the span cannot carry, cannot
    carry consistently, or already states in its own text. Those are reported
    by `postcoordinate` and modelled nowhere: writing the attribute anyway is
    what put `hasDeviation` on `Hypoparathyroidism` and left the node unlabelled
    in a real run, which is the opposite of reporting it.
    """
    dropped = {id(m) for m in misapplied_role_changing(span)}
    dropped |= {id(m) for m in contradictory_deviations(span)}
    dropped |= {id(m) for m in redundant_deviations(span)}
    return [m for m in span.modifiers if id(m) not in dropped]


def _last_segment(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1] if uri else ""


def value_name(mod) -> str:
    """A READABLE name for a modifier value: 'deficiency', never '260372006'.

    The lexicon answers first, exactly as it does for the attribute value. A
    value linked by Stage 3 resolves to a terminology code, so composing the
    label from the URI produced `iodine [260372006]` in a real run — technically
    correct, unreadable, and worse than the wrong triple it replaced. The code
    still goes on the instance as the attribute VALUE; only the human-facing
    label prefers a word.
    """
    key = mod.text.strip().lower()
    name = _last_segment(lexicon_uri(key))
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


def instance_id(doc: Document, span, attributes) -> str:
    """The node URI for `span`'s post-coordinated instance.

    Keyed on the concept and the attributes, NOT on where the mention sits. The
    offset used to be part of the key, which made every mention its own node:
    "cortisol [deficiency]" and "Cortisol [deficiency]" came back as two rows of
    the `relations` query for one fact, and 73 instances in a ten-document run
    collapsed to 53 once identity was taken from what the node means. The
    source_id stays in the key because instance URIs are minted per document
    graph and two documents must not share a node.

    Re-running on unchanged input still produces identical ids, which is what
    the offset was there for.
    """
    return config.INST + det_id(doc.source.source_id, span.cui,
                                sorted(attributes))


def postcoordinate(doc: Document) -> Document:
    """Mint the instance node for every modified, linked, affirmed span.

    Returns the same Document with `instances` extended and any unresolvable
    modifier value, misapplied deviation, contradictory deviation pair or
    deviation the span already states recorded in `needs_review`. Spans that
    mean the same thing share one instance id, so `instances` may hold several
    records pointing at one node; only the first carries the label, so the node
    does not end up with one `rdfs:label` per mention.

    Idempotent: a span that already has an instance is skipped, so a resumed
    corpus run does not build a second parallel set of nodes.
    """
    started_at = time.time()
    console.announce_stage(4, "post-coordinate",
                           "modified spans -> instance nodes")
    # The ids are deterministic, so a second set of nodes would be invisible in
    # the graph and would steadily inflate both the IR and the review queue.
    already = {inst.source_span for inst in doc.instances}
    labelled = {inst.inst_id for inst in doc.instances if inst.label}
    candidates = [span for span in doc.spans
                  if span.modifiers and span.uri and not span.negated]
    console.announce_step(
        f"{console.format_count(len(candidates), 'linked, affirmed span')} "
        f"carry modifiers; {len(already)} already have an instance")
    minted_before = len(doc.instances)
    attribute_count, redirected, suppressed = 0, 0, 0
    for span in doc.spans:
        if not span.modifiers or not span.uri or span.negated:
            continue
        if span.span_id in already:
            continue
        for mod in misapplied_role_changing(span):
            doc.needs_review.append(
                {"stage": "postcoord-deviation-type", "span_id": span.span_id,
                 "modifier": mod.text, "span": span.text, "label": span.label,
                 "reason": f"a {mod.type} modifier was attached to a "
                           f"{span.label} span; only "
                           f"{list(config.DEVIATION_APPLIES_TO)} can carry one, "
                           f"so it was neither modelled nor redirected"})
        conflicting = contradictory_deviations(span)
        conflict_text = ", ".join(sorted(m.text for m in conflicting))
        for mod in conflicting:
            doc.needs_review.append(
                {"stage": "postcoord-deviation-conflict",
                 "span_id": span.span_id, "modifier": mod.text,
                 "span": span.text,
                 "reason": f"{span.text!r} carries deviations that disagree "
                           f"({conflict_text}); none was modelled"})
        for mod in redundant_deviations(span):
            doc.needs_review.append(
                {"stage": "postcoord-deviation-redundant",
                 "span_id": span.span_id, "modifier": mod.text,
                 "span": span.text,
                 "reason": f"{span.text!r} already states this deviation, so "
                           f"attaching it would mint a second node for the "
                           f"concept the span already denotes"})

        modelled = modelled_modifiers(span)
        if not modelled:
            suppressed += 1
            continue
        attrs: list[tuple[str, str]] = []
        for mod in modelled:
            prop = config.SNOMED_ATTR.get(mod.type, config.ONT + "hasModifier")
            val_uri, review = _value_uri(mod)
            attrs.append((prop, val_uri))
            if review:
                doc.needs_review.append(
                    {"stage": "postcoord", "span_id": span.span_id,
                     "modifier": mod.text, "type": mod.type,
                     "reason": f"{mod.text!r} is not in the modifier value "
                               f"lexicon; the instance carries a placeholder "
                               f"URI rather than a guessed terminology code"})
        inst_id = instance_id(doc, span, attrs)
        label = None if inst_id in labelled else compose_label(span)
        if label:
            labelled.add(inst_id)
        doc.instances.append(
            Instance(inst_id=inst_id, type_uri=span.uri, source_span=span.span_id,
                     attributes=attrs, label=label))
        attribute_count += len(attrs)
        if role_changing(span):
            redirected += 1
    minted = len(doc.instances) - minted_before
    distinct = len({inst.inst_id for inst in doc.instances})
    console.announce_detail(
        f"{console.format_count(minted, 'instance record')} minted over "
        f"{console.format_count(distinct, 'distinct node')}, "
        f"{console.format_count(attribute_count, 'attribute triple')}")
    if suppressed:
        console.announce_detail(
            f"{suppressed} span(s) modelled no attribute at all and were sent "
            f"to review instead")
    console.announce_detail(
        f"{redirected} relation endpoint(s) redirected onto an instance "
        f"by a role-changing modifier")
    console.announce_finished("stage 4", started_at)
    return doc
