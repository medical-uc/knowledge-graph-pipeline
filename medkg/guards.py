"""Deterministic post-extraction guards for Stage 2 relations.

Every check here is pure stdlib and runs over an already-extracted `Document`,
so it can be applied to a saved Stage-2/Stage-3 IR without paying for the LLM
again (`run.py --mode guards`).

WHY THIS MODULE EXISTS
----------------------
The prompt already states head/tail direction per relation type, and
`flag_direction_conflicts` already catches a fact asserted BOTH ways round. An
audit of a real thyroid run showed those two together still let 12 of 18
relations through wrong, in four families:

  1. `goiter caused_by iodine`          <- "A **deficiency in** dietary iodine"
     `hyperthyroidism caused_by thyroid hormones` <- "an **excess of**"
     `Hashimoto's caused_by thyroid peroxidase`   <- "autoimmune reactions **against**"
     `calcitonin inhibits bone`         <- "inhibiting bone **resorption**"
     The span is a bare noun; the source attributed the effect to a PHRASE built
     around that noun. The resulting triple asserts the reverse of the document,
     and reads as perfectly plausible medicine. -> `flag_governed_roles`

  2. `Graves' disease caused_by hyperthyroidism` <- "Graves' ... **is a cause of**
     hyperthyroidism". Reversal survived only where causation was expressed
     NOMINALLY; the verbal frame in the prompt's own example came out right, so
     the model was pattern-matching the example rather than applying the rule.
     -> `flag_causal_direction`

  3. `cognitive impairments caused_by thyroid hormones` <- quoted evidence was
     "A deficiency during pregnancy can result in cognitive impairments in the
     child", which does not contain the tail at all. Stage 2 checked that the
     quote came from the chunk, never that it covered the endpoints.
     -> `flag_ungrounded_endpoints`

  4. `hyperthyroidism caused_by thyroid hormones` alongside
     `hypothyroidism caused_by thyroid hormones` -- one cause, two opposite
     effects. Not a reversal, so the existing both-ways check cannot see it.
     -> `flag_antonym_contradictions`

DESIGN
------
* **Demote, never delete.** Everything here sets `polarity = "uncertain"` and
  files a `needs_review` entry naming the cue that fired. Stage 6 keeps
  `uncertain` out of the asserted graph but reifies it, so a flagged relation is
  still retrievable via the `flagged` query. Deleting a correct fact is not an
  improvement on flagging a suspect one.
* **Recall is deliberately traded for precision.** These are lexicons, so they
  will sometimes fire on a sound relation. That is the intended direction of
  failure and it matches invariant 7: a weaker signal must degrade into MORE
  REVIEW, never into worse triples.
* **Cue lists are data.** They live at module scope so a corpus with different
  phrasing can extend them without touching the logic.
"""
from __future__ import annotations

import re

from . import config

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# A phrase BEFORE the mention that changes what the mention denotes. "iodine"
# governed by "a deficiency in" is not iodine -- it is iodine deficiency, whose
# causal role is the opposite one.
GOVERNING_PRE = [
    # deviation: the abnormality is the cause, not the substance
    r"deficien(?:cy|t|cies)", r"insufficien(?:t|cy)", r"inadequate",
    r"lack of", r"absence of", r"deplet(?:ion|ed)", r"loss of",
    r"excess(?:ive)?", r"elevated", r"raised", r"overproduction",
    r"underproduction", r"hyper-?secretion", r"hypo-?secretion",
    r"too much", r"too little", r"impaired", r"reduced", r"decreased",
    r"increased", r"resistance to", r"unopposed", r"reverse", r"inactive",
    # the mention is the TARGET or the SUBSTRATE of a process, not an agent
    r"against", r"attacks?", r"antibodies (?:to|against)", r"autoantibodies",
    r"destruction of", r"removal of", r"removing", r"conversion of",
    r"converting", r"breakdown of", r"degradation of", r"excretion of",
    r"uptake of", r"transport of", r"binding of", r"levels? of",
    r"concentrations? of", r"amounts? of",
]

# A word IMMEDIATELY AFTER the mention showing the mention was only the first
# half of the real noun phrase. "bone" + "resorption" is a process, not an organ.
GOVERNING_POST = [
    r"resorption", r"secretion", r"synthesis", r"uptake", r"transport",
    r"excretion", r"metabolism", r"de-?iodination", r"iodination",
    r"conversion", r"production", r"release", r"formation", r"degradation",
    r"breakdown", r"clearance", r"turnover", r"storage", r"binding",
    r"atoms?", r"levels?", r"concentrations?", r"deficiency", r"excess",
    r"intake", r"content", r"activity", r"receptors?", r"antibodies",
    r"antagonists?", r"agonists?", r"replacement", r"supplementation",
    r"hormones?",          # "thyroid" swallowed by "thyroid hormone"
    # head nouns of compounds the NER routinely clips off the front of:
    # "antithyroid |drugs|", "hypothalamic-pituitary-thyroid |axis|",
    # "pituitary gland |dysfunction|"
    r"drugs?", r"medications?", r"medicines?", r"agents?", r"axis",
    r"therapy", r"therapies", r"treatments?", r"disorders?", r"dysfunction",
    r"failure", r"insufficiency", r"disease", r"syndrome", r"deficiency",
    r"systems?",           # "organ" swallowed by "organ systems"
]

# Words that name a category rather than a thing. As a whole span they carry no
# concept: `thyroid hormones finding_site organ` says nothing about which organ.
VACUOUS_SPANS = {
    "organ", "organs", "tissue", "tissues", "body", "system", "systems",
    "structure", "structures", "condition", "conditions", "disease",
    "disorder", "disorders", "hormone", "hormones", "level", "levels",
    "function", "cell", "cells", "gland", "glands", "patient", "patients",
    "area", "areas", "region", "regions", "process", "processes", "factor",
    "factors", "substance", "substances", "symptom", "symptoms",
}

# Relations describing where something sits. A post-coordinated deviation is an
# abstraction, not a locatable thing: `iodine [deficiency] located_in thyroid`
# is a category error even though `iodine located_in thyroid` would be fine.
STRUCTURAL_TYPES = {"located_in", "part_of", "composed_of", "connects_to",
                    "finding_site"}

# Causal connectives, split by which side of the connective the CAUSE sits on.
# Forward: the earlier-mentioned entity is the cause ("A leads to B").
CAUSAL_FORWARD = [
    r"causes", r"cause of", r"leads? to", r"leading to", r"results? in",
    r"resulting in", r"induces?", r"produces?", r"contributes? to",
    r"contributing to", r"progresses? to", r"gives? rise to",
    r"responsible for", r"predisposes? to", r"triggers?", r"precipitates?",
]
# Backward: the earlier-mentioned entity is the effect ("A results from B").
CAUSAL_BACKWARD = [
    r"caused by", r"due to", r"results? from", r"resulting from",
    r"arises? from", r"arising from", r"secondary to", r"consequence of",
    r"complication of", r"attributable to", r"owing to", r"stems? from",
    r"cause is", r"causes are", r"brought on by", r"provoked by",
]

# Chemical conversion, split the same way. Forward: the earlier-mentioned
# species is the STARTING material ("T4 into T3", "iodate is reduced to iodide").
CONVERSION_FORWARD = [
    r"into", r"converted to", r"converts? to", r"reduced to", r"oxidi[sz]ed to",
    r"deiodinated to", r"metaboli[sz]ed to", r"transformed to", r"yields?",
    r"to form", r"to produce", r"to yield", r"becomes?", r"giving",
]
# Backward: the earlier-mentioned species is the PRODUCT.
CONVERSION_BACKWARD = [
    r"derived from", r"formed from", r"produced from", r"converted from",
    r"reduced from", r"originates from", r"arises from", r"from",
]

# Relation types whose direction the causal lexicon can adjudicate: head is the
# effect / downstream member in all three.
CAUSAL_TYPES = {"caused_by", "complication_of"}
# risk_factor_for runs the other way (head is the risk factor), so it is scored
# with the orientation inverted rather than skipped.
CAUSAL_TYPES_INVERTED = {"risk_factor_for"}

# Morphological antonym pairs used to spot one cause with two opposite effects.
ANTONYM_PREFIXES = [("hyper", "hypo"), ("over", "under"), ("hyper-", "hypo-")]

PRE_WINDOW = 48      # chars before a mention searched for a governing phrase
MAX_GAP_WORDS = 4    # words allowed between the governing phrase and the mention
POST_WINDOW = 3      # a post-cue must be adjacent (one space or hyphen)


def _compile(patterns) -> re.Pattern:
    return re.compile(r"(?:" + "|".join(patterns) + r")", re.IGNORECASE)


_PRE_RE = _compile(GOVERNING_PRE)
# An abbreviation in parentheses may sit between the span and its head noun:
# the NER clips "hypothalamic-pituitary-thyroid" out of
# "hypothalamic-pituitary-thyroid (HPT) axis", so the cue is not adjacent.
_POST_RE = re.compile(
    r"^[\s\-]{0,%d}(?:\([A-Za-z0-9/\-]{1,12}\)[\s\-]{0,2})?(?:%s)\b"
    % (POST_WINDOW, "|".join(GOVERNING_POST)), re.IGNORECASE)
_FWD_RE = _compile(CAUSAL_FORWARD)
_BWD_RE = _compile(CAUSAL_BACKWARD)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chunk_for(doc, span):
    return doc.chunk_by_id(span.chunk_id) if span is not None else None


def _evidence_text(doc, rel, head) -> str:
    """The quote the extractor cited, recovered from its chunk. '' if unusable."""
    chunk = _chunk_for(doc, head)
    if chunk is None or rel.evidence_start is None or rel.evidence_end is None:
        return ""
    a = rel.evidence_start - chunk.char_start
    b = rel.evidence_end - chunk.char_start
    if a < 0 or b > len(chunk.text) or a >= b:
        return ""
    return chunk.text[a:b]


SENTENCE_ENDS = ".!?\n"


def sentence_window(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to the sentence(s) it falls inside. Pure; stdlib only.

    `text` is the chunk, `start`/`end` are offsets into it. Returns the widened
    pair, clipped to the chunk. A quote already spanning whole sentences comes
    back unchanged.
    """
    left = start
    while left > 0 and text[left - 1] not in SENTENCE_ENDS:
        left -= 1
    right = end
    while right < len(text) and text[right - 1] not in SENTENCE_ENDS:
        right += 1
    return left, right


def _evidence_scope(doc, rel, head) -> tuple[int, int]:
    """The document-offset range the cited quote is taken to support.

    The quote itself is not that range. Extractors quote the clause that carries
    the relation and leave the subject in the one before it -- 82% of the quotes
    in a ten-document run were not whole sentences, and the demotions that
    produced read like `stimulates` missing 'PTH' over the quote "on kidneys to
    enhance calcium reabsorption". The entity is one clause to the left, in the
    same sentence, and the relation is sound.

    Widening to sentence boundaries keeps what the check is actually for -- the
    support must be LOCAL, not gathered from across a 1,500-character chunk --
    while dropping a requirement the prompt cannot reliably enforce. Returns
    (start, end) in document offsets, or the raw quote range when the chunk
    cannot be recovered.
    """
    chunk = _chunk_for(doc, head)
    if chunk is None or rel.evidence_start is None or rel.evidence_end is None:
        return rel.evidence_start, rel.evidence_end
    a = rel.evidence_start - chunk.char_start
    b = rel.evidence_end - chunk.char_start
    if a < 0 or b > len(chunk.text) or a >= b:
        return rel.evidence_start, rel.evidence_end
    left, right = sentence_window(chunk.text, a, b)
    return chunk.char_start + left, chunk.char_start + right


def _evidence_scope_text(doc, rel, head) -> str:
    """The text of `_evidence_scope`, for the surface-form tests.

    '' when the chunk cannot be recovered.
    """
    chunk = _chunk_for(doc, head)
    start, end = _evidence_scope(doc, rel, head)
    if chunk is None or start is None or end is None:
        return ""
    a, b = start - chunk.char_start, end - chunk.char_start
    if a < 0 or b > len(chunk.text) or a >= b:
        return ""
    return chunk.text[a:b]


def _demote(doc, rel, stage: str, reason: str, **extra) -> dict:
    """Mark a relation uncertain and file it for review. Idempotent."""
    rel.polarity = "uncertain"
    entry = {"stage": stage, "rel_id": rel.rel_id, "relation": rel.type,
             "reason": reason}
    entry.update(extra)
    doc.needs_review.append(entry)
    return entry


def concept_of(span) -> str:
    """Identity for cross-relation comparison: the LINKED CONCEPT, not the span.

    Two mentions of hypothyroidism in different chunks are different spans and
    the same fact. Keying on span_id is why a first cut of
    `prune_subsumed_relations` left `cretinism caused_by hypothyroidism` sitting
    next to `cretinism complication_of hypothyroidism` in a real run: the pair
    was extracted from two chunks, so the ids never matched.
    `flag_direction_conflicts` already got this right; everything that reasons
    across relations has to.
    """
    return span.uri or span.text.strip().lower()


def _concept_pairs(doc):
    """rel -> (head_concept, tail_concept) for relations with both endpoints."""
    spans = {s.span_id: s for s in doc.spans}
    out = {}
    for rel in doc.relations:
        h = spans.get(rel.head)
        t = spans.get(rel.tail) if rel.tail else None
        if h is not None and t is not None:
            out[rel.rel_id] = (concept_of(h), concept_of(t))
    return out


def _endpoints(doc, rel):
    head = doc.span_by_id(rel.head)
    tail = doc.span_by_id(rel.tail) if rel.tail else None
    return head, tail


# ---------------------------------------------------------------------------
# G0 — the quote must actually contain the entities it relates
# ---------------------------------------------------------------------------

def flag_ungrounded_endpoints(doc) -> list[dict]:
    """Reject relations whose cited evidence does not cover both endpoints.

    Stage 2's grounding guard asks only that the quote be a substring of the
    chunk. That is not the same as the quote SUPPORTING the relation: given a
    long chunk, a model can cite one sentence while relating an entity from
    another. That is how `cognitive impairments caused_by thyroid hormones`
    arrived carrying the quote "A deficiency during pregnancy can result in
    cognitive impairments in the child" -- a sentence in which no thyroid
    hormone is mentioned, and whose actual subject ("a deficiency") is the
    opposite of the tail that got asserted.

    A concept may legitimately occur several times in a chunk and the linked
    span may be a different occurrence than the quoted one, so offset
    containment OR a surface-form match both count as grounded.

    All three tests run over `_evidence_scope` — the quote widened to the
    sentences it sits in — rather than over the quote itself. See that function
    for the measurement that motivated it.
    """
    # every mention of each concept, so an abbreviation counts as its expansion
    by_concept: dict[str, list] = {}
    for sp in doc.spans:
        by_concept.setdefault(concept_of(sp), []).append(sp)

    def mentioned_in_scope(rel, span, start, end) -> bool:
        """Is this CONCEPT inside [start, end), under any of its surface forms?

        Offsets first, then the span's own text, then any co-referent mention.
        The third test matters because scispacy resolves abbreviations: a span
        can carry the text "triiodothyronine" while the sentence says "T3", and
        an exact-string check calls that ungrounded. It also covers a concept
        stated twice in one chunk, where the linked span may be a different
        occurrence than the quoted one.
        """
        if start is None or end is None:
            return False
        if start <= span.char_start and span.char_end <= end:
            return True
        scope = _evidence_scope_text(doc, rel, span)
        if scope and span.text.lower() in scope.lower():
            return True
        return any(other.span_id != span.span_id
                   and start <= other.char_start and other.char_end <= end
                   for other in by_concept.get(concept_of(span), ()))

    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed":
            continue
        head, tail = _endpoints(doc, rel)
        if head is None:
            continue
        evidence = _evidence_text(doc, rel, head)
        if not evidence:
            continue
        start, end = _evidence_scope(doc, rel, head)
        for role, span in (("head", head), ("tail", tail)):
            if span is None or mentioned_in_scope(rel, span, start, end):
                continue
            out.append(_demote(
                doc, rel, "re-evidence-scope",
                f"cited evidence does not mention the {role} "
                f"({span.text!r}); neither the quote nor the sentence around "
                f"it can support this relation",
                evidence=evidence, missing=span.text,
                scope=_evidence_scope_text(doc, rel, head)))
            break
    return out


# ---------------------------------------------------------------------------
# G1 — the span is a fragment of a phrase that changes its role
# ---------------------------------------------------------------------------

def governing_phrase(text: str, start: int, end: int):
    """If the mention at [start, end) is governed by a role-changing phrase,
    return (position, cue). Otherwise None. Pure; the unit under test.

    Two shapes are recognised:
      * a phrase just BEFORE the mention -- "a deficiency in dietary |iodine|",
        "autoimmune reactions against |thyroid peroxidase|", "by removing
        |iodine| atoms";
      * a process noun immediately AFTER it -- "|bone| resorption",
        "|iodine| atoms", "insufficient |hormone| levels".
    Both mean the document is talking about something the span does not name.
    """
    pre = text[max(0, start - PRE_WINDOW):start]
    best = None
    for m in _PRE_RE.finditer(pre):
        gap = pre[m.end():]
        if "." in gap or ";" in gap:            # cue belongs to another clause
            continue
        if len(gap.split()) > MAX_GAP_WORDS:
            continue
        best = ("pre", m.group(0))              # keep the closest match
    if best:
        return best
    m = _POST_RE.match(text[end:end + POST_WINDOW + 24])
    if m:
        return ("post", m.group(0).strip(" -"))
    return None


def flag_governed_roles(doc) -> list[dict]:
    """Demote relations whose endpoint is only part of the phrase that carries
    the meaning.

    This is the single largest error family in a real run (9 of 12 defects) and
    the most dangerous, because the triple is not merely imprecise -- it asserts
    the reverse of the source. `goiter caused_by iodine` is the exact inverse of
    "a deficiency in dietary iodine can lead to ... goiter", and iodine is the
    TREATMENT for goiter.

    The structural repair is the `deviation` modifier plus Stage-4
    post-coordination, which lets the tail become an "iodine [deficiency]"
    instance node. This guard is the backstop for everything post-coordination
    does not catch, and it fires whether or not the modifier pass ran.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed":
            continue
        head, tail = _endpoints(doc, rel)
        if head is None:
            continue
        chunk = _chunk_for(doc, head)
        if chunk is None:
            continue
        for role, span in (("tail", tail), ("head", head)):
            if span is None or span.chunk_id != chunk.chunk_id:
                continue
            # A span already post-coordinated with a deviation modifier has
            # been repaired properly; do not flag it twice.
            if any(m.type in config.ROLE_CHANGING_MODIFIERS for m in span.modifiers):
                continue
            local_start = span.char_start - chunk.char_start
            local_end = span.char_end - chunk.char_start
            if local_start < 0 or local_end > len(chunk.text):
                continue
            found = governing_phrase(chunk.text, local_start, local_end)
            if not found:
                continue
            where, cue = found
            out.append(_demote(
                doc, rel, "re-governed-role",
                f"{role} {span.text!r} is governed by {cue!r} "
                f"({where}); the source is talking about the phrase, not the "
                f"bare concept",
                cue=cue, role=role, span_text=span.text))
            break
    return out


# ---------------------------------------------------------------------------
# G2 — causal direction, checked against the connective in the quote
# ---------------------------------------------------------------------------

def orientation(evidence: str, first_end: int, second_start: int,
                fwd_re=None, bwd_re=None):
    """Which side of the connective is the cause? -> 'forward' | 'backward' | None.

    'forward' means the EARLIER-mentioned entity is the cause. Only the text
    BETWEEN the two mentions is considered, and a window carrying cues of both
    classes returns None rather than a guess -- an ambiguous sentence should
    produce no verdict, not a coin flip.
    """
    if second_start <= first_end:
        return None
    between = evidence[first_end:second_start]
    fwd = bool((fwd_re or _FWD_RE).search(between))
    bwd = bool((bwd_re or _BWD_RE).search(between))
    if fwd == bwd:                     # neither, or contradictory
        return None
    return "forward" if fwd else "backward"


causal_orientation = orientation          # kept: the original, causal-only name


_CONV_FWD_RE = _compile(CONVERSION_FORWARD)
_CONV_BWD_RE = _compile(CONVERSION_BACKWARD)


def flag_conversion_direction(doc) -> list[dict]:
    """`converts_to` is directional and had no direction check.

    A run asserted `iodide converts_to iodate` from "iodate ions are reduced to
    iodide" -- exactly backwards, and in a metabolic graph a reversed reaction
    arrow is as wrong as a reversed cause. `flag_causal_direction` covers only
    the caused_by family, so this reuses the same machinery with a conversion
    lexicon. Note the orientation flips relative to caused_by: `converts_to`
    runs FROM substrate TO product, so a forward cue means the head should come
    first.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.type != "converts_to":
            continue
        head, tail = _endpoints(doc, rel)
        if head is None or tail is None:
            continue
        chunk = _chunk_for(doc, head)
        if chunk is None or tail.chunk_id != chunk.chunk_id:
            continue
        if head.char_start == tail.char_start:
            continue
        head_first = head.char_start < tail.char_start
        first_end, second_start = ((head.char_end, tail.char_start) if head_first
                                   else (tail.char_end, head.char_start))
        orient = orientation(chunk.text, first_end - chunk.char_start,
                             second_start - chunk.char_start,
                             _CONV_FWD_RE, _CONV_BWD_RE)
        if orient is None or head_first == (orient == "forward"):
            continue
        out.append(_demote(
            doc, rel, "re-conversion-direction",
            f"the sentence runs {'start-material first' if orient == 'forward' else 'product first'} "
            f"but {head.text!r} was used as the substrate; the reaction arrow is "
            f"probably reversed"))
    return out


def flag_causal_direction(doc) -> list[dict]:
    """Check `caused_by`-family direction against the sentence's own connective.

    `flag_direction_conflicts` needs the SAME fact asserted both ways before it
    can say anything. A confidently wrong single assertion is invisible to it,
    and that is what actually happens: `Graves' disease caused_by
    hyperthyroidism` was extracted once, from "Graves' disease ... is a frequent
    cause of hyperthyroidism".

    Notice which reversals survived the prompt: only the ones where causation
    was expressed nominally ("is a cause of", "the most common cause is"). The
    verbal frame used in the prompt's own worked example came out right every
    time. The model was matching the example, not applying the rule -- so the
    rule has to be enforced outside the model.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed":
            continue
        inverted = rel.type in CAUSAL_TYPES_INVERTED
        if rel.type not in CAUSAL_TYPES and not inverted:
            continue
        head, tail = _endpoints(doc, rel)
        if head is None or tail is None:
            continue
        chunk = _chunk_for(doc, head)
        if chunk is None or tail.chunk_id != chunk.chunk_id:
            continue
        h0, h1 = head.char_start, head.char_end
        t0, t1 = tail.char_start, tail.char_end
        if h0 == t0:
            continue
        head_first = h0 < t0
        first_end, second_start = (h1, t0) if head_first else (t1, h0)
        orient = causal_orientation(chunk.text, first_end - chunk.char_start,
                                    second_start - chunk.char_start)
        if orient is None:
            continue
        # forward => the earlier entity is the cause. For caused_by the head is
        # the EFFECT, so the head must be the later entity.
        head_should_be_first = (orient == "backward")
        if inverted:                      # risk_factor_for: head IS the cause
            head_should_be_first = (orient == "forward")
        if head_first == head_should_be_first:
            continue
        out.append(_demote(
            doc, rel, "re-causal-direction",
            f"the sentence reads {'cause-first' if orient == 'forward' else 'effect-first'} "
            f"but {head.text!r} was used as the head of {rel.type}; direction is "
            f"probably reversed",
            head_text=head.text, tail_text=tail.text, orientation=orient))
    return out


# ---------------------------------------------------------------------------
# G3 — one cause, two opposite effects
# ---------------------------------------------------------------------------

def are_antonyms(a: str, b: str) -> bool:
    """hyperthyroidism / hypothyroidism -> True. Pure, morphological."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return False
    for x, y in ANTONYM_PREFIXES:
        for p, q in ((x, y), (y, x)):
            if a.startswith(p) and b.startswith(q) and a[len(p):] == b[len(q):]:
                return True
    return False


def flag_antonym_contradictions(doc) -> list[dict]:
    """One cause cannot produce a condition and its opposite.

    A real run asserted `hyperthyroidism caused_by thyroid hormones` and
    `hypothyroidism caused_by thyroid hormones` side by side. Both come from
    dropping the deviation ("an excess of" / "insufficient"), and neither is a
    reversal, so the both-ways check sees nothing wrong. Read as a pair they are
    self-refuting, which makes them mechanically detectable -- the same argument
    that justifies `flag_direction_conflicts`, applied one step further out.
    """
    out = []
    by_tail: dict[tuple, list] = {}
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.tail is None:
            continue
        head, tail = _endpoints(doc, rel)
        if head is None or tail is None:
            continue
        key = (rel.type, tail.uri or tail.text.lower())
        by_tail.setdefault(key, []).append((head, rel))

    for (rtype, tail_key), items in sorted(by_tail.items()):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ha, ra), (hb, rb) = items[i], items[j]
                if not are_antonyms(ha.text, hb.text):
                    continue
                reason = (f"{ha.text!r} and {hb.text!r} are opposites but share "
                          f"one {rtype} tail; at least one is missing a "
                          f"deviation (excess / deficiency)")
                for rel in (ra, rb):
                    out.append(_demote(doc, rel, "re-antonym-conflict", reason,
                                       tail=tail_key))
    return out


# ---------------------------------------------------------------------------
# G4 — a hormone that stimulates the gland that secretes it
# ---------------------------------------------------------------------------

# Types meaning "A is the source of B".
PRODUCTION_TYPES = {"secretes", "synthesizes", "produces", "expresses"}
# Signed-POSITIVE control only. `inhibits` is deliberately absent: a hormone
# suppressing its own source is negative feedback, which is how every endocrine
# axis in the corpus actually works.
ACTIVATION_TYPES = {"stimulates", "regulates"}


def flag_positive_feedback_loops(doc) -> list[dict]:
    """`hypothalamus secretes TRH` + `TRH stimulates hypothalamus` -> flag.

    From the sentence "The hypothalamus releases thyrotropin-releasing hormone
    (TRH), which stimulates the pituitary gland to release TSH". The tail should
    be the pituitary; the extractor took the sentence's SUBJECT instead, which
    is a plausible-looking triple and a physiological absurdity: a releasing
    hormone driving its own release is runaway positive feedback.

    Only positive control is checked. `TRH inhibits hypothalamus` would be
    ordinary negative feedback and must pass, so the asymmetry in
    `ACTIVATION_TYPES` vs `PRODUCTION_TYPES` is load-bearing, not an oversight.
    """
    pairs = _concept_pairs(doc)
    produced: set[tuple] = set()
    for rel in doc.relations:
        if rel.polarity == "affirmed" and rel.type in PRODUCTION_TYPES \
                and rel.rel_id in pairs:
            produced.add(pairs[rel.rel_id])          # (source, product)

    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.type not in ACTIVATION_TYPES:
            continue
        if rel.rel_id not in pairs:
            continue
        head_c, tail_c = pairs[rel.rel_id]
        if (tail_c, head_c) not in produced:         # tail is not its source
            continue
        head, tail = _endpoints(doc, rel)
        out.append(_demote(
            doc, rel, "re-feedback-loop",
            f"{head.text!r} {rel.type} {tail.text!r}, but {tail.text!r} is "
            f"recorded as its source; a hormone driving its own secretion is "
            f"positive feedback, so the tail is probably the sentence's subject "
            f"rather than the actual target"))
    return out


# ---------------------------------------------------------------------------
# G5 — a concept related to itself
# ---------------------------------------------------------------------------

def flag_self_loops(doc) -> list[dict]:
    """Both endpoints resolve to one concept. `Thyroxine converts_to thyroxine`.

    Nothing converts to, causes, or is part of itself. The pair usually comes
    from a sentence naming a substance twice ("thyroxine (T4) ... converted to
    the active form"), where the extractor grabbed the wrong second mention.
    Symmetric types are exempt by declaration, not by guesswork.
    """
    pairs = _concept_pairs(doc)
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.rel_id not in pairs:
            continue
        if config.RELATION_ONTOLOGY.get(rel.type, {}).get("symmetric"):
            continue
        head, tail = _endpoints(doc, rel)
        head_c, tail_c = pairs[rel.rel_id]
        # URI identity OR surface identity. `T3 converts_to T3` survived a URI-only
        # check because Stage 3 linked the two mentions to different concepts --
        # which is itself a defect, and either way the triple is wrong.
        same_text = head.text.strip().lower() == tail.text.strip().lower()
        if head_c != tail_c and not same_text:
            continue
        out.append(_demote(
            doc, rel, "re-self-loop",
            f"head and tail are the same concept ({head.text!r}); "
            f"{rel.type} cannot hold between a thing and itself"))
    return out


# ---------------------------------------------------------------------------
# G6 — spans that are not concepts
# ---------------------------------------------------------------------------

# A span that is only an enumerator has lost its head noun: "Type 1 and Type 2
# deiodinases" yields the spans "Type 1" and "Type 2", and `Type 3 catalyzes T3`
# was asserted from one of them.
FRAGMENT_RE = re.compile(
    r"^(?:type|class|stage|grade|group|phase|step|form)\s*[0-9ivx]+$", re.IGNORECASE)


def flag_fragment_endpoints(doc) -> list[dict]:
    """Demote relations whose endpoint is an enumerator with no head noun.

    Narrow on purpose. A general "is this span a real concept" test needs a
    parser; this catches the one pattern the corpus actually produces, and says
    so rather than pretending to more coverage.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed":
            continue
        head, tail = _endpoints(doc, rel)
        for role, span in (("head", head), ("tail", tail)):
            if span is None or not FRAGMENT_RE.match(span.text.strip()):
                continue
            out.append(_demote(
                doc, rel, "re-fragment-span",
                f"{role} {span.text!r} is an enumerator with no head noun; the "
                f"concept it numbers was not captured"))
            break
    return out


def flag_vacuous_endpoints(doc) -> list[dict]:
    """An endpoint that is a bare category word names nothing.

    `thyroid hormones finding_site organ` passes every type check -- "organ" is
    a BodyStructure -- and tells a reader nothing. Exact whole-span matches
    only, so "parafollicular cells" and "thyroid gland" are untouched.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed":
            continue
        head, tail = _endpoints(doc, rel)
        for role, span in (("tail", tail), ("head", head)):
            if span is None or span.text.strip().lower() not in VACUOUS_SPANS:
                continue
            out.append(_demote(
                doc, rel, "re-vacuous-endpoint",
                f"{role} {span.text!r} is a bare category word, not a concept; "
                f"the specific entity the sentence named was not captured"))
            break
    return out


def flag_structural_deviation(doc) -> list[dict]:
    """A deviation cannot be located, contained, or made of anything.

    Post-coordination is right to build `iodine [deficiency]` for a causal tail,
    but the same node reached `located_in thyroid` in a real run. A shortage is
    not somewhere.
    """
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.type not in STRUCTURAL_TYPES:
            continue
        head, tail = _endpoints(doc, rel)
        for role, span in (("head", head), ("tail", tail)):
            if span is None:
                continue
            if not any(m.type in config.ROLE_CHANGING_MODIFIERS
                       for m in span.modifiers):
                continue
            out.append(_demote(
                doc, rel, "re-structural-deviation",
                f"{role} {span.text!r} carries a deviation; {rel.type} needs a "
                f"thing, and a deficiency or excess is not one"))
            break
    return out


def flag_enzyme_as_substrate(doc) -> list[dict]:
    """A protein the document calls an enzyme is not a reaction product.

    `rT3 converts_to Deiodinases`, asserted alongside `Deiodinases catalyzes
    thyroxine`. The second relation is what identifies deiodinases as the
    catalyst, so the first is self-refuting -- the same cross-relation argument
    as the feedback-loop check, and it needs no lexicon at all: the document
    supplies the evidence.
    """
    pairs = _concept_pairs(doc)
    enzymes = {pairs[r.rel_id][0] for r in doc.relations
               if r.type == "catalyzes" and r.polarity == "affirmed"
               and r.rel_id in pairs}
    if not enzymes:
        return []
    out = []
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.type != "converts_to":
            continue
        if rel.rel_id not in pairs:
            continue
        hit = [c for c in pairs[rel.rel_id] if c in enzymes]
        if not hit:
            continue
        head, tail = _endpoints(doc, rel)
        out.append(_demote(
            doc, rel, "re-enzyme-as-substrate",
            f"{head.text!r} converts_to {tail.text!r}, but the document also "
            f"records one of them catalysing a reaction; an enzyme is not a "
            f"substrate or product"))
    return out


# ---------------------------------------------------------------------------
# Property hierarchy — assert the specific, infer the parent
# ---------------------------------------------------------------------------

def prune_subsumed_relations(doc) -> list[dict]:
    """Drop a relation when a more specific one holds over the same CONCEPT pair.

    A run produced `TRH regulates TSH` next to `TRH stimulates TSH`, and
    `cretinism caused_by hypothyroidism` next to `cretinism complication_of
    hypothyroidism`. Neither pair is wrong; both are redundant, and the
    ontology's own gloss already says to prefer the specific type ("Use only
    when direction of effect is unstated").

    Asserting only the child is safe because `stimulates rdfs:subPropertyOf
    regulates` now ships in the schema, so Stage 7 materializes the parent into
    `urn:graph:inferred`. The general fact is still queryable; it is just marked
    as derived rather than claimed, which is the distinction Stage 7 exists to
    preserve.

    Matching is on linked concepts, not span ids -- see `concept_of`.

    The parent is recorded in `needs_review` rather than vanishing: an
    extraction that was made and then withdrawn should stay auditable.
    """
    parents = config.RELATION_PARENTS
    if not parents:
        return []
    pairs = _concept_pairs(doc)
    specific: set[tuple] = set()
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.rel_id not in pairs:
            continue
        if rel.type in parents:
            specific.add((parents[rel.type], *pairs[rel.rel_id]))

    kept, removed = [], []
    for rel in doc.relations:
        key = (rel.type, *pairs[rel.rel_id]) if rel.rel_id in pairs else None
        if rel.polarity == "affirmed" and key in specific:
            entry = {"stage": "re-subsumed", "rel_id": rel.rel_id,
                     "relation": rel.type,
                     "reason": "redundant: a subPropertyOf this type was asserted "
                               "over the same concept pair; Stage 7 will infer it"}
            doc.needs_review.append(entry)
            removed.append(entry)
        else:
            kept.append(rel)
    doc.relations = kept
    return removed


def dedupe_relations(doc) -> list[dict]:
    """Collapse relations that assert the same fact about the same concepts.

    A run emitted `TRH stimulates TSH` twice, at confidence 0.9 and 0.7, because
    the pair is stated in two chunks. Stage 6 writes both, and RDF is a set, so
    the triple appears once but carries two contradictory confidence
    annotations -- and the `relations` query, which LEFT-JOINs the annotation,
    prints the fact twice.

    The highest-scoring copy wins. That is not a claim that the higher score is
    better calibrated (see `flag_degenerate_confidence` -- it usually is not);
    it is just a stable, deterministic rule for picking one, and the discarded
    copies stay in `needs_review` with their scores so the disagreement is not
    lost.
    """
    pairs = _concept_pairs(doc)
    groups: dict[tuple, list] = {}
    for rel in doc.relations:
        if rel.polarity != "affirmed" or rel.rel_id not in pairs:
            continue
        groups.setdefault((rel.type, *pairs[rel.rel_id]), []).append(rel)

    dropped_ids, out = set(), []
    for key, rels in groups.items():
        if len(rels) < 2:
            continue
        winner = max(rels, key=lambda r: (r.score if r.score is not None else -1.0))
        for rel in rels:
            if rel is winner:
                continue
            dropped_ids.add(rel.rel_id)
            entry = {"stage": "re-duplicate", "rel_id": rel.rel_id,
                     "relation": rel.type, "score": rel.score,
                     "reason": f"same fact as {winner.rel_id} (score "
                               f"{winner.score}) over the same concept pair"}
            doc.needs_review.append(entry)
            out.append(entry)
    if dropped_ids:
        doc.relations = [r for r in doc.relations if r.rel_id not in dropped_ids]
    return out


# ---------------------------------------------------------------------------
# Diagnostic — is the confidence floor doing anything at all?
# ---------------------------------------------------------------------------

def flag_degenerate_confidence(doc, min_relations: int = 8,
                              share: float = 0.9) -> list[dict]:
    """Warn when self-reported confidence carries no information.

    In the audited run all 18 relations scored exactly 0.9, which makes
    `RE_CONFIDENCE_FLOOR` inert: nothing can ever fall below it, so the safety
    valve in invariant 6 is wired up but never trips. This does not demote
    anything -- it tells the operator that one of their two quality controls is
    silently switched off, which is worth more than another flagged triple.
    """
    scored = [r.score for r in doc.relations if r.score is not None]
    if len(scored) < min_relations:
        return []
    counts: dict[float, int] = {}
    for s in scored:
        counts[round(float(s), 3)] = counts.get(round(float(s), 3), 0) + 1
    value, n = max(counts.items(), key=lambda kv: kv[1])
    if n / len(scored) < share:
        return []
    entry = {"stage": "re-confidence-degenerate",
             "reason": f"{n}/{len(scored)} relations report the identical "
                       f"confidence {value}; the RE floor "
                       f"({config.RE_CONFIDENCE_FLOOR}) cannot filter anything. "
                       f"Treat these scores as constants, not evidence.",
             "value": value, "count": n, "total": len(scored)}
    doc.needs_review.append(entry)
    return [entry]


# ---------------------------------------------------------------------------

def backfill_review_codes(doc) -> int:
    """Recover the rejection code for Stage-2 entries written before codes existed.

    The entry keeps the raw `item` and its chunk, which is everything
    `_validate` needs, so an IR from an older run can be triaged without
    re-extracting it. Also re-checks against the CURRENT ontology, so an entry
    that a new label_hierarchy or alias would now accept is marked
    `would-now-pass` rather than left in the pile.
    """
    from . import stage2_extract          # local: stage2 imports this module
    spans = {s.span_id: s for s in doc.spans}
    chunks = {c.chunk_id: c for c in doc.chunks}
    filled = 0
    for entry in doc.needs_review:
        if entry.get("stage") != "re" or entry.get("code") or "item" not in entry:
            continue
        chunk = chunks.get(entry.get("chunk"))
        if chunk is None:
            continue
        rel, problem = stage2_extract._validate(entry["item"], spans, chunk.text,
                                                chunk.char_start)
        if rel is not None:
            entry["code"] = "would-now-pass"
            entry["reason"] = (f"the current ontology accepts this as "
                               f"{rel.type!r}; re-run Stage 2 to recover it")
        else:
            entry["code"], entry["reason"] = problem
        filled += 1
    return filled


def summarize_review(doc) -> list[tuple]:
    """(stage, code, count, one example reason), commonest first.

    `needs_review` is a single flat list holding everything every stage was
    unsure about, which is the right shape to write and a poor one to read: a
    real run put 167 entries in it and the only way to tell an ontology
    misconfiguration from a genuine extraction failure was to re-implement
    Stage 2's checks by hand against the IR. Grouping by (stage, code) turns
    that into one glance -- 88% of that run shared a single cause.
    """
    backfill_review_codes(doc)
    groups: dict[tuple, list] = {}
    for entry in doc.needs_review:
        key = (entry.get("stage", "?"), entry.get("code") or "-")
        groups.setdefault(key, []).append(entry)
    rows = [(stage, code, len(items),
             next((i.get("reason", "") for i in items if i.get("reason")), ""))
            for (stage, code), items in groups.items()]
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))
    return rows


def run_all(doc) -> dict:
    """Every guard, in dependency order. Returns a per-guard count report.

    Flags run before pruning so a relation is never dropped as "redundant"
    because of a sibling that is itself suspect.
    """
    # The antonym check reasons about the extractor's output AS A SET, so it
    # runs before anything has been demoted -- otherwise a pair is only
    # reported as self-contradictory when neither half happened to trip a
    # per-relation guard first, and the most informative finding is the one
    # most easily suppressed.
    report = {
        "antonym_conflict":   len(flag_antonym_contradictions(doc)),
        "evidence_scope":     len(flag_ungrounded_endpoints(doc)),
        "governed_role":      len(flag_governed_roles(doc)),
        "causal_direction":   len(flag_causal_direction(doc)),
        "conversion_direction": len(flag_conversion_direction(doc)),
        "feedback_loop":      len(flag_positive_feedback_loops(doc)),
        "self_loop":          len(flag_self_loops(doc)),
        "fragment_span":      len(flag_fragment_endpoints(doc)),
        "vacuous_endpoint":   len(flag_vacuous_endpoints(doc)),
        "structural_deviation": len(flag_structural_deviation(doc)),
        "enzyme_as_substrate": len(flag_enzyme_as_substrate(doc)),
        # dedupe before pruning: two copies of the child must not let the parent
        # survive because only one of them matched.
        "duplicate":          len(dedupe_relations(doc)),
        "subsumed":           len(prune_subsumed_relations(doc)),
        "degenerate_conf":    len(flag_degenerate_confidence(doc)),
    }
    report["affirmed_remaining"] = sum(
        1 for r in doc.relations if r.polarity == "affirmed")
    return report
