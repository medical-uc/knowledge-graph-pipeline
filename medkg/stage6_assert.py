"""Stage 6 — assert the enriched Document into RDF.

Design: `build_quads` is PURE PYTHON and returns (quads, annotations) so the
triple-generation logic is unit-testable with only stdlib. `to_dataset` is a
thin rdflib adapter (lazy import) that loads them into a named-graph Dataset,
attaching per-triple confidence via RDF-star, with rdf:Statement reification as
a portable fallback.

Key properties enforced here:
  * one named graph per source-version; provenance asserted ONCE on the graph
  * deterministic instance/assertion URIs -> idempotent re-runs
  * negated/uncertain relations become reified assertion nodes, never dropped,
    never asserted as fact
"""
from __future__ import annotations

from collections import namedtuple
from typing import Optional

from .ir import Document
from . import config
from . import stage5_images
from . import stage4_postcoord
from .stage4_postcoord import det_id

Lit = namedtuple("Lit", ["value"])            # marks an object literal
# quad:       (s, p, o, graph)          o may be str-uri or Lit
# annotation: ((s, p, o), {prop: val}, graph)


def _cap(polarity: str) -> str:
    return {"negated": "Negated", "uncertain": "Uncertain", "affirmed": "Affirmed"}.get(
        polarity, polarity.title())


def build_quads(doc: Document, snomed_version: str = config.SNOMED_VERSION):
    """Return (quads, annotations). Pure stdlib."""
    quads: list[tuple] = []
    annotations: list[tuple] = []
    g = config.GRAPH + doc.source.source_id
    pg = config.GRAPH + "provenance"
    spans = {s.span_id: s for s in doc.spans}
    # Endpoints whose meaning is carried by a post-coordinated instance rather
    # than the bare concept -- "iodine [deficiency]", not "iodine". Pointing a
    # caused_by tail at the concept would assert that iodine causes goiter.
    postcoord = stage4_postcoord.instance_by_span(doc)

    def endpoint(span):
        """The URI a relation should actually attach to for this span."""
        return postcoord.get(span.span_id, span.uri)

    # --- relations ---------------------------------------------------------
    for rel in doc.relations:
        head = spans.get(rel.head)
        tail = spans.get(rel.tail) if rel.tail else None
        if head is None or head.uri is None:
            continue
        pred = config.RELATION_ONTOLOGY[rel.type]["uri"]

        if rel.polarity == "affirmed" and tail is not None and tail.uri:
            triple = (endpoint(head), pred, endpoint(tail))
            quads.append((*triple, g))
            if rel.score is not None:
                annotations.append((triple, {
                    config.ONT + "confidence": Lit(round(rel.score, 3)),
                    config.ONT + "extractedBy": Lit("llm-re"),
                }, g))
        else:
            # negated / uncertain (or tail-less) -> reified assertion node
            stmt = config.INST + "assert/" + det_id(
                doc.source.source_id, rel.rel_id, head.uri, rel.type)
            quads += [
                (stmt, config.RDF_TYPE, config.ONT + "ClinicalAssertion", g),
                (stmt, config.ONT + "about", endpoint(head), g),
                (stmt, config.ONT + "assertionType", config.RELATION_ONTOLOGY[rel.type]["uri"], g),
                (stmt, config.ONT + "polarity", config.ONT + _cap(rel.polarity), g),
            ]
            if tail is not None and tail.uri:
                quads.append((stmt, config.ONT + "target", endpoint(tail), g))

    # --- concept labels ----------------------------------------------------
    # Without these, every query answer is a bare `http://snomed.info/id/...`
    # and the graph is effectively unreadable. The label is the document's own
    # most frequent surface form for that concept -- the mention, not the
    # ontology's preferred term, which the triplestore can supply later from a
    # SNOMED release if you want the canonical name.
    forms: dict[str, dict[str, int]] = {}
    for sp in doc.spans:
        if sp.uri:
            forms.setdefault(sp.uri, {})
            forms[sp.uri][sp.text] = forms[sp.uri].get(sp.text, 0) + 1
    for uri, counts in forms.items():
        best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        quads.append((uri, config.RDFS_LABEL, Lit(best), g))
        quads.append((uri, config.ONT + "mentionCount", Lit(sum(counts.values())), g))

    # --- instance nodes (Stage 4) -----------------------------------------
    for inst in doc.instances:
        quads.append((inst.inst_id, config.RDF_TYPE, inst.type_uri, g))
        for prop, val in inst.attributes:
            quads.append((inst.inst_id, prop, val, g))
        # An instance used as a relation endpoint renders as a bare inst/<hash>
        # in every query answer unless it is labelled. `relations` COALESCEs
        # rdfs:label, so this is what makes the repaired triple read as
        # "goiter caused_by iodine [deficiency]".
        if inst.label:
            quads.append((inst.inst_id, config.RDFS_LABEL, Lit(inst.label), g))

    # --- image nodes + Stage 5 depicts edges -------------------------------
    # The figure node is registered even if Stage 5 hasn't run; `depicts` edges
    # appear only for figures whose text was bridged and linked.
    quads += stage5_images.build_depicts_quads(doc, g)

    # --- provenance: asserted ONCE about the whole graph -------------------
    # Stage 1's input is a REWRITTEN document, not the original material, so the
    # two are asserted separately. `dcterms:source` names what the material came
    # from; `prov:wasGeneratedBy` names the model that wrote the prose the
    # triples were actually extracted from. Collapsing them would let a claim
    # traceable to fresh LLM output pass as one traceable to the textbook.
    quads += [
        (g, config.DCTERMS + "source",
         Lit(doc.source.origin or doc.source.title), pg),
        (g, config.DCTERMS + "hasVersion", Lit(doc.source.edition), pg),
        (g, config.ONT + "snomedVersion", Lit(snomed_version), pg),
    ]
    if doc.source.generator:
        agent = config.INST + "agent/" + det_id("agent", doc.source.generator)
        quads += [
            (agent, config.RDF_TYPE, config.SOFTWARE_AGENT, pg),
            (agent, config.RDFS_LABEL, Lit(doc.source.generator), pg),
            (g, config.WAS_GENERATED_BY, agent, pg),
        ]
        if doc.source.origin:
            quads.append((g, config.WAS_DERIVED_FROM, Lit(doc.source.origin), pg))
        if doc.source.generated_at:
            quads.append((g, config.GENERATED_AT, Lit(doc.source.generated_at), pg))
    return quads, annotations


# ---------------------------------------------------------------------------
# rdflib adapter (lazy)
# ---------------------------------------------------------------------------

def _term(x):
    from rdflib import URIRef, Literal
    return Literal(x.value) if isinstance(x, Lit) else URIRef(x)


def looks_malformed_uri(text) -> bool:
    """Would rdflib accept this URI now and refuse to serialize it later?

    This is the exact failure this module guards against. A turtle parser
    WITHOUT RDF-star support does not reject `<< <http://x> <http://y> ... >>`.
    It reads the leading `<` as the start of a URI, consumes up to the next `>`,
    and yields the term `< <http://x` — emitting a warning, not an exception.
    The parse "succeeds", the annotation is silently corrupt, and the traceback
    only arrives in Stage 8 when the N-Quads serializer refuses it.

    Pure stdlib so it can be tested without rdflib installed.
    """
    t = str(text)
    return (not t.strip()) or any(ch in t for ch in "<> \t\n")


def _rdfstar_usable() -> bool:
    """Can this rdflib both PARSE `<< s p o >>` and SERIALIZE the result?

    Feature detection, not a version check — and it probes serialization too,
    because parsing is precisely where the silent corruption hides. Cached: the
    answer cannot change within a process.
    """
    global _RDFSTAR_OK
    if _RDFSTAR_OK is not None:
        return _RDFSTAR_OK
    import logging
    import warnings
    try:
        from rdflib import Graph
        # The probe deliberately feeds the parser something it may mangle, so
        # its complaint is the expected result, not news for the operator.
        rdflib_log = logging.getLogger("rdflib")
        prior = rdflib_log.level
        rdflib_log.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = Graph()
                g.parse(data='<< <urn:x:s> <urn:x:p> <urn:x:o> >> <urn:x:m> "v" .',
                        format="turtle")
                ok = not any(looks_malformed_uri(t) for triple in g for t in triple)
                if ok:
                    g.serialize(format="nt")  # the call that raises downstream
        finally:
            rdflib_log.setLevel(prior)
        _RDFSTAR_OK = ok
    except Exception:                         # noqa: BLE001
        _RDFSTAR_OK = False
    return _RDFSTAR_OK


_RDFSTAR_OK = None


def to_dataset(quads, annotations, use_rdfstar: bool = True):
    """Load quads/annotations into an rdflib Dataset with named graphs."""
    from rdflib import Dataset, URIRef, Literal, Namespace
    from rdflib.namespace import RDF

    ds = Dataset()
    for s, p, o, g in quads:
        ds.graph(URIRef(g)).add((_term(s), _term(p), _term(o)))

    for (s, p, o), meta, g in annotations:
        graph = ds.graph(URIRef(g))
        placed = False
        if use_rdfstar and _rdfstar_usable():
            try:
                from rdflib import Graph as _Graph
                lines = []
                for mp, mv in meta.items():
                    obj = f'"{mv.value}"' if isinstance(mv, Lit) else f"<{mv}>"
                    lines.append(f"<< <{s}> <{p}> <{o}> >> <{mp}> {obj} .")
                # Parse into scratch and inspect before committing. Writing
                # straight into `graph` meant a silently mis-parsed annotation
                # was already in the dataset by the time anything noticed, and
                # `placed = True` then suppressed the reification fallback that
                # would have saved it.
                scratch = _Graph()
                scratch.parse(data="\n".join(lines), format="turtle")
                if not any(looks_malformed_uri(t) for triple in scratch for t in triple):
                    for triple in scratch:
                        graph.add(triple)
                    placed = True
            except Exception:                 # noqa: BLE001
                placed = False
        if not placed:
            # Portable fallback: rdf:Statement reification.
            stmt = URIRef(config.INST + "stmt/" + det_id(s, p, o))
            graph.add((stmt, RDF.type, RDF.Statement))
            graph.add((stmt, RDF.subject, _term(s)))
            graph.add((stmt, RDF.predicate, _term(p)))
            graph.add((stmt, RDF.object, _term(o)))
            for mp, mv in meta.items():
                graph.add((stmt, URIRef(mp), _term(mv)))
    return ds


def assert_document(doc: Document, snomed_version: str = config.SNOMED_VERSION,
                    use_rdfstar: bool = True):
    quads, annotations = build_quads(doc, snomed_version)
    return to_dataset(quads, annotations, use_rdfstar=use_rdfstar)
