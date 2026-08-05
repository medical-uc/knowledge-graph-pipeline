"""Stage 7 — reasoning / inference.

Baseline: in-process RDFS/OWL-RL materialization with `owlrl`, kept in a SEPARATE
graph from asserted triples so you can always tell "the source said this" from
"the reasoner derived this" and re-reason from scratch.

Production note: full SNOMED CT classification is authored for OWL 2 EL and is
done with a dedicated EL reasoner such as ELK over the SNOMED OWL release — that
path needs Java + the licensed ontology and is intentionally out of scope for
this Python module. The subclass demo below exercises the same idea at small
scale.
"""
from __future__ import annotations

from typing import Optional


def is_literal(term) -> bool:
    """True for an RDF literal, without needing rdflib at import time."""
    try:
        from rdflib.term import Literal
        return isinstance(term, Literal)
    except ImportError:                       # pragma: no cover - offline path
        return type(term).__name__ == "Literal"


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_RESOURCE = "http://www.w3.org/2000/01/rdf-schema#Resource"


def filter_materialized(triples, drop_axiomatic: bool = True):
    """Drop inferred triples that are invalid or vacuous. Returns (kept, dropped).

    RDFS rule **rdfs4b** says `uuu aaa vvv -> vvv rdf:type rdfs:Resource`, and
    owlrl applies it to *literal* objects too. rdflib will hold
    `"bones" rdf:type rdfs:Resource` in memory quite happily and even serialize
    it, but a literal cannot be a subject in RDF, so the resulting N-Quads file
    is unparseable — it fails on the way back IN, long after the run that wrote
    it looked successful.

    `rdf:type rdfs:Resource` is dropped wholesale by default for a second
    reason: it holds of every term in the graph, so it is pure bulk. Pass
    `drop_axiomatic=False` to keep it.
    """
    kept, dropped = [], {"literal_subject": 0, "axiomatic": 0}
    for triple in triples:
        s, p, o = triple
        if is_literal(s):
            dropped["literal_subject"] += 1
            continue
        if drop_axiomatic and str(p) == RDF_TYPE and str(o) == RDFS_RESOURCE:
            dropped["axiomatic"] += 1
            continue
        kept.append(triple)
    return kept, dropped


def property_axioms() -> str:
    """`rdfs:subPropertyOf` axioms generated from the live relation ontology.

    Stage 2 asserts only the most specific type that fits and prunes the parent
    (`guards.prune_subsumed_relations`), because a run that emitted both
    `TRH regulates TSH` and `TRH stimulates TSH` was claiming one fact twice.
    These axioms are what make that safe: the general triple is re-derived here
    into `urn:graph:inferred`, so it stays queryable while remaining
    distinguishable from what a source actually said — which is the whole
    reason asserted and inferred live in separate graphs.

    Generated rather than written out, so adding `subPropertyOf` to the JSON is
    a one-line change that cannot fall out of step with this module.
    """
    from . import config
    lines = []
    for name, spec in sorted(config.RELATION_ONTOLOGY.items()):
        parent = spec.get("subPropertyOf")
        if not parent:
            continue
        parent_uri = config.RELATION_ONTOLOGY[parent]["uri"]
        lines.append(f"<{spec['uri']}> rdfs:subPropertyOf <{parent_uri}> .")
    return "\n".join(lines)


def load_ontology_graph(ontology_ttl: Optional[str] = None):
    """A tiny schema: the STEMI -> MI -> IHD subclass chain + a transitive prop,
    plus the relation-type hierarchy read from the ontology JSON.
    Replace with your imported SNOMED/RxNorm schema in production."""
    from rdflib import Graph
    g = Graph()
    if ontology_ttl:
        g.parse(ontology_ttl, format="turtle")
        # The property hierarchy is OUR vocabulary, not the imported schema's,
        # so it is added even when a custom ontology file is supplied.
        axioms = property_axioms()
        if axioms:
            g.parse(data="@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                         + axioms, format="turtle")
        return g
    g.parse(data="@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
                 + property_axioms(), format="turtle")
    g.parse(data="""
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl:  <http://www.w3.org/2002/07/owl#> .
        @prefix sct:  <http://snomed.info/id/> .
        @prefix ont:  <http://example.org/medkg/ont#> .

        sct:401303003 rdfs:subClassOf sct:22298006 .    # STEMI  is-a MI
        sct:22298006  rdfs:subClassOf sct:414545008 .   # MI     is-a Ischemic heart disease
        ont:caused_by a owl:TransitiveProperty .
    """, format="turtle")
    return g


def materialize(dataset, ontology_ttl: Optional[str] = None,
                drop_axiomatic: bool = True, report=None):
    """Expand the union of (all asserted named graphs + ontology) and return the
    NEWLY inferred triples, minus the ones that cannot legally be written."""
    from rdflib import Graph
    from owlrl import DeductiveClosure, RDFS_Semantics

    base = Graph()
    for ctx in dataset.contexts():
        if str(ctx.identifier).endswith("provenance"):
            continue
        for triple in ctx:
            base.add(triple)
    for triple in load_ontology_graph(ontology_ttl):
        base.add(triple)

    before = set(base)
    DeductiveClosure(RDFS_Semantics).expand(base)
    inferred, dropped = filter_materialized(set(base) - before, drop_axiomatic)
    if report is not None:
        report.update(dropped)

    # Park inferred triples in their own named graph.
    from rdflib import URIRef
    inf_graph = dataset.graph(URIRef("urn:graph:inferred"))
    for triple in inferred:
        inf_graph.add(triple)
    return inferred
