"""Stage 8 — storage & serving.

For the runnable demo we query the in-memory rdflib Dataset directly. In
production, serialize with `dataset.serialize(format="nquads")` and load into a
triplestore (GraphDB / Stardog / Fuseki / Neptune), then issue the same SPARQL.
"""
from __future__ import annotations

import time

from . import console


def flatten(dataset):
    """Union all named graphs (incl. inferred) into one queryable Graph."""
    console.announce_stage(8, "serve", "union the named graphs for querying")
    contexts = list(dataset.contexts())
    console.announce_step(
        f"flattening {console.format_count(len(contexts), 'named graph')} "
        f"into one queryable graph")
    from rdflib import Graph
    g = Graph()
    for ctx in console.with_progress(contexts, "graphs merged"):
        for triple in ctx:
            g.add(triple)
    console.announce_detail(console.format_count(len(g), "triple"))
    return g


# "Find any ischemic-heart-disease concept (via inferred subclass) treated with
#  aspirin." Returns STEMI because Stage 7 materialized STEMI subClassOf IHD.
QUERY_IHD_TREATED_WITH_ASPIRIN = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sct:  <http://snomed.info/id/>
PREFIX ont:  <http://example.org/medkg/ont#>
SELECT ?disease WHERE {
  ?disease rdfs:subClassOf sct:414545008 .
  ?disease ont:treated_with ?drug .
}
"""

# Confirm a negated finding is NOT asserted as fact but IS present as a
# negated assertion node.
QUERY_NEGATED = """
PREFIX ont: <http://example.org/medkg/ont#>
SELECT ?about WHERE {
  ?stmt a ont:ClinicalAssertion ;
        ont:about ?about ;
        ont:polarity ont:Negated .
}
"""


def run_query(graph, sparql: str):
    return list(graph.query(sparql))


def serialize(dataset, path: str, fmt: str = "nquads", verify: bool = True) -> dict:
    """Write the dataset, then read it back.

    The round-trip is not paranoia. rdflib will happily serialize a triple whose
    subject is a literal -- `"bones" <rdf:type> <rdfs:Resource>` -- producing a
    file that reports success on the way out and is unparseable on the way back
    in, potentially long afterwards. Writing an artefact nobody can read is a
    worse outcome than failing loudly at write time, so any offending quads are
    removed first and the result is proved readable before we claim success.
    """
    from rdflib.term import Literal

    started_at = time.time()
    console.announce_stage(8, "serve", f"write and verify {path}")
    console.announce_step("removing triples a literal subject makes unwritable")
    removed = 0
    for ctx in list(dataset.contexts()):
        for triple in [t for t in ctx if isinstance(t[0], Literal)]:
            ctx.remove(triple)
            removed += 1
    console.announce_detail(f"{removed} removed")

    console.announce_step(f"serializing as {fmt} -> {path}")
    dataset.serialize(destination=path, format=fmt)

    if verify:
        from rdflib import Dataset
        console.announce_step("reading it back to prove the file parses")
        try:
            reloaded = Dataset()
            reloaded.parse(path, format=fmt)
        except Exception as exc:              # noqa: BLE001
            raise ValueError(
                f"{path} was written but cannot be read back ({exc}). "
                "The graph contains something this syntax cannot express; "
                "serializing it would leave you with an unusable artefact."
            ) from exc
        console.announce_detail(
            f"parsed back "
            f"{console.format_count(sum(1 for _ in reloaded.quads()), 'quad')}")
    console.announce_finished("stage 8", started_at)
    return {"removed_literal_subjects": removed}
