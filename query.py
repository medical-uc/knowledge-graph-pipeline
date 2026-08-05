#!/usr/bin/env python3
"""Query a serialized medkg graph.

    python query.py graph.nq --list                 # what's available
    python query.py graph.nq relations              # a built-in query
    python query.py graph.nq --about "thyroid gland"  # everything about a concept
    python query.py graph.nq --sparql "SELECT ..."  # ad hoc
    python query.py graph.nq --file myquery.rq

The built-ins are generated from `medkg.config` rather than hardcoded, so they
stay correct if the relation ontology changes. They are also written against
the shapes Stage 6 actually emits, which is worth stating because two of them
are easy to get wrong:

  * An **affirmed** relation is a plain triple `head ont:treated_with tail`.
    Its confidence lives on a separate `rdf:Statement` node (or an RDF-star
    annotation, where the installed rdflib supports it) — so `relations`
    LEFT-JOINs that in rather than requiring it.
  * A **negated or uncertain** relation is deliberately NOT a triple. It is an
    `ont:ClinicalAssertion` node with `ont:about`, `ont:polarity` and possibly
    `ont:target`. That is the point: "no evidence of pericarditis" must never
    become `patient has pericarditis`, so it is unreachable by the queries that
    look for facts, and only `negated` finds it.

Needs rdflib. Everything else in this file is stdlib.
"""
from __future__ import annotations

import argparse
import sys

import medkg                       # noqa: F401 -- pins native thread counts first
from medkg import config

PREFIXES = f"""
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sct:  <{config.SNOMED}>
PREFIX rxn:  <{config.RXNORM}>
PREFIX ont:  <{config.ONT}>
PREFIX inst: <{config.INST}>
PREFIX prov: <{config.PROV}>
PREFIX dct:  <http://purl.org/dc/terms/>
"""


def _relation_values() -> str:
    uris = sorted(spec["uri"] for spec in config.RELATION_ONTOLOGY.values())
    return " ".join(f"<{u}>" for u in uris)


def build_queries() -> dict[str, tuple[str, str]]:
    """name -> (description, sparql). Generated from the live ontology."""
    return {
        "relations": (
            "Relations the SOURCE asserted, with labels and confidence. "
            "Excludes anything flagged uncertain (see `flagged`) and anything "
            "the reasoner derived (see `inferred`). An object rendered as "
            "'iodine [deficiency]' is a post-coordinated instance node, not the "
            "bare concept: the deviation is part of what the source claimed.",
            PREFIXES + f"""
SELECT ?subject ?predicate ?object ?confidence WHERE {{
  VALUES ?p {{ {_relation_values()} }}
  ?s ?p ?o .
  OPTIONAL {{ ?s rdfs:label ?sl }}
  OPTIONAL {{ ?o rdfs:label ?ol }}
  OPTIONAL {{ ?stmt rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o ;
                    ont:confidence ?confidence }}
  BIND(COALESCE(?sl, STR(?s)) AS ?subject)
  BIND(COALESCE(?ol, STR(?o)) AS ?object)
  BIND(REPLACE(STR(?p), "^.*[#/]", "") AS ?predicate)
}} ORDER BY ?subject ?predicate"""),

        "flagged": (
            "Relations demoted to uncertain by the Stage-2 guards -- reversed "
            "causal direction, an endpoint that is only part of the phrase the "
            "source meant (iodine vs iodine deficiency), evidence that does not "
            "mention both entities, contradictory directions, or one cause with "
            "two opposite effects. Read these first; medkg/guards.py explains "
            "each check and needs_review names the cue that fired.",
            PREFIXES + """
SELECT ?polarity ?about ?target ?assertionType WHERE {
  ?stmt a ont:ClinicalAssertion ; ont:about ?a ; ont:polarity ?pol .
  FILTER(?pol != ont:Affirmed)
  OPTIONAL { ?stmt ont:target ?t . OPTIONAL { ?t rdfs:label ?tl } }
  OPTIONAL { ?stmt ont:assertionType ?at }
  OPTIONAL { ?a rdfs:label ?al }
  BIND(COALESCE(?al, STR(?a)) AS ?about)
  BIND(COALESCE(?tl, STR(?t), "") AS ?target)
  BIND(REPLACE(STR(?pol), "^.*[#/]", "") AS ?polarity)
  BIND(REPLACE(STR(?at), "^.*[#/]", "") AS ?assertionType)
}"""),

        "inferred": (
            "Relations Stage 7 DERIVED rather than read: subPropertyOf parents "
            "(complication_of -> caused_by), subclass and transitivity. These "
            "carry no confidence because no extractor produced them.",
            PREFIXES + f"""
SELECT ?subject ?predicate ?object WHERE {{
  VALUES ?p {{ {_relation_values()} }}
  ?s ?p ?o .
  OPTIONAL {{ ?s rdfs:label ?sl }}
  OPTIONAL {{ ?o rdfs:label ?ol }}
  BIND(COALESCE(?sl, STR(?s)) AS ?subject)
  BIND(COALESCE(?ol, STR(?o)) AS ?object)
  BIND(REPLACE(STR(?p), "^.*[#/]", "") AS ?predicate)
}} ORDER BY ?subject ?predicate"""),

        "concepts": (
            "Linked concepts, most-mentioned first. The reading list for the document.",
            PREFIXES + """
SELECT ?label ?concept ?mentions WHERE {
  ?concept rdfs:label ?label .
  OPTIONAL { ?concept ont:mentionCount ?mentions }
} ORDER BY DESC(?mentions) ?label"""),

        "negated": (
            "Negated/uncertain assertions. These are NOT facts in the graph -- "
            "the whole point is that they are unreachable by the queries above.",
            PREFIXES + """
SELECT ?polarity ?about ?assertionType ?target WHERE {
  ?stmt a ont:ClinicalAssertion ;
        ont:about ?a ;
        ont:polarity ?pol .
  OPTIONAL { ?stmt ont:assertionType ?at }
  OPTIONAL { ?stmt ont:target ?t . OPTIONAL { ?t rdfs:label ?tl } }
  OPTIONAL { ?a rdfs:label ?al }
  BIND(COALESCE(?al, STR(?a)) AS ?about)
  BIND(COALESCE(?tl, STR(?t), "") AS ?target)
  BIND(REPLACE(STR(?pol), "^.*[#/]", "") AS ?polarity)
  BIND(REPLACE(STR(?at), "^.*[#/]", "") AS ?assertionType)
}"""),

        "figures": (
            "Figures, their captions, and the concepts they depict.",
            PREFIXES + """
SELECT ?figure ?caption ?depicts WHERE {
  ?fig a ont:FigureImage .
  OPTIONAL { ?fig ont:caption ?caption }
  OPTIONAL { ?fig ont:depicts ?d . OPTIONAL { ?d rdfs:label ?dl } }
  BIND(REPLACE(STR(?fig), "^.*/", "") AS ?figure)
  BIND(COALESCE(?dl, STR(?d), "") AS ?depicts)
} ORDER BY ?figure"""),

        "instances": (
            "Post-coordinated instance nodes (Stage 4) and their attributes.",
            PREFIXES + ("""
SELECT ?instance ?property ?value WHERE {
  ?i a ?type .
  ?i ?prop ?value .
  FILTER(STRSTARTS(STR(?i), "{INST}"))
  FILTER(?prop != rdf:type)
  BIND(REPLACE(STR(?i), "^.*/", "") AS ?instance)
  BIND(REPLACE(STR(?prop), "^.*[#/]", "") AS ?property)
} ORDER BY ?instance""").replace("{INST}", config.INST)),

        "provenance": (
            "Where this graph came from -- and, critically, what wrote its prose.",
            PREFIXES + """
SELECT ?graph ?property ?value WHERE {
  ?g ?prop ?value .
  VALUES ?prop { dct:source dct:hasVersion prov:wasGeneratedBy
                 prov:wasDerivedFrom prov:generatedAtTime ont:snomedVersion }
  BIND(STR(?g) AS ?graph)
  BIND(REPLACE(STR(?prop), "^.*[#/]", "") AS ?property)
}"""),

        "evidence": (
            "Relations with the text span they were extracted from.",
            PREFIXES + """
SELECT ?subject ?predicate ?object ?confidence ?extractedBy WHERE {
  ?stmt rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o .
  OPTIONAL { ?stmt ont:confidence ?confidence }
  OPTIONAL { ?stmt ont:extractedBy ?extractedBy }
  OPTIONAL { ?s rdfs:label ?sl }
  OPTIONAL { ?o rdfs:label ?ol }
  BIND(COALESCE(?sl, STR(?s)) AS ?subject)
  BIND(COALESCE(?ol, STR(?o)) AS ?object)
  BIND(REPLACE(STR(?p), "^.*[#/]", "") AS ?predicate)
} ORDER BY DESC(?confidence)"""),
    }


ABOUT_QUERY = PREFIXES + """
SELECT ?direction ?predicate ?other WHERE {
  {
    ?c ?p ?o . FILTER(?p != rdfs:label)
    OPTIONAL { ?o rdfs:label ?ol }
    BIND("out" AS ?direction) BIND(COALESCE(?ol, STR(?o)) AS ?other)
    BIND(REPLACE(STR(?p), "^.*[#/]", "") AS ?predicate)
  } UNION {
    ?s ?p ?c .
    OPTIONAL { ?s rdfs:label ?sl }
    BIND("in" AS ?direction) BIND(COALESCE(?sl, STR(?s)) AS ?other)
    BIND(REPLACE(STR(?p), "^.*[#/]", "") AS ?predicate)
  }
} ORDER BY ?direction ?predicate"""

RESOLVE_LABEL = PREFIXES + """
SELECT ?concept WHERE {
  ?concept rdfs:label ?l .
  FILTER(LCASE(STR(?l)) = LCASE(?term))
}"""


INFERRED_GRAPH = "inferred"          # suffix of urn:graph:inferred


def load_graph(path: str, only=None, skip=(), keep_labels: bool = False):
    """Load an N-Quads file and union the selected named graphs into one Graph.

    `only` / `skip` match on the END of the graph URI. They exist because
    Stage 7 parks materialized triples in `urn:graph:inferred` precisely so that
    "the source said this" stays distinguishable from "the reasoner derived
    this" -- and unioning everything threw that distinction away at the last
    step. A real run showed `myxedema caused_by hypothyroidism` with a blank
    confidence sitting beside `myxedema complication_of hypothyroidism` at 0.9,
    which reads as a duplicated extraction and is in fact the subPropertyOf
    axiom working correctly.

    `keep_labels` carries rdfs:label in from every graph regardless, since
    labels are asserted, never inferred: without it a query restricted to the
    inferred graph can only print bare URIs.
    """
    from rdflib import Dataset, Graph
    ds = Dataset()
    ds.parse(path, format="nquads")
    g = Graph()
    for ctx in ds.contexts():
        cid = str(ctx.identifier)
        selected = ((only is None or any(cid.endswith(x) for x in only))
                    and not any(cid.endswith(x) for x in skip))
        for triple in ctx:
            if selected or (keep_labels and str(triple[1]) == config.RDFS_LABEL):
                g.add(triple)
    return g


def print_rows(rows, limit: int = 0) -> int:
    rows = list(rows)
    if not rows:
        print("  (no results)")
        return 0
    headers = [str(v) for v in rows[0].labels] if hasattr(rows[0], "labels") else []
    shown = rows[:limit] if limit else rows
    table = [[("" if v is None else str(v)) for v in r] for r in shown]
    widths = [max(len(h), *(len(r[i]) for r in table)) if table else len(h)
              for i, h in enumerate(headers)]
    if headers:
        print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  " + "  ".join("-" * w for w in widths))
    for r in table:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if limit and len(rows) > limit:
        print(f"  ... {len(rows) - limit} more (raise --limit)")
    return len(rows)


def main(argv=None) -> int:
    queries = build_queries()
    ap = argparse.ArgumentParser(
        description="Query a serialized medkg graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="built-in queries: " + ", ".join(sorted(queries)))
    ap.add_argument("graph", nargs="?", default="graph.nq", help="N-Quads file")
    ap.add_argument("query", nargs="?", help="name of a built-in query")
    ap.add_argument("--list", action="store_true", help="list built-in queries")
    ap.add_argument("--about", metavar="TERM",
                    help="everything the graph says about a concept, by label or URI")
    ap.add_argument("--sparql", help="run an ad-hoc SPARQL query")
    ap.add_argument("--file", help="run SPARQL from a file")
    ap.add_argument("--limit", type=int, default=50, help="rows to print (0 = all)")
    ap.add_argument("--include-inferred", action="store_true",
                    help="include Stage 7's derived triples in every query "
                         "(default: only `inferred` and ad-hoc SPARQL see them)")
    args = ap.parse_args(argv)

    if args.list:
        width = max(len(n) for n in queries)
        for name in sorted(queries):
            print(f"{name.ljust(width)}  {queries[name][0]}")
        return 0

    # Resolve the query name first: it decides which named graphs to load.
    name = None
    if args.sparql is None and args.file is None and not args.about:
        name = args.query or "relations"
        if name not in queries:
            sys.stderr.write(f"error: unknown query {name!r}. Try --list.\n")
            return 2
    ad_hoc = args.sparql is not None or args.file is not None
    load_kw = {}
    if name == "inferred":
        load_kw = {"only": (INFERRED_GRAPH,), "keep_labels": True}
    elif not (args.include_inferred or ad_hoc):
        load_kw = {"skip": (INFERRED_GRAPH,)}

    try:
        graph = load_graph(args.graph, **load_kw)
    except FileNotFoundError:
        sys.stderr.write(f"error: no such graph file: {args.graph}\n")
        return 2
    print(f"{args.graph}: {len(graph)} triples\n")

    if args.about:
        from rdflib import URIRef, Literal
        term = args.about
        if term.startswith(("http://", "https://", "urn:")):
            concept = URIRef(term)
        else:
            hits = list(graph.query(RESOLVE_LABEL, initBindings={"term": Literal(term)}))
            if not hits:
                sys.stderr.write(
                    f"error: no concept labelled {term!r}.\n"
                    "       Try `concepts` to see what the document actually linked.\n")
                return 1
            concept = hits[0][0]
            print(f"resolved {term!r} -> {concept}\n")
        print_rows(graph.query(ABOUT_QUERY, initBindings={"c": concept}), args.limit)
        return 0

    sparql = args.sparql
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            sparql = fh.read()
    if sparql is None:
        desc, sparql = queries[name]
        print(f"# {name}: {desc}\n")

    n = print_rows(graph.query(sparql), args.limit)
    if n:
        print(f"\n{n} row(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # `python query.py graph.nq concepts | head` is the normal way to read
        # this; a traceback for it is just noise.
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
