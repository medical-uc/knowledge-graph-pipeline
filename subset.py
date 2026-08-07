#!/usr/bin/env python3
"""Extract part of a corpus graph as a standalone graph.

    python subset.py artifacts/graph.nq --list-scopes
    python subset.py artifacts/graph.nq --doc thyroid_source --out thyroid.nq
    python subset.py artifacts/graph.nq --doc thyroid_source --section "Hormone Synthesis"
    python subset.py artifacts/graph.nq --group endocrine --out endocrine.nq
    python subset.py artifacts/graph.nq --section s3 --dry-run

A scope is a set of named graphs, because that is how the knowledge is stored:
Stage 6 asserts everything extracted from one section into
`urn:graph:<source_id>/<section_id>`. So a section is one graph, a document is
its graph plus its sections', and a group is the union over its members. Nothing
is inferred about what "belongs" to a section — the subset is exactly what was
asserted there.

Three things always come along, and each is there because leaving it out makes
the output worse than useless rather than merely smaller:

  * **labels**, or every row of every query is a bare `http://snomed.info/id/...`
  * **provenance** for the documents in the subset, because an extract of a
    traceable graph that cannot say where its claims came from is not a smaller
    version of that graph, it is a different and less honest one
  * **the catalog** rows for what was selected, so the file can state what it is
    a subset of

Inferred triples are NOT carried by default: Stage 7 reasoned over the whole
corpus, so a derivation in the subset may depend on a document the subset does
not contain. `--include-inferred` keeps them anyway; `--reason` re-derives them
from the subset alone, which is the answer that is actually true of it.

Needs rdflib.
"""
from __future__ import annotations

import argparse
import sys

import medkg                       # noqa: F401 -- pins native thread counts first
from medkg import config, console, corpus


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract a section, a document or a group as its own graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="scopes combine: --doc X --section Y means that section OF that "
               "document.")
    ap.add_argument("graph", nargs="?", default="artifacts/graph.nq",
                    help="N-Quads file written by `run.py --mode corpus`")
    ap.add_argument("--doc", "--document", dest="docs", action="append", default=[],
                    metavar="NAME",
                    help="source_id, graph URI, or part of the document title "
                         "(repeatable)")
    ap.add_argument("--section", dest="sections", action="append", default=[],
                    metavar="NAME",
                    help="section id (s3), <doc>/<id>, or part of its heading "
                         "(repeatable)")
    ap.add_argument("--group", dest="groups", action="append", default=[],
                    metavar="NAME", help="a group declared in the corpus manifest")
    ap.add_argument("--out", "--output", dest="out", default=None,
                    help="where to write the subset (default: derived from the "
                         "scope, in the artifacts directory)")
    ap.add_argument("--list-scopes", action="store_true",
                    help="print the documents, sections and groups this graph has")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be extracted; write nothing")
    ap.add_argument("--include-inferred", action="store_true",
                    help="carry Stage 7's derived triples across as they stand")
    ap.add_argument("--reason", action="store_true",
                    help="re-run Stage 7 over the subset, so its inferred graph "
                         "follows from the subset itself")
    args = ap.parse_args(argv)

    print(f"loading {args.graph} (N-Quads)")
    try:
        dataset = corpus.load_dataset(args.graph)
    except FileNotFoundError:
        sys.stderr.write(f"error: no such graph file: {args.graph}\n")
        return 2
    catalog = corpus.Catalog.from_dataset(dataset)
    console.announce_step(
        f"{len(corpus.graph_uris(dataset))} named graph(s) in the catalog")

    if args.list_scopes:
        print(catalog.format())
        return 0

    if not (args.docs or args.sections or args.groups):
        sys.stderr.write("error: nothing to extract -- pass --doc, --section or "
                         "--group (see --list-scopes).\n")
        return 2

    graphs, problems = corpus.resolve_scopes(
        catalog, documents=args.docs, sections=args.sections, groups=args.groups,
        all_graphs=corpus.graph_uris(dataset))
    for problem in problems:
        sys.stderr.write(f"error: {problem}\n")
    if problems:
        sys.stderr.write("       `--list-scopes` prints what this graph has.\n")
        return 1
    if not graphs:
        sys.stderr.write("error: that scope selected no graphs.\n")
        return 1

    print(f"selecting {len(graphs)} named graph(s) plus the corpus graphs")
    out_ds = corpus.select(dataset, graphs, include_inferred=args.include_inferred)
    if args.reason:
        from medkg import stage7_reason
        report: dict = {}
        derived = stage7_reason.materialize(out_ds, report=report)
        print(f"re-reasoned over the subset alone: {len(derived)} inferred "
              f"triple(s)")

    rows = corpus.describe(out_ds)
    total = sum(n for _, n in rows)
    print()
    print(f"{args.graph}: selected {len(graphs)} graph(s), {total} triples")
    for uri, count in rows:
        print(f"  {count:7d}  {uri}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    out = config.artifact_path(args.out or _default_name(args))
    from medkg import stage8_serve
    stage8_serve.serialize(out_ds, out)
    print()
    print(f"subset -> {out}")
    print(f"query it: python query.py {out} relations")
    return 0


def _default_name(args) -> str:
    """A filename that says what the subset is, so two extracts never collide."""
    from medkg.stage1_parse import slug
    parts = [slug(p) for p in (args.groups + args.docs + args.sections) if p]
    return "subset-" + "-".join(parts)[:80] + ".nq"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
