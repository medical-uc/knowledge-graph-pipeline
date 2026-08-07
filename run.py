"""Orchestrate the pipeline.

Modes:
  stage1       Stage 1 only: the TAGGED markdown that rewrite_medical_md.py
               produces -> a tag-free normalized text file (the provenance
               anchor every chunk offset indexes into) + the Stage-1 IR JSON.
               Tags map onto chunk kinds, so no LLM and no network are involved.
  stage3       Stages 3+5 only: a Stage-2 IR JSON -> linked IR JSON. Use this
               rather than re-running `full` after a Stage-3 failure; Stage 2's
               LLM calls are the expensive part and there is no reason to pay
               for them twice.
  stage2       Stage 2 only (NER + LLM modifiers + LLM relations), starting from
               a Stage-1 IR; writes the enriched IR as JSON. Needs scispaCy + an
               LLM backend, but NOT the UMLS index — use it to iterate on Stage 2
               (fragments, Meditron) in isolation.
  full         Stages 2->8. Also needs a built SapBERT/FAISS index (see
               stage3_link.build_index).
  from-stage3  Stages 4,6,7,8 only, starting from an already-linked IR
               (sample_stage3_output.json). Needs only rdflib + owlrl — no
               models, no network. Use this to see the graph-building half run.
  review       Group an IR's `needs_review` by (stage, code) and print it,
               commonest first. Nothing is modified. Start here when the review
               queue is large: entries are written by every stage into one flat
               list, and the grouping is what tells an ontology
               misconfiguration apart from a genuine extraction failure.
  corpus       Every document in a directory (or several --input paths) into ONE
               graph. Stages 1-5 run per document into its own artifact
               directory; Stage 6 merges them into a single named-graph dataset,
               Stage 7 reasons over the union, and Stage 8 writes one .nq.
               Documents are asserted into `urn:graph:<source_id>/<section_id>`,
               so a section, a document or a declared group of documents can be
               lifted back out with subset.py. Resumes by default: a document
               whose IR is already on disk and newer than its input is not
               re-extracted, so a failure at document 4 does not re-buy the LLM
               passes for documents 1-3.
  guards       Re-run ONLY the deterministic relation guards (medkg/guards.py)
               over an existing Stage-2 or Stage-3 IR, and write the corrected
               IR back out. No LLM, no models, no network. This is how you
               audit a graph you have already paid to extract: suspect
               relations are demoted to `uncertain` (so Stage 6 reifies them
               instead of asserting them) and each one gets a needs_review
               entry naming the cue that fired. Follow it with `from-stage3`
               to rebuild the .nq.

The --llm flag (claude | meditron) selects the Stage 2 backend (and Stage 3
rerank) in both `stage2` and `full` modes.

Examples:
  python run.py --mode corpus      --input inputs/ --index-dir ./umls_index --llm huatuo
  python run.py --mode stage1      --input rewritten.md --out rewritten.stage1.json
  python run.py --mode full        --input rewritten.md --index-dir ./umls_index
  python run.py --mode stage2      --input sample_stage1_output.json --llm meditron --out stage2_output.json
  python run.py --mode from-stage3 --input sample_stage3_output.json
  python run.py --mode full        --input sample_stage1_output.json --index-dir ./umls_index
  python run.py --mode review      --input graph.ir.json
  python run.py --mode guards      --input stage3_output.json --out stage3_guarded.json
  python run.py --mode from-stage3 --input stage3_guarded.json --out graph.nq
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# --ontology and --ner have to reach `config` before it is imported, because the
# ontology is loaded (and the LLM output schemas built from it) at import time.
# Hence this pre-scan rather than waiting for argparse.
_NER_FLAG_GIVEN = any(a == "--ner" or a.startswith("--ner=") for a in sys.argv)
for _i, _a in enumerate(sys.argv):
    for _flag, _var in (("--ontology", "MEDKG_ONTOLOGY"), ("--ner", "MEDKG_NER_MODELS")):
        if _a == _flag and _i + 1 < len(sys.argv):
            os.environ[_var] = sys.argv[_i + 1]
        elif _a.startswith(_flag + "="):
            os.environ[_var] = _a.split("=", 1)[1]

import medkg            # noqa: E402,F401 -- pins native thread counts before torch/spaCy/faiss
from medkg import ontology                                          # noqa: E402
from medkg.ir import Document
from medkg import config
from medkg import console
from medkg import stage4_postcoord, stage6_assert, stage7_reason, stage8_serve


def run_stage1(md_path: str, figures_dir="", normalized_out=None,
               verify_images=False, figures_ext="") -> Document:
    """Tagged markdown -> normalized text + Stage-1 IR. Chunk offsets index into
    the normalized file, written next to the source unless overridden."""
    from medkg import stage1_parse
    return stage1_parse.parse_file(md_path, figures_dir=figures_dir,
                                   normalized_path=normalized_out,
                                   verify_images=verify_images, figures_ext=figures_ext)


def run_stage2(doc: Document, call_llm=None) -> Document:
    """Stage 2 over one document. `call_llm` overrides the default backend.

    Stage 2 is one sequential LLM call per chunk per pass, so it reports its
    own progress; nothing is printed here.
    """
    from medkg import stage2_extract
    kw = {} if call_llm is None else {"call_llm": call_llm}
    return stage2_extract.run(doc, **kw)                # NER + LLM modifiers + RE


def run_from_stage3(doc: Document, report: dict = None):
    """Stages 4, 6, 7 and 8 over a linked IR.

    Returns (doc, dataset, inferred_triples, flattened_graph). `report` is
    filled in with Stage 7's dropped-triple counts if given. Nothing is
    written to disk; the caller decides where the graph goes.
    """
    doc = stage4_postcoord.postcoordinate(doc)
    ds = stage6_assert.assert_document(doc)
    inferred = stage7_reason.materialize(ds, report=report)
    graph = stage8_serve.flatten(ds)
    return doc, ds, inferred, graph


def run_stage3(doc: Document, index_dir: str, rerank=None) -> Document:
    """Stages 3 and 5: link mentions, then bridge figure captions.

    Separable from `full` on purpose. Stage 3 loads faiss and torch into one
    process, which is where native-library crashes happen; without a resume
    point, one of those costs you the entire Stage-2 LLM pass as well.
    """
    from medkg import stage3_link, stage5_images
    started_at = time.time()
    console.announce_stage(3, "link",
                           "mentions -> UMLS concepts (SapBERT + FAISS)")
    linker = stage3_link.SapBertLinker(index_dir)
    doc = stage3_link.link_document(doc, linker, rerank=rerank)
    console.announce_finished("stage 3", started_at)
    return stage5_images.bridge_figures(doc, linker=linker, rerank=rerank)


def run_full(doc: Document, index_dir: str, call_llm=None, rerank=None,
             report: dict = None):
    doc = run_stage2(doc, call_llm=call_llm)                       # Stage 2
    doc = run_stage3(doc, index_dir, rerank=rerank)                # Stages 3 + 5
    return run_from_stage3(doc, report=report)


def _extract_one(md_path: str, paths: dict, index_dir: str, call_llm=None,
                 rerank=None, figures_dir="", figures_ext="",
                 verify_figures=False, resume=True) -> Document:
    """Stages 1-5 for ONE document of a corpus, into its own artifact directory.

    Stops at the linked IR rather than building RDF: the whole point of a corpus
    run is that Stage 6 sees every document at once, so it can name one label per
    concept and refuse a source_id collision before anything is written.
    """
    if resume and os.path.exists(paths["ir"]):
        if os.path.getmtime(paths["ir"]) >= os.path.getmtime(md_path):
            doc = Document.from_json(paths["ir"])
            console.announce_step(
                f"resuming from {paths['ir']}: stages 1 to 5 already ran "
                f"({len(doc.spans)} spans, {len(doc.relations)} relations)")
            return doc
        console.announce_step(
            f"{paths['ir']} is older than the input; extracting again")

    os.makedirs(paths["dir"], exist_ok=True)
    doc = run_stage1(md_path, figures_dir=figures_dir,
                     normalized_out=paths["normalized"],
                     verify_images=verify_figures, figures_ext=figures_ext)
    doc.to_json(paths["stage1"])
    console.announce_step(f"stage-1 IR -> {paths['stage1']}")

    doc = run_stage2(doc, call_llm=call_llm)
    doc.to_json(paths["stage2"])
    console.announce_step(f"stage-2 checkpoint -> {paths['stage2']}")

    doc = run_stage3(doc, index_dir, rerank=rerank)
    doc.to_json(paths["ir"])
    console.announce_step(f"linked IR -> {paths['ir']}")
    return doc


def run_corpus(inputs, index_dir: str, declared_groups: dict = None,
               call_llm=None, rerank=None, report: dict = None,
               figures_dir="", figures_ext="", verify_figures=False,
               resume=True):
    """Many documents -> one graph.

    Returns (docs, dataset, inferred, graph, groups, unknown_members).
    """
    from medkg import corpus, stage6_assert

    docs: list[Document] = []
    started_at = time.time()
    for i, md_path in enumerate(inputs, start=1):
        stem = corpus.stem_of(md_path)
        print()
        print(f"document {i}/{len(inputs)}  {md_path}")
        # Everything a document's own stages print is nested under its banner,
        # so a corpus run reads as a list of documents rather than as one
        # undifferentiated stream of stage output.
        outer = console.set_base_indent_level(1)
        try:
            docs.append(_extract_one(
                md_path, corpus.doc_artifacts(stem), index_dir, call_llm=call_llm,
                rerank=rerank, figures_dir=figures_dir, figures_ext=figures_ext,
                verify_figures=verify_figures, resume=resume))
        finally:
            console.set_base_indent_level(outer)
    print()
    print(f"extracted {len(inputs)} document(s) in "
          f"{console.format_duration(time.time() - started_at)}")

    # Groups are declared against source_ids, which only exist once Stage 1 has
    # read each <src> line -- so they are resolved here, not from the manifest
    # alone. A manifest written against input filenames still works: both are
    # accepted as member names.
    known: dict[str, str] = {}
    for path, doc in zip(inputs, docs):
        known[doc.source.source_id] = doc.source.source_id
        known[corpus.stem_of(path)] = doc.source.source_id
    groups, unknown = corpus.resolve_group_members(declared_groups or {}, known)

    print()
    print(f"merging {len(docs)} document(s) into one graph")
    outer = console.set_base_indent_level(1)
    try:
        docs = [stage4_postcoord.postcoordinate(doc) for doc in docs]
    finally:
        console.set_base_indent_level(outer)
    console.announce_step("writing each post-coordinated IR back to its "
                          "artifact directory")
    for doc, md_path in zip(docs, inputs):
        doc.to_json(corpus.doc_artifacts(corpus.stem_of(md_path))["ir"])
    ds = stage6_assert.assert_documents(docs, groups=groups)
    inferred = stage7_reason.materialize(ds, report=report)
    graph = stage8_serve.flatten(ds)
    return docs, ds, inferred, graph, groups, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["stage1", "stage2", "stage3", "full", "corpus",
                             "from-stage3", "guards", "review"],
                    default="from-stage3")
    ap.add_argument("--llm", choices=["claude", "meditron", "huatuo"], default="claude",
                    help="LLM backend for Stage 2 (and the Stage 3 rerank); used in stage2/full modes")
    ap.add_argument("--input", required=False, nargs="+",
                    help="tagged .md from rewrite_medical_md.py (stage1/full), or IR "
                         "JSON. In corpus mode: any number of .md files and/or "
                         "directories to read them from")
    ap.add_argument("--groups", default=None,
                    help="corpus manifest declaring document groups "
                         "(default: corpus.json beside the inputs, if present). "
                         'Shape: {"groups": {"endocrine": ["thyroid_source", ...]}}')
    ap.add_argument("--no-resume", action="store_true",
                    help="corpus mode: re-extract every document even if its IR "
                         "is already on disk and newer than its input")
    ap.add_argument("--index-dir", default="./umls_index")
    ap.add_argument("--figures-dir", default="",
                    help="prefix prepended to relative [FIGURE:...] references")
    ap.add_argument("--figures-ext", default="",
                    help="extension appended to extensionless [FIGURE:...] refs, "
                         "e.g. .png -- turns the id 'p2_b3' into a locator")
    ap.add_argument("--verify-figures", action="store_true",
                    help="check that each resolved image path exists (off by default: "
                         "parsing stays a pure function of the document)")
    ap.add_argument("--normalized-out", default=None,
                    help="path for the intermediate all-prose markdown "
                         "(default: <input>.normalized.md)")
    ap.add_argument("--ontology", default=None,
                    help="relation/modifier ontology JSON (default: the bundled "
                         "med-school one; see build_ontology.py to extend it)")
    ap.add_argument("--ner", default=None,
                    help="comma-separated scispaCy NER models. A relation can only "
                         "fire if some model here emits both its endpoint labels, "
                         "e.g. en_ner_bc5cdr_md,en_ner_bionlp13cg_md")
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip the LLM disambiguation step in Stage 3 and take "
                         "top-cosine. Much faster; the linker is already ~95%% "
                         "accurate on unambiguous mentions")
    ap.add_argument("--ontology-check", action="store_true",
                    help="print the ontology/NER reachability report and exit")
    ap.add_argument("--out", "--output", dest="out", default="graph.nq")
    ap.add_argument("--artifacts", default=None,
                    help=f"directory for generated files (default: "
                         f"{config.ARTIFACTS_DIR}/, or $MEDKG_ARTIFACTS). A bare "
                         f"--out filename lands here; an --out with any "
                         f"directory component is used verbatim.")
    args = ap.parse_args()
    # --input is a list for corpus mode; every other mode reads exactly one
    # thing, and being handed two is a mistake worth naming rather than
    # silently processing the first.
    args.inputs = list(args.input or [])
    args.input = args.inputs[0] if args.inputs else None
    if len(args.inputs) > 1 and args.mode != "corpus":
        ap.error(f"--mode {args.mode} takes one --input; use --mode corpus to "
                 f"build one graph from several documents")
    if args.artifacts:
        config.ARTIFACTS_DIR = args.artifacts
    # Resolved once here rather than at each write: the Stage-2 checkpoint and
    # the .ir.json are both derived from args.out by string surgery, so fixing
    # it up front is what keeps the whole set together in one directory.
    args.out = config.artifact_path(args.out)
    if getattr(args, "normalized_out", None):
        args.normalized_out = config.artifact_path(args.normalized_out)

    if args.ontology_check:
        print(ontology.summary(config.ONTOLOGY, config.NER_MODELS,
                               default_models=not _NER_FLAG_GIVEN))
        return

    reasoner_report: dict = {}
    rerank_errors: list = []
    llm_errors: list = []
    call_llm, rerank = None, None
    if not args.input:
        ap.error("--input is required (or use --ontology-check)")

    run_started_at = time.time()
    print(f"medkg pipeline  mode {args.mode}")
    console.announce_step(f"input      {', '.join(args.inputs)}")
    console.announce_step(f"artifacts  {config.ARTIFACTS_DIR}/")
    console.announce_step(f"ontology   "
                          f"{len(config.RELATION_ONTOLOGY)} relation type(s), "
                          f"{len(config.SNOMED_ATTR)} modifier type(s)")
    console.announce_step(f"NER        {', '.join(config.NER_MODELS)}")
    if args.mode in ("stage2", "stage3", "full", "corpus"):
        console.announce_step(
            f"LLM        {args.llm}"
            + ("" if args.no_rerank else " (also reranking Stage 3 links)"))
    if args.mode in ("stage3", "full", "corpus"):
        console.announce_step(f"index      {args.index_dir}")

    # Surface unusable relation types before spending any LLM budget: a type
    # whose endpoint labels no NER model emits contributes nothing, silently.
    if args.mode in ("stage2", "full", "corpus"):
        problems = ontology.validate(config.ONTOLOGY, config.NER_MODELS)
        if problems:
            named = ", ".join(p["relation"] for p in problems[:3])
            more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
            # Repeat --ner in the suggested command: reachability depends
            # entirely on the model list, so `--ontology-check` without it
            # reports on a DIFFERENT configuration and contradicts this warning.
            sys.stderr.write(
                f"warning: {len(problems)}/{len(config.RELATION_ONTOLOGY)} relation "
                f"types cannot fire with this NER: {named}{more}\n"
                f"         detail: python run.py --ontology-check "
                f"--ner {','.join(config.NER_MODELS)}\n")

    if args.mode in ("stage2", "stage3", "full", "corpus") and args.llm != "claude":
        from medkg import llm_backends

        def _call_failed(exc, stats):
            llm_errors.append(type(exc).__name__)
            if len(llm_errors) == 1:
                sys.stderr.write(
                    f"\nwarning: LLM call failed ({type(exc).__name__}); that "
                    "chunk is recorded in needs_review and the run continues. "
                    f"Raise {llm_backends.get_profile(args.llm).env_prefix}_TIMEOUT "
                    "if this repeats.\n")

        def _rerank_failed(mention, exc):
            rerank_errors.append(mention)
            if len(rerank_errors) == 1:
                sys.stderr.write(
                    f"\nwarning: rerank unavailable ({type(exc).__name__}); "
                    "falling back to top-cosine linking. Stage 3 continues.\n")

        profile = llm_backends.get_profile(args.llm)
        call_llm = llm_backends.make_call(profile, on_error=_call_failed)
        rerank = None if args.no_rerank else llm_backends.make_rerank(
            profile, on_error=_rerank_failed)

    if args.mode == "corpus":
        from medkg import corpus

        inputs = corpus.discover(args.inputs)
        if not inputs:
            sys.stderr.write(f"error: no documents found under {args.inputs}\n")
            raise SystemExit(2)
        missing = [p for p in inputs if not os.path.exists(p)]
        if missing:
            sys.stderr.write("error: no such input: " + ", ".join(missing) + "\n")
            raise SystemExit(2)

        manifest_path = corpus.manifest_path_for(args.inputs, args.groups)
        manifest = corpus.load_manifest(manifest_path)
        print()
        print(f"corpus: {len(inputs)} document(s) to build into one graph")
        for position, path in enumerate(inputs, start=1):
            console.announce_step(f"[{position}/{len(inputs)}] {path}")
        if manifest["groups"]:
            console.announce_step(
                f"{len(manifest['groups'])} declared group(s) from "
                f"{manifest_path}")

        docs, ds, inferred, graph, groups, unknown = run_corpus(
            inputs, args.index_dir, declared_groups=manifest["groups"],
            call_llm=call_llm, rerank=rerank, report=reasoner_report,
            figures_dir=args.figures_dir, figures_ext=args.figures_ext,
            verify_figures=args.verify_figures, resume=not args.no_resume)

        for item in unknown:
            sys.stderr.write(f"warning: group member not in this corpus -- {item}\n")
        live = {n: m for n, m in groups.items() if m}
        if live:
            print("groups: " + ", ".join(
                f"{n} ({len(m)})" for n, m in sorted(live.items())))

        print()
        print("what each document contributed")
        for path, doc in zip(inputs, docs):
            linked = sum(1 for sp in doc.spans if sp.uri)
            print(f"  {doc.source.source_id:32s} {len(doc.sections()):3d} sections  "
                  f"{len(doc.spans):5d} spans ({linked} linked)  "
                  f"{len(doc.relations):4d} relations  "
                  f"{len(doc.needs_review):5d} review")
        print(f"  {'TOTAL':32s} {sum(len(d.sections()) for d in docs):3d} sections  "
              f"{sum(len(d.spans) for d in docs):5d} spans  "
              f"{sum(len(d.relations) for d in docs):4d} relations")
        print(f"  {'inferred triples':32s} {len(inferred)}")
        print(f"  {'named graphs':32s} {len(corpus.graphs_of(ds))}")

        stage8_serve.serialize(ds, args.out)
        print()
        print(f"corpus graph -> {args.out}")
        print("whole run took " + console.format_duration(
            time.time() - run_started_at))
        print(f"query it : python query.py {args.out} relations")
        print(f"subset it: python subset.py {args.out} --list-scopes")
        return

    is_md = args.input.lower().endswith((".md", ".markdown"))
    # Stage 3 links spans that Stage 2 produced. Handed markdown it used to run
    # Stage 1 and then link a document with no spans at all -- succeeding, and
    # emitting an empty graph. Refuse instead.
    if is_md and args.mode in ("stage3", "from-stage3", "guards", "review"):
        sys.stderr.write(
            f"error: --mode {args.mode} needs an IR JSON, not markdown.\n"
            f"       Stage 2 has to run first, or there are no spans to link:\n"
            f"         python run.py --mode stage2 --input {args.input} "
            f"--llm {args.llm} --out stage2_output.json\n"
            f"         python run.py --mode {args.mode} --input stage2_output.json "
            f"--index-dir {args.index_dir} --out stage3_output.json\n")
        raise SystemExit(2)

    # Stage 1 runs whenever the input is markdown; JSON inputs are already IR.
    if args.mode == "stage1" or is_md:
        doc = run_stage1(args.input,
                         figures_dir=args.figures_dir, normalized_out=args.normalized_out,
                         verify_images=args.verify_figures,
                         figures_ext=args.figures_ext)
        if doc.source.generator:
            console.announce_step(
                f"provenance: {doc.source.origin or '?'} rewritten by "
                f"{doc.source.generator} at {doc.source.generated_at or '?'}")
        console.announce_step(
            f"{len(doc.figures)} figure(s), "
            f"{len(doc.needs_review)} review entry/entries")
        if args.mode == "stage1":
            outfile = args.out if args.out.endswith(".json") else config.artifact_path("stage1_output.json")
            doc.to_json(outfile)
            console.announce_step(f"stage-1 IR -> {outfile}")
            print()
            print(f"stage 1 only, done in "
                  f"{console.format_duration(time.time() - run_started_at)}")
            return
    else:
        print(f"reading IR from {args.input}")
        doc = Document.from_json(args.input)
        console.announce_step(
            f"{len(doc.chunks)} chunk(s), {len(doc.spans)} span(s), "
            f"{len(doc.relations)} relation(s), "
            f"{len(doc.instances)} instance(s)")

    if args.mode == "review":
        from medkg import guards
        rows = guards.summarize_review(doc)
        total = len(doc.needs_review)
        if not total:
            print("needs_review is empty.")
            return
        print(f"needs_review: {total} entries in {len(rows)} groups\n")
        w = max(len(f"{r[0]}/{r[1]}") for r in rows)
        for stage, code, n, example in rows:
            share = 100 * n // total
            print(f"  {n:5d} ({share:2d}%)  {(stage + '/' + code).ljust(w)}  {example[:88]}")
        print("\n  codes: head-label/tail-label = the span's NER label is not one the "
              "relation type accepts\n"
              "         (check `run.py --ontology-check`; a label_hierarchy entry may "
              "be all that is missing)")
        return

    if args.mode == "guards":
        if not doc.relations:
            sys.stderr.write("error: input has no relations to check.\n")
            raise SystemExit(2)
        from medkg import guards
        before = len(doc.relations)
        affirmed_before = sum(1 for r in doc.relations if r.polarity == "affirmed")
        print()
        print(f"re-running the deterministic guards over "
              f"{before} relation(s); no LLM, no models, no network")
        report = guards.run_all(doc)
        outfile = args.out if args.out.endswith(".json") else config.artifact_path("guarded_output.json")
        doc.to_json(outfile)
        print(f"relations: {before} -> {len(doc.relations)} "
              f"(affirmed {affirmed_before} -> {report['affirmed_remaining']})")
        for name in ("evidence_scope", "governed_role", "causal_direction", "conversion_direction",
                     "antonym_conflict", "feedback_loop", "self_loop",
                     "fragment_span", "vacuous_endpoint",
                     "structural_deviation", "enzyme_as_substrate",
                     "duplicate", "subsumed"):
            if report[name]:
                print(f"  {name:18s} {report[name]:4d}")
        if report["degenerate_conf"]:
            print("  WARNING: relation confidences are degenerate -- the RE floor "
                  "is filtering nothing (see needs_review)")
        print(f"guarded IR -> {outfile}   (rebuild with --mode from-stage3)")
        return

    if args.mode == "stage3":
        if not doc.spans:
            sys.stderr.write(
                "error: input has no spans -- Stage 2 has not run on it, so there\n"
                "       is nothing for Stage 3 to link.\n")
            raise SystemExit(2)
        doc = run_stage3(doc, args.index_dir, rerank=rerank)
        outfile = args.out if args.out.endswith(".json") else config.artifact_path("stage3_output.json")
        doc.to_json(outfile)
        linked = sum(1 for s_ in doc.spans if s_.uri)
        print()
        print(f"linked IR -> {outfile}")
        print(f"  {linked}/{len(doc.spans)} spans linked, "
              f"{len(doc.needs_review)} review item(s)")
        print(f"stages 3 and 5 done in "
              f"{console.format_duration(time.time() - run_started_at)}")
        return

    if args.mode == "stage2":
        doc = run_stage2(doc, call_llm=call_llm)
        outfile = args.out if args.out.endswith(".json") else config.artifact_path("stage2_output.json")
        doc.to_json(outfile)
        n_mods = sum(len(s.modifiers) for s in doc.spans)
        print()
        print(f"stage-2 IR -> {outfile}")
        print(f"  spans: {len(doc.spans)}  relations: {len(doc.relations)}  "
              f"modifiers: {n_mods}  needs_review: {len(doc.needs_review)}")
        print(f"stage 2 done in "
              f"{console.format_duration(time.time() - run_started_at)}")
        return

    if args.mode == "full":
        # Checkpoint between the expensive stage and the fragile one. Stage 2 is
        # ~50 LLM calls; Stage 3 loads FAISS and SapBERT and reaches the network
        # again. Without this, a Stage-3 failure discards all of Stage 2 -- which
        # has now happened twice.
        doc = run_stage2(doc, call_llm=call_llm)
        ckpt = (args.out.rsplit(".", 1)[0] if "." in args.out else args.out) + ".stage2.json"
        doc.to_json(ckpt)
        console.announce_step(f"stage-2 checkpoint -> {ckpt}")
        doc = run_stage3(doc, args.index_dir, rerank=rerank)
        doc, ds, inferred, graph = run_from_stage3(doc, report=reasoner_report)
    else:
        doc, ds, inferred, graph = run_from_stage3(doc, report=reasoner_report)

    linked = sum(1 for sp in doc.spans if sp.uri)
    print()
    print("what the run produced")
    print(f"  {'spans':17s}: {len(doc.spans)} ({linked} linked)")
    print(f"  {'relations':17s}: {len(doc.relations)}")
    print(f"  {'instances minted':17s}: {len(doc.instances)}")
    print(f"  {'inferred triples':17s}: {len(inferred)}")
    if any(reasoner_report.values()):
        print(f"    (dropped {reasoner_report.get('literal_subject', 0)} literal-subject "
              f"and {reasoner_report.get('axiomatic', 0)} vacuous rdfs:Resource triples)")
    print(f"  {'needs_review':17s}: {len(doc.needs_review)}")
    if llm_errors:
        print(f"    ({len(llm_errors)} LLM call(s) failed; those chunks are in "
              f"needs_review as *-no-response and can be re-run)")
    if rerank_errors:
        print(f"    (rerank failed for {len(rerank_errors)} mention(s); those fell "
              f"back to top-cosine)")
    # Demo queries, hardcoded for the cardiology sample -- empty output on any
    # other document is expected, not a failure.
    print("\ndemo queries (written for the cardiology sample; empty elsewhere)")
    print("  Q1  IHD (inferred) treated with aspirin:")
    for row in stage8_serve.run_query(graph, stage8_serve.QUERY_IHD_TREATED_WITH_ASPIRIN):
        print("     ", row.disease)
    print("  Q2  negated assertions (should list pericarditis, not diagnose it):")
    for row in stage8_serve.run_query(graph, stage8_serve.QUERY_NEGATED):
        print("     ", row.about)

    stage8_serve.serialize(ds, args.out)

    # The IR is what `needs_review` lives in, and a low relation count is only
    # diagnosable by reading it. Writing only the graph meant every failed
    # extraction was invisible.
    ir_out = (args.out.rsplit(".", 1)[0] if "." in args.out else args.out) + ".ir.json"
    doc.to_json(ir_out)
    print()
    print(f"graph -> {args.out}")
    print(f"IR (incl. needs_review) -> {ir_out}")
    print("whole run took " + console.format_duration(
        time.time() - run_started_at))
    print(f"query it: python query.py {args.out} --list")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:            # `run.py --ontology-check | head`
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
