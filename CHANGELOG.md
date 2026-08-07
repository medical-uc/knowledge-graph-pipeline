# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `medkg/console.py`: the progress reporting every stage now shares. It prints
  a stage header, the steps inside it, and batched counts (`20/96 chunks
  tagged`) on plain lines, picking a batch size that reports roughly ten times
  per loop. `with_progress` wraps a sequence so a loop reports itself,
  `set_quiet` silences the module for library use, and
  `set_base_indent_level` nests a document's stages under its banner in a
  corpus run.
- Every stage now says what it is doing while it does it, instead of only
  reporting its totals at the end: Stage 1 names each parsing step, Stage 2
  reports the NER label mix and how many LLM calls each pass actually made,
  Stage 3 reports index loading and the link cache hit rate, Stage 4 the
  instances and attribute triples minted, Stage 5 the concepts depicted, Stage
  6 quad building and dataset loading, Stage 7 the closure expansion, and
  Stage 8 the write and the read-back that verifies it.
- `run.py` opens with the configuration the run will use (mode, inputs,
  artifacts directory, ontology size, NER models, LLM backend, index
  directory) and closes with the wall-clock time each stage and the whole run
  took.
- Progress reporting for the long silent loops in `mcq.py` (stem rephrasing),
  `build_ontology.py` (Semantic Network reading and corpus induction) and
  `build_index.py` (atom scanning and embedding).
- `query.py` and `subset.py` announce the N-Quads load, which on a corpus
  graph is the slowest thing they do.
- `run.py --mode corpus`: build one graph from many documents. Stages 1–5 run
  per document into `artifacts/docs/<stem>/`, Stage 6 merges them into a single
  named-graph dataset, Stage 7 reasons over the union, and Stage 8 writes one
  `.nq`. `--input` now accepts several paths and/or directories.
- Corpus runs resume by default: a document whose IR is already on disk and
  newer than its input is not re-extracted, so a failure part-way through a
  corpus does not re-buy the LLM passes for the documents that finished.
  `--no-resume` forces a full re-extraction.
- `subset.py`: extract a section, a document, or a group of documents from a
  corpus graph as a standalone `.nq`. `--list-scopes` prints what a graph
  contains, `--dry-run` reports a selection without writing, `--reason`
  re-derives the inferred graph from the subset alone.
- Section-scoped named graphs. Everything extracted from one section is
  asserted into `urn:graph:<source_id>/<section_id>`, so a section, a document
  and a group are each a set of named graphs rather than a query.
- `urn:graph:catalog`, describing the corpus's own structure: documents, their
  sections and headings, and any declared groups. It is what `--list-scopes`
  reads and what lets `--section "Hormone Synthesis"` resolve to a graph URI.
- Document groups, declared in a corpus manifest (`--groups`, or `corpus.json`
  beside the inputs) as `{"groups": {"name": ["source_id", ...]}}`. Members may
  be named by `source_id` or by input filename.
- `query.py --doc/--section/--group` restrict any built-in query to a scope,
  using the same resolver `subset.py` extracts with, and `--scopes` lists them.
- `Chunk.section_id`, `Passage.section_id` and `Figure.section_id`, carrying the
  rewriter's own `<sec id="...">` through the IR. Stage 1 parsed it and threw it
  away.
- `Document.sections()` and `Document.chunk_section()`.
- A source-id collision check: two documents claiming the same `source_id` are
  refused rather than merged. The id names the graph and seeds the Stage-4
  instance hashes, so a collision silently mints one document's instance nodes
  onto another's.

### Changed

- Stage progress is printed as batched counts on ordinary lines instead of
  carriage-return redraws on stderr. The old heartbeat rewrote one line in
  place, which is unreadable on a terminal that does not handle `\r` and
  useless in a captured log.
- `stage2_extract.run`, `extract_relations`, `extract_modifiers_llm` and
  `stage3_link.link_document` no longer take a `progress` callback, and
  `run.py`'s `run_stage3` no longer takes `quiet`. Reporting is the console
  module's job now, and `console.set_quiet` is how a library caller turns it
  off.
- `stage3_link.build_index` reports embedding progress as batched counts of
  its own rather than through `tqdm`, which is no longer a dependency.
- Concept labels and mention counts are now corpus-level, asserted once into
  `urn:graph:labels` instead of once per document. A concept mentioned in three
  documents used to carry three `rdfs:label` values, and the `relations` query
  returned every relation about it three times.
- The `relations` query groups by subject/predicate/object and reports an
  `extractions` column, so a fact several documents assert is one row that says
  how many extractions support it rather than one row per document.
- Reified statement nodes are identified by graph as well as by triple, so two
  documents asserting the same relation no longer share a statement node — and
  no longer attach two confidences to it.
- Stage 7 no longer feeds the provenance and catalog graphs to the reasoner;
  they describe the corpus rather than asserting anything about medicine.
- Stage 6 provenance is still asserted once per document, on the document graph,
  with the catalog's `dcterms:isPartOf` edges carrying it down to sections.
- `stage5_images.build_depicts_quads` accepts a `figures` argument so figures
  can be routed to the section they were declared in.
- `run.py` refuses more than one `--input` outside corpus mode instead of
  silently processing the first.

### Removed

- The `tqdm` dependency, along with the optional progress bar it drove in
  `build_index.py`.

### Fixed

- `stage4_postcoord.postcoordinate` is now idempotent: spans that already have
  an instance are skipped, so re-running it over a saved IR (which a resumed
  corpus run does) no longer appends a second, identical set of instance nodes
  and duplicate review entries.
