# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Span labels are now re-derived from the UMLS semantic type of the concept a
  mention linked to, instead of being taken on trust from the NER model that
  tagged it. `build_index.py --mrsty MRSTY.RRF` writes a `semantic_types.json`
  sidecar next to the FAISS index, `stage3_link.reconcile_labels` applies it,
  and `Span.ner_label` keeps whatever the NER model had said so the override
  stays auditable. The measurement behind it: on a ten-document anatomy corpus,
  1,414 of 1,965 relation rejections (72%) were label-constraint failures, and
  they trace to `en_ner_bionlp13cg_md` reading anatomical cavities as
  `PATHOLOGICAL_FORMATION` and glands as `CANCER` outside its training domain.
  A surface-form override list cannot fix that — the rejections spread over 482
  distinct (surface, label) pairs.
- `build_index.py --semantic-types-only` adds the sidecar to an index that
  already exists. It parses MRSTY and embeds nothing, so an index built before
  this change gains semantic types in minutes rather than hours.
- `stage3_link.parse_mrsty`, `collect_semantic_types`, `build_semantic_types`
  and `indexed_cuis`: the sidecar build, pure-stdlib apart from the reads. The
  sidecar is restricted to the CUIs the index can actually return, which keeps
  it proportional to the index rather than to all of UMLS.
- `ontology_medschool.json` gains a `semantic_types` block (TUI -> span label)
  and `semantic_type_priority` (which label wins when a concept carries several
  types, most specific first, because the label hierarchy only weakens upward).
- `ontology_medschool.json` gains `modifier_value_antonyms`, so Stage 4 can tell
  a contradictory pair of deviations from a merely different one.
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

- Stage 3 now runs between Stage 2's two halves rather than after it. A span's
  label decides which relation types it can take part in, which candidate pairs
  are offered to the extractor and whether the validator keeps what comes back,
  so it has to be settled before the LLM passes. `stage2_extract.run` is split
  into `run_ner` and `run_llm` (the combined `run` is kept, and is what
  `--mode stage2` still uses), and `run.py` gains `run_stages_2_to_5`, which is
  the one place the interleaved order is written down.
- The corpus and `full` checkpoints now sit after linking rather than after the
  LLM passes, so a resume carries the reconciled labels the LLM passes were
  about to run against.
- `run.py` builds the SapBERT/FAISS linker once per run and shares it
  (`LinkerCache`). A ten-document corpus run used to read the multi-gigabyte
  index ten times.
- `stage3_link.link_document` no longer links modifier values, because at its
  new position the modifiers do not exist yet. `link_modifiers` is a separate
  pass that Stage 5 runs once Stage 2's modifier extraction has happened.
- Stage 4 resolves modifier values from the ontology lexicon first, and for a
  role-changing modifier will not accept a Stage-3 link at all. A deviation
  value is a closed vocabulary of about thirty surface forms; asking the linker
  to resolve a bare qualifier word with no context sent `hasDeviation` to
  `sct:36976004` (Hypoparathyroidism, a disease) for "not enough PTH" and to
  `sct:88323005` (Adequate) for "adequate", while `ont:Deficiency` — the value
  the ontology defines for exactly this — went unused in all 73 instances of a
  ten-document run. Off-lexicon deviations now go to review instead of to a
  plausible-looking wrong code.
- Post-coordinated instance ids are keyed on the concept and its attributes
  instead of the mention's character offset, so mentions that mean the same
  thing share one node. The offset key made every occurrence its own node:
  `Addison's disease caused_by cortisol [deficiency]` and
  `... Cortisol [deficiency]` came back as two rows of the `relations` query
  for one fact. Re-running on unchanged input is still byte-identical, which is
  what the offset was there for. Only the first record for a node carries the
  label, so a shared node does not collect one `rdfs:label` per mention.
- `Protein` is now declared a subtype of `Substance` in `label_hierarchy`, on
  the same reasoning that already put `Drug` there: every constraint that
  admits a `Protein` alongside a `Substance` still does, and a peptide hormone
  can now satisfy the `Substance` tail of `treated_with`. Constraints are not
  weakened the other way — `catalyzes` still demands a real `Protein`.
- `ontology.emitted_labels`, `validate` and `summary` count Stage 3's semantic
  types as a source of span labels, not just the NER models. `Finding` and
  `Symptom` are reachable only that way, which is why `has_function` was being
  reported as a relation type that could never fire.
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

- The evidence-scope guard checks the sentences the cited quote sits in, not
  the quote itself. Extractors quote the clause carrying the relation and leave
  the subject in the one before it — 82% of the quotes in a ten-document run
  were not whole sentences, producing demotions like `stimulates` missing 'PTH'
  over the quote "on kidneys to enhance calcium reabsorption". Re-running the
  guard over that run's saved IRs: 581 demotions become 275, so 306 sound
  relations stay affirmed. The property the check is for is unchanged — support
  must still be local, not gathered from across a 1,500-character chunk.
- Stage 4 no longer models a deviation modifier it has just reported as
  misapplied. `Hypoparathyroidism`, `Cushing's syndrome`, `Acromegaly` and
  `Dwarfism` were each getting an instance node carrying `hasDeviation` and no
  label, alongside the review entry saying the modifier did not belong there.
- Stage 4 rejects a pair of deviations that contradict each other
  (`deficiency` + `excess` on one span) and a deviation the span's own text
  already states (`Iodine deficiency` + `deviation: deficiency`, which is where
  `ADH deficiency caused_by ADH` came from). Both go to review and neither is
  modelled. Only antonym pairs count as contradictory, so insulin deficiency
  and insulin resistance are still both recorded.
- `stage4_postcoord.postcoordinate` is now idempotent: spans that already have
  an instance are skipped, so re-running it over a saved IR (which a resumed
  corpus run does) no longer appends a second, identical set of instance nodes
  and duplicate review entries.
