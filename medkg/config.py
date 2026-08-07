"""Central config: vocabularies, ontology mappings, thresholds, model ids.

NOTE ON IDENTIFIERS: the SNOMED/RxNorm codes below are ILLUSTRATIVE placeholders
chosen to make the worked example concrete. Verify every code against your
licensed UMLS/SNOMED release before production use — a wrong code silently
corrupts the graph.
"""
from __future__ import annotations

import os

from . import ontology

# --- Namespaces -------------------------------------------------------------
SNOMED = "http://snomed.info/id/"
RXNORM = "http://purl.bioontology.org/ontology/RXNORM/"
UMLS   = "http://linkedlifedata.com/resource/umls/id/"
ONT    = "http://example.org/medkg/ont#"
INST   = "http://example.org/medkg/inst/"
GRAPH  = "urn:graph:"
DCTERMS = "http://purl.org/dc/terms/"
GROUP  = "urn:group:"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_LABEL    = "http://www.w3.org/2000/01/rdf-schema#label"

# --- Relation ontology ------------------------------------------------------
# --- Relation & modifier ontology (DATA, not code) ---------------------------
# Loaded from JSON so it can be swapped per corpus without editing the package:
#   MEDKG_ONTOLOGY=/path/to/ontology.json     (or run.py --ontology)
# A cardiology ontology extracts almost nothing from a physiology chapter, so
# this is the knob you will turn most often. `medkg.ontology.validate` reports
# relation types whose head/tail labels no configured NER model can emit --
# those can never fire, and that is the usual reason yield looks broken.
ONTOLOGY = ontology.load()

ANY = "*ANY*"        # sentinel: any entity label is allowed

# --- Where generated files go ----------------------------------------------
# Everything the pipeline writes -- normalized text, stage checkpoints, the
# .nq, the IR, MCQs -- lands here instead of scattering across the project
# root. Overridable per-run with --artifacts or MEDKG_ARTIFACTS.
ARTIFACTS_DIR = os.environ.get("MEDKG_ARTIFACTS", "artifacts")


def artifact_path(name, artifacts_dir=None) -> str:
    """Resolve an output filename into the artifacts directory.

    A BARE FILENAME is a name and gets redirected; anything carrying a directory
    component is a path and is left exactly as given. So `--out graph.nq` writes
    `artifacts/graph.nq`, while `--out ./graph.nq`, `--out /tmp/g.nq` and
    `--out sub/dir/g.nq` all mean what they say. That distinction is the whole
    contract: a caller who has typed a path has already decided where the file
    goes, and silently moving it would be worse than not having the feature.

    The directory is created on demand, so nothing is made until something is
    actually written.
    """
    if not name:
        return name
    head, tail = os.path.split(name)
    if head or os.path.isabs(name):
        # Honour the path, but make sure it can be written: a full run failing
        # at serialisation time because a parent directory is missing throws
        # away everything the pipeline just computed.
        if head:
            os.makedirs(head, exist_ok=True)
        return name
    target = artifacts_dir or ARTIFACTS_DIR
    os.makedirs(target, exist_ok=True)
    return os.path.join(target, tail)
# tail is None        # sentinel: UNARY relation (no tail), e.g. patient-level assertions
RELATION_ONTOLOGY: dict[str, dict] = ONTOLOGY["relations"]

# A child type is asserted INSTEAD OF its parent; Stage 7 re-derives the parent
# from an rdfs:subPropertyOf axiom, so the general fact stays queryable but is
# marked derived rather than claimed twice.
RELATION_PARENTS: dict[str, str] = ontology.parent_map(ONTOLOGY)

# Span-label subsumption: a constraint written as `Substance` is satisfied by a
# span the NER called `Drug`. See ontology.label_satisfies for why.
LABEL_HIERARCHY: dict[str, str] = ONTOLOGY["label_hierarchy"]

# other-name -> (canonical relation type, swap head/tail)
RELATION_ALIASES: dict[str, tuple] = ontology.alias_map(ONTOLOGY)

# --- Post-coordination: modifier type -> attribute property -----------------
SNOMED_ATTR: dict[str, str] = ONTOLOGY["modifiers"]
# Full modifier specs (uri + source + gloss); the glosses are prompt material,
# not documentation, exactly as the relation glosses are.
MODIFIER_META: dict[str, dict] = ONTOLOGY["modifier_meta"]

# Modifiers that change WHAT THE SPAN DENOTES rather than merely qualifying it.
# "severe pain" is still pain, so a relation about it can point at the concept;
# "iodine deficiency" is not iodine, and a relation pointing at the bare concept
# asserts the opposite of the source. Stage 6 therefore redirects an endpoint
# carrying one of these onto its Stage-4 post-coordinated instance, and only
# for these -- routing every modified mention through an instance would shatter
# the graph into per-occurrence nodes and defeat the point of linking.
ROLE_CHANGING_MODIFIERS: tuple = ("deviation",)

# ...and the span labels a deviation can sensibly apply to. A deficiency or an
# excess is a property of a SUBSTANCE. A run attached deviation:excess to the
# span "hyperthyroidism" (from "accelerate metabolism excessively") and minted
# `hyperthyroidism [excess]` as a second, parallel subject node — asserting the
# same fact twice under two URIs. A disease is already a deviation; it does not
# take one. Labels outside this set keep the attribute (it stays auditable on
# the instance) but do NOT redirect the relation endpoint.
DEVIATION_APPLIES_TO: tuple = ("Substance", "Drug", "Protein")

# --- Lexicon for modifier VALUES --------------------------------------------
# Consulted BEFORE the Stage-3 link, and the only source accepted for a
# role-changing modifier: a deviation value is a closed vocabulary, and asking
# the linker to resolve a bare "deficiency" with no context returns whatever
# SNOMED concept is nearest in embedding space, which in one run was a disease.
MODIFIER_VALUE_LEXICON: dict[str, str] = ONTOLOGY["modifier_values"]
# Value pairs that cannot both hold of one mention -- deficiency and excess.
MODIFIER_VALUE_ANTONYMS: frozenset = ONTOLOGY["modifier_value_antonyms"]

# NER model -> {raw label: ontology span label}. A relation can only fire if
# some model here emits both of its endpoint labels.
#
# These are the labels a span STARTS with. Stage 3 re-derives them from the
# linked concept's UMLS semantic type wherever it can (`SEMANTIC_TYPE_LABELS`),
# because an NER model outside its training domain gets this badly wrong: on an
# anatomy corpus bionlp13cg tags cavities as PATHOLOGICAL_FORMATION and glands
# as CANCER, which was 72% of one run's relation rejections.
NER_LABEL_MAPS: dict[str, dict] = ONTOLOGY["ner_models"]

# UMLS semantic type (TUI) -> ontology span label, and the order to prefer them
# in when a concept carries several. Needs `semantic_types.json` beside the
# FAISS index; without it Stage 3 leaves the NER label alone.
SEMANTIC_TYPE_LABELS: dict[str, str] = ONTOLOGY["semantic_types"]
SEMANTIC_TYPE_PRIORITY: tuple = ONTOLOGY["semantic_type_priority"]
NER_MODELS: tuple = tuple(
    os.environ.get("MEDKG_NER_MODELS", "en_ner_bc5cdr_md").split(","))

# Adjective lexicons used by the Stage 2 modifier heuristic.
SEVERITY_WORDS = {"severe", "mild", "moderate", "marked", "slight"}
ACUITY_WORDS = {"acute", "chronic", "subacute"}
LOCATION_WORDS = {"substernal", "retrosternal", "epigastric", "precordial"}

# --- The graphs that are ABOUT the corpus rather than part of it -------------
# Asserted facts live in one graph per (document, section) -- see
# `stage6_assert.graph_for` -- so a section, a document or a group of documents
# can be lifted out as a set of graphs rather than reconstructed by query. These
# four are the exceptions: they describe the corpus, hold what is true of it
# corpus-wide, or hold what the reasoner derived, and none of them belongs to a
# single section.
#
#   provenance  what each document is and what model wrote its prose
#   catalog     the document/section/group structure a subset is selected BY
#   labels      one rdfs:label + mentionCount per concept, aggregated over every
#               document. Per-document labels would give a concept mentioned in
#               three documents three labels, and every relation row would come
#               back three times.
#   inferred    Stage 7's materialized closure
GRAPH_PROVENANCE = GRAPH + "provenance"
GRAPH_CATALOG    = GRAPH + "catalog"
GRAPH_LABELS     = GRAPH + "labels"
GRAPH_INFERRED   = GRAPH + "inferred"
# Graphs a subset always carries: without labels every answer is a bare URI, and
# without the catalog the extracted file cannot say what it is a subset OF.
CORPUS_GRAPHS = (GRAPH_PROVENANCE, GRAPH_CATALOG, GRAPH_LABELS, GRAPH_INFERRED)

# Catalog vocabulary.
DOCUMENT_CLASS  = ONT + "Document"
SECTION_CLASS   = ONT + "Section"
GROUP_CLASS     = ONT + "DocumentGroup"
HAS_SECTION     = ONT + "hasSection"
IN_DOCUMENT     = ONT + "inDocument"
HAS_MEMBER      = ONT + "hasMember"
SECTION_PATH    = ONT + "sectionPath"
SECTION_ORDER   = ONT + "sectionOrder"
FROM_SECTION    = ONT + "fromSection"

PROV               = "http://www.w3.org/ns/prov#"
WAS_GENERATED_BY   = PROV + "wasGeneratedBy"
WAS_DERIVED_FROM   = PROV + "wasDerivedFrom"
SOFTWARE_AGENT     = PROV + "SoftwareAgent"
GENERATED_AT       = PROV + "generatedAtTime"

# --- Stage 5: figure vocabulary ---------------------------------------------
DEPICTS            = ONT + "depicts"
CAPTION            = ONT + "caption"
VISUAL_DESCRIPTION = ONT + "visualDescription"
FIGURE_IMAGE       = ONT + "FigureImage"
FIGURE_REF         = ONT + "figureRef"

# --- Stage 1: markdown parsing ----------------------------------------------
# Chunk kinds. Only EXTRACTABLE_CHUNK_KINDS reach Stage 2; captions and figure
# descriptions are routed to Stage 5 so image prose never becomes a clinical fact.
# Stage 1's input is the tagged document the rewriter emits, so chunk kinds map
# straight onto its tags rather than being inferred.
#   prose <con>   definition <def>   keypoint <key>   clinical <clin>
#   summary <sum> caption <cap>      figure_description <desc>
#   table <tbl>   reference <ref>    objective <obj>  duplicate (deduped)
# A chunk is assertable text, full stop. Figure captions and descriptions live on
# `Figure` (storing them twice bought nothing — no stage read the chunk copies);
# objectives, tables and bibliography live in `Document.passages`.
CHUNK_KINDS = ("prose", "definition", "keypoint", "clinical", "summary", "duplicate")
PASSAGE_KINDS = ("objective", "table", "reference")
# Which chunk kinds Stage 2 reads. Everything but `duplicate` by default; narrow
# it to drop e.g. the closing <sum> without re-running Stage 1.
EXTRACTABLE_CHUNK_KINDS = ("prose", "definition", "keypoint", "clinical", "summary")


# --- Thresholds -------------------------------------------------------------
# Stage 3 only pays for an LLM rerank when cosine is genuinely undecided: if the
# top candidate beats the runner-up by more than this margin, the embedding has
# already answered and a 72B call would just agree with it slowly.
RERANK_MARGIN         = float(os.environ.get("MEDKG_RERANK_MARGIN", "0.05"))
LINK_CONFIDENCE_FLOOR = float(os.environ.get("MEDKG_LINK_FLOOR", "0.70"))
RE_CONFIDENCE_FLOOR   = float(os.environ.get("MEDKG_RE_FLOOR", "0.50"))
# Consecutive same-kind chunks in one section merge up to this size, so a
# relation spanning two paragraphs stays inside one Stage-2 call.
MAX_CHUNK_CHARS       = int(os.environ.get("MEDKG_MAX_CHUNK_CHARS", "1500"))

# --- Models -----------------------------------------------------------------
SAPBERT_MODEL = os.environ.get("MEDKG_SAPBERT", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# Set to a model string you have access to; see https://docs.claude.com for current ids.
LLM_MODEL = os.environ.get("MEDKG_LLM_MODEL", "claude-sonnet-4-5")

# SNOMED release the current run is linked against (recorded as provenance).
SNOMED_VERSION = os.environ.get("MEDKG_SNOMED_VERSION", "2026-03")
