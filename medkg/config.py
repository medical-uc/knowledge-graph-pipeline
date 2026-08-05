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

# --- Fallback lexicon for common modifier VALUES ----------------------------
# Used when a modifier value can't be linked by the linker.
MODIFIER_VALUE_LEXICON: dict[str, str] = ONTOLOGY["modifier_values"]

# NER model -> {raw label: ontology span label}. A relation can only fire if
# some model here emits both of its endpoint labels.
NER_LABEL_MAPS: dict[str, dict] = ONTOLOGY["ner_models"]
NER_MODELS: tuple = tuple(
    os.environ.get("MEDKG_NER_MODELS", "en_ner_bc5cdr_md").split(","))

# Adjective lexicons used by the Stage 2 modifier heuristic.
SEVERITY_WORDS = {"severe", "mild", "moderate", "marked", "slight"}
ACUITY_WORDS = {"acute", "chronic", "subacute"}
LOCATION_WORDS = {"substernal", "retrosternal", "epigastric", "precordial"}

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
