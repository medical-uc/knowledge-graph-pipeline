# medkg — how the pipeline works

## 1. What it does

medkg turns a medical textbook chapter into a queryable RDF knowledge graph in
which every asserted fact traces back to the sentence it came from.

The input is not raw textbook prose. It is a **tagged markdown rewrite**: an LLM
has already read the source chapter and re-emitted it with structural tags
(`<con>` for concept prose, `<def>` for definitions, `<clin>` for clinical notes,
`<fig>` for figures, and so on). The pipeline consumes that rewrite.

The output is an N-Quads file containing typed relations between ontology-linked
concepts (SNOMED CT and RxNorm URIs), plus a parallel IR file recording
everything the pipeline was *unsure* about.

The whole design turns on one distinction:

> An extracted fact and a true fact are not the same thing.

A pipeline that emits triples without saying which ones it doubted produces an
artefact that looks authoritative and cannot be audited. So every stage that can
fail is built to **degrade into review rather than into worse triples**, and the
graph keeps three things separate at all times: what the source said, what the
reasoner derived, and what the pipeline could not verify.

### The end-to-end shape

```
tagged markdown
  → Stage 1  parse        → normalized text + chunks + figures + passages
  → Stage 2  extract      → spans, negation, relations, modifiers
             guards       → demote the relations that fail deterministic checks
  → Stage 3  link         → CUIs + SNOMED/RxNorm URIs on every span
  → Stage 4  postcoord    → instance nodes for qualified mentions
  → Stage 5  images       → figure captions linked as depictions
  → Stage 6  assert       → named-graph RDF quads
  → Stage 7  reason       → materialized inferences, in their own graph
  → Stage 8  serve        → SPARQL + N-Quads serialization
```

Stages 1, 4, 5, 6 and the guard layer are pure Python and fully testable
offline. Stage 2 needs an LLM and NER models; Stage 3 needs a UMLS-derived FAISS
index; Stage 7 needs `owlrl`. 62 offline tests cover the pure paths.

### A single fact, traced end to end

Take one sentence from the thyroid document:

> A deficiency in dietary iodine can lead to insufficient thyroid hormone
> production, resulting in conditions such as goiter and hypothyroidism.

| stage | what it produces |
|---|---|
| 1 | a `prose` chunk with the sentence at known character offsets |
| 2 | spans `iodine` (Substance), `goiter` (Disease); relation `goiter caused_by iodine`; modifier `deviation: "A deficiency in"` on `iodine` |
| guards | the relation survives (direction is right); the deviation is present so the endpoint is not flagged |
| 3 | `iodine → C0021968 → sct:62834003`; `goiter → C0018021 → sct:3716002` |
| 4 | mints `inst:7f49bd589122 a sct:62834003 ; ont:hasDeviation ont:Deficiency`, labelled `iodine [deficiency]` |
| 6 | `sct:3716002 ont:caused_by inst:7f49bd589122` in graph `urn:graph:thyroid_source` |
| 8 | the `relations` query prints `goiter caused_by iodine [deficiency]` |

Note what the graph does *not* say: it does not say `goiter caused_by iodine`.
Iodine prevents goiter. That single character of difference — a bare concept
versus a post-coordinated deviation node — is the difference between a correct
graph and one that inverts the source's clinical meaning, and Stage 4 exists
largely to hold it.

---

## 2. The invariants

These are enforced across stages and explain most of the design decisions below.

1. **Offsets are the contract.** Stage 1 writes a tag-free normalized rendering
   and asserts `normalized[chunk.char_start:chunk.char_end] == chunk.text` for
   every chunk. Every later stage's grounding check depends on that holding.
2. **Figure prose never becomes a clinical fact.** Captions and figure
   descriptions are routed away from relation extraction. A figure that *shows*
   pericarditis has diagnosed no one.
3. **Deterministic URIs.** Instance and assertion node ids are hashes of stable
   inputs, so re-running the pipeline on unchanged input produces byte-identical
   node identities rather than a second parallel graph.
4. **Negation is reified, never dropped and never asserted.** "No evidence of
   pericarditis" becomes a retrievable `ClinicalAssertion` with
   `ont:polarity ont:Negated`.
5. **Confidence floors abstain rather than guess.** Below the floor, content
   goes to review instead of the graph.
6. **Grounding guards.** An LLM-proposed relation must quote text that actually
   exists in the chunk, and that quote must mention both entities.
7. **A prompt is not a guarantee; guards are.** Deterministic checks re-examine
   every relation after the model has spoken.
8. **Asserted and inferred live in different named graphs**, permanently.
9. **The LLM backend is injectable.** No stage hardcodes a provider.

---

## 3. Stage 1 — parse

**Purpose.** Turn tagged markdown into offset-anchored chunks, and record where
the text came from.

**Input.**

```xml
<meta>
<title>The Thyroid Gland and Its Hormones</title>
<src>source: thyroid_source.md | figures: 33 | model: huatuogpt-o1-72b-4bit | generated: 2026-07-31 09:16:43</src>
</meta>

<sec id="s1" level="2">
<head>Introduction to the Thyroid Gland</head>
<con>
The thyroid gland is the largest endocrine organ in the human body. Located in
the neck, just below the larynx, it consists of two lobes connected by a thin
strip of tissue called the isthmus. [[FIG:p5_b2]]
</con>
<fig id="p2_b3">
<cap>The figure illustrates a labeled diagram of the human endocrine system.</cap>
<desc>The diagram provides a side view of the human body from the head down.</desc>
</fig>
</sec>
```

**Output.** A normalized text file plus a `Document`:

```json
{
  "chunk_id": "c001",
  "text": "The thyroid gland is the largest endocrine organ in the human body. Located in the neck, just below the larynx, it consists of two lobes connected by a thin strip of tissue called the isthmus.",
  "char_start": 194,
  "char_end": 494,
  "section_path": ["The Thyroid Gland and Its Hormones", "Introduction to the Thyroid Gland"],
  "kind": "prose",
  "figure_refs": ["p5_b2"]
}
```

Chunk kinds are `prose`, `definition`, `keypoint`, `clinical`, `summary`,
`caption`, `figure_description` and `duplicate`. Only the first five reach
Stage 2.

### Design decisions

**`chunks` holds only extractable body text.** Captions and figure descriptions
live on `Figure`; objectives, tables and bibliography live on `Document.passages`.
Chunks used to hold captions as well as storing them on `Figure`, which was pure
duplication and cost 27% of chunk text even on the small sample. But the
non-extractable material is *moved*, not deleted — everything is still written to
the normalized file and still carries a span, so a depiction edge can point at
where its caption sits. Deleting source material to tidy a data structure is not
a trade worth making.

**Provenance is separated from the source.** The document being parsed was
written by a model. `<src>` records what the rewriter was given and which model
wrote it; these become `Source.origin`, `.generator` and `.generated_at`, and
Stage 6 asserts the original as `dcterms:source` and the rewrite as
`prov:wasGeneratedBy` a `prov:SoftwareAgent`. Collapsing the two would let a
triple extracted from fresh LLM prose pass as one traceable to the textbook. This
is the load-bearing honesty property of the design: the rewriting step is
unguarded, so the graph must at least be explicit about where its sentences came
from.

**`[[FIG:id]]` anchors are lifted out of the prose** into `Chunk.figure_refs`.
Left inline they sit inside Stage 2's evidence substrings and shift every span
offset after them.

**Exact duplicates are demoted, not dropped.** The rewriter's `<key>` blocks
frequently restate `<con>` verbatim. A byte-identical repeat becomes
`kind="duplicate"` and is review-flagged: the text stays in the normalized file,
it just is not sent to the LLM a second time.

**Markdown markers are a regression signal.** The current rewriter emits no `#`,
`**` or `- ` inside tag content — the tags carry the structure. The parser
therefore *reports* any that appear while still repairing them, because repairing
silently would hide the fact that the rewriter regressed.

**Unanchored figures are surfaced.** `Figure.referenced_from` is bound only from
an anchor in prose, never from the figure's own caption. Binding it to the
caption would make every figure look referenced and hide the ones the rewriter
dropped into a section without ever mentioning them.

The parser is hand-rolled rather than XML, because the body carries markdown,
bare ampersands and `[[FIG:…]]` anchors that a strict parser would reject.

---

## 4. Stage 2 — extract

**Purpose.** Find entities, their negation, the relations among them, and their
modifiers.

**Input.** The extractable chunks.

**Output.**

```json
"spans": [
  {"span_id":"s001","text":"STEMI","label":"Disease","negated":false,
   "modifiers":[{"type":"acuity","text":"Acute"}]},
  {"span_id":"s003","text":"chest pain","label":"Symptom",
   "modifiers":[{"type":"severity","text":"severe"},
                {"type":"location","text":"substernal"}]},
  {"span_id":"s006","text":"pericarditis","label":"Disease","negated":true}
],
"relations": [
  {"rel_id":"r001","head":"s001","tail":"s002","type":"caused_by",
   "polarity":"affirmed","score":0.86,"evidence_start":42,"evidence_end":118},
  {"rel_id":"r003","head":"s006","tail":null,"type":"present_in_patient",
   "polarity":"negated","score":0.90}
]
```

### Four passes

**NER.** Two scispaCy models, `en_ner_bc5cdr_md` and `en_ner_bionlp13cg_md`,
with `AbbreviationDetector` and negspacy. Both models are required: bc5cdr alone
emits only DISEASE and CHEMICAL, which leaves 9 of 27 relation types with no
reachable head or tail label — no BodyStructure, Cell or Protein means no
`secretes`, `part_of`, `located_in`, `catalyzes` or `converts_to`. Abbreviation
resolution is not optional in medicine, where "MS" is three different concepts.

**Relations.** One LLM call per chunk. `candidate_pairs()` uses the ontology's
head/tail type constraints to prune which entity pairs are even offered, which
shrinks the model's job and blocks structurally impossible proposals before they
are made. `validate_relation()` then enforces ontology membership, type
constraints, grounded evidence, and the confidence floor (0.50).

**Modifiers.** A second LLM call, not a dependency parse. Bulleted and
fragmentary medical text does not produce clean dependency trees, and the
heuristic version failed on exactly the material this corpus is made of. The
modifier menu is generated from the ontology rather than hardcoded: an earlier
hand-written list named five types while the ontology had nine, so four types
were accepted by the validator and never once requested from the model — a
silent recall hole invisible to tests, because both ends were self-consistent.

**Guards.** Covered in section 5.

### Fragment awareness

`looks_structured()` detects titles and bullets, and `_context_header()` injects
the section heading and a scope hint into both prompts. A bullet under
"Contraindications" keeps the meaning its heading gives it. This is the single
highest-leverage adaptation for non-prose sources.

### Rejection reasons

Every rejected candidate is recorded with a **structured code**, not just the raw
item: `unknown-type`, `head-span-missing`, `tail-span-missing`,
`head-label:<type>`, `tail-label:<type>`, `below-floor`, `ungrounded-evidence`.
Before codes existed, a real run produced 144 rejections that were
indistinguishable from one another, and diagnosing them meant re-implementing the
validator by hand against the IR. 88% turned out to share a single cause.

### Aliases and inverses

Extractors reach for the inverse of a relation type when the menu offers only one
direction. A real run invented 17 types — `produced_by`, `stored_in`,
`required_for`, `affected_by` — nearly all of which named a relation the ontology
already had, read from the other end. The ontology now declares `aliases` (same
direction, different name) and `inverseOf` (accepted by swapping head and tail),
37 names in total. `binds_to` and `incorporates_into` remain unmapped: no
existing type means either, and mapping them to something approximate would be
worse than the rejection.

---

## 5. The guard layer

Fourteen deterministic checks run over the extracted relations after the LLM has
spoken. They are pure stdlib, so they can be re-run over a saved IR without
paying for extraction again.

**Why they exist.** The prompt already stated head/tail direction per relation
type, and a both-ways contradiction check was already running. An audit of a real
run found **12 of 18 relations still wrong**. More tellingly, the reversals that
survived were exactly the ones whose phrasing differed from the prompt's worked
example — the model was matching the example, not applying the rule. That is not
fixable by better prompting.

| guard | catches |
|---|---|
| `flag_ungrounded_endpoints` | the cited quote does not *mention* both entities |
| `flag_governed_roles` | the span is a fragment of the phrase that carries the meaning |
| `flag_causal_direction` | `caused_by` direction disagrees with the sentence's connective |
| `flag_conversion_direction` | same, for `converts_to` |
| `flag_antonym_contradictions` | one cause, two opposite effects |
| `flag_positive_feedback_loops` | a hormone stimulating the gland recorded as secreting it |
| `flag_self_loops` | head and tail are the same concept |
| `flag_fragment_endpoints` | an enumerator with no head noun (`Type 3`) |
| `flag_vacuous_endpoints` | a bare category word (`organ`) |
| `flag_structural_deviation` | a deviation node used as a location |
| `flag_enzyme_as_substrate` | a protein the document calls a catalyst used as a reaction product |
| `dedupe_relations` | one fact extracted from two chunks |
| `prune_subsumed_relations` | a parent type asserted beside its child |
| `flag_degenerate_confidence` | self-reported scores that carry no information |

### The three that matter most

**Governed roles** was the largest error family — 9 of 12 defects — and the most
dangerous, because the resulting triple is not merely imprecise, it asserts the
*reverse* of the source. `goiter caused_by iodine` came from "a **deficiency in**
dietary iodine"; `calcitonin inhibits bone` from "inhibiting bone **resorption**";
`Hashimoto's caused_by thyroid peroxidase` from "autoimmune reactions **against**
thyroid peroxidase". In each case the span is a bare noun and the sentence
attributes the effect to a phrase built around it.

The guard uses two lexicons: a set of phrases that may appear before the span
within about 48 characters, and a set of head nouns that must sit immediately
after it. One omission is load-bearing: `inhibiting` is *not* a pre-cue, even
though "inhibiting bone resorption" is the sentence that motivated the guard.
Adding it would flag every correct `inhibits` relation in the corpus, since the
relation verb naturally sits just before its object. The trailing `resorption` is
what makes that case detectable, which is why the post-cue list exists at all.

**Causal direction** adjudicates from the sentence's own connective: forward cues
(`causes`, `leads to`, `is a cause of`) mean the earlier entity is the cause;
backward cues (`caused by`, `results from`, `secondary to`) mean the opposite. A
window carrying cues of both classes returns no verdict rather than a guess.
`converts_to` uses the same machinery with a chemical lexicon, and its
orientation is **opposite**: `caused_by` runs effect → cause, `converts_to` runs
substrate → product, so a forward cue means the head should come first.

**Evidence scope** turned out to be the guard that fired most often, and the one
that most needed refining. Stage 2 originally asked only that the quote be a
substring of the chunk, which is not the same as the quote *supporting* the
relation: a model can cite one sentence while relating an entity from another.
But an exact-string check over-fires, because scispaCy resolves abbreviations —
a span can carry the text `triiodothyronine` while the sentence says `T3`.
Grounding therefore also accepts a co-referent mention of the same linked concept
inside the quote.

### Two rules that shape all of them

**Demote, never delete.** Every guard sets `polarity = "uncertain"` and files a
review entry naming the cue that fired. Stage 6 keeps uncertain relations out of
the asserted graph but reifies them, so they stay retrievable. Deleting a correct
fact is not an improvement on flagging a suspect one.

**Cross-relation checks key on the linked concept, never the span id.** Two
mentions of hypothyroidism in different chunks are two spans and one fact. An
early version of the subsumption check keyed on span ids and let a redundant pair
from two chunks straight through.

---

## 6. Stage 3 — entity linking

**Purpose.** Map each mention to a stable CUI and ontology URI, so synonyms
across sources collapse to one node. This is the alignment step.

**Input.** Spans plus surrounding chunk text for disambiguation.

**Output.**

```json
{"span_id":"s001","text":"STEMI","label":"Disease",
 "cui":"C0340305","uri":"http://snomed.info/id/401303003","link_score":0.95}
```

**Three phases.**

1. *Index build, offline and once.* Streams `MRCONSO.RRF`, embeds each English
   atom with **SapBERT** ([CLS], L2-normalized), and builds a cosine FAISS
   `IndexFlatIP`. Streaming keeps memory bounded; a vocabulary filter is the main
   size lever.
2. *Candidate generation.* Embed the mention, ANN-search top-k, collapse to
   best-score-per-CUI.
3. *Disambiguation.* An optional LLM rerank (candidates carry their names) picks
   the context-correct CUI; otherwise top-1 cosine wins. Below the link
   confidence floor of 0.70, the span goes to review rather than the graph.

**Why SapBERT.** It is trained on UMLS synonym pairs, which is precisely the task.
BiomedCLIP, the obvious alternative, is an image–text model and solves a
different problem.

**URI choice.** `cui_to_uri` prefers RxNorm, then SNOMED, then the raw UMLS CUI
URI. Only drug concepts carry RxNorm codes, so preferring RxNorm routes drugs to
RxNorm and everything else to SNOMED without needing a type check.

**Failure mode.** A confidently wrong high-similarity link corrupts the graph
silently, which is why the floor and abstention exist. The index must be built
with the same SapBERT model used at query time, or the vectors live in different
spaces.

---

## 7. Stage 4 — post-coordination

**Purpose.** Turn qualifiers into first-class queryable structure instead of
pre-coordinated classes, which explode combinatorially.

**Input.** Linked spans carrying modifiers.

**Output.** An instance node per qualified, non-negated span, typed as the
concept, with one attribute triple per modifier:

```turtle
inst:7f49bd589122  a sct:29857009 ;                      # chest pain
    sct:246112005 sct:24484000 ;                         # severity: Severe
    sct:363698007 sct:51185008 ;                         # location: Substernal
    rdfs:label "chest pain [severe]" .
```

Instance ids are `det_id(source, char_offset, cui)`, so re-running is idempotent.
Modifier values resolve via the Stage-3 link, then a configured lexicon, then a
review-flagged fallback. Negated spans get no instance.

### The deviation modifier

Ten modifier types are defined; one of them does something the others do not.
`severity`, `laterality`, `acuity` and the rest *qualify* a concept — severe pain
is still pain, so a relation about it can point at the shared concept. But
**`deviation`** — deficiency, excess, resistance — changes what the span
*denotes*. Iodine deficiency is not a kind of iodine. It is arguably its
opposite.

So `ROLE_CHANGING_MODIFIERS` names the deviation type, and Stage 6 redirects a
relation endpoint carrying one onto the instance minted here. Only that type
redirects: routing every "severe" through an instance would shatter the graph
into per-occurrence nodes and undo Stage 3's entire purpose.

This was added because the modifier vocabulary originally had **no slot for
deviation at all**. "A deficiency in dietary iodine" was structurally
unrepresentable, so the graph asserted that iodine causes goiter — the exact
inverse of the source, and the opposite of the clinical advice.

**Deviation is restricted by span type.** `DEVIATION_APPLIES_TO` is Substance,
Drug and Protein. A run attached `deviation: excess` to the span
"hyperthyroidism" and minted `hyperthyroidism [excess]` as a second subject node,
asserting one fact twice under two URIs. A disease *is* a deviation; it cannot
take one. A misapplied modifier is reported rather than silently modelled.

**Labels prefer words to codes.** Stage 3 links modifier values, so composing a
label from the resolved URI produced `iodine [260372006]` — technically correct
and unreadable. Label composition consults the lexicon first and never uses a
numeric segment. The code still goes on the instance as the attribute value; only
the human-facing label prefers a word.

---

## 8. Stage 5 — images

**Purpose.** Bring figures into the graph as concept-linked nodes, and catch the
caption text Stage 1 routed away from relation extraction.

**Input.** Figures plus their caption and description text.

**Output.** `Depiction` records, which Stage 6 turns into `ont:depicts` edges,
with `ont:caption` and `ont:visualDescription` literals on the figure node.

Figure text is run through the same NER and the same linker (both injected, so
nothing heavy is imported at module scope). Two rules make this safe:

**`Depiction`, not `Span`.** Figure concepts never enter `doc.spans`, so they
cannot head a relation, cannot mint an instance, and are never asserted as fact.

**Caption over description.** Depiction edges are restricted to the caption,
which carries the medicine. The visual description — "a gradient of light blue",
"a leader line points to" — is kept as a literal for retrieval and provenance
rather than linked. Negated caption mentions depict nothing.

This firewall is testable and has been observed holding: a rewriter hallucination
in one figure description ("MIT (monochloride)" for monoiodotyrosine) never
reached the graph.

---

## 9. Stage 6 — RDF assertion

**Purpose.** Commit the enriched IR to RDF. This is where annotations *about a
source* become *held facts*.

**Output.** A named-graph dataset:

```turtle
# graph urn:graph:thyroid_source
sct:3716002  ont:caused_by  inst:7f49bd589122 .        # goiter ← iodine [deficiency]
inst:7f49bd589122 a sct:62834003 ; ont:hasDeviation ont:Deficiency ;
                  rdfs:label "iodine [deficiency]" .

<< sct:3716002 ont:caused_by inst:7f49bd589122 >> ont:confidence 0.9 .

inst:assert_a1b2c3 a ont:ClinicalAssertion ;           # negation, reified
    ont:about sct:3238004 ; ont:polarity ont:Negated .

# graph urn:graph:provenance — asserted once, not per triple
urn:graph:thyroid_source dcterms:source "thyroid_source.md" ;
    prov:wasGeneratedBy [ a prov:SoftwareAgent ; rdfs:label "huatuogpt-o1-72b-4bit" ] .
```

`build_quads()` is pure Python and returns quads plus annotations, so triple
generation is unit-testable; the rdflib adapter is a thin wrapper.

**One named graph per (source-version, section)**, with provenance asserted once
*on the document graph* rather than stamped on every triple. Per-triple
confidence uses RDF-star, serialized through a Turtle-star round trip, with an
`rdf:Statement` reification fallback if the installed rdflib rejects it — and the
statement node's id includes its graph, so two documents that extracted the same
fact do not share one node and attach two confidences to it.

The section granularity is what makes a subset extractable, and it follows from
the choice that was already made here: provenance lives on the graph, not the
triple. Making the section a graph is therefore a change of grain, not of design.

```
urn:graph:thyroid_source        the document -- its provenance, and anything
                                outside every <sec>
urn:graph:thyroid_source/s3     one section of it
urn:graph:labels                one label per concept, corpus-wide
urn:graph:catalog               which documents, sections and groups exist
urn:graph:provenance            what each document is, and what wrote its prose
urn:graph:inferred              Stage 7
```

**Concept labels are corpus-level.** `rdfs:label` is functional in practice even
though RDF does not enforce it: the `relations` query OPTIONAL-joins a label onto
each endpoint, so a concept carrying a label from each of three documents returns
every relation about it three times. One label per concept, in `urn:graph:labels`,
chosen by frequency across the whole corpus.

**A `source_id` collision is refused, not merged.** The id names the graph *and*
seeds the Stage-4 instance hashes, so two documents sharing one would mint the
same instance node for two different spans at the same offset, and the attributes
of one would land on the other. It is a live risk rather than a theoretical one,
because `source_id` comes from `<src>source:` — the original material — not from
the input filename: two chapters rewritten from one textbook legitimately name
the same origin. In this corpus, `parathyroid_gland.md` already asserts into
`urn:graph:thyroid_source`.

**Uncertain relations are reified, not asserted.** A guard-demoted relation
becomes a `ClinicalAssertion` with `ont:polarity ont:Uncertain` — kept out of the
asserted triples, still retrievable through the `flagged` query.

---

## 10. Stage 7 — reasoning

**Purpose.** Materialize inferences into a separate graph.

`owlrl` computes an RDFS/OWL-RL closure in process, and everything derived lands
in `urn:graph:inferred`. Keeping it separate is what lets a reader tell "the
source said this" from "the reasoner derived this", and makes re-reasoning clean.

Two kinds of axiom drive it. Subclass chains give the classic case: asserted
`STEMI ⊑ MI ⊑ IHD` means a query for ischemic heart disease returns STEMI though
no source said so.

The second kind is generated from the relation ontology. Three types declare a
`subPropertyOf` parent — `stimulates` and `inhibits` under `regulates`,
`complication_of` under `caused_by` — and the axioms are emitted from that
declaration rather than written out, so adding one is a single line of JSON.

This is what makes the subsumption guard lossless. A run that emitted both
`TRH regulates TSH` and `TRH stimulates TSH` was claiming one fact twice; the
guard drops the parent, and Stage 7 derives it back into the inferred graph. The
general fact stays queryable while remaining distinguishable from what a source
actually said.

Full OWL-DL over SNOMED-sized data will not terminate; SNOMED classification is
OWL 2 EL, and the shipped schema is a small stand-in.

---

## 11. Stage 8 — serving, and the query layer

`flatten()` unions named graphs into one queryable graph, `run_query()` runs
SPARQL, and `serialize()` writes N-Quads for a triplestore.

Nine built-in queries are generated from the ontology, so a new relation type
appears in them automatically:

| query | shows |
|---|---|
| `relations` | what the **source asserted** — excludes uncertain and excludes derived |
| `inferred` | what the **reasoner derived** — no confidence, because nothing extracted it |
| `flagged` | relations the guards demoted, and why |
| `evidence` | per-triple confidence and extracting model |
| `concepts`, `instances`, `negated`, `figures`, `provenance` | the rest |

Graph selection is explicit. `relations` loads every named graph *except* the
inferred one; `inferred` loads only that one, carrying labels in from elsewhere
since labels are asserted and never derived; `--include-inferred` overrides.

This split exists because a run once showed `myxedema caused_by hypothyroidism`
with a blank confidence sitting beside `myxedema complication_of hypothyroidism`
at 0.9. That reads as a duplicated extraction and is in fact the subPropertyOf
axiom working correctly — the query was unioning every graph and discarding the
distinction Stage 7 exists to maintain.

---

## 11b. Corpora, and subsets of them

`run.py --mode corpus` builds one graph from many documents. Stages 1–5 run per
document into `artifacts/docs/<stem>/`; Stage 6 sees all of them at once, Stage 7
reasons over the union, Stage 8 writes one `.nq`.

**Documents merge at the RDF layer, not the IR layer.** Concatenating IRs first
looks simpler and is not: chunk offsets index into a per-document normalized
file, and chunk and span ids are only unique within a document, so a merged IR
would have to renumber both — and renumbering breaks the offset contract every
grounding check depends on. Merging at the graph layer needs neither, because the
documents were already destined for different named graphs. It also means Stage 6
is the first stage that has to know a corpus exists.

**A corpus run resumes.** Stage 2 is an LLM call per chunk and Stage 3 loads
FAISS and SapBERT; over four documents that is hundreds of calls, and a failure
at the fourth used to be indistinguishable from a failure at the first. A
document whose IR is already on disk and newer than its input is not re-extracted.
This is the same reasoning as the single-document Stage-2 checkpoint, one level up.

**Extraction is set selection, not a query.** A section is a graph, a document is
its graph plus its sections', a group is the union over its members. `subset.py`
copies those contexts into a new dataset. Nothing decides what "belongs" to a
section by reachability — the subset is exactly what was asserted there, which is
the only answer that stays true as the graph grows.

Three things always come with a subset, each because leaving it out makes the
output worse than useless rather than merely smaller: **labels**, or every row is
a bare SNOMED URI; **provenance** for the documents included, because an extract
of a traceable graph that cannot say where its claims came from is not a smaller
version of that graph but a less honest one; and the **catalog** rows for what
was selected, so the file can state what it is a subset of.

**Inferred triples do not come along by default.** Stage 7 reasoned over the
whole corpus, so a derivation inside a scope may rest on a document outside it.
`--include-inferred` carries them as they stand; `--reason` re-derives them from
the subset alone, which is the answer that is actually true of it. For the same
reason `query.py` refuses `inferred` together with a scope rather than answering
a question it cannot answer honestly.

**Groups are declared, never inferred.** Two documents sharing a word in their
titles is not evidence that a reader wants them queried together, and a guessed
grouping would be indistinguishable in the graph from one a curator meant. They
live in a `corpus.json` manifest beside the inputs.

---

## 12. Reviewing what the pipeline was unsure about

Everything uncertain lands in one flat list on the IR, which is the right shape
to write and a poor one to read. A run produced 274 entries. Grouping them by
`(stage, code)` is what makes them tractable, and shows things like "104 of 111
type rejections trace to a handful of mislabelled spans, and 55 of those to one
entity". Review-mode grouping also **backfills codes onto older IRs** by
re-running the validator over the stored candidate, and re-checks against the
*current* ontology, marking an entry `would-now-pass` when a new label-hierarchy
entry or alias has since made it valid.

Because the guards are pure functions over a `Document`, they can also be re-run
standalone over a saved IR — no LLM, no models, no network — so auditing a graph
you have already paid to extract costs nothing.

---

## 13. The ontology as data

The relation ontology is a JSON document, not code. It defines 27 relation types,
each with a head and tail label constraint, a URI, and a **gloss** that is prompt
material rather than documentation — the glosses are what the model reads.

**Type constraints are hierarchical.** A `label_hierarchy` declares `Drug ⊑
Substance`, `MultiTissueStructure ⊑ BodyStructure`, `AnatomicalSystem ⊑
BodyStructure`, and a constraint is satisfied by the label or any ancestor. This
exists for a measured reason: bc5cdr maps every CHEMICAL to `Drug`, so in one run
115 of 539 spans could not satisfy a single `Substance` constraint, and 127
otherwise-valid relations were discarded on type grounds. Subsumption is
deliberately one-way — `treated_with` still demands a real `Drug`, and a bare
`Substance` will not do.

**A constraint that contradicts its own gloss is a bug in the constraint.**
`catalyzes` accepted `['Protein', 'Substance']` as a head while its gloss said
"HEAD is the ENZYME". Once `Drug ⊑ Substance` landed, every chemical in the
document became a candidate enzyme. It was narrowed to `Protein`.

Ten modifier types map to SNOMED attribute properties, so post-coordinated
expressions are valid SNOMED rather than a private vocabulary. Load-time
validation reports hierarchy cycles, unknown `subPropertyOf` targets, and relation
types no configured NER model can reach — a cycle would make subsumption pruning
delete both members of a pair and the reasoner loop.

---

## 14. LLM backends

Three backends satisfy one interface — `call_llm(system, user, schema=None)` —
and no stage hardcodes a provider. The two self-hosted ones are *profiles*
recording only what differs: base URL, model id, constraint mechanism, token
budget, and whether the model emits chain-of-thought. Reasoning traces are
stripped in the backend, so no stage has to know about them.

The failure policy is deliberate. **One failed call is not fatal**: Stage 2 makes
a call per chunk, and a single timeout at chunk 40 of 50 used to discard 39
chunks of completed work. A failure now returns empty, which every caller already
treats as "no items" and routes to review — recorded data loss, not silent, and
re-runnable. **A run of failures is fatal**: if the server is down, returning
empty forever produces an empty graph and reports success, so a circuit breaker
raises after a threshold of consecutive failures. In a medical pipeline, loud
failure beats a hollow artefact.

---

## 15. MCQ generation

Single-best-answer questions are generated from the asserted graph, with the
source sentence as a verified rationale. Selection and distractor logic are pure;
the LLM only rephrases stems.

Generation reads the IR rather than SPARQL, because it needs what the graph does
not carry: polarity to exclude demoted relations, evidence offsets for the
rationale, span labels for typed distractor pooling, and the uncertain relations
themselves as a blocklist.

**The open-world problem is the whole difficulty.** A graph records what was
extracted, not what is true. The absence of `hypothyroidism caused_by pituitary
failure` does not make it false. Three defences:

1. A distractor is rejected if it touches the head under **any** predicate in
   **any** state — asserted, inferred, or flagged. `uncertain` means unverified,
   never verified-false, so a demoted relation is the most dangerous thing to
   offer as a wrong answer.
2. Any edge to the head disqualifies, even under a different predicate. A
   defensible answer is a broken question.
3. Stems say "a recognised cause", never "the cause". This is what makes several
   true answers in the graph harmless: only one can reach the options, and an
   indefinite stem stays true when it does.

Distractors are tiered — same-role siblings first, then the confusions the guards
exist to catch (antonym pairs, deviation nodes), then same-semantic-type — and
each item records which tier each distractor came from. Fewer than three safe
distractors means no item; padding an option set is a giveaway, which is worse
than generating nothing.

**Negative stems are refused in code, not merely discouraged in the prompt.**
"All of the following EXCEPT" requires establishing that three statements are
false, which an open-world graph cannot do about anything. The stem guard also
rejects a rewrite that is not a question, exceeds 300 characters, or names the
key *or any distractor*. Every rejection falls back to the template stem, so a
misbehaving model costs prose and never correctness — the same failure direction
as Stage 2's grounding rule.

---

## 16. Where output goes

Everything the pipeline writes — normalized text, stage checkpoints, the
N-Quads, the IR, generated questions — goes to an artifacts directory. A corpus
run gives each document its own subdirectory, `artifacts/docs/<stem>/`, and
writes only the merged graph at the top. Sharing one set of filenames across
documents would have the second document's checkpoint overwrite the first's,
which is noticed at the end of the run, when the graph is short and everything
that could diagnose it is gone.

The path contract: a **bare filename** is a name and gets redirected, so
`--out graph.nq` writes into artifacts. Anything with a **directory component**
is a path and is used verbatim. Someone who typed a directory has already decided
where the file goes, and silently relocating it would be worse than not having
the feature. Parent directories are created either way, because a full run dying
at serialization time for a missing directory discards everything just computed.

The output path is resolved once, immediately after argument parsing, which is
what keeps the derived paths together: the Stage-2 checkpoint and the IR file are
both built from it by string surgery.

The UMLS/FAISS index is **not** routed there. It is an input to the pipeline and a
reusable asset built once, not a per-run output.

---

## 17. Known failure modes

These are properties of the pipeline as it stands.

**NER labels are the dominant constraint bottleneck.** In one run 104 of 111 type
rejections traced to mislabelled spans, and 55 to a single family: bc5cdr's
DISEASE recognizer fires on the document's central noun phrase, so `thyroid
gland`, `thyroid hormone` and `thyroid peroxidase` all come back as `Disease`.
The graph stays model-driven — these are reported rather than overridden — so
they remain visible in the review queue.

**The guards are lexicons.** They will miss phrasings not in their cue lists, and
will occasionally fire on a sound relation. That is the intended direction of
failure: degrade into more review, never into worse triples. The lists are
module-level data so a different corpus can extend them without touching logic.

**Nothing repairs a reversed relation.** A flagged relation is demoted and a
human decides.

**Self-reported confidence is degenerate.** Every relation in one run scored
exactly 0.9, which makes the RE floor inert — the safety valve is wired up and
cannot trip. The degenerate-confidence check reports this rather than fixing it.

**Coordination crossing is not detectable deterministically.** A run produced
`hypothyroidism treated_with antithyroid` from a sentence that states the pairing
correctly: "hormone replacement therapy for hypothyroidism and antithyroid
medications for hyperthyroidism". The correct pairing is carried by the two `for`
prepositions, and proximity actively misleads, since "hypothyroidism" and
"antithyroid" are adjacent in the string.

**Wrong relation *type* is harder than wrong arguments.** `TSH receptor
converts_to cAMP` came from "the conversion of ATP to cyclic AMP via adenylate
cyclase". The arguments are wrong and so is the type; no cheap deterministic
signal separates it from a real conversion.

**Bullet structure survives even when bullet markers are stripped.** A `<con>`
the rewriter returned as a list arrives one item per line, complete with anaphora
("They are crucial for…") whose referent the extractor cannot see. The rewriter
owns restructuring, which is where relation recall leaks.

**Linking errors are hard to see from the graph.** Two mentions of "T3" in one
document have linked to different concepts, and "thyroid follicles" and
"follicular cells" have linked to the same one. Both are invisible in the
relations output because entries render under a single label.
