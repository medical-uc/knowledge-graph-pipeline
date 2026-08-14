"""Stage 6 — assert the enriched Document into RDF.

Design: `build_quads` is PURE PYTHON and returns (quads, annotations) so the
triple-generation logic is unit-testable with only stdlib. `to_dataset` is a
thin rdflib adapter (lazy import) that loads them into a named-graph Dataset,
attaching per-triple confidence via RDF-star, with rdf:Statement reification as
a portable fallback.

Key properties enforced here:
  * one named graph per (source-version, SECTION); provenance asserted ONCE per
    source on the document graph, never per triple
  * deterministic instance/assertion URIs -> idempotent re-runs
  * negated/uncertain relations become reified assertion nodes, never dropped,
    never asserted as fact
  * concept labels are corpus-level, not per document

The graph granularity is what makes a subset extractable. Provenance was always
carried by the graph rather than the triple, so the way to make "the knowledge in
this section" a first-class thing is to make the section a graph:

    urn:graph:thyroid_source            the document (its provenance, and any
                                        content outside every <sec>)
    urn:graph:thyroid_source/s3         one section of it

A document is then the set of graphs under its prefix, a group is the union over
its members' documents, and a section is a single graph -- all three are set
selection over contexts, with no query needed and no triple left behind. The
alternative, stamping a section onto every triple through its reification node,
makes extraction a CONSTRUCT over a graph that has to be fully loaded first, and
leaves unreified material (labels, figures, instances) with no section at all.
"""
from __future__ import annotations

import time
from collections import namedtuple
from typing import Optional

from .ir import Document
from . import config
from . import console
from . import stage5_images
from . import stage4_postcoord
from .stage4_postcoord import det_id

Lit = namedtuple("Lit", ["value"])            # marks an object literal
# quad:       (s, p, o, graph)          o may be str-uri or Lit
# annotation: ((s, p, o), {prop: val}, graph)


def _cap(polarity: str) -> str:
    return {"negated": "Negated", "uncertain": "Uncertain", "affirmed": "Affirmed"}.get(
        polarity, polarity.title())


def document_graph(source_id: str) -> str:
    return config.GRAPH + source_id


def graph_for(source_id: str, section_id: str = "") -> str:
    """The named graph a section's assertions belong in.

    Content that sits outside every `<sec>` -- the closing `<sum>`, a stray
    passage -- has no section and stays on the document graph. It is still
    reachable from a document-scoped subset (which takes the document graph and
    everything under its prefix), just not from a section-scoped one, which is
    the honest answer: it was not in a section.
    """
    base = document_graph(source_id)
    return f"{base}/{section_id}" if section_id else base


def chunk_uri(source_id: str, chunk_id: str) -> str:
    """The node standing for one chunk of normalized text.

    Scoped by `source_id` because chunk ids restart at `c001` in every document,
    so the bare id collides the moment two chapters share a store.
    """
    return f"{config.INST}chunk/{source_id}/{chunk_id}"


def table_uri(source_id: str, table_id: str) -> str:
    """The node standing for a `<tbl>` block, scoped the same way as a chunk."""
    return f"{config.INST}tbl/{source_id}/{table_id}"


def build_table_quads(doc: Document, graph: str, tables=None) -> list[tuple]:
    """(s, p, o, g) for every table: the node, its label and caption, and edges
    to the chunk holding its restated grid and to the prose that cites it.

    The grid's sentences are not repeated here. They are an ordinary chunk and
    Stage 2 extracts relations from them like any other, so the table node's job
    is to say which chunk that is, not to carry the text a second time.

    `tables` narrows the set so Stage 6 can call this once per section graph.
    """
    sid = doc.source.source_id
    quads: list[tuple] = []
    for tbl in (doc.tables if tables is None else tables):
        node = table_uri(sid, tbl.table_id)
        quads.append((node, config.RDF_TYPE, config.TABLE_BLOCK, graph))
        if tbl.label:
            quads.append((node, config.RDFS_LABEL, Lit(tbl.label), graph))
            quads.append((node, config.TABLE_LABEL, Lit(tbl.label), graph))
        if tbl.caption:
            quads.append((node, config.CAPTION, Lit(tbl.caption), graph))
        if tbl.content_chunk:
            quads.append((node, config.TABLE_CONTENT,
                          chunk_uri(sid, tbl.content_chunk), graph))
        if tbl.referenced_from:
            quads.append((node, config.TABLE_REF,
                          chunk_uri(sid, tbl.referenced_from), graph))
    return quads


def build_chunk_quads(doc: Document, chunk_ids, graph_of_chunk) -> list[tuple]:
    """Declare the chunk nodes that figure and table edges point at.

    Only the cited chunks are minted, not every chunk in the document: a chunk
    node exists so a figure reference can be an edge into the text rather than
    an id stranded in a literal, and one that nothing points at would be a node
    with no incoming edge and no query that reaches it. Each carries its kind
    and offsets into `Document.normalized_path`, enough to fetch the text.

    `chunk_ids` is the set to declare and `graph_of_chunk` maps a chunk id to
    the named graph it belongs in.
    """
    sid = doc.source.source_id
    quads: list[tuple] = []
    for chunk in doc.chunks:
        if chunk.chunk_id not in chunk_ids:
            continue
        node = chunk_uri(sid, chunk.chunk_id)
        g = graph_of_chunk(chunk.chunk_id)
        quads += [
            (node, config.RDF_TYPE, config.CHUNK_NODE, g),
            (node, config.CHUNK_KIND, Lit(chunk.kind), g),
            (node, config.CHAR_START, Lit(chunk.char_start), g),
            (node, config.CHAR_END, Lit(chunk.char_end), g),
        ]
    return quads


def build_quads(doc: Document, snomed_version: str = config.SNOMED_VERSION):
    """Return (quads, annotations) for ONE document. Pure stdlib.

    Concept labels are NOT emitted here -- they are corpus-level and come from
    `build_corpus_quads`, which sees every document and can pick one label per
    concept. See `assert_documents`.
    """
    quads: list[tuple] = []
    annotations: list[tuple] = []
    doc_g = document_graph(doc.source.source_id)
    pg = config.GRAPH_PROVENANCE
    spans = {s.span_id: s for s in doc.spans}
    section_of_chunk = {c.chunk_id: c.section_id for c in doc.chunks}

    def graph_of_span(span) -> str:
        """The section graph a span's assertions belong in."""
        if span is None:
            return doc_g
        return graph_for(doc.source.source_id,
                         section_of_chunk.get(span.chunk_id, ""))
    # Endpoints whose meaning is carried by a post-coordinated instance rather
    # than the bare concept -- "iodine [deficiency]", not "iodine". Pointing a
    # caused_by tail at the concept would assert that iodine causes goiter.
    postcoord = stage4_postcoord.instance_by_span(doc)

    def endpoint(span):
        """The URI a relation should actually attach to for this span."""
        return postcoord.get(span.span_id, span.uri)

    # --- relations ---------------------------------------------------------
    for rel in doc.relations:
        head = spans.get(rel.head)
        tail = spans.get(rel.tail) if rel.tail else None
        if head is None or head.uri is None:
            continue
        pred = config.RELATION_ONTOLOGY[rel.type]["uri"]
        # The head's section owns the relation. A relation is extracted from one
        # chunk, so its head and tail are almost always in the same section; when
        # a cross-chunk dedupe has moved them apart, the head is the endpoint the
        # assertion is *about*, so it decides.
        g = graph_of_span(head)

        if rel.polarity == "affirmed" and tail is not None and tail.uri:
            triple = (endpoint(head), pred, endpoint(tail))
            quads.append((*triple, g))
            if rel.score is not None:
                annotations.append((triple, {
                    config.ONT + "confidence": Lit(round(rel.score, 3)),
                    config.ONT + "extractedBy": Lit("llm-re"),
                }, g))
        else:
            # negated / uncertain (or tail-less) -> reified assertion node
            stmt = config.INST + "assert/" + det_id(
                doc.source.source_id, rel.rel_id, head.uri, rel.type)
            quads += [
                (stmt, config.RDF_TYPE, config.ONT + "ClinicalAssertion", g),
                (stmt, config.ONT + "about", endpoint(head), g),
                (stmt, config.ONT + "assertionType", config.RELATION_ONTOLOGY[rel.type]["uri"], g),
                (stmt, config.ONT + "polarity", config.ONT + _cap(rel.polarity), g),
            ]
            if tail is not None and tail.uri:
                quads.append((stmt, config.ONT + "target", endpoint(tail), g))

    # --- instance nodes (Stage 4) -----------------------------------------
    for inst in doc.instances:
        ig = graph_of_span(spans.get(inst.source_span))
        quads.append((inst.inst_id, config.RDF_TYPE, inst.type_uri, ig))
        for prop, val in inst.attributes:
            quads.append((inst.inst_id, prop, val, ig))
        # An instance used as a relation endpoint renders as a bare inst/<hash>
        # in every query answer unless it is labelled. `relations` COALESCEs
        # rdfs:label, so this is what makes the repaired triple read as
        # "goiter caused_by iodine [deficiency]".
        if inst.label:
            quads.append((inst.inst_id, config.RDFS_LABEL, Lit(inst.label), ig))

    # --- image nodes + Stage 5 depicts edges -------------------------------
    # The figure node is registered even if Stage 5 hasn't run; `depicts` edges
    # appear only for figures whose text was bridged and linked. Figures go to
    # the section they were declared in, so a section subset shows its own
    # figures and no others.
    by_section: dict[str, list] = {}
    for fig in doc.figures:
        by_section.setdefault(fig.section_id, []).append(fig)
    for section_id, figs in by_section.items():
        quads += stage5_images.build_depicts_quads(
            doc, graph_for(doc.source.source_id, section_id), figures=figs)

    # --- table nodes -------------------------------------------------------
    # Same treatment as figures, and for the same reason: a table declared in a
    # section belongs to that section's graph, not to the document's.
    tables_by_section: dict[str, list] = {}
    for tbl in doc.tables:
        tables_by_section.setdefault(tbl.section_id, []).append(tbl)
    for section_id, tbls in tables_by_section.items():
        quads += build_table_quads(
            doc, graph_for(doc.source.source_id, section_id), tables=tbls)

    # --- chunk nodes the figure and table edges land on --------------------
    cited_chunks = {cid for fig in doc.figures
                    for cid in (fig.referenced_from, fig.content_chunk) if cid}
    cited_chunks |= {cid for tbl in doc.tables
                     for cid in (tbl.referenced_from, tbl.content_chunk) if cid}
    quads += build_chunk_quads(
        doc, cited_chunks,
        lambda cid: graph_for(doc.source.source_id,
                              section_of_chunk.get(cid, "")))

    # --- provenance: asserted ONCE about the whole graph -------------------
    # Stage 1's input is a REWRITTEN document, not the original material, so the
    # two are asserted separately. `dcterms:source` names what the material came
    # from; `prov:wasGeneratedBy` names the model that wrote the prose the
    # triples were actually extracted from. Collapsing them would let a claim
    # traceable to fresh LLM output pass as one traceable to the textbook.
    # It stays on the DOCUMENT graph even though facts now live in the section
    # graphs beneath it: the rewriter generated a document, not a section, and
    # the catalog's isPartOf edges are what carry provenance down to a section.
    quads += [
        (doc_g, config.DCTERMS + "source",
         Lit(doc.source.origin or doc.source.title), pg),
        (doc_g, config.DCTERMS + "hasVersion", Lit(doc.source.edition), pg),
        (doc_g, config.ONT + "snomedVersion", Lit(snomed_version), pg),
    ]
    if doc.source.generator:
        agent = config.INST + "agent/" + det_id("agent", doc.source.generator)
        quads += [
            (agent, config.RDF_TYPE, config.SOFTWARE_AGENT, pg),
            (agent, config.RDFS_LABEL, Lit(doc.source.generator), pg),
            (doc_g, config.WAS_GENERATED_BY, agent, pg),
        ]
        if doc.source.origin:
            quads.append((doc_g, config.WAS_DERIVED_FROM, Lit(doc.source.origin), pg))
        if doc.source.generated_at:
            quads.append((doc_g, config.GENERATED_AT, Lit(doc.source.generated_at), pg))

    quads += build_catalog_quads(doc)
    return quads, annotations


def build_catalog_quads(doc: Document) -> list[tuple]:
    """The structure a subset is selected BY: document, its sections, their
    headings. Pure stdlib.

    This is what lets `subset.py --list-scopes` name a section in words instead
    of making you grep the .nq for graph URIs, and what lets `--section
    "Hormone Synthesis"` resolve to `urn:graph:thyroid_source/s2`.
    """
    cg = config.GRAPH_CATALOG
    doc_g = document_graph(doc.source.source_id)
    quads: list[tuple] = [
        (doc_g, config.RDF_TYPE, config.DOCUMENT_CLASS, cg),
        (doc_g, config.RDFS_LABEL, Lit(doc.source.title or doc.source.source_id), cg),
        (doc_g, config.DCTERMS + "identifier", Lit(doc.source.source_id), cg),
    ]
    for order, (section_id, section_path) in enumerate(doc.sections(), start=1):
        sg = graph_for(doc.source.source_id, section_id)
        quads += [
            (sg, config.RDF_TYPE, config.SECTION_CLASS, cg),
            (sg, config.RDFS_LABEL, Lit(section_path[-1] if section_path else section_id), cg),
            (sg, config.DCTERMS + "identifier", Lit(section_id), cg),
            (sg, config.SECTION_PATH, Lit(" > ".join(section_path)), cg),
            (sg, config.SECTION_ORDER, Lit(order), cg),
            (sg, config.IN_DOCUMENT, doc_g, cg),
            (sg, config.DCTERMS + "isPartOf", doc_g, cg),
            (doc_g, config.HAS_SECTION, sg, cg),
        ]
    return quads


def build_group_quads(groups: dict) -> list[tuple]:
    """`{"endocrine": ["thyroid_source", "anatomy_of_endocrine_glands"]}` ->
    catalog quads. Pure stdlib.

    Groups are declared, never inferred. Two documents sharing a word in their
    titles is not evidence that a reader wants them queried together, and a
    guessed grouping would be indistinguishable in the graph from one the
    curator meant.
    """
    cg = config.GRAPH_CATALOG
    quads: list[tuple] = []
    for name, members in sorted(groups.items()):
        if not members:
            # A group none of whose documents are in this corpus is not a scope
            # anyone can select; writing it into the catalog would advertise an
            # extraction that always comes back empty. The caller reports the
            # unresolved members.
            continue
        gid = config.GROUP + name
        quads += [
            (gid, config.RDF_TYPE, config.GROUP_CLASS, cg),
            (gid, config.RDFS_LABEL, Lit(name), cg),
        ]
        for source_id in members:
            quads.append((gid, config.HAS_MEMBER, document_graph(source_id), cg))
    return quads


def build_label_quads(docs) -> list[tuple]:
    """One `rdfs:label` and one `ont:mentionCount` per concept, over the WHOLE
    corpus. Pure stdlib.

    Corpus-level rather than per document because `rdfs:label` is functional in
    practice even though RDF does not enforce it: the `relations` query
    OPTIONAL-joins a label onto each endpoint, so a concept carrying three
    labels from three documents returns every relation about it three times.
    Counts are summed across documents and the most frequent surface form in the
    corpus wins, with the shorter form breaking ties.
    """
    forms: dict[str, dict[str, int]] = {}
    for doc in docs:
        for sp in doc.spans:
            if sp.uri:
                forms.setdefault(sp.uri, {})
                forms[sp.uri][sp.text] = forms[sp.uri].get(sp.text, 0) + 1
    quads: list[tuple] = []
    lg = config.GRAPH_LABELS
    for uri, counts in forms.items():
        best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        quads.append((uri, config.RDFS_LABEL, Lit(best), lg))
        quads.append((uri, config.ONT + "mentionCount", Lit(sum(counts.values())), lg))
    return quads


# ---------------------------------------------------------------------------
# rdflib adapter (lazy)
# ---------------------------------------------------------------------------

def _term(x):
    from rdflib import URIRef, Literal
    return Literal(x.value) if isinstance(x, Lit) else URIRef(x)


def looks_malformed_uri(text) -> bool:
    """Would rdflib accept this URI now and refuse to serialize it later?

    This is the exact failure this module guards against. A turtle parser
    WITHOUT RDF-star support does not reject `<< <http://x> <http://y> ... >>`.
    It reads the leading `<` as the start of a URI, consumes up to the next `>`,
    and yields the term `< <http://x` — emitting a warning, not an exception.
    The parse "succeeds", the annotation is silently corrupt, and the traceback
    only arrives in Stage 8 when the N-Quads serializer refuses it.

    Pure stdlib so it can be tested without rdflib installed.
    """
    t = str(text)
    return (not t.strip()) or any(ch in t for ch in "<> \t\n")


def _rdfstar_usable() -> bool:
    """Can this rdflib both PARSE `<< s p o >>` and SERIALIZE the result?

    Feature detection, not a version check — and it probes serialization too,
    because parsing is precisely where the silent corruption hides. Cached: the
    answer cannot change within a process.
    """
    global _RDFSTAR_OK
    if _RDFSTAR_OK is not None:
        return _RDFSTAR_OK
    import logging
    import warnings
    try:
        from rdflib import Graph
        # The probe deliberately feeds the parser something it may mangle, so
        # its complaint is the expected result, not news for the operator.
        rdflib_log = logging.getLogger("rdflib")
        prior = rdflib_log.level
        rdflib_log.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = Graph()
                g.parse(data='<< <urn:x:s> <urn:x:p> <urn:x:o> >> <urn:x:m> "v" .',
                        format="turtle")
                ok = not any(looks_malformed_uri(t) for triple in g for t in triple)
                if ok:
                    g.serialize(format="nt")  # the call that raises downstream
        finally:
            rdflib_log.setLevel(prior)
        _RDFSTAR_OK = ok
    except Exception:                         # noqa: BLE001
        _RDFSTAR_OK = False
    return _RDFSTAR_OK


_RDFSTAR_OK = None


def to_dataset(quads, annotations, use_rdfstar: bool = True):
    """Load quads/annotations into an rdflib Dataset with named graphs."""
    from rdflib import Dataset, URIRef, Literal, Namespace
    from rdflib.namespace import RDF

    console.announce_step(
        f"loading {console.format_count(len(quads), 'quad')} into an rdflib "
        f"Dataset")
    ds = Dataset()
    for s, p, o, g in console.with_progress(quads, "quads loaded"):
        ds.graph(URIRef(g)).add((_term(s), _term(p), _term(o)))

    annotation_style = ("RDF-star" if use_rdfstar and _rdfstar_usable()
                        else "rdf:Statement reification")
    counted = console.format_count(len(annotations), "confidence annotation")
    console.announce_step(f"attaching {counted} as {annotation_style}")
    for (s, p, o), meta, g in annotations:
        graph = ds.graph(URIRef(g))
        placed = False
        if use_rdfstar and _rdfstar_usable():
            try:
                from rdflib import Graph as _Graph
                lines = []
                for mp, mv in meta.items():
                    obj = f'"{mv.value}"' if isinstance(mv, Lit) else f"<{mv}>"
                    lines.append(f"<< <{s}> <{p}> <{o}> >> <{mp}> {obj} .")
                # Parse into scratch and inspect before committing. Writing
                # straight into `graph` meant a silently mis-parsed annotation
                # was already in the dataset by the time anything noticed, and
                # `placed = True` then suppressed the reification fallback that
                # would have saved it.
                scratch = _Graph()
                scratch.parse(data="\n".join(lines), format="turtle")
                if not any(looks_malformed_uri(t) for triple in scratch for t in triple):
                    for triple in scratch:
                        graph.add(triple)
                    placed = True
            except Exception:                 # noqa: BLE001
                placed = False
        if not placed:
            # Portable fallback: rdf:Statement reification. The graph is part of
            # the id: two documents that extracted the same fact must not share
            # one statement node, or the node ends up carrying both confidences
            # and neither can be traced back to the source that reported it.
            stmt = URIRef(config.INST + "stmt/" + det_id(g, s, p, o))
            graph.add((stmt, RDF.type, RDF.Statement))
            graph.add((stmt, RDF.subject, _term(s)))
            graph.add((stmt, RDF.predicate, _term(p)))
            graph.add((stmt, RDF.object, _term(o)))
            for mp, mv in meta.items():
                graph.add((stmt, URIRef(mp), _term(mv)))
    console.announce_detail(
        f"{console.format_count(len(list(ds.contexts())), 'named graph')}")
    return ds


def check_source_ids(docs) -> None:
    """Refuse a corpus in which two documents claim the same `source_id`.

    The id names the document's graph AND seeds the Stage-4 instance hashes, so
    a collision does not merely mix two documents into one graph: two spans at
    the same offset in different documents mint the SAME instance node, and the
    attributes of one land on the other. It is silent, and it is not repairable
    after the fact.

    It is a real risk rather than a theoretical one, because `source_id` comes
    from `<src>source:` — the ORIGINAL material — not from the input filename.
    Two chapters rewritten from one textbook legitimately name the same origin.
    """
    seen: dict[str, str] = {}
    for doc in docs:
        sid = doc.source.source_id
        if sid in seen:
            raise ValueError(
                f"two documents share source_id {sid!r}: {seen[sid]} and "
                f"{doc.source_path or doc.doc_id}. It names the graph and seeds "
                f"the instance hashes, so they cannot be asserted together. Set "
                f"a distinct <src>source:</src> in one of them, or pass "
                f"--source-id to override.")
        seen[sid] = doc.source_path or doc.doc_id


def assert_documents(docs, snomed_version: str = config.SNOMED_VERSION,
                     use_rdfstar: bool = True, groups: dict = None):
    """Assert a whole corpus into one named-graph Dataset.

    Documents are merged HERE, at the RDF layer, rather than by concatenating
    their IRs earlier. Chunk offsets index into a per-document normalized file
    and span/chunk ids are only unique within a document, so a merged IR would
    have to renumber both and would break the offset contract every grounding
    check depends on. Merging at the graph layer needs neither: the documents
    were already destined for different named graphs.
    """
    started_at = time.time()
    console.announce_stage(6, "assert", "IR -> RDF named graphs")
    docs = list(docs)
    console.announce_step("checking for source_id collisions")
    check_source_ids(docs)
    quads: list[tuple] = []
    annotations: list[tuple] = []
    console.announce_step(
        f"building quads for {console.format_count(len(docs), 'document')}")
    for doc in console.with_progress(docs, "documents built"):
        q, a = build_quads(doc, snomed_version)
        quads += q
        annotations += a
    console.announce_step("naming each concept once across the corpus")
    label_quads = build_label_quads(docs)
    quads += label_quads
    console.announce_detail(
        f"{console.format_count(len(label_quads) // 2, 'concept')} labelled")
    if groups:
        quads += build_group_quads(groups)
        console.announce_step("cataloguing " + console.format_count(
            len(groups), "document group"))
    dataset = to_dataset(quads, annotations, use_rdfstar=use_rdfstar)
    console.announce_finished(
        "stage 6", started_at,
        f"{console.format_count(len(quads), 'quad')}, "
        f"{console.format_count(len(annotations), 'annotation')}")
    return dataset


def assert_document(doc: Document, snomed_version: str = config.SNOMED_VERSION,
                    use_rdfstar: bool = True):
    return assert_documents([doc], snomed_version, use_rdfstar=use_rdfstar)
