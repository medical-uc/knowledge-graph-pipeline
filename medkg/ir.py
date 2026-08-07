"""Stage 0 — the intermediate representation.

Every stage reads a Document, ADDS fields, and never mutates prior ones. This
module is pure stdlib so it can be imported and round-tripped anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional


@dataclass
class Source:
    source_id: str
    title: str
    edition: str = ""
    section: str = ""
    page: Optional[int] = None
    # Stage 1's input is now a REWRITTEN document, not the original material.
    # These three keep that distinction in the IR so Stage 6 can assert the
    # original as dcterms:source and the rewrite as prov:wasGeneratedBy, rather
    # than letting model-authored prose pass as what the textbook said.
    origin: str = ""            # filename the rewriter was given
    generator: str = ""         # model that wrote the document Stage 1 parsed
    generated_at: str = ""      # when it did


@dataclass
class Modifier:
    type: str                       # "severity" | "location" | "radiation" | "acuity" | ...
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    cui: Optional[str] = None       # filled if the modifier VALUE gets linked (Stage 3/4)
    uri: Optional[str] = None


@dataclass
class Span:
    span_id: str
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    label: str                      # entity type (Disease, Drug, Symptom, ...)
    negated: bool = False           # Stage 2 (negspacy)
    modifiers: list[Modifier] = field(default_factory=list)
    cui: Optional[str] = None       # Stage 3
    uri: Optional[str] = None       # Stage 3 (ontology URI)
    link_score: Optional[float] = None
    # What NER originally called this span, kept when Stage 3 re-derived `label`
    # from the linked concept's semantic type. Empty when the two agreed or no
    # reconciliation ran. The override decides which relation types the span can
    # take part in, so the graph has to stay able to show what it overrode.
    ner_label: str = ""


@dataclass
class Relation:
    rel_id: str
    head: str                       # span_id
    tail: Optional[str]             # span_id, or None for patient-level assertions
    type: str                       # must be a key in config.RELATION_ONTOLOGY
    polarity: str = "affirmed"      # "affirmed" | "negated" | "uncertain"
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    score: Optional[float] = None


@dataclass
class Instance:
    """A minted, occurrence-specific node (Stage 4). Instantiates a concept and
    carries post-coordinated modifiers as (attribute_property_uri, value_uri)."""
    inst_id: str
    type_uri: str
    source_span: str
    attributes: list[tuple[str, str]] = field(default_factory=list)
    # Composed surface form, e.g. "iodine [deficiency]". Without it a
    # post-coordinated relation endpoint renders as a bare inst/ hash and the
    # query output is less readable than the wrong triple it replaced.
    label: Optional[str] = None


@dataclass
class Chunk:
    """A unit of ASSERTABLE text — the corpus Stage 2 extracts relations from,
    and nothing else. Figure captions and descriptions live on `Figure`;
    objectives, tables and bibliography live in `Document.passages`.

    `kind` is `prose | definition | keypoint | clinical | summary` (from the
    rewriter's `<con>`, `<def>`, `<key>`, `<clin>`, `<sum>`), or `duplicate` for
    a block that exactly restates an earlier one. `EXTRACTABLE_CHUNK_KINDS`
    decides which of those Stage 2 actually reads, so a kind can be switched off
    without re-running Stage 1.

    Offsets index into `Document.normalized_path`, never the raw source.

    `section_id` is the rewriter's own `<sec id="...">`, kept because it is the
    unit Stage 6 names a graph after: everything extracted from one section is
    asserted into `urn:graph:<source_id>/<section_id>`, which is what makes a
    section retrievable as a subset. `section_path` stays the human-readable
    heading trail; the id is what stays stable when a heading is reworded.
    """
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    section_path: list[str] = field(default_factory=list)
    kind: str = "prose"
    figure_refs: list[str] = field(default_factory=list)  # [[FIG:id]] anchors lifted out of prose
    section_id: str = ""


@dataclass
class Passage:
    """Document text that is retained but never asserted: learning objectives,
    tables, bibliography. Kept out of `chunks` so that "chunk" means exactly
    "something Stage 2 may extract from", but kept in the IR — and with offsets —
    because deleting source material to tidy a data structure is not a trade
    worth making. `kind` is `objective | table | reference`.
    """
    passage_id: str
    kind: str
    text: str
    char_start: int
    char_end: int
    section_path: list[str] = field(default_factory=list)
    section_id: str = ""


@dataclass
class Depiction:
    """A concept a figure depicts (Stage 5). Deliberately NOT a Span: figure
    text must never flow into clinical relations."""
    text: str
    cui: Optional[str] = None
    uri: Optional[str] = None
    score: Optional[float] = None
    # [start, end] of the figure text this concept came from. Captions are no
    # longer chunks, so this is how a depicts edge keeps a position.
    source_span: Optional[list[int]] = None


@dataclass
class Figure:
    fig_id: str
    image_path: str
    caption: str
    referenced_from: str = ""
    marker: str = ""                    # verbatim source marker, e.g. "[FIGURE:p2_b3]"
    description: str = ""               # <desc> text (Stage 1)
    # [start, end] into the normalized file. The caption and description are not
    # chunks -- they would be pure duplication of these two fields -- but they
    # are still written to the normalized file, so a depicts edge can point at
    # where its text sits, the same as any other assertion in the graph.
    caption_span: Optional[list[int]] = None
    description_span: Optional[list[int]] = None
    depicts: list[Depiction] = field(default_factory=list)   # Stage 5
    section_id: str = ""                # the <sec> the figure block sits in


@dataclass
class Document:
    doc_id: str
    source: Source
    chunks: list[Chunk] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    needs_review: list[dict] = field(default_factory=list)
    source_path: str = ""               # raw markdown Stage 1 was given
    normalized_path: str = ""           # intermediate all-prose markdown; chunk offsets index into THIS

    # ---- helpers -------------------------------------------------------
    def span_by_id(self, span_id: str) -> Optional[Span]:
        for s in self.spans:
            if s.span_id == span_id:
                return s
        return None

    def chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def figure_by_id(self, fig_id: str) -> Optional[Figure]:
        for f in self.figures:
            if f.fig_id == fig_id:
                return f
        return None

    def sections(self) -> list[tuple[str, list[str]]]:
        """(section_id, section_path) in document order, deduplicated.

        The catalog Stage 6 writes is built from this, and `--list-scopes`
        prints it: a section is only addressable if something in the document
        actually landed in it.
        """
        out: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        for item in list(self.chunks) + list(self.passages):
            if item.section_id and item.section_id not in seen:
                seen.add(item.section_id)
                out.append((item.section_id, list(item.section_path)))
        return out

    def chunk_section(self, chunk_id: str) -> str:
        chunk = self.chunk_by_id(chunk_id)
        return chunk.section_id if chunk is not None else ""

    def extractable_chunks(self, kinds=("prose",)) -> list[Chunk]:
        """Chunks Stage 2 may extract from. Every chunk is assertable text by
        construction; this filter exists so a kind (say `summary`) can be turned
        off without re-running Stage 1, and to drop `duplicate` blocks."""
        return [c for c in self.chunks if c.kind in kinds]

    def passages_of(self, kind: str) -> list[Passage]:
        return [p for p in self.passages if p.kind == kind]

    # ---- JSON round-trip ----------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        # Fields this IR no longer carries are dropped rather than raising. The
        # schema has moved repeatedly (chunk kinds, figure spans, passages), and
        # an old Stage-1 JSON should still load rather than crash on a key that
        # was retired -- the stage that produced it is cheap to re-run, but only
        # if you can read the file to notice.
        def build(klass, data, **extra):
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in data.items() if k in names}, **extra)

        src = build(Source, d["source"])
        chunks = [build(Chunk, c) for c in d.get("chunks", [])]
        figures = []
        for f in d.get("figures", []):
            deps = [build(Depiction, x) for x in f.get("depicts", [])]
            figures.append(build(Figure, {k: v for k, v in f.items() if k != "depicts"},
                                 depicts=deps))
        spans = []
        for s in d.get("spans", []):
            mods = [build(Modifier, m) for m in s.get("modifiers", [])]
            spans.append(build(Span, {k: v for k, v in s.items() if k != "modifiers"},
                               modifiers=mods))
        passages = [build(Passage, p) for p in d.get("passages", [])]
        relations = [build(Relation, r) for r in d.get("relations", [])]
        instances = []
        for i in d.get("instances", []):
            attrs = [tuple(a) for a in i.get("attributes", [])]
            instances.append(build(Instance, {k: v for k, v in i.items()
                                              if k != "attributes"}, attributes=attrs))
        return cls(
            doc_id=d["doc_id"], source=src, chunks=chunks, figures=figures,
            spans=spans, relations=relations, instances=instances,
            passages=passages, needs_review=d.get("needs_review", []),
            source_path=d.get("source_path", ""),
            normalized_path=d.get("normalized_path", ""),
        )

    @classmethod
    def from_json(cls, path: str) -> "Document":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
