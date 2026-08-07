"""Stage 5 — images, via the caption bridge (minimal build).

Stage 1 stores a figure's caption and description on the `Figure` itself rather
than as chunks, so they never reach Stage 2. This stage is what "catches" them:
it runs the figure's own text through the existing NER + linking machinery and
records the resulting concepts on the Figure as `depicts`, each carrying the
`[start, end]` span of the text it came from.

Two rules make this safe, and they are the reason figure text is not simply
merged into the ordinary pipeline:

  * **`Depiction`, not `Span`.** Concepts found in figure text never enter
    `doc.spans`, so they can never become the head or tail of a clinical
    relation, never mint a Stage-4 instance, and never get asserted as fact.
    A figure that *shows* pericarditis has not diagnosed anyone with it.
  * **Caption over description.** The caption carries the medically meaningful
    content; the visual description ("gradient of light blue", "leader line")
    is kept as a literal on the figure node for retrieval and provenance, and is
    linked only opportunistically. `CAPTION_ONLY_DEPICTS=True` (the default)
    restricts depicts edges to caption text.

BiomedCLIP embeddings and region annotations (W3C Web Annotation / IIIF) are the
next increments and are deliberately absent here; the interfaces below leave
room for them. Nothing in this module auto-asserts a model-guessed finding.

`build_depicts_quads` is pure stdlib so Stage 6's output stays offline-testable.
"""
from __future__ import annotations

import time
from typing import Optional

from .ir import Document, Depiction, Figure
from . import config
from . import console

CAPTION_ONLY_DEPICTS = True


# ---------------------------------------------------------------------------
# Gathering the text a figure owns
# ---------------------------------------------------------------------------

def figure_texts(doc: Document, fig: Figure) -> dict[str, str]:
    """The text bound to one figure, split by role. Pure stdlib."""
    return {"caption": fig.caption or "", "description": fig.description or ""}


def bridge_text(doc: Document, fig: Figure, caption_only: bool = None) -> str:
    only = CAPTION_ONLY_DEPICTS if caption_only is None else caption_only
    t = figure_texts(doc, fig)
    return t["caption"] if only else (t["caption"] + " " + t["description"]).strip()


# ---------------------------------------------------------------------------
# The bridge itself
# ---------------------------------------------------------------------------

def bridge_figures(doc: Document, nlp=None, linker=None, rerank=None,
                   caption_only: bool = None) -> Document:
    """Run each figure's text through NER + linking; record `Figure.depicts`.

    `nlp` and `linker` are injected exactly as elsewhere (Stage 2's scispaCy
    pipeline and Stage 3's `SapBertLinker`), so this module imports nothing
    heavy at module scope. Unlinkable mentions go to `needs_review`, never to
    the graph.

    Uses ALL configured NER models, not just the first. Captions are unusually
    anatomy-dense — "thyroid cartilage", "follicular cells", "isthmus" — and
    `en_ner_bc5cdr_md` recognises none of those, so a single-model bridge finds
    almost nothing in exactly the text figures are made of.
    """
    from .stage2_extract import load_pipelines

    started_at = time.time()
    console.announce_stage(5, "images",
                           "figure captions -> depicts edges")
    if not doc.figures:
        console.announce_step("no figures in this document; nothing to bridge")
        return doc
    pipelines = [("", nlp)] if nlp is not None else load_pipelines()

    only = CAPTION_ONLY_DEPICTS if caption_only is None else caption_only
    console.announce_step(
        f"bridging {console.format_count(len(doc.figures), 'figure')} "
        f"({'captions only' if only else 'captions and descriptions'})")
    empty, depicted, unlinked = 0, 0, 0
    for fig in console.with_progress(doc.figures, "figures bridged"):
        text = bridge_text(doc, fig, caption_only)
        if not text.strip():
            empty += 1
            continue
        seen: set[str] = set()
        ents = [e for _, pipe in pipelines for e in pipe(text).ents]
        for ent in ents:
            mention = ent.text.strip()
            key = mention.lower()
            if not mention or key in seen:
                continue
            seen.add(key)
            if getattr(ent._, "negex", False):
                continue                     # "no evidence of X" in a caption depicts nothing
            dep = Depiction(text=mention, source_span=(
                fig.caption_span if only else (fig.caption_span or fig.description_span)))
            if linker is not None:
                result = linker.link(mention, context=text, rerank=rerank)
                if result is None:
                    unlinked += 1
                    doc.needs_review.append({"stage": "stage5-link", "fig_id": fig.fig_id,
                                             "text": mention})
                    continue
                dep.cui, dep.uri, dep.score = result
            fig.depicts.append(dep)
            depicted += 1
    console.announce_detail(
        f"{console.format_count(depicted, 'concept')} depicted, "
        f"{unlinked} mention(s) unlinked, {empty} figure(s) without text")
    console.announce_finished("stage 5", started_at)
    return doc


# ---------------------------------------------------------------------------
# Quads (pure) — consumed by Stage 6
# ---------------------------------------------------------------------------

def figure_uri(fig_id: str) -> str:
    return config.INST + "fig/" + fig_id


def build_depicts_quads(doc: Document, graph: str, figures=None) -> list[tuple]:
    """(s, p, o, g) for every figure: the node, its caption/description literals,
    its source chunk, and one `ont:depicts` edge per linked concept. Pure stdlib
    — mirrors `stage6_assert.build_quads`'s contract, `Lit` marks literals.

    `figures` narrows the set, so Stage 6 can call this once per section graph
    rather than dumping every figure in the document into one of them.
    """
    from .stage6_assert import Lit

    quads: list[tuple] = []
    for fig in (doc.figures if figures is None else figures):
        node = figure_uri(fig.fig_id)
        quads.append((node, config.RDF_TYPE, config.FIGURE_IMAGE, graph))
        if fig.image_path:
            quads.append((node, config.DCTERMS + "source", Lit(fig.image_path), graph))
        if fig.caption:
            quads.append((node, config.CAPTION, Lit(fig.caption), graph))
        if fig.description:
            # kept as a literal, not linked: it describes the rendering, not the medicine
            quads.append((node, config.VISUAL_DESCRIPTION, Lit(fig.description), graph))
        if fig.referenced_from:
            quads.append((node, config.FIGURE_REF, Lit(fig.referenced_from), graph))
        for dep in fig.depicts:
            if dep.uri:
                quads.append((node, config.DEPICTS, dep.uri, graph))
    return quads


def run(doc: Document, **kw) -> Document:
    return bridge_figures(doc, **kw)
