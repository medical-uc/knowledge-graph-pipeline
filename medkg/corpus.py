"""Corpora: many documents in one graph, and subsets back out of it.

Two things live here.

**Ingestion.** `discover` turns a directory into an ordered list of inputs and
`plan_artifacts` gives each document its own artifact directory, because a corpus
run is not one long run — it is N independent runs whose outputs merge at Stage 6.
Keeping each document's normalized text, checkpoints and IR separate is what
makes the expensive half resumable: one document failing Stage 3 must not cost
the Stage-2 LLM pass of the three that already finished.

**Selection.** A scope — a section, a document, a group of documents — names a
set of named graphs. `resolve_scopes` turns what a human types into those graph
URIs, and `select` copies the matching contexts out of a dataset. Extraction is
therefore set selection over contexts, not a query: the subset of a graph that
represents one section is exactly the quads that were asserted into that
section's graph, with no reachability heuristic deciding what "belongs".

Pure stdlib except `select`/`load_dataset`, which need rdflib.
"""
from __future__ import annotations

import json
import os

from . import config
from .stage1_parse import slug

MANIFEST_NAME = "corpus.json"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def discover(paths, exts=(".md", ".markdown")) -> list[str]:
    """Expand directories into their markdown files; keep explicit files as given.

    Sorted, so a corpus run is reproducible and the Nth document is the same one
    on every machine. Hidden files and the normalized artefacts of a previous run
    (`*.normalized.md`) are skipped — re-ingesting the pipeline's own output
    would assert a document twice under a name that looks legitimate.
    """
    out: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if (name.lower().endswith(exts) and not name.startswith(".")
                        and not name.endswith(".normalized.md")):
                    out.append(os.path.join(path, name))
        else:
            out.append(path)
    seen, unique = set(), []
    for p in out:
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def doc_artifacts(stem: str, root: str = None) -> dict[str, str]:
    """Per-document artifact paths: `artifacts/docs/<stem>/...`.

    A corpus run writes one set of these per document and one merged `.nq` at the
    top. Without the per-document directory the second document's `stage2.json`
    overwrites the first's, which is only noticed at the end, when the graph is
    short and there is nothing left to diagnose it with.
    """
    base = os.path.join(root or config.ARTIFACTS_DIR, "docs", stem)
    return {
        "dir": base,
        "normalized": os.path.join(base, f"{stem}.normalized.md"),
        "stage1": os.path.join(base, "stage1.json"),
        "stage2": os.path.join(base, "stage2.json"),
        "ir": os.path.join(base, "ir.json"),
    }


def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def load_manifest(path: str = None) -> dict:
    """Read a corpus manifest. Returns `{"groups": {name: [source_id, ...]}}`.

    Absent file -> no groups, which is the honest default: the corpus has no
    grouping until someone declares one. The manifest is looked for next to the
    inputs so it travels with the material it describes.
    """
    if not path or not os.path.exists(path):
        return {"groups": {}}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    groups = {slug(k): list(v) for k, v in (data.get("groups") or {}).items()}
    return {"groups": groups}


def manifest_path_for(inputs, explicit: str = None) -> str:
    """`--groups` if given, else `corpus.json` beside the first input."""
    if explicit:
        return explicit
    for path in inputs:
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        candidate = os.path.join(directory or ".", MANIFEST_NAME)
        if os.path.exists(candidate):
            return candidate
    return ""


def resolve_group_members(groups: dict, known: dict) -> tuple[dict, list[str]]:
    """Map declared member names onto real source_ids; report the ones that miss.

    A group naming a document that is not in the corpus is reported rather than
    dropped: the usual cause is a manifest written against input FILENAMES while
    `source_id` comes from `<src>source:`, and silently producing an empty group
    would look exactly like a group whose documents had nothing to say.
    """
    resolved: dict[str, list[str]] = {}
    unknown: list[str] = []
    for name, members in groups.items():
        ids = []
        for member in members:
            sid = known.get(member) or known.get(stem_of(member))
            if sid is None:
                unknown.append(f"{name}: {member}")
            else:
                ids.append(sid)
        resolved[name] = ids
    return resolved, unknown


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

class Catalog:
    """What the graph says about its own structure, read back from the catalog.

    Built from `urn:graph:catalog`, so the file itself is what a subset is
    selected from — not a side-car index that can drift from the graph it
    describes.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}          # graph uri -> {label, id}
        self.sections: dict[str, dict] = {}           # graph uri -> {label, id, doc, path, order}
        self.groups: dict[str, list[str]] = {}        # group name -> [doc graph uri]

    # -- construction ----------------------------------------------------
    @classmethod
    def from_graph(cls, graph) -> "Catalog":
        from rdflib import URIRef
        cat = cls()

        def objects(s, p):
            return [str(o) for o in graph.objects(URIRef(s), URIRef(p))]

        def one(s, p, default=""):
            vals = objects(s, p)
            return vals[0] if vals else default

        for s in graph.subjects(URIRef(config.RDF_TYPE), URIRef(config.DOCUMENT_CLASS)):
            uri = str(s)
            cat.documents[uri] = {
                "label": one(uri, config.RDFS_LABEL),
                "id": one(uri, config.DCTERMS + "identifier"),
            }
        for s in graph.subjects(URIRef(config.RDF_TYPE), URIRef(config.SECTION_CLASS)):
            uri = str(s)
            order = one(uri, config.SECTION_ORDER, "0")
            cat.sections[uri] = {
                "label": one(uri, config.RDFS_LABEL),
                "id": one(uri, config.DCTERMS + "identifier"),
                "doc": one(uri, config.IN_DOCUMENT),
                "path": one(uri, config.SECTION_PATH),
                "order": int(order) if str(order).isdigit() else 0,
            }
        for s in graph.subjects(URIRef(config.RDF_TYPE), URIRef(config.GROUP_CLASS)):
            uri = str(s)
            name = one(uri, config.RDFS_LABEL) or uri[len(config.GROUP):]
            cat.groups[name] = sorted(objects(uri, config.HAS_MEMBER))
        return cat

    @classmethod
    def from_dataset(cls, dataset) -> "Catalog":
        from rdflib import Graph, URIRef
        g = Graph()
        for triple in dataset.graph(URIRef(config.GRAPH_CATALOG)):
            g.add(triple)
        return cls.from_graph(g)

    @classmethod
    def from_nquads(cls, path: str) -> "Catalog":
        return cls.from_dataset(load_dataset(path))

    # -- reporting -------------------------------------------------------
    def format(self) -> str:
        """What a reader has to see before they can name a scope."""
        if not self.documents:
            return ("This graph has no catalog. It was built before section "
                    "scoping existed, or by a single-document run. Rebuild it "
                    "with:\n  python run.py --mode corpus --input inputs/ "
                    "--index-dir ./umls_index")
        lines = ["documents"]
        for uri, doc in sorted(self.documents.items(), key=lambda kv: kv[1]["id"]):
            secs = self.sections_of(uri)
            lines.append(f"  {doc['id']:34s} {len(secs):3d} sections  {doc['label']}")
            for suri in secs:
                sec = self.sections[suri]
                lines.append(f"      {sec['id']:8s} {sec['label']}")
        lines.append("")
        lines.append("groups")
        if self.groups:
            for name, members in sorted(self.groups.items()):
                ids = [self.documents.get(m, {}).get("id", m) for m in members]
                lines.append(f"  {name:34s} {', '.join(ids)}")
        else:
            lines.append("  (none declared -- see --groups / corpus.json)")
        return "\n".join(lines)

    # -- lookup ----------------------------------------------------------
    def sections_of(self, doc_uri: str) -> list[str]:
        return sorted((uri for uri, s in self.sections.items() if s["doc"] == doc_uri),
                      key=lambda u: self.sections[u]["order"])

    def find_document(self, term: str) -> list[str]:
        """Match a document by source_id, graph URI, or a substring of its title."""
        term_l = term.lower()
        exact = [uri for uri, d in self.documents.items()
                 if term_l in (d["id"].lower(), uri.lower(),
                               config.GRAPH + d["id"].lower())]
        if exact:
            return exact
        return sorted(uri for uri, d in self.documents.items()
                      if term_l in d["label"].lower() or term_l in d["id"].lower())

    def find_section(self, term: str, docs: list[str] = None) -> list[str]:
        """Match a section by graph URI, `<doc>/<sec id>`, bare id, or heading text.

        Heading matching is what makes this usable: nobody remembers that the
        parathyroid chapter's regulation section is `s3`, and a scope you have to
        look up defeats the purpose of being able to ask for one.
        """
        term_l = term.lower()
        pool = {uri: s for uri, s in self.sections.items()
                if not docs or s["doc"] in docs}
        exact = [uri for uri in pool if uri.lower() == term_l
                 or uri.lower().endswith("/" + term_l)
                 or uri.lower() == (config.GRAPH + term_l)]
        if exact:
            return exact
        by_id = [uri for uri, s in pool.items() if s["id"].lower() == term_l]
        if by_id:
            return by_id
        return sorted((uri for uri, s in pool.items()
                       if term_l in s["label"].lower() or term_l in s["path"].lower()),
                      key=lambda u: self.sections[u]["order"])

    def find_group(self, term: str) -> list[str]:
        name = slug(term)
        members = self.groups.get(name) or self.groups.get(term)
        return list(members or [])


def graphs_for_document(doc_uri: str, catalog: Catalog = None,
                        all_graphs=()) -> list[str]:
    """Every graph belonging to a document: the document graph and its sections.

    Falls back to prefix matching over the graphs actually present when there is
    no catalog, so a graph built before the catalog existed can still be sliced
    by document.
    """
    found = {doc_uri}
    if catalog is not None:
        found |= set(catalog.sections_of(doc_uri))
    prefix = doc_uri + "/"
    found |= {g for g in all_graphs if g.startswith(prefix)}
    return sorted(found)


def resolve_scopes(catalog: Catalog, *, documents=(), sections=(), groups=(),
                   all_graphs=()) -> tuple[list[str], list[str]]:
    """Turn typed scopes into named-graph URIs. Returns (graphs, problems).

    Unmatched scopes are returned rather than raised so a caller can report all
    of them at once — an extraction that silently produced a smaller subset than
    asked for is the failure mode worth spending an error message on.
    """
    problems: list[str] = []
    doc_uris: list[str] = []

    for term in groups:
        members = catalog.find_group(term)
        if not members:
            problems.append(f"no such group: {term!r}")
        doc_uris += members
    for term in documents:
        hits = catalog.find_document(term)
        if not hits:
            problems.append(f"no document matches {term!r}")
        doc_uris += hits

    selected: set[str] = set()
    if sections:
        # A --section alongside --doc/--group NARROWS rather than adds: "that
        # section of that document" is what it reads as in English, and it is
        # the only way a heading as common as "Introduction" can be addressed.
        for term in sections:
            hits = catalog.find_section(term, docs=doc_uris or None)
            if not hits:
                where = f" in {', '.join(sorted(set(doc_uris)))}" if doc_uris else ""
                problems.append(f"no section matches {term!r}{where}")
                continue
            selected.update(hits)
    else:
        for doc_uri in doc_uris:
            selected.update(graphs_for_document(doc_uri, catalog, all_graphs))

    return sorted(selected), problems


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    from rdflib import Dataset
    ds = Dataset()
    ds.parse(path, format="nquads")
    return ds


def graphs_of(dataset):
    """Every named graph, across rdflib versions.

    `Dataset.contexts` is deprecated in favour of `Dataset.graphs` in rdflib 7.6
    but is the only spelling in the versions before it, and the deprecation
    warning fires straight into the console output of any script that calls it.
    """
    getter = getattr(dataset, "graphs", None) or dataset.contexts
    return list(getter())


def graph_uris(dataset) -> list[str]:
    return sorted(str(ctx.identifier) for ctx in graphs_of(dataset))


def select(dataset, graphs, include_inferred: bool = False,
           carry_corpus_graphs: bool = True):
    """Copy the selected named graphs into a new Dataset.

    The corpus graphs come along by default and are the reason a subset is
    readable on its own: labels, or every answer is a bare SNOMED URI; the
    catalog, or the file cannot say which section it is; provenance, or a
    subset of a traceable graph is an untraceable one. Only the provenance and
    catalog rows of the documents actually selected are carried, so a
    single-section extract does not ship a description of the whole corpus.
    """
    from rdflib import Dataset, URIRef

    wanted = set(graphs)
    out = Dataset()
    for ctx in graphs_of(dataset):
        cid = str(ctx.identifier)
        if cid in wanted:
            target = out.graph(URIRef(cid))
            for triple in ctx:
                target.add(triple)

    if include_inferred:
        for triple in dataset.graph(URIRef(config.GRAPH_INFERRED)):
            out.graph(URIRef(config.GRAPH_INFERRED)).add(triple)

    if carry_corpus_graphs:
        subjects = {URIRef(g) for g in wanted}
        # A section's provenance hangs off its DOCUMENT, so the document graph
        # URI has to be in the set even when only the section was selected.
        for g in list(wanted):
            head = g.rsplit("/", 1)[0] if g.startswith(config.GRAPH) else g
            subjects.add(URIRef(head))
        for meta in (config.GRAPH_PROVENANCE, config.GRAPH_CATALOG):
            target = out.graph(URIRef(meta))
            for s, p, o in dataset.graph(URIRef(meta)):
                if s in subjects or o in subjects:
                    target.add((s, p, o))
        # Labels are kept only for terms the subset actually mentions; a
        # corpus-wide label dump would be most of the file.
        mentioned = {t for ctx in graphs_of(out) for triple in ctx for t in triple}
        target = out.graph(URIRef(config.GRAPH_LABELS))
        for s, p, o in dataset.graph(URIRef(config.GRAPH_LABELS)):
            if s in mentioned:
                target.add((s, p, o))
    return out


def describe(dataset) -> list[tuple[str, int]]:
    """(graph uri, triple count), largest first — what a subset actually got.

    Empty graphs are omitted, including rdflib's default context, which every
    Dataset carries whether or not anything was put in it.
    """
    rows = [(str(ctx.identifier), len(list(ctx))) for ctx in graphs_of(dataset)]
    return sorted(((uri, n) for uri, n in rows if n), key=lambda r: (-r[1], r[0]))
