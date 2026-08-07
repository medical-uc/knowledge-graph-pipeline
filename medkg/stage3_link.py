"""Stage 3 — entity linking with SapBERT + FAISS over UMLS MRCONSO.

Requires a UMLS license. Two phases:
  1. build_index(...)  — offline, once: parse MRCONSO, embed every English atom
     string with SapBERT, build a cosine FAISS index, and record CUI->codes.
  2. SapBertLinker      — per-mention candidate generation + (optional) rerank.

Heavy libs (torch, transformers, faiss) are imported lazily so `parse_mrconso`
and the URI logic can be tested with only the standard library.

Native threading note (macOS especially)
----------------------------------------
torch, thinc/blis (spaCy's backend) and faiss each bring their own threading
runtime, and two of them building pools in one process is a known cause of a
bare `Segmentation fault: 11`. The primary guard is `medkg.native_env`, which
pins thread counts at package import — before any of them load.
`configure_native_threads` below is the runtime follow-up for the libraries
that expose an API, and cannot substitute for the env-level pinning.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator, Optional

import numpy as np

from .ir import Document
from . import config
from . import console

# Env-level thread pinning happens in `medkg.native_env`, at package import --
# it has to run before these libraries load, which is too early for this module.
# What follows is the runtime belt-and-braces for the libraries that expose an
# API for it.
NATIVE_THREADS = int(os.environ.get("MEDKG_NATIVE_THREADS", "1"))


def configure_native_threads(faiss=None, torch=None, threads: int = None) -> int:
    """Pin faiss's and torch's OpenMP pools to the same small thread count.

    Two runtimes each spinning up a pool sized to the machine is what tips the
    conflict over on macOS. One thread each is slower but stable; raise
    `MEDKG_NATIVE_THREADS` once the process is known to survive — index building
    in particular benefits from more.
    """
    n = NATIVE_THREADS if threads is None else threads
    if n <= 0:
        return 0
    for lib, setter in ((faiss, "omp_set_num_threads"), (torch, "set_num_threads")):
        fn = getattr(lib, setter, None)
        if fn is not None:
            try:
                fn(n)
            except Exception:                    # noqa: BLE001 - never fatal
                pass
    return n


# MRCONSO.RRF column order (0-indexed) per UMLS spec.
_CUI, _LAT, _SAB, _CODE, _STR, _SUPPRESS = 0, 1, 11, 13, 14, 16


def parse_mrconso(path: str, langs=("ENG",)) -> Iterator[tuple[str, str, str, str]]:
    """Yield (cui, sab, code, string) for non-suppressed atoms in `langs`.

    Pure stdlib — unit-testable on a small synthetic RRF snippet.
    """
    langset = set(langs)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) <= _SUPPRESS:
                continue
            if cols[_LAT] not in langset:
                continue
            if cols[_SUPPRESS] in {"O", "E", "Y"}:  # suppressed atoms
                continue
            yield cols[_CUI], cols[_SAB], cols[_CODE], cols[_STR]


def cui_to_uri(cui: str, codes: dict[str, dict[str, str]]) -> str:
    """Map a CUI to an ontology URI. RxNorm is preferred (only drug concepts
    carry RxNorm codes, so this routes drugs -> RxNorm), then SNOMED CT for
    everything else, else fall back to the UMLS CUI URI."""
    entry = codes.get(cui, {})
    if "RXNORM" in entry:
        return config.RXNORM + entry["RXNORM"]
    if "SNOMEDCT_US" in entry:
        return config.SNOMED + entry["SNOMEDCT_US"]
    return config.UMLS + cui


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def _embed(model, tokenizer, texts: list[str], device: str, batch: int = 256) -> np.ndarray:
    import torch
    out = []
    for i in range(0, len(texts), batch):
        toks = tokenizer(texts[i:i + batch], padding=True, truncation=True,
                         max_length=32, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**toks).last_hidden_state[:, 0, :]  # [CLS] — SapBERT convention
        out.append(emb.cpu().numpy())
    vecs = np.vstack(out).astype("float32")
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)  # normalize -> cosine via IP
    return vecs


def plan_atoms(rows, keep_sabs, embed_sabs=None, max_atoms=None):
    """Pure, stdlib-only selection core of the index build (unit-testable).

    Walks (cui, sab, code, string) rows and returns:
      to_embed : list[(cui, string)]  atoms to embed into FAISS
      codes    : cui -> {SAB: code}   crosswalk (built from keep_sabs, for ALL
                                       rows seen, regardless of the embed filter)
      names    : cui -> representative name (first embedded atom per cui)

    embed_sabs=None embeds every English atom; a set restricts embedding to those
    source vocabularies (the main lever to shrink the index). max_atoms caps the
    number of EMBEDDED atoms (so smoke tests embed exactly that many).

    Reports every millionth row read: MRCONSO runs to tens of millions of lines
    and the scan is minutes of otherwise silent work before the first embedding.
    """
    to_embed: list[tuple[str, str]] = []
    codes: dict[str, dict[str, str]] = {}
    names: dict[str, str] = {}
    for rows_read, (cui, sab, code, string) in enumerate(rows, start=1):
        if not rows_read % 1_000_000:
            console.announce_detail(
                f"{rows_read // 1_000_000}M atom rows read, "
                f"{len(to_embed):,} selected")
        if sab in keep_sabs:
            codes.setdefault(cui, {})[sab] = code
        if embed_sabs is not None and sab not in embed_sabs:
            continue
        names.setdefault(cui, string)
        to_embed.append((cui, string))
        if max_atoms and len(to_embed) >= max_atoms:
            break
    return to_embed, codes, names


def build_index(mrconso_path: str, out_dir: str,
                model_name: str = config.SAPBERT_MODEL,
                keep_sabs=("SNOMEDCT_US", "RXNORM"), embed_sabs=None,
                max_atoms: Optional[int] = None, batch: int = 256,
                progress: bool = True) -> None:
    """Embed MRCONSO's atoms and write the four files `SapBertLinker` reads.

    `mrconso_path` is the UMLS MRCONSO.RRF; `out_dir` receives umls.faiss,
    cuis.npy, codes.json and names.json, and is created if missing. `keep_sabs`
    names the vocabularies whose codes go into the CUI crosswalk, `embed_sabs`
    (None for every English atom) the ones whose surface forms are embedded,
    and `max_atoms` caps the embedded count for a smoke test. `batch` is the
    embedding batch size, and `progress` toggles the step-by-step commentary
    (the closing summary is printed either way).

    Raises ValueError if the filters selected no atoms at all. Runs for hours
    on a full release, so it is deliberately noisy.
    """
    import faiss
    import torch
    from transformers import AutoTokenizer, AutoModel
    configure_native_threads(faiss, torch)

    was_quiet = console.set_quiet(not progress)
    try:
        os.makedirs(out_dir, exist_ok=True)
        console.announce_step(f"reading atoms from {mrconso_path}")
        to_embed, codes, names = plan_atoms(
            parse_mrconso(mrconso_path), set(keep_sabs),
            set(embed_sabs) if embed_sabs else None, max_atoms)
        console.announce_detail(
            f"{len(to_embed):,} atom(s) to embed, "
            f"{len(codes):,} concept(s) with a crosswalk code")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.announce_step(f"loading SapBERT ({model_name}) on {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        index = None
        cuis: list[str] = []
        starts = list(range(0, len(to_embed), batch))
        console.announce_step(
            f"embedding {len(to_embed):,} atom(s) in {len(starts):,} batch(es) "
            f"of {batch}")
        # stream: embed a batch, add it to the index, repeat
        for i in console.with_progress(starts, "batches embedded"):
            chunk = to_embed[i:i + batch]
            vecs = _embed(model, tokenizer, [s for _, s in chunk], device)
            if index is None:
                index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            cuis.extend(c for c, _ in chunk)

        if index is None:
            raise ValueError("no atoms selected — check --sabs / --max-atoms")

        console.announce_step(f"writing the index to {out_dir}")
        faiss.write_index(index, os.path.join(out_dir, "umls.faiss"))
        np.save(os.path.join(out_dir, "cuis.npy"), np.array(cuis))
        with open(os.path.join(out_dir, "codes.json"), "w") as fh:
            json.dump(codes, fh)
        with open(os.path.join(out_dir, "names.json"), "w") as fh:
            json.dump(names, fh)
        console.announce_detail(
            "umls.faiss, cuis.npy, codes.json, names.json")
    finally:
        console.set_quiet(was_quiet)
    console.announce_step(
        f"indexed {len(cuis):,} atoms, {len(codes):,} CUIs with codes "
        f"-> {out_dir}")


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------

class SapBertLinker:
    """A loaded UMLS index plus the SapBERT encoder that queries it."""

    def __init__(self, index_dir: str, model_name: str = config.SAPBERT_MODEL,
                 threads: int = None):
        """Load the FAISS index, the CUI crosswalk and the encoder.

        `index_dir` is a directory `build_index` wrote; `model_name` must be
        the model it embedded with, or the vectors will not be comparable.
        `threads` overrides the native thread pinning. Loading takes tens of
        seconds and allocates the whole index, so it is reported step by step
        and a linker is worth building once per run rather than per document.
        """
        import faiss
        import torch
        from transformers import AutoTokenizer, AutoModel
        configure_native_threads(faiss, torch, threads)

        console.announce_step(f"loading the UMLS index from {index_dir}")
        self.index = faiss.read_index(os.path.join(index_dir, "umls.faiss"))
        self.cuis = np.load(os.path.join(index_dir, "cuis.npy"))
        with open(os.path.join(index_dir, "codes.json")) as fh:
            self.codes = json.load(fh)
        names_path = os.path.join(index_dir, "names.json")
        self.names = json.load(open(names_path)) if os.path.exists(names_path) else {}
        console.announce_detail(
            f"{len(self.cuis):,} embedded atom(s), "
            f"{len(self.codes):,} concept(s) with a crosswalk code")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        console.announce_step(
            f"loading SapBERT ({model_name}) on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def candidates(self, mention: str, k: int = 20) -> list[tuple[str, float]]:
        vec = _embed(self.model, self.tokenizer, [mention], self.device)
        scores, idxs = self.index.search(vec, k)
        # Collapse duplicate CUIs, keeping the best score per CUI.
        best: dict[str, float] = {}
        for score, idx in zip(scores[0], idxs[0]):
            cui = str(self.cuis[idx])
            if cui not in best or score > best[cui]:
                best[cui] = float(score)
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def needs_rerank(self, cands, margin: float = None) -> bool:
        """Is cosine actually undecided here?

        Reranking every mention meant one 72B call per span -- 539 of them on a
        textbook chapter, most asking a large model to confirm a decision the
        embedding had already made unambiguously. When the top candidate leads
        by a clear margin the reranker only agrees, slowly. Ambiguity is where
        it earns its cost.
        """
        m = config.RERANK_MARGIN if margin is None else margin
        if len(cands) < 2:
            return False
        return (cands[0][1] - cands[1][1]) < m

    def link(self, mention: str, context: Optional[str] = None,
             rerank=None) -> Optional[tuple[str, str, float]]:
        """Return (cui, uri, score) for the best candidate above the floor.
        `rerank(mention, context, candidates) -> cui` optionally reorders using
        context (e.g. an LLM/cross-encoder); candidates are (cui, name, score)
        so the reranker can disambiguate on names. By default we take top-1 by
        cosine similarity, and the reranker is consulted only when the top two
        are close enough for the choice to be in doubt.
        """
        cands = self.candidates(mention)
        if not cands:
            return None
        if rerank is not None and context and self.needs_rerank(cands):
            detailed = [(cui, self.names.get(cui, cui), score) for cui, score in cands]
            chosen = rerank(mention, context, detailed)
            score = dict(cands).get(chosen, 0.0)
            cui = chosen
        else:
            cui, score = cands[0]
        if score < config.LINK_CONFIDENCE_FLOOR:
            return None
        return cui, cui_to_uri(cui, self.codes), score


def link_document(doc: Document, linker: SapBertLinker, rerank=None) -> Document:
    """Link every span. Identical mentions are resolved once and reused: a
    textbook chapter repeats "thyroid" and "T3" dozens of times, and both the
    FAISS search and any rerank call would otherwise be repeated verbatim.

    Reports the cache hit rate alongside the count, because it is what says
    whether a slow link pass is doing real work or re-asking the reranker the
    same question. The Stage 3 header is left to the caller, which owns the
    linker construction that precedes this."""
    cache: dict[tuple, Optional[tuple]] = {}
    console.announce_step(
        f"linking {console.format_count(len(doc.spans), 'span')}"
        + (" with LLM rerank on ambiguous mentions" if rerank else
           " by top cosine (no rerank)"))
    reused, unlinked = 0, 0
    for span in console.with_progress(doc.spans, "spans linked"):
        context = doc.chunk_by_id(span.chunk_id).text if span.chunk_id else None
        key = (span.text.lower(), span.chunk_id)
        if key in cache:
            result = cache[key]
            reused += 1
        else:
            result = linker.link(span.text, context=context, rerank=rerank)
            cache[key] = result
        if result is None:
            unlinked += 1
            doc.needs_review.append({"stage": "link", "span_id": span.span_id, "text": span.text})
            continue
        span.cui, span.uri, span.link_score = result
        # Also try to link modifier VALUES so "severe" becomes a concept node.
        for m in span.modifiers:
            mkey = (m.text.lower(), None)
            if mkey not in cache:
                cache[mkey] = linker.link(m.text)
            r = cache[mkey]
            if r is not None:
                m.cui, m.uri, _ = r
    linked = sum(1 for span in doc.spans if span.uri)
    console.announce_detail(
        f"{linked}/{len(doc.spans)} linked, {unlinked} below the "
        f"{config.LINK_CONFIDENCE_FLOOR} confidence floor")
    console.announce_detail(
        f"{reused} mention(s) answered from cache, "
        f"{console.format_count(len(cache), 'distinct mention')} resolved")
    return doc
