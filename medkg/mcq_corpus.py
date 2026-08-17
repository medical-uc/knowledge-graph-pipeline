"""MCQ generation across every document a corpus run left in `artifacts`.

`medkg.mcq` builds a paper from one IR. A corpus run writes one IR per document
under `artifacts/docs/<stem>/ir.json`, so this walks that directory and drops a
`questions.json` beside each one.

Kept per-document rather than pooled into a single paper. The safety rules in
`medkg.mcq` are all document-local: `blocklist` reads the relations of the
document it was given, and a distractor that is safe against chapter 2's graph
may be the asserted answer in chapter 17. Merging the IRs first would widen
every blocklist and cost most of the distractors, and merging the papers
afterwards would ship items whose distractors were never checked against the
document they now sit next to.

One document failing does not stop the run: the failure is reported and the
remaining documents are still generated, which is the same bargain the corpus
extraction makes.
"""
from __future__ import annotations

import os
import traceback

from . import config
from . import console
from . import mcq
from .ir import Document

QUESTIONS_NAME = "questions.json"


def docs_root(artifacts_dir: str = None) -> str:
    """The directory a corpus run writes its per-document artifacts into."""
    return os.path.join(artifacts_dir or config.ARTIFACTS_DIR, "docs")


def discover_documents(artifacts_dir: str = None) -> list[tuple[str, str]]:
    """Every document with an IR on disk, as sorted `(stem, ir_path)` pairs.

    Reads the artifact tree rather than the inputs, so a document whose
    extraction has not finished is simply absent instead of failing here. A
    subdirectory without an `ir.json` is skipped for the same reason: a corpus
    run that died in Stage 2 leaves the directory and its checkpoints behind.
    """
    root = docs_root(artifacts_dir)
    if not os.path.isdir(root):
        return []
    found = []
    for stem in sorted(os.listdir(root)):
        ir_path = os.path.join(root, stem, "ir.json")
        if os.path.isfile(ir_path):
            found.append((stem, ir_path))
    return found


def questions_path(ir_path: str) -> str:
    """Where a document's paper goes: `questions.json` beside its IR."""
    return os.path.join(os.path.dirname(ir_path), QUESTIONS_NAME)


def is_current(out_path: str, ir_path: str) -> bool:
    """Whether an existing paper was generated from the IR as it stands now.

    Compares modification times, matching how `run.py --mode corpus` decides an
    IR can be resumed from. A paper older than its IR is stale by definition:
    the relations it was built from have been re-extracted since.
    """
    if not os.path.exists(out_path):
        return False
    return os.path.getmtime(out_path) >= os.path.getmtime(ir_path)


def generate_for_document(ir_path: str, out_path: str, call_llm=None,
                          limit: int = 0) -> int:
    """Build one document's paper and write it. Returns the item count.

    `call_llm` is passed through to `mcq.generate` for stem phrasing; without
    one the stems stay templated. `limit` truncates the paper after the safety
    rules have run, so it never changes which items are eligible.

    The file is written even when no item survives, because an empty paper is a
    result: the alternative leaves a stale file from an earlier run standing
    beside a graph that no longer supports it.
    """
    doc = Document.from_json(ir_path)
    console.announce_step(
        f"{len(doc.relations)} relation(s), {len(doc.spans)} span(s) read")
    items, rejected = mcq.generate(doc, call_llm)
    if limit:
        items = items[:limit]
        console.announce_step(f"keeping the first {limit} item(s)")
    if call_llm:
        rephrased = sum(1 for i in items if i.stem_source == "llm")
        console.announce_detail(
            f"stems: {rephrased} rephrased, {len(items) - rephrased} "
            f"template ({len(rejected)} rewrites rejected)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(mcq.to_json(items))
    console.announce_step(f"{len(items)} item(s) -> {out_path}")
    return len(items)


def run(documents: list[tuple[str, str]], call_llm=None, limit: int = 0,
        resume: bool = True) -> tuple[int, int, list[str]]:
    """Generate a paper for each document. Returns (written, items, failures).

    `documents` comes from `discover_documents`. `resume` skips a document whose
    `questions.json` is already newer than its IR, so a run interrupted halfway
    picks up where it stopped. `failures` holds one line per document that
    raised, and those documents keep whatever paper they already had.

    Each document's own output is indented under its banner, the same nesting a
    corpus extraction uses, so the run reads as a list of documents rather than
    one undifferentiated stream.
    """
    written = skipped = total_items = 0
    failures: list[str] = []
    for index, (stem, ir_path) in enumerate(documents, start=1):
        out_path = questions_path(ir_path)
        if resume and is_current(out_path, ir_path):
            skipped += 1
            continue
        print()
        print(f"document {index}/{len(documents)}  {stem}")
        outer = console.set_base_indent_level(1)
        try:
            total_items += generate_for_document(
                ir_path, out_path, call_llm=call_llm, limit=limit)
            written += 1
        except Exception as error:
            failures.append(f"{stem}: {error}")
            console.announce_step(f"failed: {error}")
            for line in traceback.format_exc().strip().splitlines()[-3:]:
                console.announce_detail(line)
        finally:
            console.set_base_indent_level(outer)
    if skipped:
        print()
        print(f"{skipped} document(s) already had a paper newer than their IR "
              f"(--no-resume to rebuild)")
    return written, total_items, failures


def main(argv=None):
    """CLI: generate `questions.json` for every document under `artifacts`."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Single-best-answer MCQs for every document a corpus run "
                    "left in the artifacts directory.")
    ap.add_argument("--artifacts", default=None,
                    help=f"directory the corpus run wrote into "
                         f"(default: {config.ARTIFACTS_DIR}/)")
    ap.add_argument("--llm", default=None,
                    help="backend profile for stem phrasing; omit for "
                         "templates only")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep at most this many items per document")
    ap.add_argument("--only", nargs="+", default=None,
                    help="document stems to generate for, e.g. "
                         "chapter_2_water_and_ph; default is all of them")
    ap.add_argument("--no-resume", action="store_true",
                    help="rebuild every paper, even one already newer than "
                         "its IR")
    args = ap.parse_args(argv)

    if args.artifacts:
        config.ARTIFACTS_DIR = args.artifacts

    documents = discover_documents()
    if args.only:
        wanted = set(args.only)
        missing = wanted - {stem for stem, _ in documents}
        documents = [d for d in documents if d[0] in wanted]
        for stem in sorted(missing):
            print(f"no IR for {stem} under {docs_root()}")
    print(f"generating MCQs for {len(documents)} document(s) in "
          f"{docs_root()}/")
    if not documents:
        print(f"nothing to do: no ir.json found. Run "
              f"`python run.py --mode corpus` first.")
        return 1

    call = None
    if args.llm:
        console.announce_step(
            f"loading the {args.llm} backend for stem phrasing")
        from .llm_backends import get_backend
        call, _ = get_backend(args.llm)
    else:
        console.announce_step("no --llm: stems stay templated")

    written, total_items, failures = run(
        documents, call_llm=call, limit=args.limit, resume=not args.no_resume)

    print()
    print(f"{written} document(s) written, "
          f"{console.format_count(total_items, 'item')} in total")
    print(f"questions -> {docs_root()}/<document>/{QUESTIONS_NAME}")
    if failures:
        print(f"{len(failures)} document(s) failed:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
