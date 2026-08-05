#!/usr/bin/env python3
"""Build the SapBERT/FAISS UMLS index (./umls_index) from MRCONSO.RRF.

One command produces the four files SapBertLinker reads at run time
(umls.faiss, cuis.npy, codes.json, names.json).

Examples:
  # smoke test: embed 5k atoms, then round-trip a few mentions
  python build_index.py --mrconso /data/UMLS/META/MRCONSO.RRF --max-atoms 5000

  # recommended full build: only SNOMED + RxNorm surface forms (far smaller
  # than all of English MRCONSO, and those are the vocabularies we crosswalk)
  python build_index.py --mrconso /data/UMLS/META/MRCONSO.RRF --sabs SNOMEDCT_US,RXNORM

  # everything English (largest index, highest recall)
  python build_index.py --mrconso /data/UMLS/META/MRCONSO.RRF --sabs all
"""
import argparse
import time

from medkg import config
from medkg.stage3_link import build_index, SapBertLinker

DEFAULT_PROBES = ["myocardial infarction", "heart attack", "aspirin", "chest pain"]


def parse_sabs(s):
    """'SNOMEDCT_US,RXNORM' -> {'SNOMEDCT_US','RXNORM'}; 'all'/''/None -> None (no filter)."""
    if not s or s.strip().lower() in {"all", "*", ""}:
        return None
    return {tok.strip() for tok in s.split(",") if tok.strip()}


def sanity_check(out_dir, model, probes):
    """Round-trip known mentions to confirm the four files + model agree.

    Prints top candidates (no floor) and the floored link() result. With a small
    --max-atoms subset the 'right' concept is usually absent — that's expected;
    the point is to prove embed -> FAISS search -> cui map -> uri all round-trip.
    """
    print("\n[sanity] loading linker and probing known mentions...")
    linker = SapBertLinker(out_dir, model_name=model)
    for m in probes:
        cands = linker.candidates(m, k=3)
        best = linker.link(m, context=m)
        names = getattr(linker, "names", {})
        top = ", ".join(f"{cui}({names.get(cui, '?')}):{s:.2f}" for cui, s in cands) or "(none)"
        print(f"  {m!r:34} -> {top}")
        if best:
            print(f"       linked: {best[0]}  {best[1]}  score={best[2]:.2f}")
        else:
            print(f"       below floor / not in subset (expected on --max-atoms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mrconso", required=True, help="path to UMLS MRCONSO.RRF")
    ap.add_argument("--out-dir", default="./umls_index")
    ap.add_argument("--sabs", default="SNOMEDCT_US,RXNORM",
                    help="comma-separated SABs to EMBED, or 'all' for every English atom "
                         "(default: SNOMEDCT_US,RXNORM)")
    ap.add_argument("--keep-sabs", default="SNOMEDCT_US,RXNORM",
                    help="SABs whose codes go in the CUI->code crosswalk (default: SNOMEDCT_US,RXNORM)")
    ap.add_argument("--max-atoms", type=int, default=None, help="cap embedded atoms (smoke test)")
    ap.add_argument("--model", default=config.SAPBERT_MODEL)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--no-sanity", action="store_true", help="skip the post-build round-trip")
    ap.add_argument("--probe", nargs="*", default=None, help="custom mentions for the sanity check")
    args = ap.parse_args()

    embed_sabs = parse_sabs(args.sabs)
    keep_sabs = parse_sabs(args.keep_sabs) or {"SNOMEDCT_US", "RXNORM"}

    print(f"[build] mrconso={args.mrconso}")
    print(f"[build] embed SABs={embed_sabs or 'ALL English'} | crosswalk SABs={sorted(keep_sabs)}"
          f"{f' | max_atoms={args.max_atoms}' if args.max_atoms else ''}")
    t0 = time.time()
    build_index(args.mrconso, args.out_dir, model_name=args.model,
                keep_sabs=tuple(keep_sabs), embed_sabs=embed_sabs,
                max_atoms=args.max_atoms, batch=args.batch, progress=not args.no_progress)
    print(f"[build] done in {time.time() - t0:.1f}s")

    if not args.no_sanity:
        sanity_check(args.out_dir, args.model, args.probe or DEFAULT_PROBES)


if __name__ == "__main__":
    main()
