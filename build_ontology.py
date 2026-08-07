#!/usr/bin/env python3
"""Extend the relation ontology — by adoption first, induction second.

    # 1. Adopt: UMLS already enumerated biomedical relations. You have the files.
    python build_ontology.py --from-srdef  UMLS/META/NET/SRDEF \\
                             --srstre2     UMLS/META/NET/SRSTRE2 \\
                             --out proposals.json

    # 2. Induce: what the corpus expressed that the ontology has no type for.
    python build_ontology.py --induce graph.ir.json --llm huatuo --out proposals.json

    # 3. Review proposals.json BY HAND, then merge.
    python build_ontology.py --merge proposals.json --into my_ontology.json

Why induction is a build step and not a runtime one
---------------------------------------------------
Letting the extractor invent a relation type per chunk is the one change that
would destroy the graph's value, and it is worth being precise about why:

  * **Queries stop working.** The same fact arrives as `secretes`, `produces`,
    `synthesizes`, `releases` and `secreted_by` depending on the sentence, so
    `?cell ont:secretes ?hormone` silently misses most of them. A knowledge
    graph is only worth building because the predicates are *consistent*; an
    open vocabulary is a pile of text with angle brackets.
  * **Reasoning stops working.** Stage 7 materializes over declared domains and
    ranges. A predicate first seen at runtime has neither, so it infers nothing
    and cannot be checked for misuse.
  * **Post-coordination stops working.** Stage 4 maps a modifier type to an
    attribute URI. An invented type has no URI, so it becomes an unqueryable
    `ont:hasModifier` blob or is dropped.
  * **The errors become invisible.** Today a bad relation type is *rejected* by
    `validate_relation` and lands in `needs_review` where you can count it.
    Accept anything and the failure mode changes from "visibly rejected" to
    "silently inconsistent", which is far worse in a medical context.

So: the LLM proposes, offline, over a corpus sample; a human disposes; the
runtime vocabulary stays closed. `needs_review` is the feedback loop — rejected
relation types accumulate there, and a *pattern* in them is the signal to run
this tool again.

And prefer adoption over invention. UMLS's Semantic Network, SNOMED's concept
model, RO and BioLink have each spent years enumerating biomedical relations
with real domains and ranges. Anything you mint yourself is a term no other
dataset will ever join on.

Stdlib only; the LLM is injected.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import medkg                                    # noqa: F401 -- thread pinning
from medkg import config, console, ontology

# ---------------------------------------------------------------------------
# 1. Adoption: UMLS Semantic Network
# ---------------------------------------------------------------------------

# UMLS semantic type -> our span labels. Deliberately partial: an unmapped type
# is reported as a TODO rather than guessed, because a wrong domain silently
# suppresses every relation that uses it.
STY_TO_LABEL = {
    "Disease or Syndrome": "Disease", "Neoplastic Process": "Disease",
    "Injury or Poisoning": "Disease", "Mental or Behavioral Dysfunction": "Disease",
    "Sign or Symptom": "Symptom", "Finding": "Finding",
    "Laboratory or Test Result": "Finding",
    "Body Part, Organ, or Organ Component": "BodyStructure",
    "Body System": "BodyStructure", "Tissue": "BodyStructure",
    "Body Location or Region": "BodyStructure", "Body Space or Junction": "BodyStructure",
    "Cell": "Cell", "Cell Component": "CellComponent",
    "Pharmacologic Substance": "Drug", "Clinical Drug": "Drug",
    "Antibiotic": "Drug",
    "Organic Chemical": "Substance", "Biologically Active Substance": "Substance",
    "Hormone": "Substance", "Enzyme": "Protein",
    "Amino Acid, Peptide, or Protein": "Protein",
    "Gene or Genome": "Protein",
    "Element, Ion, or Isotope": "Substance", "Inorganic Chemical": "Substance",
    "Body Substance": "Substance",
    "Bacterium": "Organism", "Virus": "Organism", "Fungus": "Organism",
    "Anatomical Abnormality": "Morphology",
}


def parse_srdef(text: str) -> list[dict]:
    """UMLS NET/SRDEF -> relation records. Pipe-delimited; RT is the first field
    and is `RL` for relations (`STY` rows are semantic types, skipped)."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        if len(f) < 4 or f[0] != "RL":
            continue
        out.append({"ui": f[1], "name": f[2], "tree": f[3],
                    "definition": f[4] if len(f) > 4 else "",
                    "inverse": f[9] if len(f) > 9 else ""})
    return out


def parse_srstre2(text: str) -> dict[str, dict[str, set]]:
    """UMLS NET/SRSTRE2 -> {relation: {"head": {sty}, "tail": {sty}}}.

    Rows are `SemanticType|Relation|SemanticType`, giving the relation's real
    domain and range — the part that makes an adopted type usable rather than
    just named.
    """
    dr: dict[str, dict[str, set]] = {}
    for line in text.splitlines():
        f = line.strip().split("|")
        if len(f) < 3 or not f[1]:
            continue
        slot = dr.setdefault(f[1], {"head": set(), "tail": set()})
        slot["head"].add(f[0])
        slot["tail"].add(f[2])
    return dr


def _labels_for(stys: set) -> tuple[list, list]:
    """(mapped labels, unmapped semantic types)."""
    mapped = sorted({STY_TO_LABEL[s] for s in stys if s in STY_TO_LABEL})
    todo = sorted(s for s in stys if s not in STY_TO_LABEL)
    return mapped, todo


def propose_from_umls(srdef_text: str, srstre2_text: str, existing: dict) -> dict:
    """Relation types present in the UMLS Semantic Network but not in the ontology."""
    known = {n.lower() for n in existing["relations"]}
    dr = parse_srstre2(srstre2_text)
    proposals = {}
    for rec in parse_srdef(srdef_text):
        name = rec["name"].strip().replace(" ", "_").lower()
        if name in known or name in ("isa", "inverse_isa"):
            continue
        head, head_todo = _labels_for(dr.get(rec["name"], {}).get("head", set()))
        tail, tail_todo = _labels_for(dr.get(rec["name"], {}).get("tail", set()))
        if not head and not tail:
            continue                        # no domain/range: nothing usable yet
        proposals[name] = {
            "head": head, "tail": tail,
            "uri": f"ont:{name}",
            "source": f"UMLS-SN:{rec['ui']} {rec['name']}",
            "gloss": " ".join(rec["definition"].split())[:300],
            "_review": {"inverse": rec["inverse"],
                        "unmapped_head_sty": head_todo[:8],
                        "unmapped_tail_sty": tail_todo[:8]},
        }
    return proposals


# ---------------------------------------------------------------------------
# 2. Induction: what the corpus said that the ontology has no word for
# ---------------------------------------------------------------------------

INDUCE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name":     {"type": "string"},
            "head":     {"type": "array", "items": {"type": "string"}},
            "tail":     {"type": "array", "items": {"type": "string"}},
            "gloss":    {"type": "string"},
            "example":  {"type": "string"},
        },
        "required": ["name", "head", "tail", "gloss", "example"],
    },
}


def build_induce_prompt(samples: list[str], existing: list[str], labels: list[str]) -> str:
    return (
        "Below are passages from a medical textbook.\n\n"
        "These relation types already exist; do NOT propose synonyms of them:\n"
        + ", ".join(sorted(existing)) + "\n\n"
        "Entity labels available (head/tail must be drawn from this list only):\n"
        + ", ".join(sorted(labels)) + "\n\n"
        "=== PASSAGES ===\n" + "\n\n".join(samples) + "\n=== END ===\n\n"
        "Propose relation types that these passages express and the existing list "
        "cannot capture. Rules:\n"
        "1. A type must be reusable across many documents, not a restatement of "
        "one sentence.\n"
        "2. Prefer the most general form that stays precise: 'secretes', not "
        "'secretes_thyroid_hormone'.\n"
        "3. Do not propose a near-synonym of an existing type. If 'secretes' "
        "exists, 'releases' is not a new type.\n"
        "4. head and tail must be entity labels from the list above.\n"
        "5. Give a real example sentence from the passages for each.\n\n"
        "Return ONLY a JSON array of "
        '{name, head, tail, gloss, example}.'
    )


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z]+", name.lower()) if len(t) > 2}


def near_duplicate(name: str, existing: list[str], threshold: float = 0.6) -> str:
    """The closest existing type, or "". Token-overlap only — deliberately crude,
    because its job is to make a reviewer look, not to decide."""
    cand = _tokens(name)
    best, best_score = "", 0.0
    for other in existing:
        toks = _tokens(other)
        if not cand or not toks:
            continue
        score = len(cand & toks) / len(cand | toks)
        if score > best_score:
            best, best_score = other, score
    return best if best_score >= threshold else ""


def induce_from_corpus(doc, call_llm, sample_size: int = 12) -> dict:
    """Propose relation types from a Stage-2+ IR document."""
    existing = list(config.RELATION_ONTOLOGY)
    labels = sorted({s.label for s in doc.spans} or ontology.emitted_labels(config.ONTOLOGY))
    chunks = [c.text for c in doc.extractable_chunks(config.EXTRACTABLE_CHUNK_KINDS)]
    samples = chunks[:sample_size]
    if not samples:
        return {}
    raw = call_llm("You are a biomedical ontologist. Reply with JSON only.",
                   build_induce_prompt(samples, existing, labels),
                   schema=INDUCE_SCHEMA)
    try:
        items = json.loads(re.sub(r"```(?:json)?|```", "", raw or "").strip())
    except json.JSONDecodeError:
        return {}
    proposals = {}
    for item in items if isinstance(items, list) else []:
        name = str(item.get("name", "")).strip().lower().replace(" ", "_")
        if not name or name in config.RELATION_ONTOLOGY:
            continue
        dup = near_duplicate(name, existing)
        proposals[name] = {
            "head": [h for h in item.get("head", []) if h in labels],
            "tail": [t for t in item.get("tail", []) if t in labels],
            "uri": f"ont:{name}",
            "source": "induced (LLM proposal — UNVERIFIED, map to UMLS-SN/RO before use)",
            "gloss": str(item.get("gloss", ""))[:300],
            "_review": {"example": str(item.get("example", ""))[:300],
                        "possible_duplicate_of": dup},
        }
    return proposals


# ---------------------------------------------------------------------------
# 3. Merge (after human review)
# ---------------------------------------------------------------------------

def merge(base: dict, proposals: dict) -> tuple[dict, list[str]]:
    """Add proposals to a raw ontology dict. Never overwrites an existing type,
    and drops anything still missing head/tail — an unreviewed proposal is not
    an ontology entry."""
    out = json.loads(json.dumps(base))
    added, skipped = [], []
    for name, spec in proposals.items():
        if name in out.get("relations", {}):
            skipped.append(f"{name}: already present")
            continue
        if not spec.get("head") or ("tail" in spec and spec["tail"] == []):
            skipped.append(f"{name}: head/tail not filled in — review it first")
            continue
        clean = {k: v for k, v in spec.items() if not k.startswith("_")}
        out.setdefault("relations", {})[name] = clean
        added.append(name)
    return out, (added, skipped)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Propose relation-ontology extensions. Adoption first, induction second.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Proposals are NEVER applied automatically — review, then --merge.")
    ap.add_argument("--from-srdef", help="UMLS NET/SRDEF")
    ap.add_argument("--srstre2", help="UMLS NET/SRSTRE2 (domains and ranges)")
    ap.add_argument("--induce", help="Stage-2+ IR JSON to induce from")
    ap.add_argument("--llm", choices=["claude", "meditron", "huatuo"], default="huatuo")
    ap.add_argument("--sample-size", type=int, default=12)
    ap.add_argument("--merge", help="proposals JSON to merge")
    ap.add_argument("--into", help="ontology JSON to merge into (default: the bundled one)")
    ap.add_argument("--out", default="proposals.json")
    args = ap.parse_args(argv)

    if args.merge:
        base_path = args.into or ontology.DEFAULT_ONTOLOGY_PATH
        print(f"merging {args.merge} into {base_path}")
        with open(base_path, encoding="utf-8") as fh:
            base = json.load(fh)
        with open(args.merge, encoding="utf-8") as fh:
            proposals = json.load(fh).get("relations", {})
        console.announce_step(
            f"{len(base.get('relations', {}))} existing type(s), "
            f"{len(proposals)} proposed")
        merged, (added, skipped) = merge(base, proposals)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False)
        print(f"merged {len(added)} type(s) into {args.out}")
        for name in added:
            print("  +", name)
        for note in skipped:
            print("  skipped", note)
        return 0

    proposals: dict = {}
    if args.from_srdef:
        if not args.srstre2:
            sys.stderr.write("error: --from-srdef also needs --srstre2 for domains/ranges\n")
            return 2
        print(f"reading the UMLS Semantic Network")
        console.announce_step(f"definitions {args.from_srdef}")
        with open(args.from_srdef, encoding="utf-8", errors="replace") as fh:
            srdef = fh.read()
        console.announce_step(f"domains and ranges {args.srstre2}")
        with open(args.srstre2, encoding="utf-8", errors="replace") as fh:
            srstre2 = fh.read()
        proposals.update(propose_from_umls(srdef, srstre2, config.ONTOLOGY))
        print(f"UMLS Semantic Network: {len(proposals)} candidate type(s)")

    if args.induce:
        from medkg.ir import Document
        from medkg import llm_backends
        print(f"inducing types from {args.induce} with the {args.llm} backend")
        doc = Document.from_json(args.induce)
        call_llm = (llm_backends.get_backend(args.llm)[0] if args.llm != "claude"
                    else __import__("medkg.stage2_extract", fromlist=["x"])._call_anthropic)
        console.announce_step(
            f"one LLM call over {args.sample_size} sampled chunk(s)")
        induced = induce_from_corpus(doc, call_llm, args.sample_size)
        print(f"corpus induction: {len(induced)} candidate type(s)")
        for name, spec in induced.items():
            if spec["_review"]["possible_duplicate_of"]:
                print(f"  ! {name} may duplicate {spec['_review']['possible_duplicate_of']}")
        proposals.update(induced)

    if not proposals:
        sys.stderr.write("nothing proposed (give --from-srdef/--srstre2 or --induce)\n")
        return 1

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"relations": proposals}, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(proposals)} proposal(s) -> {args.out}")
    print("REVIEW IT, then: python build_ontology.py --merge "
          f"{args.out} --out my_ontology.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
