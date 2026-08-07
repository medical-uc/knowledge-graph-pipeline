"""Relation/modifier ontology: loading, CURIE expansion, and validation.

The ontology used to be a literal in `config.py`. It is now data, because it is
the thing you will change most often and it is corpus-specific: a cardiology
ontology extracts almost nothing from a physiology chapter, and vice versa.

    MEDKG_ONTOLOGY=/path/to/ontology.json      # or run.py --ontology

The load-bearing idea is `validate`. A relation's `head`/`tail` name **span
labels**, and a span label only exists if some configured NER model emits it.
The shipped default used to reference `BodyStructure`, `Finding` and `Symptom`
while the only NER model in the pipeline was `en_ner_bc5cdr_md`, which emits
`DISEASE` and `CHEMICAL` and nothing else — so three of five relation types
could never fire, no matter how the prompt was worded. That failure is silent:
extraction just returns little, and nothing says why. `validate` makes it loud.

Adding a relation type is therefore a two-part change: the type here, and an
NER model that can produce its endpoints.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONTOLOGY_PATH = os.path.join(_HERE, "ontology_medschool.json")

# CURIE prefixes understood in `uri` fields.
PREFIXES = {
    "ont": "http://example.org/medkg/ont#",
    "sct": "http://snomed.info/id/",
    "rxn": "http://purl.bioontology.org/ontology/RXNORM/",
    "inst": "http://example.org/medkg/inst/",
}


def expand(curie: str, prefixes: dict = None) -> str:
    """`sct:363698007` -> `http://snomed.info/id/363698007`. Full URIs pass through."""
    pref = PREFIXES if prefixes is None else prefixes
    if not isinstance(curie, str) or "://" in curie:
        return curie
    head, _, tail = curie.partition(":")
    return pref[head] + tail if head in pref and tail else curie


def load(path: str = None) -> dict:
    """Read an ontology file and expand every CURIE. Pure apart from the read."""
    path = path or os.environ.get("MEDKG_ONTOLOGY") or DEFAULT_ONTOLOGY_PATH
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return normalize(data, path)


def normalize(data: dict, path: str = "<memory>") -> dict:
    """Expand CURIEs and coerce to the shapes the stages expect.

    `head`/`tail` become sets because Stage 2 does membership tests on them;
    `tail: null` is preserved as None, which is the sentinel for a UNARY
    relation and must not be confused with "no constraint".
    """
    relations: dict[str, dict] = {}
    for name, spec in (data.get("relations") or {}).items():
        relations[name] = {
            "head": set(spec["head"]) if spec.get("head") else None,
            "tail": set(spec["tail"]) if spec.get("tail") else None,
            "uri": expand(spec["uri"]),
            "source": spec.get("source", ""),
            # `gloss` is not documentation -- it is sent to the extractor, and is
            # the ONLY thing that tells it which way round a relation goes.
            "gloss": spec.get("gloss", ""),
            "symmetric": bool(spec.get("symmetric", False)),
            # A more general type. Stage 2 asserts only the most specific type
            # that fits; Stage 7 emits the rdfs:subPropertyOf axiom so the
            # parent is INFERRED rather than claimed twice.
            "subPropertyOf": spec.get("subPropertyOf") or None,
            # other names for the same relation, and for it read from the other
            # end (Stage 2 accepts the latter by swapping head and tail)
            "aliases": list(spec.get("aliases") or ()),
            "inverseOf": list(spec.get("inverseOf") or ()),
        }
    label_parents = {str(k): str(v) for k, v in (data.get("label_hierarchy") or {}).items()}
    modifiers = {name: expand(spec["uri"])
                 for name, spec in (data.get("modifiers") or {}).items()}
    modifier_meta = {name: spec for name, spec in (data.get("modifiers") or {}).items()}
    values = {k: expand(v) for k, v in (data.get("modifier_values") or {}).items()}
    # Value pairs that cannot both hold of one mention. Only genuine opposites
    # belong here: insulin deficiency and insulin resistance are two facts about
    # one hormone, and treating "two different values" as a contradiction would
    # discard both.
    antonyms = frozenset(
        frozenset(expand(v) for v in pair)
        for pair in (data.get("modifier_value_antonyms") or ())
        if len(pair) == 2)
    ner_models = {m: dict(labels) for m, labels in (data.get("ner_models") or {}).items()}
    semantic_types = {
        str(tui): str(label)
        for tui, label in (data.get("semantic_types") or {}).items()}
    # Most specific label first. The label hierarchy weakens upward only, so
    # when a concept carries several semantic types the specific answer keeps
    # the most relation types reachable.
    priority = tuple(data.get("semantic_type_priority") or ())
    return {
        "version": data.get("version", "unversioned"),
        "path": path,
        "relations": relations,
        "modifiers": modifiers,
        "modifier_meta": modifier_meta,
        "modifier_values": values,
        "modifier_value_antonyms": antonyms,
        "ner_models": ner_models,
        "label_hierarchy": label_parents,
        "semantic_types": semantic_types,
        "semantic_type_priority": priority,
    }


def emitted_labels(onto: dict, models=None,
                   semantic_types: bool = True) -> set[str]:
    """Span labels a run can actually produce.

    Two sources. NER emits the label a span starts with; Stage 3 then re-derives
    it from the linked concept's UMLS semantic type, which reaches labels no
    model in `ner_models` emits — `Finding` and `Symptom` are only reachable
    that way, and `has_function` was reported dead for exactly that reason.
    Pass `semantic_types=False` to ask what NER alone can reach, which is what a
    run against an index with no `semantic_types.json` gets.
    """
    models = list(onto["ner_models"]) if models is None else list(models)
    out: set[str] = set()
    for m in models:
        out |= set(onto["ner_models"].get(m, {}).values())
    if semantic_types:
        out |= set(onto.get("semantic_types", {}).values())
    return out


def ancestors(label: str, hierarchy: dict[str, str]) -> list[str]:
    """[label, parent, grandparent, ...]. Cycle-safe."""
    out, seen, cur = [], set(), label
    while cur and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = hierarchy.get(cur)
    return out


def label_satisfies(label, allowed, hierarchy: dict[str, str]) -> bool:
    """Does a span with `label` satisfy a head/tail constraint?

    Membership is checked against the label AND its ancestors, so a constraint
    written as `Substance` is met by a span the NER called `Drug`. This is the
    span-label counterpart of `subPropertyOf` for relations, and it exists for a
    measured reason: bc5cdr maps every CHEMICAL to Drug, so in one real run 115
    of 539 spans could not satisfy a single `Substance` constraint and 127
    otherwise-valid relations were discarded on type grounds alone.

    Constraints are NOT weakened in the other direction: `treated_with` still
    demands a Drug, and a span labelled only `Substance` will not do.
    """
    if allowed is None:
        return True
    from . import config
    if allowed is config.ANY or allowed == config.ANY:
        return True
    if label is None:
        return False
    return any(a in allowed for a in ancestors(label, hierarchy))


def alias_map(onto: dict) -> dict[str, tuple[str, bool]]:
    """other-name -> (canonical type, swap head/tail).

    Built from every relation's `aliases` (swap=False) and `inverseOf`
    (swap=True). Canonical names win over aliases if they ever collide.
    """
    out: dict[str, tuple[str, bool]] = {}
    for name, spec in onto["relations"].items():
        for alt in spec.get("aliases", ()):
            out.setdefault(alt, (name, False))
        for alt in spec.get("inverseOf", ()):
            out.setdefault(alt, (name, True))
    for name in onto["relations"]:
        out.pop(name, None)
    return out


def validate_labels(onto: dict) -> list[dict]:
    """Report label-hierarchy cycles and parents that no relation type mentions."""
    hier = onto.get("label_hierarchy", {})
    problems = []
    for child in hier:
        seen, cur = {child}, hier.get(child)
        while cur:
            if cur in seen:
                problems.append({"label": child,
                                 "reason": f"label_hierarchy cycles at {cur!r}"})
                break
            seen.add(cur)
            cur = hier.get(cur)
    return problems


def parent_map(onto: dict) -> dict[str, str]:
    """child relation type -> parent type, for the types that declare one."""
    return {name: spec["subPropertyOf"]
            for name, spec in onto["relations"].items()
            if spec.get("subPropertyOf")}


def validate_hierarchy(onto: dict) -> list[dict]:
    """Report `subPropertyOf` targets that are missing, self-referential or
    cyclic. A cycle here would make Stage 2's subsumption pruning delete both
    members of a pair and Stage 7 loop, so it is checked once at load rather
    than discovered at runtime."""
    rels = onto["relations"]
    problems: list[dict] = []
    for name, spec in rels.items():
        parent = spec.get("subPropertyOf")
        if not parent:
            continue
        if parent not in rels:
            problems.append({"relation": name,
                             "reason": f"subPropertyOf names an unknown type {parent!r}"})
            continue
        seen, cur = {name}, parent
        while cur:
            if cur in seen:
                problems.append({"relation": name,
                                 "reason": f"subPropertyOf chain cycles at {cur!r}"})
                break
            seen.add(cur)
            cur = rels.get(cur, {}).get("subPropertyOf")
    return problems


def validate(onto: dict, models=None,
             semantic_types: bool = True) -> list[dict]:
    """Report relation types that cannot fire with the configured NER models.

    Returns one entry per unusable type. An empty list means every relation in
    the ontology has at least one reachable head label and (unless unary) at
    least one reachable tail label. `semantic_types` says whether Stage 3's
    label reconciliation is counted as a source of labels; see `emitted_labels`.
    """
    available = emitted_labels(onto, models, semantic_types)
    problems: list[dict] = []
    problems += validate_hierarchy(onto)
    problems += validate_labels(onto)
    hier = onto.get("label_hierarchy", {})
    # a model that emits Drug can satisfy a Substance constraint, so expand the
    # available set upward before testing reachability -- otherwise
    # --ontology-check reports types as dead that the hierarchy just revived.
    available = {a for lab in available for a in ancestors(lab, hier)}
    for name, spec in onto["relations"].items():
        missing_head = spec["head"] and not (spec["head"] & available)
        missing_tail = spec["tail"] is not None and not (spec["tail"] & available)
        if missing_head or missing_tail:
            problems.append({
                "relation": name,
                "reason": "no NER model emits its "
                          + (" and ".join(x for x in (
                              "head label" if missing_head else "",
                              "tail label" if missing_tail else "") if x)),
                "head": sorted(spec["head"] or []),
                "tail": sorted(spec["tail"]) if spec["tail"] else None,
                "available": sorted(available),
            })
    return problems


def summary(onto: dict, models=None, default_models: bool = False,
            semantic_types: bool = True) -> str:
    """Reachability report. `models` MUST be the same list the run will use.

    Reachability used to be entirely a function of the NER models: with bc5cdr
    alone only Disease and Drug exist, so 10 relation types are dead. Stage 3's
    label reconciliation is the second source, and it reaches labels no model
    emits, so the report names both and says which it counted. A report that
    silently described a different configuration than the warning that sent you
    to it is worse than no report.
    """
    problems = validate(onto, models, semantic_types)
    usable = len(onto["relations"]) - len(problems)
    shown = list(models) if models is not None else list(onto["ner_models"])
    from_ner = sorted(emitted_labels(onto, models, semantic_types=False))
    available = sorted(emitted_labels(onto, models, semantic_types))
    lines = [f"ontology {onto['version']} ({onto['path']})",
             f"  NER models checked: {', '.join(shown) or '(none)'}"
             + ("   <- DEFAULT; pass --ner to check the models you actually run"
                if default_models else ""),
             f"  relations: {len(onto['relations'])} ({usable} usable)",
             f"  modifiers: {len(onto['modifiers'])}",
             f"  span labels from NER: {', '.join(from_ner) or '(none)'}",
             f"  span labels available: {', '.join(available) or '(none)'}"
             + ("   (NER + Stage-3 semantic types)" if semantic_types
                else "   (NER only: the index has no semantic_types.json)")]
    for p in problems:
        lines.append(f"  UNUSABLE {p['relation']}: {p['reason']} "
                     f"(head={p['head']} tail={p['tail']})")
    return "\n".join(lines)
