"""Meditron backend — a thin shim over `llm_backends`.

The generic OpenAI-compatible client now lives in `llm_backends.py` as a set of
per-model profiles (Meditron on llama.cpp, HuatuoGPT-o1 on vLLM). This module
keeps Meditron's original public API so existing callers, tests and docs keep
working:

    export MEDITRON_URL=http://jupyter-02.aml1.id.iosda.org:8888   # '/v1' auto-appended
    export MEDITRON_MODEL=meditron3-8b-gguf
    export MEDITRON_GUIDED=json_schema|vllm|none

Meditron3 base checkpoints are NOT instruction-tuned, so output is constrained
rather than trusted. Constraint guarantees STRUCTURE, not correctness: Stage 2's
`validate_relation` / `validate_modifier` and Stage 1's `check_faithful` still
drop off-ontology, ungrounded, low-confidence or unfaithful items, so a weaker
model routes more work to `needs_review` instead of corrupting the graph.

If the GGUF has no chat template the /chat endpoint may misbehave — set
MEDITRON_GUIDED=none and/or use a few-shot prompt. Research-only model; not
validated for clinical use.
"""
from __future__ import annotations

from . import llm_backends as _b
from .llm_backends import build_rerank_messages  # noqa: F401  (re-exported)

PROFILE = _b.PROFILES["meditron"]

_DEFAULT_URL = PROFILE.default_url
_DEFAULT_MODEL = PROFILE.default_model


def _base_url() -> str:
    return _b.base_url(PROFILE)


def _model() -> str:
    return _b.model_id(PROFILE)


def _guided_mode() -> str:
    return _b.guided_mode(PROFILE)


def json_kwargs(schema) -> dict:
    """Server-specific kwargs to constrain a call to `schema` (a JSON array)."""
    return _b.json_kwargs(schema, PROFILE)


def choice_kwargs(choices) -> dict:
    """Constrain output to exactly one of `choices` (used by the rerank)."""
    return _b.choice_kwargs(choices, PROFILE)


def call_meditron(system: str, user: str, schema=None) -> str:
    """Drop-in for stage2_extract._call_anthropic."""
    return _b.make_call(PROFILE)(system, user, schema)


def meditron_rerank(mention: str, context: str, candidates) -> str:
    """Stage 3 rerank hook; falls back to the top-cosine CUI on a bad reply."""
    return _b.make_rerank(PROFILE)(mention, context, candidates)
