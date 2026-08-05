"""HuatuoGPT-o1 backend — `mlx-community/HuatuoGPT-o1-72B-4bit` on vLLM.

Served alongside Meditron on the same host, as model id `huatuogpt-o1-72b-4bit`:

    export HUATUO_URL=http://jupyter-02.aml1.id.iosda.org:8888   # '/v1' auto-appended
    export HUATUO_MODEL=huatuogpt-o1-72b-4bit
    export HUATUO_GUIDED=vllm|json_schema|none
    export HUATUO_MAX_TOKENS=4096

Two differences from Meditron drive the profile in `llm_backends`:

  * **vLLM guided decoding** uses `extra_body={"guided_json"|"guided_choice"}`,
    not `response_format` — so the default guided mode is `vllm`.
  * **It reasons before answering.** Replies may carry a chain-of-thought
    preamble, which `strip_reasoning` / `extract_json_text` remove before any
    stage sees the text, and `max_tokens` defaults to 4096 because the reasoning
    spends the same budget as the answer.

Worth knowing: guided decoding forces JSON immediately and therefore suppresses
the reasoning this model was trained to produce. The default stays `vllm`
because an unparseable reply becomes silent `needs_review` volume that no
validator can recover — but `HUATUO_GUIDED=none` lets it think and is the
setting to try for the Stage-3 rerank, where medical judgement matters more than
output shape. Being a 72B 4-bit model it is much slower per call than Meditron
3-8B; Stage 1 batches one call per figure-adjacent paragraph and one per list,
so cost scales with structure, not document length.

Research-only model; not validated for clinical use.
"""
from __future__ import annotations

from . import llm_backends as _b
from .llm_backends import build_rerank_messages  # noqa: F401  (re-exported)

PROFILE = _b.PROFILES["huatuo"]


def _base_url() -> str:
    return _b.base_url(PROFILE)


def _model() -> str:
    return _b.model_id(PROFILE)


def _guided_mode() -> str:
    return _b.guided_mode(PROFILE)


def json_kwargs(schema) -> dict:
    return _b.json_kwargs(schema, PROFILE)


def choice_kwargs(choices) -> dict:
    return _b.choice_kwargs(choices, PROFILE)


def call_huatuo(system: str, user: str, schema=None) -> str:
    """Drop-in for stage2_extract._call_anthropic / Stage 1's call_llm."""
    return _b.make_call(PROFILE)(system, user, schema)


def huatuo_rerank(mention: str, context: str, candidates) -> str:
    """Stage 3 rerank hook; falls back to the top-cosine CUI on a bad reply."""
    return _b.make_rerank(PROFILE)(mention, context, candidates)
