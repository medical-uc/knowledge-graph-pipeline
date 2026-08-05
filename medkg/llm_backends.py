"""Self-hosted OpenAI-compatible LLM backends, as per-model *profiles*.

One client, one `call_llm(system, user, schema=None)` contract, several models.
Each profile records only what actually differs between servers: base URL, model
id, how to constrain output, how much room the model needs, and whether it emits
chain-of-thought that has to be stripped before the JSON is parsed.

Shipped profiles
----------------
**meditron** — `meditron3-8b-gguf` on llama.cpp. Base checkpoint, not
instruction-tuned, so output is *constrained* rather than trusted; llama.cpp
speaks `response_format={"type":"json_schema"}`.

**huatuo** — `mlx-community/HuatuoGPT-o1-72B-4bit` served by vLLM as
`huatuogpt-o1-72b-4bit`. Two things make it different from Meditron:

  * *vLLM guided decoding.* The constraint keys are `guided_json` /
    `guided_choice` under `extra_body`, not `response_format` — hence
    `default_guided="vllm"`.
  * *It is a reasoning model.* HuatuoGPT-o1 is trained to think before it
    answers, so a reply can carry a chain-of-thought preamble ahead of the JSON.
    `strip_reasoning` + `extract_json_text` recover the payload, and
    `max_tokens` is far larger because the reasoning is billed against the same
    budget. Note the tension: guided decoding forces JSON immediately and so
    suppresses the reasoning this model was trained to do. The default is still
    `vllm`, because a reply we cannot parse becomes silent `needs_review` volume
    and no validator can recover it — but set `HUATUO_GUIDED=none` to let the
    model reason and lean on `extract_json_text`, which is worth trying for the
    Stage-3 rerank, where medical judgement matters more than structure.

Every profile is configurable by env var, prefixed per profile:
`<PREFIX>_URL`, `<PREFIX>_MODEL`, `<PREFIX>_GUIDED`, `<PREFIX>_MAX_TOKENS`
(e.g. `HUATUO_URL`, `MEDITRON_GUIDED`).

Constraints guarantee *structure*; the stages' validators (`validate_relation`,
`validate_modifier`, `check_faithful`) guarantee *correctness*. A weaker model
therefore raises review volume rather than corrupting the graph.

Research-only models; not validated for clinical use.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

_SHARED_URL = "http://jupyter-02.aml1.id.iosda.org:8888"


@dataclass(frozen=True)
class Profile:
    name: str
    env_prefix: str
    default_url: str
    default_model: str
    default_guided: str          # "json_schema" | "vllm" | "none"
    default_max_tokens: int
    reasoning: bool = False      # emits chain-of-thought before the answer
    # Reranking picks one CUI from a supplied list. Even on a reasoning model it
    # needs a handful of tokens, and `guided_choice` suppresses the reasoning
    # anyway -- giving it the full budget bought nothing and cost a timeout per
    # ambiguous mention.
    default_rerank_max_tokens: int = 24
    default_timeout: float = 120.0


PROFILES: dict[str, Profile] = {
    "meditron": Profile("meditron", "MEDITRON", _SHARED_URL,
                        "meditron3-8b-gguf", "json_schema", 1024, False,
                        default_rerank_max_tokens=24, default_timeout=120.0),
    # 72B at 4-bit on MLX is slow per call; the timeout is generous for one
    # extraction call and still bounds a hung request.
    "huatuo": Profile("huatuo", "HUATUO", _SHARED_URL,
                      "huatuogpt-o1-72b-4bit", "vllm", 4096, True,
                      default_rerank_max_tokens=24, default_timeout=300.0),
}


def get_profile(name: str) -> Profile:
    key = (name or "").lower()
    aliases = {"huatuogpt": "huatuo", "huatuogpt-o1": "huatuo",
               "huatuogpt-o1-72b-4bit": "huatuo", "meditron3": "meditron"}
    key = aliases.get(key, key)
    if key not in PROFILES:
        raise ValueError(f"unknown backend {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[key]


# --- env resolution (pure) --------------------------------------------------

def _env(profile: Profile, suffix: str, default):
    return os.environ.get(f"{profile.env_prefix}_{suffix}", default)


def base_url(profile: Profile) -> str:
    """Ensure the URL ends with /v1 — the OpenAI-compatible path both llama.cpp
    and vLLM serve."""
    base = str(_env(profile, "URL", profile.default_url)).rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def model_id(profile: Profile) -> str:
    return str(_env(profile, "MODEL", profile.default_model))


def guided_mode(profile: Profile) -> str:
    return str(_env(profile, "GUIDED", profile.default_guided))


def max_tokens(profile: Profile) -> int:
    return int(_env(profile, "MAX_TOKENS", profile.default_max_tokens))


def rerank_max_tokens(profile: Profile) -> int:
    return int(_env(profile, "RERANK_MAX_TOKENS", profile.default_rerank_max_tokens))


def timeout(profile: Profile) -> float:
    return float(_env(profile, "TIMEOUT", profile.default_timeout))


# --- constrained-decoding kwargs (pure, unit-testable) ---------------------

def json_kwargs(schema, profile: Profile) -> dict:
    """Server-specific kwargs constraining a call to `schema`."""
    mode = guided_mode(profile)
    if schema and mode == "vllm":
        return {"extra_body": {"guided_json": schema}}
    if schema and mode == "json_schema":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "medkg_out", "schema": schema}}}
    return {}                                   # 'none': prompt + validators only


def choice_kwargs(choices, profile: Profile) -> dict:
    """Constrain output to exactly one of `choices` (used by the Stage-3 rerank)."""
    mode = guided_mode(profile)
    if mode == "vllm":
        return {"extra_body": {"guided_choice": choices}}
    if mode == "json_schema":
        return {"response_format": {"type": "json_schema",
                "json_schema": {"name": "cui", "schema": {"type": "string", "enum": choices}}}}
    return {}


# --- reasoning / JSON recovery (pure, unit-testable) -----------------------

# Markers reasoning models use to close their chain-of-thought. If your build
# emits a different one, add it here — it is the single place that matters.
REASONING_MARKERS = ("## Final Response", "## Final response", "## Final Answer",
                     "</think>", "<|end_of_thought|>", "<|start_of_solution|>",
                     "**Final Response**")


def strip_reasoning(text: str) -> str:
    """Keep only what follows the LAST end-of-thought marker. HuatuoGPT-o1 emits
    `## Thinking … ## Final Response …`; the thinking must not reach the JSON
    parser, and taking the last marker is what makes nested mentions harmless."""
    if not text:
        return ""
    cut = -1
    for marker in REASONING_MARKERS:
        idx = text.rfind(marker)
        if idx > cut:
            cut = idx + len(marker)
    return text[cut:].strip() if cut >= 0 else text.strip()


def _balanced(text: str, start: int) -> Optional[str]:
    """The complete JSON value beginning at `start`, respecting strings/escapes."""
    opener = text[start]
    closer = {"[": "]", "{": "}"}[opener]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json_text(raw: str) -> str:
    """Best-effort JSON payload from a chatty or reasoning reply.

    Tries, in order: the whole (de-fenced, de-reasoned) reply; then the last
    balanced `[...]` or `{...}` in it. Returns "" when nothing parses, which the
    stages already treat as "no items" and route to `needs_review` — a chatty
    model loses items, it never invents them.
    """
    text = re.sub(r"```(?:json)?|```", "", strip_reasoning(raw or "")).strip()
    if not text:
        return ""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] in "[{":
            candidate = _balanced(text, idx)
            if candidate:
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    return ""


# --- client + the two injectable hooks -------------------------------------

@lru_cache(maxsize=8)
def _client(url: str, timeout_s: float = 300.0, retries: int = 2):
    from openai import OpenAI      # lazy: module stays importable without openai
    # An explicit timeout matters more than its value: the SDK default is 10
    # minutes, so a hung request stalls the run long enough to look like a crash.
    return OpenAI(base_url=url, api_key="not-needed",   # local servers ignore the key
                  timeout=timeout_s, max_retries=retries)


def build_rerank_messages(mention: str, context: str, candidates):
    """Pure helper: the (system, user) prompt for picking the best CUI.
    `candidates` is a list of (cui, name, score)."""
    listing = "\n".join(f"{cui}: {name}" for cui, name, _ in candidates)
    system = ("You are a biomedical entity disambiguator. Given a mention in "
              "context and candidate UMLS concepts, choose the single CUI whose "
              "concept best matches the mention. Reply with only the CUI.")
    user = (f"Mention: {mention!r}\nContext: {context!r}\n"
            f"Candidates:\n{listing}\n\nReturn the best CUI.")
    return system, user


class BackendUnavailable(RuntimeError):
    """Raised when the backend has failed too many times in a row to continue."""


# Consecutive failures after which we stop rather than keep going. A single
# timeout is a hiccup; a run of them means the server is down, and continuing
# would produce a graph that looks successful and contains almost nothing --
# the worst possible outcome for a medical KG, because it is silent.
CIRCUIT_BREAKER = int(os.environ.get("MEDKG_LLM_MAX_CONSECUTIVE_FAILURES", "5"))


def make_call(profile: Profile, on_error=None, breaker: int = None):
    """Build a `call_llm(system, user, schema=None)` for this profile — the exact
    hook Stage 1's classify/normalize passes and Stage 2's RE/modifier passes take.

    Failure policy, which is a judgement call worth stating:

    * **One failed call is not fatal.** Stage 2 makes a call per chunk, so a
      single timeout at chunk 40 of 50 used to discard 39 chunks of completed
      extraction. A failure now returns `""`, which every caller already treats
      as "no items" and routes to `needs_review` — recorded data loss, not
      silent, and re-runnable.
    * **A run of failures IS fatal.** If the server is down, returning `""`
      forever produces an empty graph and reports success. `BackendUnavailable`
      after `CIRCUIT_BREAKER` consecutive failures stops that. In a medical
      pipeline, loud failure beats a hollow artefact.
    """
    limit = CIRCUIT_BREAKER if breaker is None else breaker
    state = {"consecutive": 0, "failures": 0, "calls": 0}

    def call(system: str, user: str, schema=None) -> str:
        state["calls"] += 1
        try:
            resp = _client(base_url(profile), timeout(profile)).chat.completions.create(
                model=model_id(profile),
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens(profile), temperature=0.0,
                **json_kwargs(schema, profile),
            )
        except Exception as exc:                      # noqa: BLE001
            state["consecutive"] += 1
            state["failures"] += 1
            if on_error is not None:
                on_error(exc, dict(state))
            if limit and state["consecutive"] >= limit:
                raise BackendUnavailable(
                    f"{state['consecutive']} consecutive failures from "
                    f"{model_id(profile)} at {base_url(profile)} "
                    f"(last: {type(exc).__name__}). Stopping rather than emitting "
                    f"a graph built from nothing. Check the server, or raise "
                    f"{profile.env_prefix}_TIMEOUT (currently {timeout(profile)}s)."
                ) from exc
            return ""
        state["consecutive"] = 0
        raw = resp.choices[0].message.content or ""
        # Adapt the model's quirks HERE so no stage has to know about reasoning
        # traces; stages stay backend-agnostic, per the injectable-backend rule.
        return extract_json_text(raw) or raw

    call.stats = state                                # inspectable by callers
    return call


def make_rerank(profile: Profile, on_error=None):
    """Build a Stage-3 `rerank(mention, context, candidates) -> cui`.

    Two properties matter more than the reranking itself:

    * **It never raises.** Stage 3 runs after Stage 2's LLM pass, which is the
      expensive part of the pipeline. A network timeout on an optional
      disambiguation step must not destroy that work, so any failure falls back
      to the top-cosine CUI -- the answer we would have had without a reranker
      at all -- and is reported through `on_error`.
    * **It is cached.** The same mention recurs constantly in one document
      ("thyroid", "T3", "TSH"), and re-asking a 72B model for an answer it just
      gave is the difference between a run that finishes and one that doesn't.
    """
    cache: dict[tuple, str] = {}

    def rerank(mention: str, context: str, candidates) -> str:
        choices = [c[0] for c in candidates]
        if not choices:
            return ""
        key = (mention.lower(), tuple(choices))
        if key in cache:
            return cache[key]
        fallback = choices[0]
        try:
            system, user = build_rerank_messages(mention, context, candidates)
            resp = _client(base_url(profile), timeout(profile)).chat.completions.create(
                model=model_id(profile),
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=rerank_max_tokens(profile),
                temperature=0.0,
                **choice_kwargs(choices, profile),
            )
            out = strip_reasoning(resp.choices[0].message.content or "").strip().strip('"')
        except Exception as exc:                      # noqa: BLE001
            if on_error is not None:
                on_error(mention, exc)
            cache[key] = fallback
            return fallback
        # a reasoning model may wrap the CUI in a sentence
        m = re.search(r"\bC\d{6,8}\b", out)
        chosen = m.group(0) if (m and m.group(0) in choices) else (
            out if out in choices else fallback)
        cache[key] = chosen
        return chosen
    return rerank


def get_backend(name: str, on_error=None):
    """(call_llm, rerank) for a profile name."""
    profile = get_profile(name)
    return make_call(profile, on_error=on_error), make_rerank(profile)
