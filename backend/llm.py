import os
import re
import json
import time
import asyncio
import logging
from typing import Any

import httpx
from dotenv import load_dotenv

from backend.gemini import gemini_available, gemini_chat_completion, gemini_chat_completion_stream, GEMINI_MODEL


load_dotenv()

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
LLM_BASE_URL   = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY    = os.getenv("LLM_API_KEY",  "").strip().strip("'\"")
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Explicit provider override.  Values: auto | openai | azure | vllm
# 'auto' detects Azure from the URL; openai and vllm both use OpenAI-compatible format.
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "auto").strip().lower()

# RAGService (standalone microservice)
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8600").rstrip("/")
RAG_COLLECTION  = os.getenv("RAG_COLLECTION",  "kiosk-rnsit")
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "5"))
# Minimum cosine-similarity score (0-1) from RAGService to include a chunk.
# BGE embeddings: >0.55 = relevant, >0.40 = loosely related, <0.30 = noise.
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))

# ── Multi-LLM fallback config (Tier 1: local Qwen -> Tier 2: Gemini -> Tier 3: RAG-only) ──
PRIMARY_LLM            = os.getenv("PRIMARY_LLM", "local").strip().lower()
ENABLE_GEMINI_FALLBACK = os.getenv("ENABLE_GEMINI_FALLBACK", "false").strip().lower() in ("1", "true", "yes")

JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "college_info.json")

PROFANITY_BLOCKLIST = {
    "badword1", "badword2", "abuse", "stupid", "idiot"
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "from", "and", "or", "but",
    "not", "no", "it", "its", "this", "that", "there", "what", "which",
    "who", "how", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "their", "tell", "me", "please", "know",
}

_SHARED_ASYNC_CLIENT: httpx.AsyncClient | None = None
_rag_seeded: bool = False

# Local copy of main.py's Q/A label stripper — used only by the Tier-3
# RAG-only safety net below, so raw "Q: ... A: ..." scaffolding from a
# retrieved chunk never reaches the visitor when both LLM tiers are down.
#
# BUG FIX: this used to require a literal "?" between "Q:" and "A:"
# (r"Q:\s*.+?\?\s*A:\s*"). Most KB entries are phrased as questions and do
# have one, but not all — e.g. the "Q: What can you do A: ..." entry has no
# question mark, so the old pattern silently failed to match and the raw
# "Q: ... A:" scaffolding leaked straight into a visitor-facing reply (seen
# in production: "I am currently unable to generate a conversational
# response, but based on the available RNSIT information: Q: What can you
# do A: I can help you with..."). Dropping the "?" requirement fixes every
# case the old pattern covered plus this one.
_QA_LABEL_RE_LLM = re.compile(r"Q:\s*.*?\s*A:\s*", re.IGNORECASE)

# Same idea for the other two label shapes seen in this KB's chunks —
# "Facility: X. Details: ..." and "Department: X. ... Head of Department: ..."
# — so the Tier-3 safety net strips scaffolding regardless of which chunk
# shape happens to be top-ranked, not just the Q/A one.
_FACILITY_LABEL_RE_LLM = re.compile(r"^Facility:\s*.*?\.\s*Details:\s*", re.IGNORECASE)


# ==========================================
# SHARED HTTP CLIENT
# ==========================================
def get_shared_client() -> httpx.AsyncClient:
    global _SHARED_ASYNC_CLIENT
    if _SHARED_ASYNC_CLIENT is None or _SHARED_ASYNC_CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        # connect=4.0: fail fast when the LLM server isn't reachable at all
        # (unreachable host/port should be near-instant, not a long hang —
        # a real visitor stood at the kiosk for ~21s in silence before this
        # was tightened). read=30.0 stays generous since a slow LLM that IS
        # connected and actually generating tokens shouldn't be cut off.
        _SHARED_ASYNC_CLIENT = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(30.0, connect=4.0),
        )
    return _SHARED_ASYNC_CLIENT


async def close_llm_client():
    global _SHARED_ASYNC_CLIENT
    if _SHARED_ASYNC_CLIENT and not _SHARED_ASYNC_CLIENT.is_closed:
        await _SHARED_ASYNC_CLIENT.aclose()
        logger.info("[RAG] Persistent connection pool cleanly terminated.")


def _require_llm_config():
    if not LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL is not configured.")
    # NOTE: LLM_API_KEY is intentionally NOT required here anymore. The
    # team's actual server (confirmed working via their standalone test
    # script) runs with no API key at all — LLM_API_KEY="" in .env — so
    # hard-requiring a truthy key here made every call fail before it even
    # left this machine, regardless of network/URL correctness.


def _provider() -> str:
    """
    Resolve the active LLM provider:
      'azure'  – Azure OpenAI (different URL structure + 'api-key' header)
      'openai' – Standard OpenAI (api.openai.com)
      'vllm'   – vLLM / Uvicorn / LiteLLM / Ollama (OpenAI-compatible REST)
    Both 'openai' and 'vllm' use identical wire format; they are treated the same.
    """
    if LLM_PROVIDER != "auto":
        return LLM_PROVIDER                          # explicit env var wins
    if "openai.azure.com" in LLM_BASE_URL.lower():
        return "azure"
    return "openai"                                  # covers vLLM too


def _is_azure() -> bool:
    return _provider() == "azure"


def _is_new_model_family() -> bool:
    """
    True for model families that:
      - use 'max_completion_tokens' instead of 'max_tokens'
      - only support temperature=1 (default), so the param must be omitted
    Covers: o1/o2/o3/o4 reasoning series, gpt-5.x and later.
    """
    m = LLM_CHAT_MODEL.lower().lstrip("/-")
    return m.startswith(("o1", "o2", "o3", "o4", "gpt-5", "gpt5"))


def _max_tokens_param() -> str:
    """
    Newer model families (o1/o2/o3/o4, gpt-5.x) require 'max_completion_tokens'.
    Legacy models use 'max_tokens'.  Override with LLM_MAX_TOKENS_PARAM in .env.
    """
    override = os.getenv("LLM_MAX_TOKENS_PARAM", "").strip()
    if override:
        return override
    return "max_completion_tokens" if _is_new_model_family() else "max_tokens"


def _api_style() -> str:
    """
    'openai' (default) — standard {"messages": [...]} to POST {base}/chat/completions,
        Authorization: Bearer header. Works for real OpenAI, vLLM, LiteLLM, Ollama, etc.
    'query' — the other team's actual confirmed-working server: POST {base}/chat
        with {"query": "<flattened prompt>", "model": ..., "max_tokens": ...,
        "temperature": ...}, X-API-Key header, response shaped {"answer": "..."}.
        Set LLM_API_STYLE=query in .env to use this.
    """
    return os.getenv("LLM_API_STYLE", "openai").strip().lower()


def _auth_headers() -> dict[str, str]:
    _require_llm_config()
    if _api_style() == "query":
        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["X-API-Key"] = LLM_API_KEY
        return headers
    if _provider() == "azure":
        # Azure OpenAI uses 'api-key', not 'Authorization: Bearer'
        return {"api-key": LLM_API_KEY, "Content-Type": "application/json"}
    # Standard OpenAI, vLLM, Uvicorn, LiteLLM, Ollama — all accept Bearer token.
    # vLLM ignores the key value; set LLM_API_KEY=EMPTY in .env.
    return {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }


# ==========================================
# DEFENSIVE TYPE-SAFE UTILITIES
# ==========================================
def safe_float(val) -> float:
    if val is None:
        return 0.0
    while isinstance(val, (list, tuple)):
        if not val:
            return 0.0
        val = val[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def get_nested_value(data: Any, keys: list[Any]):
    curr = data
    for key in keys:
        if isinstance(key, str) and isinstance(curr, dict):
            curr = curr.get(key)
        elif isinstance(key, int) and isinstance(curr, (list, tuple)):
            if 0 <= key < len(curr):
                curr = curr[key]
            else:
                return None
        else:
            return None
    return curr


def parse_history_message(msg) -> tuple[str | None, str | None]:
    if not msg:
        return None, None
    if isinstance(msg, dict):
        speaker = msg.get("speaker") or msg.get("role")
        text    = msg.get("text")    or msg.get("content")
        return (str(speaker) if speaker else None, str(text) if text else None)
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        return (str(msg[0]) if msg[0] else None, str(msg[1]) if msg[1] else None)
    speaker = getattr(msg, "speaker", getattr(msg, "role", None))
    text    = getattr(msg, "text",    getattr(msg, "content", None))
    return (str(speaker) if speaker else None, str(text) if text else None)


def _tokenise(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# ==========================================
# OPENAI-COMPATIBLE CHAT API
# ==========================================
async def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
    client: httpx.AsyncClient = None,
) -> str:
    _require_llm_config()
    prov = _provider()
    style = _api_style()

    if style == "query":
        # The team's actual server: POST {base}/chat, {"query": "<flat prompt>"}.
        # It has no concept of a role-based messages array, so flatten the
        # system+user messages into one plain-text prompt — same as sending
        # a single combined instruction to a text-completion-style endpoint.
        url = f"{LLM_BASE_URL.rstrip('/')}/chat"
        prompt_text = "\n\n".join(
            f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}" for m in messages
        )
        payload = {
            "query": prompt_text,
            "model": LLM_CHAT_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        active_client = client if client is not None else get_shared_client()
        try:
            response = await active_client.post(
                url, json=payload, headers=_auth_headers(),
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("[LLM] HTTP %s from query-style API — %s", exc.response.status_code, exc.response.text[:400])
            raise
        data = response.json()
        if isinstance(data, dict):
            if data.get("answer") is not None:
                return str(data["answer"]).strip()
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    if isinstance(choice.get("message"), dict):
                        return str(choice["message"].get("content", "")).strip()
                    if choice.get("text"):
                        return str(choice["text"]).strip()
        raise ValueError(f"Unable to extract LLM output from query-style response: {data}")

    _AZURE_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    if prov == "azure":
        url = (
            f"{LLM_BASE_URL.rstrip('/')}/openai/deployments/"
            f"{LLM_CHAT_MODEL}/chat/completions?api-version={_AZURE_API_VER}"
        )
    else:
        # Works for: standard OpenAI, vLLM, Uvicorn/LiteLLM, Ollama, text-generation-webui
        url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    tok_key = _max_tokens_param()
    logger.debug("[LLM] provider=%s  url=%s  model=%s  tok_param=%s", prov, url, LLM_CHAT_MODEL, tok_key)
    payload: dict = {
        "model":    LLM_CHAT_MODEL,
        "messages": messages,
        tok_key:    max_tokens,
    }
    # Newer model families (o1/o3/gpt-5.x) only accept temperature=1 (default);
    # omitting the param is the safe cross-provider approach.
    if not _is_new_model_family():
        payload["temperature"] = temperature

    active_client = client if client is not None else get_shared_client()
    try:
        response = await active_client.post(
            url,
            json=payload,
            headers=_auth_headers(),
            # NOTE: previously `timeout=30.0` (a flat float) here silently
            # overrode the shared client's fast connect-timeout
            # (httpx.Timeout(30.0, connect=4.0), set where the client is
            # created) with one flat 30s timeout applied to every phase —
            # so an unreachable LLM_BASE_URL took up to 30s to fail instead
            # of ~4-5s. Passing an explicit httpx.Timeout here keeps the
            # generous 30s allowance for slow LLM *inference* while still
            # failing fast if the server isn't even reachable.
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[LLM] HTTP %s from %s provider — %s",
            exc.response.status_code, prov, exc.response.text[:400]
        )
        raise

    data    = response.json()
    content = get_nested_value(data, ["choices", 0, "message", "content"])

    if content and isinstance(content, str):
        return content.strip()

    raise ValueError(f"Invalid chat completion response format: {data}")


async def chat_completion_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
    client: httpx.AsyncClient = None,
):
    """
    Streaming sibling of chat_completion() — yields text DELTAS as the
    local LLM generates them, using the standard OpenAI-compatible
    `stream: true` SSE contract (vLLM, Ollama, text-generation-webui, and
    real OpenAI/Azure all speak this the same way).

    LIMITATION, ON PURPOSE: the "query"-style API (LLM_API_STYLE=query —
    the team's own custom POST /chat {"query": ...} server) has no defined
    streaming contract; guessing at SSE framing for a proprietary endpoint
    we don't actually know supports it would be worse than just being
    honest about it. For that style only, this falls back to calling the
    regular non-streaming chat_completion() and yielding its result as one
    whole chunk — functionally identical to what Tier 1 already did before
    real streaming existed, so nothing regresses for that style; it simply
    doesn't gain the new benefit.

    Raises on failure, same contract as chat_completion — infra failures
    (connection refused, timeout, 5xx) propagate to the caller so
    chat_completion_with_fallback_stream's existing Tier-1-failed handling
    (circuit breaker, fallback to Gemini) keeps working unchanged.
    """
    if _api_style() == "query":
        text = await chat_completion(messages, temperature=temperature, max_tokens=max_tokens, client=client)
        yield text
        return

    _require_llm_config()
    prov = _provider()

    _AZURE_API_VER = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    if prov == "azure":
        url = (
            f"{LLM_BASE_URL.rstrip('/')}/openai/deployments/"
            f"{LLM_CHAT_MODEL}/chat/completions?api-version={_AZURE_API_VER}"
        )
    else:
        url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    tok_key = _max_tokens_param()
    payload: dict = {
        "model":    LLM_CHAT_MODEL,
        "messages": messages,
        tok_key:    max_tokens,
        "stream":   True,
    }
    if not _is_new_model_family():
        payload["temperature"] = temperature

    active_client = client if client is not None else get_shared_client()
    got_any_text = False

    try:
        async with active_client.stream(
            "POST", url, json=payload, headers=_auth_headers(),
            timeout=httpx.Timeout(30.0, connect=5.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk_raw = line[len("data:"):].strip()
                if not chunk_raw or chunk_raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(chunk_raw)
                except ValueError:
                    continue
                delta = get_nested_value(chunk, ["choices", 0, "delta", "content"])
                if delta:
                    got_any_text = True
                    yield delta
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[LLM] HTTP %s from %s provider (stream) — %s",
            exc.response.status_code, prov, exc.response.text[:400] if exc.response.text else ""
        )
        raise

    if not got_any_text:
        raise ValueError("Local LLM stream returned no text")


# ==========================================
# TIER-1 CIRCUIT BREAKER
# ==========================================
# Every /ask call was paying the full ~4-5s Local-Qwen connect-timeout
# before falling back to Gemini, on EVERY turn, because the local server
# just isn't reachable in this deployment (see the repeated
# "LLM Attempt: Local Qwen | STATUS: FAILED | Reason: ConnectTimeout"
# lines in the logs). That's dead, guaranteed-to-fail latency added to
# every single kiosk response. This breaker still honours the "always try
# local first" requirement — it tries Tier 1 normally until it sees
# _CIRCUIT_FAILURE_THRESHOLD consecutive infra failures, then "opens" and
# skips straight to Gemini for _CIRCUIT_COOLDOWN_SECONDS, after which it
# automatically tries Tier 1 again once (so a local server coming back up
# is picked up without a restart).
_CIRCUIT_FAILURE_THRESHOLD = 2
_CIRCUIT_COOLDOWN_SECONDS  = 60
_circuit_consecutive_failures = 0
_circuit_open_until            = 0.0


def _tier1_circuit_open() -> bool:
    return time.monotonic() < _circuit_open_until


def _tier1_record_success() -> None:
    global _circuit_consecutive_failures, _circuit_open_until
    _circuit_consecutive_failures = 0
    _circuit_open_until = 0.0


def _tier1_record_failure() -> None:
    global _circuit_consecutive_failures, _circuit_open_until
    _circuit_consecutive_failures += 1
    if _circuit_consecutive_failures >= _CIRCUIT_FAILURE_THRESHOLD:
        _circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS
        logger.warning(
            "[LLM CIRCUIT] Local Qwen failed %d times in a row — skipping Tier 1 "
            "for the next %ds and going straight to Gemini.",
            _circuit_consecutive_failures, _CIRCUIT_COOLDOWN_SECONDS,
        )


# ==========================================
# 3-TIER LLM FALLBACK ORCHESTRATOR
#   Tier 1: Local Qwen (always tried first — see project requirement —
#           unless the circuit breaker above has it open)
#   Tier 2: Gemini (only on genuine infra failure, not poor quality)
#   Tier 3: caller's responsibility — see generate_rag_kiosk_response's
#           RAG-only safety net, triggered when this raises
# ==========================================
def _is_infra_failure(exc: Exception) -> bool:
    """
    True only for the failure classes the spec calls out as fallback-worthy:
    connection refused/timeout, network failure, server offline, HTTP 5xx.
    False for anything else (e.g. a 4xx from a malformed request, which is
    OUR bug, not the server being down) — falling back to Gemini for that
    would just paper over a real bug instead of surfacing it. Response
    *quality* is never a reason to fall back; that's not caught here at all
    since a successful call never raises.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                         httpx.WriteTimeout, httpx.PoolTimeout, httpx.NetworkError,
                         httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    # A response that came back 200 but was empty/unparseable (ValueError from
    # chat_completion's own validation) means the local server is up but not
    # producing usable output — treated as infra-level for fallback purposes,
    # same as the spec's "API unavailable" case.
    if isinstance(exc, ValueError):
        return True
    return False


async def chat_completion_with_fallback(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> tuple[str, str, str]:
    """
    Tries local Qwen first, falls over to Gemini ONLY on genuine infra
    failure, per project spec. Returns (text, tier_label, model_used) on
    success. Raises RuntimeError if every enabled tier failed — the caller
    (generate_rag_kiosk_response) is responsible for the final Tier-3
    RAG-only safety net, since only it has the retrieved context to build
    that response from.
    """
    # ── Tier 1: Local Qwen (skipped while the circuit breaker is open) ──
    if _tier1_circuit_open():
        logger.info(
            "LLM Attempt: Local Qwen | SKIPPED | Reason: circuit breaker open "
            "(recent consecutive failures) — going straight to Gemini"
        )
    else:
        try:
            text = await chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
            logger.info("LLM Attempt: Local Qwen | STATUS: SUCCESS | MODEL USED: %s", LLM_CHAT_MODEL)
            _tier1_record_success()
            return text, "local", LLM_CHAT_MODEL
        except Exception as e:
            # %s on some httpx timeout/connect exceptions stringifies to "" with
            # no useful text at all — logging only str(e) then produced literally
            # blank "Reason: " lines with no way to tell what actually failed.
            # Including the exception type name guarantees something diagnosable
            # is always printed, even when the exception has no message body.
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            if not _is_infra_failure(e):
                # Not an infra failure (e.g. a 4xx from our own bad request) —
                # don't mask it by silently trying Gemini. Let it propagate.
                logger.error("LLM Attempt: Local Qwen | STATUS: FAILED (non-infra, not falling back) | Reason: %s", reason)
                raise
            logger.warning("LLM Attempt: Local Qwen | STATUS: FAILED | Reason: %s", reason)
            _tier1_record_failure()

    # ── Tier 2: Gemini (only reached on a genuine Tier-1 infra failure) ──
    if ENABLE_GEMINI_FALLBACK and gemini_available():
        try:
            text = await gemini_chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
            logger.info("Fallback: Gemini | STATUS: SUCCESS | MODEL USED: %s", GEMINI_MODEL)
            return text, "gemini", GEMINI_MODEL
        except Exception as e:
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error("Fallback: Gemini | STATUS: FAILED | Reason: %s", reason)
    elif ENABLE_GEMINI_FALLBACK and not gemini_available():
        logger.warning("Fallback: Gemini | SKIPPED | Reason: ENABLE_GEMINI_FALLBACK=true but GEMINI_API_KEY not set")
    else:
        logger.info("Fallback: Gemini | SKIPPED | Reason: ENABLE_GEMINI_FALLBACK is false")

    # Both tiers exhausted — caller falls back to Tier 3 (RAG-only).
    raise RuntimeError("Local Qwen and Gemini fallback both failed or unavailable.")


async def chat_completion_with_fallback_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
):
    """
    Streaming sibling of chat_completion_with_fallback(). Yields
    (text_delta, tier, model_used) tuples as they become available instead
    of returning one complete string at the end.

    Tier 1 (Local Qwen): now streams for real via chat_completion_stream
    (standard OpenAI-compatible `stream: true` SSE, which vLLM/Ollama/
    text-generation-webui/OpenAI/Azure all support) — added after
    confirming the deployment's model (Qwen2.5-7B-Instruct via the default
    "openai" API style, not the proprietary "query" style) supports it.
    The one deliberate exception: LLM_API_STYLE=query (the team's own
    custom POST /chat {"query": ...} server) has no defined streaming
    contract, so chat_completion_stream falls back to yielding the whole
    answer as one chunk for that style only — see its docstring.

    Tier 2 (Gemini): yields real incremental deltas via
    gemini_chat_completion_stream.

    EDGE CASE, ACKNOWLEDGED NOT SOLVED: if Local Qwen streams a few
    sentences successfully and THEN fails partway through (e.g. the
    connection drops mid-response), those already-yielded sentences may
    already be playing on the kiosk by the time we fall back to Gemini,
    which starts its own answer from scratch — the visitor could hear a
    few duplicate/inconsistent sentences in that specific scenario. In
    every failure actually observed in this deployment's logs, Local Qwen
    fails at connect time (ConnectTimeout) before yielding anything, so
    this doesn't come up in practice — but it's a known gap if Local Qwen
    becomes reliably reachable-but-flaky later.

    Raises RuntimeError if every enabled tier failed, same contract as the
    non-streaming version — caller still owns the Tier-3 RAG-only fallback.
    """
    if _tier1_circuit_open():
        logger.info(
            "LLM Attempt: Local Qwen | SKIPPED | Reason: circuit breaker open "
            "(recent consecutive failures) — going straight to Gemini"
        )
    else:
        try:
            any_yielded = False
            async for delta in chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens):
                any_yielded = True
                yield delta, "local", LLM_CHAT_MODEL
            if any_yielded:
                logger.info("LLM Attempt: Local Qwen (stream) | STATUS: SUCCESS | MODEL USED: %s", LLM_CHAT_MODEL)
                _tier1_record_success()
                return
            raise ValueError("Local Qwen stream yielded no text")
        except Exception as e:
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            if not _is_infra_failure(e):
                logger.error("LLM Attempt: Local Qwen | STATUS: FAILED (non-infra, not falling back) | Reason: %s", reason)
                raise
            logger.warning("LLM Attempt: Local Qwen | STATUS: FAILED | Reason: %s", reason)
            _tier1_record_failure()

    if ENABLE_GEMINI_FALLBACK and gemini_available():
        try:
            any_yielded = False
            async for delta in gemini_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens):
                any_yielded = True
                yield delta, "gemini", GEMINI_MODEL
            if any_yielded:
                logger.info("Fallback: Gemini (stream) | STATUS: SUCCESS | MODEL USED: %s", GEMINI_MODEL)
                return
            raise ValueError("Gemini stream yielded no text")
        except Exception as e:
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error("Fallback: Gemini (stream) | STATUS: FAILED | Reason: %s", reason)
    elif ENABLE_GEMINI_FALLBACK and not gemini_available():
        logger.warning("Fallback: Gemini | SKIPPED | Reason: ENABLE_GEMINI_FALLBACK=true but GEMINI_API_KEY not set")
    else:
        logger.info("Fallback: Gemini | SKIPPED | Reason: ENABLE_GEMINI_FALLBACK is false")

    raise RuntimeError("Local Qwen and Gemini fallback both failed or unavailable.")


# Sentence-boundary splitter — deliberately the same rule the frontend uses
# (WelcomeScreen.js's `raw` regex) so a sentence closed here is exactly what
# the client would have chunked it into anyway: end on . ! or ?, optionally
# followed by a closing quote, with any trailing whitespace consumed.
_SENTENCE_END_RE = re.compile(r'[^.!?]*[.!?]+["\']?\s*')


def _pop_complete_sentences(buf: str) -> tuple[list[str], str]:
    """Given text accumulated so far, split off every COMPLETE sentence at
    the front, returning (finished_sentences, remaining_incomplete_tail).
    Sentences keep their trailing whitespace (the regex's trailing \\s*)
    so that "".join()-ing them back together (generate_rag_kiosk_response's
    non-streaming path) reproduces the original spacing exactly — stripping
    it here would silently glue "sentence one.sentence two" together with
    no space between them."""
    sentences = []
    pos = 0
    for m in _SENTENCE_END_RE.finditer(buf):
        if m.end() == m.start():
            break
        piece = buf[m.start():m.end()]
        if piece.strip():
            sentences.append(piece)
        pos = m.end()
    return sentences, buf[pos:]


# ==========================================
# MEMORY-AWARE RE-ENGAGEMENT — TOPIC EXTRACTION
# ==========================================
# Turns a visitor's recent stored questions from their last session into a
# short topic summary (e.g. "hostel facilities and fees") so the greeting
# can say "last time you were asking about X" instead of repeating their
# question verbatim. Deliberately conservative: cheap, low-temperature,
# tightly bounded, and guarded on the caller side (build_greeting in
# main.py) — if this returns nothing usable, the caller MUST fall back to
# a generic re-engagement line rather than inventing a topic. This function
# never answers or elaborates, it only labels. Goes through the same
# chat_completion() used by the main pipeline, so it's subject to the same
# provider/auth configuration — no separate LLM wiring needed.
_TOPIC_EXTRACT_PROMPT = (
    "You will be given one or more visitor questions asked at a college "
    "kiosk during their last visit, oldest first. Summarize what they "
    "were asking about in 2 to 6 words (a short noun phrase covering all "
    "of them if there's more than one), e.g. 'hostel facilities', "
    "'placement statistics and fees', 'admission process'. Do not answer "
    "the question(s). Do not add punctuation, quotes, or a leading "
    "article like 'the'. If everything given is only a greeting, "
    "farewell, or too vague to label, reply with exactly: NONE.\n\n"
    "Visitor questions:\n{questions}\n"
    "Topic:"
)

async def extract_topic_label(questions) -> str | None:
    """
    Returns a short topic phrase summarizing `questions`, or None if
    extraction is empty/uncertain. Accepts either a single question
    string or a list of question strings (oldest first).
    """
    if isinstance(questions, str):
        questions = [questions]
    questions = [q.strip() for q in (questions or []) if q and q.strip()]
    if not questions:
        return None

    questions_block = "\n".join(f"- {q}" for q in questions)
    try:
        # THE FIX: this used to call the bare chat_completion() (Local Qwen
        # only, no fallback), so whenever the local server was down, topic
        # extraction failed even though the main answer pipeline was
        # successfully falling back to Gemini at that exact same moment —
        # producing a generic "no record" greeting/recall despite real
        # history existing (see get_recent_interactions succeeding right
        # above this call). Using the same 3-tier fallback the rest of the
        # app uses means a Local Qwen outage no longer silently disables
        # re-engagement.
        raw, _tier, _model = await chat_completion_with_fallback(
            messages=[{"role": "user", "content": _TOPIC_EXTRACT_PROMPT.format(questions=questions_block)}],
            temperature=0.0,
            max_tokens=16,
        )
    except Exception as e:
        logger.warning("[TOPIC EXTRACT] LLM call failed on all tiers, skipping topic label: %s", e)
        return None

    label = (raw or "").strip().strip(".\"'").strip()

    # Confidence gate — reject anything that isn't a clean short label.
    if not label or label.lower() in ("none", "unknown"):
        return None
    if len(label.split()) > 8:
        return None
    if any(ch in label for ch in ("\n", "{", "}", ":")):
        return None

    return label


# ==========================================
# RAGService INTEGRATION
# ==========================================

def _json_to_text_chunks(json_path: str) -> list[str]:
    """Convert college_info.json into readable text chunks for RAGService indexing."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[str] = []

    c = data.get("college", {})
    chunks.append(
        f"{c.get('name')} ({c.get('short_name')}) was established in {c.get('established')} "
        f"by {c.get('founder')}. It is a {c.get('type')} affiliated to {c.get('affiliation')}. "
        f"Location: {c.get('location')}. Website: {c.get('website')}. "
        f"COMEDK code: {c.get('admission_codes', {}).get('comedk')}, "
        f"CET code: {c.get('admission_codes', {}).get('cet')}."
    )

    # ── Administration: one focused chunk per fact instead of one bundled
    # paragraph, so a query like "who is the principal" has a dedicated,
    # undiluted chunk to match against instead of competing with working
    # hours / director / admissions info all mashed into the same vector. ──
    adm = data.get("administration", {})
    if adm.get("working_hours"):
        chunks.append(f"RNSIT working hours: {adm.get('working_hours')}.")
    director = adm.get("director", {})
    if director.get("name"):
        chunks.append(f"RNSIT Director: {director.get('name')}. Contact: {director.get('phone')}.")
    principal = adm.get("principal", {})
    if principal.get("name"):
        chunks.append(f"RNSIT Principal: {principal.get('name')}. Contact: {principal.get('phone')}.")
    contacts = adm.get("contacts", {})
    if contacts.get("admissions_phone") or contacts.get("admissions_email"):
        chunks.append(
            f"RNSIT admissions phone: {contacts.get('admissions_phone')}. "
            f"Admissions email: {contacts.get('admissions_email')}."
        )
    enquiry = contacts.get("enquiry_phone", [])
    if enquiry:
        chunks.append(f"RNSIT general enquiry numbers: {', '.join(enquiry)}.")

    for code, dept in data.get("departments", {}).items():
        parts = [f"Department: {dept.get('name')} ({code.upper()})."]
        if dept.get("block"):
            parts.append(f"Located in: {dept['block']}.")
        if dept.get("hod"):
            parts.append(f"Head of Department: {dept['hod']}.")
        if dept.get("intake"):
            parts.append(f"Annual intake: {dept['intake']} students.")
        if dept.get("phd_center"):
            parts.append("Has a PhD research centre.")
        chunks.append(" ".join(parts))

    for fname, fdet in data.get("facilities", {}).items():
        if isinstance(fdet, dict):
            label = fname.replace("_", " ").title()
            parts = [f"Facility: {label}."]
            for key in ("name", "location", "timings", "details", "capacity", "platform"):
                if fdet.get(key):
                    parts.append(f"{key.title()}: {fdet[key]}.")
            chunks.append(" ".join(parts))
        elif isinstance(fdet, str):
            chunks.append(f"Facility {fname.replace('_', ' ').title()}: {fdet}.")

    pl = data.get("placements", {})
    if pl:
        parts = [f"RNSIT Placements:"]
        if pl.get("total_companies"):
            parts.append(f"{pl['total_companies']} companies recruit from RNSIT.")
        if pl.get("recent_recruiters"):
            parts.append(f"Recent recruiters: {', '.join(pl['recent_recruiters'][:15])}.")
        chunks.append(" ".join(parts))

        for yr, stats in pl.get("stats", {}).items():
            s_parts = [f"Placement stats {yr}:"]
            if stats.get("highest_ctc_lpa"):
                s_parts.append(f"Highest package: {stats['highest_ctc_lpa']} LPA.")
            if stats.get("average_ctc_lpa"):
                s_parts.append(f"Average package: {stats['average_ctc_lpa']} LPA.")
            if stats.get("students_placed"):
                s_parts.append(f"Students placed: {stats['students_placed']}.")
            chunks.append(" ".join(s_parts))

    # ── Admissions: dedicated builder (not the generic key:val loop below) ──
    # because admissions now holds nested dicts/lists (eligibility per
    # course, process steps, documents, fee/scholarship disclaimers) that
    # the generic loop can't turn into natural sentences. One topic per
    # chunk, same pattern as departments/placements above.
    adms = data.get("admissions", {})
    if adms:
        headline = [f"RNSIT admissions status: {adms.get('status', 'contact the admissions office for current status')}."]
        if adms.get("academic_year"):
            headline.append(f"Academic year: {adms['academic_year']}.")
        if adms.get("contact_phone") or adms.get("contact_email"):
            headline.append(f"Admissions contact: phone {adms.get('contact_phone', 'N/A')}, "
                             f"email {adms.get('contact_email', 'N/A')}.")
        if adms.get("cet_code"):
            headline.append(f"KCET institute code: {adms['cet_code']}.")
        if adms.get("comedk_code"):
            headline.append(f"COMEDK institute code: {adms['comedk_code']}.")
        chunks.append(" ".join(headline))

        modes = adms.get("modes", {})
        if isinstance(modes, dict):
            for course, exams in modes.items():
                if exams:
                    label = course.replace("_", " ").upper()
                    chunks.append(f"RNSIT admission route for {label}: {', '.join(exams)}.")
        elif isinstance(modes, list) and modes:
            chunks.append(f"RNSIT admission modes: {', '.join(modes)}.")

        elig = adms.get("eligibility", {})
        if isinstance(elig, dict):
            for course, text in elig.items():
                if isinstance(text, str) and text.strip():
                    label = course.replace("_", " ").upper()
                    chunks.append(f"RNSIT eligibility for {label}: {text}")

        steps = adms.get("process_steps", [])
        if steps:
            numbered = " ".join(f"({i+1}) {s}" for i, s in enumerate(steps))
            chunks.append(f"RNSIT admission process: {numbered}")

        docs = adms.get("documents_required", [])
        if docs:
            chunks.append("Documents required for RNSIT admission: " + "; ".join(docs) + ".")

        fees = adms.get("fees", {})
        if isinstance(fees, dict):
            if fees.get("quota_types"):
                chunks.append(f"RNSIT admission quota types: {', '.join(fees['quota_types'])}.")
            if fees.get("note"):
                chunks.append(f"RNSIT fee information: {fees['note']}")

        schol = adms.get("scholarships", {})
        if isinstance(schol, dict):
            if schol.get("general"):
                chunks.append("RNSIT scholarship options: " + "; ".join(schol["general"]) + ".")
            if schol.get("note"):
                chunks.append(f"RNSIT scholarship note: {schol['note']}")

    for section, content in data.items():
        if section in ("meta", "college", "administration", "departments",
                        "facilities", "placements", "admissions", "faqs"):
            continue
        if isinstance(content, dict):
            for key, val in content.items():
                if isinstance(val, str) and val.strip():
                    chunks.append(f"{section.replace('_', ' ').title()} — {key}: {val}")
                elif isinstance(val, list):
                    chunks.append(f"{section.replace('_', ' ').title()} — {key}: {', '.join(str(v) for v in val)}")
        elif isinstance(content, str):
            chunks.append(f"{section.replace('_', ' ').title()}: {content}")

    # ── FAQs: these are hand-written, single-topic Q&A pairs — the best
    # possible chunks for direct-match retrieval. Previously silently
    # dropped by the generic loop above (a list-of-dicts matches neither
    # the isinstance(content, dict) nor isinstance(content, str) branch),
    # so none of the 40+ FAQ entries ever reached the vector store. ──
    #
    # EXCLUDE conversational entries (greetings/farewells/thanks). These are
    # conversation mechanics, not campus facts — they don't belong in a
    # fact-retrieval index. Leaving them in causes two real bugs: (1) they
    # can win the similarity search for vague/short queries and get echoed
    # back as if they were the answer, and (2) they're now handled by
    # deterministic GREETING_PHRASES/THANK_YOU_PHRASES checks in main.py
    # instead, so indexing them here would just create a second,
    # competing (and inconsistent) path to the same behaviour.
    _CONVERSATIONAL_FAQ_QUESTIONS = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "thank you", "thanks", "bye", "goodbye",
    }
    skipped = 0
    for faq in data.get("faqs", []):
        q = (faq.get("question") or "").strip()
        a = (faq.get("answer") or "").strip()
        if not (q and a):
            continue
        if q.lower().strip(" ?!.") in _CONVERSATIONAL_FAQ_QUESTIONS:
            skipped += 1
            continue
        chunks.append(f"Q: {q} A: {a}")
    if skipped:
        logger.info("[RAG SEED] Skipped %d conversational FAQ entries (handled deterministically, not indexed).", skipped)

    return [c.strip() for c in chunks if c.strip()]


async def initialize_rag_knowledge_base():
    """Seed the RAGService collection from college_info.json if the collection is empty."""
    global _rag_seeded
    if _rag_seeded:
        return

    client = get_shared_client()
    try:
        # Check whether collection already has data
        resp = await client.get(f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}",
                                timeout=5.0)
        if resp.status_code == 200:
            stats = resp.json()
            if stats.get("total_chunks", 0) > 0:
                logger.info(
                    "[RAG] Collection '%s' already has %d chunks — skipping seed.",
                    RAG_COLLECTION, stats["total_chunks"]
                )
                _rag_seeded = True
                return
        # Collection missing or empty — seed from JSON
        chunks = _json_to_text_chunks(JSON_PATH)
        if not chunks:
            logger.warning("[RAG] college_info.json not found or empty — nothing to seed.")
            _rag_seeded = True
            return

        logger.info("[RAG] Seeding collection '%s' with %d chunks from college_info.json …",
                    RAG_COLLECTION, len(chunks))
        added = 0
        for chunk in chunks:
            r = await client.post(
                f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/index/text",
                json={"text": chunk, "source": "college_info.json"},
                timeout=15.0,
            )
            if r.status_code == 200:
                added += r.json().get("added", 0)

        logger.info("[RAG] Seed complete — %d chunks indexed into '%s'.", added, RAG_COLLECTION)
        _rag_seeded = True

    except Exception as e:
        logger.warning("[RAG] RAGService unreachable during init (%s). Will retry on next query.", e)


async def retrieve_relevant_context(user_query: str, top_k: int = None) -> tuple[str, float, list[dict]]:
    """
    Semantic search via RAGService.
    Returns (context_text, best_score, raw_results). raw_results is the
    ranked list of {text, score, metadata} from RAGService, kept around so
    callers can log exactly what was retrieved (see Step-15 debug logging
    in generate_rag_kiosk_response) — this is what lets us actually PROVE
    what the LLM was given, instead of just trusting it happened.
    Falls back to empty context on error.
    """
    k = top_k or RAG_TOP_K
    client = get_shared_client()
    try:
        resp = await client.post(
            f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/search",
            json={"query": user_query, "k": k},
            timeout=10.0,
        )
        resp.raise_for_status()
        results = resp.json()   # list of {text, score, metadata}
        if not results:
            return "No relevant context found.", 0.0, []

        best_score = max(safe_float(r.get("score", 0)) for r in results)
        context = "\n\n".join(r["text"] for r in results if r.get("text"))
        return context, best_score, results

    except Exception as e:
        logger.warning("[RAG] Search failed: %s", e)
        return "No context available.", 0.0, []


# ==========================================
# CONTEXTUAL QUERY CONDENSER
# ==========================================
async def condense_query(question: str, history: list | None) -> str:
    q_lower = question.lower().strip()

    if not history:
        return question

    bypass_keywords = {"where is", "location", "timing", "hours", "who is", "what is", "address"}
    if any(keyword in q_lower for keyword in bypass_keywords):
        return question

    if len(question.split()) > 3:
        return question

    history_lines = []
    for msg in history[-3:]:
        speaker_val, text_val = parse_history_message(msg)
        if speaker_val and text_val:
            speaker = "Visitor" if speaker_val.lower() in ("visitor", "user") else "Kiosk"
            history_lines.append(f"{speaker}: {text_val}")

    history_context = "\n".join(history_lines)
    condense_prompt = (
        "Given the following conversation history and a short follow-up question, rewrite "
        "the follow-up into a single, standalone search query for a database. "
        "Do not answer the question. Return only the rewritten query.\n\n"
        f"History:\n{history_context}\n\n"
        f"Follow-up question: {question}"
    )

    try:
        rewritten = await chat_completion(
            messages=[
                {"role": "system",
                 "content": "You rewrite short follow-up questions into standalone retrieval queries only."},
                {"role": "user", "content": condense_prompt}
            ],
            temperature=0.0,
            max_tokens=60,
        )
        if rewritten and isinstance(rewritten, str):
            rewritten = rewritten.strip().strip('"')
            if rewritten:
                logger.info("[CONTEXT REWRITER] '%s' -> '%s'", question, rewritten)
                return rewritten
    except Exception as e:
        logger.warning("[CONTEXT REWRITER] Failed, falling back to original query: %s", e)

    return question


# ==========================================
# RAG RESPONSE GENERATION
# ==========================================
async def generate_rag_kiosk_response_stream(question: str, history: list = None):
    """
    Streaming sibling of generate_rag_kiosk_response() — yields the answer
    SENTENCE BY SENTENCE as it's generated, instead of the whole thing at
    once. This is the single source of truth for the RAG/LLM pipeline;
    generate_rag_kiosk_response() below just joins this generator's output,
    so the streaming and non-streaming paths can never drift apart in
    behavior — same routing, same prompt, same fallbacks, same logging.

    All the deterministic pre-RAG routes (profanity guard, weather/traffic
    live-info, off-topic) resolve to a single complete string quickly
    anyway, so those are yielded as ONE chunk — there's nothing to gain by
    streaming something that's already fast. Only the RNSIT_RAG / LLM-
    generation path (the slow one: RAG search + Local Qwen/Gemini) streams
    for real, sentence by sentence, as chat_completion_with_fallback_stream
    produces text.
    """
    words = question.lower().split()
    if any(bad_word in words for bad_word in PROFANITY_BLOCKLIST):
        logger.warning("[SAFETY TRIGGERED] Blocked inappropriate query words.")
        yield (
            "Let's keep our conversation respectful! I am the official RNSIT kiosk guide. "
            "How can I assist you politely with campus layouts, departments, or admissions today?"
        )
        return

    await initialize_rag_knowledge_base()

    weather_answer = await _try_fetch_weather(question)
    if weather_answer:
        logger.info("=" * 60)
        logger.info("TRANSCRIPT: %r", question)
        logger.info("ROUTE: LIVE_INFO (weather, pre-RAG)")
        logger.info("LLM CALLED: NO")
        logger.info("FINAL RESPONSE: %r", weather_answer)
        logger.info("=" * 60)
        yield weather_answer
        return

    traffic_answer = await _try_fetch_traffic(question)
    if traffic_answer:
        logger.info("=" * 60)
        logger.info("TRANSCRIPT: %r", question)
        logger.info("ROUTE: LIVE_INFO (traffic, pre-RAG)")
        logger.info("LLM CALLED: NO")
        logger.info("FINAL RESPONSE: %r", traffic_answer)
        logger.info("=" * 60)
        yield traffic_answer
        return

    search_query = await condense_query(question, history or [])
    context_text, max_score, raw_results = await retrieve_relevant_context(search_query)
    max_score = safe_float(max_score)

    logger.info("=" * 60)
    logger.info("TRANSCRIPT: %r", question)
    logger.info("NORMALIZED QUERY: %r", search_query)
    for i, r in enumerate(raw_results[:5], start=1):
        preview = (r.get("text") or "")[:80].replace("\n", " ")
        logger.info("  RETRIEVED #%d — score=%.3f — %s...", i, safe_float(r.get("score", 0)), preview)
    logger.info("THRESHOLD: %.2f | TOP SCORE: %.4f", RAG_SIMILARITY_THRESHOLD, max_score)

    if max_score < RAG_SIMILARITY_THRESHOLD:
        route, answer = await _handle_offtopic(question, history or [])
        logger.info("ROUTE: %s", route)
        logger.info("LLM CALLED: %s", "YES" if route != "RNSIT_UNKNOWN" else "NO")
        logger.info("FINAL RESPONSE: %r", answer)
        logger.info("=" * 60)
        yield answer
        return

    logger.info("ROUTE: RNSIT_RAG")
    logger.info("LLM CALLED: YES | MODEL: %s", LLM_CHAT_MODEL)

    system_prompt = (
        "You are Nova, the official AI Digital Receptionist for RNS Institute of Technology (RNSIT), Bengaluru.\n"
        "Your name is Nova. NEVER call yourself by the visitor's name or any name other than Nova. The person speaking to you is a visitor, and you are Nova.\n"
        "Your workspace is a public campus kiosk visible to parents, children, and students. "
        "Your tone must remain completely child-safe, welcoming, polite, and professional at all times.\n\n"
        f"Use ONLY the following verified campus facts to answer the visitor:\n\n"
        f"{context_text}\n\n"
        "CRITICAL RESPONSE CONSTRAINTS:\n"
        "1. Your name is Nova. If asked who you are or what your name is, always say you are Nova.\n"
        "2. Rely only on the facts provided above. If the context does not contain the answer, "
        "say: 'I don't have that detail — please visit the Admin Block or call our admissions desk.'\n"
        "3. Keep responses snappy and punchy (2-3 sentences maximum). Avoid long paragraphs.\n"
        "4. Do not answer out-of-domain questions (politics, celebrities, general trivia). "
        "Guide them back to college topics.\n"
        "5. COMPARISON / RANKING QUESTIONS (e.g. 'which department is best for placements'): "
        "the placement figures you have are INSTITUTE-WIDE totals, not broken down per department. "
        "Do NOT invent a per-department ranking or imply one department outperforms another unless "
        "the context above explicitly states department-specific figures. If asked to compare "
        "departments and you only have overall numbers, say so plainly — e.g. 'I only have "
        "placement numbers for RNSIT overall, not broken down by department, so I can't say which "
        "is best — but I can tell you the departments and overall placement stats we do have.' "
        "Never present a guess as if it were verified data.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-4:]:
            speaker_val, text_val = parse_history_message(msg)
            if speaker_val and text_val:
                role = "user" if speaker_val.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text_val})
    messages.append({"role": "user", "content": question})

    buf = ""
    full_answer_parts: list[str] = []
    tier_seen, model_seen = "", ""
    try:
        async for delta, tier, model_used in chat_completion_with_fallback_stream(
            messages=messages, temperature=0.2, max_tokens=180,
        ):
            tier_seen, model_seen = tier, model_used
            buf += delta
            ready, buf = _pop_complete_sentences(buf)
            for s in ready:
                full_answer_parts.append(s)
                yield s
        if buf.strip():
            full_answer_parts.append(buf)
            yield buf
        full_answer = "".join(full_answer_parts).strip()
        if not full_answer:
            full_answer = "I am having trouble formatting the response. Please try again."
            yield full_answer
        logger.info("FINAL RESPONSE: %r (tier=%s, model=%s)", full_answer, tier_seen, model_seen)
        logger.info("=" * 60)

    except Exception as e:
        # ── Tier 3: both Local Qwen and Gemini failed — return the top
        # retrieved RAG fact directly instead of an internal error.
        logger.error("Local Qwen: FAILED")
        logger.error("Gemini: FAILED (or disabled) — %s", e)
        logger.info("Returning: Top RAG Answer")
        top_fact = raw_results[0].get("text", "") if raw_results else ""
        top_fact = _QA_LABEL_RE_LLM.sub("", top_fact).strip()
        top_fact = _FACILITY_LABEL_RE_LLM.sub("", top_fact).strip()
        if top_fact:
            answer = (
                "I am currently unable to generate a conversational response, "
                f"but based on the available RNSIT information: {top_fact}"
            )
        else:
            answer = "The kiosk AI engine is currently experiencing connectivity issues. Please visit the Admin Block."
        logger.info("FINAL RESPONSE: %r (tier=rag_only)", answer)
        logger.info("=" * 60)
        yield answer


async def generate_rag_kiosk_response(question: str, history: list = None) -> str:
    """Non-streaming entry point — kept for callers that just want the full
    string (e.g. anything hitting Redis cache logic before TTS). Internally
    just joins generate_rag_kiosk_response_stream()'s output, so this can
    never drift out of sync with the streaming path — one implementation,
    two ways to consume it."""
    parts = [s async for s in generate_rag_kiosk_response_stream(question, history)]
    return "".join(parts).strip()


# ==========================================
# OFF-TOPIC ROUTER — GENERAL_LLM / LIVE_INFO / UNSUPPORTED_EXTERNAL / RNSIT_UNKNOWN
# ==========================================
# Only runs when RAG scored below threshold, so it costs nothing on the
# common case (a real RNSIT question that retrieval already answered).
# One cheap LLM call classifies + drafts a response in a single round trip
# instead of a large keyword/if-else tree.
_OFFTOPIC_ROUTER_PROMPT = (
    "You are Nova, the AI voice receptionist behind the RNSIT campus kiosk. Your name is always Nova.\n"
    "The visitor's question did not match anything in the RNSIT knowledge base. "
    "Classify it into exactly one category and reply with ONLY that category word "
    "on the first line, then (if GENERAL_LLM) a short 1-2 sentence helpful answer "
    "on the second line. If you introduce yourself or answer small talk in GENERAL_LLM, you are Nova (never call yourself by the visitor's name).\n"
    "Categories:\n"
    "GENERAL_LLM — harmless general-knowledge or small-talk question you can answer "
    "yourself (e.g. 'what is machine learning', 'how are you', 'who are you').\n"
    "LIVE_INFO — needs real-time/current data you cannot know (weather, today's date, "
    "live traffic, current news).\n"
    "UNSUPPORTED_EXTERNAL — asks a specific factual question about a DIFFERENT "
    "institution/company/person you have no verified data on.\n"
    "RNSIT_UNKNOWN — seems to be about RNSIT but you have no matching verified fact.\n\n"
    f"Visitor question: {{question}}"
)


async def _try_fetch_weather(question: str) -> str | None:
    """
    Real weather integration for LIVE_INFO — Open-Meteo (no API key needed).
    Only fires for questions that actually look weather-related; returns None
    for other LIVE_INFO questions (news/traffic/etc.) so the caller falls back
    to an honest "I don't have that" message instead of guessing.
    Coordinates are RNSIT's campus location (R R Nagar, Bengaluru).
    """
    q = question.lower()
    if not any(w in q for w in ("weather", "rain", "raining", "temperature", "hot", "cold", "climate", "sunny", "humid")):
        return None

    _WMO_CODES = {
        0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "light showers",
        81: "showers", 82: "heavy showers", 95: "thunderstorm", 96: "thunderstorm with hail",
        99: "severe thunderstorm with hail",
    }
    try:
        client = get_shared_client()
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": 12.9081, "longitude": 77.5218,
                    "current_weather": "true", "timezone": "Asia/Kolkata"},
            timeout=httpx.Timeout(8.0, connect=4.0),
        )
        resp.raise_for_status()
        cw = resp.json().get("current_weather", {})
        temp = cw.get("temperature")
        code = cw.get("weathercode")
        if temp is None:
            return None
        desc = _WMO_CODES.get(code, "")
        desc_part = f" and {desc}" if desc else ""
        return f"It's currently around {temp}°C{desc_part} here at RNSIT in Bengaluru."
    except Exception as e:
        logger.warning("[LIVE_INFO] Weather fetch failed: %s", e)
        return None


async def _try_fetch_traffic(question: str) -> str | None:
    """
    Real traffic integration for LIVE_INFO — TomTom Traffic Flow API.
    Unlike weather, traffic isn't free-and-keyless anywhere reputable, so this
    requires TOMTOM_API_KEY in .env (free tier, no cost, just a signup at
    developer.tomtom.com). Only fires for traffic-looking questions; if no key
    is configured, returns None so the caller falls back to an honest message
    instead of silently doing nothing or guessing.
    Reports live road-flow conditions at a fixed point on RNSIT's approach
    road (Dr. Vishnuvardhan Road) rather than route-based ETA, since the
    visitor is already at the kiosk — there's no "origin" to route from.
    """
    q = question.lower()
    if not any(w in q for w in ("traffic", "congestion", "jam", "road condition", "how's the road")):
        return None

    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = get_shared_client()
        resp = await client.get(
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
            params={"point": "12.9081,77.5218", "key": api_key},
            timeout=httpx.Timeout(8.0, connect=4.0),
        )
        resp.raise_for_status()
        seg = resp.json().get("flowSegmentData", {})
        current = seg.get("currentSpeed")
        free_flow = seg.get("freeFlowSpeed")
        if not current or not free_flow:
            return None
        ratio = current / free_flow
        if ratio >= 0.8:
            desc = "flowing smoothly"
        elif ratio >= 0.5:
            desc = "moderately busy"
        else:
            desc = "quite congested"
        return f"Traffic on the road near RNSIT is currently {desc}."
    except Exception as e:
        logger.warning("[LIVE_INFO] Traffic fetch failed: %s", e)
        return None


async def _handle_offtopic(question: str, history: list) -> tuple[str, str]:
    try:
        raw, _tier, _model = await chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": "You are a strict single-word classifier plus optional short answer generator."},
                {"role": "user", "content": _OFFTOPIC_ROUTER_PROMPT.format(question=question)},
            ],
            temperature=0.2,
            max_tokens=120,
        )
    except Exception as e:
        logger.warning("[ROUTER] Classification call failed on all tiers (%s) — defaulting to RNSIT_UNKNOWN.", e)
        raw = "RNSIT_UNKNOWN"

    lines = (raw or "").strip().splitlines()
    route = (lines[0].strip().upper() if lines else "RNSIT_UNKNOWN")
    if route not in ("GENERAL_LLM", "LIVE_INFO", "UNSUPPORTED_EXTERNAL", "RNSIT_UNKNOWN"):
        route = "RNSIT_UNKNOWN"

    if route == "GENERAL_LLM" and len(lines) > 1 and lines[1].strip():
        return route, lines[1].strip()

    if route == "LIVE_INFO":
        weather_answer = await _try_fetch_weather(question)
        if weather_answer:
            return route, weather_answer
        traffic_answer = await _try_fetch_traffic(question)
        if traffic_answer:
            return route, traffic_answer
        return route, (
            "I don't have access to that kind of live information right now. "
            "For campus-related timings or schedules, I'm happy to help!"
        )

    if route == "UNSUPPORTED_EXTERNAL":
        variants = [
            "I currently have detailed information about RNSIT, so I don't have reliable information about that institution.",
            "I'm specialized in RNSIT information and don't have verified details on that institution — sorry!",
            "That's outside what I have verified data on — I can only speak confidently about RNSIT.",
        ]
        return route, variants[hash(question) % len(variants)]

    # RNSIT_UNKNOWN — plausibly about RNSIT but nothing in the knowledge base
    return "RNSIT_UNKNOWN", (
        "I don't have that detail on hand — please check with the Admin Block "
        "or call our admissions desk, and I'll be able to help with most other "
        "RNSIT questions!"
    )