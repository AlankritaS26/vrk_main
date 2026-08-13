"""
gemini.py — thin, self-contained Gemini client used ONLY as Tier-2 fallback
when the local Qwen LLM is genuinely unavailable (connection refused,
timeout, or 5xx). Never called as a primary path.

Kept deliberately separate from llm.py so Gemini-specific request/response
shaping never leaks into the main RAG/LLM pipeline — per the project
requirement to keep this modular and easy to swap for another provider
later (OpenAI, Claude, Ollama, vLLM, etc.).
"""
from __future__ import annotations

import os
import json
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")   # fast + cheap: right choice for a fallback tier
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_STREAM_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"


def gemini_available() -> bool:
    """Cheap check callers use before attempting a fallback call."""
    return bool(GEMINI_API_KEY)


def _to_gemini_payload(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict:
    """
    Convert the standard [{"role": "system"/"user"/"assistant", "content": ...}]
    shape (identical to what llm.chat_completion expects) into Gemini's own
    systemInstruction + contents format.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    system_instruction = "\n\n".join(system_parts) if system_parts else None

    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": m.get("content", "")}],
        })

    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return payload


async def gemini_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    """
    Same call shape as llm.chat_completion() — a drop-in alternate used by
    chat_completion_with_fallback(). Raises on any failure; the caller
    (llm.py) is responsible for treating that as "both tiers failed" and
    moving on to the Tier-3 RAG-only last resort.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    payload = _to_gemini_payload(messages, temperature, max_tokens)

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
        resp = await client.post(GEMINI_URL, params={"key": GEMINI_API_KEY}, json=payload)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        # Most common real cause: the prompt tripped Gemini's safety filters.
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        raise ValueError(f"Gemini returned no candidates (blockReason={block_reason}): {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise ValueError(f"Gemini returned empty text: {data}")
    return text


async def gemini_chat_completion_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
):
    """
    Streaming sibling of gemini_chat_completion() — yields text DELTAS
    (small pieces, not complete sentences; the caller buffers those into
    sentences) as Gemini generates them, instead of waiting for the whole
    reply. This is what lets speech start on the first sentence instead of
    after the full 1-9s (sometimes more) generation completes.

    Uses Gemini's :streamGenerateContent endpoint with alt=sse, which
    returns a standard Server-Sent-Events stream of
    `data: {...same candidates/parts shape as the non-streaming response...}`
    lines, one per incremental chunk.

    Raises on any failure, exactly like gemini_chat_completion — the caller
    (llm.py) treats that as "Gemini tier failed" the same way either way.
    A failure partway through (after some text was already yielded) is
    still raised; the caller decides what, if anything, to do with the
    partial text it already has.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    payload = _to_gemini_payload(messages, temperature, max_tokens)
    got_any_text = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        async with client.stream(
            "POST", GEMINI_STREAM_URL,
            params={"key": GEMINI_API_KEY, "alt": "sse"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk_raw = line[len("data:"):].strip()
                if not chunk_raw or chunk_raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(chunk_raw)
                except ValueError:
                    continue
                candidates = chunk.get("candidates", [])
                if not candidates:
                    block_reason = chunk.get("promptFeedback", {}).get("blockReason")
                    if block_reason:
                        raise ValueError(f"Gemini blocked the prompt (blockReason={block_reason})")
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                delta = "".join(p.get("text", "") for p in parts)
                if delta:
                    got_any_text = True
                    yield delta

    if not got_any_text:
        raise ValueError("Gemini stream returned no text")