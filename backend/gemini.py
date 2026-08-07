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
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")   # fast + cheap: right choice for a fallback tier
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


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