import os
import re
import json
import asyncio
import logging
from typing import Any

import httpx
from dotenv import load_dotenv


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


# ==========================================
# SHARED HTTP CLIENT
# ==========================================
def get_shared_client() -> httpx.AsyncClient:
    global _SHARED_ASYNC_CLIENT
    if _SHARED_ASYNC_CLIENT is None or _SHARED_ASYNC_CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        _SHARED_ASYNC_CLIENT = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(30.0, connect=10.0),
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
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not configured.")


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


def _auth_headers() -> dict[str, str]:
    _require_llm_config()
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
            timeout=30.0,
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

    adm = data.get("administration", {})
    chunks.append(
        f"RNSIT working hours: {adm.get('working_hours')}. "
        f"Director: {adm.get('director', {}).get('name')} ({adm.get('director', {}).get('phone')}). "
        f"Principal: {adm.get('principal', {}).get('name')} ({adm.get('principal', {}).get('phone')}). "
        f"Admissions phone: {adm.get('contacts', {}).get('admissions_phone')}. "
        f"Admissions email: {adm.get('contacts', {}).get('admissions_email')}."
    )
    enquiry = adm.get("contacts", {}).get("enquiry_phone", [])
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

    for section, content in data.items():
        if section in ("meta", "college", "administration", "departments", "facilities", "placements"):
            continue
        if isinstance(content, dict):
            for key, val in content.items():
                if isinstance(val, str) and val.strip():
                    chunks.append(f"{section.replace('_', ' ').title()} — {key}: {val}")
                elif isinstance(val, list):
                    chunks.append(f"{section.replace('_', ' ').title()} — {key}: {', '.join(str(v) for v in val)}")
        elif isinstance(content, str):
            chunks.append(f"{section.replace('_', ' ').title()}: {content}")

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


async def retrieve_relevant_context(user_query: str, top_k: int = None) -> tuple[str, float]:
    """
    Semantic search via RAGService.
    Returns (context_text, best_score).  Falls back to empty string on error.
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
            return "No relevant context found.", 0.0

        best_score = max(safe_float(r.get("score", 0)) for r in results)
        context = "\n\n".join(r["text"] for r in results if r.get("text"))
        return context, best_score

    except Exception as e:
        logger.warning("[RAG] Search failed: %s", e)
        return "No context available.", 0.0


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
async def generate_rag_kiosk_response(question: str, history: list = None) -> str:
    words = question.lower().split()
    if any(bad_word in words for bad_word in PROFANITY_BLOCKLIST):
        logger.warning("[SAFETY TRIGGERED] Blocked inappropriate query words.")
        return (
            "Let's keep our conversation respectful! I am the official RNSIT kiosk guide. "
            "How can I assist you politely with campus layouts, departments, or admissions today?"
        )

    # Ensure RAGService collection is seeded (no-op after first call)
    await initialize_rag_knowledge_base()

    search_query            = await condense_query(question, history or [])
    context_text, max_score = await retrieve_relevant_context(search_query)

    max_score = safe_float(max_score)
    logger.info("[RAG] query='%s' | top_score=%.4f | threshold=%.2f",
                search_query, max_score, RAG_SIMILARITY_THRESHOLD)

    if max_score < RAG_SIMILARITY_THRESHOLD:
        logger.info("[RAG] Score below threshold — off-topic guard triggered.")
        return (
            "I am the RNSIT Campus Kiosk virtual assistant. I can help you with "
            "campus directions, departments, fees, placements, and administrative queries. "
            "Please ask a campus-related question!"
        )

    system_prompt = (
        "You are the official AI Digital Receptionist for RNS Institute of Technology (RNSIT), Bengaluru.\n"
        "Your workspace is a public campus kiosk visible to parents, children, and students. "
        "Your tone must remain completely child-safe, welcoming, polite, and professional at all times.\n\n"
        f"Use ONLY the following verified campus facts to answer the visitor:\n\n"
        f"{context_text}\n\n"
        "CRITICAL RESPONSE CONSTRAINTS:\n"
        "1. Rely only on the facts provided above. If the context does not contain the answer, "
        "say: 'I don't have that detail — please visit the Admin Block or call our admissions desk.'\n"
        "2. Keep responses snappy and punchy (2-3 sentences maximum). Avoid long paragraphs.\n"
        "3. Do not answer out-of-domain questions (politics, celebrities, general trivia). "
        "Guide them back to college topics.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-4:]:
            speaker_val, text_val = parse_history_message(msg)
            if speaker_val and text_val:
                role = "user" if speaker_val.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text_val})

    messages.append({"role": "user", "content": question})

    try:
        text_out = await chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=180,
        )
        return text_out.strip() if text_out else "I am having trouble formatting the response. Please try again."

    except httpx.HTTPStatusError as e:
        logger.error("[LLM API] Status %s: %s", e.response.status_code, e.response.text)
        return "I am having trouble accessing my AI engine. Please try again in a moment."
    except Exception as e:
        logger.exception("[LLM API] Connection failure: %s", e)
        return "The kiosk AI engine is currently experiencing connectivity issues. Please visit the Admin Block."
