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
# CONFIGURATION & GLOBAL KNOWLEDGE CACHE
# ==========================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "").strip().strip("'\"")

LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Keyword-overlap threshold (0-1).  Replaces the old cosine-similarity threshold.
# A score of 0.15 means at least ~15 % of the query's meaningful words must
# appear in the best-matching chunk — keeps off-topic questions filtered out.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))

JSON_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "college_info.json")

KNOWLEDGE_CHUNKS: list[str] = []

PROFANITY_BLOCKLIST = {
    "badword1", "badword2", "abuse", "stupid", "idiot"
}

# Common English stop-words to skip during keyword matching
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


def _auth_headers() -> dict[str, str]:
    _require_llm_config()
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


# ==========================================
# KEYWORD TOKENISER (no external deps)
# ==========================================
def _tokenise(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stop-words. Returns a set of stems."""
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

    url     = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model":       LLM_CHAT_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    active_client = client if client is not None else get_shared_client()
    response = await active_client.post(
        url,
        json=payload,
        headers=_auth_headers(),
        timeout=30.0,
    )
    response.raise_for_status()

    data    = response.json()
    content = get_nested_value(data, ["choices", 0, "message", "content"])

    if content and isinstance(content, str):
        return content.strip()

    raise ValueError(f"Invalid chat completion response format: {data}")


# ==========================================
# DATA INGESTION  (chunks only, no vectors)
# ==========================================
async def initialize_rag_knowledge_base():
    global KNOWLEDGE_CHUNKS

    if KNOWLEDGE_CHUNKS:
        return

    if not os.path.exists(JSON_PATH):
        logger.warning(f"[RAG] Knowledge base file not found at: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = []

    c_info = data.get("college", {})
    chunks.append(
        f"{c_info.get('name')} ({c_info.get('short_name')}) was established in {c_info.get('established')} "
        f"by founder {c_info.get('founder')}. It is an autonomous {c_info.get('type')} located at {c_info.get('location')}."
    )

    admin = data.get("administration", {})
    chunks.append(f"The college working hours are {admin.get('working_hours')}.")
    chunks.append(
        f"The Director of RNSIT is {admin.get('director', {}).get('name')}. "
        f"Contact phone: {admin.get('director', {}).get('phone')}."
    )
    chunks.append(
        f"The Principal of RNSIT is {admin.get('principal', {}).get('name')}. "
        f"Contact phone: {admin.get('principal', {}).get('phone')}."
    )

    for code, details in data.get("departments", {}).items():
        chunks.append(
            f"The department of {details.get('name')} ({code.upper()}) is located in the "
            f"{details.get('block', 'Main campus block')}. The HOD is "
            f"{details.get('hod', 'the appointed department head')} and intake capacity is "
            f"{details.get('intake', '180')} students per year."
        )

    for name, details in data.get("facilities", {}).items():
        if isinstance(details, dict):
            chunks.append(
                f"Facility: {name.replace('_', ' ').title()}. "
                f"Details: {details.get('name', '')} {details.get('location', '')} "
                f"{details.get('timings', '')} {details.get('details', '')}."
            )

    placements = data.get("placements", {})
    chunks.append(
        f"RNSIT placements feature over {placements.get('total_companies')} total companies. "
        f"Major recent recruiters include: {', '.join(placements.get('recent_recruiters', []))}."
    )
    for year, stats in placements.get("stats", {}).items():
        chunks.append(
            f"In {year}, the highest package offered was {stats.get('highest_ctc_lpa')} LPA."
        )

    KNOWLEDGE_CHUNKS = [c for c in chunks if c and c.strip()]
    logger.info(f"[RAG] Knowledge base ready — {len(KNOWLEDGE_CHUNKS)} chunks loaded (keyword mode).")


# ==========================================
# KEYWORD-OVERLAP RETRIEVAL  (no embeddings)
# ==========================================
def _keyword_score(query_tokens: set[str], chunk: str) -> float:
    """
    Jaccard-style overlap:  |query ∩ chunk| / |query|
    Returns 0-1.  1.0 means every query word appeared in the chunk.
    """
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenise(chunk)
    if not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    return overlap / len(query_tokens)


async def retrieve_relevant_context(user_query: str, top_k: int = 3) -> tuple[str, float]:
    if not KNOWLEDGE_CHUNKS:
        return "No context found.", 0.0

    query_tokens = _tokenise(user_query)

    scored_chunks = [
        (_keyword_score(query_tokens, chunk), chunk)
        for chunk in KNOWLEDGE_CHUNKS
    ]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    max_score  = scored_chunks[0][0] if scored_chunks else 0.0
    top_matches = [chunk for _, chunk in scored_chunks[:top_k]]

    return "\n\n".join(top_matches), max_score


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
                {
                    "role": "system",
                    "content": "You rewrite short follow-up questions into standalone retrieval queries only."
                },
                {
                    "role": "user",
                    "content": condense_prompt
                }
            ],
            temperature=0.0,
            max_tokens=60,
        )

        if rewritten and isinstance(rewritten, str):
            rewritten = rewritten.strip().strip('"')
            if rewritten:
                logger.info(f"[CONTEXT REWRITER] '{question}' -> '{rewritten}'")
                return rewritten

    except Exception as e:
        logger.warning(f"[CONTEXT REWRITER] Failed, falling back to original query: {e}")

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

    await initialize_rag_knowledge_base()

    search_query             = await condense_query(question, history or [])
    context_text, max_score  = await retrieve_relevant_context(search_query, top_k=2)

    max_score = safe_float(max_score)
    logger.info(f"[RAG] Search Term: '{search_query}' | Keyword score: {max_score:.4f}")

    if max_score < SIMILARITY_THRESHOLD:
        logger.info(f"[RAG] Guardrail triggered — score {max_score:.4f} below threshold.")
        return (
            "I am the RNSIT Campus Kiosk virtual assistant. I can only help you with "
            "campus directions, layouts, departments, fees, and administrative guidelines. "
            "Please ask a campus-related question!"
        )

    system_prompt = (
        "You are the official AI Digital Receptionist for RNS Institute of Technology (RNSIT), Bengaluru.\n"
        "Your workspace is a public campus kiosk visible to parents, children, and students. "
        "Your tone must remain completely child-safe, welcoming, polite, and professional at all times.\n\n"
        f"Use the following verified campus facts to answer the visitor:\n{context_text}\n\n"
        "CRITICAL RESPONSE CONSTRAINTS:\n"
        "1. Rely only on the facts above. If the database context doesn't contain the answer, "
        "tell them to visit the Admin Block or reception counter for accurate human help.\n"
        "2. Keep responses snappy and punchy (2-3 sentences maximum). Avoid long paragraphs.\n"
        "3. If the user says nonsensical words, argues, or uses subtle harassment, ignore the tone and respond: "
        "'I am here to guide you with RNSIT campus routes, admissions, and facilities. Let me know how I can help.'\n"
        "4. Do not answer out-of-domain questions like politics, celebrities, or general trivia. "
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
        return text_out.strip() if text_out else "I am having trouble formatting the response text. Please try again."

    except httpx.HTTPStatusError as e:
        logger.error(f"[LLM API] Status {e.response.status_code}: {e.response.text}")
        return "I am having trouble accessing my AI engine. Please try again in a moment."
    except Exception as e:
        logger.exception(f"[LLM API] Connection failure: {e}")
        return "The kiosk AI engine is currently experiencing connectivity issues. Please visit the Admin Block."