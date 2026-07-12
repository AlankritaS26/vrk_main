import os
import json
import math
import asyncio
import logging
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION & GLOBAL VECTOR CACHE
# ==========================================
GENERATE_MODEL = "gemini-2.5-flash"
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
SIMILARITY_THRESHOLD = 0.42

JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "college_info.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings_cache.json")

KNOWLEDGE_CHUNKS: list[str] = []
KNOWLEDGE_EMBEDDINGS: list[list[float]] = []

PROFANITY_BLOCKLIST = {
    "badword1", "badword2", "abuse", "stupid", "idiot"
}

raw_key = os.getenv("GEMINI_API_KEY", "")
LLM_API_KEY = raw_key.strip().strip("'\"")

# PRODUCTION REFACTOR: Single, shared persistent HTTP Client container
_SHARED_ASYNC_CLIENT: httpx.AsyncClient | None = None

def get_shared_client() -> httpx.AsyncClient:
    """Returns a long-lived, reusable HTTP client maintaining a warm connection pool."""
    global _SHARED_ASYNC_CLIENT
    if _SHARED_ASYNC_CLIENT is None or _SHARED_ASYNC_CLIENT.is_closed:
        # Configured with production limits to handle concurrent scaling smoothly
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        _SHARED_ASYNC_CLIENT = httpx.AsyncClient(limits=limits, timeout=30.0)
    return _SHARED_ASYNC_CLIENT

async def close_llm_client():
    """Clean teardown hook to safely close connections on server shutdown."""
    global _SHARED_ASYNC_CLIENT
    if _SHARED_ASYNC_CLIENT and not _SHARED_ASYNC_CLIENT.is_closed:
        await _SHARED_ASYNC_CLIENT.aclose()
        logger.info("[RAG] Persistent connection pool cleanly terminated.")


# ==========================================
# DEFENSIVE TYPE-SAFE UTILITIES
# ==========================================
def safe_float(val) -> float:
    """Aggressively flattens and converts any value/nested layouts to a scalar float."""
    if val is None:
        return 0.0
    while isinstance(val, (list, tuple)):
        if not val:
            return 0.0
        val = val[0]  # FIX: actually step inside the nested structure (was `val = val`, an infinite loop)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def get_nested_value(data, keys: list):
    """Safely navigates through mixed dicts/lists using keys or integer indices to prevent crashes."""
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
    """Safely extracts role/speaker and text from any inbound message schema or list structure."""
    if not msg:
        return None, None
    if isinstance(msg, dict):
        speaker = msg.get("speaker") or msg.get("role")
        text = msg.get("text") or msg.get("content")
        return (str(speaker) if speaker else None, str(text) if text else None)
    if isinstance(msg, (list, tuple)) and len(msg) >= 2:
        # FIX: use the actual elements (speaker, text), not str(msg) for both
        return (str(msg[0]) if msg[0] else None, str(msg[1]) if msg[1] else None)
    speaker = getattr(msg, "speaker", getattr(msg, "role", None))
    text = getattr(msg, "text", getattr(msg, "content", None))
    return (str(speaker) if speaker else None, str(text) if text else None)


# ==========================================
# GEMINI CLOUD EMBEDDING API
# ==========================================
async def get_embedding(text: str, client: httpx.AsyncClient = None, semaphore: asyncio.Semaphore = None) -> list[float]:
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not configured.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={LLM_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"content": {"parts": [{"text": text}]}}

    # OPTIMIZATION: Pull from the shared connection pool if an explicit client wasn't passed down
    active_client = client if client is not None else get_shared_client()
    local_sem = semaphore if semaphore is not None else asyncio.Semaphore(50)

    max_retries = 3
    backoff_seconds = 1.5

    async with local_sem:
        for attempt in range(max_retries):
            try:
                response = await active_client.post(url, json=payload, headers=headers, timeout=15.0)

                if response.status_code in {429, 500, 502, 503, 504}:
                    logger.warning(
                        f"[RAG] Remote service issued status {response.status_code} "
                        f"on attempt {attempt + 1}/{max_retries}. Retrying in {backoff_seconds}s..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_seconds)
                        backoff_seconds *= 2
                        continue

                response.raise_for_status()
                data = response.json()
                return data["embedding"]["values"]

            except (httpx.HTTPError, Exception) as e:
                if attempt == max_retries - 1:
                    logger.error(f"[RAG] Permanent embedding extraction failure after {max_retries} attempts: {e}")
                    raise e
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2


# ==========================================
# COSINE SIMILARITY
# ==========================================
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return safe_float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0


# ==========================================
# DATA INGESTION & VECTOR INDEXING
# ==========================================
async def initialize_rag_knowledge_base():
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_EMBEDDINGS
    if KNOWLEDGE_CHUNKS:
        return
    if not os.path.exists(JSON_PATH):
        logger.warning(f"[RAG] Knowledge base file not found at: {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as file:
        data = json.load(file)

    chunks = []
    c_info = data.get("college", {})
    chunks.append(
        f"{c_info.get('name')} ({c_info.get('short_name')}) was established in {c_info.get('established')} "
        f"by founder {c_info.get('founder')}. It is an autonomous {c_info.get('type')} located at {c_info.get('location')}."
    )

    admin = data.get("administration", {})
    chunks.append(f"The college working hours are {admin.get('working_hours')}.")
    chunks.append(f"The Director of RNSIT is {admin.get('director', {}).get('name')}. Contact phone: {admin.get('director', {}).get('phone')}.")
    chunks.append(f"The Principal of RNSIT is {admin.get('principal', {}).get('name')}. Contact phone: {admin.get('principal', {}).get('phone')}.")

    for code, details in data.get("departments", {}).items():
        chunks.append(
            f"The department of {details.get('name')} ({code.upper()}) is located in the {details.get('block', 'Main campus block')}. "
            f"The HOD is {details.get('hod', 'the appointed department head')} and intake capacity is {details.get('intake', '180')} students per year."
        )

    for name, details in data.get("facilities", {}).items():
        if isinstance(details, dict):
            chunks.append(
                f"Facility: {name.replace('_', ' ').title()}. "
                f"Details: {details.get('name', '')} {details.get('location', '')} {details.get('timings', '')} {details.get('details', '')}."
            )

    placements = data.get("placements", {})
    chunks.append(
        f"RNSIT placements feature over {placements.get('total_companies')} total companies. "
        f"Major recent recruiters include: {', '.join(placements.get('recent_recruiters', []))}."
    )
    for year, stats in placements.get("stats", {}).items():
        chunks.append(f"In {year}, the highest package offered was {stats.get('highest_ctc_lpa')} LPA.")

    chunks = [c for c in chunks if c and c.strip()]
    KNOWLEDGE_CHUNKS = chunks

    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as cache_file:
                cached_data = json.load(cache_file)
                if cached_data.get("chunks") == chunks:
                    KNOWLEDGE_EMBEDDINGS = cached_data.get("embeddings", [])
                    logger.info("[RAG] Loaded vector cache instantly from local data file.")
                    return
        except Exception as cache_err:
            logger.warning(f"[RAG] Could not parse vector cache file, falling back to API: {cache_err}")

    logger.info(f"[RAG] Indexing {len(chunks)} chunks smoothly using controlled concurrency...")

    sem = asyncio.Semaphore(4)
    shared_client = get_shared_client()

    try:
        KNOWLEDGE_EMBEDDINGS = await asyncio.gather(
            *[get_embedding(chunk, client=shared_client, semaphore=sem) for chunk in chunks]
        )
        logger.info("[RAG] Knowledge base fully loaded into memory from Gemini API.")

        with open(CACHE_PATH, "w") as cache_file:
            json.dump({"chunks": chunks, "embeddings": KNOWLEDGE_EMBEDDINGS}, cache_file)
            logger.info("[RAG] Saved updated vector embeddings to local disk cache.")

    except Exception as e:
        logger.error(f"[RAG] Failed to initialize embeddings: {e}")
        raise


# ==========================================
# SEMANTIC RETRIEVAL
# ==========================================
async def retrieve_relevant_context(user_query: str, top_k: int = 3) -> tuple[str, float]:
    # Reuse pool automatically via get_embedding's internal default hook
    query_vector = await get_embedding(user_query)
    if not query_vector:
        return "No context found.", 0.0

    scored_chunks = [
        (safe_float(cosine_similarity(query_vector, chunk_vector)), KNOWLEDGE_CHUNKS[idx])
        for idx, chunk_vector in enumerate(KNOWLEDGE_EMBEDDINGS)
    ]
    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # FIX: extract the actual top score (a float), not the whole list of tuples
    max_score = scored_chunks[0][0] if scored_chunks else 0.0

    top_matches = [chunk for _, chunk in scored_chunks[:top_k]]
    return "\n\n".join(top_matches), max_score


async def condense_query(question: str, history: list) -> str:
    # Clean the query string for analysis
    q_lower = question.lower().strip()

    # CRITICAL SAFEGUARD: If there's no chat history, there is nothing to contextualize
    if not history:
        return question

    # PRODUCTION GUARDRAIL: If it's a direct, well-formed question, BYPASS the rewriter.
    # This prevents the rewriter from stripping out crucial keywords like 'library' or 'admin block'.
    bypass_keywords = {"where is", "location", "timing", "hours", "who is", "what is", "address"}
    if any(keyword in q_lower for keyword in bypass_keywords):
        return question

    # If the user's query is long and fully formed, it doesn't need to be condensed
    if len(question.split()) > 3:
        return question

    # Assemble conversation lines for ambiguous short follow-ups (e.g., "for cse")
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
        "Do NOT answer the question. Return ONLY the rewritten query string.\n\n"
        "History:\n" f"{history_context}\n"
        f"Follow-up Question: {question}\n"
        "Standalone Query:"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATE_MODEL}:generateContent?key={LLM_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": condense_prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 30}
    }
    headers = {"Content-Type": "application/json"}

    try:
        client = get_shared_client()
        response = await client.post(url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 200:
            res_json = response.json()
            rewritten = get_nested_value(res_json, ["candidates", 0, "content", "parts", 0, "text"])
            if rewritten and isinstance(rewritten, str) and rewritten.strip():
                rewritten = rewritten.strip().strip('"')
                logger.info(f"[CONTEXT REWRITER] Transformed short query '{question}' -> '{rewritten}'")
                return rewritten
    except Exception as e:
        logger.warning(f"[CONTEXT REWRITER] Exception caught, falling back to original query: {e}")

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

    # ==========================================
    # UPDATE INSIDE YOUR generate_rag_kiosk_response FUNCTION
    # ==========================================

    # LATENCY: condense_query is a full extra LLM round-trip whose only job
    # is resolving pronouns ("its fees" -> "CSE fees") from earlier turns.
    # On the first turns there is nothing to resolve — skip the trip.
    if history and len(history) >= 2:
        search_query = await condense_query(question, history)
    else:
        search_query = question
    # CHANGE: Increase top_k to 3 for better context depth coverage
    context_text, max_score = await retrieve_relevant_context(search_query, top_k=3)

    max_score = safe_float(max_score)
    logger.info(f"[RAG] Search Term: '{search_query}' | Max similarity: {max_score:.4f}")

    if max_score < SIMILARITY_THRESHOLD:
        logger.info(f"[RAG] Guardrail triggered — score {max_score:.4f} below threshold.")
        return (
            "I am the RNSIT Campus Kiosk virtual assistant. I can only help you with "
            "campus directions, layouts, departments, fees, and administrative guidelines. "
            "Please ask a campus-related question!"
        )

    # REINFORCED PROMPT CONFIGURATION
    system_prompt = (
        "You are the official AI Digital Receptionist for RNS Institute of Technology (RNSIT), Bengaluru.\n"
        "Your workspace is a public campus kiosk visible to parents, children, and students. "
        "Your tone must remain completely child-safe, welcoming, polite, and professional at all times.\n\n"
        f"Use the following verified campus facts to answer the visitor:\n{context_text}\n\n"
        "CRITICAL RESPONSE CONSTRAINTS:\n"
        "1. Prioritize reading individual names, designations, HODs, and block locations directly from the context facts provided above. "
        "Do NOT state you do not know the information if a name or department head is written in the facts block above.\n"
        "2. Keep responses snappy and punchy (2-3 sentences maximum). Avoid long paragraphs.\n"
        "3. ABSURDITY OR GIBBERISH: If the user says nonsensical words, attempts to argue, or uses subtle "
        "harassment that bypasses core safety filters, ignore the tone completely. State smoothly: "
        "'I am here to guide you with RNSIT campus routes, admissions, and facilities. Let me know how I can help.'\n"
        "4. OUT-OF-DOMAIN: Do not give life advice or answer general trivia. If they ask about sports, celebrities, "
        "or politics, cleanly guide them back to college topics.\n"
    )

    contents = []
    if history:
        for msg in history[-4:]:
            speaker_val, text_val = parse_history_message(msg)
            if speaker_val and text_val:
                role = "user" if speaker_val.lower() in ("visitor", "user") else "model"
                contents.append({"role": role, "parts": [{"text": text_val}]})

    contents.append({"role": "user", "parts": [{"text": question}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATE_MODEL}:generateContent?key={LLM_API_KEY}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.1,  # Lowering from 0.2 to 0.1 forces strict factual retrieval
            "maxOutputTokens": 500
        }
    }
    headers = {"Content-Type": "application/json"}
    contents = []
    if history:
        for msg in history[-4:]:
            speaker_val, text_val = parse_history_message(msg)
            if speaker_val and text_val:
                role = "user" if speaker_val.lower() in ("visitor", "user") else "model"
                contents.append({"role": role, "parts": [{"text": text_val}]})

    contents.append({"role": "user", "parts": [{"text": question}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATE_MODEL}:generateContent?key={LLM_API_KEY}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
    }
    headers = {"Content-Type": "application/json"}

    try:
        # OPTIMIZATION: Fire final content payload using pool session
        client = get_shared_client()
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            res_json = response.json()
            logger.info(f"[RAW GEMINI RESPONSE] {json.dumps(res_json, indent=2)}")

            text_out = get_nested_value(res_json, ["candidates", 0, "content", "parts", 0, "text"])
            if text_out and isinstance(text_out, str):
                return text_out.strip()

            return "I am having trouble formatting the response text. Please try again."

        logger.error(f"[Gemini Cloud] Status {response.status_code}: {response.text}")
        return "I am having trouble accessing my AI engine. Please try again in a moment."
    except Exception as e:
        logger.exception(f"[Gemini Cloud] Connection failure: {e}")
        return "The kiosk AI engine is currently experiencing connectivity issues. Please visit the Admin Block."