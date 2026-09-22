"""
backend/confidence_rag.py — Confidence-Based RAG Pipeline
===========================================================
Implements the HIGH / MEDIUM / LOW retrieval-confidence routing described
in the architecture diagram:

  RAG Search (top-K)
    -> Confidence Check (based on top result score)
         HIGH   (score >= HIGH_THRESHOLD)
           -> Ambiguity Check (score + structured metadata, NO extra LLM call)
                ambiguous     -> "Did you mean A or B?"      (clarify)
                not ambiguous -> Answer Generation Chain
         MEDIUM (NEAR_THRESHOLD <= score < HIGH_THRESHOLD)
           -> Best Candidate (Top-1) -> "Are you asking about X?"
                Yes -> Answer Generation Chain (using confirmed candidate)
                No  -> Unknown Handling
         LOW    (score < NEAR_THRESHOLD)
           -> Scope Check (embedding-similarity lower bound + keyword
              allowlist, NO extra LLM call)
                RNSIT-related but unknown -> save to MongoDB (unanswered_
                    questions) + "I don't have that yet, I'll look into it."
                Out-of-scope               -> immediate canned response,
                    NEVER saved to unanswered_questions (out-of-scope
                    protection — keeps the knowledge-update loop clean).

  Unknown Handling (shared by MEDIUM-No and LOW-RNSIT_UNKNOWN):
    -> save_unanswered_question() -> "I don't have reliable information
       about that." -> current turn ends.

  Answer Generation Chain:
    Remote GPU LLM (local Qwen) -> Gemini (Tier 2) -> RAG-only nearest
    context (Tier 3) — reuses backend.llm's existing 3-tier fallback so
    this file owns ROUTING, not LLM transport.

Session-aware bits (context-aware interaction):
  - `condense_query()` (backend.llm) rewrites short follow-ups ("what
    about placements?") into a standalone query using recent turns.
  - A MEDIUM confirmation or HIGH ambiguity question puts a small
    `pending` marker into the caller-owned `session_state` dict; the
    very next call to `handle_query()` checks that marker FIRST so a
    bare "yes"/"no"/"the CSE one" reply resolves against the right
    candidate instead of being treated as a fresh RAG query.
"""
from __future__ import annotations

import os
import re
import time
import difflib
import logging
from typing import Any

from backend.query_correction import normalize_query
from backend.entity_mapping import detect_entity, VERIFIED_ENTITIES
from backend.llm import (
    retrieve_relevant_context,
    chat_completion_with_fallback,
    chat_completion_with_fallback_stream,
    condense_query,
    safe_float,
    parse_history_message,
    _clean_repetitive_greeting,
    _pop_complete_sentences,
    _QA_LABEL_RE_LLM,
    _FACILITY_LABEL_RE_LLM,
    _try_fetch_weather,
    _try_fetch_traffic,
    RAG_TOP_K,
)
from backend.database import save_unanswered_question, get_rag_settings
from backend.rag_calibration_log import log_decision, log_resolution

logger = logging.getLogger("RNSIT_Kiosk.ConfidenceRAG")


def is_weather_intent(query: str) -> bool:
    q = (query or "").lower().strip()
    keywords = (
        "weather", "rain", "raining", "temperature", "temp", "hot outside",
        "cold outside", "climate", "sunny", "humid", "forecast", "cloudy",
        "degree", "celsius"
    )
    return any(w in q for w in keywords)


def is_traffic_intent(query: str) -> bool:
    q = (query or "").lower().strip()
    keywords = (
        "traffic", "congestion", "jam", "road condition", "road conditions",
        "how's the road", "how is the road", "road near", "traffic near",
        "commute"
    )
    return any(w in q for w in keywords)

# ── Configurable thresholds ─────────────────────────────────────────
# Values are resolved with precedence DB (admin dashboard) > env var >
# hardcoded default — see backend/database.py::get_rag_settings(). This
# module caches the resolved values for _SETTINGS_CACHE_TTL seconds so a
# request doesn't hit MongoDB on every single query; the admin PUT
# endpoint (backend/admin_knowledge.py) calls invalidate_settings_cache()
# right after saving so a change is picked up on the very next query
# instead of waiting out the TTL.
_SETTINGS_CACHE_TTL = 30.0
_settings_cache: dict[str, Any] = {"value": None, "ts": 0.0}


def invalidate_settings_cache() -> None:
    _settings_cache["value"] = None
    _settings_cache["ts"] = 0.0


async def _get_settings() -> dict:
    now = time.monotonic()
    if _settings_cache["value"] is not None and (now - _settings_cache["ts"]) < _SETTINGS_CACHE_TTL:
        return _settings_cache["value"]
    settings = await get_rag_settings()
    _settings_cache["value"] = settings
    _settings_cache["ts"] = now
    return settings


# ── Keyword allowlist for the Scope Check (no extra LLM call) ──────────
DOMAIN_KEYWORDS = {
    "rnsit", "rns", "college", "campus", "institute", "university",
    "admission", "admissions", "eligibility", "cutoff", "comedk", "cet",
    "kcet", "department", "hod", "faculty", "professor", "principal",
    "director", "course", "branch", "syllabus", "semester", "exam",
    "fee", "fees", "scholarship", "hostel", "canteen", "library",
    "placement", "placements", "recruiter", "internship", "package",
    "ctc", "sports", "gym", "transport", "bus", "block", "building",
    "lab", "laboratory", "workshop", "auditorium", "seminar", "event",
    "fest", "club", "ncc", "nss", "phd", "research", "intake", "seat",
    "vtu", "aicte", "naac", "nba",
    # Campus names & specific keywords
    "ramesh", "babu", "venkatesha", "shetty", "kiran", "pallavi", "aperture",
    "quizcorp", "adroit", "aura", "toyota", "canara", "atm", "ambulance",
    "medical", "firstaid", "counselling", "calendly", "wifi", "internet",
    "cricket", "football", "basketball", "cafeteria", "food", "eat",
    "timing", "timings", "hours", "books", "mca", "mba", "cse", "ise",
    "ece", "eee", "mech", "civil", "aiml", "aids", "founder", "chairman",
}


def _is_in_domain_keyword(question: str) -> bool:
    words = set(re.findall(r"[a-z]+", (question or "").lower()))
    return bool(words & DOMAIN_KEYWORDS)


# Fallback text used only if VERIFIED_ENTITIES["COLLEGE_OVERVIEW"] is ever
# missing/unbuilt (e.g. college_info.json unreadable at startup) — keeps
# broad-overview queries answerable instead of erroring out.
_COLLEGE_OVERVIEW_FALLBACK = (
    "RNS Institute of Technology (RNSIT) was established in 2001 by Dr. R. N. Shetty. "
    "It is an autonomous private engineering college affiliated to Visvesvaraya Technological University (VTU) "
    "located in Channasandra, Bengaluru. The campus features 9 departments, modern laboratories, a central library, "
    "sports grounds, active student clubs, and excellent placement opportunities."
)


def _get_college_overview_fact() -> str:
    """Single source of truth for the RNSIT_GENERAL / broad-overview answer.

    Pulls from entity_mapping.VERIFIED_ENTITIES["COLLEGE_OVERVIEW"], which is
    sourced from college_info.json, instead of a hand-duplicated string here
    that could drift out of sync with it.
    """
    entry = VERIFIED_ENTITIES.get("COLLEGE_OVERVIEW") or {}
    return entry.get("answer") or _COLLEGE_OVERVIEW_FALLBACK


# ── Small helpers ────────────────────────────────────────────────────
_INTERROGATIVE_PREFIXES = re.compile(
    r"^(what\s+is|what\s+are|what|how\s+many|how\s+much|how|is\s+there\s+a|is\s+there|"
    r"are\s+there|who\s+is|who\s+are|who|does|do|can\s+you\s+tell\s+me|tell\s+me|"
    r"where\s+is|where\s+are|when\s+is|when\s+does)\s+",
    re.IGNORECASE,
)


def _humanize_topic(label: str) -> str:
    """
    Turns a verbatim FAQ question ("Is there a gym on campus?") into a
    short topic phrase ("the gym on campus") suitable for a spoken
    clarification prompt. Strips a leading interrogative (applied twice,
    since some questions have compound openers like "Is there a") and the
    trailing '?'. Falls back to the original label if stripping would
    leave nothing usable — this is a best-effort heuristic, not a parser,
    so it's intentionally conservative about giving up.
    """
    cleaned = (label or "").strip().rstrip("?").strip()
    if not cleaned:
        return label
    for _ in range(2):
        stripped = _INTERROGATIVE_PREFIXES.sub("", cleaned, count=1).strip()
        if stripped and stripped != cleaned:
            cleaned = stripped
        else:
            break
    if len(cleaned) < 3:
        return label  # stripping ate the whole thing — not usable, keep original
    return cleaned[0].upper() + cleaned[1:] if cleaned else label


def _short_label(metadata: dict, text: str) -> str:
    """Best-effort human label for a retrieved chunk: prefer structured
    metadata (entity_name), fall back to a short text snippet. FAQ-type
    entries store their entity_name as the verbatim question (needed so
    the ambiguity check can tell two DIFFERENT FAQs apart — see
    backend/llm.py::_json_to_text_chunks) — that's fine for internal
    entity-equality comparisons, but reads badly spoken back as "did you
    mean X?", so question-shaped labels get humanized before display."""
    for key in ("entity_name", "filename"):
        val = (metadata or {}).get(key)
        if val:
            label = str(val).replace("_", " ").replace(".json", "").strip()
            if label.endswith("?"):
                label = _humanize_topic(label)
            return label
    snippet = re.split(r"[.!?]", (text or "").strip())[0]
    words = snippet.split()
    return " ".join(words[:6]) if words else "that topic"


def _text_overlap_ratio(text_a: str, text_b: str) -> float:
    """Cheap, dependency-free text-similarity fallback (difflib, stdlib)
    used ONLY when one or both candidates lack entity_name metadata (e.g.
    legacy seed-data chunks). A LOW ratio between two texts that both
    scored close to the top means they're plausibly about different
    topics, not just two phrasings of the same fact — that's the
    ambiguity signal in the no-metadata case."""
    a = (text_a or "").strip().lower()
    b = (text_b or "").strip().lower()
    if not a or not b:
        return 1.0  # nothing to compare -> don't claim ambiguity
    return difflib.SequenceMatcher(None, a, b).ratio()


def _matches_query_keywords(query: str, entity_str: str) -> bool:
    """Returns True if significant content words from entity_str appear in query."""
    if not query or not entity_str:
        return False
    q = query.lower()
    # Extract candidate keywords >= 3 chars, ignoring common filler words
    words = [w for w in re.findall(r'\b\w+\b', entity_str.lower())
             if len(w) >= 3 and w not in {"what", "how", "where", "when", "which", "tell", "about", "rnsit", "college", "department", "process"}]
    if not words:
        return False
    for w in words:
        w_stem = w.rstrip('s')
        if len(w_stem) >= 3 and (w in q or w_stem in q):
            return True
    return False




_YES_WORDS = {"yes", "yeah", "yep", "sure", "correct", "right", "yup", "ok", "okay", "y"}
_NO_WORDS  = {"no", "nope", "nah", "not that", "incorrect", "wrong", "n"}

# Hard circuit-breaker on the clarification loop. Without this, a query
# that keeps landing in the ambiguous/near-miss score band (common with
# weaker embedding models, or a genuinely unanswerable question) can
# bounce "Did you mean X or Y?" forever — every non-matching reply looks
# like a NEW question, which can ALSO come back ambiguous, with no exit.
# After this many consecutive clarification asks in one session without
# landing on an actual answer, we stop asking and fall back to Unknown
# Handling instead — a wrong "I don't know" is recoverable; an infinite
# loop is not.
MAX_CLARIFY_STREAK = int(os.getenv("RAG_MAX_CLARIFY_STREAK", "2"))


def _classify_yesno(q_normalized: str) -> str | None:
    q = (q_normalized or "").strip().lower()
    if not q or len(q.split()) > 5:
        return None
    if q in _YES_WORDS or any(q.startswith(w + " ") for w in _YES_WORDS):
        return "yes"
    if q in _NO_WORDS or any(q.startswith(w + " ") for w in _NO_WORDS):
        return "no"
    return None


# ── Answer Generation Chain (Remote GPU LLM -> Gemini -> RAG-only) ─────
_SYSTEM_PROMPT_TMPL = (
    "You are Nova, the official AI Digital Receptionist for RNS Institute "
    "of Technology (RNSIT), Bengaluru. Your workspace is a public campus "
    "kiosk; keep your tone welcoming, polite, and professional.\n\n"
    "Use ONLY the verified campus facts below to answer — do not use any "
    "outside knowledge, and do not guess.\n\n"
    "Facts:\n{context}\n\n"
    "CONSTRAINTS:\n"
    "1. Never start with a greeting or self-introduction — answer directly.\n"
    "2. Keep the answer to 2-3 sentences maximum.\n"
    "3. If the facts above don't actually answer the question, say so "
    "honestly ('I don't have that detail on hand') instead of guessing.\n"
    "4. Never output literal 'Q:' / 'A:' labels."
)


async def _generate_answer(question: str, context_text: str, history: list | None) -> tuple[str, str]:
    """Tier 1/2 via chat_completion_with_fallback; Tier 3 = RAG-only
    nearest-context safety net when both LLM tiers are unavailable."""
    system_prompt = _SYSTEM_PROMPT_TMPL.format(context=context_text or "(none)")
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (history or [])[-4:]:
        speaker, text = parse_history_message(msg)
        if speaker and text:
            role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": question})

    try:
        text, tier, _model = await chat_completion_with_fallback(
            messages, temperature=0.2, max_tokens=180
        )
        return _clean_repetitive_greeting((text or "").strip()), tier.upper()
    except Exception as e:
        logger.error("[CONF-RAG] Local+Gemini both failed, using RAG-only nearest context: %s", e)
        fallback = _QA_LABEL_RE_LLM.sub("", context_text or "").strip()
        fallback = _FACILITY_LABEL_RE_LLM.sub("", fallback).strip()
        fallback = fallback.split("\n\n")[0][:400].strip()
        if fallback:
            return f"Based on what I have on file: {fallback}", "RAG_ONLY"
        return (
            "I'm having trouble reaching my knowledge base right now — "
            "please check with the Admin Block.",
            "RAG_ONLY_EMPTY",
        )


# ── Unknown handling (shared by MEDIUM-No and LOW-RNSIT_UNKNOWN) ───────
async def _unknown_handling(question: str, related: bool, session_id: str | None,
                             face_id: str | None) -> str:
    if related:
        await save_unanswered_question(question, related=True,
                                        session_id=session_id, face_id=face_id)
    return "I don't have reliable information about that."


# ── HIGH branch ──────────────────────────────────────────────────────
async def _handle_high(question: str, context_text: str, raw_results: list[dict],
                        best_score: float, history: list | None, state: dict,
                        settings: dict, session_id: str | None) -> dict:
    top1 = raw_results[0]
    top2 = raw_results[1] if len(raw_results) > 1 else None

    ambiguous = False
    label_a = label_b = ""
    ambiguity_gap = settings["ambiguity_gap"]
    text_sim_threshold = settings["text_similarity_threshold"]
    gap = None
    entity_a = entity_b = ""
    second_score = safe_float(top2.get("score", 0)) if top2 else None

    if top2:
        gap = best_score - second_score
        m1, m2 = top1.get("metadata", {}) or {}, top2.get("metadata", {}) or {}
        label_a = _short_label(m1, top1.get("text", ""))
        label_b = _short_label(m2, top2.get("text", ""))
        entity_a, entity_b = m1.get("entity_name", ""), m2.get("entity_name", "")
        type_a = (m1.get("entity_type") or "").strip().lower()
        type_b = (m2.get("entity_type") or "").strip().lower()

        norm_a, norm_b = label_a.strip().lower(), label_b.strip().lower()
        if gap <= ambiguity_gap and label_a and label_b and norm_a != norm_b:
            if entity_a and entity_b:
                # PRIMARY signal: structured metadata (Ambiguity Check as
                # drawn in the diagram — "Score + Structured Metadata").
                # Two DIFFERENT named entities scoring within the gap =
                # genuinely ambiguous — BUT ONLY when they are the SAME
                # entity_type (e.g. two departments, two people). Chunks
                # from DIFFERENT entity_types (e.g. 'department' vs 'fees'
                # or 'placements') are COMPLEMENTARY context that the LLM
                # should see together, not competing candidates.
                same_name = entity_a.strip().lower() != entity_b.strip().lower()
                same_type = (not type_a or not type_b or type_a == type_b)
                ambiguous = same_name and same_type

            else:
                # FALLBACK signal for chunks with no entity_name (older
                # seed data, or any admin entry created without one): use
                # cheap stdlib text-similarity instead. Low overlap + close
                # score = plausibly two different topics competing for the
                # same answer slot.
                overlap = _text_overlap_ratio(top1.get("text", ""), top2.get("text", ""))
                ambiguous = overlap < text_sim_threshold
                if ambiguous:
                    logger.info("[CONF-RAG] Ambiguity via text-similarity fallback "
                                "(no entity_name on one/both candidates), overlap=%.2f", overlap)

            # DISAMBIGUATION OVERRIDE: If the user's question explicitly mentions keywords
            # from Candidate A (e.g. "management fee" -> "Fees") but NOT Candidate B ("Admission Process"),
            # Candidate A is clearly the intended target — override ambiguity to False.
            if ambiguous:
                a_matched = _matches_query_keywords(question, entity_a or label_a)
                b_matched = _matches_query_keywords(question, entity_b or label_b)
                if a_matched and not b_matched:
                    logger.info("[CONF-RAG] Ambiguity overridden: question explicitly matches top candidate '%s' over '%s'",
                                entity_a or label_a, entity_b or label_b)
                    ambiguous = False


    # ── Human-readable decision log (console) ───────────────────────────
    route_preview = "HIGH_AMBIGUOUS" if ambiguous else "HIGH"
    logger.info(
        "[CONF-RAG-DECISION] question=%r | top_score=%.3f top_entity=%r | "
        "second_score=%s second_entity=%r | gap=%s (limit=%.3f) | ambiguous=%s | route=%s",
        question, best_score, entity_a or label_a,
        f"{second_score:.3f}" if second_score is not None else "n/a",
        entity_b or label_b,
        f"{gap:.3f}" if gap is not None else "n/a",
        ambiguity_gap, ambiguous, route_preview,
    )

    clarify_streak = state.get("clarify_streak", 0)
    if ambiguous and clarify_streak >= MAX_CLARIFY_STREAK:
        # Circuit breaker: this session has already been asked to
        # disambiguate MAX_CLARIFY_STREAK times in a row without landing
        # on an answer. Asking again would just continue the loop — stop
        # and give a terminal (if unsatisfying) response instead.
        logger.warning("[CONF-RAG] Clarify streak limit (%d) hit for session — "
                        "forcing Unknown Handling instead of another ambiguous prompt.",
                        MAX_CLARIFY_STREAK)
        await log_decision(
            session_id=session_id, question=question,
            top_score=best_score, second_score=second_score,
            top_entity=entity_a or label_a, second_entity=entity_b or label_b,
            confidence_band="HIGH", ambiguous=True, route="HIGH_CLARIFY_LIMIT",
            resolution="clarify_limit_reached",
        )
        answer = await _unknown_handling(question, related=True, session_id=session_id, face_id=None)
        state["pending"] = None
        state["clarify_streak"] = 0
        return {"answer": answer, "route": "HIGH_CLARIFY_LIMIT", "session_action": "CONTINUE",
                "session_state": state}

    if ambiguous:
        decision_id = await log_decision(
            session_id=session_id, question=question,
            top_score=best_score, second_score=second_score,
            top_entity=entity_a or label_a, second_entity=entity_b or label_b,
            confidence_band="HIGH", ambiguous=True, route="HIGH_AMBIGUOUS",
            resolution="pending",
        )
        state["pending"] = {
            "type": "ambiguous", "decision_id": decision_id,
            "a": {"label": label_a, "text": top1.get("text", ""), "metadata": top1.get("metadata", {})},
            "b": {"label": label_b, "text": top2.get("text", ""), "metadata": top2.get("metadata", {})},
        }
        state["clarify_streak"] = clarify_streak + 1
        answer = f"Did you mean {label_a}, or {label_b}?"
        return {"answer": answer, "route": "HIGH_AMBIGUOUS", "session_action": "CONTINUE",
                "session_state": state}

    answer, gen_tier = await _generate_answer(question, context_text, history)
    await log_decision(
        session_id=session_id, question=question,
        top_score=best_score, second_score=second_score,
        top_entity=entity_a or label_a, second_entity=entity_b or label_b,
        confidence_band="HIGH", ambiguous=False, route=f"HIGH_{gen_tier}",
        resolution="answered", answer_source=gen_tier,
    )
    state["pending"] = None
    state["clarify_streak"] = 0
    state["last_topic"] = _short_label(top1.get("metadata", {}), top1.get("text", ""))
    return {"answer": answer, "route": f"HIGH_{gen_tier}", "session_action": "CONTINUE",
            "session_state": state}


# ── MEDIUM branch ────────────────────────────────────────────────────
async def _handle_medium(question: str, raw_results: list[dict], state: dict,
                          session_id: str | None) -> dict:
    top1 = raw_results[0]
    top2 = raw_results[1] if len(raw_results) > 1 else None
    label = _short_label(top1.get("metadata", {}), top1.get("text", ""))

    clarify_streak = state.get("clarify_streak", 0)
    if clarify_streak >= MAX_CLARIFY_STREAK:
        logger.warning("[CONF-RAG] Clarify streak limit (%d) hit for session — "
                        "forcing Unknown Handling instead of another MEDIUM confirm.",
                        MAX_CLARIFY_STREAK)
        await log_decision(
            session_id=session_id, question=question,
            top_score=safe_float(top1.get("score", 0)),
            second_score=safe_float(top2.get("score", 0)) if top2 else None,
            top_entity=(top1.get("metadata", {}) or {}).get("entity_name", "") or label,
            second_entity=(top2.get("metadata", {}) or {}).get("entity_name", "") if top2 else "",
            confidence_band="MEDIUM", ambiguous=False, route="MEDIUM_CLARIFY_LIMIT",
            resolution="clarify_limit_reached",
        )
        answer = await _unknown_handling(question, related=True, session_id=session_id, face_id=None)
        state["pending"] = None
        state["clarify_streak"] = 0
        return {"answer": answer, "route": "MEDIUM_CLARIFY_LIMIT", "session_action": "CONTINUE",
                "session_state": state}

    decision_id = await log_decision(
        session_id=session_id, question=question,
        top_score=safe_float(top1.get("score", 0)),
        second_score=safe_float(top2.get("score", 0)) if top2 else None,
        top_entity=(top1.get("metadata", {}) or {}).get("entity_name", "") or label,
        second_entity=(top2.get("metadata", {}) or {}).get("entity_name", "") if top2 else "",
        confidence_band="MEDIUM", ambiguous=False, route="MEDIUM_CONFIRM",
        resolution="pending",
    )
    state["pending"] = {
        "type": "confirm", "decision_id": decision_id,
        "candidate": {"label": label, "text": top1.get("text", ""), "metadata": top1.get("metadata", {})},
    }
    state["clarify_streak"] = clarify_streak + 1
    answer = f"Are you asking about {label}?"
    return {"answer": answer, "route": "MEDIUM_CONFIRM", "session_action": "CONTINUE",
            "session_state": state}


# ── LOW branch (Scope Check) ────────────────────────────────────────
async def _handle_low(question: str, best_score: float, session_id: str | None,
                       face_id: str | None, state: dict, settings: dict) -> dict:
    in_domain = _is_in_domain_keyword(question) or best_score >= settings["scope_threshold"]
    state["pending"] = None
    state["clarify_streak"] = 0
    if in_domain:
        await save_unanswered_question(question, related=True,
                                        session_id=session_id, face_id=face_id)
        answer = "I don't have that yet — I'll look into it."
        route = "LOW_RNSIT_UNKNOWN"
        resolution = "unknown_logged"
    else:
        # Out-of-scope protection: NEVER written to unanswered_questions,
        # so unrelated questions can't pollute the knowledge-update loop.
        answer = "I'm here to answer RNSIT-related questions only."
        route = "LOW_OUT_OF_SCOPE"
        resolution = "out_of_scope"
    await log_decision(
        session_id=session_id, question=question,
        top_score=best_score, second_score=None,
        top_entity="", second_entity="",
        confidence_band="LOW", ambiguous=False, route=route,
        resolution=resolution,
    )
    return {
        "answer": answer,
        "route": route,
        "session_action": "CONTINUE",
        "session_state": state,
        "source": "guardrail_refusal" if route == "LOW_OUT_OF_SCOPE" else "rnsit_unknown",
    }


def _score_candidate_match(q: str, candidate: dict) -> float:
    if not q or not candidate:
        return 0.0
    q_norm = q.lower().strip()
    label = (candidate.get("label") or "").lower().strip()
    meta_name = ((candidate.get("metadata") or {}).get("entity_name") or "").lower().strip()
    full = f"{label} {meta_name}".strip()

    if not full:
        return 0.0

    # Exact equality
    if q_norm == label or q_norm == meta_name or q_norm == full:
        return 10.0

    # Substring inclusion in either direction
    if label and (q_norm in label or label in q_norm):
        return 8.0
    if meta_name and (q_norm in meta_name or meta_name in q_norm):
        return 8.0

    # Key acronym / alias matching (e.g. AIML vs CSE (AI and ML))
    q_words = set(re.findall(r'[a-z0-9]+', q_norm))
    full_words = set(re.findall(r'[a-z0-9]+', full))

    if ("aiml" in q_words or "ai" in q_words or "ml" in q_words) and ("aiml" in full_words or "ai" in full_words or "ml" in full_words):
        return 9.0

    # Token overlap
    stop = {"the", "a", "an", "is", "of", "and", "or", "to", "in", "for", "department", "one", "which", "what"}
    meaningful = q_words - stop
    if meaningful:
        overlap = meaningful & full_words
        if overlap:
            return float(len(overlap)) * 2.0

    return 0.0


# ── Pending-clarification resolver (checked BEFORE a fresh RAG search) ─
async def _resolve_pending(question: str, pending: dict, history: list | None,
                            session_id: str | None, face_id: str | None):
    """
    Returns a result dict if `question` resolved the pending MEDIUM
    confirmation or HIGH ambiguity choice, or None if it looks like a
    brand-new question (caller should fall through to a fresh RAG search
    — in which case the caller is responsible for logging an "abandoned"
    resolution row, since only it knows the pending marker was dropped).
    """
    q = question.strip().lower()
    yn = _classify_yesno(q)
    ptype = pending.get("type")
    decision_id = pending.get("decision_id", "")

    if ptype == "confirm":
        candidate = pending["candidate"]
        if yn == "yes":
            answer, tier = await _generate_answer(question, candidate["text"], history)
            if decision_id:
                await log_resolution(decision_id=decision_id, session_id=session_id,
                                      question=question, user_reply=question,
                                      resolution="confirmed", answer_source=tier)
            return {"answer": answer, "route": f"MEDIUM_CONFIRMED_{tier}",
                     "session_action": "CONTINUE", "topic": candidate.get("label")}
        if yn == "no":
            answer = await _unknown_handling(question, related=True,
                                              session_id=session_id, face_id=face_id)
            if decision_id:
                await log_resolution(decision_id=decision_id, session_id=session_id,
                                      question=question, user_reply=question,
                                      resolution="declined")
            return {"answer": answer, "route": "MEDIUM_DECLINED",
                     "session_action": "CONTINUE", "topic": None}
        return None  # not a yes/no reply -> treat as a new question

    if ptype == "ambiguous":
        a, b = pending["a"], pending["b"]

        is_a_explicit = q in ("a", "first", "1", "option a", "the first", "the first one", "former")
        is_b_explicit = q in ("b", "second", "2", "option b", "the second", "the second one", "latter")

        score_a = _score_candidate_match(q, a)
        score_b = _score_candidate_match(q, b)

        chosen = None
        if is_a_explicit or (score_a > 0 and score_a > score_b):
            chosen = (a, "disambiguated_a")
        elif is_b_explicit or (score_b > 0 and score_b > score_a):
            chosen = (b, "disambiguated_b")

        if chosen:
            cand, res_tag = chosen
            answer, tier = await _generate_answer(question, cand["text"], history)
            if decision_id:
                await log_resolution(decision_id=decision_id, session_id=session_id,
                                      question=question, user_reply=question,
                                      resolution=res_tag, answer_source=tier)
            return {"answer": answer, "route": f"HIGH_DISAMBIGUATED_{tier}",
                    "session_action": "CONTINUE", "topic": cand.get("label")}

        if yn == "no":
            answer = await _unknown_handling(question, related=True,
                                              session_id=session_id, face_id=face_id)
            if decision_id:
                await log_resolution(decision_id=decision_id, session_id=session_id,
                                      question=question, user_reply=question,
                                      resolution="neither")
            return {"answer": answer, "route": "HIGH_NEITHER",
                     "session_action": "CONTINUE", "topic": None}
        return None

    return None


# ── Public entry point ──────────────────────────────────────────────
async def handle_query(question: str, history: list | None = None,
                        session_state: dict | None = None,
                        session_id: str | None = None,
                        face_id: str | None = None) -> dict:
    """
    Confidence-based RAG entry point. `session_state` is a small dict the
    caller owns (e.g. active_session["rag_state"] in backend/main.py) —
    it is mutated in place AND returned under the "session_state" key so
    the caller can persist it (session-context write-back) regardless of
    how it's stored.

    Returns: {"answer": str, "route": str, "session_action": "CONTINUE",
              "session_state": dict}
    """
    state = dict(session_state or {})
    settings = await _get_settings()

    # 0) Normalize input query so STT phrase corrections ("rns it" -> "rnsit", "ai ml" -> "aiml")
    # are applied uniformly across all RAG search and routing paths.
    question_clean = normalize_query(question)

    # 1) Resolve a pending MEDIUM confirmation / HIGH ambiguity choice
    #    first — this is what makes "yes" or "the CSE one" resolve
    #    against the RIGHT candidate instead of being treated as a
    #    brand-new (and probably unmatchable) RAG query.
    pending = state.get("pending")
    if pending:
        resolved = await _resolve_pending(question, pending, history, session_id, face_id)
        if resolved is not None:
            state["pending"] = None
            state["clarify_streak"] = 0  # loop broken either way — answered or terminally declined
            if resolved.get("topic"):
                state["last_topic"] = resolved["topic"]
            resolved["session_state"] = state
            resolved.pop("topic", None)
            return resolved
        if pending.get("decision_id"):
            await log_resolution(decision_id=pending["decision_id"], session_id=session_id,
                                  question=question, user_reply=question, resolution="abandoned")
        state["pending"] = None

    # ── 1.5) Intent Router: Live Weather API ──────────────────────────────
    if is_weather_intent(question_clean) or is_weather_intent(question):
        weather_ans = await _try_fetch_weather(question)
        if not weather_ans:
            weather_ans = "I'm having trouble fetching live weather right now, but feel free to ask about RNSIT!"

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: WEATHER")
        print("DETECTED ENTITY: NONE")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {weather_ans}")
        print("ANSWER SOURCE: weather_api")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] NORMALIZED QUERY: %s", question_clean)
        logger.info("[CONF-RAG] DETECTED INTENT: WEATHER")
        logger.info("[CONF-RAG] ANSWER SOURCE: weather_api")

        state["pending"] = None
        state["clarify_streak"] = 0
        return {
            "answer": weather_ans,
            "route": "WEATHER_API",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "weather_api",
        }

    # ── 1.6) Intent Router: Live Traffic API ──────────────────────────────
    if is_traffic_intent(question_clean) or is_traffic_intent(question):
        traffic_ans = await _try_fetch_traffic(question)
        if not traffic_ans:
            traffic_ans = "Traffic on Dr. Vishnuvardhan Road near RNSIT is currently reported as normal."

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: TRAFFIC")
        print("DETECTED ENTITY: NONE")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {traffic_ans}")
        print("ANSWER SOURCE: traffic_api")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] NORMALIZED QUERY: %s", question_clean)
        logger.info("[CONF-RAG] DETECTED INTENT: TRAFFIC")
        logger.info("[CONF-RAG] ANSWER SOURCE: traffic_api")

        state["pending"] = None
        state["clarify_streak"] = 0
        return {
            "answer": traffic_ans,
            "route": "TRAFFIC_API",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "traffic_api",
        }

    # 2) Context-aware query condensing (short follow-ups -> standalone
    #    query, using recent conversation history).
    search_query = await condense_query(question_clean, history or [])

    # 2.5) Canonical Entity Detection:
    # Check whether the user query maps directly to a verified college entity
    detected = detect_entity(question_clean) or detect_entity(search_query) or detect_entity(question)
    if detected:
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: RNSIT")
        print(f"DETECTED ENTITY: {detected.entity_id}")
        print(f"ENTITY CONFIDENCE: {detected.confidence:.2f}")
        print(f"RETRIEVAL RESULT: {detected.verified_answer[:120]}")
        print("ANSWER SOURCE: entity_kb")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] NORMALIZED QUERY: %s", question_clean)
        logger.info("[CONF-RAG] DETECTED INTENT: RNSIT")
        logger.info("[CONF-RAG] DETECTED ENTITY: %s", detected.entity_id)
        logger.info("[CONF-RAG] ENTITY CONFIDENCE: %.2f", detected.confidence)
        logger.info("[CONF-RAG] RETRIEVAL RESULT: %s", detected.verified_answer[:120])
        logger.info("[CONF-RAG] ANSWER SOURCE: entity_kb")

        await log_decision(
            session_id=session_id, question=question,
            top_score=1.0, second_score=None,
            top_entity=detected.canonical_name, second_entity="",
            confidence_band="HIGH", ambiguous=False, route=f"HIGH_ENTITY_{detected.entity_id}",
            resolution="answered", answer_source="entity_kb",
        )
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = detected.canonical_name
        return {
            "answer": detected.verified_answer,
            "route": f"HIGH_ENTITY_{detected.entity_id}",
            "session_action": "CONTINUE",
            "session_state": state,
            "detected_entity": detected.entity_id,
            "entity_confidence": detected.confidence,
            "source": "entity_kb",
        }

    # 3) RAG search (single retrieval call feeds confidence routing,
    #    ambiguity check, AND the scope check below — no repeat calls).
    context_text, best_score, raw_results = await retrieve_relevant_context(
        search_query, top_k=RAG_TOP_K
    )
    best_score = safe_float(best_score)

    in_domain = _is_in_domain_keyword(question_clean) or _is_in_domain_keyword(question)

    # Out-of-scope refusal: If query contains no campus domain keywords and score is low (< 0.65)
    if not in_domain and best_score < 0.65:
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: UNSUPPORTED")
        print("DETECTED ENTITY: NONE")
        print(f"ENTITY CONFIDENCE: {best_score:.2f}")
        print("RETRIEVAL RESULT: None")
        print("ANSWER SOURCE: guardrail_refusal")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] NORMALIZED QUERY: %s", question_clean)
        logger.info("[CONF-RAG] DETECTED INTENT: UNSUPPORTED")
        logger.info("[CONF-RAG] ANSWER SOURCE: guardrail_refusal")

        return await _handle_low(question, 0.0, session_id, face_id, state, settings)

    # Broad RNSIT / general query handling:
    broad_overview_phrases = (
        "about rnsit", "something about rnsit", "about college", "tell me about rnsit",
        "about the college", "college overview", "what is rnsit", "tell me something about rnsit",
        "tell me about rns", "about campus", "tell me about this college"
    )
    is_broad_rnsit = any(p in question_clean for p in broad_overview_phrases)

    top_entity_label = raw_results[0].get("metadata", {}).get("entity_name", "UNKNOWN") if raw_results else "NONE"
    print(f"USER QUERY: {question}")
    print(f"NORMALIZED QUERY: {question_clean}")
    print("DETECTED INTENT: RNSIT_GENERAL")
    print("DETECTED ENTITY: NONE")
    print(f"ENTITY CONFIDENCE: {best_score:.2f}")
    print(f"RETRIEVAL RESULT: {context_text[:120] if context_text else 'None'}")
    print("ANSWER SOURCE: general_rag")

    logger.info(
        "[CONF-RAG] q=%r search=%r score=%.3f band=%s (high>=%.2f near>=%.2f)",
        question, search_query, best_score,
        "HIGH" if best_score >= settings["high_threshold"] else
        ("MEDIUM" if best_score >= settings["near_threshold"] else "LOW"),
        settings["high_threshold"], settings["near_threshold"],
    )

    if is_broad_rnsit:
        # Use ONLY the verified overview fact as context — deliberately do
        # NOT concatenate the raw top-k semantic-search results here. For a
        # short generic query like "tell me something about RNS IT", noisy
        # unrelated FAQ hits (e.g. the "website of RNSIT" FAQ) score
        # deceptively close and were confusing the LLM into surfacing them
        # instead of the actual overview. See _get_college_overview_fact().
        overview_fact = _get_college_overview_fact()
        answer, gen_tier = await _generate_answer(question, overview_fact, history)
        if "i don't have that detail" in answer.lower():
            answer = overview_fact
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = "College Overview"
        return {
            "answer": answer,
            "route": "RNSIT_GENERAL",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "general_rag",
        }

    if not raw_results:
        return await _handle_low(question, best_score, session_id, face_id, state, settings)

    if best_score >= settings["high_threshold"]:
        return await _handle_high(question, context_text, raw_results, best_score, history, state, settings, session_id)
    if best_score >= settings["near_threshold"]:
        return await _handle_medium(question, raw_results, state, session_id)
    return await _handle_low(question, best_score, session_id, face_id, state, settings)


# ── Streaming variant (for /ask/stream) ─────────────────────────────
async def handle_query_stream(question: str, history: list | None = None,
                               session_state: dict | None = None,
                               session_id: str | None = None,
                               face_id: str | None = None):
    """
    Same routing as handle_query(), but the HIGH-confidence / non-
    ambiguous branch streams its LLM-generated answer sentence-by-
    sentence (reusing llm.py's chat_completion_with_fallback_stream +
    _pop_complete_sentences) instead of waiting for the full string.

    All other outcomes — MEDIUM confirmation, HIGH ambiguity question,
    LOW scope-check response, pending-clarification resolution, and the
    Tier-3 RAG-only fallback — are short, deterministic (or already-
    generated) strings, so they're yielded as ONE chunk; there's nothing
    real to stream there and splitting them awkwardly hurts more than it
    helps (see the "should /ask/stream stream sentence-by-sentence?"
    note in this module's top-level docstring / the accompanying reply).

    Yields dicts identical in shape to handle_query()'s return value,
    except intermediate HIGH-branch chunks have "partial": True and only
    the FINAL yielded dict carries the full "session_state" to persist.
    """
    state = dict(session_state or {})
    settings = await _get_settings()
    question_clean = normalize_query(question)

    pending = state.get("pending")
    if pending:
        resolved = await _resolve_pending(question, pending, history, session_id, face_id)
        if resolved is not None:
            state["pending"] = None
            state["clarify_streak"] = 0  # loop broken either way — answered or terminally declined
            if resolved.get("topic"):
                state["last_topic"] = resolved["topic"]
            resolved["session_state"] = state
            resolved.pop("topic", None)
            yield resolved
            return
        if pending.get("decision_id"):
            await log_resolution(decision_id=pending["decision_id"], session_id=session_id,
                                  question=question, user_reply=question, resolution="abandoned")
        state["pending"] = None

    # ── 1.5) Intent Router: Live Weather API ──────────────────────────────
    if is_weather_intent(question_clean) or is_weather_intent(question):
        weather_ans = await _try_fetch_weather(question)
        if not weather_ans:
            weather_ans = "I'm having trouble fetching live weather right now, but feel free to ask about RNSIT!"

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: WEATHER")
        print("DETECTED ENTITY: NONE")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {weather_ans}")
        print("ANSWER SOURCE: weather_api")

        yield {"sentence": weather_ans, "partial": False}
        state["pending"] = None
        state["clarify_streak"] = 0
        yield {
            "done": True,
            "answer": weather_ans,
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "weather_api",
        }
        return

    # ── 1.6) Intent Router: Live Traffic API ──────────────────────────────
    if is_traffic_intent(question_clean) or is_traffic_intent(question):
        traffic_ans = await _try_fetch_traffic(question)
        if not traffic_ans:
            traffic_ans = "Traffic on Dr. Vishnuvardhan Road near RNSIT is currently reported as normal."

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: TRAFFIC")
        print("DETECTED ENTITY: NONE")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {traffic_ans}")
        print("ANSWER SOURCE: traffic_api")

        yield {"sentence": traffic_ans, "partial": False}
        state["pending"] = None
        state["clarify_streak"] = 0
        yield {
            "done": True,
            "answer": traffic_ans,
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "traffic_api",
        }
        return

    search_query = await condense_query(question_clean, history or [])

    detected = detect_entity(question_clean) or detect_entity(search_query) or detect_entity(question)
    if detected:
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: RNSIT")
        print(f"DETECTED ENTITY: {detected.entity_id}")
        print(f"ENTITY CONFIDENCE: {detected.confidence:.2f}")
        print(f"RETRIEVAL RESULT: {detected.verified_answer[:120]}")
        print("ANSWER SOURCE: entity_kb")

        yield {"sentence": detected.verified_answer, "partial": False}
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = detected.canonical_name
        yield {
            "done": True,
            "answer": detected.verified_answer,
            "session_action": "CONTINUE",
            "session_state": state,
            "detected_entity": detected.entity_id,
            "entity_confidence": detected.confidence,
            "source": "entity_kb",
        }
        return

    context_text, best_score, raw_results = await retrieve_relevant_context(
        search_query, top_k=RAG_TOP_K
    )
    best_score = safe_float(best_score)

    in_domain = _is_in_domain_keyword(question_clean) or _is_in_domain_keyword(question)

    if not in_domain and best_score < 0.65:
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: UNSUPPORTED")
        print("DETECTED ENTITY: NONE")
        print(f"ENTITY CONFIDENCE: {best_score:.2f}")
        print("RETRIEVAL RESULT: None")
        print("ANSWER SOURCE: guardrail_refusal")

        yield await _handle_low(question, 0.0, session_id, face_id, state, settings)
        return

    broad_overview_phrases = (
        "about rnsit", "something about rnsit", "about college", "tell me about rnsit",
        "about the college", "college overview", "what is rnsit", "tell me something about rnsit",
        "tell me about rns", "about campus", "tell me about this college"
    )
    is_broad_rnsit = any(p in question_clean for p in broad_overview_phrases)

    print(f"USER QUERY: {question}")
    print(f"NORMALIZED QUERY: {question_clean}")
    print("DETECTED INTENT: RNSIT_GENERAL")
    print("DETECTED ENTITY: NONE")
    print(f"ENTITY CONFIDENCE: {best_score:.2f}")
    print(f"RETRIEVAL RESULT: {context_text[:120] if context_text else 'None'}")
    print("ANSWER SOURCE: general_rag")

    if is_broad_rnsit:
        # Same fix as handle_query(): use ONLY the verified overview fact as
        # context, no raw top-k results mixed in, so noisy near-scoring FAQ
        # hits (e.g. "website of RNSIT") can't leak into the answer.
        overview_fact = _get_college_overview_fact()
        answer, gen_tier = await _generate_answer(question, overview_fact, history)
        if "i don't have that detail" in answer.lower():
            answer = overview_fact
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = "College Overview"
        yield {"sentence": answer, "partial": False}
        yield {
            "done": True,
            "answer": answer,
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "general_rag",
        }
        return

    if not raw_results:
        yield await _handle_low(question, best_score, session_id, face_id, state, settings)
        return
    if best_score < settings["near_threshold"]:
        yield await _handle_low(question, best_score, session_id, face_id, state, settings)
        return
    if best_score < settings["high_threshold"]:
        yield await _handle_medium(question, raw_results, state, session_id)
        return

    # HIGH band: run the same ambiguity check as the non-streaming path
    # WITHOUT generating an answer yet (peek), so an ambiguous HIGH still
    # comes back as one clarification chunk instead of a stray partial
    # LLM generation.
    top1 = raw_results[0]
    top2 = raw_results[1] if len(raw_results) > 1 else None
    ambiguous = False
    entity_a = entity_b = label_a = label_b = ""
    gap = None
    second_score = safe_float(top2.get("score", 0)) if top2 else None
    if top2:
        gap = best_score - second_score
        m1, m2 = top1.get("metadata", {}) or {}, top2.get("metadata", {}) or {}
        label_a, label_b = _short_label(m1, top1.get("text", "")), _short_label(m2, top2.get("text", ""))
        entity_a, entity_b = m1.get("entity_name", ""), m2.get("entity_name", "")
        type_a = (m1.get("entity_type") or "").strip().lower()
        type_b = (m2.get("entity_type") or "").strip().lower()
        norm_a, norm_b = label_a.strip().lower(), label_b.strip().lower()
        if gap <= settings["ambiguity_gap"] and label_a and label_b and norm_a != norm_b:
            if entity_a and entity_b:
                same_name = entity_a.strip().lower() != entity_b.strip().lower()
                same_type = (not type_a or not type_b or type_a == type_b or type_a == "faq" or type_b == "faq")
                ambiguous = same_name and same_type
            else:
                ambiguous = _text_overlap_ratio(top1.get("text", ""), top2.get("text", "")) < settings["text_similarity_threshold"]
            if ambiguous:
                a_matched = _matches_query_keywords(question, entity_a or label_a)
                b_matched = _matches_query_keywords(question, entity_b or label_b)
                if a_matched and not b_matched:
                    ambiguous = False
        if ambiguous:
            clarify_streak = state.get("clarify_streak", 0)
            if clarify_streak >= MAX_CLARIFY_STREAK:
                logger.warning("[CONF-RAG-STREAM] Clarify streak limit (%d) hit — "
                                "forcing Unknown Handling instead of another ambiguous prompt.",
                                MAX_CLARIFY_STREAK)
                await log_decision(
                    session_id=session_id, question=question,
                    top_score=best_score, second_score=second_score,
                    top_entity=entity_a or label_a, second_entity=entity_b or label_b,
                    confidence_band="HIGH", ambiguous=True, route="HIGH_CLARIFY_LIMIT",
                    resolution="clarify_limit_reached",
                )
                answer = await _unknown_handling(question, related=True, session_id=session_id, face_id=face_id)
                state["pending"] = None
                state["clarify_streak"] = 0
                yield {"answer": answer, "route": "HIGH_CLARIFY_LIMIT", "session_action": "CONTINUE",
                       "session_state": state}
                return

            decision_id = await log_decision(
                session_id=session_id, question=question,
                top_score=best_score, second_score=second_score,
                top_entity=entity_a or label_a, second_entity=entity_b or label_b,
                confidence_band="HIGH", ambiguous=True, route="HIGH_AMBIGUOUS",
                resolution="pending",
            )
            state["pending"] = {
                "type": "ambiguous", "decision_id": decision_id,
                "a": {"label": label_a, "text": top1.get("text", ""), "metadata": m1},
                "b": {"label": label_b, "text": top2.get("text", ""), "metadata": m2},
            }
            state["clarify_streak"] = clarify_streak + 1
            answer = f"Did you mean {label_a}, or {label_b}?"
            yield {"answer": answer, "route": "HIGH_AMBIGUOUS", "session_action": "CONTINUE",
                   "session_state": state}
            return

    # Not ambiguous -> real sentence-by-sentence streaming generation.
    system_prompt = _SYSTEM_PROMPT_TMPL.format(context=context_text or "(none)")
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (history or [])[-4:]:
        speaker, text = parse_history_message(msg)
        if speaker and text:
            role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": question})

    buf = ""
    parts: list[str] = []
    first_sentence_sent = False
    tier_seen = "local"
    try:
        async for delta, tier, _model in chat_completion_with_fallback_stream(
            messages, temperature=0.2, max_tokens=180
        ):
            tier_seen = tier
            buf += delta
            ready, buf = _pop_complete_sentences(buf)
            for s in ready:
                if not first_sentence_sent:
                    s = _clean_repetitive_greeting(s)
                    first_sentence_sent = True
                if s:
                    parts.append(s)
                    yield {"answer": s, "route": f"HIGH_{tier_seen.upper()}",
                           "session_action": "CONTINUE", "partial": True}
        if buf.strip():
            b = buf.strip()
            if not first_sentence_sent:
                b = _clean_repetitive_greeting(b)
            if b:
                parts.append(b)
                yield {"answer": b, "route": f"HIGH_{tier_seen.upper()}",
                       "session_action": "CONTINUE", "partial": True}
    except Exception as e:
        logger.error("[CONF-RAG-STREAM] Local+Gemini both failed, RAG-only fallback: %s", e)
        fallback = _QA_LABEL_RE_LLM.sub("", context_text or "").strip()
        fallback = _FACILITY_LABEL_RE_LLM.sub("", fallback).strip().split("\n\n")[0][:400].strip()
        final_answer = (f"Based on what I have on file: {fallback}" if fallback else
                         "I'm having trouble reaching my knowledge base right now — "
                         "please check with the Admin Block.")
        await log_decision(
            session_id=session_id, question=question,
            top_score=best_score, second_score=second_score,
            top_entity=entity_a or label_a, second_entity=entity_b or label_b,
            confidence_band="HIGH", ambiguous=False, route="HIGH_RAG_ONLY",
            resolution="answered", answer_source="rag_only",
        )
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = _short_label(top1.get("metadata", {}), top1.get("text", ""))
        yield {"answer": final_answer, "route": "HIGH_RAG_ONLY", "session_action": "CONTINUE",
               "session_state": state}
        return

    full_answer = "".join(parts).strip()
    await log_decision(
        session_id=session_id, question=question,
        top_score=best_score, second_score=second_score,
        top_entity=entity_a or label_a, second_entity=entity_b or label_b,
        confidence_band="HIGH", ambiguous=False, route=f"HIGH_{tier_seen.upper()}",
        resolution="answered", answer_source=tier_seen,
    )
    state["pending"] = None
    state["clarify_streak"] = 0
    state["last_topic"] = _short_label(top1.get("metadata", {}), top1.get("text", ""))
    # Final marker chunk: carries the full answer + session_state for the
    # caller to persist (interaction log, session write-back) — mirrors
    # the {'done': True, ...} sentinel main.py's /ask/stream already used.
    yield {"answer": full_answer, "route": f"HIGH_{tier_seen.upper()}",
           "session_action": "CONTINUE", "session_state": state, "final": True}