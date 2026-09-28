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
from backend.entity_mapping import detect_entity, VERIFIED_ENTITIES, get_kb_data
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
    "rnsit", "rns", "college", "colleges", "campus", "institute", "university",
    "admission", "admissions", "eligibility", "cutoff", "cutoffs", "comedk", "cet",
    "kcet", "department", "departments", "dept", "depts", "hod", "hods", "faculty", "faculties",
    "professor", "professors", "principal", "director", "course", "courses", "branch", "branches",
    "syllabus", "semester", "semesters", "exam", "exams",
    "fee", "fees", "scholarship", "scholarships", "hostel", "hostels", "canteen", "library",
    "placement", "placements", "recruiter", "recruiters", "internship", "internships", "package", "packages",
    "ctc", "sports", "gym", "transport", "bus", "buses", "block", "blocks", "building", "buildings",
    "lab", "labs", "laboratory", "laboratories", "workshop", "workshops", "auditorium", "seminar", "seminars", "event", "events",
    "fest", "fests", "club", "clubs", "ncc", "nss", "phd", "research", "intake", "seat", "seats",
    "vtu", "aicte", "naac", "nba",
    # Campus names & specific keywords
    "ramesh", "babu", "venkatesha", "shetty", "kiran", "pallavi", "aperture",
    "quizcorp", "adroit", "aura", "toyota", "canara", "atm", "ambulance",
    "medical", "firstaid", "counselling", "calendly", "wifi", "internet",
    "cricket", "football", "basketball", "cafeteria", "food", "eat",
    "timing", "timings", "hours", "books", "mca", "mba", "cse", "ise",
    "ece", "eee", "mech", "civil", "aiml", "aids", "founder", "chairman",
    # Spelled-out department/branch names (the abbreviations above — cse,
    # ece, eee, aiml, aids — only match if the visitor says the acronym.
    # A visitor who instead spells the branch out, e.g. "computer science",
    # "electronics and communication", "information science", was falling
    # through with NO keyword hit at all and getting refused as
    # out-of-scope even though it's a clearly on-topic follow-up.
    "computer", "science", "electronics", "communication", "communications",
    "information", "electrical", "mechanical", "artificial", "intelligence",
    "machine", "learning", "robotics", "biotechnology", "engineering",
    "technology", "telecommunication", "instrumentation",
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


def is_broad_department_query(query: str) -> bool:
    """True if query asks broadly about departments, branch comparisons, or choosing a branch."""
    q = (query or "").lower().strip()
    if not q:
        return False
    patterns = (
        r"\bwhich\s+(?:department|dept|branch|course)\s+(?:is\s+)?(?:good|better|best|preferred)\b",
        r"\bwhich\s+(?:department|dept|branch|course)\s+should\s+i\s+choose\b",
        r"\bwhich\s+(?:department|dept|branch|course)\s+to\s+choose\b",
        r"\bwhich\s+(?:department|dept|branch|course)\s+is\s+best\s+for\s+me\b",
        r"\bwhich\s+(?:department|dept|branch|course)\s+(?:is\s+)?better\s+for\s+(?:software|coding|placements?|hardware|core)\b",
        r"\bwhich\s+(?:department|dept|branch|course)\s+has\s+(?:good|better|best|high)\s+placements?\b",
        r"\bwhat\s+departments?\s+(?:does\s+rnsit\s+have|are\s+there|exist|are\s+available|offered)\b",
        r"\b(?:list\s+of\s+|all\s+)?departments?\s+(?:list|overview)?\b",
        r"\b(?:all\s+)?branches\s+offered\b",
        r"\b(?:is|are)\s+(?:cse|ise|ece|eee|aiml|aids|mech|civil)\s+(?:a\s+)?good\b",
        r"\bcompare\s+(?:the\s+)?departments?\b",
        r"\bwhich\s+branch\s+is\s+(?:good|better|best)\b",
        r"\bwhich\s+department\s+is\s+(?:good|better|best)\b",
    )
    for pat in patterns:
        if re.search(pat, q):
            return True
    if any(k in q for k in ("department", "departments", "branch", "branches")):
        if any(w in q for w in ("good", "better", "best", "choose", "suggest", "recommend", "compare", "preferred", "scope")):
            return True
    return False


# Known valid RNSIT department abbreviations / names
_KNOWN_DEPT_CODES = {"CSE", "ECE", "ISE", "EEE", "AIML", "AIDS", "MECH", "CIVIL", "MCA", "MBA"}

# Regex patterns to detect department mentions in comparison queries
_DEPT_PATTERNS = [
    ("CSE", r"\b(?:cse|computer\s+science(?:\s+and\s+engineering)?)\b"),
    ("ECE", r"\b(?:ece|electronics(?:\s+and\s+communication)?)\b"),
    ("ISE", r"\b(?:ise|information\s+science(?:\s+and\s+engineering)?)\b"),
    ("EEE", r"\b(?:eee|electrical(?:\s+and\s+electronics)?)\b"),
    ("AIML", r"\b(?:aiml|ai\s+(?:and|&)\s+ml)\b"),
    ("AIDS", r"\b(?:aids|ai\s+(?:and|&)\s+ds)\b"),
    ("MECH", r"\b(?:mech|mechanical(?:\s+engineering)?)\b"),
    ("CIVIL", r"\b(?:civil(?:\s+engineering)?)\b"),
]

# Pattern to detect unrecognised uppercase 2-4-letter abbreviations used as dept names
_UNKNOWN_DEPT_ABBREV_RE = re.compile(r"\b([A-Z]{2,4})\b")


def extract_comparison_departments(query: str) -> tuple[str, str] | tuple[str, None] | None:
    """Detects if query asks to compare two specific departments.

    Returns:
      (dept_a, dept_b) — both verified  → do the comparison.
      (dept_a, None)   — one known, one unknown/unrecognised → ask clarification.
      None             — not a comparison query at all.
    """
    q = (query or "").lower()
    q_orig = (query or "")
    comp_keywords = ("which is good", "which is better", "which one is good", "which one is better",
                     "compare", "difference", "versus", " vs ", " vs. ", " or ", "better between",
                     "choose between", "good cse", "good ece", "prefer")
    if not any(k in q for k in comp_keywords):
        return None

    found = []
    for code, pat in _DEPT_PATTERNS:
        if re.search(pat, q):
            if code not in found:
                found.append(code)

    if len(found) >= 2:
        return found[0], found[1]

    # One known dept found — check whether an unrecognised abbreviation is also present.
    # e.g. 'CSE or ESE?' → found=['CSE'], but 'ESE' is unknown.
    if len(found) == 1:
        # Look for uppercase abbreviations in the ORIGINAL query that weren't matched above.
        for abbrev in _UNKNOWN_DEPT_ABBREV_RE.findall(q_orig):
            if abbrev not in _KNOWN_DEPT_CODES and abbrev not in {"RNSIT", "VTU", "PhD", "HOD"}:
                # Unknown abbreviation alongside a known dept → signal for clarification.
                return found[0], None  # sentinel: second dept unknown

    return None


def _get_department_comparison_facts(dept_a: str, dept_b: str, user_interest: str | None = None) -> str:
    """Builds verified factual context for comparing two departments."""
    facts = {
        "CSE": (
            "Computer Science & Engineering (CSE):\n"
            "- Located in the CSE Block, HOD is Dr. Kiran Y.C., annual intake is 720 students.\n"
            "- VTU recognized PhD research center.\n"
            "- Focus areas & curriculum: Software engineering, algorithms, programming languages, data structures, "
            "computing systems, operating systems, and computer architecture fundamentals."
        ),
        "ECE": (
            "Electronics & Communication Engineering (ECE):\n"
            "- Located in the Main Campus with a recognized PhD research center.\n"
            "- Focus areas & curriculum: Hardware systems, VLSI design, embedded systems, electronic circuits, "
            "microcontrollers, telecommunications, and digital signal processing."
        ),
        "ISE": (
            "Information Science & Engineering (ISE):\n"
            "- Located in the CSE Block.\n"
            "- Focus areas & curriculum: Software systems, information architecture, data engineering, web applications, and network management."
        ),
        "EEE": (
            "Electrical & Electronics Engineering (EEE):\n"
            "- Equipped with state-of-the-art laboratory facilities.\n"
            "- Focus areas & curriculum: Electrical power systems, electrical machines, control systems, and electronic instrumentation."
        ),
        "MECH": (
            "Mechanical Engineering:\n"
            "- Located in the Mechanical Block, houses the Toyota Center of Excellence and recognized PhD research center.\n"
            "- Focus areas: Mechanical design, manufacturing, thermal engineering, robotics, and CAD/CAM."
        ),
        "CIVIL": (
            "Civil Engineering:\n"
            "- Located in the Civil Block with recognized PhD research center.\n"
            "- Focus areas: Structural engineering, geotechnical engineering, environmental engineering, and surveying."
        ),
        "AIML": (
            "Artificial Intelligence & Machine Learning (AIML):\n"
            "- Located in the CSE Block, HOD is Dr. Andhe Pallavi.\n"
            "- Focus areas: Artificial intelligence, machine learning algorithms, data analysis, and intelligent systems."
        ),
    }

    fact_a = facts.get(dept_a, f"{dept_a} Department at RNSIT.")
    fact_b = facts.get(dept_b, f"{dept_b} Department at RNSIT.")

    interest_context = f"\nVisitor's Stated Interest: {user_interest}\n" if user_interest else ""
    return (
        f"Verified Department Facts for Comparison:\n\n"
        f"{fact_a}\n\n"
        f"{fact_b}\n"
        f"{interest_context}\n"
        f"Comparison Guidelines:\n"
        f"- Explain the factual differences between {dept_a} and {dept_b} as relevant to the visitor's question and stated interest.\n"
        f"- State what each department specifically focuses on (e.g. curriculum, labs, systems).\n"
        f"- Do NOT make unsupported subjective claims like 'Both offer strong programs' or 'Both are excellent choices'.\n"
        f"- Do NOT declare one department objectively better.\n"
        f"- Let the visitor decide based on these factual differences."
    )


def is_department_interest_followup(query: str, state: dict, history: list | None) -> str | None:
    """Checks if the query is a user specifying an interest/area following a broad department discussion."""
    last_topic = (state.get("last_topic") or "").lower()
    last_kiosk_text = ""
    if history:
        for msg in reversed(history):
            spk, txt = parse_history_message(msg)
            if spk and spk.lower() in ("assistant", "kiosk"):
                last_kiosk_text = txt.lower()
                break

    is_dept_discussion = (
        "department" in last_topic
        or "branch" in last_topic
        or "which department" in last_kiosk_text
        or "what area" in last_kiosk_text
        or "interested in" in last_kiosk_text
        or "software and computing" in last_kiosk_text
    )
    if not is_dept_discussion:
        return None

    q_clean = (query or "").lower().strip().rstrip(".!?")
    _INTEREST_PATTERNS = {
        "hardware": "Hardware Engineering",
        "hardware engineering": "Hardware Engineering",
        "embedded": "Embedded Systems",
        "embedded systems": "Embedded Systems",
        "vlsi": "VLSI Design",
        "software": "Software Engineering",
        "software engineering": "Software Engineering",
        "coding": "Software Engineering",
        "programming": "Software Engineering",
        "ai": "Artificial Intelligence",
        "artificial intelligence": "Artificial Intelligence",
        "machine learning": "Machine Learning",
        "data science": "Data Science",
        "cyber security": "Cyber Security",
        "networking": "Computer Networks",
        "core": "Core Engineering",
        "robotics": "Robotics",
    }
    for pat, label in _INTEREST_PATTERNS.items():
        if q_clean == pat or q_clean == f"in {pat}" or q_clean == f"interested in {pat}" or q_clean == f"{pat} engineering":
            return label

    return None


def _get_interest_departments_fact(interest: str) -> str:
    """Verified factual context tailored to a specific student interest area."""
    if "hardware" in interest.lower() or "embedded" in interest.lower() or "vlsi" in interest.lower():
        return (
            "RNSIT Departments for Hardware Engineering:\n"
            "- Electronics & Communication Engineering (ECE) and Electrical & Electronics Engineering (EEE) focus directly on hardware systems, electronic circuits, embedded systems, microcontrollers, and VLSI design. ECE is located in the Main Campus with state-of-the-art labs and a PhD research center.\n"
            "- Computer Science & Engineering (CSE) focuses primarily on software and algorithms, but covers computer organization and computing hardware fundamentals.\n"
            "If you are deciding between branches like ECE and CSE, I can explain the specific differences between them."
        )
    elif "software" in interest.lower():
        return (
            "RNSIT Departments for Software Engineering:\n"
            "- Computer Science & Engineering (CSE) and Information Science & Engineering (ISE) focus directly on software development, algorithms, data structures, and computing systems. CSE has an annual intake of 720 students in the CSE Block.\n"
            "- Specialized branches like CSE (AI & ML), CSE (Data Science), and CSE (Cyber Security) offer targeted software curriculums."
        )
    return _get_broad_departments_fact()


def is_incomplete_or_garbled_query(query: str, q_normalized: str | None = None) -> bool:
    """Detects incomplete, truncated, or unintelligible utterances.
    
    Returns True ONLY when input is genuinely incomplete or garbled, asking
    for clarification without running RAG or returning canned unknowns.
    Valid short queries (e.g. 'Placements?', 'ECE?', 'What is RNSIT?') return False.
    """
    q = (query or "").strip()
    qn = (q_normalized or normalize_query(q)).strip().lower()
    if not qn:
        return True

    # 1. Valid checks: If entity detected, NOT incomplete
    try:
        from backend.entity_mapping import detect_entity
        if detect_entity(qn) is not None or detect_entity(q) is not None:
            return False
    except Exception:
        pass

    # Valid if broad department or weather or traffic or broad college evaluation
    if is_broad_department_query(qn) or is_broad_department_query(q):
        return False
    if is_weather_intent(qn) or is_traffic_intent(qn):
        return False
    if is_college_evaluation_query(qn) or is_college_evaluation_query(q):
        return False

    # Check for valid standalone keyword queries (e.g. 'placements', 'admissions', 'fees', 'hostel', etc.)
    _VALID_STANDALONE = {
        "placement", "placements", "admission", "admissions", "fee", "fees", "hostel", "hostels",
        "canteen", "library", "sports", "gym", "bus", "transport", "principal", "director",
        "chairman", "cse", "ise", "ece", "eee", "mech", "civil", "aiml", "aids", "mca", "mba",
        "rnsit", "college", "campus", "curriculum", "syllabus", "academics"
    }
    words = qn.split()
    if len(words) == 1 and words[0] in _VALID_STANDALONE:
        return False

    # SINGLE-WORD filler: a lone function word with no content is incomplete.
    # ("the", "a", "can", etc.) — these would slip past the multi-word checks below.
    _SINGLE_FILLER = {
        "the", "a", "an", "is", "are", "of", "to", "for", "in", "on", "at", "by",
        "with", "from", "and", "or", "so", "was", "were", "that", "this", "it",
        "its", "there", "here", "can", "could", "would", "will", "do", "does",
    }
    if len(words) == 1 and words[0] in _SINGLE_FILLER:
        return True

    # Trailing hyphen/dash/ellipsis indicates truncation (e.g. "Can you tell me about-")
    if re.search(r"[-—–]\s*$", q) or q.endswith("..."):
        return True

    # Incomplete prompt openers with no topic specified (e.g. "Can you tell me", "Tell me about")
    _INCOMPLETE_OPENER_RE = re.compile(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+|i\s+want\s+to\s+|would\s+you\s+)?"
        r"(?:tell\s+me|tell|explain|show\s+me|show|know|ask|give\s+me)(?:\s+about|\s+me)?$",
        re.IGNORECASE
    )
    if _INCOMPLETE_OPENER_RE.match(qn):
        return True

    # Dangling trailing stopwords in multi-word sentence fragments (e.g. "This is the")
    _DANGLING_STOPWORDS = {
        "the", "a", "an", "of", "about", "to", "for", "in", "on", "at", "by", "with",
        "from", "and", "or", "so", "is", "are", "was", "were", "that", "this", "my", "your"
    }
    if len(words) >= 2 and words[-1] in _DANGLING_STOPWORDS:
        return True

    # Short fragments (<= 3 words) with no substantive RNSIT domain signal or interrogative
    # e.g. "The world.", "The world", "A person", "This thing", "World"
    _INTERROGATIVE_STARTS = (
        "who", "what", "where", "when", "why", "how", "which",
        "is", "are", "do", "does", "did", "can", "could", "would",
    )
    has_domain_word = bool(set(words) & DOMAIN_KEYWORDS)
    starts_interrogative = any(qn.startswith(s + " ") or qn == s for s in _INTERROGATIVE_STARTS)
    if len(words) <= 3 and not has_domain_word and not starts_interrogative:
        return True

    # Garbled repetitions / disordered syntax
    _GARBLED_PATTERNS = (
        r"\bis\s+what\s+is\b",
        r"\bwhat\s+is\s+what\s+is\b",
        r"\bthe\s+is\s+the\b",
        r"\bname\s+is\s+what\s+is\b",
        r"\bthis\s+is\s+the\b",
    )
    for pat in _GARBLED_PATTERNS:
        if re.search(pat, qn):
            return True

    # Utterances composed purely of filler/function words with no substantive noun or verb
    _PURE_FILLER_WORDS = {
        "this", "that", "the", "a", "an", "is", "are", "was", "were", "what", "where",
        "how", "who", "which", "name", "it", "its", "there", "here"
    }
    if len(words) >= 2 and all(w in _PURE_FILLER_WORDS for w in words):
        return True

    return False


def is_emotional_reengagement_response(text: str) -> bool:
    """Returns True only when the user's text is a clear emotional/wellbeing
    response to the re-engagement greeting 'How are you doing today?'.

    Uses POSITIVE detection: the text must match an emotional pattern.
    Anything that doesn't positively match is treated as a new question
    and falls through to normal routing.

    Matches:  'I'm good', 'Fine', 'Stressed', 'Not very good', 'Pretty bad'...
    Non-matches: 'Who is the principal?', 'Did I ask you...', 'Which department?'
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    # Pattern 1: starts with a personal pronoun + emotional state
    # 'i am good', "i'm stressed", 'im fine'
    _EMOTION_STATE_WORDS = (
        "good", "well", "fine", "great", "okay", "ok", "alright", "alrite",
        "bad", "not good", "not great", "not okay", "not well", "not fine",
        "stress", "stressed", "anxious", "worried", "nervous", "overwhelmed",
        "exhausted", "tired", "hectic", "rough",
        "sad", "low", "depressed", "unhappy", "terrible", "awful",
        "frustrated", "annoyed", "irritated", "angry", "upset",
        "excited", "pumped", "thrilled", "wonderful", "fantastic", "amazing",
        "happy", "pleased", "cheerful",
        "okay", "so so", "normal", "same old", "pretty good", "pretty bad",
        "doing well", "doing good", "doing fine", "doing great", "doing okay",
        "not too bad", "not bad", "quite good", "fairly good",
    )
    # Short 1-3 word emotional replies without any interrogative structure
    _SHORT_EMOTIONAL = {
        "good", "fine", "well", "great", "okay", "ok", "alright",
        "bad", "tired", "stressed", "happy", "sad", "exhausted",
        "not good", "not bad", "not great", "not okay",
        "pretty good", "so so", "quite good", "doing well",
    }
    if t in _SHORT_EMOTIONAL:
        return True

    # Pattern 2: "I'm <state>", "I am <state>", "Im <state>"
    for prefix in ("i'm ", "im ", "i am ", "i feel ", "feeling "):
        if t.startswith(prefix):
            rest = t[len(prefix):].strip()
            if rest and len(rest.split()) <= 4:  # short after prefix = emotional, not a new question
                return True

    # Pattern 3: explicit emotion word at start of short sentence (<= 5 words, no interrogative start)
    interrogative_starts = ("who", "what", "where", "when", "why", "how", "which",
                            "can", "could", "would", "did", "do", "does", "is", "are",
                            "tell", "please", "show", "find", "explain")
    if any(t.startswith(s) for s in interrogative_starts):
        return False  # interrogative opener → definitely a new question

    words_t = t.split()
    if len(words_t) <= 5:
        for state in _EMOTION_STATE_WORDS:
            if state in t:
                return True

    return False


def is_college_evaluation_query(query: str) -> bool:
    """Returns True for broad 'Is RNSIT a good college?' style evaluation queries.

    These must use verified college overview context, NOT random RAG chunks
    about buses/ATMs/clubs.
    """
    q = (query or "").lower().strip()
    # Patterns: 'is rnsit a good college', 'is rnsit good', 'rnsit good college',
    # 'is rns it a good college', 'is rnsit worth it', 'is rnsit recommended'
    patterns = (
        r"\brnsit\s+(?:a\s+)?good\s+college\b",
        r"\brns\s+(?:it\s+)?(?:a\s+)?good\s+college\b",
        r"\bis\s+rnsit\s+good\b",
        r"\bis\s+rns\s+(?:it\s+)?good\b",
        r"\bis\s+rnsit\s+(?:a\s+)?(?:good|great|decent|worth|best|top|reputed|recommended)\b",
        r"\bis\s+rns\s+(?:it\s+)?(?:a\s+)?(?:good|great|decent|worth|best|top|reputed|recommended)\b",
        r"\brnsit\s+is\s+(?:a\s+)?(?:good|great|decent|nice|best|top|reputed|recommended)\b",
        r"\brnsit\s+(?:is\s+)?(?:a\s+)?(?:good|great|decent|nice|best|top|reputed|recommended)\b",
        r"\b(?:good|great|best|top|reputed)\s+(?:college|institute).*rnsit\b",
        r"\brnsit\s+worth\s+(?:it|joining|studying)\b",
        r"\bshould\s+i\s+(?:join|choose|go to|study at)\s+rnsit\b",

    )
    for pat in patterns:
        if re.search(pat, q):
            return True
    return False


def _get_broad_departments_fact() -> str:
    """Builds a rich, factual overview of RNSIT departments and streams for broad synthesis."""
    try:
        kb = get_kb_data() or {}
        placements = kb.get("placements", {})
        total_cos = placements.get("total_companies", "200+")
        highest = placements.get("stats", {}).get("2025", {}).get("highest_ctc_lpa", "26.1")
        avg = placements.get("stats", {}).get("2025", {}).get("average_ctc_lpa", "6.5")
    except Exception:
        total_cos, highest, avg = "200+", "26.1", "6.5"

    return (
        "RNSIT Departments and Academic Offerings:\n"
        "- Software & Computing Branches: Computer Science & Engineering (CSE - annual intake 720, located in CSE Block, HOD Dr. Kiran Y.C., VTU PhD research center), Information Science & Engineering (ISE, located in CSE Block), Artificial Intelligence & Machine Learning (AIML, HOD Dr. Andhe Pallavi), Artificial Intelligence & Data Science (AIDS), CSE Data Science, and CSE Cyber Security. These branches focus on software development, algorithms, artificial intelligence, and computing systems.\n"
        "- Electronics & Electrical Branches: Electronics & Communication Engineering (ECE - located in Main Campus with PhD research center) and Electrical & Electronics Engineering (EEE - state-of-the-art labs). These branches focus on embedded systems, telecommunications, VLSI, and electrical systems.\n"
        "- Core Engineering Branches: Mechanical Engineering (located in Mechanical Block, houses the Toyota Center of Excellence and PhD center) and Civil Engineering (located in Civil Block with recognized PhD research center).\n"
        "- Postgraduate Programs: Master of Computer Applications (MCA) and Master of Business Administration (MBA).\n"
        f"- Placements Across Departments: Over {total_cos} companies recruit from RNSIT (highest CTC {highest} LPA, average CTC {avg} LPA) including recruiters like Adobe, Amazon, Cisco, Cognizant, Infosys, and IBM. Campus-wide placement training and opportunities are available across engineering disciplines."
    )


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
    "Use the verified campus facts below to answer the visitor's question:\n\n"
    "Facts:\n{context}\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Never start with a greeting or self-introduction — answer directly.\n"
    "2. Keep the answer concise (2-3 sentences maximum).\n"
    "3. Ground your response in the facts provided above. Synthesize the facts naturally to address the question.\n"
    "4. For broad, subjective, or comparison questions (e.g. 'Which department is good?', 'Which branch should I choose?', 'Which department is better for software?'): "
    "summarize the relevant options from the facts (for example, software/computing branches like CSE/ISE vs electronics like ECE/EEE or core engineering) "
    "and ask what area or career path they are interested in so you can help them compare. Do NOT claim one department is objectively 'best' unless the facts explicitly say so.\n"
    "5. CRITICAL: Never casually say 'I don't have that detail on hand', 'I don't know', or 'I don't have information' when relevant facts are present in the context above. Synthesize the best grounded answer possible.\n"
    "6. Only if the provided facts contain zero relevant information about the question should you say: 'I don't have that detail on hand — please check with the Admin Block.'\n"
    "7. Never output literal 'Q:' / 'A:' labels.\n"
    "8. For comparisons between specific departments or branches (e.g. 'Which is good, CSE or ECE?'), use the visitor's stated interest if provided, explain the factual differences relevant to that interest using only the verified facts, and let the visitor decide. Never make unsupported generic claims like 'Both offer strong programs', 'Both are excellent choices', 'CSE is better', or 'ECE is better'."
)


def _naturalize_rag_only_fallback(context_text: str, max_chars: int = 450) -> str:
    """
    Tier-3 (RAG-only, no LLM available) fallback formatter — used only when
    BOTH LLM tiers (local Qwen + Gemini) have failed and there's no model
    left to phrase an answer, so this has to work from the raw retrieved
    context alone. Fixes two production bugs that were both living in this
    one spot (duplicated at both call sites below):

    1. Previously prefixed the answer with the literal internal phrase
       "Based on what I have on file: ..." — an implementation detail that
       leaked straight to the visitor. Removed; this returns the cleaned
       fact text directly, as Nova would actually say it.

    2. Previously took ONLY the first "\\n\\n"-separated chunk of
       context_text (`context_text.split("\\n\\n")[0]`). context_text is the
       concatenation of the top-K retrieved chunks (see
       retrieve_relevant_context in llm.py), ordered by score — so for a
       broad query like "what facilities does the college provide?", a
       single specific facility (e.g. "Canteen") could get returned ALONE
       even when a proper aggregate chunk ("Campus Facilities Overview")
       was ALSO present in context_text, just not ranked first. This now
       walks every retrieved chunk (each cleaned of its "Q:"/"A:" and
       "Facility:" labels), dedupes, and aggregates as many as fit under
       max_chars — so a broad/category question is represented properly
       instead of being truncated to whichever chunk happened to rank
       first.
    """
    if not context_text:
        return ""

    seen: set[str] = set()
    cleaned_chunks: list[str] = []
    for raw_chunk in context_text.split("\n\n"):
        cleaned = _QA_LABEL_RE_LLM.sub("", raw_chunk).strip()
        cleaned = _FACILITY_LABEL_RE_LLM.sub("", cleaned).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_chunks.append(cleaned)

    out = ""
    for chunk in cleaned_chunks:
        candidate = f"{out} {chunk}".strip() if out else chunk
        if out and len(candidate) > max_chars:
            break
        out = candidate
        if len(out) > max_chars:
            out = out[:max_chars].rsplit(" ", 1)[0].strip()
            break
    return out


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
        cleaned = _clean_repetitive_greeting((text or "").strip())
        # Critical Fallback Rule: if context exists, do not let LLM casually refuse
        if context_text and len(context_text.strip()) > 30 and any(ref in cleaned.lower() for ref in ("i don't have that detail", "i don't know", "i do not know", "i don't have information")):
            fallback_synth = _naturalize_rag_only_fallback(context_text)
            if fallback_synth:
                cleaned = fallback_synth
        return cleaned, tier.upper()
    except Exception as e:
        logger.error("[CONF-RAG] Local+Gemini both failed, using RAG-only nearest context: %s", e)
        fallback = _naturalize_rag_only_fallback(context_text)
        if fallback:
            return fallback, "RAG_ONLY"
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

            # Comparison query override: queries asking to compare or choose between branches/departments
            # are not ambiguous ambiguities; they should be synthesized together by the LLM.
            if ambiguous:
                comparison_words = ("compare", "difference", "between", "versus", " vs ", " or ", "which")
                if any(w in question.lower() for w in comparison_words):
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

    # ── 1.2) Incomplete / Garbled Utterance Check ─────────────────────────
    if is_incomplete_or_garbled_query(question, question_clean):
        clarify_ans = "Sure, what would you like to know?"
        logger.info("[CONF-RAG] Incomplete/garbled query detected: %r -> clarification", question)
        state["pending"] = None
        state["clarify_streak"] = 0
        return {
            "answer": clarify_ans,
            "route": "CLARIFICATION_INCOMPLETE",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "clarification_incomplete",
        }

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

    # ── 1.7) Canonical Entity Detection (Standalone topic check): ─────────
    # If the user asks directly about a verified entity (e.g. 'Who is the principal?'),
    # route to that entity without forcing previous unrelated department context into it.
    detected = detect_entity(question_clean) or detect_entity(question)
    if detected:
        last_top = (state.get("last_topic") or "").lower()
        is_dept_earlier = any(d in last_top for d in ("department", "cse", "ece", "hardware", "software", "branch"))
        is_new_unrelated = (
            is_dept_earlier
            and detected.entity_id in ("PRINCIPAL", "DIRECTOR", "CHAIRMAN", "ADMIN_BLOCK", "CANTEEN", "LIBRARY", "SPORTS_GROUND", "GYM")
        )
        history_for_gen = None if is_new_unrelated else history

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

        answer, gen_tier = await _generate_answer(question, detected.verified_answer, history_for_gen)
        if not answer or any(p in answer.lower() for p in ("i don't have that detail", "i don't know", "i do not know", "i don't have information")):
            answer = detected.verified_answer

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
            "answer": answer,
            "route": f"HIGH_ENTITY_{detected.entity_id}",
            "session_action": "CONTINUE",
            "session_state": state,
            "detected_entity": detected.entity_id,
            "entity_confidence": detected.confidence,
            "source": "entity_kb",
        }

    # ── 1.8) Specific Department Comparison Routing (e.g. 'Which is good, CSE or ECE?') ──
    comparison_depts = extract_comparison_departments(question_clean) or extract_comparison_departments(question)
    if comparison_depts is not None:
        dept_a, dept_b = comparison_depts

        # dept_b is None → one department is unrecognised/unknown (e.g. 'ESE').
        # Never invent what an abbreviation means — ask a short clarification instead.
        if dept_b is None:
            clarify_answer = (
                f"I know {dept_a} but I'm not sure which department you mean by the other abbreviation. "
                f"Did you perhaps mean ECE, ISE, EEE, or another department?"
            )
            logger.info("[CONF-RAG] Comparison: known=%s, unknown dept abbrev → clarification", dept_a)
            state["pending"] = None
            state["clarify_streak"] = 0
            return {
                "answer": clarify_answer,
                "route": "CLARIFICATION_UNKNOWN_DEPT",
                "session_action": "CONTINUE",
                "session_state": state,
                "source": "clarification_unknown_dept",
            }

        user_interest = state.get("user_interest")
        # Also check recent history if user previously stated an interest
        if not user_interest and history:
            for h_msg in reversed(history[-4:]):
                _, h_txt = parse_history_message(h_msg)
                if h_txt and "hardware" in h_txt.lower():
                    user_interest = "Hardware Engineering"
                    state["user_interest"] = user_interest
                    break

        comp_facts = _get_department_comparison_facts(dept_a, dept_b, user_interest=user_interest)
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print(f"DETECTED INTENT: DEPARTMENT_COMPARISON ({dept_a} vs {dept_b})")
        print(f"USER INTEREST: {user_interest}")
        print(f"RETRIEVAL RESULT: {comp_facts[:120]}")
        print("ANSWER SOURCE: department_comparison_kb")

        logger.info("[CONF-RAG] Comparison query: %s vs %s, interest=%r", dept_a, dept_b, user_interest)
        answer, gen_tier = await _generate_answer(question, comp_facts, history)
        if not answer or any(p in answer.lower() for p in ("i don't have that detail", "i don't know", "i do not know")):
            if "hardware" in (user_interest or "").lower():
                answer = (
                    "Electronics and Communication Engineering (ECE) focuses on hardware systems, VLSI design, electronic circuits, and embedded systems, located in the Main Campus with specialized labs. "
                    "Computer Science and Engineering (CSE) focuses on software engineering, algorithms, and computing systems, with introductory hardware concepts. "
                    "Depending on whether you prefer hands-on electronics hardware or software development, both paths offer distinct career options."
                )
            else:
                answer = (
                    f"{dept_a} and {dept_b} each have distinct focus areas. "
                    "I recommend choosing based on whether your primary interest is software, hardware, or another engineering field."
                )

        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = f"{dept_a} vs {dept_b} Comparison"
        return {
            "answer": answer,
            "route": f"HIGH_DEPARTMENT_COMPARISON_{gen_tier}",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": f"dept_comparison_{gen_tier.lower()}",
        }

    # ── 1.9) Broad Department Query Check (runs BEFORE generic RAG search) ──
    # Prevents generic RAG chunks (clubs, academics) from contaminating broad department context.
    if is_broad_department_query(question_clean) or is_broad_department_query(question):
        broad_dept_context = _get_broad_departments_fact()
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: BROAD_DEPARTMENT")
        print("DETECTED ENTITY: DEPARTMENTS_OVERVIEW")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {broad_dept_context[:120]}")
        print("ANSWER SOURCE: broad_department_rag")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] DETECTED INTENT: BROAD_DEPARTMENT")
        logger.info("[CONF-RAG] ANSWER SOURCE: broad_department_rag")

        answer, gen_tier = await _generate_answer(question, broad_dept_context, history)
        if not answer or any(ref in answer.lower() for ref in ("i don't have that detail", "i don't know", "i do not know")):
            answer = (
                "RNSIT has several departments across software, electronics, and core engineering. "
                "If you're interested in software and computing, CSE or ISE may be relevant, while ECE and EEE focus more on electronics and electrical fields. "
                "If you tell me what area you're interested in, I can help you compare them."
            )

        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = "Departments Overview"
        return {
            "answer": answer,
            "route": f"HIGH_BROAD_DEPT_{gen_tier}",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": f"broad_dept_{gen_tier.lower()}",
        }

    # ── 1.10) Department Follow-up: User specifies an interest area ────────
    dept_interest = is_department_interest_followup(question_clean, state, history) or is_department_interest_followup(question, state, history)
    if dept_interest:
        state["user_interest"] = dept_interest
        interest_facts = _get_interest_departments_fact(dept_interest)

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print(f"DETECTED INTENT: BROAD_DEPARTMENT_FOLLOWUP ({dept_interest})")
        print("ANSWER SOURCE: interest_followup_kb")

        logger.info("[CONF-RAG] Department interest follow-up: %s", dept_interest)
        answer, gen_tier = await _generate_answer(question, interest_facts, history)
        if not answer or any(ref in answer.lower() for ref in ("i don't have that detail", "i don't know", "i do not know")):
            answer = interest_facts

        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = f"Department Interest: {dept_interest}"
        return {
            "answer": answer,
            "route": f"HIGH_BROAD_DEPT_FOLLOWUP_{gen_tier}",
            "session_action": "CONTINUE",
            "session_state": state,
            "source": f"dept_interest_{gen_tier.lower()}",
        }

    # 2) Context-aware query condensing for remaining follow-up questions
    search_query = await condense_query(question_clean, history or [])
    if search_query != question_clean:
        detected_search = detect_entity(search_query)
        if detected_search:
            answer, gen_tier = await _generate_answer(question, detected_search.verified_answer, history)
            state["pending"] = None
            state["clarify_streak"] = 0
            state["last_topic"] = detected_search.canonical_name
            return {
                "answer": answer,
                "route": f"HIGH_ENTITY_{detected_search.entity_id}",
                "session_action": "CONTINUE",
                "session_state": state,
                "detected_entity": detected_search.entity_id,
                "entity_confidence": detected_search.confidence,
                "source": "entity_kb",
            }

    # 3) RAG search (single retrieval call feeds confidence routing,
    #    ambiguity check, AND the scope check below — no repeat calls).
    context_text, best_score, raw_results = await retrieve_relevant_context(
        search_query, top_k=RAG_TOP_K
    )
    best_score = safe_float(best_score)

    in_domain = (
        _is_in_domain_keyword(question_clean)
        or _is_in_domain_keyword(question)
        or _is_in_domain_keyword(search_query)
    )

    # ── Broad RNSIT overview / college-evaluation queries ─────────────────
    # These must use verified college_overview context, NOT random RAG chunks.
    broad_overview_phrases = (
        "about rnsit", "something about rnsit", "about college", "tell me about rnsit",
        "about the college", "college overview", "what is rnsit", "tell me something about rnsit",
        "tell me about rns", "about campus", "tell me about this college",
        "about rns institute", "about r n s", "give me an overview", "overview of rnsit",
        "overview of rns", "what is rns institute", "what is rns", "about this institute",
        "about this college", "what does rnsit", "what does rns institute",
        "introduce rnsit", "introduce rns", "know about rnsit", "know about this college",
        "information about rnsit", "info about rnsit", "general info about rnsit",
    )
    is_broad_rnsit = (
        any(p in question_clean for p in broad_overview_phrases)
        or is_college_evaluation_query(question_clean)
        or is_college_evaluation_query(question)
    )

    if is_broad_rnsit:
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

    # Out-of-scope / weak-query protection:
    # If the query contains NO campus domain keywords, DO NOT let generic RAG retrieval
    # produce an answer from a weak semantic match.
    if not in_domain:
        words_q = question_clean.split()
        if len(words_q) <= 4:
            clarify_ans = "Sure, what would you like to know?"
            logger.info("[CONF-RAG] Weak/unclear query with no domain keywords: %r -> clarification", question)
            state["pending"] = None
            state["clarify_streak"] = 0
            return {
                "answer": clarify_ans,
                "route": "CLARIFICATION_INCOMPLETE",
                "session_action": "CONTINUE",
                "session_state": state,
                "source": "clarification_incomplete",
            }

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

        return await _handle_low(question, best_score, session_id, face_id, state, settings)

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

    if not raw_results:
        return await _handle_low(question, best_score, session_id, face_id, state, settings)

    # ── LOW confidence with ambiguous/unrelated retrieval ────────────────
    # Prevent low-relevance generic chunks (buses, ATM, clubs) from becoming
    # answers for queries that had weak retrieval signal.
    # If best_score is HIGH-band but the top chunk entity doesn't semantically
    # relate to the query, let the normal HIGH handler proceed — ambiguity
    # check will handle competing chunks. Only intercept truly weak retrievals.
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

    # ── 1.2) Incomplete / Garbled Utterance Check ─────────────────────────
    if is_incomplete_or_garbled_query(question, question_clean):
        clarify_ans = "Sure, what would you like to know?"
        logger.info("[CONF-RAG-STREAM] Incomplete/garbled query detected: %r -> clarification", question)
        yield {"sentence": clarify_ans, "partial": False}
        state["pending"] = None
        state["clarify_streak"] = 0
        yield {
            "done": True,
            "answer": clarify_ans,
            "session_action": "CONTINUE",
            "session_state": state,
            "source": "clarification_incomplete",
        }
        return

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

    # ── 1.7) Canonical Entity Detection (Standalone topic check): ─────────
    detected = detect_entity(question_clean) or detect_entity(question)
    if detected:
        last_top = (state.get("last_topic") or "").lower()
        is_dept_earlier = any(d in last_top for d in ("department", "cse", "ece", "hardware", "software", "branch"))
        is_new_unrelated = (
            is_dept_earlier
            and detected.entity_id in ("PRINCIPAL", "DIRECTOR", "CHAIRMAN", "ADMIN_BLOCK", "CANTEEN", "LIBRARY", "SPORTS_GROUND", "GYM")
        )
        history_for_gen = None if is_new_unrelated else history

        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: RNSIT")
        print(f"DETECTED ENTITY: {detected.entity_id}")
        print(f"ENTITY CONFIDENCE: {detected.confidence:.2f}")
        print(f"RETRIEVAL RESULT: {detected.verified_answer[:120]}")
        print("ANSWER SOURCE: entity_kb")

        # QUESTION + RETRIEVED CONTEXT -> LLM streaming synthesis
        system_prompt = _SYSTEM_PROMPT_TMPL.format(context=detected.verified_answer)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history_for_gen or [])[-4:]:
            speaker, text = parse_history_message(msg)
            if speaker and text:
                role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})

        parts: list[str] = []
        first_sentence_sent = False
        tier_seen = "local"
        try:
            buf = ""
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
                        yield {"answer": s, "route": f"HIGH_ENTITY_{detected.entity_id}",
                               "session_action": "CONTINUE", "partial": True}
            if buf.strip():
                b = buf.strip()
                if not first_sentence_sent:
                    b = _clean_repetitive_greeting(b)
                if b:
                    parts.append(b)
                    yield {"answer": b, "route": f"HIGH_ENTITY_{detected.entity_id}",
                           "session_action": "CONTINUE", "partial": True}
            full_ans = "".join(parts).strip()
            if not full_ans or any(p in full_ans.lower() for p in ("i don't have that detail", "i don't know", "i do not know", "i don't have information")):
                full_ans = detected.verified_answer
                if not parts:
                    yield {"answer": full_ans, "route": f"HIGH_ENTITY_{detected.entity_id}", "partial": False}
        except Exception:
            full_ans = detected.verified_answer
            yield {"sentence": full_ans, "partial": False}

        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = detected.canonical_name
        yield {
            "done": True,
            "answer": full_ans,
            "session_action": "CONTINUE",
            "session_state": state,
            "detected_entity": detected.entity_id,
            "entity_confidence": detected.confidence,
            "source": "entity_kb",
        }
        return

    # ── 1.8) Specific Department Comparison Routing (streaming) ───────────
    comparison_depts = extract_comparison_departments(question_clean) or extract_comparison_departments(question)
    if comparison_depts is not None:
        dept_a, dept_b = comparison_depts

        # Unknown/unrecognised second dept → clarify instead of inventing
        if dept_b is None:
            clarify_answer = (
                f"I know {dept_a} but I'm not sure which department you mean by the other abbreviation. "
                f"Did you perhaps mean ECE, ISE, EEE, or another department?"
            )
            logger.info("[CONF-RAG-STREAM] Comparison: known=%s, unknown dept → clarification", dept_a)
            state["pending"] = None
            state["clarify_streak"] = 0
            yield {"answer": clarify_answer, "route": "CLARIFICATION_UNKNOWN_DEPT",
                   "session_action": "CONTINUE", "session_state": state}
            return

        user_interest = state.get("user_interest")
        if not user_interest and history:
            for h_msg in reversed(history[-4:]):
                _, h_txt = parse_history_message(h_msg)
                if h_txt and "hardware" in h_txt.lower():
                    user_interest = "Hardware Engineering"
                    state["user_interest"] = user_interest
                    break

        comp_facts = _get_department_comparison_facts(dept_a, dept_b, user_interest=user_interest)
        system_prompt = _SYSTEM_PROMPT_TMPL.format(context=comp_facts)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or [])[-4:]:
            speaker, text = parse_history_message(msg)
            if speaker and text:
                role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})

        parts = []
        first_sentence_sent = False
        tier_seen = "local"
        try:
            buf = ""
            async for delta, tier, _model in chat_completion_with_fallback_stream(messages, temperature=0.2, max_tokens=180):
                tier_seen = tier
                buf += delta
                ready, buf = _pop_complete_sentences(buf)
                for s in ready:
                    if not first_sentence_sent:
                        s = _clean_repetitive_greeting(s)
                        first_sentence_sent = True
                    if s:
                        parts.append(s)
                        yield {"answer": s, "route": f"HIGH_DEPARTMENT_COMPARISON_{tier_seen.upper()}", "session_action": "CONTINUE", "partial": True}
            if buf.strip():
                b = _clean_repetitive_greeting(buf.strip()) if not first_sentence_sent else buf.strip()
                if b:
                    parts.append(b)
                    yield {"answer": b, "route": f"HIGH_DEPARTMENT_COMPARISON_{tier_seen.upper()}", "session_action": "CONTINUE", "partial": True}
        except Exception:
            pass

        full_answer = "".join(parts).strip()
        if not full_answer or any(p in full_answer.lower() for p in ("i don't have that detail", "i don't know", "i do not know")):
            if "hardware" in (user_interest or "").lower():
                full_answer = (
                    "Electronics and Communication Engineering (ECE) focuses on hardware systems, VLSI design, electronic circuits, and embedded systems, located in the Main Campus with specialized labs. "
                    "Computer Science and Engineering (CSE) focuses on software engineering, algorithms, and computing systems, with introductory hardware concepts. "
                    "Depending on whether you prefer hands-on electronics hardware or software development, both paths offer distinct career options."
                )
            else:
                full_answer = f"CSE focuses on software development and computing systems, while {dept_b} focuses on its specific field. I recommend choosing based on whether your primary interest is software or hardware."

        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = f"{dept_a} vs {dept_b} Comparison"
        yield {
            "done": True,
            "answer": full_answer,
            "session_action": "CONTINUE",
            "session_state": state,
            "source": f"dept_comparison_{tier_seen.lower()}",
        }
        return

    # ── 1.9) Broad Department Query Check (streaming, runs BEFORE generic RAG) ──
    if is_broad_department_query(question_clean) or is_broad_department_query(question):
        broad_dept_context = _get_broad_departments_fact()
        print(f"USER QUERY: {question}")
        print(f"NORMALIZED QUERY: {question_clean}")
        print("DETECTED INTENT: BROAD_DEPARTMENT")
        print("DETECTED ENTITY: DEPARTMENTS_OVERVIEW")
        print("ENTITY CONFIDENCE: 1.00")
        print(f"RETRIEVAL RESULT: {broad_dept_context[:120]}")
        print("ANSWER SOURCE: broad_department_rag")

        logger.info("[CONF-RAG] USER QUERY: %s", question)
        logger.info("[CONF-RAG] DETECTED INTENT: BROAD_DEPARTMENT")
        logger.info("[CONF-RAG] ANSWER SOURCE: broad_department_rag")

        system_prompt = _SYSTEM_PROMPT_TMPL.format(context=broad_dept_context)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or [])[-4:]:
            speaker, text = parse_history_message(msg)
            if speaker and text:
                role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})

        parts: list[str] = []
        first_sentence_sent = False
        tier_seen = "local"
        try:
            buf = ""
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
                        yield {"answer": s, "route": f"HIGH_BROAD_DEPT_{tier_seen.upper()}",
                               "session_action": "CONTINUE", "partial": True}
            if buf.strip():
                b = buf.strip()
                if not first_sentence_sent:
                    b = _clean_repetitive_greeting(b)
                if b:
                    parts.append(b)
                    yield {"answer": b, "route": f"HIGH_BROAD_DEPT_{tier_seen.upper()}",
                           "session_action": "CONTINUE", "partial": True}
        except Exception as e:
            logger.error("[CONF-RAG-STREAM] Broad dept generation failed, using fallback: %s", e)
            final_fallback = (
                "RNSIT has several departments across software, electronics, and core engineering. "
                "If you're interested in software and computing, CSE or ISE may be relevant, while ECE and EEE focus more on electronics and electrical fields. "
                "If you tell me what area you're interested in, I can help you compare them."
            )
            state["pending"] = None
            state["clarify_streak"] = 0
            state["last_topic"] = "Departments Overview"
            yield {"answer": final_fallback, "route": "HIGH_BROAD_DEPT_FALLBACK",
                   "session_action": "CONTINUE", "session_state": state}
            return

        full_answer = "".join(parts).strip()
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = "Departments Overview"
        yield {"answer": full_answer, "route": f"HIGH_BROAD_DEPT_{tier_seen.upper()}",
               "session_action": "CONTINUE", "session_state": state, "final": True}
        return

    # ── 1.10) Department Interest Follow-up (streaming) ───────────────────
    dept_interest = is_department_interest_followup(question_clean, state, history) or is_department_interest_followup(question, state, history)
    if dept_interest:
        state["user_interest"] = dept_interest
        interest_facts = _get_interest_departments_fact(dept_interest)
        system_prompt = _SYSTEM_PROMPT_TMPL.format(context=interest_facts)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or [])[-4:]:
            speaker, text = parse_history_message(msg)
            if speaker and text:
                role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})

        parts = []
        tier_seen = "local"
        try:
            buf = ""
            async for delta, tier, _model in chat_completion_with_fallback_stream(messages, temperature=0.2, max_tokens=180):
                tier_seen = tier
                buf += delta
                ready, buf = _pop_complete_sentences(buf)
                for s in ready:
                    if s:
                        parts.append(s)
                        yield {"answer": s, "route": f"HIGH_BROAD_DEPT_FOLLOWUP_{tier_seen.upper()}", "session_action": "CONTINUE", "partial": True}
            if buf.strip():
                parts.append(buf.strip())
                yield {"answer": buf.strip(), "route": f"HIGH_BROAD_DEPT_FOLLOWUP_{tier_seen.upper()}", "session_action": "CONTINUE", "partial": True}
        except Exception:
            pass

        full_answer = "".join(parts).strip()
        if not full_answer:
            full_answer = interest_facts
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = f"Department Interest: {dept_interest}"
        yield {"answer": full_answer, "route": f"HIGH_BROAD_DEPT_FOLLOWUP_{tier_seen.upper()}", "session_action": "CONTINUE", "session_state": state, "final": True}
        return

    search_query = await condense_query(question_clean, history or [])

    # ── College evaluation / broad RNSIT overview (streaming) ────────────
    # 'Is RNSIT a good college?' must use verified overview, not generic RAG.
    broad_overview_phrases = (
        "about rnsit", "something about rnsit", "about college", "tell me about rnsit",
        "about the college", "college overview", "what is rnsit", "tell me something about rnsit",
        "tell me about rns", "about campus", "tell me about this college",
        "about rns institute", "about r n s", "give me an overview", "overview of rnsit",
        "overview of rns", "what is rns institute", "what is rns", "about this institute",
        "about this college", "what does rnsit", "what does rns institute",
        "introduce rnsit", "introduce rns", "know about rnsit", "know about this college",
        "information about rnsit", "info about rnsit", "general info about rnsit",
    )
    is_broad_rnsit_stream = (
        any(p in question_clean for p in broad_overview_phrases)
        or is_college_evaluation_query(question_clean)
        or is_college_evaluation_query(question)
    )
    if is_broad_rnsit_stream:
        overview_fact = _get_college_overview_fact()
        system_prompt = _SYSTEM_PROMPT_TMPL.format(context=overview_fact)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in (history or [])[-4:]:
            speaker, text = parse_history_message(msg)
            if speaker and text:
                role = "user" if speaker.lower() in ("visitor", "user") else "assistant"
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": question})

        parts: list[str] = []
        first_sentence_sent = False
        tier_seen = "local"
        try:
            buf = ""
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
                        yield {"answer": s, "route": f"RNSIT_GENERAL",
                               "session_action": "CONTINUE", "partial": True}
            if buf.strip():
                b = buf.strip()
                if not first_sentence_sent:
                    b = _clean_repetitive_greeting(b)
                if b:
                    parts.append(b)
                    yield {"answer": b, "route": "RNSIT_GENERAL",
                           "session_action": "CONTINUE", "partial": True}
        except Exception as e:
            logger.error("[CONF-RAG-STREAM] College overview generation failed: %s", e)
            parts = [overview_fact]
            yield {"answer": overview_fact, "route": "RNSIT_GENERAL", "partial": False}

        full_answer = "".join(parts).strip() or overview_fact
        state["pending"] = None
        state["clarify_streak"] = 0
        state["last_topic"] = "College Overview"
        yield {"answer": full_answer, "route": "RNSIT_GENERAL",
               "session_action": "CONTINUE", "session_state": state, "final": True}
        return

    context_text, best_score, raw_results = await retrieve_relevant_context(search_query, top_k=RAG_TOP_K)
    best_score = safe_float(best_score)

    in_domain = (
        _is_in_domain_keyword(question_clean)
        or _is_in_domain_keyword(question)
        or _is_in_domain_keyword(search_query)
    )
    if not in_domain:
        words_q = question_clean.split()
        if len(words_q) <= 4:
            clarify_ans = "Sure, what would you like to know?"
            logger.info("[CONF-RAG-STREAM] Weak/unclear query with no domain keywords: %r -> clarification", question)
            yield {"sentence": clarify_ans, "partial": False}
            state["pending"] = None
            state["clarify_streak"] = 0
            yield {
                "done": True,
                "answer": clarify_ans,
                "session_action": "CONTINUE",
                "session_state": state,
                "source": "clarification_incomplete",
            }
            return
        yield await _handle_low(question, best_score, session_id, face_id, state, settings)
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
                comparison_words = ("compare", "difference", "between", "versus", " vs ", " or ", "which")
                if any(w in question.lower() for w in comparison_words):
                    ambiguous = False
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
        fallback = _naturalize_rag_only_fallback(context_text)
        final_answer = fallback if fallback else (
            "I'm having trouble reaching my knowledge base right now — "
            "please check with the Admin Block."
        )
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