"""
backend/query_correction.py — STT / domain query correction.

Pulled out of main.py so BOTH the deterministic fast-path routing (/ask)
and the new confidence-based RAG pipeline (backend/confidence_rag.py) use
the exact same normalization — previously the raw, uncorrected question
could leak into retrieval on one path but not the other. This module has
no side effects and no dependency on FastAPI/Mongo, so it's safe to import
from anywhere.

Two layers, applied in order (phrase-level first, then word-level):
  1. PHRASE_CORRECTIONS — STT sometimes splits "RNSIT" across multiple
     tokens ("r n s fit", "run sit") instead of mis-hearing it as one
     word; those can't be fixed by a single-word dict lookup, so they're
     corrected as whole phrases BEFORE the text is split into words.
  2. DOMAINS_CORRECTIONS — single mis-heard/mis-typed words.

Both dictionaries are intentionally simple, editable Python dicts (not
config-file driven) so a developer can extend them in one place; admins
extend the KNOWLEDGE side of the system (facts), not STT correction
rules, which is a code-owned concern.
"""
from __future__ import annotations
import re
import string

DOMAINS_CORRECTIONS: dict[str, str] = {
    "pricipal":  "principal",
    "prinsipal": "principal",
    "libary":    "library",
    "placment":  "placement",
    "fees":      "fee",
    # Common STT mis-hearings of "RNSIT" (the college's own name!)
    "rnsfit":    "rnsit",
    "ransit":    "rnsit",
    "rnscit":    "rnsit",
    "arnsit":    "rnsit",
    "rnsit's":   "rnsit",
    "rnsits":    "rnsit",
}

PHRASE_CORRECTIONS: dict[str, str] = {
    "rns fit":     "rnsit",
    "r n s fit":   "rnsit",
    "run sit":     "rnsit",
    "rn sit":      "rnsit",
    "r and s fit": "rnsit",
    "r n site":    "rnsit",
    "rns it":      "rnsit",
    "r n s it":    "rnsit",
    "r n s i t":   "rnsit",
    "ai ml":       "aiml",
    "ai & ml":     "aiml",
    "ai and ml":   "aiml",
    "artificial intelligence and machine learning": "aiml",
    "artificial intelligence & machine learning":   "aiml",
    "artificial intelligence":                      "aiml",
    "management phase":                             "management fee",
    "management fees":                              "management fee",
    "management quota fees":                        "management fee",
    "college timings":                              "college working hours",
    "college timing":                               "college working hours",
    "timings of college":                           "college working hours",
    "timing of college":                            "college working hours",
    "highest package":                              "highest placement package",
    "highest ctc":                                  "highest placement package",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_query(raw_question: str) -> str:
    """
    Full normalization pipeline: lowercase -> strip punctuation ->
    phrase-level STT corrections -> word-level STT/typo corrections.
    This is the single source of truth for what gets sent to retrieval —
    callers should never hand-roll their own version of this.
    """
    q_clean = (raw_question or "").lower().strip()
    q_clean = q_clean.translate(_PUNCT_TABLE).strip()

    for wrong_phrase, right_phrase in PHRASE_CORRECTIONS.items():
        q_clean = re.sub(
            rf"(?:^|\s){re.escape(wrong_phrase)}(?:$|\s)", f" {right_phrase} ", q_clean
        )
    q_clean = q_clean.strip()

    words = q_clean.split()
    corrected_words = [DOMAINS_CORRECTIONS.get(w, w) for w in words]
    return " ".join(corrected_words)