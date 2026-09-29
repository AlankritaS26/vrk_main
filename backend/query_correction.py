"""
backend/query_correction.py
---------------------------
Fast, conservative normalization and canonical-entity resolution for STT queries.

Design:
- Canonical entities are the source of truth.
- Normalization happens before entity/routing logic.
- Known entities can be resolved with a lightweight fuzzy check.
- Fuzzy matching is deliberately conservative: low-confidence candidates are
  left unchanged so the normal RAG/clarification flow can handle them.
- No LLM call is used for entity correction.
"""

from __future__ import annotations

import re
import string


# ---------------------------------------------------------------------------
# Deterministic corrections
# ---------------------------------------------------------------------------
# These are general/domain spelling and phrase normalizations, NOT a growing
# list of Whisper variants for individual canonical entities.
DOMAINS_CORRECTIONS = {
    "pricipal": "principal",
    "prinsipal": "principal",
    "libary": "library",
    "placment": "placement",
    "fees": "fee",
}

PHRASE_CORRECTIONS = {
    # RNSIT STT variants are handled by the canonical resolver below.

    # AI/ML formatting
    "ai ml": "aiml",
    "ai and ml": "aiml",
    "ai & ml": "aiml",

    # Fee phrasing
    "management fee": "management fees",
    "management quota fee": "management fees",

    # Timings / package terminology
    "college timing": "college timings",
    "highest package": "highest package",
    "highest ctc": "highest ctc",
}


# ---------------------------------------------------------------------------
# Canonical entities
# ---------------------------------------------------------------------------
# Exact entity mappings elsewhere in the application remain authoritative.
# Only entities in this allowlist may ever be fuzzy-resolved.
CANONICAL_FUZZY_ENTITIES = (
    "rnsit",
    "cse",
    "ece",
    "ise",
    "eee",
    "mech",
    "civil",
    "aiml",
    "aids",
    "mca",
    "mba",
)

ENTITY_FUZZY_MIN_CONF = 0.80
_MIN_CANON_LEN = 5
_MAX_JOIN_TOKENS = 4
_MAX_JOIN_TOKEN_LEN = 3


# ---------------------------------------------------------------------------
# Lightweight similarity helpers
# ---------------------------------------------------------------------------
def _fold(text: str) -> str:
    """Small phonetic normalization used only for entity candidates."""
    return text.lower().replace("m", "n")


def _skeleton(text: str) -> str:
    """Remove vowels after folding, giving a cheap phonetic signature."""
    folded = _fold(text)
    return "".join(ch for ch in folded if ch not in "aeiou")


def _is_subsequence(shorter: str, longer: str) -> bool:
    if not shorter:
        return True
    i = 0
    for ch in longer:
        if i < len(shorter) and shorter[i] == ch:
            i += 1
    return i == len(shorter)


def _edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance, dependency-free."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Standard dynamic programming with adjacent transposition support.
    da = {}
    maxdist = len(a) + len(b)
    d = [[0] * (len(b) + 2) for _ in range(len(a) + 2)]
    d[0][0] = maxdist

    for i in range(len(a) + 1):
        d[i + 1][0] = maxdist
        d[i + 1][1] = i
    for j in range(len(b) + 1):
        d[0][j + 1] = maxdist
        d[1][j + 1] = j

    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db

            cost = 1
            if a[i - 1] == b[j - 1]:
                cost = 0
                db = j

            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
        da[a[i - 1]] = i

    return d[len(a) + 1][len(b) + 1]


def _similarity(candidate: str, canonical: str) -> float:
    """Conservative similarity score for one canonical entity."""
    if not candidate or not canonical:
        return 0.0

    if candidate == canonical:
        return 1.0

    # Canonical short department codes should not be fuzzy-guessed.
    # They continue to work through their existing exact entity mappings.
    if len(canonical) < _MIN_CANON_LEN:
        return 0.0

    if len(candidate) != len(canonical):
        return 0.0

    distance = _edit_distance(candidate, canonical)
    if distance > 1:
        return 0.0

    score = 1.0 - (distance / len(canonical))

    # Additional phonetic guard.  This is intentionally only a guard, never
    # a standalone reason to resolve an entity.
    cand_skeleton = _skeleton(candidate)
    canon_skeleton = _skeleton(canonical)

    if distance == 0:
        return 1.0

    # For a one-character STT substitution such as RMSIT -> RNSIT, require
    # the phonetic signatures to remain structurally close.
    if not _is_subsequence(canon_skeleton, cand_skeleton) and not _is_subsequence(
        cand_skeleton, canon_skeleton
    ):
        return 0.0

    return score


def match_canonical_entity(span: str) -> tuple[str, float] | None:
    """
    Resolve one candidate span against ONLY the canonical entity allowlist.

    Returns (canonical_entity, confidence) only when confidence is high enough.
    Otherwise returns None and leaves the original text untouched.
    """
    candidate = re.sub(r"[^a-z]", "", (span or "").lower())

    if not candidate:
        return None

    best_entity = None
    best_score = 0.0

    for canonical in CANONICAL_FUZZY_ENTITIES:
        canonical_clean = re.sub(r"[^a-z]", "", canonical)

        # Exact canonical spelling always wins.
        if candidate == canonical_clean:
            return canonical, 1.0

        score = _similarity(candidate, canonical_clean)
        if score > best_score:
            best_entity = canonical
            best_score = score

    if best_entity is None or best_score < ENTITY_FUZZY_MIN_CONF:
        return None

    return best_entity, best_score


def _entity_candidate_spans(tokens: list[str]):
    """
    Generate short candidate spans for STT formatting such as:
      RNS IT
      RMS IT
      R N S IT

    We intentionally cap token count/length so normal sentences are never
    treated as entity candidates.
    """
    n = len(tokens)

    for start in range(n):
        joined = ""
        for end in range(start, min(n, start + _MAX_JOIN_TOKENS)):
            token = re.sub(r"[^a-z]", "", tokens[end].lower())

            # A single STT token may already be the full canonical entity
            # (e.g. RMSIT/RNSIT). Multi-token spans remain tightly bounded
            # to short STT fragments.
            if end == start:
                if not token or len(token) > max(_MAX_JOIN_TOKEN_LEN, 5):
                    break
            elif not token or len(token) > _MAX_JOIN_TOKEN_LEN:
                break

            joined += token

            # RNSIT is a five-letter canonical entity.  For fuzzy matching,
            # requiring the same total length prevents phrases such as
            # "runs it" from becoming RNSIT.
            if len(joined) >= _MIN_CANON_LEN:
                yield start, end, joined


def resolve_entities(q_clean: str) -> str:
    """
    Replace only high-confidence canonical-entity candidates.

    The resolver is intentionally conservative:
      - exact canonical entity -> resolve
      - one-character high-confidence deviation of a known >=5-char entity
        -> resolve
      - anything else -> untouched
    """
    if not q_clean:
        return q_clean

    tokens = q_clean.split()
    if not tokens:
        return q_clean

    replacements: list[tuple[int, int, str]] = []

    for start, end, joined in _entity_candidate_spans(tokens):
        # Avoid resolving the same token range twice.
        if any(start >= s and end <= e for s, e, _ in replacements):
            continue

        matched = match_canonical_entity(joined)
        if matched is None:
            continue

        canonical, confidence = matched

        # Explicitly require the high-confidence threshold here as a second
        # safety gate before modifying user/STT text.
        if confidence < ENTITY_FUZZY_MIN_CONF:
            continue

        replacements.append((start, end, canonical))

    if not replacements:
        return q_clean

    # Apply from right to left so token indexes remain valid.
    for start, end, canonical in sorted(replacements, reverse=True):
        tokens[start : end + 1] = [canonical]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public normalization entry point
# ---------------------------------------------------------------------------
def normalize_query(raw_question: str) -> str:
    """
    Normalize an STT/user query while preserving the existing routing contract.

    Order:
      1. lowercase / whitespace normalization
      2. punctuation cleanup
      3. deterministic phrase corrections
      4. deterministic word corrections
      5. conservative canonical-entity resolution
    """
    if not raw_question:
        return ""

    q = str(raw_question).strip().lower()

    # Normalize apostrophes/dashes and remove punctuation while preserving
    # spaces so STT-separated entity tokens can still be joined safely.
    q = q.replace("’", "'").replace("–", "-").replace("—", "-")
    q = q.translate(str.maketrans({ch: " " for ch in string.punctuation if ch != "-"}))
    q = re.sub(r"\s+", " ", q).strip()

    # Phrase corrections are applied longest-first.
    for source, target in sorted(PHRASE_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        q = re.sub(rf"\b{re.escape(source)}\b", target, q)

    # Word-level deterministic corrections.
    words = q.split()
    words = [DOMAINS_CORRECTIONS.get(word, word) for word in words]
    q = " ".join(words)

    # Conservative canonical entity resolution is the final normalization
    # stage, immediately before existing entity/routing logic consumes it.
    q = resolve_entities(q)

    return q
