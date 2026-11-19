"""Quarantine FAQ and entity answers that mention retired college facts."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("faq_guard")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETIRED_PATH = DATA_DIR / ".kb_retired.json"
QUARANTINE_PATH = DATA_DIR / ".faq_quarantine.json"

_NON_SOURCE_SECTIONS = {"faqs", "meta"}
_MIN_VALUE_LEN = 4
_MAX_VALUE_LEN = 80


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _has_term(haystack_norm: str, term_norm: str) -> bool:
    return bool(term_norm) and f" {term_norm} " in f" {haystack_norm} "


def _leaves(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _leaves(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _leaves(value)
    elif obj is not None and not isinstance(obj, bool):
        yield str(obj)


def _source_sections(kb: dict | None) -> dict:
    return {key: value for key, value in (kb or {}).items() if key not in _NON_SOURCE_SECTIONS}


def _live_blob(kb: dict | None) -> str:
    return " | ".join(_norm(value) for value in _leaves(_source_sections(kb)))


def _value_terms(kb: dict | None) -> set[str]:
    values = set()
    for raw in _leaves(_source_sections(kb)):
        normalized = _norm(raw)
        digits = normalized.replace(" ", "")
        if not normalized or len(raw) > _MAX_VALUE_LEN:
            continue
        if digits.isdigit():
            if len(digits) >= 3:
                values.add(normalized)
        elif len(normalized) >= _MIN_VALUE_LEN:
            values.add(normalized)
    return values


def _name_variants(name: str) -> set[str]:
    normalized = _norm(name)
    variants = {normalized}
    without_department = re.sub(r"^department of ", "", normalized)
    variants.add(without_department)
    without_engineering = re.sub(r"\s+(?:and\s+)?engineering$", "", without_department).strip()
    if len(without_engineering) >= 4:
        variants.add(without_engineering)
    return {variant for variant in variants if len(variant) >= 3}


def _entry_terms(key: str, node: Any) -> set[str]:
    terms = {_norm(key.replace("_", " "))}
    if isinstance(node, dict):
        for field in ("name", "title"):
            if isinstance(node.get(field), str):
                terms |= _name_variants(node[field])
    elif isinstance(node, str) and len(node) <= _MAX_VALUE_LEN:
        terms.add(_norm(node))
    return {term for term in terms if len(term) >= 3}


def _entries(kb: dict | None) -> dict[tuple[str, str], Any]:
    """Return each source section and its first-level keyed entries."""
    entries: dict[tuple[str, str], Any] = {}
    for section, content in _source_sections(kb).items():
        entries[(section, "")] = content
        if isinstance(content, dict):
            for key, node in content.items():
                entries[(section, key)] = node
    return entries


def _load() -> dict:
    try:
        data = json.loads(RETIRED_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"values": [], "keys": []}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("[FAQ-GUARD] Cannot load retired-term state %s: %s", RETIRED_PATH, exc)
        raise

    if not isinstance(data, dict) or not isinstance(data.get("values", []), list) or not isinstance(data.get("keys", []), list):
        raise ValueError(f"Invalid retired-term state format in {RETIRED_PATH}")
    return data


def _save(data: dict) -> None:
    RETIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RETIRED_PATH.with_suffix(RETIRED_PATH.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(temporary_path, RETIRED_PATH)


def update_retired(old_kb: dict | None, new_kb: dict) -> dict[str, int]:
    """Record retired source terms and release terms that are live again."""
    data = _load()
    values = set(data.get("values", []))
    keys = {
        (item["section"], item["key"]): set(item["terms"])
        for item in data.get("keys", [])
        if isinstance(item, dict)
        and isinstance(item.get("section"), str)
        and isinstance(item.get("key"), str)
        and isinstance(item.get("terms"), list)
    }
    stats = {"values_added": 0, "keys_added": 0}

    if old_kb:
        new_blob = _live_blob(new_kb)
        for value in _value_terms(old_kb):
            if not _has_term(new_blob, value) and value not in values:
                values.add(value)
                stats["values_added"] += 1

        old_entries, new_entries = _entries(old_kb), _entries(new_kb)
        for (section, key), node in old_entries.items():
            if not key or (section, key) in new_entries:
                continue
            surviving_terms = set()
            for (_, new_key), new_node in new_entries.items():
                if new_key:
                    surviving_terms |= _entry_terms(new_key, new_node)
            terms = _entry_terms(key, node) - surviving_terms
            if terms and (section, key) not in keys:
                keys[(section, key)] = terms
                stats["keys_added"] += 1

    live_blob = _live_blob(new_kb)
    values = {value for value in values if not _has_term(live_blob, value)}
    present_entries = _entries(new_kb)
    keys = {entry: terms for entry, terms in keys.items() if entry not in present_entries}

    _save({
        "values": sorted(values),
        "keys": [
            {"section": section, "key": key, "terms": sorted(terms)}
            for (section, key), terms in sorted(keys.items())
        ],
    })
    return stats


def _active_terms(kb: dict | None) -> list[tuple[str, str]]:
    """Return retired terms that are still absent from the current source."""
    if kb is None:
        return []

    data = _load()
    live_blob = _live_blob(kb)
    present_entries = _entries(kb)
    terms = []
    for value in data.get("values", []):
        if isinstance(value, str) and not _has_term(live_blob, value):
            terms.append((value, f"value '{value}' was removed or changed in college_info.json"))
    for entry in data.get("keys", []):
        if not isinstance(entry, dict):
            continue
        section, key = entry.get("section"), entry.get("key")
        if (section, key) not in present_entries:
            for term in entry.get("terms", []):
                if isinstance(term, str):
                    terms.append((term, f"{section}/{key} was deleted from college_info.json"))
    return terms


def stale_reason(text: str, kb: dict | None) -> str | None:
    """Return why text is stale, or None when it contains no retired term."""
    normalized_text = _norm(text)
    for term, reason in _active_terms(kb):
        if _has_term(normalized_text, term):
            return reason
    return None


def _faq_list(kb: Any) -> list:
    faqs = kb.get("faqs") if isinstance(kb, dict) else None
    return faqs if isinstance(faqs, list) else []


def fresh_faqs(kb: dict | None) -> list:
    """Return valid FAQ records whose questions and answers are not stale."""
    faqs = _faq_list(kb)
    active_terms = _active_terms(kb)
    if not active_terms:
        return [faq for faq in faqs if isinstance(faq, dict)]

    fresh = []
    for faq in faqs:
        if not isinstance(faq, dict):
            continue
        text = _norm(f"{faq.get('question', '')} {faq.get('answer', '')}")
        if not any(_has_term(text, term) for term, _ in active_terms):
            fresh.append(faq)
    return fresh


def quarantine_report(kb: dict | None) -> list[dict[str, str]]:
    """Describe each FAQ currently withheld by the retired-term guard."""
    report = []
    active_terms = _active_terms(kb)
    for faq in _faq_list(kb):
        if not isinstance(faq, dict):
            continue
        text = _norm(f"{faq.get('question', '')} {faq.get('answer', '')}")
        for term, reason in active_terms:
            if _has_term(text, term):
                report.append({
                    "question": str(faq.get("question", "")),
                    "reason": reason,
                    "term": term,
                })
                break
    return report


def write_report(kb: dict | None) -> list[dict[str, str]]:
    """Persist and log the FAQ quarantine report for administrators."""
    report = quarantine_report(kb)
    QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = QUARANTINE_PATH.with_suffix(QUARANTINE_PATH.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(temporary_path, QUARANTINE_PATH)
    for item in report:
        logger.warning(
            "[FAQ-GUARD] withheld FAQ %r - %s. Edit this FAQ in college_info.json to release it.",
            item["question"][:70],
            item["reason"],
        )
    return report


def filter_entity_answer(answer: Any, kb: dict | None) -> bool:
    """Return whether a deterministic entity answer avoids retired facts."""
    return stale_reason(str(answer or ""), kb) is None
