"""
kb_sync.py - keeps EVERYTHING that answers visitors in step with the source files.

Sources of truth
    data/college_info.json   structured facts + FAQs
    data/knowledge/*         PDF / DOCX / PPTX / XLSX / CSV / TXT / MD documents

Targets kept in sync (additions, modifications AND deletions)
    1. RAGService vector store      exact sync: new/changed chunks added, removed ones deleted
    2. entity_mapping               verified-answer entities + auto rules, reloaded in place
    3. FAQ answers                  FAQs that mention deleted/changed facts are withheld (faq_guard)
    4. Redis answer cache           cache keys carry a knowledge VERSION; old keys are purged
    5. the kiosk                    /ask reads all of the above, so it is current without a restart

How a pass works (sync_once)
    * hashes the files on disk and compares with data/.kb_state.json
      -> catches changes made while the backend was OFF, not only live edits
    * while changes are pending the cache version is "<hash>-u" (unsynced): answers
      produced mid-sync are cached under a key that is never read again
    * on full success the version becomes "<hash>", stale cache keys are deleted
    * failures are retried with exponential backoff and never mark anything as synced

Run order at startup: startup_sync() finishes (or times out) BEFORE the server accepts
requests; sync_loop() then repeats every KB_SYNC_INTERVAL seconds.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from backend import faq_guard

logger = logging.getLogger("kb_sync")

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
JSON_FILE     = DATA_DIR / "college_info.json"
STATE_FILE    = DATA_DIR / ".kb_state.json"
SNAPSHOT_FILE = DATA_DIR / ".kb_snapshot.json"      # last college_info.json that synced fully

RAG_URL    = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8600").rstrip("/")
COLLECTION = os.getenv("RAG_COLLECTION", "kiosk-rnsit")
INTERVAL   = int(os.getenv("KB_SYNC_INTERVAL", "10"))
RETRIES    = max(1, int(os.getenv("KB_RETRY_ATTEMPTS", "3")))
KB_SOURCE  = "kb_folder"            # source tag on every chunk indexed from data/knowledge
CACHE_PREFIX = "kiosk:cache:"       # Redis answer cache (companion tokens use another prefix)
EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".txt", ".md"}


# ═══════════════════════════ small file helpers ═══════════════════════════
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_state() -> dict:
    try:
        st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        st.setdefault("files", {})
        return st
    except Exception:
        return {"json": "", "json_entities": "", "files": {}}


def _save_state(state: dict) -> None:
    _atomic_write(STATE_FILE, json.dumps(state, indent=2))


def _load_snapshot() -> dict | None:
    try:
        return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _disk_state() -> tuple[str, dict[str, str]]:
    """(json_hash, {filename: hash}) as the files are RIGHT NOW."""
    json_hash = _sha(JSON_FILE) if JSON_FILE.exists() else ""
    files: dict[str, str] = {}
    if KNOWLEDGE_DIR.exists():
        for p in KNOWLEDGE_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(("~", ".")):
                try:
                    files[p.name] = _sha(p)
                except OSError:
                    pass                                    # file vanished mid-scan
    return json_hash, files


def _version_of(json_hash: str, files: dict[str, str]) -> str:
    blob = json_hash + "|" + "|".join(f"{n}:{h}" for n, h in sorted(files.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ═══════════════════════ knowledge version (Redis keys) ═══════════════════════
# main.py builds its cache key as  kiosk:cache:<kb_version()>:<md5(question)>.
# Because the version changes whenever the knowledge changes, an answer computed
# from OLD data can never be read back - not even if a request that started
# before the sync finishes writes it to Redis afterwards.
_version: str = _version_of(*_disk_state()) + "-u"          # "-u" = not yet verified in sync
_last_purged_version: str | None = None
_purge_pending: bool = False


def kb_version() -> str:
    return _version


def _set_version(v: str) -> None:
    global _version
    if v != _version:
        logger.info("[KB-SYNC] knowledge version %s -> %s", _version, v)
    _version = v


def _redis_client():
    """The shared Redis client created in main.py. Looked up in sys.modules (never
    imported) so this can never trigger a second import of the whole backend."""
    import sys
    for module_name in ("backend.main", "main", "__main__"):
        mod = sys.modules.get(module_name)
        client = getattr(mod, "redis_client", None) if mod is not None else None
        if client is not None:
            return client
    return None


def _purge_cache_sync(keep_version: str) -> int:
    rc = _redis_client()
    if rc is None:
        return 0            # Redis not in use (kiosk runs without it) -> nothing cached, nothing to purge
    keep = f"{CACHE_PREFIX}{keep_version}:"
    deleted, batch = 0, []
    for key in rc.scan_iter(match=f"{CACHE_PREFIX}*", count=500):
        if key.startswith(keep):
            continue
        batch.append(key)
        if len(batch) >= 500:
            deleted += rc.unlink(*batch)
            batch = []
    if batch:
        deleted += rc.unlink(*batch)
    return deleted


async def invalidate_answer_caches(keep_version: str | None = None) -> int:
    """Delete every cached answer except those for the current knowledge version."""
    global _purge_pending, _last_purged_version
    keep_version = keep_version or _version
    try:
        n = await asyncio.to_thread(_purge_cache_sync, keep_version)
        _purge_pending = False
        _last_purged_version = keep_version
        logger.info("[KB-SYNC] Redis: purged %d outdated cached answer(s)", n)
        return n
    except Exception as e:
        _purge_pending = True      # safe meanwhile: old keys have a different version prefix
        logger.warning("[KB-SYNC] Redis purge failed (%s) - will retry next pass", e)
        return 0


# ═══════════════════════════ status / observability ═══════════════════════════
_status: dict = {
    "synced": False, "last_attempt": None, "last_success": None, "last_error": None,
    "consecutive_failures": 0, "last_summary": None, "quarantined_faqs": [],
}


def is_synced() -> bool:
    return bool(_status["synced"])


def get_status() -> dict:
    return {**_status, "version": _version, "redis_purge_pending": _purge_pending,
            "interval_seconds": INTERVAL}


# ═══════════════════════════════ HTTP with retries ═══════════════════════════════
async def _rag_call(client: httpx.AsyncClient, method: str, path: str, what: str, **kw) -> httpx.Response:
    """RAGService call with exponential backoff on connection errors / 5xx."""
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = await client.request(method, f"{RAG_URL}{path}", **kw)
            if r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:120]}", request=r.request, response=r)
            return r                                        # 2xx / 4xx are final answers
        except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last = e
            if attempt < RETRIES:
                delay = 2 ** (attempt - 1)
                logger.warning("[KB-SYNC] %s failed (attempt %d/%d): %s - retrying in %ds",
                               what, attempt, RETRIES, e, delay)
                await asyncio.sleep(delay)
    raise last if last else RuntimeError(what)


# ═════════════════════════════════ the sync pass ═════════════════════════════════
_locks: dict[int, asyncio.Lock] = {}


def _lock() -> asyncio.Lock:
    loop_id = id(asyncio.get_running_loop())
    return _locks.setdefault(loop_id, asyncio.Lock())


async def sync_once() -> dict:
    """One full pass. Serialised: loop, startup and manual triggers never overlap."""
    async with _lock():
        return await _sync_locked()


async def _sync_locked() -> dict:
    global _purge_pending
    t0 = time.time()
    _status["last_attempt"] = t0
    summary: dict = {"json_reseeded": False, "entities_reloaded": 0, "indexed": [],
                     "removed": [], "faq_withheld": 0, "cache_purged": 0, "errors": []}
    errors: list[str] = summary["errors"]

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    disk_json, disk_files = _disk_state()
    disk_ver = _version_of(disk_json, disk_files)

    pending = (disk_json != state.get("json") or disk_json != state.get("json_entities")
               or disk_files != state.get("files", {}))
    if pending:
        _set_version(disk_ver + "-u")        # answers made while syncing go to a throw-away key

    # ── 1) college_info.json ─────────────────────────────────────────────
    if disk_json and (disk_json != state.get("json") or disk_json != state.get("json_entities")):
        try:
            new_kb = json.loads(JSON_FILE.read_text(encoding="utf-8"))
            if not isinstance(new_kb, dict) or not new_kb:
                raise ValueError("top level is not a non-empty object")
        except Exception as e:
            # half-saved / invalid file: keep serving the last good data, never wipe it
            errors.append(f"college_info.json is not valid JSON ({e}); keeping previous knowledge")
            logger.error("[KB-SYNC] college_info.json invalid (%s) - previous knowledge kept", e)
            new_kb = None

        if new_kb is not None:
            if disk_json != state.get("json"):
                # what disappeared/changed since the last good sync -> FAQ quarantine
                stats = faq_guard.update_retired(_load_snapshot(), new_kb)
                withheld = faq_guard.write_report(new_kb)
                summary["faq_withheld"] = len(withheld)
                _status["quarantined_faqs"] = withheld
                if stats["values_added"] or stats["keys_added"]:
                    logger.info("[KB-SYNC] retired %d changed value(s) and %d deleted entr(ies)",
                                stats["values_added"], stats["keys_added"])

            if disk_json != state.get("json_entities"):
                try:
                    from backend.entity_mapping import reload_entities
                    summary["entities_reloaded"] = reload_entities()
                    state["json_entities"] = disk_json
                    logger.info("[KB-SYNC] entity mapping reloaded (%d entities)", summary["entities_reloaded"])
                except Exception as e:
                    errors.append(f"entity reload: {e}")
                    logger.error("[KB-SYNC] entity reload failed: %s", e)

            if disk_json != state.get("json"):
                logger.info("[KB-SYNC] college_info.json changed - syncing RAG store")
                try:
                    from backend.llm import initialize_rag_knowledge_base   # lazy: avoid import cycle
                    if await initialize_rag_knowledge_base(force_reseed=True):
                        state["json"] = disk_json
                        _atomic_write(SNAPSHOT_FILE, json.dumps(new_kb))
                        summary["json_reseeded"] = True
                    else:
                        errors.append("college_info.json: RAG sync did not complete (will retry)")
                except Exception as e:
                    errors.append(f"college_info.json RAG sync: {e}")
                    logger.error("[KB-SYNC] RAG sync failed: %s", e)
    elif not disk_json:
        errors.append("college_info.json not found")

    # ── 2) data/knowledge documents (added / modified / deleted) ─────────────
    known: dict = state["files"]
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            for name, h in disk_files.items():
                if known.get(name) == h:
                    continue
                try:
                    path = KNOWLEDGE_DIR / name
                    r = await _rag_call(
                        client, "POST", f"/v1/collections/{COLLECTION}/index/file", f"index {name}",
                        files={"file": (name, path.read_bytes(), "application/octet-stream")},
                        data={"source": KB_SOURCE, "replace": "true"},
                    )
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
                    known[name] = h
                    summary["indexed"].append(name)
                    logger.info("[KB-SYNC] %s: %s", name, r.json().get("message"))
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    logger.error("[KB-SYNC] failed to index %s: %s", name, e)

            # deleted files (known to the state file) ...
            for name in [n for n in list(known) if n not in disk_files]:
                try:
                    r = await _rag_call(client, "DELETE",
                                        f"/v1/collections/{COLLECTION}/files/{quote(name, safe='')}",
                                        f"remove {name}")
                    if r.status_code in (200, 404):
                        known.pop(name, None)
                        summary["removed"].append(name)
                        logger.info("[KB-SYNC] removed deleted document %s from RAG", name)
                    else:
                        errors.append(f"remove {name}: HTTP {r.status_code}")
                except Exception as e:
                    errors.append(f"remove {name}: {e}")

            # ... and orphans the state file does not know (state lost / deleted while off)
            try:
                r = await _rag_call(client, "GET",
                                    f"/v1/collections/{COLLECTION}/source/{KB_SOURCE}/files",
                                    "orphan sweep")
                if r.status_code == 200:
                    for name in r.json().get("files", []):
                        if name not in disk_files:
                            d = await _rag_call(client, "DELETE",
                                                f"/v1/collections/{COLLECTION}/files/{quote(name, safe='')}",
                                                f"remove orphan {name}")
                            if d.status_code in (200, 404):
                                known.pop(name, None)
                                summary["removed"].append(name)
                                logger.info("[KB-SYNC] removed orphaned chunks of %s", name)
            except Exception as e:
                errors.append(f"orphan sweep: {e}")
    except Exception as e:
        errors.append(f"document sync: {e}")

    state["files"] = known
    _save_state(state)

    # ── 3) verdict, cache version and Redis purge ────────────────────────────
    fully_synced = (not errors and state.get("json") == disk_json
                    and state.get("json_entities") == disk_json and known == disk_files)
    if fully_synced:
        _set_version(disk_ver)
        if pending or _purge_pending or _last_purged_version != disk_ver:
            summary["cache_purged"] = await invalidate_answer_caches(disk_ver)
        _status.update(synced=True, last_success=time.time(), last_error=None, consecutive_failures=0)
    else:
        _status.update(synced=False, last_error="; ".join(errors)[:500] or "incomplete",
                       consecutive_failures=_status["consecutive_failures"] + 1)
    _status["last_summary"] = {**summary, "seconds": round(time.time() - t0, 2)}
    return summary


# ═════════════════════════════ startup + background loop ═════════════════════════════
async def startup_sync(timeout: float | None = None) -> bool:
    """
    Run full passes BEFORE the app starts serving, retrying while RAGService boots,
    so anything added/changed/deleted while the backend was OFF is already reflected
    when the first visitor asks. Bounded by `timeout`; returns whether it fully synced.
    The periodic loop keeps trying afterwards if it did not.
    """
    timeout = timeout or float(os.getenv("KB_STARTUP_SYNC_TIMEOUT", "120"))
    deadline = time.time() + timeout
    delay = 2.0
    while True:
        try:
            summary = await asyncio.wait_for(sync_once(), timeout=max(1.0, deadline - time.time()))
            if not summary["errors"]:
                logger.info("[KB-SYNC] startup sync complete: %s", summary)
                return True
            logger.warning("[KB-SYNC] startup sync incomplete: %s", summary["errors"])
        except asyncio.TimeoutError:
            logger.warning("[KB-SYNC] startup sync exceeded %.0fs", timeout)
            break
        except Exception as e:
            logger.warning("[KB-SYNC] startup sync pass failed: %s", e)
        if time.time() + delay >= deadline:
            break
        await asyncio.sleep(delay)
        delay = min(delay * 2, 15.0)
    logger.error("[KB-SYNC] startup sync NOT complete - serving with last known knowledge; "
                 "background loop will keep retrying (status: /api/kb/status)")
    return False


async def sync_loop() -> None:
    while True:
        fails = _status["consecutive_failures"]
        # healthy: every INTERVAL; failing: back off 2x per failure up to 5 min
        await asyncio.sleep(INTERVAL if fails == 0 else min(INTERVAL * (2 ** min(fails, 5)), 300))
        try:
            summary = await sync_once()
            fails = _status["consecutive_failures"]
            if fails and (fails == 3 or fails % 10 == 0):
                logger.error("[KB-SYNC] still failing after %d passes: %s", fails, summary["errors"])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _status["consecutive_failures"] += 1
            _status["last_error"] = str(e)
            logger.exception("[KB-SYNC] pass crashed: %s", e)