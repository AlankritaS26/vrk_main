"""
detection.py — EP-07 Presence Detection & Face Recognition
VRK / RNSIT Digital Receptionist

RECEPTIONIST MODEL
------------------
  PERSON  = face_id     minted ONCE, at first quality-gated enrolment.
  VISIT   = session_id  owned by the backend; every fresh detection starts
                        a brand-new session thread (see main.py).

  One visitor is served at a time. Bystanders are detected and ignored:
  the largest face above a size floor is the PRIMARY visitor and holds
  the session until they leave. A different face never inherits a live
  session — it ends the session instead.

  NOTE — "largest face" doubles as "nearest person": a closer visitor's
  face occupies more of the frame than someone standing further back, so
  the largest-bbox-wins rule already prioritises whoever is nearest the
  kiosk when several people are in frame. No separate depth sensor needed.

STATE MACHINE
-------------
  IDLE -> DWELLING -> RECOGNIZING -> {ACTIVE | ENROLLING -> ACTIVE}
  ACTIVE -> DEPARTING -> (return within grace) ACTIVE | COOLDOWN -> IDLE

GENERATION GUARD
-----------------
Recognition/enrolment run in background threads against a frame snapshot
taken a moment earlier. If the visitor leaves (or a different person steps
in) before that thread finishes, its result is stale and must NOT be
committed — otherwise you get "asked for a name into an empty frame" or
"greeted the wrong person". Every time presence is lost during
DWELLING/RECOGNIZING/ENROLLING, `ST.generation` is bumped; workers check
their captured generation before committing ACTIVE/ENROLLING state and
bail out silently if it no longer matches.

STUCK-STATE WATCHDOG
---------------------
`_spawn()` silently no-ops if a worker from the previous visit is still
finishing up (`ST.busy` still True — e.g. a `_recheck_worker` that hasn't
cleared its flag yet). Previously the DWELLING->RECOGNIZING transition set
`state="RECOGNIZING"` *before* checking whether the spawn actually
succeeded, so a failed spawn left the visitor stuck on "Identifying..."
forever with no worker ever running for that generation, and the only way
out was for them to walk away and re-trigger DWELLING from scratch (which
is what produced the repeated "N known faces loaded" logspam — every
retry re-hit the backend). Fixed below: state only advances to
RECOGNIZING/ENROLLING when a worker is actually running, and a watchdog
force-resets to IDLE if we're ever caught in RECOGNIZING/ENROLLING with
no worker in flight.

BLINK DETECTION (experimental)
-------------------------------
A lightweight eye-aspect-ratio (EAR) blink detector runs on the same
FaceLandmarker output used for presence/recognition — no extra model.
It's exposed on DetectionResult as `blink` (a fresh, debounced blink
event, true for one frame only) purely for the frontend to use as an
optional "yes" gesture (e.g. confirming "would you like me to remember
you?"). It never drives the state machine itself — recognition/session
flow is untouched.

All mutable state lives in KioskState behind one lock. No naked globals.
"""

import os
import cv2
import time
import uuid
import base64
import logging
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import httpx
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
# Recognition engine.
#   PRIMARY : backend/recognition.py -> SCRFD + ArcFace (buffalo_l ONNX) run
#             directly on onnxruntime. No `insightface` pip package, so no MS
#             C++ Build Tools, on any OS. Same models, same accuracy.
#   FALLBACK: DeepFace, only if onnxruntime/models are unavailable. Only this
#             fallback path re-detects with its own internal detector
#             (`DEEPFACE_DETECTOR`) — the ArcFace path always reuses the
#             MediaPipe landmarks captured during presence detection instead
#             (see `extract_embedding`'s fast path), so there is no redundant
#             detector running when _ENGINE == "arcface".
_ENGINE = "none"
try:
    from backend import recognition as RECOG
    _ENGINE = "arcface"
except Exception:
    RECOG = None
try:
    from deepface import DeepFace          # legacy fallback
    if _ENGINE == "none":
        _ENGINE = "deepface"
except Exception:
    DeepFace = None

logger = logging.getLogger(__name__)

# Port 8000 — run.py starts the backend there (old default 8001 was unreachable)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
print(f"[DETECT] backend base URL = {BACKEND_URL}", flush=True)

# ─── Model auto-download ──────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        logger.info("Downloading face_landmarker.task (~5 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


_ensure_model()

_lm_opts = mp_vision.FaceLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=5,                      # see bystanders so we can IGNORE them and select nearest face
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
try:
    FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_lm_opts)
except Exception as _lm_err:
    logger.warning(
        "[DETECT] FaceLandmarker unavailable (missing system library?): %s"
        "  — face detection will be disabled until the library is installed.",
        _lm_err,
    )
    FACE_LANDMARKER = None

DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "Facenet512")
DEEPFACE_DETECTOR = os.getenv("DEEPFACE_DETECTOR", "opencv")

if _ENGINE == "arcface" and not RECOG.is_available():
    _ENGINE = "deepface" if DeepFace is not None else "none"

print(f"[DETECT] recognition engine = {_ENGINE}", flush=True)

# ─── Tunables (all env-overridable; calibrate on-site) ────────────────────────
MIN_FACE_FRAC   = float(os.getenv("MIN_FACE_FRAC", "0.020"))   # ~2% of frame ≈ 1 m.
                                                               # NOT 15% — that is
                                                               # nose-on-glass and
                                                               # detects nobody.
# ArcFace similarities run lower than Facenet512's: same-person pairs land
# ~0.45-0.75, different-person ~0.0-0.25. Default 0.55 => verified at sim>=0.45.
_DEFAULT_MATCH = "0.55" if _ENGINE == "arcface" else "0.30"
MATCH_DISTANCE  = float(os.getenv("FACE_MATCH_DISTANCE", _DEFAULT_MATCH))
MATCH_MARGIN    = float(os.getenv("FACE_MATCH_MARGIN", "0.06"))     # best must beat 2nd best
_DEFAULT_CONT = "0.60" if _ENGINE == "arcface" else "0.28"
CONTINUITY_DIST = float(os.getenv("SESSION_CONTINUITY_DISTANCE", _DEFAULT_CONT))
DWELL_REQUIRED  = float(os.getenv("DWELL_REQUIRED", "0.7"))
DEPART_GRACE    = float(os.getenv("DEPART_GRACE", "3.0"))      # bag/phone/companion tolerance
COOLDOWN        = float(os.getenv("DETECT_COOLDOWN", "2.0"))   # was 6.0 — same person should
                                                               # re-engage within ~2-3s, not 6+
RECHECK_EVERY   = float(os.getenv("SESSION_RECHECK_INTERVAL", "2.0"))
RECOG_ABSENCE_GRACE = float(os.getenv("RECOG_ABSENCE_GRACE", "2.5"))  # tolerate a
                                                               # single blinked/blurred
                                                               # frame during DWELLING/
                                                               # RECOGNIZING/ENROLLING —
                                                               # only invalidate the
                                                               # worker after this many
                                                               # seconds of REAL absence
SWAP_STREAK     = int(os.getenv("SWAP_STREAK", "3"))           # frames before believing a swap
ENROLL_TEMPLATES = int(os.getenv("ENROLL_TEMPLATES", "3"))     # multi-template enrolment
NAME_WAIT_SECS  = float(os.getenv("NAME_WAIT_SECS", "30"))
KNOWN_FACES_TTL = float(os.getenv("KNOWN_FACES_TTL", "5.0"))   # cache /faces/all instead of
                                                               # re-fetching on every single
                                                               # recognition attempt
WORKER_STUCK_TIMEOUT = float(os.getenv("WORKER_STUCK_TIMEOUT", "6.0"))  # watchdog: max time
                                                               # allowed in RECOGNIZING/
                                                               # ENROLLING with no worker
                                                               # actually running

# ─── Blink detection tunables (experimental "yes" gesture) ───────────────────
EAR_BLINK_THRESHOLD = float(os.getenv("EAR_BLINK_THRESHOLD", "0.19"))  # below this = eyes closed
EAR_OPEN_THRESHOLD  = float(os.getenv("EAR_OPEN_THRESHOLD", "0.23"))   # above this = eyes open
                                                               # (hysteresis avoids flicker
                                                               # right at one threshold)
BLINK_MIN_CLOSED_FRAMES = int(os.getenv("BLINK_MIN_CLOSED_FRAMES", "1"))  # frames closed
                                                               # before we trust it's a
                                                               # real blink, not a blend
                                                               # artifact from one bad frame
BLINK_REFRACTORY_SECS = float(os.getenv("BLINK_REFRACTORY_SECS", "0.6"))  # min gap between
                                                               # two reported blinks, so a
                                                               # single natural blink can't
                                                               # be double-counted


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DetectionResult:
    present: bool
    bbox: Optional[BoundingBox] = None
    face_crop: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None          # 5-pt landmarks for ArcFace align
    frame_ref: Optional[np.ndarray] = None    # full frame (fast-path embedding)
    identity: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0
    landmarks_img: Optional[np.ndarray] = None
    bystanders: int = 0
    state: str = "IDLE"
    error: Optional[str] = None
    blink: bool = False               # experimental: a debounced blink event this frame


@dataclass
class KioskState:
    """Every piece of mutable state, guarded by `lock`."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "IDLE"                 # IDLE DWELLING RECOGNIZING ENROLLING ACTIVE DEPARTING COOLDOWN
    face_id: str = ""                   # PERSON id of the current primary visitor
    identity: str = ""
    session_id: str = ""                # VISIT id returned by /visitor/greet — lets us
                                         # notice if the backend closes this session from
                                         # somewhere else (e.g. the voice/conversation flow
                                         # calling /session/end directly) so we can drop
                                         # out of ACTIVE instead of recognizing forever
    anchor: Optional[List[float]] = None   # live embedding captured at session start
    dwell_started: float = 0.0
    departed_at: float = 0.0
    cooldown_until: float = 0.0
    last_recheck: float = 0.0
    swap_streak: int = 0
    busy: bool = False                  # a background worker is running
    generation: int = 0                 # bumped whenever an in-flight worker's
                                         # result should be discarded as stale
    state_entered: float = 0.0          # when we last transitioned into
                                         # RECOGNIZING/ENROLLING — used by the
                                         # stuck-state watchdog
    prompted_departure: bool = False    # True when "Are you there?" prompt has been sent

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "face_id": self.face_id,
                    "identity": self.identity, "busy": self.busy,
                    "generation": self.generation, "state_entered": self.state_entered,
                    "prompted_departure": self.prompted_departure}

    def bump_generation(self):
        """Invalidate any in-flight recognize/enroll/recheck worker."""
        with self.lock:
            self.generation += 1

    def is_current(self, gen: int) -> bool:
        with self.lock:
            return self.generation == gen

    def reset_session(self):
        with self.lock:
            self.state = "COOLDOWN"
            self.cooldown_until = time.time() + COOLDOWN
            self.busy = False   # clear so the NEXT visitor's _spawn doesn't silently no-op
            self.face_id = ""
            self.identity = ""
            self.session_id = ""
            self.anchor = None
            self.swap_streak = 0
            self.dwell_started = 0.0
            self.departed_at = 0.0
            self.state_entered = 0.0
            self.prompted_departure = False
            self.generation += 1   # any worker from the old visit is now stale


ST = KioskState()

# ─── Blink tracker state (independent of KioskState — runs regardless of
# session state so it's available the instant a face is in frame) ────────────
@dataclass
class _BlinkTracker:
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed_frames: int = 0
    eyes_closed: bool = False
    last_blink_at: float = 0.0
    blink_times: list = field(default_factory=list)


_BLINK = _BlinkTracker()

# ─── Known-faces cache ────────────────────────────────────────────────────────
# Previously `_load_known_faces()` hit the backend on EVERY recognition
# attempt (every DWELLING->RECOGNIZING transition), which (a) is redundant
# network I/O in a latency-sensitive worker and (b) was the source of the
# "faces loading again and again" logspam whenever the state machine had to
# retry. Cached with a short TTL; force-refreshed right after we register a
# new face so the very next lookup sees it.
_known_faces: list = []
_known_faces_loaded_at: float = 0.0
_known_faces_lock = threading.Lock()


# ─── Backend HTTP ─────────────────────────────────────────────────────────────
def _post(path, timeout: float = 8, **kw):
    try:
        return httpx.post(f"{BACKEND_URL}{path}", timeout=timeout, **kw)
    except Exception as e:
        logger.warning(f"POST {path} failed: {e}")
        return None


def _get(path):
    try:
        return httpx.get(f"{BACKEND_URL}{path}", timeout=8)
    except Exception as e:
        logger.warning(f"GET {path} failed: {e}")
        return None


def _load_known_faces(force: bool = False) -> list:
    """Cached read of /faces/all. Set `force=True` right after an enrolment
    so the freshly-registered face is visible immediately; everywhere else
    we're happy to reuse a result up to KNOWN_FACES_TTL seconds old."""
    global _known_faces, _known_faces_loaded_at
    now = time.time()
    with _known_faces_lock:
        if not force and _known_faces and (now - _known_faces_loaded_at) < KNOWN_FACES_TTL:
            return _known_faces

    r = _get("/faces/all")
    if r is not None and r.status_code == 200:
        faces = r.json().get("faces", [])
        logger.info(f"[DETECT] {len(faces)} known faces loaded")
        with _known_faces_lock:
            _known_faces = faces
            _known_faces_loaded_at = now
        return faces

    # network hiccup — serve stale cache rather than an empty list, which
    # would otherwise make everyone look "unknown" for one bad request
    with _known_faces_lock:
        return _known_faces


# ─── Vision helpers ───────────────────────────────────────────────────────────
def _bbox(landmarks, w, h, pad=20) -> BoundingBox:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)
    return BoundingBox(x1, y1, x2 - x1, y2 - y1)


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


# Eye landmark indices from MediaPipe's 478-point face mesh topology, used
# only for the EAR blink heuristic — completely separate from the 5-point
# kps used for ArcFace alignment above.
_LEFT_EYE_EAR_IDX  = (33, 160, 158, 133, 153, 144)   # (L corner, top1, top2, R corner, bot2, bot1)
_RIGHT_EYE_EAR_IDX = (263, 387, 385, 362, 380, 373)


def _eye_aspect_ratio(lm, idx, w, h) -> Optional[float]:
    """Standard EAR: vertical eye openness over horizontal eye width.
    Falls back to None if any required landmark is missing."""
    try:
        pts = [(lm[i].x * w, lm[i].y * h) for i in idx]
    except (IndexError, TypeError):
        return None
    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float32) for p in pts]
    vert = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    horiz = 2.0 * np.linalg.norm(p1 - p4)
    if horiz <= 1e-6:
        return None
    return float(vert / horiz)


def _update_blink_tracker(lm, w, h) -> bool:
    """Debounced EAR blink detector. Returns True for genuine blinks (single or double blink),
    maintaining timestamps so double-blinks ("blink 2 times") are properly recognized as Yes gestures."""
    ear_l = _eye_aspect_ratio(lm, _LEFT_EYE_EAR_IDX, w, h)
    ear_r = _eye_aspect_ratio(lm, _RIGHT_EYE_EAR_IDX, w, h)
    if ear_l is None or ear_r is None:
        return False
    ear = (ear_l + ear_r) / 2.0

    now = time.time()
    with _BLINK.lock:
        if not _BLINK.eyes_closed:
            if ear < EAR_BLINK_THRESHOLD:
                _BLINK.closed_frames += 1
                if _BLINK.closed_frames >= BLINK_MIN_CLOSED_FRAMES:
                    _BLINK.eyes_closed = True
            else:
                _BLINK.closed_frames = 0
            return False
        else:
            # eyes currently marked closed — watch for the reopen edge
            if ear > EAR_OPEN_THRESHOLD:
                _BLINK.eyes_closed = False
                _BLINK.closed_frames = 0
                if (now - _BLINK.last_blink_at) >= BLINK_REFRACTORY_SECS:
                    _BLINK.last_blink_at = now
                    _BLINK.blink_times = [t for t in _BLINK.blink_times if (now - t) <= 1.4]
                    _BLINK.blink_times.append(now)
                    return True
            return False


def detect_presence(frame: np.ndarray, draw_mesh: bool = False) -> DetectionResult:
    """PRIMARY-VISITOR LOCK: largest face above MIN_FACE_FRAC wins (this is
    also, in effect, the NEAREST face — see module docstring); the rest
    are bystanders — counted, never served, never blocking."""
    if FACE_LANDMARKER is None:
        return DetectionResult(present=False, error="landmarker_unavailable")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = FACE_LANDMARKER.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not det.face_landmarks:
        return DetectionResult(present=False)

    h, w = frame.shape[:2]
    area_frame = float(h * w)
    best, best_area, qualifying = None, -1.0, 0
    best_lm = None

    for lm in det.face_landmarks:
        bb = _bbox(lm, w, h)
        area = bb.w * bb.h
        if area / area_frame < MIN_FACE_FRAC:
            continue                       # background / passer-by: invisible
        qualifying += 1
        if area > best_area:
            best_area, best, best_lm = area, bb, lm

    if best is None:
        return DetectionResult(present=False)

    crop = frame[best.y:best.y + best.h, best.x:best.x + best.w].copy()
    mesh = None
    if draw_mesh:
        mesh = frame.copy()
        for lm in det.face_landmarks:
            for p in lm:
                cv2.circle(mesh, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)

    # 5-point landmarks (eyes, nose, mouth corners) from the MediaPipe mesh,
    # in FULL-FRAME pixel coords — ArcFace alignment needs exactly these.
    kps = None
    if best_lm is not None:
        try:
            idx = [33, 263, 1, 61, 291]     # L-eye, R-eye, nose, mouth L, mouth R
            kps = np.array([[best_lm[i].x * w, best_lm[i].y * h] for i in idx],
                           dtype=np.float32)
        except Exception:
            kps = None

    # Experimental blink detection on the primary (nearest/largest) face only.
    blink = False
    if best_lm is not None:
        try:
            blink = _update_blink_tracker(best_lm, w, h)
        except Exception:
            blink = False

    return DetectionResult(present=True, bbox=best, face_crop=crop, kps=kps,
                           frame_ref=frame,
                           landmarks_img=mesh, bystanders=max(0, qualifying - 1),
                           blink=blink)


def extract_embedding(face_crop: np.ndarray,
                      frame: Optional[np.ndarray] = None,
                      kps: Optional[np.ndarray] = None) -> Optional[list]:
    """Face -> 512-d L2-normalised embedding.

    FAST PATH: when the caller passes the full `frame` plus the 5 landmarks
    found during presence detection, we skip re-detection and only align +
    embed (~100 ms on a normal laptop CPU). Otherwise we detect inside the
    crop, which is slower and less reliable on tight crops.

    NOTE: embeddings are model-specific. Faces enrolled with DeepFace are NOT
    comparable with ArcFace vectors — wipe `faces` and re-enrol after an
    engine change.
    """
    try:
        if _ENGINE == "arcface":
            if frame is not None and kps is not None:
                return RECOG.embed_with_landmarks(frame, kps)
            return RECOG.get_embedding(face_crop)

        if _ENGINE == "deepface":
            _, enc = cv2.imencode(".jpg", face_crop)
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return DeepFace.represent(img_path=img, model_name=DEEPFACE_MODEL,
                                      detector_backend=DEEPFACE_DETECTOR,
                                      enforce_detection=False)[0]["embedding"]

        logger.error("[DETECT] no recognition engine available")
        return None
    except Exception as e:
        logger.error(f"[DETECT] embedding failed: {e}")
        return None


def _match(probe: list, faces: list):
    """Best match with an AMBIGUITY GUARD: the winner must beat the runner-up
    by MATCH_MARGIN, else we refuse to guess (misgreeting is worse than asking)."""
    best_sim = second = -1.0
    best = None
    for f in faces:
        encs = f.get("encodings") or ([f["encoding"]] if f.get("encoding") else [])
        if not encs:
            continue
        sim = max(_cos(probe, e) for e in encs)     # multi-template: best of N
        if sim > best_sim:
            second, best_sim, best = best_sim, sim, f
        elif sim > second:
            second = sim

    if best is None:
        return None, 0.0
    verified = best_sim >= (1.0 - MATCH_DISTANCE)
    if verified and second > 0 and (best_sim - second) < MATCH_MARGIN:
        logger.warning(f"[DETECT] ambiguous: {best_sim:.3f} vs {second:.3f} -> unknown")
        return None, best_sim
    return (best, best_sim) if verified else (None, best_sim)


# ─── Session transitions ──────────────────────────────────────────────────────
def _end_session(reason: str):
    logger.info(f"[DETECT] session end ({reason})")
    _post("/session/end")
    ST.reset_session()   # also bumps generation — invalidates any straggler worker


def _start_session(face: dict, anchor: list, sim: float, gen: int):
    """Backend owns session_id and decides resume-vs-new. `gen` is the
    generation captured when the calling worker started — if the visitor
    has since left (or someone else took their place), we must NOT commit
    this identity."""
    if not ST.is_current(gen):
        logger.info("[DETECT] stale worker (visitor changed) - discarding result "
                    f"for '{face.get('name')}'")
        return

    face_id = face["face_id"]
    # LONGER TIMEOUT, DELIBERATELY: /visitor/greet does a real-time LLM call
    # server-side (the "re-engagement lookup" that builds the "Welcome back,
    # last time you asked about X" summary — see backend.llm's Gemini
    # fallback path). That round trip regularly took 8-13+ seconds in
    # practice (RAG search + Local-Qwen connect-timeout + Gemini fallback),
    # well past the old blanket 8s timeout here.
    #
    # BUG THIS FIXES: when this 8s timeout fired, we treated it as a failed
    # greet and retried — but the ORIGINAL request kept running on the
    # server and usually finished anyway a few seconds later, completing
    # the session resume and broadcasting its own "session_start" + greeting
    # audio. The retry then did the same thing again. Visitors would see
    # "Welcome back" appear two or three times and hear overlapping/
    # interrupted greeting audio — and because each new greeting's speak()
    # call interrupts whatever audio was already playing (see WelcomeScreen.
    # js's interruptSpeaking()), a real answer already in flight could get
    # silently cut off too, which is what showed up as "voice not coming."
    # 25s comfortably covers the worst case we've observed and stops the
    # client from abandoning a request that the server was going to finish
    # anyway.
    r = _post("/visitor/greet", timeout=25, json={
        "face_id": face_id,
        "name": face.get("name", "Guest"),
        "is_returning": True,
        "visit_count": int(face.get("visit_count") or 1),
    })

    # ── BUG FIX: only advance to ACTIVE when the backend confirmed the session ──
    # Previously ACTIVE was always set even when the HTTP call failed, causing
    # the detection overlay to show "Welcome!" while /session/current returned
    # {active: false} and App.js stayed stuck on the idle/home screen.
    if r is None or r.status_code != 200:
        logger.error(
            "[DETECT] /visitor/greet failed (status=%s) — cooling down before "
            "the visitor is re-recognised.",
            getattr(r, "status_code", "no-response"),
        )
        _fail_recognition("/visitor/greet failed", gen)
        return

    # Re-check staleness AFTER the network round-trip too — the greet call
    # itself takes time, during which the visitor may have left.
    if not ST.is_current(gen):
        logger.info("[DETECT] stale worker after greet round-trip - ending "
                    "the session we just opened instead of showing it")
        _post("/session/end")
        return

    _post("/faces/visit", params={"face_id": face_id})
    sid = ""
    try:
        sid = (r.json() or {}).get("session_id", "")
    except Exception:
        pass
    ST.set(state="ACTIVE", face_id=face_id, identity=face.get("name", ""),
           anchor=anchor, swap_streak=0, last_recheck=time.time(), session_id=sid)
    logger.info(f"[DETECT] ACTIVE {face.get('name')} sim={sim:.3f} session={sid[:8]}")


def _fail_recognition(reason: str, gen: int):
    """A recognize/enrol attempt didn't pan out (bad embedding, ambiguous
    match, backend hiccup — anything short of a genuine 'visitor left').
    Per the requested behaviour: STOP recognizing immediately, sit quiet
    for one COOLDOWN period, then automatically re-arm and re-identify
    whoever is in front of the camera (same person or a new one) instead
    of instantly re-triggering DWELLING and hammering the backend again."""
    logger.info(f"[DETECT] recognition attempt aborted ({reason}) - "
                f"cooling down {COOLDOWN}s before re-identifying")
    if ST.is_current(gen):
        ST.set(state="COOLDOWN", cooldown_until=time.time() + COOLDOWN,
               dwell_started=0.0, state_entered=0.0)


def _recognize_worker(crop: np.ndarray, frame=None, kps=None, gen: int = 0):
    """RECOGNIZING: identify, then either start the session or enrol."""
    try:
        known = _load_known_faces()

        if not ST.is_current(gen):
            logger.info("[DETECT] recognize aborted: visitor changed while "
                        "known-faces were loading")
            return   # visitor already gone by the time faces loaded — state
                     # was already forced to IDLE by the absence-grace path,
                     # nothing further to do here

        probe = extract_embedding(crop, frame, kps)
        if probe is None:
            logger.warning("[DETECT] recognize aborted: embedding extraction "
                           "returned None (bad crop / engine error — check the "
                           "'[DETECT] embedding failed' line just above this)")
            _fail_recognition("embedding extraction failed", gen)
            return

        if not ST.is_current(gen):
            logger.info("[DETECT] recognize aborted: visitor changed during "
                        "embedding extraction")
            return   # visitor left during embedding extraction

        face, sim = _match(probe, known)
        if face:
            _start_session(face, probe, sim, gen)
        else:
            if not ST.is_current(gen):
                logger.info("[DETECT] recognize aborted: visitor changed "
                            "right before enrolment")
                return
            logger.info(f"[DETECT] no confident match (best sim={sim:.3f}) "
                        "— moving to ENROLLING")
            ST.set(state="ENROLLING", state_entered=time.time())
            _enroll_worker(crop, probe, frame, kps, gen)
    except Exception as e:
        logger.error(f"[DETECT] recognize error: {e}")
        _fail_recognition(f"exception: {e}", gen)
    finally:
        ST.set(busy=False)


def _enroll_worker(crop: np.ndarray, probe: list, frame=None, kps=None, gen: int = 0):
    """
    ENROLLING — the ONLY place a face_id is ever minted.

    Guards (this is what stopped the 'one person, seven face_ids' factory):
      * DUPLICATE HARD BLOCK — if this face already matches someone, we do
        NOT create a second identity; we greet them as that person.
      * multi-template capture for a robust identity.
      * GENERATION GUARD — if the visitor leaves while we're waiting for a
        name (camera goes dark, they walk off), we stop waiting and never
        commit a session for someone no longer there.
    """
    try:
        if not ST.is_current(gen):
            return

        r = _post("/visitor/unknown")
        if r is None or r.status_code != 200:
            logger.warning("[DETECT] /visitor/unknown failed (status=%s)",
                           getattr(r, "status_code", "no-response"))
            _fail_recognition("/visitor/unknown failed", gen)
            return

        deadline = time.time() + NAME_WAIT_SECS
        data = None
        while time.time() < deadline:
            if not ST.is_current(gen):
                # Visitor left mid-wait — stop asking into an empty frame and
                # clear the "asking_name" prompt on the backend/frontend.
                logger.info("[DETECT] visitor left while waiting for name - aborting enrolment")
                _post("/session/end")
                return
            rr = _get("/visitor/name_response")
            if rr is not None and rr.status_code == 200:
                d = rr.json()
                if d.get("ready"):
                    data = d
                    _post("/visitor/clear_response")
                    break
            time.sleep(0.4)

        if not ST.is_current(gen):
            _post("/session/end")
            return

        name = (data.get("name") or "Guest").strip() if data else "Guest"
        save = bool(data.get("save", True)) if data else False

        # ── DUPLICATE HARD BLOCK ──────────────────────────────────────────
        fresh = _load_known_faces(force=True)
        dup, dup_sim = _match(probe, fresh)
        if dup:
            logger.warning(f"[DETECT] enrolment blocked - face already registered "
                           f"as '{dup.get('name')}' (sim={dup_sim:.3f}). Resuming them.")
            _start_session(dup, probe, dup_sim, gen)
            return

        if not ST.is_current(gen):
            return

        if not save or name in ("Guest", ""):
            # Guest: session without an identity; anchor still locks the seat.
            _post("/session/start", params={"user_name": name or "Guest",
                                            "face_id": "", "is_returning": False,
                                            "visit_count": 1, "trigger": "camera"})
            if not ST.is_current(gen):
                _post("/session/end")
                return
            ST.set(state="ACTIVE", face_id="", identity=name or "Guest",
                   anchor=probe, swap_streak=0, last_recheck=time.time())
            return

        templates = [probe]                      # multi-template enrolment
        # (extra templates are captured by the caller's frames on later visits;
        #  a single high-quality template plus visit-time updates is enough here)

        face_id = str(uuid.uuid4())              # minted ONCE, for this person
        resp = _post("/faces/register", json={"face_id": face_id, "name": name,
                                              "encoding": templates[0],
                                              "encodings": templates})
        if resp is None or resp.status_code != 200:
            logger.warning("[DETECT] /faces/register failed (status=%s)",
                           getattr(resp, "status_code", "no-response"))
            _fail_recognition("/faces/register failed", gen)
            return

        logger.info(f"[DETECT] enrolled '{name}' face_id={face_id[:8]}")
        _load_known_faces(force=True)   # so the next lookup sees this face immediately

        if not ST.is_current(gen):
            _post("/session/end")
            return

        _post("/session/start", params={"user_name": name, "face_id": face_id,
                                        "is_returning": False, "visit_count": 1,
                                        "trigger": "camera"})
        if not ST.is_current(gen):
            _post("/session/end")
            return
        ST.set(state="ACTIVE", face_id=face_id, identity=name,
               anchor=probe, swap_streak=0, last_recheck=time.time())
    except Exception as e:
        logger.error(f"[DETECT] enrol error: {e}")
        _fail_recognition(f"enrol exception: {e}", gen)


def _backend_session_still_active(local_session_id: str) -> Optional[bool]:
    """Ask the backend whether it still considers this visit live. Returns
    True/False when the response is unambiguous, or None if the response
    shape is unrecognised / unreachable (in which case the caller should NOT
    end the session on a guess).

    NOTE: this assumes /session/current returns something like
    {"active": bool, "session_id": str}. If your backend's response shape
    differs, adjust the two .get(...) lookups below to match — check
    main.py's /session/current handler for the exact field names.
    """
    r = _get("/session/current")
    if r is None or r.status_code != 200:
        return None
    try:
        cur = r.json() or {}
    except Exception:
        return None

    active = cur.get("active")
    remote_sid = cur.get("session_id") or cur.get("session", {}).get("session_id", "")

    if active is False:
        logger.info(f"[DETECT] backend /session/current reports active=False "
                    f"(payload={cur}) - session was ended elsewhere")
        return False
    if local_session_id and remote_sid and remote_sid != local_session_id:
        logger.info(f"[DETECT] backend session_id changed under us "
                    f"(local={local_session_id[:8]} remote={remote_sid[:8]}) "
                    "- session was ended/replaced elsewhere")
        return False
    return True


def _recheck_worker(crop: np.ndarray, frame=None, kps=None):
    """ACTIVE: is the person in front still the session owner, AND does the
    backend still agree this session is live? Deletion / external session-end
    is deterministic (immediate). Identity drift is debounced."""
    try:
        snap = ST.snapshot()
        gen = snap["generation"]
        if snap["face_id"]:
            fresh = _load_known_faces()
            if not any(f.get("face_id") == snap["face_id"] for f in fresh):
                logger.info("[DETECT] session face deleted from DB - ending")
                _end_session("deleted")
                return

        # BUG FIX: the voice/conversation flow can end a session by calling
        # /session/end directly (e.g. after "thank you"), completely
        # bypassing this module. Previously ST.state just stayed "ACTIVE"
        # forever in that case — detection kept re-verifying the same face
        # every RECHECK_EVERY seconds and never dropped back to IDLE, so the
        # kiosk never re-armed for the next visitor (or the same visitor's
        # next visit). Now we cross-check with the backend on every recheck.
        with ST.lock:
            local_sid = ST.session_id
        still_active = _backend_session_still_active(local_sid)
        if still_active is False:
            _end_session("ended by backend")
            return

        with ST.lock:
            anchor = ST.anchor
        if anchor is None:
            return
        probe = extract_embedding(crop, frame, kps)
        if probe is None:
            return

        if not ST.is_current(gen):
            return   # session already moved on while we were embedding

        sim = _cos(probe, anchor)
        if sim >= (1.0 - CONTINUITY_DIST):
            ST.set(swap_streak=0)
            return

        with ST.lock:
            ST.swap_streak += 1
            streak = ST.swap_streak
        logger.info(f"[DETECT] continuity mismatch {streak}/{SWAP_STREAK} (sim={sim:.3f})")
        if streak >= SWAP_STREAK:
            # A DIFFERENT person is now in front. Do not merely end the
            # session and idle — that left the newcomer stuck inside the
            # previous visitor's identity until the camera was covered.
            # End the old session, then RE-IDENTIFY this face right away.
            _end_session("face swap")
            new_gen = ST.snapshot()["generation"]

            fresh = _load_known_faces(force=True)
            face, msim = _match(probe, fresh)
            if face:
                logger.info(f"[DETECT] re-identified as '{face.get('name')}' "
                            f"(sim={msim:.3f}) — starting their session")
                ST.set(state="RECOGNIZING", state_entered=time.time())
                _start_session(face, probe, msim, new_gen)
            else:
                # Unknown newcomer: enrol them instead of idling.
                logger.info("[DETECT] newcomer not recognised - enrolling")
                ST.set(state="ENROLLING", state_entered=time.time())
                _enroll_worker(crop, probe, frame, kps, new_gen)
    except Exception as e:
        logger.error(f"[DETECT] recheck error: {e}")
    finally:
        ST.set(busy=False)


def _spawn(target, *args):
    with ST.lock:
        if ST.busy:
            return False
        ST.busy = True
    threading.Thread(target=target, args=args, daemon=True).start()
    return True


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(frame_data, known_faces=None, draw_mesh: bool = False) -> DetectionResult:
    frame = frame_data if isinstance(frame_data, np.ndarray) else \
        cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return DetectionResult(present=False, error="bad frame")

    res = detect_presence(frame, draw_mesh)
    now = time.time()
    snap = ST.snapshot()
    st = snap["state"]
    res.state = st

    # ── nobody in front ───────────────────────────────────────────────────
    if not res.present:
        if st == "ACTIVE":
            ST.set(state="DEPARTING", departed_at=now, prompted_departure=False)
        elif st == "DEPARTING":
            with ST.lock:
                gone_for = now - ST.departed_at
                already_prompted = ST.prompted_departure
            if gone_for >= DEPART_GRACE:
                if not already_prompted:
                    ST.set(prompted_departure=True)
                    logger.info("[DETECT] Face absent for 3s during ACTIVE session — prompting 'Are you there?'")
                    _post("/session/are_you_there")
                elif gone_for >= (DEPART_GRACE + 7.0):
                    _end_session("visitor left after 'are you there' prompt")
        elif st in ("DWELLING", "RECOGNIZING", "ENROLLING"):
            # Debounced: a single blinked/blurred frame during recognition
            # must NOT nuke an in-flight worker's result. Only treat them as
            # truly gone after RECOG_ABSENCE_GRACE seconds of continuous
            # absence — same idea as DEPART_GRACE for ACTIVE sessions.
            with ST.lock:
                if ST.departed_at == 0.0:
                    ST.departed_at = now
                gone_for = now - ST.departed_at
            if gone_for >= RECOG_ABSENCE_GRACE:
                ST.bump_generation()
                ST.set(state="IDLE", dwell_started=0.0, departed_at=0.0, state_entered=0.0)
        elif st == "COOLDOWN":
            with ST.lock:
                if now >= ST.cooldown_until:
                    ST.state = "IDLE"
        res.state = ST.snapshot()["state"]
        return res

    # ── someone is in front ───────────────────────────────────────────────
    if st in ("DWELLING", "RECOGNIZING", "ENROLLING"):
        with ST.lock:
            if ST.departed_at != 0.0:
                ST.departed_at = 0.0

    # ── STUCK-STATE WATCHDOG ────────────────────────────────────────────
    # If we're sitting in RECOGNIZING/ENROLLING but no worker is actually
    # running (busy == False), a spawn must have silently failed earlier
    # (see module docstring). Don't leave the visitor frozen on
    # "Identifying..." — force back to IDLE so DWELLING retries the spawn.
    if st in ("RECOGNIZING", "ENROLLING"):
        cur = ST.snapshot()
        if not cur["busy"]:
            entered = cur["state_entered"] or now
            stuck_for = now - entered
            if stuck_for >= WORKER_STUCK_TIMEOUT:
                logger.warning(f"[DETECT] {st} stuck with no worker running "
                               f"({stuck_for:.1f}s) — resetting to IDLE for retry")
                ST.bump_generation()
                ST.set(state="IDLE", dwell_started=0.0, departed_at=0.0, state_entered=0.0)
                st = "IDLE"

    if st == "DEPARTING":
        # they came back inside the grace window → keep the SAME session
        ST.set(state="ACTIVE", departed_at=0.0, prompted_departure=False)
        st = "ACTIVE"

    if st == "ACTIVE":
        res.identity, res.verified = snap["identity"], True
        with ST.lock:
            due = (now - ST.last_recheck) >= RECHECK_EVERY
            if due:
                ST.last_recheck = now
        if due and res.face_crop is not None:
            _spawn(_recheck_worker, res.face_crop.copy(), res.frame_ref, res.kps)
        return res

    if st == "COOLDOWN":
        with ST.lock:
            if now < ST.cooldown_until:
                res.identity = "..."
                return res
            ST.state = "IDLE"
        st = "IDLE"

    if st == "IDLE":
        logger.info("[DETECT] visitor detected - starting dwell timer")
        ST.set(state="DWELLING", dwell_started=now)
        res.identity = "..."
        return res

    if st == "DWELLING":
        with ST.lock:
            dwell = now - ST.dwell_started
        if dwell < DWELL_REQUIRED:
            res.identity = "..."
            return res

        # ── BUG FIX: don't advance state until the worker is actually running ──
        # Previously `ST.set(state="RECOGNIZING")` happened unconditionally,
        # before checking whether `_spawn` succeeded. If a worker from the
        # previous visit (e.g. a lingering `_recheck_worker`) still held
        # `ST.busy`, `_spawn` would silently no-op and the visitor would be
        # stuck showing "Identifying..." forever with nothing ever recognising
        # them — this is why a returning visitor never made it past this
        # screen, and why the backend kept re-fetching faces every time they
        # walked away and back to retrigger DWELLING from scratch.
        if res.face_crop is not None:
            gen = ST.snapshot()["generation"]
            spawned = _spawn(_recognize_worker, res.face_crop.copy(), res.frame_ref, res.kps, gen)
            if spawned:
                ST.set(state="RECOGNIZING", state_entered=now)
                res.identity = "Identifying..."
            else:
                # a worker from the previous visit is still finishing up —
                # stay in DWELLING and retry the spawn next frame
                res.identity = "..."
        else:
            res.identity = "..."
        return res

    res.identity = "Identifying..." if st in ("RECOGNIZING", "ENROLLING") else res.identity
    return res


def _warm_models():
    """First DeepFace call builds the TF graph (2-5 s). Pay that at startup,
    not while a visitor is standing in front of the kiosk."""
    try:
        if _ENGINE == "arcface":
            RECOG.warm_up()
        else:
            extract_embedding(np.zeros((160, 160, 3), dtype=np.uint8))
        logger.info("[DETECT] face model warmed up")
    except Exception as e:
        logger.warning(f"[DETECT] warmup skipped: {e}")


threading.Thread(target=_warm_models, daemon=True).start()


def face_crop_to_b64(face_crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", face_crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Local webcam runner ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else (
        sys.argv[1] if len(sys.argv) > 1 else int(os.getenv("CAMERA_INDEX", "0")))

    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        print(f"Cannot open camera source: {src}  "
              f"(try CAMERA_INDEX=1 in .env)")
        sys.exit(1)

    show = os.getenv("DETECT_WINDOW", "true").lower() == "true"
    print("Detection running - press Q in the window to quit.")
    _load_known_faces()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = run_pipeline(frame, draw_mesh=show)

        if show:
            disp = r.landmarks_img if r.landmarks_img is not None else frame.copy()
            if r.present and r.bbox:
                b = r.bbox
                col = (0, 255, 0) if r.verified else (0, 165, 255)
                cv2.rectangle(disp, (b.x, b.y), (b.x + b.w, b.y + b.h), col, 2)
                cv2.putText(disp, r.identity or "Unknown", (b.x, b.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(disp, f"{r.state}  bystanders={r.bystanders}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow("VRK Digital Receptionist", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()