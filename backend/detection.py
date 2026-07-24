"""
detection.py — EP-07 Presence Detection & Face Recognition Pipeline
VRK/RNS Digital Receptionist
"""

import os
import cv2
import numpy as np
import base64
import logging
import urllib.request
import time
import uuid
import threading
import httpx
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from deepface import DeepFace

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

# ─── Model auto-download ──────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        logger.info("Downloading face_landmarker.task (~5 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model downloaded.")

_ensure_model()

# ─── MediaPipe FaceLandmarker (tracks multiple faces so bystanders can be
#     detected AND ignored, instead of being detected and treated as a reason
#     to stop recognizing the primary/closest visitor) ─────────────────────
_base_opts = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
_lm_opts   = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=4,                        # Detect up to 4 faces to handle bystanders
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_lm_opts)

# ─── DeepFace config ──────────────────────────────────────────────────────────
DEEPFACE_MODEL    = "Facenet512"
COSINE_THRESHOLD  = 0.30

# ─── Tunable constants ────────────────────────────────────────────────────────
_COOLDOWN:                 float = 120.0  # seconds before same person can retrigger
_DWELL_REQUIRED:           float = 0.8    # seconds face must be present before triggering
_GOODBYE_DELAY:            float = 3.0    # seconds face must be ABSENT/SWAPPED before ending session
_SESSION_RECHECK_INTERVAL: float = 1.0    # how often we re-check the active identity
SESSION_CONTINUITY_THRESHOLD: float = 0.40  # loose threshold for frame-to-frame consistency
_POST_SESSION_LOCKOUT:     float = 0.0    # allow the next visitor to be re-identified immediately after session end
_SWAP_MISMATCH_LIMIT:      int   = 3      # backup: consecutive mismatched rechecks before ending a session
_SYNC_INTERVAL:            float = 2.0    # how often we poll the backend to see if it ended our session

# ─── Global state ─────────────────────────────────────────────────────────────
_asking_name:         bool  = False
_recognizing:         bool  = False  # background recognition in-flight
_last_trigger_ts:      float = 0.0
_last_known_identity: str   = ""
_last_known_face_id:  str   = ""
_known_faces_cache:    list  = []

# Passerby filter
_face_first_seen_ts: float = 0.0
_face_present_last:  bool  = False

# Goodbye detection
_face_gone_ts:        float = 0.0
_session_active:      bool  = False
_last_session_end_ts: float = 0.0  # Tracks absolute termination window epoch timestamp

# Throttle + in-flight guard for the mid-session identity recheck
_last_session_recheck_ts: float = 0.0
_rechecking_session:      bool  = False
_session_anchor_embedding: Optional[list] = None  # anchor live embedding captured at session start
_swap_mismatch_count:     int   = 0    # consecutive continuity mismatches seen this session (backup debounce)
_mismatch_since_ts:       float = 0.0  # wall-clock time the FIRST mismatch was seen (primary debounce)

# What the UI/caller should currently show while a session is active.
# Kept separate from _last_known_identity so a mismatched (e.g. a friend's)
# face is displayed correctly instead of being mislabeled with the
# original visitor's name until the session finally ends.
_current_display_identity: str   = ""
_current_display_verified: bool  = False
_current_display_confidence: float = 0.0

# Throttle for the (now background, non-blocking) backend session sync
_last_sync_ts: float = 0.0
_syncing:      bool  = False

# Spatial continuity tracking for the "primary" face. Picking the largest
# face fresh every single frame causes flicker when two people are close in
# size/distance (e.g. a friend standing next to you) — one noisy frame can
# make their face briefly "win", which then poisons the session's embedding
# checks. Instead we track the same physical position frame-to-frame and
# only re-lock onto "largest face" when nobody was being tracked yet.
_primary_track_center: Optional[tuple] = None  # normalized (cx, cy) of tracked primary face

# Lock so mutations of shared session state don't race across threads
_pipeline_lock = threading.Lock()


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class DetectionResult:
    present:        bool
    bbox:           Optional[BoundingBox]
    face_crop:      Optional[np.ndarray]
    identity:       Optional[str]        = None
    verified:       bool                 = False
    confidence:     float                = 0.0
    landmarks_img: Optional[np.ndarray] = None
    error:          Optional[str]        = None
    multiple_faces: bool                 = False  # informational only — bystanders present
    bystander_count: int                 = 0       # how many extra faces besides the primary one


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_frame(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes.")
    return frame

def _get_bbox(landmarks, img_w, img_h, pad=20) -> BoundingBox:
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x1 = max(0,     int(min(xs)) - pad)
    y1 = max(0,     int(min(ys)) - pad)
    x2 = min(img_w, int(max(xs)) + pad)
    y2 = min(img_h, int(max(ys)) + pad)
    return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

def _bbox_center_norm(bbox: BoundingBox, img_w, img_h) -> tuple:
    cx = (bbox.x + bbox.w / 2) / img_w
    cy = (bbox.y + bbox.h / 2) / img_h
    return (cx, cy)

def _draw_landmarks(frame, landmarks, img_w, img_h) -> np.ndarray:
    out = frame.copy()
    for lm in landmarks:
        cv2.circle(out, (int(lm.x * img_w), int(lm.y * img_h)), 1, (0, 255, 0), -1)
    return out

def _cosine_similarity(a, b) -> float:
    a, b  = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ─── Backend HTTP helpers ─────────────────────────────────────────────────────

def _post(path, **kwargs):
    try:
        return httpx.post(f"{BACKEND_URL}{path}", timeout=5, **kwargs)
    except Exception as e:
        logger.warning(f"POST {path} failed: {e}")
        return None

def _get(path):
    try:
        return httpx.get(f"{BACKEND_URL}{path}", timeout=5)
    except Exception as e:
        logger.warning(f"GET {path} failed: {e}")
        return None

def _load_known_faces() -> list:
    try:
        r = httpx.get(f"{BACKEND_URL}/faces/all", timeout=5)
        if r and r.status_code == 200:
            faces = r.json().get("faces", [])
            logger.info(f"[DETECTION] Loaded {len(faces)} known faces from DB")
            return faces
        else:
            logger.warning(f"[DETECTION] /faces/all returned {r.status_code if r else 'no response'}")
    except Exception as e:
        logger.warning(f"Could not load known faces: {e}")
    return []


# ─── Step 1: Presence ─────────────────────────────────────────────────────────

def detect_presence(frame: np.ndarray, draw_mesh: bool = False) -> DetectionResult:
    global _primary_track_center

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = FACE_LANDMARKER.detect(mp_image)

    if not detection.face_landmarks:
        return DetectionResult(present=False, bbox=None, face_crop=None, multiple_faces=False)

    num_detected    = len(detection.face_landmarks)
    multiple_faces  = num_detected > 1
    bystander_count = max(0, num_detected - 1)

    h, w = frame.shape[:2]

    candidates = []
    for landmarks in detection.face_landmarks:
        bbox   = _get_bbox(landmarks, w, h)
        center = _bbox_center_norm(bbox, w, h)
        area   = bbox.w * bbox.h
        candidates.append((bbox, center, area))

    primary_bbox   = None
    primary_center = None

    if _primary_track_center is not None:
        # Someone is already being tracked (a session may be starting or
        # active). Keep following the SAME physical position frame-to-frame
        # instead of re-picking "largest face" every frame — this is what
        # stops a nearby bystander from momentarily stealing the primary
        # slot just because they were fractionally bigger in one frame.
        best_dist = None
        for bbox, center, area in candidates:
            dist = ((center[0] - _primary_track_center[0]) ** 2
                    + (center[1] - _primary_track_center[1]) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist      = dist
                primary_bbox   = bbox
                primary_center = center
    else:
        # Nobody being tracked yet — lock onto whoever is largest/closest,
        # i.e. treat this as a brand-new visitor initiating.
        max_area = -1
        for bbox, center, area in candidates:
            if area > max_area:
                max_area       = area
                primary_bbox   = bbox
                primary_center = center

    _primary_track_center = primary_center

    face_crop = frame[primary_bbox.y : primary_bbox.y + primary_bbox.h, primary_bbox.x : primary_bbox.x + primary_bbox.w].copy()

    # Draw landmarks on all detected faces if requested
    mesh_img = frame.copy() if draw_mesh else None
    if draw_mesh:
        for landmarks in detection.face_landmarks:
            mesh_img = _draw_landmarks(mesh_img, landmarks, w, h)

    return DetectionResult(
        present=True,
        bbox=primary_bbox,
        face_crop=face_crop,
        landmarks_img=mesh_img,
        multiple_faces=multiple_faces,
        bystander_count=bystander_count,
    )


# ─── Step 2: Extract embedding ────────────────────────────────────────────────

def extract_embedding(face_crop: np.ndarray) -> Optional[list]:
    try:
        if face_crop is None or face_crop.size == 0:
            return None
        # mediaPipe isolates crops explicitly, detector_backend="skip" maintains 1 active detector engine
        res = DeepFace.represent(
            img_path          = face_crop,
            model_name        = DEEPFACE_MODEL,
            detector_backend  = "skip",
            enforce_detection = False,
        )
        return res[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


# ─── Step 3: Recognise ────────────────────────────────────────────────────────

def recognize_face(face_crop: np.ndarray, known_faces: list) -> tuple:
    if not known_faces:
        probe_emb = extract_embedding(face_crop)
        return False, None, None, 0.0, probe_emb

    probe_emb = extract_embedding(face_crop)
    if probe_emb is None:
        return False, None, None, 0.0, None

    best_sim     = -1.0
    best_name    = None
    best_face_id = None

    for entry in known_faces:
        stored_enc = entry.get("encoding")
        if not stored_enc:
            continue
        sim = _cosine_similarity(probe_emb, stored_enc)
        if sim > best_sim:
            best_sim     = sim
            best_name    = entry["name"]
            best_face_id = entry["face_id"]

    verified   = best_sim >= (1.0 - COSINE_THRESHOLD)
    confidence = round(max(0.0, best_sim), 4)
    logger.info(f"[DETECTION] Best match: '{best_name}' sim={confidence:.4f} verified={verified}")

    if verified:
        return True, best_name, best_face_id, confidence, probe_emb
    return False, None, None, confidence, probe_emb


# ─── Goodbye & Cleanup ────────────────────────────────────────────────────────

def _reset_local_session_state():
    global _session_active, _last_known_identity, _last_known_face_id, _last_trigger_ts
    global _session_anchor_embedding, _last_session_end_ts, _swap_mismatch_count, _mismatch_since_ts
    global _asking_name, _recognizing, _face_first_seen_ts, _face_present_last, _face_gone_ts
    global _last_session_recheck_ts, _rechecking_session
    global _current_display_identity, _current_display_verified, _current_display_confidence
    global _primary_track_center

    _session_active             = False
    _primary_track_center       = None
    _last_known_identity        = ""
    _last_known_face_id         = ""
    _last_trigger_ts            = 0.0
    _asking_name                = False
    _recognizing                = False
    _face_first_seen_ts         = 0.0
    _face_present_last          = False
    _face_gone_ts               = 0.0
    _session_anchor_embedding   = None
    _last_session_recheck_ts    = 0.0
    _rechecking_session         = False
    _last_session_end_ts        = time.time()
    _swap_mismatch_count        = 0
    _mismatch_since_ts          = 0.0
    _current_display_identity   = ""
    _current_display_verified   = False
    _current_display_confidence = 0.0


def _sync_local_session_state_with_backend():
    """If the backend already ended the session, clear the detector's stale local state.
    Runs in a background thread (throttled) — never blocks the frame loop."""
    global _session_active, _last_known_identity, _last_known_face_id, _last_trigger_ts
    global _session_anchor_embedding, _last_session_recheck_ts, _rechecking_session, _swap_mismatch_count
    global _asking_name, _recognizing, _face_first_seen_ts, _face_present_last, _face_gone_ts, _last_session_end_ts
    global _syncing

    try:
        r = _get("/session/current")
        if not r or r.status_code != 200:
            return False

        data = r.json() or {}
        backend_active = bool(data.get("active"))
        if backend_active:
            return False

        with _pipeline_lock:
            # IMPORTANT: if recognition or the "ask for name" flow is
            # currently in flight locally, the backend simply hasn't caught
            # up yet (it only learns about a session once /session/start or
            # /visitor/unknown actually completes, which can take a couple
            # of seconds for embedding extraction + HTTP round trips). Do
            # NOT treat that in-progress window as a "stale session" — doing
            # so was cancelling recognition mid-run and immediately kicking
            # off a duplicate, causing continuous re-loading and never
            # actually reaching a result.
            if _recognizing or _asking_name or _rechecking_session:
                return False

            if (_session_active or _last_known_identity or _last_known_face_id
                    or _last_session_recheck_ts > 0.0
                    or _session_anchor_embedding is not None):
                logger.info("[DETECTION] Backend session inactive; resetting local face state for next visitor")
                _reset_local_session_state()
                return True
    except Exception as e:
        logger.warning(f"[DETECTION] Could not sync session state with backend: {e}")
    finally:
        _syncing = False
    return False


def _end_session_background():
    """Called to clean up and officially close a session."""
    logger.info("[DETECTION] Ending active session.")
    _post("/session/end")
    with _pipeline_lock:
        _reset_local_session_state()


# ─── Mid-session identity check (Debounced termination + swap identification) ─

def _recheck_session_face_background(face_crop: np.ndarray):
    """
    Runs in a background thread while a session is active.

    - If the current primary face has been deleted from the DB, the session
      ends immediately (deterministic — no ambiguity there).
    - If the current primary face does not match the active session's
      anchor embedding, we've most likely got a different person in frame
      (e.g. a friend). We:
        1. Try to recognize THEM against the known-faces DB so the UI shows
           their real name (or "Unknown" if they're not registered) instead
           of continuing to show the original visitor's name.
        2. Start (or continue) a wall-clock timer. Once the original
           visitor has been missing for >= _GOODBYE_DELAY seconds (~3s),
           the session ends — regardless of whether the new face was
           recognized or not.
      A single matching frame (original visitor back in view) resets the
      timer/counter back to zero.
    """
    global _rechecking_session, _session_active, _last_known_identity, _last_known_face_id
    global _last_trigger_ts, _known_faces_cache, _session_anchor_embedding
    global _swap_mismatch_count, _mismatch_since_ts
    global _current_display_identity, _current_display_verified, _current_display_confidence

    try:
        if not _last_known_face_id and _session_anchor_embedding is None:
            return

        fresh_faces        = _load_known_faces()
        _known_faces_cache = fresh_faces

        # 1. Deterministic deletion check — always instant, no debounce needed
        if _last_known_face_id:
            still_exists = any(f.get("face_id") == _last_known_face_id for f in fresh_faces)
            if not still_exists:
                logger.info(f"[DETECTION] Active face_id={_last_known_face_id} deleted — ending session immediately")
                _end_session_background()
                return

        # 2. Identity continuity check (time-debounced swap detection)
        if _session_anchor_embedding is None:
            return

        probe_emb = extract_embedding(face_crop)
        if probe_emb is None:
            return  # Skip bad frames without penalizing

        sim        = _cosine_similarity(probe_emb, _session_anchor_embedding)
        still_same = sim >= (1.0 - SESSION_CONTINUITY_THRESHOLD)

        if still_same:
            with _pipeline_lock:
                if _swap_mismatch_count or _mismatch_since_ts:
                    logger.info("[DETECTION] Continuity match recovered — resetting mismatch state")
                _swap_mismatch_count        = 0
                _mismatch_since_ts          = 0.0
                _current_display_identity   = _last_known_identity
                _current_display_verified   = True
                _current_display_confidence = round(max(0.0, sim), 4)
            return

        # ── Mismatch: a different face is in frame. Try to recognize THEM. ──
        now = time.time()
        with _pipeline_lock:
            _swap_mismatch_count += 1
            if _mismatch_since_ts == 0.0:
                _mismatch_since_ts = now
            elapsed = now - _mismatch_since_ts

        verified, name, _fid, conf, _ = recognize_face(face_crop, fresh_faces)
        with _pipeline_lock:
            if verified and name:
                _current_display_identity   = name          # friend, recognized by their own face
                _current_display_verified   = True
                _current_display_confidence = conf
                logger.info(f"[DETECTION] Different face in frame, recognized as '{name}' (sim={conf})")
            else:
                _current_display_identity   = "Unknown"      # friend, not registered
                _current_display_verified   = False
                _current_display_confidence = conf

        logger.info(
            f"[DETECTION] Continuity mismatch {_swap_mismatch_count}/{_SWAP_MISMATCH_LIMIT} "
            f"(similarity={sim:.4f}, original visitor absent {elapsed:.1f}s)"
        )

        if elapsed >= _GOODBYE_DELAY or _swap_mismatch_count >= _SWAP_MISMATCH_LIMIT:
            logger.info(
                f"[DETECTION] Original visitor gone ~{_GOODBYE_DELAY:.0f}s — ending session "
                f"(new face displayed as '{_current_display_identity}')"
            )
            _end_session_background()

    except Exception as e:
        logger.error(f"[DETECTION] _recheck_session_face_background error: {e}")
    finally:
        _rechecking_session = False


# ─── Background Flows ─────────────────────────────────────────────────────────

def _handle_recognition_background(face_crop: np.ndarray):
    global _recognizing, _asking_name, _known_faces_cache
    global _last_known_identity, _last_known_face_id, _last_trigger_ts, _session_active
    global _last_session_recheck_ts, _session_anchor_embedding, _swap_mismatch_count, _mismatch_since_ts
    global _current_display_identity, _current_display_verified, _current_display_confidence

    try:
        fresh_faces = _load_known_faces()
        _known_faces_cache = fresh_faces

        verified, name, face_id, confidence, probe_emb = recognize_face(face_crop, fresh_faces)

        if verified and name:
            with _pipeline_lock:
                _last_known_identity        = name
                _last_known_face_id         = face_id
                _last_trigger_ts            = time.time()
                _session_active             = True
                _last_session_recheck_ts    = time.time()
                _session_anchor_embedding   = probe_emb
                _swap_mismatch_count        = 0  # fresh session — reset debounce
                _mismatch_since_ts          = 0.0
                _current_display_identity   = name
                _current_display_verified   = True
                _current_display_confidence = confidence

            _post("/session/start", params={
                "user_name":    name,
                "face_id":      face_id,
                "is_returning": True,
                "visit_count":  1,
                "trigger":      "camera",
            })
            _post("/faces/visit", params={"face_id": face_id})
            logger.info(f"[DETECTION] Returning visitor: {name} (sim={confidence})")

        else:
            with _pipeline_lock:
                _last_known_identity = ""
                _last_known_face_id  = ""
                _last_trigger_ts     = time.time()
                _asking_name         = True
            _handle_unknown_visitor(face_crop)

    except Exception as e:
        logger.error(f"[DETECTION] _handle_recognition_background error: {e}")
    finally:
        _recognizing = False


def _handle_unknown_visitor(face_crop: np.ndarray):
    global _asking_name, _known_faces_cache, _last_known_identity, _last_known_face_id, _last_trigger_ts, _session_active
    global _last_session_recheck_ts, _session_anchor_embedding, _swap_mismatch_count, _mismatch_since_ts
    global _current_display_identity, _current_display_verified, _current_display_confidence
    try:
        r = _post("/visitor/unknown")
        if not r or r.status_code != 200:
            logger.warning("[DETECTION] /visitor/unknown call failed")
            return

        logger.info("[DETECTION] Waiting for visitor name...")

        deadline  = time.time() + 30.0
        name_data = None

        while time.time() < deadline:
            r = _get("/visitor/name_response")
            if r and r.status_code == 200:
                data = r.json()
                if data.get("ready"):
                    name_data = data
                    _post("/visitor/clear_response")
                    break
            time.sleep(0.5)

        visitor_name = (name_data.get("name") or "Guest").strip() if name_data else "Guest"
        save_face    = name_data.get("save", True)                 if name_data else False
        new_face_id  = str(uuid.uuid4())

        logger.info(f"[DETECTION] Name received: '{visitor_name}' save={save_face}")

        embedding = extract_embedding(face_crop)

        if save_face and visitor_name not in ("Guest", ""):
            if embedding:
                resp = _post("/faces/register", json={
                    "face_id":  new_face_id,
                    "name":     visitor_name,
                    "encoding": embedding,
                })
                if resp and resp.status_code == 200:
                    logger.info(f"[DETECTION] Face registered: {visitor_name}")
                    _known_faces_cache   = _load_known_faces()
                    _last_known_identity = visitor_name
                    _last_known_face_id  = new_face_id
                else:
                    logger.warning(f"[DETECTION] /faces/register failed: {resp}")
            else:
                logger.warning("[DETECTION] Embedding failed — face not saved")

        _post("/session/start", params={
            "user_name":    visitor_name,
            "face_id":      new_face_id if (save_face and visitor_name not in ("Guest", "")) else "",
            "is_returning": False,
            "visit_count":  1,
            "trigger":      "camera",
        })
        with _pipeline_lock:
            _session_active              = True
            _last_session_recheck_ts     = time.time()
            _session_anchor_embedding    = embedding
            _swap_mismatch_count         = 0  # fresh session — reset debounce
            _mismatch_since_ts           = 0.0
            _current_display_identity    = visitor_name
            _current_display_verified    = bool(save_face and visitor_name not in ("Guest", ""))
            _current_display_confidence  = 0.0

    except Exception as e:
        logger.error(f"[DETECTION] _handle_unknown_visitor error: {e}")
    finally:
        _last_trigger_ts = time.time() + 60.0
        _asking_name     = False


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(frame_data, known_faces: list = None, draw_mesh: bool = False) -> DetectionResult:
    global _asking_name, _recognizing, _last_trigger_ts, _known_faces_cache, _last_known_identity
    global _face_first_seen_ts, _face_present_last, _face_gone_ts, _session_active
    global _last_session_recheck_ts, _rechecking_session, _session_anchor_embedding, _last_session_end_ts
    global _last_sync_ts, _syncing
    global _current_display_identity, _current_display_verified, _current_display_confidence
    global _primary_track_center

    try:
        frame = _decode_frame(frame_data)
    except Exception as e:
        return DetectionResult(present=False, bbox=None, face_crop=None, error=str(e))

    now = time.time()

    # ─── Throttled, non-blocking backend sync ────────────────────────────────
    # This used to run synchronously on EVERY frame (a blocking HTTP call),
    # which was the main source of camera lag. Now it runs in the background
    # at most once every _SYNC_INTERVAL seconds.
    if not _syncing and (now - _last_sync_ts) >= _SYNC_INTERVAL:
        _last_sync_ts = now
        _syncing      = True
        threading.Thread(target=_sync_local_session_state_with_backend, daemon=True).start()

    # Process core visibility presence tracking normally so state variables match frames perfectly
    result = detect_presence(frame, draw_mesh=draw_mesh)

    # ─── Goodbye detection (nobody in frame at all) ──────────────────────────
    if not result.present:
        if _face_present_last:
            _face_gone_ts = now
            logger.info("[DETECTION] Face disappeared — starting goodbye timer")

        elif _session_active and _face_gone_ts > 0 and (now - _face_gone_ts) >= _GOODBYE_DELAY:
            _face_gone_ts = 0.0
            threading.Thread(target=_end_session_background, daemon=True).start()

        _face_present_last  = False
        _face_first_seen_ts = 0.0
        return result

    # ─── Face IS present ─────────────────────────────────────────────────────
    _face_gone_ts = 0.0

    if not _face_present_last:
        _face_first_seen_ts = now
        logger.info("[DETECTION] Face appeared — starting dwell timer")

    _face_present_last = True

    # ─── Bystander info ──────────────────────────────────────────────────────
    # Bystanders are recorded on the result (result.multiple_faces /
    # result.bystander_count) for logging/UI purposes, but they no longer
    # block recognition of the primary (largest/closest) face below. A
    # friend standing nearby is politely ignored rather than freezing
    # identification of the actual visitor.
    if result.multiple_faces:
        logger.info(f"[DETECTION] {result.bystander_count} bystander face(s) present — ignoring, tracking primary face")

    # ─── Passerby filter ─────────────────────────────────────────────────────
    dwell_time = now - _face_first_seen_ts if _face_first_seen_ts > 0 else 0.0
    if dwell_time < _DWELL_REQUIRED:
        result.identity = "..."
        return result

    # ─── Processing States ────────────────────────────────────────────────────
    if _asking_name or _recognizing:
        result.identity = "Identifying..."
        return result

    # ─── Session active — verification & swap identification ────────────────
    if _session_active:
        if (not _rechecking_session
                and (_last_known_face_id or _session_anchor_embedding is not None)
                and result.face_crop is not None
                and (now - _last_session_recheck_ts) >= _SESSION_RECHECK_INTERVAL):
            _last_session_recheck_ts = now
            _rechecking_session      = True
            threading.Thread(
                target=_recheck_session_face_background,
                args=(result.face_crop.copy(),),
                daemon=True,
            ).start()

        # Show whoever is currently confirmed to be in frame — the original
        # visitor, or (mid-swap, before the 3s grace period elapses) whoever
        # has replaced them, recognized by their own face.
        if _session_active:
            result.identity   = _current_display_identity or _last_known_identity or "..."
            result.verified   = _current_display_verified
            result.confidence = _current_display_confidence
        return result

    # ─── Cooldown ─────────────────────────────────────────────────────────────
    if (now - _last_trigger_ts) < _COOLDOWN:
        if _last_known_identity:
            result.identity = _last_known_identity
            result.verified = True
        return result

    # ─── Kick off background recognition ──────────────────────────────────────
    _recognizing = True
    threading.Thread(
        target=_handle_recognition_background,
        args=(result.face_crop.copy(),),
        daemon=True,
    ).start()

    result.identity = "Identifying..."
    return result


def face_crop_to_b64(face_crop: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", face_crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Local webcam test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else \
          sys.argv[1]       if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {src}")
        sys.exit(1)

    print("Running — press Q to quit.")
    _known_faces_cache = _load_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result  = run_pipeline(frame, draw_mesh=True)
        display = result.landmarks_img if result.landmarks_img is not None else frame.copy()

        if result.present and result.bbox:
            b     = result.bbox
            color = (0, 255, 0) if result.verified else (0, 165, 255)
            cv2.rectangle(display, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
            label = result.identity or "Unknown"
            if result.multiple_faces:
                label += f"  (+{result.bystander_count} bystander)"
            cv2.putText(display, label, (b.x, b.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(display, "NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("VRK Digital Receptionist", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()