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

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

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

# ─── MediaPipe FaceLandmarker (num_faces=1 — ignores anyone standing beside) ─
_base_opts = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
_lm_opts   = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,                        # ONLY primary face — bystanders ignored
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_lm_opts)

# ─── DeepFace config ──────────────────────────────────────────────────────────
DEEPFACE_MODEL    = "Facenet512"
DEEPFACE_DETECTOR = "opencv"
COSINE_THRESHOLD  = 0.30

# ─── Tunable constants ────────────────────────────────────────────────────────
_COOLDOWN:       float = 120.0  # seconds before same person can retrigger
_DWELL_REQUIRED: float = 0.8    # seconds face must be present before triggering
_GOODBYE_DELAY:  float = 3.0    # seconds face must be ABSENT before ending session
_SESSION_RECHECK_INTERVAL: float = 5.0  # how often to verify active session's face_id still exists

# ─── Global state ─────────────────────────────────────────────────────────────
_asking_name:         bool  = False
_recognizing:         bool  = False   # background recognition in-flight
_last_trigger_ts:     float = 0.0
_last_known_identity: str   = ""
_last_known_face_id:  str   = ""      # needed to check deletion mid-session
_known_faces_cache:   list  = []
_greeted:             bool  = False   # welcome greeting played this session

# Passerby filter
_face_first_seen_ts: float = 0.0
_face_present_last:  bool  = False

# Goodbye detection
_face_gone_ts:   float = 0.0
_session_active: bool  = False

# Throttle + in-flight guard for the mid-session deletion check
_last_session_recheck_ts: float = 0.0
_rechecking_session:      bool  = False

# Lock so only one background recognition/unknown-handler runs at a time
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
    present:       bool
    bbox:          Optional[BoundingBox]
    face_crop:     Optional[np.ndarray]
    identity:      Optional[str]                 = None
    verified:      bool                          = False
    confidence:    float                         = 0.0
    landmarks_img: Optional[np.ndarray] = None
    error:         Optional[str]                 = None


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
    """Reload face cache from DB — called before each recognition attempt."""
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
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = FACE_LANDMARKER.detect(mp_image)

    if not detection.face_landmarks:
        return DetectionResult(present=False, bbox=None, face_crop=None)

    h, w      = frame.shape[:2]
    landmarks = detection.face_landmarks[0]
    bbox      = _get_bbox(landmarks, w, h)
    face_crop = frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w].copy()
    mesh_img  = _draw_landmarks(frame, landmarks, w, h) if draw_mesh else None

    return DetectionResult(present=True, bbox=bbox, face_crop=face_crop, landmarks_img=mesh_img)


# ─── Step 2: Extract embedding ────────────────────────────────────────────────

def extract_embedding(face_crop: np.ndarray) -> Optional[list]:
    try:
        _, enc  = cv2.imencode(".jpg", face_crop)
        img_arr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        res = DeepFace.represent(
            img_path          = img_arr,
            model_name        = DEEPFACE_MODEL,
            detector_backend  = DEEPFACE_DETECTOR,
            enforce_detection = False,
        )
        return res[0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


# ─── Step 3: Recognise ────────────────────────────────────────────────────────

def recognize_face(face_crop: np.ndarray, known_faces: list) -> tuple:
    if not known_faces:
        return False, None, None, 0.0

    probe_emb = extract_embedding(face_crop)
    if probe_emb is None:
        return False, None, None, 0.0

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
        return True, best_name, best_face_id, confidence
    return False, None, None, confidence


# ─── Goodbye: end session when face disappears ────────────────────────────────

def _end_session_background():
    """Called in background thread when face has been gone for _GOODBYE_DELAY seconds."""
    global _session_active, _last_known_identity, _last_known_face_id, _last_trigger_ts, _greeted
    logger.info("[DETECTION] Face gone — ending session")
    _post("/session/end")
    _session_active      = False
    _last_known_identity = ""
    _last_known_face_id  = ""
    _last_trigger_ts     = 0.0
    _greeted             = False   # reset so next visitor gets greeting


# ─── Mid-session deletion check ────────────────────────────────────────────
def _recheck_session_face_background():
    """
    Runs in a background thread while a session is active. Re-fetches /faces/all
    and, if the current visitor's face_id is no longer present (i.e. they hit
    Delete My Data while still standing in frame), force-ends the session
    immediately instead of waiting for them to walk away + restart detection.py.
    """
    global _rechecking_session, _session_active, _last_known_identity, _last_known_face_id, _last_trigger_ts, _greeted

    try:
        if not _last_known_face_id:
            # Guest / unsaved visitor — nothing to verify against
            return

        fresh_faces     = _load_known_faces()
        still_exists    = any(f.get("face_id") == _last_known_face_id for f in fresh_faces)

        if not still_exists:
            logger.info(
                f"[DETECTION] Active session's face_id={_last_known_face_id} "
                f"no longer in DB (deleted) — ending session early"
            )
            _post("/session/end")
            _session_active      = False
            _last_known_identity = ""
            _last_known_face_id  = ""
            _last_trigger_ts     = 0.0
            _greeted             = False
    except Exception as e:
        logger.error(f"[DETECTION] _recheck_session_face_background error: {e}")
    finally:
        _rechecking_session = False


# ─── Background: recognition (non-blocking) ───────────────────────────────────

def _handle_recognition_background(face_crop: np.ndarray):
    """
    Runs DeepFace in a daemon thread so pipeline never blocks.
    On match  → start returning-visitor session immediately.
    No match  → fall through to unknown visitor flow.
    """
    global _recognizing, _asking_name, _known_faces_cache
    global _last_known_identity, _last_known_face_id, _last_trigger_ts, _session_active

    try:
        # ── Always reload cache fresh before matching ─────────────────────────
        fresh_faces = _load_known_faces()
        _known_faces_cache = fresh_faces

        verified, name, face_id, confidence = recognize_face(face_crop, fresh_faces)

        if verified and name:
            _last_known_identity = name
            _last_known_face_id  = face_id
            _last_trigger_ts     = time.time()
            _session_active      = True

            # Trigger greet orchestrator for matching users
            _post("/visitor/greet", json={
                "face_id":      face_id,
                "name":         name,
                "is_returning": True,
                "visit_count":  1,
            })
            _post("/faces/visit", params={"face_id": face_id})
            logger.info(f"[DETECTION] Returning visitor: {name} (sim={confidence})")

        else:
            # Unknown — hand off to name-asking flow
            _last_known_identity = ""
            _last_known_face_id  = ""
            _last_trigger_ts     = time.time()
            _asking_name         = True
            _handle_unknown_visitor(face_crop)

    except Exception as e:
        logger.error(f"[DETECTION] _handle_recognition_background error: {e}")
    finally:
        _recognizing = False


# ─── Background: unknown visitor flow ─────────────────────────────────────────

def _handle_unknown_visitor(face_crop: np.ndarray):
    """
    Polls for visitor name entered on frontend, then registers face + starts session.
    Runs inside the already-backgrounded recognition thread — no extra thread needed.
    """
    global _asking_name, _known_faces_cache, _last_known_identity, _last_known_face_id, _last_trigger_ts, _session_active
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

        # Save face FIRST so face_id exists before session references it
        if save_face and visitor_name not in ("Guest", ""):
            embedding = extract_embedding(face_crop)
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

        # Let greet block finalize routing session updates to frontend WS layers
        _post("/visitor/greet", json={
            "face_id":      new_face_id if (save_face and visitor_name not in ("Guest", "")) else "",
            "name":         visitor_name,
            "is_returning": False,
            "visit_count":  1,
        })
        _session_active = True

    except Exception as e:
        logger.error(f"[DETECTION] _handle_unknown_visitor error: {e}")
    finally:
        _last_trigger_ts = time.time() + 60.0
        _asking_name     = False


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(frame_data, known_faces: list = None, draw_mesh: bool = False) -> DetectionResult:
    global _asking_name, _recognizing, _last_trigger_ts, _known_faces_cache, _last_known_identity
    global _face_first_seen_ts, _face_present_last, _face_gone_ts, _session_active, _greeted
    global _last_session_recheck_ts, _rechecking_session

    try:
        frame = _decode_frame(frame_data)
    except Exception as e:
        return DetectionResult(present=False, bbox=None, face_crop=None, error=str(e))

    result = detect_presence(frame, draw_mesh=draw_mesh)
    now    = time.time()

    # ─── Goodbye detection ───────────────────────────────────────────────────
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

    # ─── Passerby filter ─────────────────────────────────────────────────────
    dwell_time = now - _face_first_seen_ts if _face_first_seen_ts > 0 else 0.0
    if dwell_time < _DWELL_REQUIRED:
        result.identity = "..."
        return result

    # ─── Voice greeting — once per visitor arrival ────────────────────────────
    # Fixed: Sends structured biometric payload context instead of raw text strings
    if not _greeted and not _session_active and not _asking_name and not _recognizing:
        _greeted = True
        
        payload = {
            "face_id": _last_known_face_id if _last_known_identity else "",
            "name": _last_known_identity if _last_known_identity else "Unknown",
            "is_returning": True if _last_known_identity else False,
            "visit_count": 1
        }
        
        logger.info(f"[PIPELINE] Offloading initial presence greeting payload to backend: {payload['name']}")
        threading.Thread(
            target=lambda: _post("/visitor/greet", json=payload),
            daemon=True,
        ).start()
        
        result.identity = "..."
        return result

    # ─── Already asking for name ──────────────────────────────────────────────
    if _asking_name:
        result.identity = "Identifying..."
        return result

    # ─── Recognition in-flight (non-blocking) ────────────────────────────────
    if _recognizing:
        result.identity = "Identifying..."
        return result

    # ─── Session active — lock to this person, skip recognition ──────────────
    if _session_active:
        if (not _rechecking_session
                and _last_known_face_id
                and (now - _last_session_recheck_ts) >= _SESSION_RECHECK_INTERVAL):
            _last_session_recheck_ts = now
            _rechecking_session      = True
            threading.Thread(target=_recheck_session_face_background, daemon=True).start()

        if _session_active and _last_known_identity:
            result.identity = _last_known_identity
            result.verified = True
        return result

    # ─── Cooldown ─────────────────────────────────────────────────────────────
    if now - _last_trigger_ts < _COOLDOWN:
        if _last_known_identity:
            result.identity = _last_known_identity
            result.verified = True
        return result

    # ─── Kick off background recognition (non-blocking) ──────────────────────
    _recognizing = True
    threading.Thread(
        target=_handle_recognition_background,
        args=(result.face_crop.copy(),),
        daemon=True,
    ).start()

    result.identity = "Identifying..."
    return result


# ─── Utility ──────────────────────────────────────────────────────────────────

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
            cv2.putText(display, label, (b.x, b.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(display, "NO FACE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("VRK Digital Receptionist", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()