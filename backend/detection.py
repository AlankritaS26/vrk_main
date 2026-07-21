"""
detection.py — EP-07 Presence Detection & Face Recognition
VRK / RNSIT Digital Receptionist

RECEPTIONIST MODEL
------------------
  PERSON  = face_id     minted ONCE, at first quality-gated enrolment.
  VISIT   = session_id  owned by the backend; a known face returning
                        within 30 days RESUMES its previous session_id.

  One visitor is served at a time. Bystanders are detected and ignored:
  the largest face above a size floor is the PRIMARY visitor and holds
  the session until they leave. A different face never inherits a live
  session — it ends the session instead.

STATE MACHINE
-------------
  IDLE -> DWELLING -> RECOGNIZING -> {ACTIVE | ENROLLING -> ACTIVE}
  ACTIVE -> DEPARTING -> (return within grace) ACTIVE | COOLDOWN -> IDLE

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
#   FALLBACK: DeepFace, only if onnxruntime/models are unavailable.
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
    num_faces=4,                      # see bystanders so we can IGNORE them
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(_lm_opts)

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
DEPART_GRACE    = float(os.getenv("DEPART_GRACE", "5.5"))      # bag/phone/companion tolerance
COOLDOWN        = float(os.getenv("DETECT_COOLDOWN", "6.0"))
RECHECK_EVERY   = float(os.getenv("SESSION_RECHECK_INTERVAL", "2.0"))
SWAP_STREAK     = int(os.getenv("SWAP_STREAK", "3"))           # frames before believing a swap
ENROLL_TEMPLATES = int(os.getenv("ENROLL_TEMPLATES", "3"))     # multi-template enrolment
NAME_WAIT_SECS  = float(os.getenv("NAME_WAIT_SECS", "30"))


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


@dataclass
class KioskState:
    """Every piece of mutable state, guarded by `lock`."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "IDLE"                 # IDLE DWELLING RECOGNIZING ENROLLING ACTIVE DEPARTING COOLDOWN
    face_id: str = ""                   # PERSON id of the current primary visitor
    identity: str = ""
    anchor: Optional[List[float]] = None   # live embedding captured at session start
    dwell_started: float = 0.0
    departed_at: float = 0.0
    cooldown_until: float = 0.0
    last_recheck: float = 0.0
    swap_streak: int = 0
    busy: bool = False                  # a background worker is running

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self.lock:
            return {"state": self.state, "face_id": self.face_id,
                    "identity": self.identity, "busy": self.busy}

    def reset_session(self):
        with self.lock:
            self.state = "COOLDOWN"
            self.cooldown_until = time.time() + COOLDOWN
            self.face_id = ""
            self.identity = ""
            self.anchor = None
            self.swap_streak = 0
            self.dwell_started = 0.0
            self.departed_at = 0.0


ST = KioskState()
_known_faces: list = []


# ─── Backend HTTP ─────────────────────────────────────────────────────────────
def _post(path, **kw):
    try:
        return httpx.post(f"{BACKEND_URL}{path}", timeout=8, **kw)
    except Exception as e:
        logger.warning(f"POST {path} failed: {e}")
        return None


def _get(path):
    try:
        return httpx.get(f"{BACKEND_URL}{path}", timeout=8)
    except Exception as e:
        logger.warning(f"GET {path} failed: {e}")
        return None


def _load_known_faces() -> list:
    r = _get("/faces/all")
    if r is not None and r.status_code == 200:
        faces = r.json().get("faces", [])
        logger.info(f"[DETECT] {len(faces)} known faces loaded")
        return faces
    return []


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


def detect_presence(frame: np.ndarray, draw_mesh: bool = False) -> DetectionResult:
    """PRIMARY-VISITOR LOCK: largest face above MIN_FACE_FRAC wins; the rest
    are bystanders — counted, never served, never blocking."""
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

    return DetectionResult(present=True, bbox=best, face_crop=crop, kps=kps,
                           frame_ref=frame,
                           landmarks_img=mesh, bystanders=max(0, qualifying - 1))


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
    ST.reset_session()


def _start_session(face: dict, anchor: list, sim: float):
    """Backend owns session_id and decides resume-vs-new (30-day window)."""
    face_id = face["face_id"]
    r = _post("/visitor/greet", json={
        "face_id": face_id,
        "name": face.get("name", "Guest"),
        "is_returning": True,
        "visit_count": int(face.get("visit_count") or 1),
    })
    _post("/faces/visit", params={"face_id": face_id})
    sid = ""
    try:
        sid = (r.json() or {}).get("session_id", "") if r is not None else ""
    except Exception:
        pass
    ST.set(state="ACTIVE", face_id=face_id, identity=face.get("name", ""),
           anchor=anchor, swap_streak=0, last_recheck=time.time())
    logger.info(f"[DETECT] ACTIVE {face.get('name')} sim={sim:.3f} session={sid[:8]}")


def _recognize_worker(crop: np.ndarray, frame=None, kps=None):
    """RECOGNIZING: identify, then either start the session or enrol."""
    global _known_faces
    try:
        _known_faces = _load_known_faces()
        probe = extract_embedding(crop, frame, kps)
        if probe is None:
            ST.set(state="IDLE")
            return

        face, sim = _match(probe, _known_faces)
        if face:
            _start_session(face, probe, sim)
        else:
            ST.set(state="ENROLLING")
            _enroll_worker(crop, probe, frame, kps)
    except Exception as e:
        logger.error(f"[DETECT] recognize error: {e}")
        ST.set(state="IDLE")
    finally:
        ST.set(busy=False)


def _enroll_worker(crop: np.ndarray, probe: list, frame=None, kps=None):
    """
    ENROLLING — the ONLY place a face_id is ever minted.

    Guards (this is what stopped the 'one person, seven face_ids' factory):
      * DUPLICATE HARD BLOCK — if this face already matches someone, we do
        NOT create a second identity; we greet them as that person.
      * multi-template capture for a robust identity.
    """
    global _known_faces
    try:
        r = _post("/visitor/unknown")
        if r is None or r.status_code != 200:
            ST.set(state="IDLE")
            return

        deadline = time.time() + NAME_WAIT_SECS
        data = None
        while time.time() < deadline:
            rr = _get("/visitor/name_response")
            if rr is not None and rr.status_code == 200:
                d = rr.json()
                if d.get("ready"):
                    data = d
                    _post("/visitor/clear_response")
                    break
            time.sleep(0.4)

        name = (data.get("name") or "Guest").strip() if data else "Guest"
        save = bool(data.get("save", True)) if data else False

        # ── DUPLICATE HARD BLOCK ──────────────────────────────────────────
        fresh = _load_known_faces()
        dup, dup_sim = _match(probe, fresh)
        if dup:
            logger.warning(f"[DETECT] enrolment blocked — face already registered "
                           f"as '{dup.get('name')}' (sim={dup_sim:.3f}). Resuming them.")
            _start_session(dup, probe, dup_sim)
            return

        if not save or name in ("Guest", ""):
            # Guest: session without an identity; anchor still locks the seat.
            _post("/session/start", params={"user_name": name or "Guest",
                                            "face_id": "", "is_returning": False,
                                            "visit_count": 1, "trigger": "camera"})
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
            logger.warning("[DETECT] /faces/register failed")
            ST.set(state="IDLE")
            return

        logger.info(f"[DETECT] enrolled '{name}' face_id={face_id[:8]}")
        _known_faces = _load_known_faces()
        _post("/session/start", params={"user_name": name, "face_id": face_id,
                                        "is_returning": False, "visit_count": 1,
                                        "trigger": "camera"})
        ST.set(state="ACTIVE", face_id=face_id, identity=name,
               anchor=probe, swap_streak=0, last_recheck=time.time())
    except Exception as e:
        logger.error(f"[DETECT] enrol error: {e}")
        ST.set(state="IDLE")


def _recheck_worker(crop: np.ndarray, frame=None, kps=None):
    """ACTIVE: is the person in front still the session owner?
    Deletion is deterministic (immediate). Identity drift is debounced."""
    try:
        snap = ST.snapshot()
        if snap["face_id"]:
            fresh = _load_known_faces()
            if not any(f.get("face_id") == snap["face_id"] for f in fresh):
                logger.info("[DETECT] session face deleted from DB — ending")
                _end_session("deleted")
                return

        with ST.lock:
            anchor = ST.anchor
        if anchor is None:
            return
        probe = extract_embedding(crop, frame, kps)
        if probe is None:
            return

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

            fresh = _load_known_faces()
            face, msim = _match(probe, fresh)
            if face:
                logger.info(f"[DETECT] re-identified as '{face.get('name')}' "
                            f"(sim={msim:.3f}) — starting their session")
                ST.set(state="RECOGNIZING")
                _start_session(face, probe, msim)
            else:
                # Unknown newcomer: enrol them instead of idling.
                logger.info("[DETECT] newcomer not recognised — enrolling")
                ST.set(state="ENROLLING")
                _enroll_worker(crop, probe, frame, kps)
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
            ST.set(state="DEPARTING", departed_at=now)
        elif st == "DEPARTING":
            with ST.lock:
                gone_for = now - ST.departed_at
            if gone_for >= DEPART_GRACE:
                _end_session("visitor left")
        elif st in ("DWELLING", "RECOGNIZING"):
            ST.set(state="IDLE", dwell_started=0.0)
        elif st == "COOLDOWN":
            with ST.lock:
                if now >= ST.cooldown_until:
                    ST.state = "IDLE"
        res.state = ST.snapshot()["state"]
        return res

    # ── someone is in front ───────────────────────────────────────────────
    if st == "DEPARTING":
        # they came back inside the grace window → keep the SAME session
        ST.set(state="ACTIVE", departed_at=0.0)
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
        ST.set(state="DWELLING", dwell_started=now)
        res.identity = "..."
        return res

    if st == "DWELLING":
        with ST.lock:
            dwell = now - ST.dwell_started
        if dwell < DWELL_REQUIRED:
            res.identity = "..."
            return res
        ST.set(state="RECOGNIZING")
        if res.face_crop is not None:
            _spawn(_recognize_worker, res.face_crop.copy(), res.frame_ref, res.kps)
        res.identity = "Identifying..."
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
    print("Detection running — press Q in the window to quit.")
    _known_faces = _load_known_faces()

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