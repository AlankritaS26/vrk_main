"""
recognition.py — face embeddings for the VRK kiosk.

WHY THIS FILE EXISTS
--------------------
We want InsightFace quality (ArcFace embeddings on ALIGNED faces) without the
`insightface` pip package, which fails to install on Windows because it tries
to compile a Cython 3D-mesh extension we never use (needs MS C++ Build Tools).

So we load the SAME buffalo_l ONNX models directly with onnxruntime — a clean
prebuilt wheel on Windows/Linux/macOS, no compiler anywhere. Same models, same
accuracy, same speed, zero build tooling.

WHAT IT DOES
------------
  1. SCRFD detector  -> face box + 5 landmarks (eyes, nose, mouth corners)
  2. Similarity transform aligns the face to the canonical 112x112 pose
     (this alignment step is what stops look-alike webcam crops being confused)
  3. ArcFace w600k_r50 -> 512-d L2-normalised embedding

Models download once (~280 MB) to ~/.insightface/models/buffalo_l/.
"""

from __future__ import annotations

import os
import zipfile
import logging
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(os.getenv(
    "INSIGHT_MODEL_DIR",
    Path.home() / ".insightface" / "models" / "buffalo_l"))
_MODEL_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

_DET_FILE = "det_10g.onnx"          # SCRFD detector
_REC_FILE = "w600k_r50.onnx"        # ArcFace embedder

# Canonical 5-point template for 112x112 ArcFace input (standard values).
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_det_session = None
_rec_session = None
_available = None


# ──────────────────────────────────────────────────────────────────────────
# Model files
# ──────────────────────────────────────────────────────────────────────────
def _ensure_models() -> bool:
    """Download + unpack buffalo_l once. Returns True if both models exist."""
    det, rec = _MODEL_DIR / _DET_FILE, _MODEL_DIR / _REC_FILE
    if det.exists() and rec.exists():
        return True

    try:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = _MODEL_DIR.parent / "buffalo_l.zip"
        logger.info("[RECOG] Downloading buffalo_l models (~280 MB, one time)...")
        urllib.request.urlretrieve(_MODEL_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = os.path.basename(member)
                if name in (_DET_FILE, _REC_FILE):
                    with zf.open(member) as src, open(_MODEL_DIR / name, "wb") as dst:
                        dst.write(src.read())
        zip_path.unlink(missing_ok=True)
        logger.info(f"[RECOG] Models ready in {_MODEL_DIR}")
        return det.exists() and rec.exists()
    except Exception as e:
        logger.error(f"[RECOG] Model download failed: {e}")
        return False


def is_available() -> bool:
    """True if onnxruntime + models are usable. Cached after first check."""
    global _available
    if _available is None:
        try:
            import onnxruntime  # noqa: F401
            _available = _ensure_models()
        except ImportError:
            logger.warning("[RECOG] onnxruntime not installed")
            _available = False
    return _available


def _sessions():
    """Lazy-load both ONNX sessions (CPU)."""
    global _det_session, _rec_session
    if _det_session is None or _rec_session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "4"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        _det_session = ort.InferenceSession(str(_MODEL_DIR / _DET_FILE),
                                            sess_options=opts, providers=providers)
        _rec_session = ort.InferenceSession(str(_MODEL_DIR / _REC_FILE),
                                            sess_options=opts, providers=providers)
        logger.info("[RECOG] ArcFace + SCRFD sessions ready (CPU)")
    return _det_session, _rec_session


# ──────────────────────────────────────────────────────────────────────────
# SCRFD detection
# ──────────────────────────────────────────────────────────────────────────
def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, (i + 1) % 2] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float = 0.4) -> List[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


def detect_faces(img: np.ndarray, det_thresh: float = 0.5,
                 input_size: int = 0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Returns [(bbox[x1,y1,x2,y2], kps[5,2], score), ...] sorted by area desc.
    """
    if not is_available():
        return []

    det, _ = _sessions()
    h0, w0 = img.shape[:2]

    # SCRFD runs on a square canvas; its cost scales with canvas area. Full
    # frames need 640, but the kiosk mostly passes small face crops — forcing
    # 640 there wasted ~10x the compute (measured 1.9 s vs 0.2 s per crop).
    if input_size <= 0:
        longest = max(h0, w0)
        input_size = 320 if longest <= 400 else (480 if longest <= 800 else 640)
    input_size = int(np.ceil(input_size / 32) * 32)      # stride-32 friendly
    scale = min(input_size / max(h0, w0), input_size / max(h0, w0))
    nw, nh = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized

    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (input_size, input_size),
                                 (127.5, 127.5, 127.5), swapRB=True)
    outputs = det.run(None, {det.get_inputs()[0].name: blob})

    scores_list, bboxes_list, kps_list = [], [], []
    fmc, feat_strides = 3, [8, 16, 32]
    num_anchors = 2

    for idx, stride in enumerate(feat_strides):
        scores = outputs[idx]
        bbox_preds = outputs[idx + fmc] * stride
        kps_preds = outputs[idx + fmc * 2] * stride

        height, width = input_size // stride, input_size // stride
        ax, ay = np.meshgrid(np.arange(width), np.arange(height))
        anchor_centers = np.stack([ax, ay], axis=-1).astype(np.float32) * stride
        anchor_centers = anchor_centers.reshape(-1, 2)
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(-1, 2)

        scores = scores.reshape(-1)
        pos = np.where(scores >= det_thresh)[0]
        if pos.size == 0:
            continue
        bboxes = _distance2bbox(anchor_centers, bbox_preds.reshape(-1, 4))
        kpss = _distance2kps(anchor_centers, kps_preds.reshape(-1, 10)).reshape(-1, 5, 2)

        scores_list.append(scores[pos])
        bboxes_list.append(bboxes[pos])
        kps_list.append(kpss[pos])

    if not scores_list:
        return []

    scores = np.concatenate(scores_list)
    bboxes = np.concatenate(bboxes_list) / scale
    kpss = np.concatenate(kps_list) / scale

    pre_det = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
    keep = _nms(pre_det)

    results = []
    for i in keep:
        b = bboxes[i]
        b[0::2] = np.clip(b[0::2], 0, w0)
        b[1::2] = np.clip(b[1::2], 0, h0)
        results.append((b, kpss[i], float(scores[i])))

    results.sort(key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1]), reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────
# Alignment + embedding
# ──────────────────────────────────────────────────────────────────────────
def _align(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Similarity transform of the 5 landmarks onto the ArcFace template.
    THIS is the step plain crops lack — it removes pose/scale variation."""
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), _ARCFACE_DST,
                                       method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img, (112, 112))
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


def embed_aligned(face_112: np.ndarray) -> Optional[List[float]]:
    _, rec = _sessions()
    blob = cv2.dnn.blobFromImage(face_112, 1.0 / 127.5, (112, 112),
                                 (127.5, 127.5, 127.5), swapRB=True)
    out = rec.run(None, {rec.get_inputs()[0].name: blob})[0][0]
    norm = np.linalg.norm(out)
    if norm == 0:
        return None
    return (out / norm).astype(np.float32).tolist()      # L2-normalised


def embed_with_landmarks(frame: np.ndarray, kps: np.ndarray) -> Optional[List[float]]:
    """FAST PATH — the caller already has 5-point landmarks from a full-frame
    detection, so we skip detection entirely: align + embed only (~40-80 ms).
    This is what the kiosk pipeline uses every session."""
    try:
        return embed_aligned(_align(frame, kps))
    except Exception as e:
        logger.error(f"[RECOG] embed_with_landmarks failed: {e}")
        return None


def get_embedding(image: np.ndarray) -> Optional[List[float]]:
    """
    Full pipeline on any image (frame or crop): detect -> pick largest ->
    align -> embed. Returns a 512-d L2-normalised vector, or None.
    """
    if not is_available() or image is None or image.size == 0:
        return None
    try:
        faces = detect_faces(image)
        if not faces:
            # Crop may already be a tight face: try direct alignment-free embed
            if min(image.shape[:2]) >= 40:
                return embed_aligned(cv2.resize(image, (112, 112)))
            return None
        bbox, kps, _ = faces[0]
        return embed_aligned(_align(image, kps))
    except Exception as e:
        logger.error(f"[RECOG] embedding failed: {e}")
        return None


def warm_up():
    """Build ONNX graphs at boot so the first visitor pays no penalty."""
    try:
        if is_available():
            _sessions()
            get_embedding(np.zeros((160, 160, 3), dtype=np.uint8))
            logger.info("[RECOG] warmed up")
    except Exception as e:
        logger.warning(f"[RECOG] warmup skipped: {e}")