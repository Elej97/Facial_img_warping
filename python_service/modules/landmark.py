import json

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.model_utils import get_model_path

# Try to import connection constants (may not exist in all 0.10.x builds)
try:
    from mediapipe.python.solutions.face_mesh_connections import (
        FACEMESH_CONTOURS,
        FACEMESH_TESSELATION,
    )
    _HAS_CONNECTIONS = True
except Exception:
    _HAS_CONNECTIONS = False


def _build_landmarker():
    model_path = get_model_path('face_landmarker')
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


_landmarker = _build_landmarker()


def detect_landmarks(image_np: np.ndarray) -> list[tuple] | None:
    """Detect 468 landmarks using MediaPipe.
    
    OPTIMIZATION: Downscales very large images to prevent lag.
    """
    h, w = image_np.shape[:2]
    
    # --- Optimization: Downscale for detection if extremely large ---
    # MediaPipe Tasks API handles resizing internally but very high res still adds overhead.
    target_dim = 1280
    if max(h, w) > target_dim:
        scale = target_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        proc_img = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        is_scaled = True
    else:
        proc_img = image_np
        is_scaled = False
        scale = 1.0

    rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    inv_scale = 1.0 / scale if is_scaled else 1.0
    
    # Scale coordinates back to original image size
    return [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]


def _build_blendshape_landmarker():
    model_path = get_model_path('face_landmarker')
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


_blendshape_landmarker = None


def detect_blendshapes(image_np: np.ndarray) -> dict | None:
    """Identity-independent expression scores (ARKit-style 52 blendshapes, 0..1).

    This is the expression signal that does NOT depend on who the face belongs to,
    so it can be compared/transferred across different people. Lazily initialised so
    it does not slow service startup. Returns {category_name: score} or None.
    """
    global _blendshape_landmarker
    if _blendshape_landmarker is None:
        _blendshape_landmarker = _build_blendshape_landmarker()

    h, w = image_np.shape[:2]
    target_dim = 1280
    if max(h, w) > target_dim:
        scale = target_dim / max(h, w)
        proc = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        proc = image_np

    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _blendshape_landmarker.detect(mp_image)
    if not result.face_blendshapes:
        return None
    return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}


def draw_landmarks(image_np: np.ndarray, landmarks: list[tuple]) -> np.ndarray:
    out = image_np.copy()
    n = len(landmarks)

    if _HAS_CONNECTIONS:
        # Draw tessellation at 30% opacity
        tess_layer = out.copy()
        for a, b in FACEMESH_TESSELATION:
            if a < n and b < n:
                cv2.line(tess_layer, landmarks[a], landmarks[b], (128, 128, 128), 1)
        cv2.addWeighted(tess_layer, 0.3, out, 0.7, 0, out)

        # Draw contours
        for a, b in FACEMESH_CONTOURS:
            if a < n and b < n:
                cv2.line(out, landmarks[a], landmarks[b], (0, 200, 0), 1)

    # Draw all landmark points
    for pt in landmarks[:468]:
        cv2.circle(out, pt, 1, (0, 255, 0), -1)

    return out


def export_landmarks(landmarks: list[tuple], output_path: str) -> None:
    data = {"landmarks": [[x, y] for x, y in landmarks], "count": len(landmarks)}
    with open(output_path, 'w') as f:
        json.dump(data, f)


def get_key_landmark_indices() -> dict:
    return {
        "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "mouth_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
        "mouth_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
        "nose": [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 129, 203, 205, 206, 207],
        "jaw": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "left_brow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "right_brow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    }
