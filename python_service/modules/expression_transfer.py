"""Expression transfer (Ifade aktarma).

Transfers the *expression* of a reference face onto a target face WITHOUT copying the
reference person's identity/anatomy. The expression is read as MediaPipe blendshape
scores (ARKit-style, identity-independent), so "reference minus target" is a pure
expression difference. That difference is turned into landmark displacements expressed
in the TARGET's own facial frame (scaled by the target's inter-ocular distance), then
applied with a Delaunay warp.

This replaces the previous approach, which pushed the target's landmarks toward the
reference person's absolute landmark positions — that transferred anatomy, not
expression, so it did nothing when the two faces were anatomically similar.
"""
from __future__ import annotations

import cv2
import numpy as np

from modules.landmark import detect_blendshapes
from modules.warping import apply_delaunay_warp

# Face-frame reference points (MediaPipe 468 topology).
_L_EYE_OUT, _R_EYE_OUT = 33, 263   # inter-ocular axis -> "right" direction + scale
_FOREHEAD_TOP, _CHIN = 10, 152     # vertical axis -> "up" direction

# Per-expression rules. For each rule, `bs` blendshapes are averaged into a signal; the
# signal is (reference - target) so it is a pure expression delta. Each point is moved
# by gain * signal * IOD * (d_right * right_vec + d_up * up_vec). d_right is +toward the
# subject's left in the image (outer-eye vector), d_up is +toward the forehead.
_RULES: list[dict] = [
    {
        "name": "smile", "bs": ["mouthSmileLeft", "mouthSmileRight"], "gain": 0.22,
        "points": [
            (61, -0.45, 0.32), (291, 0.45, 0.32),     # mouth corners up & out
            (146, -0.28, 0.18), (375, 0.28, 0.18),     # outer-lip near corners
            (91, -0.20, 0.14), (321, 0.20, 0.14),
            (185, -0.16, 0.10), (409, 0.16, 0.10),
            (205, 0.0, 0.12), (425, 0.0, 0.12),        # cheeks lift
        ],
    },
    {
        "name": "frown", "bs": ["mouthFrownLeft", "mouthFrownRight"], "gain": 0.20,
        "points": [(61, 0.0, -0.5), (291, 0.0, -0.5), (146, 0.0, -0.3), (375, 0.0, -0.3)],
    },
    {
        "name": "jawOpen", "bs": ["jawOpen"], "gain": 0.30,
        "points": [
            (17, 0.0, -0.65), (14, 0.0, -0.55), (84, 0.0, -0.45), (314, 0.0, -0.45),
            (181, 0.0, -0.40), (405, 0.0, -0.40), (152, 0.0, -0.85), (175, 0.0, -0.70),
            (148, 0.0, -0.50), (377, 0.0, -0.50),
        ],
    },
    {
        "name": "pucker", "bs": ["mouthPucker"], "gain": 0.16,
        "points": [(61, 0.45, 0.0), (291, -0.45, 0.0), (0, 0.0, -0.1), (17, 0.0, 0.1)],
    },
    {
        "name": "browInnerUp", "bs": ["browInnerUp"], "gain": 0.04,
        "points": [(55, 0.0, 0.85), (65, 0.0, 0.70), (285, 0.0, 0.85), (295, 0.0, 0.70),
                   (107, 0.0, 0.55), (336, 0.0, 0.55)],
    },
    {
        "name": "browDown", "bs": ["browDownLeft", "browDownRight"], "gain": 0.03,
        "points": [(55, 0.0, -0.6), (65, 0.0, -0.55), (52, 0.0, -0.5), (46, 0.0, -0.4),
                   (285, 0.0, -0.6), (295, 0.0, -0.55), (282, 0.0, -0.5), (276, 0.0, -0.4)],
    },
    {
        "name": "squint", "bs": ["eyeSquintLeft", "eyeSquintRight"], "gain": 0.06,
        "points": [(145, 0.0, 0.4), (153, 0.0, 0.35), (144, 0.0, 0.3),
                   (374, 0.0, 0.4), (380, 0.0, 0.35), (373, 0.0, 0.3)],
    },
]

# Overall strength multiplier on top of the per-rule gains. Tuned so a moderate
# expression difference produces a clearly visible (but natural) warp.
_GLOBAL_GAIN = 4.5

# Safety cap on a single point's displacement, as a fraction of inter-ocular distance.
# Kept tight so large expression differences warp partially rather than tearing the mesh
# (the sparse landmark control points flip triangles if pushed too far, esp. the brows).
_MAX_DISP_FRAC = 0.10


def _face_frame(landmarks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.array(landmarks, dtype=np.float32)

    def _safe(i, fallback):
        return pts[i] if i < len(pts) else np.array(fallback, dtype=np.float32)

    le, re = _safe(_L_EYE_OUT, [0, 0]), _safe(_R_EYE_OUT, [1, 0])
    right = re - le
    iod = float(np.linalg.norm(right))
    if iod < 1e-3:
        return np.array([1.0, 0.0], np.float32), np.array([0.0, -1.0], np.float32), 1.0
    right = right / iod

    top, chin = _safe(_FOREHEAD_TOP, [0, 0]), _safe(_CHIN, [0, 1])
    up = top - chin
    n = float(np.linalg.norm(up))
    # Fall back to right-rotated-90 if vertical landmarks are degenerate.
    up = up / n if n > 1e-3 else np.array([right[1], -right[0]], dtype=np.float32)
    return right, up, iod


def _mean_score(bs: dict, keys: list[str]) -> float:
    vals = [bs.get(k, 0.0) for k in keys]
    return float(np.mean(vals)) if vals else 0.0


def _build_destination_landmarks(
    target_landmarks: list[tuple[int, int]],
    target_bs: dict,
    reference_bs: dict,
    intensity: float,
) -> list[tuple[int, int]]:
    target_array = np.array(target_landmarks, dtype=np.float32)
    destination = target_array.copy()

    right_vec, up_vec, iod = _face_frame(target_landmarks)
    max_disp = _MAX_DISP_FRAC * iod
    n = len(target_landmarks)

    for rule in _RULES:
        signal = (_mean_score(reference_bs, rule["bs"]) - _mean_score(target_bs, rule["bs"])) * intensity
        if abs(signal) < 1e-4:
            continue
        amp = _GLOBAL_GAIN * rule["gain"] * signal * iod
        for idx, d_right, d_up in rule["points"]:
            if idx >= n:
                continue
            disp = amp * (d_right * right_vec + d_up * up_vec)
            destination[idx] = destination[idx] + disp

    # Cap per-point displacement so an over-driven blendshape can never tear the face.
    delta = destination - target_array
    mag = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.where(mag > max_disp, max_disp / np.maximum(mag, 1e-6), 1.0)
    destination = target_array + delta * scale

    return [(int(round(x)), int(round(y))) for x, y in destination]


def transfer_expression(
    target_image: np.ndarray,
    target_landmarks: list[tuple[int, int]],
    reference_landmarks: list[tuple[int, int]],
    intensity: float = 0.7,
    reference_image: np.ndarray | None = None,
    target_blendshapes: dict | None = None,
    reference_blendshapes: dict | None = None,
) -> dict:
    intensity = float(np.clip(intensity, 0.0, 1.0))

    if target_blendshapes is None:
        target_blendshapes = detect_blendshapes(target_image)
    if reference_blendshapes is None and reference_image is not None:
        reference_blendshapes = detect_blendshapes(reference_image)

    if not target_blendshapes or not reference_blendshapes:
        # Cannot read an identity-free expression for one of the faces — do not fall back
        # to anatomy transfer; leave the target unchanged so we never corrupt the face.
        return {
            "result_image": target_image.copy(),
            "destination_landmarks": [(int(x), int(y)) for x, y in target_landmarks],
            "aligned_reference_landmarks": [(int(x), int(y)) for x, y in reference_landmarks],
            "expression_applied": False,
        }

    destination_landmarks = _build_destination_landmarks(
        target_landmarks, target_blendshapes, reference_blendshapes, intensity
    )
    result_image = apply_delaunay_warp(target_image, target_landmarks, destination_landmarks)

    return {
        "result_image": result_image,
        "destination_landmarks": destination_landmarks,
        "aligned_reference_landmarks": [(int(x), int(y)) for x, y in reference_landmarks],
        "expression_applied": True,
    }
