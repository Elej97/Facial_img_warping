import importlib.util
import types
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


_VENDOR_UTILS_PATH = Path(__file__).resolve().parents[2] / "external" / "Virtual_Makeup" / "utils.py"
_VENDOR_SPEC = importlib.util.spec_from_file_location("jayanths_virtual_makeup_utils", _VENDOR_UTILS_PATH)
if _VENDOR_SPEC is None or _VENDOR_SPEC.loader is None:
    raise ImportError(f"Could not load Virtual_Makeup utils from {_VENDOR_UTILS_PATH}")

if not hasattr(mp, "solutions"):
    mp.solutions = types.SimpleNamespace(face_mesh=None, drawing_utils=None)

_vendor_utils = importlib.util.module_from_spec(_VENDOR_SPEC)
_VENDOR_SPEC.loader.exec_module(_vendor_utils)


_REGION_FEATURES = {
    "lip": ["LIP_LOWER", "LIP_UPPER"],
    "cheek": ["BLUSH_LEFT", "BLUSH_RIGHT"],
    "brow": ["EYEBROW_LEFT", "EYEBROW_RIGHT"],
    "lash": ["EYESHADOW_LEFT", "EYESHADOW_RIGHT"],
    "eye": ["EYESHADOW_LEFT", "EYESHADOW_RIGHT"],
}

_DEFAULT_COLORS = {
    "lip": (0, 0, 255),
    "cheek": (102, 0, 51),
    "brow": (19, 69, 139),
    "lash": (139, 0, 0),
    "eye": (0, 100, 0),
}

_REGION_OPACITY = {
    "lip": 0.82,
    "cheek": 0.58,
    "brow": 0.72,
    "lash": 0.70,
    "eye": 0.62,
}


def _normalize_hex_color(hex_color: str) -> str:
    value = (hex_color or "").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError("hex_color must be a 3 or 6 digit HEX value")
    int(value, 16)
    return value.upper()


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = _normalize_hex_color(hex_color)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return blue, green, red


def _landmark_dict(landmarks: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    return {idx: (int(point[0]), int(point[1])) for idx, point in enumerate(landmarks)}


def _apply_vendor_makeup(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    features: list[str],
    color_bgr: tuple[int, int, int],
    intensity: float,
    region: str,
) -> np.ndarray:
    idx_to_coordinates = _landmark_dict(landmarks)
    face_connections = [_vendor_utils.face_points[name] for name in features]
    colors = [[255, 255, 255] for _ in features]

    mask = np.zeros_like(image_np)
    mask = _vendor_utils.add_mask(
        mask,
        idx_to_coordinates=idx_to_coordinates,
        face_connections=face_connections,
        colors=colors,
    )

    opacity = float(np.clip(intensity * _REGION_OPACITY.get(region, 0.62), 0.0, 1.0))
    mask_alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = np.expand_dims(np.clip(mask_alpha * opacity, 0.0, 1.0), axis=2)
    color_layer = np.full_like(image_np, color_bgr, dtype=np.uint8)
    result = image_np.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_vendor_blush(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    idx_to_coordinates = _landmark_dict(landmarks)
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(16, int(min(h, w) * 0.085))

    for feature in _REGION_FEATURES["cheek"]:
        for idx in _vendor_utils.face_points[feature]:
            point = idx_to_coordinates.get(idx)
            if point is not None:
                cv2.circle(mask, point, radius, 255, cv2.FILLED, lineType=cv2.LINE_AA)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.55, sigmaY=radius * 0.55)
    mask_alpha = mask.astype(np.float32) / 255.0
    opacity = float(np.clip(intensity * _REGION_OPACITY["cheek"], 0.0, 0.78))
    alpha = np.expand_dims(np.clip(mask_alpha * opacity, 0.0, 1.0), axis=2)
    color_layer = np.full_like(image_np, color_bgr, dtype=np.uint8)
    result = image_np.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_vendor_teeth_passthrough(image_np: np.ndarray) -> np.ndarray:
    return image_np.copy()


def apply_makeup(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    region: str,
    hex_color: str,
    intensity: float = 0.6,
) -> dict:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    region = (region or "").lower().strip()
    if region not in {"lip", "cheek", "brow", "lash", "eye", "teeth"}:
        raise ValueError("region must be one of: lip, cheek, brow, lash, eye, teeth")

    normalized_hex = _normalize_hex_color(hex_color)
    color_bgr = _hex_to_bgr(normalized_hex)

    if region == "teeth":
        result = _apply_vendor_teeth_passthrough(image_np)
    elif region == "cheek":
        result = _apply_vendor_blush(image_np, landmarks, color_bgr, intensity)
    else:
        features = _REGION_FEATURES[region]
        result = _apply_vendor_makeup(image_np, landmarks, features, color_bgr, intensity, region)

    return {
        "result_image": result,
        "region": region,
        "hex_color": normalized_hex,
        "intensity": intensity,
    }
