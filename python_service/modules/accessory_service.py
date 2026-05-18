import math
from pathlib import Path

import cv2
import numpy as np

from modules.landmark import get_key_landmark_indices


_ACCESSORY_ALIASES = {
    "glasses": "glasses",
    "mustache": "mustache",
    "moustache": "mustache",
    "hat": "hat",
}

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "accessories"
_GLASSES_ASSETS = {
    "classic": _ASSET_ROOT / "glasses" / "user_black_square_clean.png",
    "round": _ASSET_ROOT / "glasses" / "user_gold_frame.png",
    "aviator": _ASSET_ROOT / "glasses" / "user_gold_frame.png",
    "heart": _ASSET_ROOT / "glasses" / "user_pink_heart.png",
}
_MUSTACHE_ASSETS = {
    "classic": _ASSET_ROOT / "mustache" / "handlebar_asset.png",
    "handlebar": _ASSET_ROOT / "mustache" / "handlebar_asset.png",
    "chevron": _ASSET_ROOT / "mustache" / "full_beard_asset.png",
}
_HAT_ASSETS = {
    "cowboy": _ASSET_ROOT / "hats" / "CowboyHat.jpg",
    "cap": _ASSET_ROOT / "hats" / "cap.jpg",
    "asian": _ASSET_ROOT / "hats" / "asianHat.jpg",
    "newasian": _ASSET_ROOT / "hats" / "newAsianHat.png",
    "pink": _ASSET_ROOT / "hats" / "pinkHat.jpg",
}


def _normalize_hex_color(hex_color: str, fallback: str = "1D1D1F") -> str:
    value = (hex_color or fallback).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        value = fallback
    int(value, 16)
    return value.upper()


def _hex_to_bgr(hex_color: str, fallback: str = "1D1D1F") -> tuple[int, int, int]:
    value = _normalize_hex_color(hex_color, fallback=fallback)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return blue, green, red


def _safe_points(landmarks: list[tuple[int, int]], indices: list[int]) -> np.ndarray:
    points = [landmarks[i] for i in indices if i < len(landmarks)]
    return np.array(points, dtype=np.float32)


def _point_at(landmarks: list[tuple[int, int]], index: int, fallback: tuple[float, float]) -> np.ndarray:
    if index < len(landmarks):
        return np.array(landmarks[index], dtype=np.float32)
    return np.array(fallback, dtype=np.float32)


def _blend(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    mask = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0) * opacity
    mask3 = mask[:, :, None]
    out = base.astype(np.float32) * (1.0 - mask3) + layer.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_rgba_asset(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    asset = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if asset is None or asset.ndim != 3 or asset.shape[2] != 4:
        return None
    alpha = asset[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return None

    pad = 6
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(asset.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(asset.shape[0], int(ys.max()) + pad + 1)
    return asset[y0:y1, x0:x1].copy()


def _tint_frame_pixels(asset_bgra: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    tinted = asset_bgra.copy()
    bgr = tinted[:, :, :3].astype(np.float32)
    alpha = tinted[:, :, 3]
    luminance = bgr[:, :, 2] * 0.299 + bgr[:, :, 1] * 0.587 + bgr[:, :, 0] * 0.114
    frame_mask = (alpha > 20) & (luminance < 105)
    if np.any(frame_mask):
        target = np.array(color_bgr, dtype=np.float32)
        bgr[frame_mask] = bgr[frame_mask] * 0.18 + target * 0.82
        tinted[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    return tinted


def _composite_rgba(
    image_bgr: np.ndarray,
    asset_bgra: np.ndarray,
    center: tuple[float, float],
    target_width: float,
    angle: float,
    opacity: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    src_h, src_w = asset_bgra.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_width <= 2:
        return image_bgr.copy()

    target_w = int(np.clip(target_width, 8, w * 2.2))
    target_h = max(2, int(target_w * src_h / src_w))
    resized = cv2.resize(asset_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

    rot_mat = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    bound_w = int(target_h * sin + target_w * cos)
    bound_h = int(target_h * cos + target_w * sin)
    rot_mat[0, 2] += bound_w / 2 - target_w / 2
    rot_mat[1, 2] += bound_h / 2 - target_h / 2

    rotated = cv2.warpAffine(
        resized,
        rot_mat,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    x0 = int(round(center[0] - bound_w / 2))
    y0 = int(round(center[1] - bound_h / 2))
    x1 = x0 + bound_w
    y1 = y0 + bound_h

    crop_x0 = max(0, x0)
    crop_y0 = max(0, y0)
    crop_x1 = min(w, x1)
    crop_y1 = min(h, y1)
    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return image_bgr.copy()

    asset_x0 = crop_x0 - x0
    asset_y0 = crop_y0 - y0
    asset_crop = rotated[asset_y0:asset_y0 + (crop_y1 - crop_y0), asset_x0:asset_x0 + (crop_x1 - crop_x0)]

    alpha = np.clip(asset_crop[:, :, 3].astype(np.float32) / 255.0, 0.0, 1.0)
    alpha = alpha * float(np.clip(opacity, 0.0, 1.0))
    alpha3 = alpha[:, :, None]

    out = image_bgr.copy().astype(np.float32)
    roi = out[crop_y0:crop_y1, crop_x0:crop_x1]
    asset_bgr = asset_crop[:, :, :3].astype(np.float32)
    out[crop_y0:crop_y1, crop_x0:crop_x1] = roi * (1.0 - alpha3) + asset_bgr * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_hat_asset(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    # PNG saved with .jpg extension — has real alpha channel
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3].copy()
        # Fill small holes and clean up edges
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)
        # Remove isolated background pixels that leaked into the hat
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        out = raw.copy()
        out[:, :, 3] = alpha
        ys, xs = np.where(alpha > 8)
        if len(xs) == 0 or len(ys) == 0:
            return None
        pad = 6
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(raw.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(raw.shape[0], int(ys.max()) + pad + 1)
        return out[y0:y1, x0:x1].copy()

    # True JPEG (3-channel) — detect up to 2 background colors (handles checkerboard)
    img = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    patch = max(6, min(20, h // 10, w // 10))
    corner_px = np.concatenate([
        img[:patch, :patch].reshape(-1, 3),
        img[:patch, -patch:].reshape(-1, 3),
        img[-patch:, :patch].reshape(-1, 3),
        img[-patch:, -patch:].reshape(-1, 3),
    ]).astype(np.float32)

    bg1 = np.median(corner_px, axis=0)
    alt = corner_px[np.abs(corner_px - bg1).max(axis=1) > 20]
    bg2 = np.median(alt, axis=0) if len(alt) >= 6 else bg1

    img_f = img.astype(np.float32)
    dist = np.minimum(
        np.abs(img_f - bg1).max(axis=2),
        np.abs(img_f - bg2).max(axis=2),
    )
    alpha = np.clip((dist - 15) * (255.0 / 30.0), 0, 255).astype(np.uint8)

    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k_big)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k_small)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    rgba = np.dstack([img, alpha])
    ys, xs = np.where(alpha > 20)
    if len(xs) == 0 or len(ys) == 0:
        return None

    pad = 6
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    return rgba[y0:y1, x0:x1].copy()


def _draw_hat(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    style: str,
    intensity: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]

    # Forehead top anchor (landmark 10 in MediaPipe 468-point mesh)
    forehead = _point_at(landmarks, 10, (w * 0.5, h * 0.18))
    left_face = _point_at(landmarks, 234, (w * 0.28, h * 0.5))
    right_face = _point_at(landmarks, 454, (w * 0.72, h * 0.5))

    face_width = float(np.linalg.norm(right_face - left_face))
    if face_width < 8:
        raise ValueError("Face landmarks are not reliable enough for hat placement.")

    # Tilt angle from eye geometry
    key = get_key_landmark_indices()
    left_eye = _safe_points(landmarks, key["left_eye"])
    right_eye = _safe_points(landmarks, key["right_eye"])
    if len(left_eye) < 2 or len(right_eye) < 2:
        angle = 0.0
    else:
        eye_delta = right_eye.mean(axis=0) - left_eye.mean(axis=0)
        angle = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))

    asset = _load_hat_asset(_HAT_ASSETS.get(style, _HAT_ASSETS["cowboy"]))
    if asset is None:
        raise ValueError(f"Hat asset could not be loaded for style: {style}")

    asset_h, asset_w = asset.shape[:2]
    # Hats are typically wider than the face
    width_factor = 1.55 if style == "asian" else 1.45
    target_width = face_width * width_factor * scale
    target_height = target_width * asset_h / asset_w

    # Place the hat above the forehead: center of the asset is target_height/2 above forehead
    center = (
        float(forehead[0] + offset_x),
        float(forehead[1] - target_height * 0.48 + offset_y),
    )

    return _composite_rgba(
        image_np,
        asset,
        center=center,
        target_width=target_width,
        angle=angle,
        opacity=0.78 + intensity * 0.22,
    )


def _eye_geometry(landmarks: list[tuple[int, int]]) -> dict:
    key = get_key_landmark_indices()
    left_eye = _safe_points(landmarks, key["left_eye"])
    right_eye = _safe_points(landmarks, key["right_eye"])
    if len(left_eye) < 4 or len(right_eye) < 4:
        raise ValueError("Eye landmarks are not reliable enough for glasses.")

    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    eye_delta = right_center - left_center
    eye_distance = float(np.linalg.norm(eye_delta))
    if eye_distance < 8:
        raise ValueError("Eye distance is too small for accessory placement.")

    angle = math.degrees(math.atan2(float(eye_delta[1]), float(eye_delta[0])))
    return {
        "left_center": left_center,
        "right_center": right_center,
        "eye_distance": eye_distance,
        "angle": angle,
    }


def _draw_glasses(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    style: str,
    color_bgr: tuple[int, int, int],
    intensity: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    geometry = _eye_geometry(landmarks)
    asset = _load_rgba_asset(_GLASSES_ASSETS.get(style, _GLASSES_ASSETS["classic"]))
    if asset is not None:
        asset = _tint_frame_pixels(asset, color_bgr)
        midpoint = (geometry["left_center"] + geometry["right_center"]) * 0.5
        eye_distance = geometry["eye_distance"]
        center_y_bias = 0.10 if style == "heart" else 0.03
        center = (
            float(midpoint[0] + offset_x),
            float(midpoint[1] + eye_distance * center_y_bias + offset_y),
        )
        if style == "heart":
            width_factor = 2.70
        elif style == "round":
            width_factor = 2.24
        else:
            width_factor = 2.34
        target_width = eye_distance * width_factor * scale
        return _composite_rgba(
            image_np,
            asset,
            center=center,
            target_width=target_width,
            angle=geometry["angle"],
            opacity=0.66 + intensity * 0.34,
        )

    eye_distance = geometry["eye_distance"] * scale
    angle = geometry["angle"]
    vertical_offset = eye_distance * 0.02 + offset_y
    horizontal_offset = offset_x

    left_center = geometry["left_center"] + np.array([horizontal_offset, vertical_offset], dtype=np.float32)
    right_center = geometry["right_center"] + np.array([horizontal_offset, vertical_offset], dtype=np.float32)

    frame = np.zeros_like(image_np)
    frame_mask = np.zeros((h, w), dtype=np.uint8)
    lens = np.zeros_like(image_np)
    lens_mask = np.zeros((h, w), dtype=np.uint8)

    frame_thickness = max(2, int(eye_distance * (0.035 + 0.012 * intensity)))
    bridge_thickness = max(2, int(frame_thickness * 0.85))

    if style == "round":
        axes = (int(eye_distance * 0.23), int(eye_distance * 0.20))
    elif style == "aviator":
        axes = (int(eye_distance * 0.26), int(eye_distance * 0.21))
    else:
        axes = (int(eye_distance * 0.25), int(eye_distance * 0.16))

    for center in (left_center, right_center):
        c = (int(center[0]), int(center[1]))
        cv2.ellipse(lens_mask, c, axes, angle, 0, 360, 180, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(lens, c, axes, angle, 0, 360, (38, 42, 48), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(frame_mask, c, axes, angle, 0, 360, 255, frame_thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, c, axes, angle, 0, 360, color_bgr, frame_thickness, lineType=cv2.LINE_AA)

    bridge_start = (int(left_center[0] + axes[0] * 0.74), int(left_center[1]))
    bridge_end = (int(right_center[0] - axes[0] * 0.74), int(right_center[1]))
    cv2.line(frame_mask, bridge_start, bridge_end, 255, bridge_thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, bridge_start, bridge_end, color_bgr, bridge_thickness, lineType=cv2.LINE_AA)

    temple_offset = np.array([eye_distance * 0.22, eye_distance * 0.05], dtype=np.float32)
    cv2.line(
        frame_mask,
        (int(left_center[0] - axes[0]), int(left_center[1])),
        (int(left_center[0] - axes[0] - temple_offset[0]), int(left_center[1] + temple_offset[1])),
        255,
        max(1, frame_thickness - 1),
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (int(left_center[0] - axes[0]), int(left_center[1])),
        (int(left_center[0] - axes[0] - temple_offset[0]), int(left_center[1] + temple_offset[1])),
        color_bgr,
        max(1, frame_thickness - 1),
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        frame_mask,
        (int(right_center[0] + axes[0]), int(right_center[1])),
        (int(right_center[0] + axes[0] + temple_offset[0]), int(right_center[1] + temple_offset[1])),
        255,
        max(1, frame_thickness - 1),
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (int(right_center[0] + axes[0]), int(right_center[1])),
        (int(right_center[0] + axes[0] + temple_offset[0]), int(right_center[1] + temple_offset[1])),
        color_bgr,
        max(1, frame_thickness - 1),
        lineType=cv2.LINE_AA,
    )

    out = _blend(image_np, lens, cv2.GaussianBlur(lens_mask, (5, 5), 0), opacity=0.16 + intensity * 0.22)
    return _blend(out, frame, frame_mask, opacity=0.74 + intensity * 0.26)


def _draw_mustache(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    style: str,
    color_bgr: tuple[int, int, int],
    intensity: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    mouth_left = _point_at(landmarks, 61, (w * 0.42, h * 0.62))
    mouth_right = _point_at(landmarks, 291, (w * 0.58, h * 0.62))
    upper_lip = _point_at(landmarks, 13, ((mouth_left[0] + mouth_right[0]) / 2, h * 0.62))
    nose_tip = _point_at(landmarks, 2, (upper_lip[0], upper_lip[1] - h * 0.10))
    chin = _point_at(landmarks, 152, (upper_lip[0], h * 0.82))

    mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
    if mouth_width < 8:
        raise ValueError("Mouth landmarks are not reliable enough for mustache placement.")

    asset = _load_rgba_asset(_MUSTACHE_ASSETS.get(style, _MUSTACHE_ASSETS["classic"]))
    if asset is not None:
        angle = math.degrees(math.atan2(float(mouth_right[1] - mouth_left[1]), float(mouth_right[0] - mouth_left[0])))
        if style == "chevron":
            center = (
                float((mouth_left[0] + mouth_right[0]) * 0.5 + offset_x),
                float((upper_lip[1] * 0.28 + chin[1] * 0.72) + offset_y),
            )
            target_width = mouth_width * 2.62 * scale
            opacity = 0.50 + intensity * 0.48
        else:
            center = (
                float((mouth_left[0] + mouth_right[0]) * 0.5 + offset_x),
                float((upper_lip[1] * 0.72 + nose_tip[1] * 0.28) + offset_y),
            )
            target_width = mouth_width * 1.46 * scale
            opacity = 0.55 + intensity * 0.43

        return _composite_rgba(
            image_np,
            asset,
            center=center,
            target_width=target_width,
            angle=angle,
            opacity=opacity,
        )

    center = np.array(
        [
            (mouth_left[0] + mouth_right[0]) * 0.5 + offset_x,
            upper_lip[1] * 0.68 + nose_tip[1] * 0.32 + offset_y,
        ],
        dtype=np.float32,
    )
    angle = math.degrees(math.atan2(float(mouth_right[1] - mouth_left[1]), float(mouth_right[0] - mouth_left[0])))
    width = mouth_width * (1.16 if style == "handlebar" else 0.98) * scale
    height = mouth_width * (0.20 if style == "chevron" else 0.15) * scale

    layer = np.zeros_like(image_np)
    mask = np.zeros((h, w), dtype=np.uint8)

    left_center = (int(center[0] - width * 0.22), int(center[1]))
    right_center = (int(center[0] + width * 0.22), int(center[1]))
    axes = (max(3, int(width * 0.29)), max(2, int(height)))

    if style == "handlebar":
        cv2.ellipse(mask, left_center, axes, angle - 10, 180, 360, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, right_center, axes, angle + 10, 180, 360, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(layer, left_center, axes, angle - 10, 180, 360, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(layer, right_center, axes, angle + 10, 180, 360, color_bgr, -1, lineType=cv2.LINE_AA)

        curl_radius = max(4, int(height * 0.9))
        cv2.circle(mask, (int(center[0] - width * 0.54), int(center[1] - height * 0.35)), curl_radius, 255, 2, lineType=cv2.LINE_AA)
        cv2.circle(layer, (int(center[0] - width * 0.54), int(center[1] - height * 0.35)), curl_radius, color_bgr, 2, lineType=cv2.LINE_AA)
        cv2.circle(mask, (int(center[0] + width * 0.54), int(center[1] - height * 0.35)), curl_radius, 255, 2, lineType=cv2.LINE_AA)
        cv2.circle(layer, (int(center[0] + width * 0.54), int(center[1] - height * 0.35)), curl_radius, color_bgr, 2, lineType=cv2.LINE_AA)
    else:
        cv2.ellipse(mask, left_center, axes, angle - 8, 190, 355, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(mask, right_center, axes, angle + 8, 185, 350, 255, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(layer, left_center, axes, angle - 8, 190, 355, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(layer, right_center, axes, angle + 8, 185, 350, color_bgr, -1, lineType=cv2.LINE_AA)

    notch_radius = max(2, int(height * 0.35))
    cv2.circle(mask, (int(center[0]), int(center[1] - height * 0.16)), notch_radius, 0, -1, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return _blend(image_np, layer, mask, opacity=0.46 + intensity * 0.48)


def apply_accessory(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    accessory_type: str,
    style: str = "classic",
    hex_color: str = "#1D1D1F",
    intensity: float = 0.7,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict:
    accessory = _ACCESSORY_ALIASES.get((accessory_type or "").lower().strip())
    if accessory not in {"glasses", "mustache", "hat"}:
        raise ValueError("accessory_type must be one of: glasses, mustache, hat")

    style = (style or "classic").lower().strip()
    intensity = float(np.clip(intensity, 0.0, 1.0))
    scale = float(np.clip(scale, 0.6, 1.6))
    offset_x = float(np.clip(offset_x, -120.0, 120.0))
    offset_y = float(np.clip(offset_y, -120.0, 120.0))
    color_bgr = _hex_to_bgr(hex_color)

    if accessory == "glasses":
        if style not in {"classic", "round", "aviator", "heart"}:
            style = "classic"
        result = _draw_glasses(image_np, landmarks, style, color_bgr, intensity, scale, offset_x, offset_y)
    elif accessory == "hat":
        if style not in {"cowboy", "cap", "asian", "newasian", "pink"}:
            style = "cowboy"
        result = _draw_hat(image_np, landmarks, style, intensity, scale, offset_x, offset_y)
    else:
        if style not in {"classic", "handlebar", "chevron"}:
            style = "classic"
        result = _draw_mustache(image_np, landmarks, style, color_bgr, intensity, scale, offset_x, offset_y)

    return {
        "result_image": result,
        "accessory_type": accessory,
        "style": style,
        "hex_color": _normalize_hex_color(hex_color),
        "intensity": intensity,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }
