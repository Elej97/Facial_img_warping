import cv2
import numpy as np

from modules.landmark import get_key_landmark_indices
from modules.region_map import get_region_indices


_REGION_ALIASES = {
    'lip': 'lip',
    'cheek': 'cheek',
    'brow': 'brow',
    'lash': 'lash',
}


def _normalize_hex_color(hex_color: str) -> str:
    value = (hex_color or '').strip().lstrip('#')
    if len(value) == 3:
        value = ''.join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError('hex_color must be a 3 or 6 digit HEX value')
    int(value, 16)
    return value.upper()


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    value = _normalize_hex_color(hex_color)
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return blue, green, red


def _polygon_mask(points: list[tuple[int, int]], h: int, w: int, blur: int = 21, dilate_iter: int = 1) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask.astype(np.float32) / 255.0

    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=dilate_iter)
    blur = blur | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def _ellipse_mask(
    h: int,
    w: int,
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0.0,
    blur: int = 41,
    strength: float = 1.0,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, int(255 * np.clip(strength, 0.0, 1.0)), -1)
    blur = blur | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def _line_mask(
    h: int,
    w: int,
    points: list[tuple[int, int]],
    thickness: int,
    blur: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 2:
        return mask.astype(np.float32) / 255.0

    pts = np.array(points, dtype=np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness=max(1, thickness))
    blur = blur | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def _mask_center(mask_u8: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(mask_u8)
    if moments['m00'] <= 1e-6:
        return None
    return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])


def _apply_tint(
    image_np: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    intensity: float,
    clone_mode: int,
    reinforce: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if float(mask.max()) <= 1e-6:
        return image_np.copy()

    alpha = float(np.clip(0.18 + intensity * 0.54, 0.0, 0.88))
    color_layer = np.full_like(image_np, color_bgr, dtype=np.uint8)
    tinted_source = cv2.addWeighted(image_np.astype(np.float32), 1.0 - alpha, color_layer.astype(np.float32), alpha, 0)
    tinted_source = np.clip(tinted_source, 0, 255).astype(np.uint8)

    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    center = _mask_center(mask_u8) or (w // 2, h // 2)

    try:
        cloned = cv2.seamlessClone(tinted_source, image_np, mask_u8, center, clone_mode)
    except Exception:
        cloned = image_np.copy()

    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5.5, sigmaY=5.5)
    soft_mask = np.expand_dims(np.clip(soft_mask, 0.0, 1.0), axis=2)
    result = cloned.astype(np.float32) * (1.0 - reinforce * soft_mask) + tinted_source.astype(np.float32) * (reinforce * soft_mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def _split_cheek_points(landmarks: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    key = get_region_indices()
    face_ids = [
        *key.get('upper_cheeks', []),
        *key.get('cheeks_outer', []),
        *key.get('jawline_outer', []),
    ]
    safe = [landmarks[i] for i in face_ids if i < len(landmarks)]
    if not safe:
        return [], []

    center_x = float(np.mean([pt[0] for pt in safe]))
    left = [pt for pt in safe if pt[0] <= center_x]
    right = [pt for pt in safe if pt[0] > center_x]
    return left, right


def _build_region_mask(image_np: np.ndarray, landmarks: list[tuple[int, int]], region: str) -> np.ndarray:
    h, w = image_np.shape[:2]
    region = _REGION_ALIASES.get(region.lower().strip(), region.lower().strip())
    key = get_key_landmark_indices()
    regions = get_region_indices()

    if region == 'lip':
        points = [landmarks[i] for i in key['mouth_outer'] if i < len(landmarks)]
        return _polygon_mask(points, h, w, blur=17, dilate_iter=1)

    if region == 'cheek':
        left_cheek, right_cheek = _split_cheek_points(landmarks)
        mask = np.zeros((h, w), dtype=np.float32)

        for points in (left_cheek, right_cheek):
            if len(points) < 3:
                continue
            pts = np.array(points, dtype=np.float32)
            center = np.mean(pts, axis=0)
            width_span = float(max(1.0, np.max(pts[:, 0]) - np.min(pts[:, 0])))
            height_span = float(max(1.0, np.max(pts[:, 1]) - np.min(pts[:, 1])))
            axes = (
                int(np.clip(width_span * 0.95, w * 0.09, w * 0.18)),
                int(np.clip(height_span * 1.45, h * 0.08, h * 0.18)),
            )
            cheek_mask = _ellipse_mask(h, w, (int(center[0]), int(center[1])), axes, blur=61, strength=0.95)
            mask = np.clip(mask + cheek_mask, 0.0, 1.0)

        return cv2.GaussianBlur(mask, (0, 0), sigmaX=13.0, sigmaY=13.0)

    if region == 'brow':
        left = [landmarks[i] for i in regions.get('eyebrow_left_arc', []) if i < len(landmarks)]
        right = [landmarks[i] for i in regions.get('eyebrow_right_arc', []) if i < len(landmarks)]
        left_mask = _line_mask(h, w, left, thickness=12, blur=15)
        right_mask = _line_mask(h, w, right, thickness=12, blur=15)
        return np.clip(left_mask + right_mask, 0.0, 1.0)

    if region == 'lash':
        left_eye = [landmarks[i] for i in key['left_eye'][:9] if i < len(landmarks)]
        right_eye = [landmarks[i] for i in key['right_eye'][:9] if i < len(landmarks)]
        left_mask = _line_mask(h, w, left_eye, thickness=6, blur=11)
        right_mask = _line_mask(h, w, right_eye, thickness=6, blur=11)
        return np.clip(left_mask + right_mask, 0.0, 1.0)

    raise ValueError('Unsupported makeup region. Use lip, cheek, brow, or lash.')


def apply_makeup(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    region: str,
    hex_color: str,
    intensity: float = 0.6,
) -> dict:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    region = (region or '').lower().strip()
    if region not in {'lip', 'cheek', 'brow', 'lash'}:
        raise ValueError('region must be one of: lip, cheek, brow, lash')

    color_bgr = _hex_to_bgr(hex_color)
    mask = _build_region_mask(image_np, landmarks, region)

    clone_mode = cv2.MIXED_CLONE if region in {'lip', 'cheek'} else cv2.NORMAL_CLONE
    reinforce = 0.22 if region in {'lip', 'cheek'} else 0.14
    result = _apply_tint(image_np, mask, color_bgr, intensity, clone_mode=clone_mode, reinforce=reinforce)

    if region == 'lash':
        dark_boost = np.clip(0.35 + intensity * 0.45, 0.0, 0.82)
        overlay = cv2.addWeighted(
            result.astype(np.float32),
            1.0 - dark_boost,
            np.full_like(result, color_bgr, dtype=np.uint8).astype(np.float32),
            dark_boost,
            0,
        )
        lash_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.8, sigmaY=2.8)
        lash_mask = np.expand_dims(np.clip(lash_mask, 0.0, 1.0), axis=2)
        result = np.clip(result.astype(np.float32) * (1.0 - lash_mask) + overlay * lash_mask, 0, 255).astype(np.uint8)

    return {
        'result_image': result,
        'region': region,
        'hex_color': _normalize_hex_color(hex_color),
        'intensity': intensity,
    }