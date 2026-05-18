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


def _draw_realistic_lashes(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    key: dict,
    intensity: float,
) -> np.ndarray:
    """Draws eyeliner effect along eye contours."""
    result = image_np.copy().astype(np.float32)
    h, w = image_np.shape[:2]
    
    for eye_key in ['left_eye', 'right_eye']:
        eye_points = [landmarks[i] for i in key[eye_key] if i < len(landmarks)]
        if len(eye_points) < 4:
            continue
        
        eye_points = np.array(eye_points, dtype=np.int32)
        
        # Split upper and lower lid
        upper_lid = eye_points[:len(eye_points)//2]
        lower_lid = eye_points[len(eye_points)//2:]
        
        # Draw upper eyeliner - smooth curve
        if len(upper_lid) > 2:
            eyeliner_color = (0, 0, 0)  # Black
            thickness = max(1, int(1 + intensity * 0.8))  # 1-1.8 px
            cv2.polylines(result, [upper_lid], False, eyeliner_color, thickness=thickness)
        
        # Draw lower eyeliner
        if len(lower_lid) > 2:
            thickness = max(1, int(1 + intensity * 0.8))
            cv2.polylines(result, [lower_lid], False, (0, 0, 0), thickness=thickness)
    
    # Apply slight blur to soften eyeliner
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    
    return np.uint8(np.clip(result, 0, 255))


def _apply_eye_color(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    key: dict,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """Applies iris color change."""
    result = image_np.copy().astype(np.float32)
    h, w = image_np.shape[:2]
    
    for eye_key in ['left_eye', 'right_eye']:
        eye_points = [landmarks[i] for i in key[eye_key] if i < len(landmarks)]
        if len(eye_points) < 4:
            continue
        
        eye_points = np.array(eye_points, dtype=np.float32)
        
        # Estimate iris center - approximate from eye landmarks
        center = eye_points.mean(axis=0)
        cx, cy = int(center[0]), int(center[1])
        
        # Iris radius - approximately 8-10 pixels from eye center
        iris_radius = int(8 + intensity * 3)
        
        # Create iris mask using Gaussian
        y_min = max(0, cy - iris_radius * 3)
        y_max = min(h, cy + iris_radius * 3)
        x_min = max(0, cx - iris_radius * 3)
        x_max = min(w, cx + iris_radius * 3)
        
        if y_min < y_max and x_min < x_max:
            yy, xx = np.meshgrid(
                np.arange(y_min, y_max),
                np.arange(x_min, x_max),
                indexing='ij'
            )
            # Gaussian iris mask
            iris_mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * iris_radius**2))
            
            # Blend eye color
            for c in range(3):
                result[y_min:y_max, x_min:x_max, c] = (
                    result[y_min:y_max, x_min:x_max, c] * (1 - iris_mask * intensity * 0.7) +
                    color_bgr[c] * iris_mask * intensity * 0.7
                )
    
    return np.uint8(np.clip(result, 0, 255))


def _apply_teeth_whitening(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    key: dict,
    color_bgr: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """Applies teeth whitening effect to mouth inner area."""
    result = image_np.copy().astype(np.float32)
    h, w = image_np.shape[:2]
    
    # Get mouth inner landmarks
    mouth_inner = [landmarks[i] for i in key.get('mouth_inner', []) if i < len(landmarks)]
    if len(mouth_inner) < 3:
        return np.uint8(np.clip(result, 0, 255))
    
    mouth_inner = np.array(mouth_inner, dtype=np.int32)
    
    # Create mouth inner mask
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(mouth_inner)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Apply Gaussian blur and dilation for smooth blend
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (11, 11), 1.5)
    mask = mask.astype(np.float32) / 255.0
    
    # Blend whitening color - make teeth bright
    blend_strength = intensity * 0.6
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - mask * blend_strength) + color_bgr[c] * mask * blend_strength
    
    return np.uint8(np.clip(result, 0, 255))


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
        outer = [landmarks[i] for i in key['mouth_outer'] if i < len(landmarks)]
        inner = [landmarks[i] for i in key['mouth_inner'] if i < len(landmarks)]
        points = outer + inner
        return _polygon_mask(points, h, w, blur=11, dilate_iter=2)

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
        left_eye = [landmarks[i] for i in key['left_eye'] if i < len(landmarks)]
        right_eye = [landmarks[i] for i in key['right_eye'] if i < len(landmarks)]
        left_mask = _line_mask(h, w, left_eye, thickness=10, blur=7)
        right_mask = _line_mask(h, w, right_eye, thickness=10, blur=7)
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
    if region not in {'lip', 'cheek', 'brow', 'lash', 'eye', 'teeth'}:
        raise ValueError('region must be one of: lip, cheek, brow, lash, eye, teeth')

    key = get_key_landmark_indices()
    
    # For lashes, draw realistic eyelashes
    if region == 'lash':
        result = _draw_realistic_lashes(image_np, landmarks, key, intensity)
        return {
            'result_image': result,
            'region': region,
            'hex_color': '000000',
            'intensity': intensity,
        }
    
    # For eye color
    if region == 'eye':
        color_bgr = _hex_to_bgr(hex_color)
        result = _apply_eye_color(image_np, landmarks, key, color_bgr, intensity)
        return {
            'result_image': result,
            'region': region,
            'hex_color': _normalize_hex_color(hex_color),
            'intensity': intensity,
        }
    
    # For teeth whitening
    if region == 'teeth':
        color_bgr = _hex_to_bgr(hex_color)
        result = _apply_teeth_whitening(image_np, landmarks, key, color_bgr, intensity)
        return {
            'result_image': result,
            'region': region,
            'hex_color': _normalize_hex_color(hex_color),
            'intensity': intensity,
        }

    color_bgr = _hex_to_bgr(hex_color)
    mask = _build_region_mask(image_np, landmarks, region)

    clone_mode = cv2.MIXED_CLONE if region in {'lip', 'cheek'} else cv2.NORMAL_CLONE
    reinforce = 1.0 if region == 'lip' else (0.22 if region == 'cheek' else 0.14)
    result = _apply_tint(image_np, mask, color_bgr, intensity, clone_mode=clone_mode, reinforce=reinforce)

    return {
        'result_image': result,
        'region': region,
        'hex_color': _normalize_hex_color(hex_color),
        'intensity': intensity,
    }