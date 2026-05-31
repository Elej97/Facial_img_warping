import cv2
import numpy as np
from scipy.spatial import Delaunay

from modules.landmark import get_key_landmark_indices


def _clip_points(points: list, w: int, h: int) -> list:
    clipped = []
    for x, y in points:
        clipped.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    return clipped


def _blend_output(src: np.ndarray, warped: np.ndarray, intensity: float, min_alpha: float = 0.35, max_alpha: float = 0.85) -> np.ndarray:
    alpha = float(np.clip(min_alpha + intensity * (max_alpha - min_alpha), 0.0, 1.0))
    return cv2.addWeighted(src.astype(np.float32), 1.0 - alpha, warped.astype(np.float32), alpha, 0).astype(np.uint8)


def _add_gaussian_flow(flow_x: np.ndarray, flow_y: np.ndarray, center: tuple, shift: tuple, sigma: float) -> None:
    h, w = flow_x.shape
    cx, cy = float(center[0]), float(center[1])
    sigma = max(1.0, float(sigma))
    radius = int(3.0 * sigma)

    x1 = max(0, int(cx) - radius)
    y1 = max(0, int(cy) - radius)
    x2 = min(w, int(cx) + radius + 1)
    y2 = min(h, int(cy) + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    weight = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

    flow_x[y1:y2, x1:x2] += float(shift[0]) * weight
    flow_y[y1:y2, x1:x2] += float(shift[1]) * weight


def _add_brow_lift_flow(flow_y: np.ndarray, center: tuple, shift_y: float, sigma_x: float, sigma_y: float, lower_bleed_px: float) -> None:
    h, w = flow_y.shape
    cx, cy = float(center[0]), float(center[1])
    sigma_x = max(1.0, float(sigma_x))
    sigma_y = max(1.0, float(sigma_y))
    radius_x = int(3.0 * sigma_x)
    radius_y = int(3.0 * sigma_y)

    x1 = max(0, int(cx) - radius_x)
    y1 = max(0, int(cy) - radius_y)
    x2 = min(w, int(cx) + radius_x + 1)
    y2 = min(h, int(cy) + radius_y + 1)
    if x2 <= x1 or y2 <= y1:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    weight = np.exp(-(((xx - cx) ** 2) / (sigma_x * sigma_x) + ((yy - cy) ** 2) / (sigma_y * sigma_y))).astype(np.float32)
    weight[yy > cy + lower_bleed_px] = 0.0
    flow_y[y1:y2, x1:x2] += float(shift_y) * weight


def _apply_flow_warp(image_np: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Backward mapping for cv2.remap: sample source from the inverse of desired displacement.
    map_x = np.clip(grid_x - flow_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(grid_y - flow_y, 0, h - 1).astype(np.float32)

    return cv2.remap(
        image_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _translate_brow_texture(
    image_np: np.ndarray,
    landmarks: list,
    brow_groups: list[list[int]],
    eye_groups: list[list[int]],
    intensity: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    scale = float(min(h, w))
    intensity_eff = float(np.clip(intensity, 0.0, 1.0))

    candidate_mask = np.zeros((h, w), dtype=np.uint8)
    thickness = max(4, int(scale * 0.020))

    for brow_ids in brow_groups:
        points = np.array([landmarks[i] for i in brow_ids if i < len(landmarks)], dtype=np.int32)
        if len(points) < 2:
            continue
        cv2.polylines(candidate_mask, [points], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
        for point in points:
            cv2.circle(candidate_mask, tuple(point), max(2, thickness // 2), 255, -1, lineType=cv2.LINE_AA)

    if int(candidate_mask.max()) == 0:
        return image_np.copy()

    for eye_ids in eye_groups:
        eye_points = np.array([landmarks[i] for i in eye_ids if i < len(landmarks)], dtype=np.int32)
        if len(eye_points) >= 3:
            eye_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(eye_mask, cv2.convexHull(eye_points), 255)
            eye_mask = cv2.dilate(eye_mask, np.ones((5, 5), np.uint8), iterations=1)
            candidate_mask[eye_mask > 0] = 0

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    candidate_pixels = gray[candidate_mask > 0]
    if candidate_pixels.size == 0:
        return image_np.copy()

    threshold = max(35.0, min(165.0, float(np.percentile(candidate_pixels, 45)) - 4.0))
    source_mask = np.zeros((h, w), dtype=np.uint8)
    source_mask[(candidate_mask > 0) & (gray <= threshold)] = 255
    source_mask = cv2.morphologyEx(source_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    source_mask = cv2.dilate(source_mask, np.ones((3, 3), np.uint8), iterations=1)
    source_mask[candidate_mask == 0] = 0

    if int(source_mask.max()) == 0:
        return image_np.copy()

    shift_y = -scale * (0.018 + 0.018 * intensity_eff)
    transform = np.float32([[1, 0, 0], [0, 1, shift_y]])

    erase_mask = cv2.dilate(source_mask, np.ones((3, 3), np.uint8), iterations=1)
    base = cv2.inpaint(image_np, erase_mask, max(1, int(scale * 0.004)), cv2.INPAINT_TELEA)
    shifted_image = cv2.warpAffine(image_np, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    shifted_mask = cv2.warpAffine(source_mask, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    alpha = cv2.GaussianBlur(shifted_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=max(1.0, scale * 0.004))
    alpha = np.clip(alpha * (0.92 + 0.08 * intensity_eff), 0.0, 1.0)
    alpha3 = np.stack([alpha, alpha, alpha], axis=2)

    out = base.astype(np.float32) * (1.0 - alpha3) + shifted_image.astype(np.float32) * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)


def _region_mask(points: list, h: int, w: int, blur: int = 31, dilate_iter: int = 2) -> np.ndarray:
    if not points:
        return np.zeros((h, w), dtype=np.float32)
    poly = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(poly)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=dilate_iter)
    blur = blur | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return (mask.astype(np.float32) / 255.0)


def apply_delaunay_warp(src_image: np.ndarray, src_pts: list, dst_pts: list) -> np.ndarray:
    h, w = src_image.shape[:2]
    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)

    if np.allclose(src_arr, dst_arr):
        return src_image.copy()

    # Boundary anchors so regions outside the face hull don't deform
    boundary = np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2], [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
    ], dtype=np.float32)

    src_full = np.vstack([src_arr, boundary])
    dst_full = np.vstack([dst_arr, boundary])

    tri = Delaunay(src_full)
    output = src_image.copy()

    for simplex in tri.simplices:
        src_tri = src_full[simplex].astype(np.float32)
        dst_tri = dst_full[simplex].astype(np.float32)

        # Bounding rect of dst triangle (output region)
        rx, ry, rw, rh = cv2.boundingRect(dst_tri)
        rx = max(0, rx); ry = max(0, ry)
        rx2 = min(w, rx + rw); ry2 = min(h, ry + rh)
        rw = rx2 - rx; rh = ry2 - ry
        if rw <= 0 or rh <= 0:
            continue

        # Bounding rect of src triangle (source region)
        sx, sy, srw, srh = cv2.boundingRect(src_tri)
        sx = max(0, sx); sy = max(0, sy)
        sx2 = min(w, sx + srw); sy2 = min(h, sy + srh)
        srw = sx2 - sx; srh = sy2 - sy
        if srw <= 0 or srh <= 0:
            continue

        src_tri_local = src_tri - np.array([sx, sy], dtype=np.float32)
        dst_tri_local = dst_tri - np.array([rx, ry], dtype=np.float32)

        M = cv2.getAffineTransform(src_tri_local, dst_tri_local)
        src_roi = src_image[sy:sy2, sx:sx2]
        warped_patch = cv2.warpAffine(
            src_roi, M, (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_local.astype(np.int32), 255)

        roi = output[ry:ry2, rx:rx2]
        mask_f = (mask.astype(np.float32) / 255.0)
        mask_f = cv2.GaussianBlur(mask_f, (5, 5), 0)
        mask_f = np.expand_dims(mask_f, axis=2)

        roi_f = roi.astype(np.float32)
        warped_f = warped_patch.astype(np.float32)
        blended = roi_f * (1.0 - mask_f) + warped_f * mask_f
        output[ry:ry2, rx:rx2] = np.clip(blended, 0, 255).astype(np.uint8)

    return output


def simulate_smile(image_np: np.ndarray, landmarks: list, intensity: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    intensity_eff = float(np.clip(intensity ** 0.85, 0.0, 1.0))
    scale = float(min(h, w))
    
    # Use flow-based warping for smoother, more natural results
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)
    
    # Ağız köşeleri - GÜÇLÜ dışa ve yukarı hareket (ana gülüş noktası)
    mouth_corners = [61, 291]  # Left & right mouth corners
    corner_shift_x = scale * 0.045 * intensity_eff  # 0.032 → 0.045 (çok daha güçlü)
    corner_shift_y = -scale * 0.035 * intensity_eff  # -0.024 → -0.035 (çok daha güçlü)
    corner_sigma = scale * 0.085  # Biraz daha büyük radius
    
    if mouth_corners[0] < len(landmarks):
        x, y = landmarks[mouth_corners[0]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (-corner_shift_x, corner_shift_y), corner_sigma)
    if mouth_corners[1] < len(landmarks):
        x, y = landmarks[mouth_corners[1]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (corner_shift_x, corner_shift_y), corner_sigma)
    
    # Üst dudak - yeterince yukarı
    upper_lip_points = [0, 37, 267]
    upper_shift_y = -scale * 0.014 * intensity_eff
    upper_sigma = scale * 0.055
    
    for i in upper_lip_points:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, upper_shift_y), upper_sigma)
    
    # Alt dudak - biraz aşağı
    lower_lip_points = [14, 84, 314]
    lower_shift_y = scale * 0.012 * intensity_eff
    lower_sigma = scale * 0.055
    
    for i in lower_lip_points:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, lower_shift_y), lower_sigma)
    
    # Ağız kenarları - minimal hareket
    mouth_edge = [78, 308]  
    edge_shift_x = scale * 0.012 * intensity_eff
    edge_sigma = scale * 0.050
    
    if mouth_edge[0] < len(landmarks):
        x, y = landmarks[mouth_edge[0]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (-edge_shift_x, 0.0), edge_sigma)
    if mouth_edge[1] < len(landmarks):
        x, y = landmarks[mouth_edge[1]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (edge_shift_x, 0.0), edge_sigma)
    
    # Gözler - hafif yukarı hareket + üst kapak biraz aşağı (kısılma efekti)
    eye_points = [33, 133, 362, 263]
    eye_shift_y = -scale * 0.006 * intensity_eff  # Reduced
    eye_sigma = scale * 0.070
    
    for i in eye_points:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, eye_shift_y), eye_sigma)
    
    # Üst göz kapağı - aşağı hareket (gözleri kısma efekti)
    upper_lid_points = [159, 160, 161, 386, 385, 384]
    lid_shift_y = scale * 0.004 * intensity_eff
    lid_sigma = scale * 0.040
    
    for i in upper_lid_points:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, lid_shift_y), lid_sigma)
    
    # Kaş bölgesi - hafif aşağı hareket
    brow_points = [46, 52, 276, 282]
    brow_shift_y = scale * 0.005 * intensity_eff
    brow_sigma = scale * 0.085
    
    for i in brow_points:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, brow_shift_y), brow_sigma)
    
    # Yanaklar - biraz daha belirgin yukarı ve dışa hareket
    cheek_points = [205, 425]
    cheek_shift_x = scale * 0.015 * intensity_eff  # 0.012 → 0.015
    cheek_shift_y = -scale * 0.015 * intensity_eff  # -0.012 → -0.015
    cheek_sigma = scale * 0.090
    
    if cheek_points[0] < len(landmarks):
        x, y = landmarks[cheek_points[0]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (-cheek_shift_x, cheek_shift_y), cheek_sigma)
    if cheek_points[1] < len(landmarks):
        x, y = landmarks[cheek_points[1]]
        _add_gaussian_flow(flow_x, flow_y, (x, y), (cheek_shift_x, cheek_shift_y), cheek_sigma)
    
    warped = _apply_flow_warp(image_np, flow_x, flow_y)
    # Natural blend - daha fazla original görüntü kalıyor
    return _blend_output(image_np, warped, intensity_eff, min_alpha=0.50, max_alpha=0.85)


def raise_eyebrows(image_np: np.ndarray, landmarks: list, intensity: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    intensity_eff = float(np.clip(intensity, 0.0, 1.0))
    scale = float(min(h, w))
    key = get_key_landmark_indices()

    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    # SADECE kaş noktalarını taşı - sigma arttırıldı
    # Göz yakın noktaları hariç (105, 107, 334, 336 out)
    brow_sigma = scale * 0.016
    brow_ids_full = [i for i in key["left_brow"] + key["right_brow"] if i < len(landmarks)]
    brow_ids = [i for i in brow_ids_full if i in {70, 63, 105, 66, 107, 300, 293, 334, 296, 336}]
    lift_strength = {
        70: 0.020, 63: 0.016, 105: 0.016, 66: 0.014, 107: 0.013,
        300: 0.020, 293: 0.016, 334: 0.016, 296: 0.014, 336: 0.013,
    }
    for i in brow_ids:
        if i < len(landmarks):
            _add_brow_lift_flow(
                flow_y,
                landmarks[i],
                -scale * lift_strength.get(i, 0.011) * intensity_eff,
                sigma_x=scale * 0.026,
                sigma_y=brow_sigma,
                lower_bleed_px=scale * 0.020,
            )

    # Forehead hafif yukarı - kaşın doğal kalması için
    forehead = [10, 67, 109, 338, 297]
    for i in forehead:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, -scale * 0.002 * intensity_eff), scale * 0.026)
    
    # Zone mask: Kaş + forehead - göz altı dışarıda
    left_zone = [landmarks[i] for i in brow_ids if landmarks[i][0] < w // 2]
    right_zone = [landmarks[i] for i in brow_ids if landmarks[i][0] >= w // 2]
    
    for i in [10, 67, 109]:
        if i < len(landmarks):
            left_zone.append(landmarks[i])
    for i in [10, 338, 297]:
        if i < len(landmarks):
            right_zone.append(landmarks[i])

    for i in [159, 160, 161]:
        if i < len(landmarks):
            left_zone.append(landmarks[i])
    for i in [386, 385, 384]:
        if i < len(landmarks):
            right_zone.append(landmarks[i])

    mask_left = _region_mask(left_zone, h, w, blur=11, dilate_iter=1)
    mask_right = _region_mask(right_zone, h, w, blur=11, dilate_iter=1)
    zone_mask = np.clip(mask_left + mask_right, 0.0, 1.0)
    for eye_ids in [
        [33, 133, 145, 159, 160, 161, 163, 144, 153, 154, 155, 158, 157],
        [362, 263, 374, 386, 385, 384, 390, 373, 380, 381, 382, 387, 388],
    ]:
        eye_pts = [landmarks[i] for i in eye_ids if i < len(landmarks)]
        if len(eye_pts) >= 3:
            zone_mask = np.clip(zone_mask - _region_mask(eye_pts, h, w, blur=15, dilate_iter=1), 0.0, 1.0)
    flow_x *= zone_mask
    flow_y *= zone_mask

    warped = _apply_flow_warp(image_np, flow_x, flow_y)
    return _blend_output(image_np, warped, intensity_eff, min_alpha=0.55, max_alpha=0.82)


def widen_lips(image_np: np.ndarray, landmarks: list, intensity: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    intensity_eff = float(np.clip(intensity ** 0.85, 0.0, 1.0))
    scale = float(min(h, w))
    key = get_key_landmark_indices()

    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    mouth_outer = [i for i in key["mouth_outer"] if i < len(landmarks)]
    mouth_inner = [i for i in key["mouth_inner"] if i < len(landmarks)]
    if not mouth_outer:
        return image_np.copy()

    cx = float(np.mean([landmarks[i][0] for i in mouth_outer]))

    sigma_outer = scale * 0.045
    sigma_inner = scale * 0.040
    base_shift = scale * 0.024 * intensity_eff

    for i in mouth_outer:
        x, y = landmarks[i]
        dir_x = np.sign(x - cx)
        _add_gaussian_flow(flow_x, flow_y, (x, y), (dir_x * base_shift, 0.0), sigma_outer)

    for i in mouth_inner:
        x, y = landmarks[i]
        dir_x = np.sign(x - cx)
        _add_gaussian_flow(flow_x, flow_y, (x, y), (dir_x * base_shift * 0.35, 0.0), sigma_inner)

    # Lift upper lip and lower lower-lip slightly to keep mouth natural while widening.
    upper_lip = [13, 0, 37, 267]
    lower_lip = [14, 17, 84, 314]
    for i in upper_lip:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, -scale * 0.004 * intensity_eff), scale * 0.032)
    for i in lower_lip:
        if i < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[i], (0.0, scale * 0.004 * intensity_eff), scale * 0.032)

    mouth_zone = [landmarks[i] for i in mouth_outer]
    mouth_mask = _region_mask(mouth_zone, h, w, blur=35, dilate_iter=2)
    flow_x *= mouth_mask
    flow_y *= mouth_mask

    warped = _apply_flow_warp(image_np, flow_x, flow_y)
    return _blend_output(image_np, warped, intensity_eff, min_alpha=0.36, max_alpha=0.68)


def slim_face(image_np: np.ndarray, landmarks: list, intensity: float = 0.5) -> np.ndarray:
    h, w = image_np.shape[:2]
    intensity_eff = float(np.clip(intensity ** 0.90, 0.0, 1.0))
    inward = min(h, w) * 0.14 * intensity_eff
    nose_tip = landmarks[1]
    dst = list(landmarks)

    jaw_indices = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    cheek_indices = [234, 454, 132, 361, 227, 447]

    for i in jaw_indices:
        if i < len(dst):
            x, y = dst[i]
            # Direction toward nose tip
            dx = nose_tip[0] - x
            dy = nose_tip[1] - y
            dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
            dst[i] = (int(x + dx / dist * inward), int(y + dy / dist * inward * 0.30))

    for i in cheek_indices:
        if i < len(dst):
            x, y = dst[i]
            dx = nose_tip[0] - x
            dy = nose_tip[1] - y
            dist = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
            dst[i] = (int(x + dx / dist * inward * 0.65), int(y + dy / dist * inward * 0.22))

    dst = _clip_points(dst, w, h)
    warped = apply_delaunay_warp(image_np, landmarks, dst)
    return _blend_output(image_np, warped, intensity_eff, min_alpha=0.48, max_alpha=0.86)
