import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from modules.aging_module import apply_pro_aging, apply_pro_deaging
from modules.landmark import get_key_landmark_indices
from modules.region_map import REGION_INDICES


_DEF_KERNEL = "thin_plate_spline"


def _clip_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    out = points.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def _face_mask(h: int, w: int, face_points: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(face_points) < 3:
        return mask.astype(np.float32)

    hull = cv2.convexHull(face_points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return mask.astype(np.float32) / 255.0


def _build_anchor_points(w: int, h: int) -> np.ndarray:
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2], [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
    ], dtype=np.float32)


def _rbf_warp(
    image_np: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    face_points: np.ndarray,
    smoothing: float = 2.5,
    kernel: str = _DEF_KERNEL,
    epsilon: float | None = None,
    neighbors: int | None = None,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    src_points = _clip_points(src_points.astype(np.float32), w, h)
    dst_points = _clip_points(dst_points.astype(np.float32), w, h)

    anchors = _build_anchor_points(w, h)
    src_all = np.vstack([src_points, anchors])
    dst_all = np.vstack([dst_points, anchors])

    displacement = dst_all - src_all

    kwargs: dict = {"kernel": kernel, "smoothing": smoothing}
    if epsilon is not None:
        kwargs["epsilon"] = float(max(1e-4, epsilon))
    if neighbors is not None:
        kwargs["neighbors"] = int(max(6, neighbors))

    rbf_dx = RBFInterpolator(src_all, displacement[:, 0], **kwargs)
    rbf_dy = RBFInterpolator(src_all, displacement[:, 1], **kwargs)

    # PERF: the RBF flow is smooth, so evaluate it on a COARSE grid (longest side <= 256)
    # and bilinearly upsample, instead of querying every h*w pixel. Querying all pixels with
    # neighbors=128 was the slowness (slim_face etc.); this is ~10-40x faster, no visible change.
    s = 256.0 / float(max(h, w))
    if s < 1.0:
        gw, gh = max(2, int(round(w * s))), max(2, int(round(h * s)))
        mx, my = np.meshgrid(np.linspace(0, w - 1, gw, dtype=np.float32),
                             np.linspace(0, h - 1, gh, dtype=np.float32))
        query = np.stack([mx.ravel(), my.ravel()], axis=1).astype(np.float32)
        flow_x = cv2.resize(rbf_dx(query).reshape(gh, gw).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(rbf_dy(query).reshape(gh, gw).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        ys, xs = np.mgrid[0:h, 0:w]
        query = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        flow_x = rbf_dx(query).reshape(h, w).astype(np.float32)
        flow_y = rbf_dy(query).reshape(h, w).astype(np.float32)

    # Keep deformation localized in face region for natural transitions.
    mask = _face_mask(h, w, face_points)
    flow_x *= mask
    flow_y *= mask

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x - flow_x, 0, w - 1)
    map_y = np.clip(grid_y - flow_y, 0, h - 1)

    return cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _blend(src: np.ndarray, warped: np.ndarray, intensity: float, low: float, high: float) -> np.ndarray:
    alpha = float(np.clip(low + intensity * (high - low), 0.0, 1.0))
    out = cv2.addWeighted(src.astype(np.float32), 1.0 - alpha, warped.astype(np.float32), alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _texture_preserve_blend(
    src: np.ndarray,
    warped: np.ndarray,
    intensity: float,
    low: float,
    high: float,
    detail_weight: float = 0.16,
) -> np.ndarray:
    blended = _blend(src, warped, intensity, low=low, high=high).astype(np.float32)
    src_f = src.astype(np.float32)

    # Keep high-frequency skin detail so deformation looks less smeared.
    base = cv2.GaussianBlur(src_f, (0, 0), sigmaX=1.05, sigmaY=1.05)
    detail = src_f - base
    gain = float(np.clip(detail_weight + 0.06 * intensity, 0.08, 0.28))
    out = blended + gain * detail
    return np.clip(out, 0, 255).astype(np.uint8)


def _polygon_mask(points: list[tuple[int, int]], h: int, w: int, blur: int = 31, dilate_iter: int = 1) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask.astype(np.float32)
    poly = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(poly)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=dilate_iter)
    blur = blur | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def _add_brow_lift_flow(
    flow_y: np.ndarray,
    center: tuple[int, int],
    shift_y: float,
    sigma_x: float,
    sigma_y: float,
    lower_bleed_px: float,
) -> None:
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


def _add_local_flow(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    center: tuple[int, int],
    shift: tuple[float, float],
    sigma_x: float,
    sigma_y: float,
) -> None:
    h, w = flow_x.shape
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
    flow_x[y1:y2, x1:x2] += float(shift[0]) * weight
    flow_y[y1:y2, x1:x2] += float(shift[1]) * weight


def _flow_warp(image_np: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x - flow_x, 0, w - 1)
    map_y = np.clip(grid_y - flow_y, 0, h - 1)
    return cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _flow_magnitude_mask(flow_x: np.ndarray, flow_y: np.ndarray, blur: int = 17) -> np.ndarray:
    magnitude = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    max_value = float(np.max(magnitude))
    if max_value <= 1e-6:
        return np.zeros_like(flow_x, dtype=np.float32)
    mask = np.clip(magnitude / max_value, 0.0, 1.0).astype(np.float32)
    blur = blur | 1
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def _translate_brow_texture(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
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


def _collect_points(landmarks: list[tuple[int, int]], indices: list[int]) -> list[tuple[int, int]]:
    return [landmarks[i] for i in indices if i < len(landmarks)]


def _fft_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return fshift, mag


def _spectrum_vis(mag: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _spectral_energy_components(gray: np.ndarray) -> dict:
    h, w = gray.shape
    fshift, _ = _fft_spectrum(gray)
    power = np.abs(fshift) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = min(h, w) * 0.18
    low = float(np.sum(power[dist <= r]))
    high = float(np.sum(power[dist > r]))
    total = low + high

    return {
        "total": total,
        "low": low,
        "high": high,
        "hf_lf_ratio": high / max(low, 1e-6),
    }


def _compute_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-6)
    return float(np.mean(ssim_map))


def _compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    org_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    out_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    diff = org_gray.astype(np.float32) - out_gray.astype(np.float32)
    mse = float(np.mean(diff * diff))
    psnr = float(20.0 * np.log10(255.0 / np.sqrt(max(mse, 1e-9))))
    ssim = _compute_ssim(org_gray, out_gray)

    energy_before = _spectral_energy_components(org_gray)
    energy_after = _spectral_energy_components(out_gray)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "total_spectral_energy_before": energy_before["total"],
        "total_spectral_energy_after": energy_after["total"],
        "total_spectral_energy_delta": energy_after["total"] - energy_before["total"],
        "low_frequency_energy_before": energy_before["low"],
        "low_frequency_energy_after": energy_after["low"],
        "high_frequency_energy_before": energy_before["high"],
        "high_frequency_energy_after": energy_after["high"],
        "hf_lf_ratio_before": energy_before["hf_lf_ratio"],
        "hf_lf_ratio_after": energy_after["hf_lf_ratio"],
        "hf_lf_ratio_delta": energy_after["hf_lf_ratio"] - energy_before["hf_lf_ratio"],
    }


def _procedural_wrinkle_texture(h: int, w: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)

    tex_h = cv2.GaussianBlur(noise, (0, 0), sigmaX=6.0, sigmaY=1.1)
    tex_d = cv2.GaussianBlur(noise, (0, 0), sigmaX=3.2, sigmaY=2.2)
    texture = 0.68 * tex_h + 0.32 * tex_d
    texture = cv2.normalize(np.abs(texture), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return texture


def _region_ids(name: str) -> list[int]:
    return REGION_INDICES.get(name, [])


class ProWarpManager:
    def __init__(self, kernel: str = _DEF_KERNEL):
        self.kernel = kernel

    @staticmethod
    def _safe_ids(indices: list[int], n: int) -> list[int]:
        return [i for i in indices if i < n]

    @staticmethod
    def _face_points(landmarks: list[tuple[int, int]]) -> np.ndarray:
        key = get_key_landmark_indices()
        jaw_ids = [i for i in key["jaw"] if i < len(landmarks)]
        if jaw_ids:
            return np.array([landmarks[i] for i in jaw_ids], dtype=np.float32)
        return np.array(landmarks, dtype=np.float32)

    def _rbf_warp(
        self,
        image_np: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        face_points: np.ndarray,
        smooth: float,
        kernel: str | None = None,
        epsilon: float | None = None,
        neighbors: int | None = None,
    ) -> np.ndarray:
        return _rbf_warp(
            image_np,
            src_points,
            dst_points,
            face_points,
            smoothing=float(np.clip(smooth, 0.8, 10.0)),
            kernel=kernel or self.kernel,
            epsilon=epsilon,
            neighbors=neighbors,
        )

    def plump_lips(self, image_np: np.ndarray, landmarks: list[tuple[int, int]], intensity: float = 0.6, smooth: float = 2.0) -> np.ndarray:
        h, w = image_np.shape[:2]
        n = len(landmarks)

        # Basic verification of landmark count
        if n < 468:
            return image_np.copy()

        print("[DEBUG] Moving region: lip_plump — RBF dynamic dual lip warp (Enhanced Natural & Realistic).")

        # Outer boundaries of upper and lower lips
        upper_outer = {185, 40, 39, 37, 0, 267, 269, 270, 409}
        lower_outer = {146, 91, 181, 84, 17, 314, 405, 321, 375}

        # Inner boundaries of upper and lower lips plus mouth corners.
        # Keeping these locked preserves the natural mouth line and closed seal perfectly.
        upper_inner = {191, 80, 81, 82, 13, 312, 311, 310, 415}
        lower_inner = {324, 318, 402, 317, 14, 87, 178, 88, 95}
        corner_ids = {61, 291, 78, 308}

        lip_all_ids = self._safe_ids(list(upper_outer) + list(lower_outer) + list(upper_inner) + list(lower_inner) + list(corner_ids), n)
        
        # Center of lips is computed from the inner lip line
        center_ids = [i for i in [13, 14, 0, 17] if i < n]
        center = np.mean([landmarks[i] for i in center_ids], axis=0)
        cx, cy = float(center[0]), float(center[1])

        src_list = []
        dst_list = []

        # We will dynamically generate zero-displacement outer boundary anchors to confine
        # the warp precisely to the lips, preventing any stretching of the philtrum or chin.
        dynamic_anchors = []

        # Subtle, natural scaling factor: moves landmarks vertically by at most 22% of the lip height.
        # This keeps the plumping extremely realistic and fully prevents distorted fish-face puckering.
        factor = 0.22

        for lm_id in lip_all_ids:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))

                # Corner landmarks and inner mouth contact lines are pinned to 0 displacement
                if lm_id in corner_ids or lm_id in upper_inner or lm_id in lower_inner:
                    dst_list.append((sx, sy))
                    continue

                vx, vy = sx - cx, sy - cy

                # Vertical height-proportional displacement: scales displacement directly with the 
                # local thickness of the lip. Naturally tapers to 0 at the mouth corners.
                dy = (sy - cy) * factor * intensity

                dst_list.append((sx, sy + dy))

                # Generate a dynamic outer anchor point further away from the mouth center (e.g. 1.90x distance)
                # to create a very soft, smooth transition zone on the surrounding skin.
                ax = cx + 1.90 * vx
                ay = cy + 1.90 * vy
                dynamic_anchors.append((ax, ay))

        # Add the dynamic outer anchors with zero displacement
        for ax, ay in dynamic_anchors:
            src_list.append((ax, ay))
            dst_list.append((ax, ay))

        # Standard face-wide anchors to stabilize general face structure
        anchor_ids = self._safe_ids(
            # Nose bridge & tip
            [168, 6, 197, 195, 1, 2, 94] +
            # Inner/Lower eyes
            [33, 133, 362, 263] +
            # Jawline / chin to keep jaw shape stable
            [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397] +
            # Ear side anchors
            REGION_INDICES["ear_side_anchors"],
            n
        )

        lip_set = set(lip_all_ids)
        anchor_ids = list(set(anchor_ids) - lip_set)

        for lm_id in anchor_ids:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy))

        src_arr = np.array(src_list, dtype=np.float32)
        dst_arr = np.array(dst_list, dtype=np.float32)
        face_points = self._face_points(landmarks)

        # Apply smooth thin-plate spline global warp
        warped = self._rbf_warp(image_np, src_arr, dst_arr, face_points, smooth=smooth)

        # Construct a blend zone mask covering the lips and their dynamic bounds with a soft blur
        lip_pts = np.array([landmarks[i] for i in lip_all_ids], dtype=np.float32)
        zone = _polygon_mask([(int(p[0]), int(p[1])) for p in lip_pts], h, w, blur=25, dilate_iter=3)
        zone3 = np.stack([zone, zone, zone], axis=2)

        # Composite the warped lips onto the original image
        out_f = image_np.astype(np.float32) * (1.0 - zone3) + warped.astype(np.float32) * zone3

        # Preserve high-frequency skin textures in the morphed area to prevent blurriness
        base = cv2.GaussianBlur(image_np.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
        detail = image_np.astype(np.float32) - base
        gain = float(np.clip(0.10 + 0.03 * intensity, 0.06, 0.16))
        out_f = np.clip(out_f + gain * detail, 0, 255)

        return out_f.astype(np.uint8)

    def slim_face(self, image_np: np.ndarray, landmarks: list[tuple[int, int]], intensity: float = 0.6, smooth: float = 3.0) -> np.ndarray:
        h, w = image_np.shape[:2]
        n = len(landmarks)
        jaw_ids = self._safe_ids(REGION_INDICES["jawline_outer"], n)
        cheek_ids = self._safe_ids(REGION_INDICES["cheeks_outer"], n)
        upper_cheek_ids = self._safe_ids(REGION_INDICES["upper_cheeks"], n)
        ear_anchor_ids = self._safe_ids(REGION_INDICES["ear_side_anchors"], n)

        if len(jaw_ids) < 8:
            return image_np.copy()

        pull_ids = [i for i in (jaw_ids + cheek_ids) if i not in set(upper_cheek_ids)]
        anchor_ids = upper_cheek_ids + ear_anchor_ids

        selected = pull_ids + anchor_ids
        src = np.array([landmarks[i] for i in selected], dtype=np.float32)

        center_candidates = [idx for idx in [168, 1, 2, 4] if idx < n]
        if center_candidates:
            cx = float(np.mean([landmarks[i][0] for i in center_candidates]))
        else:
            cx = float(np.mean([landmarks[i][0] for i in jaw_ids]))

        dst = src.copy()
        ys = np.array([landmarks[i][1] for i in jaw_ids], dtype=np.float32)
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_span = max(1.0, y_max - y_min)

        pull_set = set(pull_ids)

        for i in range(len(dst)):
            x, y = dst[i]
            lm_id = selected[i]
            if lm_id not in pull_set:
                continue

            dx = cx - x
            y_ratio = np.clip((y - y_min) / y_span, 0.0, 1.0)

            # Stronger pull in lower jawline/cheek hollow, weaker around upper cheekbones.
            horiz_pull = (0.08 + 0.24 * y_ratio) * intensity
            dst[i, 0] = x + dx * horiz_pull

            # Slight lift in lower contour for V-shape effect.
            dst[i, 1] = y - abs(dx) * (0.010 + 0.012 * y_ratio) * intensity

        # Expand face_points 50 px outward so _face_mask is 1.0 across the entire
        # original face boundary — full displacement at the original contour, no ghost.
        all_ctrl_pts = np.array([landmarks[i] for i in selected], dtype=np.float32)
        ctrl_center  = np.mean(all_ctrl_pts, axis=0)
        ctrl_dirs    = all_ctrl_pts - ctrl_center
        ctrl_norms   = np.linalg.norm(ctrl_dirs, axis=1, keepdims=True) + 1e-6
        ctrl_expanded = all_ctrl_pts + (ctrl_dirs / ctrl_norms) * 50.0
        face_points   = np.vstack([all_ctrl_pts, ctrl_expanded])

        warped = self._rbf_warp(
            image_np,
            src,
            dst,
            face_points,
            smooth=smooth,
            kernel=_DEF_KERNEL,
            neighbors=128,
        )

        # Build zone from ALL modified control points (jaw + cheeks + anchors) so
        # both the jaw ghost AND cheek ghost are covered.  Forehead points extend
        # the zone to the top of the face for a complete face-oval mask.
        forehead_ids = self._safe_ids([10, 9, 151, 337, 299, 109, 67, 103, 54, 21], n)
        zone_pts = [landmarks[i] for i in selected] + [landmarks[i] for i in forehead_ids]
        # dilate_iter=4 → ~20 px outward so mask=1.0 at all original contour points.
        zone = _polygon_mask(zone_pts, h, w, blur=23, dilate_iter=4)
        zone3 = np.stack([zone, zone, zone], axis=2)
        out_f = image_np.astype(np.float32) * (1.0 - zone3) + warped.astype(np.float32) * zone3

        # Restore high-frequency skin detail.
        base   = cv2.GaussianBlur(image_np.astype(np.float32), (0, 0), sigmaX=1.05)
        detail = image_np.astype(np.float32) - base
        out_f  = np.clip(out_f + 0.12 * detail, 0, 255)
        return out_f.astype(np.uint8)

    def pro_smile_enhancement(
        self,
        image_np: np.ndarray,
        landmarks: list[tuple[int, int]],
        intensity: float = 0.6,
        smooth: float = 2.7,
    ) -> np.ndarray:
        h, w = image_np.shape[:2]
        n = len(landmarks)
        if n < 468:
            return image_np.copy()

        scale = float(min(h, w))
        intensity_eff = float(np.clip(intensity, 0.0, 1.0))

        # Muscular smile shifts:
        # We define shifts for corners, adjacent lip regions, lip centers, cheeks, and eyelids
        # to simulate a natural coordinated zygomaticus muscle contraction.
        
        # 1. Corners (move up and slightly out)
        left_corners = [61, 78]
        right_corners = [291, 308]
        
        # 2. Adjacent Lip regions (move up and slightly out at 60% of corner scale)
        left_adjacent = [185, 40, 191, 80, 146, 91, 95, 88]
        right_adjacent = [409, 270, 415, 310, 375, 321, 324, 318]
        
        # 3. Upper Lip Center (very small lift to prevent compression)
        upper_center = [0, 37, 267, 13, 81, 82, 311, 312]
        
        # 4. Lower Lip Center (very small lift to follow upper lip)
        lower_center = [17, 84, 314, 14, 87, 178, 317, 402]
        
        # 5. Cheeks (lifted up and outward)
        left_cheeks = [205, 50, 187]
        right_cheeks = [425, 280, 411]
        
        # 6. Lower eyelids (Duchenne squint)
        left_eyelids = [145, 153, 154]
        right_eyelids = [374, 380, 381]

        # Shift fractions
        dx_corner = 0.010
        dy_corner = -0.018
        
        src_list = []
        dst_list = []

        # Left corners
        for lm_id in left_corners:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx - scale * dx_corner * intensity_eff, sy + scale * dy_corner * intensity_eff))

        # Right corners
        for lm_id in right_corners:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx + scale * dx_corner * intensity_eff, sy + scale * dy_corner * intensity_eff))

        # Left adjacent
        for lm_id in left_adjacent:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx - scale * dx_corner * 0.6 * intensity_eff, sy + scale * dy_corner * 0.6 * intensity_eff))

        # Right adjacent
        for lm_id in right_adjacent:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx + scale * dx_corner * 0.6 * intensity_eff, sy + scale * dy_corner * 0.6 * intensity_eff))

        # Upper center (very small lift)
        for lm_id in upper_center:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy - scale * 0.004 * intensity_eff))

        # Lower center (very small lift)
        for lm_id in lower_center:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy - scale * 0.002 * intensity_eff))

        # Left cheeks
        for lm_id in left_cheeks:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx - scale * 0.004 * intensity_eff, sy - scale * 0.007 * intensity_eff))

        # Right cheeks
        for lm_id in right_cheeks:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx + scale * 0.004 * intensity_eff, sy - scale * 0.007 * intensity_eff))

        # Left eyelids
        for lm_id in left_eyelids:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy - scale * 0.002 * intensity_eff))

        # Right eyelids
        for lm_id in right_eyelids:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy - scale * 0.002 * intensity_eff))

        # Anchor points to ensure stable nose, forehead, jaw, ears, and eyelids
        anchor_ids = self._safe_ids(
            REGION_INDICES["forehead_anchors"] +
            [168, 6, 197, 195, 1, 2, 94] +
            [159, 160, 161, 158, 157, 386, 385, 384, 387, 388] +
            REGION_INDICES["jawline_outer"] +
            REGION_INDICES["ear_side_anchors"],
            n
        )
        
        # Deduplicate anchor IDs
        moving_set = set(left_corners + right_corners + left_adjacent + right_adjacent + upper_center + lower_center + left_cheeks + right_cheeks + left_eyelids + right_eyelids)
        anchor_ids = list(set(anchor_ids) - moving_set)

        for lm_id in anchor_ids:
            if lm_id < n:
                sx, sy = landmarks[lm_id]
                src_list.append((sx, sy))
                dst_list.append((sx, sy))  # zero displacement anchor

        if not src_list:
            return image_np.copy()

        src_arr = np.array(src_list, dtype=np.float32)
        dst_arr = np.array(dst_list, dtype=np.float32)
        face_points = self._face_points(landmarks)

        # Higher smoothing (min 3.6) makes the RBF transition very global and natural
        warped = self._rbf_warp(image_np, src_arr, dst_arr, face_points, smooth=max(smooth, 3.6))

        # Create a soft blend zone mask for the lower face (lips, cheeks, lower nose, lower eyes)
        # with high blur (35px) for an extremely soft, natural falloff on the surrounding skin.
        zone_ids = self._safe_ids(
            REGION_INDICES["lips_all"] +
            REGION_INDICES["nasolabial"] +
            REGION_INDICES["eye_lower"] +
            [1, 2, 94, 168, 6],
            n
        )
        zone_pts = [landmarks[i] for i in zone_ids]
        zone = _polygon_mask(zone_pts, h, w, blur=35, dilate_iter=2)
        zone3 = np.stack([zone, zone, zone], axis=2)

        # Composite the warped smile region back onto the original image
        out_f = image_np.astype(np.float32) * (1.0 - zone3) + warped.astype(np.float32) * zone3

        # Recover fine high-frequency skin textures in the morphed area to prevent blurriness
        base = cv2.GaussianBlur(image_np.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
        detail = image_np.astype(np.float32) - base
        gain = float(np.clip(0.12 + 0.05 * intensity_eff, 0.08, 0.22))
        out_f = np.clip(out_f + gain * detail, 0, 255)

        return out_f.astype(np.uint8)

    def pro_brow_lift(
        self,
        image_np: np.ndarray,
        landmarks: list[tuple[int, int]],
        intensity: float = 0.6,
        smooth: float = 3.2,
    ) -> np.ndarray:
        n = len(landmarks)
        brow_left = self._safe_ids(REGION_INDICES["eyebrow_left_arc"], n)
        brow_right = self._safe_ids(REGION_INDICES["eyebrow_right_arc"], n)

        brows = brow_left + brow_right
        if len(brows) < 6:
            return image_np.copy()

        operation = "brow_lift"
        print(f"[DEBUG] Moving region: {operation} with {len(brows)} points.")

        h, w = image_np.shape[:2]
        scale = float(min(image_np.shape[:2]))
        intensity_eff = float(np.clip(intensity, 0.0, 1.0))

        # Per-landmark upward lift as a fraction of face scale. Brow-arc peak
        # points lift the most; tail points less, for a natural arch.
        lift_strength = {
            70: 0.020, 63: 0.016, 105: 0.016, 66: 0.014, 107: 0.013,
            300: 0.020, 293: 0.016, 334: 0.016, 296: 0.014, 336: 0.013,
        }

        # Smooth global RBF warp: each brow control point is moved up by its
        # lift amount while the eye landmarks are pinned (dst == src → zero
        # displacement). This replaces the old localized vertical flow warp,
        # which left the forehead above the brow fixed and so produced a
        # "ghost brow" smear. The RBF spreads the deformation smoothly across
        # the forehead and never tears the brow/eyelid boundary.
        src = np.array([landmarks[i] for i in brows], dtype=np.float32)
        dst = src.copy()
        for k, lm_id in enumerate(brows):
            dst[k, 1] -= scale * lift_strength.get(lm_id, 0.011) * intensity_eff

        # Eye landmarks added as zero-displacement anchors: the RBF is forced
        # to produce zero displacement at the eye boundary, preventing the eyes
        # from being dragged upward when the brows move.
        eye_anchor_ids = self._safe_ids(
            [33, 133, 145, 159,               # left eye corners + lid centers
             7, 163, 144, 153, 154, 155,       # left lower lid
             362, 263, 374, 386,               # right eye corners + lid centers
             249, 390, 373, 380, 381, 382],    # right lower lid
            n,
        )
        if eye_anchor_ids:
            eye_src = np.array([landmarks[i] for i in eye_anchor_ids], dtype=np.float32)
            src_rbf = np.vstack([src, eye_src])
            dst_rbf = np.vstack([dst, eye_src])   # dst == src → zero displacement
        else:
            src_rbf = src
            dst_rbf = dst

        face_points = self._face_points(landmarks)
        warped = self._rbf_warp(image_np, src_rbf, dst_rbf, face_points, smooth=min(smooth, 1.8))

        # Brow blend zone: brow arcs + forehead top + upper eyelid boundary.
        h, w = image_np.shape[:2]
        forehead_top = self._safe_ids([10, 9, 151, 337, 299, 109, 67], n)
        upper_lid = self._safe_ids([159, 160, 161, 386, 385, 384], n)
        brow_zone_pts = [landmarks[i] for i in brow_left + brow_right + forehead_top + upper_lid]
        zone = _polygon_mask(brow_zone_pts, h, w, blur=11, dilate_iter=1)

        # Explicitly cut out each eye hull from the blend zone so no warped eye
        # pixels are ever composited in, even if the RBF residual is non-zero.
        for eye_ids in [
            self._safe_ids([33, 133, 145, 159, 160, 161, 163, 144, 153, 154, 155, 158, 157], n),
            self._safe_ids([362, 263, 374, 386, 385, 384, 390, 373, 380, 381, 382, 387, 388], n),
        ]:
            if len(eye_ids) >= 3:
                eye_mask = np.zeros((h, w), dtype=np.uint8)
                hull = cv2.convexHull(np.array([landmarks[i] for i in eye_ids], dtype=np.int32))
                cv2.fillConvexPoly(eye_mask, hull, 255)
                eye_mask = cv2.dilate(eye_mask, np.ones((5, 5), np.uint8), iterations=1)
                eye_mask = cv2.GaussianBlur(eye_mask, (15, 15), 0).astype(np.float32) / 255.0
                zone = np.clip(zone - eye_mask, 0.0, 1.0)

        zone3 = np.stack([zone, zone, zone], axis=2)
        alpha = float(np.clip(0.55 + intensity_eff * (0.82 - 0.55), 0.0, 1.0))
        blended = image_np.astype(np.float32) * (1.0 - alpha * zone3) + warped.astype(np.float32) * (alpha * zone3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def aging_pro(
        self,
        image_np: np.ndarray,
        landmarks: list[tuple[int, int]] | None,
        intensity: float = 0.6,
    ) -> dict:
        return apply_pro_aging(image_np, landmarks=landmarks, intensity=intensity)

    def de_aging_pro(
        self,
        image_np: np.ndarray,
        landmarks: list[tuple[int, int]] | None,
        intensity: float = 0.6,
    ) -> dict:
        return apply_pro_deaging(image_np, landmarks=landmarks, intensity=intensity)


_PRO_WARP = ProWarpManager()


def plump_lips_pro(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    intensity: float = 0.6,
    smooth: float = 2.2,
) -> np.ndarray:
    return _PRO_WARP.plump_lips(image_np, landmarks, intensity=intensity, smooth=smooth)


def slim_face_pro(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    intensity: float = 0.6,
    smooth: float = 3.4,
) -> np.ndarray:
    return _PRO_WARP.slim_face(image_np, landmarks, intensity=intensity, smooth=smooth)


def smile_enhancement_pro(image_np: np.ndarray, landmarks: list[tuple[int, int]], intensity: float = 0.6, smooth: float = 2.7) -> np.ndarray:
    return _PRO_WARP.pro_smile_enhancement(image_np, landmarks, intensity=intensity, smooth=smooth)


def brow_lift_pro(image_np: np.ndarray, landmarks: list[tuple[int, int]], intensity: float = 0.6, smooth: float = 3.2) -> np.ndarray:
    return _PRO_WARP.pro_brow_lift(image_np, landmarks, intensity=intensity, smooth=smooth)


def aging_pro(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]] | None,
    intensity: float = 0.6,
) -> dict:
    return _PRO_WARP.aging_pro(image_np, landmarks, intensity=intensity)


def de_aging_pro(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]] | None,
    intensity: float = 0.6,
) -> dict:
    return _PRO_WARP.de_aging_pro(image_np, landmarks, intensity=intensity)


def quality_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    return _compute_metrics(original, processed)
