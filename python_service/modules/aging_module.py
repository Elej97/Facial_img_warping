from __future__ import annotations

import cv2
import numpy as np

try:
    import pywt  # type: ignore
except Exception:
    pywt = None

from modules.evaluation_metrics import evaluate_metrics
from modules.hair_segmentation import apply_hair_whitening as _mp_hair_whitening
from modules.landmark import detect_landmarks
from modules.region_map import REGION_INDICES


def _polygon_mask(points: list[tuple[int, int]], h: int, w: int, blur: int = 21, dilate_iter: int = 1) -> np.ndarray:
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


def _collect_points(landmarks: list[tuple[int, int]], indices: list[int]) -> list[tuple[int, int]]:
    return [landmarks[i] for i in indices if i < len(landmarks)]


def _default_skin_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.52
    rx, ry = w * 0.33, h * 0.42
    e = ((xx - cx) ** 2) / (rx * rx + 1e-6) + ((yy - cy) ** 2) / (ry * ry + 1e-6)
    mask = np.clip(1.0 - e, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (41, 41), 0)


def build_skin_mask(image_np: np.ndarray, landmarks: list[tuple[int, int]] | None = None) -> np.ndarray:
    h, w = image_np.shape[:2]

    # FR-4.2 / FR-4.4: Face parsing stage - keep effects in skin areas only.
    if landmarks is None:
        landmarks = detect_landmarks(image_np)

    if not landmarks:
        return _default_skin_mask(h, w)

    face_points = np.array(landmarks, dtype=np.int32)
    face_hull = cv2.convexHull(face_points)

    skin_mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(skin_mask_u8, face_hull, 255)

    left_eye = _collect_points(landmarks, REGION_INDICES.get("eye_lower_left", []))
    right_eye = _collect_points(landmarks, REGION_INDICES.get("eye_lower_right", []))
    lips = _collect_points(landmarks, REGION_INDICES.get("lips_all", []))

    for region in (left_eye, right_eye, lips):
        if len(region) >= 3:
            cv2.fillConvexPoly(skin_mask_u8, cv2.convexHull(np.array(region, dtype=np.int32)), 0)

    skin_mask_u8 = cv2.erode(skin_mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    skin_mask_u8 = cv2.GaussianBlur(skin_mask_u8, (31, 31), 0)
    return skin_mask_u8.astype(np.float32) / 255.0


def _laplacian_decompose(gray: np.ndarray, levels: int = 3) -> tuple[np.ndarray, list[np.ndarray]]:
    gp = [gray.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))

    high_bands: list[np.ndarray] = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        high_bands.append(gp[i] - up)
    low_band = gp[-1]
    return low_band, high_bands


def _laplacian_reconstruct(low_band: np.ndarray, high_bands: list[np.ndarray]) -> np.ndarray:
    current = low_band
    for i in reversed(range(len(high_bands))):
        current = cv2.pyrUp(current, dstsize=(high_bands[i].shape[1], high_bands[i].shape[0])) + high_bands[i]
    return current


def _gabor_wrinkle_map(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    acc = np.zeros_like(gray_f, dtype=np.float32)

    # FR-4.2: Directional wrinkle texture generation using Gabor filters.
    for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
        kernel = cv2.getGaborKernel((23, 23), sigma=4.4, theta=theta, lambd=9.5, gamma=0.55, psi=0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        acc += np.abs(resp)

    acc = cv2.normalize(acc, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(acc, (7, 7), 0)


def _wrinkle_focus_mask(landmarks: list[tuple[int, int]] | None, h: int, w: int, skin_mask: np.ndarray) -> np.ndarray:
    if not landmarks:
        return skin_mask

    # Eye-corner and smile-line emphasis regions.
    forehead = _collect_points(landmarks, REGION_INDICES.get("forehead_wrinkle", []))
    crow_l = _collect_points(landmarks, REGION_INDICES.get("left_crow_feet", []))
    crow_r = _collect_points(landmarks, REGION_INDICES.get("right_crow_feet", []))
    naso_l = _collect_points(landmarks, REGION_INDICES.get("left_nasolabial", []))
    naso_r = _collect_points(landmarks, REGION_INDICES.get("right_nasolabial", []))

    focus = np.zeros((h, w), dtype=np.float32)
    focus += _polygon_mask(forehead, h, w, blur=25, dilate_iter=1) * 0.85
    focus += _polygon_mask(crow_l, h, w, blur=19, dilate_iter=1) * 1.20
    focus += _polygon_mask(crow_r, h, w, blur=19, dilate_iter=1) * 1.20
    focus += _polygon_mask(naso_l, h, w, blur=23, dilate_iter=1) * 1.10
    focus += _polygon_mask(naso_r, h, w, blur=23, dilate_iter=1) * 1.10

    focus = np.clip(focus, 0.0, 1.0)
    return np.clip(0.40 * skin_mask + 0.90 * focus, 0.0, 1.0)


def _high_frequency_noise(h: int, w: int, seed: int = 19) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w)).astype(np.float32)
    low = cv2.GaussianBlur(n, (0, 0), sigmaX=2.2, sigmaY=2.2)
    hf = n - low
    hf = cv2.normalize(hf, None, -1.0, 1.0, cv2.NORM_MINMAX)
    return hf


def _add_gaussian_flow(flow_x: np.ndarray, flow_y: np.ndarray, center: tuple[int, int], shift: tuple[float, float], sigma: float) -> None:
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


def _apply_flow_warp(image_np: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x - flow_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(grid_y - flow_y, 0, h - 1).astype(np.float32)
    return cv2.remap(image_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _apply_aging_expression_droop(image_np: np.ndarray, landmarks: list[tuple[int, int]] | None, intensity: float) -> np.ndarray:
    if not landmarks:
        return image_np

    h, w = image_np.shape[:2]
    scale = float(min(h, w))
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    # FR-4.2: Add age-like geometric cues (brow and upper eyelid droop).
    brow_ids = REGION_INDICES.get("eyebrow_left_arc", []) + REGION_INDICES.get("eyebrow_right_arc", [])
    lid_ids = [159, 160, 161, 386, 385, 384]

    brow_shift_y = scale * (0.006 + 0.010 * intensity)
    lid_shift_y = scale * (0.004 + 0.008 * intensity)

    for idx in brow_ids:
        if idx < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[idx], (0.0, brow_shift_y), sigma=scale * 0.060)

    for idx in lid_ids:
        if idx < len(landmarks):
            _add_gaussian_flow(flow_x, flow_y, landmarks[idx], (0.0, lid_shift_y), sigma=scale * 0.045)

    return _apply_flow_warp(image_np, flow_x, flow_y)


def _apply_facial_sagging(image_np: np.ndarray, landmarks: list[tuple[int, int]] | None, intensity: float) -> np.ndarray:
    if not landmarks:
        return image_np

    h, w = image_np.shape[:2]
    scale = float(min(h, w))
    flow_x = np.zeros((h, w), dtype=np.float32)
    flow_y = np.zeros((h, w), dtype=np.float32)

    # Age sag around smile corners.
    for idx in [61, 291]:
        if idx < len(landmarks):
            _add_gaussian_flow(
                flow_x,
                flow_y,
                landmarks[idx],
                (0.0, scale * (0.006 + 0.010 * intensity)),
                sigma=scale * 0.055,
            )

    # Nasolabial region sagging.
    for idx in (REGION_INDICES.get("left_nasolabial", []) + REGION_INDICES.get("right_nasolabial", []))[::2]:
        if idx < len(landmarks):
            _add_gaussian_flow(
                flow_x,
                flow_y,
                landmarks[idx],
                (0.0, scale * (0.004 + 0.009 * intensity)),
                sigma=scale * 0.050,
            )

    # Lower-face tissue sag (jaw and lower cheeks).
    lower_face_ids = REGION_INDICES.get("jawline_outer", []) + REGION_INDICES.get("cheeks_outer", [])
    lower_face_ids = [i for i in lower_face_ids if i < len(landmarks)]
    if lower_face_ids:
        cx = float(np.mean([landmarks[i][0] for i in lower_face_ids]))
        half_w = max(1.0, w * 0.5)
        for idx in lower_face_ids[::2]:
            x, _ = landmarks[idx]
            center_weight = 1.0 - min(1.0, abs(float(x) - cx) / half_w)
            dy = scale * (0.004 + 0.011 * intensity) * (0.75 + 0.55 * center_weight)
            _add_gaussian_flow(flow_x, flow_y, landmarks[idx], (0.0, dy), sigma=scale * 0.060)

    return _apply_flow_warp(image_np, flow_x, flow_y)


def _fft_spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return fshift, mag


def _spectrum_vis(mag: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _hf_lf_energy_ratio(gray: np.ndarray) -> float:
    h, w = gray.shape
    fshift, _ = _fft_spectrum(gray)
    power = np.abs(fshift) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = min(h, w) * 0.18
    lf = float(np.sum(power[dist <= r]) + 1e-6)
    hf = float(np.sum(power[dist > r]) + 1e-6)
    return hf / lf


def _augment_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    base = evaluate_metrics(original, processed)
    org_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    out_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    hf_lf_before = _hf_lf_energy_ratio(org_gray)
    hf_lf_after = _hf_lf_energy_ratio(out_gray)

    return {
        "mse": float(base["mse"]),
        "psnr": float(base["psnr"]),
        "ssim": float(base["ssim"]),
        "hf_lf_ratio_before": float(hf_lf_before),
        "hf_lf_ratio_after": float(hf_lf_after),
        "hf_lf_ratio_delta": float(hf_lf_after - hf_lf_before),
    }


def _wavelet_aging_luma(luma: np.ndarray, skin_mask: np.ndarray, intensity: float) -> np.ndarray:
    if pywt is None:
        low_band, high_bands = _laplacian_decompose(luma, levels=3)
        boosted_bands: list[np.ndarray] = []
        for band in high_bands:
            m = cv2.resize(skin_mask, (band.shape[1], band.shape[0]), interpolation=cv2.INTER_LINEAR)
            gain = 1.0 + (0.62 + 0.46 * intensity) * m
            boosted_bands.append(band * gain)
        return np.clip(_laplacian_reconstruct(low_band, boosted_bands), 0, 255).astype(np.float32)

    coeffs = pywt.wavedec2(luma.astype(np.float32), wavelet="db2", level=2)
    boosted = [coeffs[0]]

    for level, details in enumerate(coeffs[1:], start=1):
        c_h, c_v, c_d = details
        m = cv2.resize(skin_mask, (c_h.shape[1], c_h.shape[0]), interpolation=cv2.INTER_LINEAR)
        scale = (0.54 + 0.42 * intensity) / float(level)
        gain = 1.0 + m * scale
        boosted.append((c_h * gain, c_v * gain, c_d * gain))

    recon = pywt.waverec2(boosted, wavelet="db2")
    recon = recon[: luma.shape[0], : luma.shape[1]]
    return np.clip(recon, 0, 255).astype(np.float32)


def _wavelet_deaging_luma(luma: np.ndarray, skin_mask: np.ndarray, intensity: float) -> np.ndarray:
    if pywt is None:
        low_band, high_bands = _laplacian_decompose(luma, levels=3)
        attenuated: list[np.ndarray] = []
        for band in high_bands:
            m = cv2.resize(skin_mask, (band.shape[1], band.shape[0]), interpolation=cv2.INTER_LINEAR)
            attn = 1.0 - (0.66 + 0.40 * intensity) * m
            attenuated.append(band * np.clip(attn, 0.15, 1.0))
        return np.clip(_laplacian_reconstruct(low_band, attenuated), 0, 255).astype(np.float32)

    coeffs = pywt.wavedec2(luma.astype(np.float32), wavelet="db2", level=2)
    attenuated = [coeffs[0]]

    for level, details in enumerate(coeffs[1:], start=1):
        c_h, c_v, c_d = details
        m = cv2.resize(skin_mask, (c_h.shape[1], c_h.shape[0]), interpolation=cv2.INTER_LINEAR)
        scale = (0.72 + 0.34 * intensity) / float(level)
        attn = np.clip(1.0 - m * scale, 0.12, 1.0)
        attenuated.append((c_h * attn, c_v * attn, c_d * attn))

    recon = pywt.waverec2(attenuated, wavelet="db2")
    recon = recon[: luma.shape[0], : luma.shape[1]]
    return np.clip(recon, 0, 255).astype(np.float32)


def apply_pro_aging(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]] | None = None,
    intensity: float = 0.6,
) -> dict:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]

    skin_mask = build_skin_mask(image_np, landmarks)

    # FR-4.2: Increase high-frequency components in skin region with multi-scale decomposition.
    aged_y = _wavelet_aging_luma(y, skin_mask, intensity)

    h, w = image_np.shape[:2]
    wrinkle_mask = _wrinkle_focus_mask(landmarks, h, w, skin_mask)

    # FR-4.2: Add directional wrinkle texture (Gabor) with emphasis on eye corners and smile lines.
    wrinkle = _gabor_wrinkle_map(y.astype(np.uint8))
    wrinkle_strength = (0.20 + 0.30 * intensity) * wrinkle_mask

    # Use zero-mean modulation to avoid global darkening while increasing wrinkle visibility.
    wrinkle_centered = wrinkle - cv2.GaussianBlur(wrinkle, (0, 0), sigmaX=4.0, sigmaY=4.0)
    aged_y = aged_y + wrinkle_centered * wrinkle_strength * 118.0

    # FR-4.2: Add controlled high-frequency grain/noise to strengthen aged skin texture.
    grain = _high_frequency_noise(h, w)
    grain_strength = (0.02 + 0.08 * intensity) * wrinkle_mask
    aged_y = aged_y + grain * grain_strength * 22.0

    # FR-4.2: Slight local contrast boost in luminance only (no color shift).
    aged_y = aged_y + cv2.GaussianBlur((aged_y - y), (0, 0), sigmaX=1.0, sigmaY=1.0) * (0.06 + 0.09 * intensity)

    # Keep luminance mean stable on skin to prevent perceived skin darkening.
    skin_sum = float(np.sum(skin_mask) + 1e-6)
    orig_skin_mean = float(np.sum(y * skin_mask) / skin_sum)
    aged_skin_mean = float(np.sum(aged_y * skin_mask) / skin_sum)
    aged_y = aged_y + (orig_skin_mean - aged_skin_mean) * skin_mask

    # Keep chroma channels unchanged to preserve original skin tone.
    ycrcb[:, :, 0] = np.clip(aged_y, 0, 255)
    aged = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    # Add facial sagging (jaw/cheek/smile-line) while keeping eye droop disabled.
    aged = _apply_facial_sagging(aged, landmarks, intensity)

    # Hair whitening for aged appearance (MediaPipe segmentation).
    aged = _mp_hair_whitening(aged, intensity=intensity)

    # 3 kanal Fourier
    b, g, r = cv2.split(aged)
    gray_ch = cv2.cvtColor(aged, cv2.COLOR_BGR2GRAY)
    _, mag_gray = _fft_spectrum(gray_ch)
    _, mag_b = _fft_spectrum(b)
    _, mag_r = _fft_spectrum(r)

    return {
        "result_image": aged,
        "spectrum_gray": _spectrum_vis(mag_gray),
        "spectrum_blue": _spectrum_vis(mag_b),
        "spectrum_red": _spectrum_vis(mag_r),
        "metrics": _augment_metrics(image_np, aged),
    }


def apply_pro_deaging(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]] | None = None,
    intensity: float = 0.6,
) -> dict:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]

    skin_mask = build_skin_mask(image_np, landmarks)

    # FR-4.4: Reduce high-frequency skin components with multi-scale decomposition.
    deaged_y = _wavelet_deaging_luma(y, skin_mask, intensity)

    ycrcb[:, :, 0] = np.clip(deaged_y, 0, 255)
    freq_smoothed = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    # FR-4.5: Preserve important facial edges while smoothing using edge-preserving filter.
    edge_pres = cv2.edgePreservingFilter(
        src=image_np,
        flags=1,
        sigma_s=int(45 + 55 * intensity),
        sigma_r=float(0.24 + 0.16 * intensity),
    )

    edge_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(edge_gray, 48, 132).astype(np.float32) / 255.0
    edge_map = cv2.GaussianBlur(edge_map, (7, 7), 0)

    skin = np.expand_dims(np.clip(skin_mask, 0.0, 1.0), axis=2)
    edges = np.expand_dims(np.clip(edge_map, 0.0, 1.0), axis=2)

    smooth_alpha = np.clip((0.58 + 0.34 * intensity) * (1.0 - edges), 0.0, 1.0)
    smooth_alpha3 = skin * smooth_alpha

    blended = freq_smoothed.astype(np.float32) * (1.0 - smooth_alpha3) + edge_pres.astype(np.float32) * smooth_alpha3

    # FR-4.5: Add gentle bilateral pass only in skin to enhance rejuvenation while preserving edges.
    bilateral = cv2.bilateralFilter(image_np, d=7, sigmaColor=35 + int(45 * intensity), sigmaSpace=30 + int(40 * intensity)).astype(np.float32)
    bil_alpha = skin * (0.20 + 0.24 * intensity) * (1.0 - edges)
    blended = blended * (1.0 - bil_alpha) + bilateral * bil_alpha

    blended = blended * skin + image_np.astype(np.float32) * (1.0 - skin)
    deaged = np.clip(blended, 0, 255).astype(np.uint8)

    # 3 kanal Fourier
    b, g, r = cv2.split(deaged)
    gray_ch = cv2.cvtColor(deaged, cv2.COLOR_BGR2GRAY)
    _, mag_gray = _fft_spectrum(gray_ch)
    _, mag_b = _fft_spectrum(b)
    _, mag_r = _fft_spectrum(r)

    return {
        "result_image": deaged,
        "spectrum_gray": _spectrum_vis(mag_gray),
        "spectrum_blue": _spectrum_vis(mag_b),
        "spectrum_red": _spectrum_vis(mag_r),
        "metrics": _augment_metrics(image_np, deaged),
    }
