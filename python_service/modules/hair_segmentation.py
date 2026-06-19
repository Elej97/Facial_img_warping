"""Hair segmentation and whitening using MediaPipe multiclass selfie segmenter.

The selfie_multiclass_256x256 model outputs per-pixel category masks:
  0 = background
  1 = hair
  2 = body skin
  3 = face skin
  4 = clothes
  5 = others (accessories, etc.)

We extract category 1 (hair) for a precise hair mask, then apply a
natural-looking silver/gray whitening effect.
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.model_utils import get_model_path


# ---------------------------------------------------------------------------
# Segmenter singleton
# ---------------------------------------------------------------------------

_segmenter: mp_vision.ImageSegmenter | None = None


def _get_segmenter() -> mp_vision.ImageSegmenter:
    global _segmenter
    if _segmenter is None:
        model_path = get_model_path("selfie_multiclass")
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _segmenter = mp_vision.ImageSegmenter.create_from_options(options)
    return _segmenter


# ---------------------------------------------------------------------------
# Hair mask via MediaPipe segmentation
# ---------------------------------------------------------------------------

def get_hair_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Return a soft [0..1] float32 hair mask using MediaPipe segmenter.

    OPTIMIZATION: Downscales large images for segmentation phase.
    """
    h, w = image_bgr.shape[:2]
    try:
        segmenter = _get_segmenter()
        
        # --- Optimization: Downscale for segmenter if large ---
        # The model uses 256x256 internally anyway.
        target_dim = 1024
        if max(h, w) > target_dim:
            scale = target_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            proc_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_img = image_bgr

        rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment(mp_image)

        if result.category_mask is not None:
            cat = result.category_mask.numpy_view()           # (H, W) uint8
            hair_raw = (cat == 1).astype(np.uint8) * 255      # category 1 = hair
            # Resize to original dimensions
            hair_raw = cv2.resize(hair_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            # Morphological cleanup
            hair_raw = cv2.morphologyEx(hair_raw, cv2.MORPH_CLOSE,
                                        np.ones((5, 5), np.uint8), iterations=2)
            hair_raw = cv2.morphologyEx(hair_raw, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8), iterations=1)
            # Smooth edges for natural blending
            hair_raw = cv2.GaussianBlur(hair_raw, (21, 21), 0)
            return hair_raw.astype(np.float32) / 255.0

    except Exception as exc:
        print(f"[hair_segmentation] Model fallback: {exc}")

    # Fallback: empty mask
    return np.zeros((h, w), dtype=np.float32)


# ---------------------------------------------------------------------------
# Hair whitening / silver effect
# ---------------------------------------------------------------------------

def apply_hair_whitening(
    image_bgr: np.ndarray,
    hair_mask: np.ndarray | None = None,
    intensity: float = 0.6,
) -> np.ndarray:
    """Apply a natural silver/gray hair whitening effect."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.05:
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]

    if hair_mask is None:
        hair_mask = get_hair_mask(image_bgr)

    # Skip if no hair detected
    if float(np.max(hair_mask)) < 0.1:
        return image_bgr.copy()

    result = image_bgr.astype(np.float32)
    hair3 = np.expand_dims(hair_mask, axis=2)

    # --- Step 1: Convert to LAB for perceptual color manipulation ---
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]  # Luminance
    a_channel = lab[:, :, 1]  # Green-Red
    b_channel = lab[:, :, 2]  # Blue-Yellow

    # --- Step 2: Desaturate hair (push a,b channels toward neutral 128) ---
    desat_strength = 0.70 + 0.30 * intensity  # 0.70 .. 1.0
    a_neutral = a_channel * (1.0 - desat_strength * hair_mask) + 128.0 * (desat_strength * hair_mask)
    b_neutral = b_channel * (1.0 - desat_strength * hair_mask) + 128.0 * (desat_strength * hair_mask)

    # --- Step 3: Brighten luminance toward silver ---
    # Target luminance for silver hair: bright but not blown out
    target_l = 180.0 + 40.0 * intensity  # 180..220
    l_bright = l_channel + (target_l - l_channel) * (0.35 + 0.35 * intensity) * hair_mask

    # --- Step 4: Add subtle cool tint (slightly blue-ish silver) ---
    b_neutral = b_neutral - 4.0 * intensity * hair_mask  # slight blue shift

    # --- Step 5: Reconstruct and preserve texture ---
    # Extract luminance detail from original (high-frequency texture)
    l_smooth = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=2.0, sigmaY=2.0)
    l_detail = l_channel - l_smooth  # hair strand texture

    # Apply detail back to brightened luminance
    l_final = l_bright + l_detail * (0.6 + 0.4 * intensity)

    lab_out = np.stack([
        np.clip(l_final, 0, 255),
        np.clip(a_neutral, 0, 255),
        np.clip(b_neutral, 0, 255),
    ], axis=2).astype(np.uint8)

    silver_hair = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

    # --- Step 6: Blend with original using the hair mask ---
    blend_alpha = hair3 * (0.50 + 0.50 * intensity)  # 0.50..1.0 within hair
    out = result * (1.0 - blend_alpha) + silver_hair * blend_alpha

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Hair color change — any target color
# ---------------------------------------------------------------------------

def apply_hair_color(
    image_bgr: np.ndarray,
    hex_color: str,
    intensity: float = 0.85,
    hair_mask: np.ndarray | None = None,
    preserve_highlights: bool = True,
) -> np.ndarray:
    """
    Professional-quality hair recoloring.

    Algorithm (Photoshop "Color" blend mode equivalent):
      1. MediaPipe segmentation → soft [0,1] hair mask
      2. Convert to LAB — replace A/B (chrominance), keep L (luminance)
         This preserves every shadow, strand highlight and texture automatically.
      3. For vivid colors on dark hair: gently lift L so the color becomes visible
         (realistic "dark-tinted" look, e.g. black → deep navy/dark purple).
      4. High-frequency luminance detail recovered → strand texture stays sharp.
      5. Specular highlight protection — very bright pixels keep original tone.
      6. Feathered mask blend → no hard edges at hairline.

    Args:
        image_bgr: Input image (BGR, uint8).
        hex_color: Target hair color, e.g. "#A020F0" or "A020F0".
        intensity: 0.0 = no change, 1.0 = full replacement.
        hair_mask: Pre-computed [0,1] float32 mask. Computed if None.
        preserve_highlights: Keep specular shine areas mostly original.

    Returns:
        uint8 BGR image.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.02:
        return image_bgr.copy()

    # --- 1. Parse target hex color ---
    hex_clean = (hex_color or '').strip().lstrip('#')
    if len(hex_clean) == 3:
        hex_clean = ''.join(c * 2 for c in hex_clean)
    if len(hex_clean) != 6:
        raise ValueError(f"Invalid hex_color: {hex_color!r}. Use '#RRGGBB' or 'RRGGBB'.")
    try:
        r_int = int(hex_clean[0:2], 16)
        g_int = int(hex_clean[2:4], 16)
        b_int = int(hex_clean[4:6], 16)
    except ValueError:
        raise ValueError(f"Invalid hex_color: {hex_color!r}.")

    target_bgr_px = np.array([[[b_int, g_int, r_int]]], dtype=np.uint8)

    # Target color in LAB (OpenCV LAB: L∈[0,255], A/B∈[0,255] centred at 128)
    target_lab_px = cv2.cvtColor(target_bgr_px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    target_L_val = float(target_lab_px[0])
    target_A_val = float(target_lab_px[1])
    target_B_val = float(target_lab_px[2])

    # Determine saturation to classify natural vs vivid
    target_hsv_px = cv2.cvtColor(target_bgr_px, cv2.COLOR_BGR2HSV)[0, 0]
    target_sat = float(target_hsv_px[1]) / 255.0   # 0.0 – 1.0
    is_vivid = target_sat > 0.40

    # --- 2. Acquire hair mask ---
    if hair_mask is None:
        hair_mask = get_hair_mask(image_bgr)

    if float(np.max(hair_mask)) < 0.05:
        # No hair detected — return unchanged
        return image_bgr.copy()

    h, w = image_bgr.shape[:2]

    # --- 3. Convert full image to LAB ---
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch = lab[:, :, 0]   # luminance
    A_ch = lab[:, :, 1]   # green–red axis
    B_ch = lab[:, :, 2]   # blue–yellow axis

    # Mean luminance inside the hair region (used for L-lifting decision)
    hair_px_mask = hair_mask > 0.3
    mean_hair_L = float(np.mean(L_ch[hair_px_mask])) if np.any(hair_px_mask) else 100.0

    # --- 4. A/B channel replacement (main color change) ---
    blend = intensity * hair_mask                        # per-pixel blend amount
    new_A = A_ch + (target_A_val - A_ch) * blend
    new_B = B_ch + (target_B_val - B_ch) * blend

    # --- 5. Luminance adjustment ---
    # Vivid colors on dark hair need a gentle L-lift so the hue becomes visible.
    # Natural colors: very small L shift, just to compensate light→dark or dark→light.
    if is_vivid and mean_hair_L < 80.0:
        # e.g. black hair → blue: lift dark regions enough to show the hue
        # The brighter the target saturation, the higher we lift.
        min_L_target = 55.0 + target_sat * 40.0          # 55 – 95
        l_deficit = np.maximum(0.0, min_L_target - L_ch) # how much lift is needed
        l_lift = l_deficit * intensity * hair_mask         # apply proportionally to mask
        new_L = L_ch + l_lift
    else:
        # Gentle luminance blend (≤ 25% of the target L contribution)
        l_blend_strength = 0.20 * intensity
        new_L = L_ch + (target_L_val - L_ch) * (l_blend_strength * hair_mask)

    # --- 6. Preserve high-frequency strand detail ---
    # Recover fine hair texture by adding back high-freq luminance detail from original.
    l_smooth = cv2.GaussianBlur(L_ch, (0, 0), sigmaX=3.0)
    l_detail = L_ch - l_smooth                           # ± fine strand texture
    texture_strength = np.clip(0.85 - 0.35 * intensity, 0.4, 0.85)
    new_L = new_L + l_detail * (texture_strength * hair_mask)

    # --- 7. Specular highlight protection ---
    # Very bright areas (L > 175) should not be heavily recoloured.
    # This keeps the natural shine/gloss on hair.
    if preserve_highlights:
        # Smooth ramp: 0 at L=170, 1 at L=215
        highlight_alpha = np.clip((L_ch - 170.0) / 45.0, 0.0, 1.0)
        protect = highlight_alpha * 0.88                 # up to 88 % revert
        new_A = new_A * (1.0 - protect) + A_ch * protect
        new_B = new_B * (1.0 - protect) + B_ch * protect
        new_L = new_L * (1.0 - protect * 0.65) + L_ch * (protect * 0.65)

    # --- 8. Reconstruct LAB → BGR ---
    lab_out = np.stack([
        np.clip(new_L, 0.0, 255.0),
        np.clip(new_A, 0.0, 255.0),
        np.clip(new_B, 0.0, 255.0),
    ], axis=2).astype(np.uint8)

    colored = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

    # --- 9. Final blend with original ---
    hair3 = hair_mask[:, :, np.newaxis]                  # broadcast over channels
    original = image_bgr.astype(np.float32)
    result = original * (1.0 - hair3) + colored * hair3

    return np.clip(result, 0.0, 255.0).astype(np.uint8)
