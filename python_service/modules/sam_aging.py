"""
SAM (Style-based Age Manipulation) — sadece foto modunda kullanılır.
Live modda çağrılmaz çünkü CPU'da 10-30s sürer.

SAM paper: https://arxiv.org/abs/2102.02754
"""
from __future__ import annotations

import os
import sys
import cv2
import numpy as np

_VENDOR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "SAM"))
_CKPT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "sam_ffhq_aging.pt"))
_SIZE_IN  = 256
_SIZE_OUT = 1024

_FFHQ_REF = np.float32([
    [ 85.6,  78.0],
    [170.4,  78.0],
    [128.0, 117.8],
    [ 92.4, 155.5],
    [163.6, 155.5],
])

_MP_L_EYE = [33, 133, 160, 159, 158, 144, 153, 145]
_MP_R_EYE = [362, 263, 387, 386, 385, 373, 380, 374]
_MP_NOSE  = [4]
_MP_M_L   = [61]
_MP_M_R   = [291]

_MP_L_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
_MP_R_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

_KEY_IDS = [4, 61, 291, 152, 234, 454, 10, 168, 94, 19]

_net  = None
_opts = None


def _lm_mean(lms, indices):
    pts = [np.array(lms[i], np.float32) for i in indices if i < len(lms)]
    return np.mean(pts, axis=0)


def _align_ffhq(img, lms):
    src = np.float32([
        _lm_mean(lms, _MP_L_EYE),
        _lm_mean(lms, _MP_R_EYE),
        _lm_mean(lms, _MP_NOSE),
        _lm_mean(lms, _MP_M_L),
        _lm_mean(lms, _MP_M_R),
    ])
    M, _ = cv2.estimateAffinePartial2D(src, _FFHQ_REF, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(src[:3], _FFHQ_REF[:3])
    aligned = cv2.warpAffine(img, M, (_SIZE_IN, _SIZE_IN),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    M_inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    return aligned, M, M_inv


def _lms_to_aligned(lms, M):
    pts  = np.array(lms, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return (np.hstack([pts, ones]) @ M.T)


def _correct_sam_alignment(sam_256, orig_lms_aligned):
    from modules.landmark_fusion import detect_landmarks_fused
    sam_lms, _ = detect_landmarks_fused(sam_256, backend="mediapipe", temporal_smoothing=False)
    if sam_lms is None or len(sam_lms) < max(_KEY_IDS) + 1:
        return sam_256
    src_pts = np.float32([sam_lms[i] for i in _KEY_IDS if i < len(sam_lms)])
    dst_pts = np.float32([orig_lms_aligned[i] for i in _KEY_IDS if i < len(orig_lms_aligned)])
    if len(src_pts) < 4:
        return sam_256
    M_corr, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M_corr is None:
        return sam_256
    # Aşırı ölçek/döndürme → düzeltme kayma yaratır, uygulama
    scale = np.sqrt(M_corr[0, 0] ** 2 + M_corr[0, 1] ** 2)
    if scale < 0.75 or scale > 1.35:
        return sam_256
    return cv2.warpAffine(sam_256, M_corr, (_SIZE_IN, _SIZE_IN),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _build_face_mask(lms_aligned):
    """Renk eşleştirmesi için — yüz merkezi (saç/göz hariç)."""
    pts = np.clip(lms_aligned.astype(np.int32), 0, _SIZE_IN - 1)
    hull = cv2.convexHull(pts)
    mask = np.zeros((_SIZE_IN, _SIZE_IN), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=4)
    return mask


def _build_blend_mask(lms_aligned):
    """Alpha blend maskesi — yüz + biraz saç, yumuşak geçiş."""
    pts = np.clip(lms_aligned.astype(np.int32), 0, _SIZE_IN - 1)
    hull_pts = cv2.convexHull(pts, returnPoints=True).reshape(-1, 2)

    x_min, x_max = hull_pts[:, 0].min(), hull_pts[:, 0].max()
    y_min, y_max = hull_pts[:, 1].min(), hull_pts[:, 1].max()
    cx = (x_min + x_max) // 2
    face_h = int(y_max - y_min)
    face_w = int(x_max - x_min)
    hair_ext = int(face_h * 0.45)
    hair_top = max(0, y_min - hair_ext)

    extra = np.array([
        [cx,                   hair_top],
        [cx - face_w // 3,     hair_top + hair_ext // 3],
        [cx + face_w // 3,     hair_top + hair_ext // 3],
    ], dtype=np.int32)

    all_pts = cv2.convexHull(np.vstack([hull_pts, extra]), returnPoints=True).reshape(-1, 1, 2)
    mask = np.zeros((_SIZE_IN, _SIZE_IN), np.uint8)
    cv2.fillConvexPoly(mask, all_pts, 255)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (91, 91), 0)
    return mask


def _build_eye_exclusion(lms_aligned):
    """Göz+gözlük bölgesini blend maskesinden çıkarır."""
    excl = np.zeros((_SIZE_IN, _SIZE_IN), np.uint8)
    for ids in [_MP_L_EYE_CONTOUR, _MP_R_EYE_CONTOUR]:
        pts = np.array(
            [[int(np.clip(lms_aligned[i][0], 0, _SIZE_IN - 1)),
              int(np.clip(lms_aligned[i][1], 0, _SIZE_IN - 1))]
             for i in ids if i < len(lms_aligned)],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            cv2.fillConvexPoly(excl, hull, 255)
    excl = cv2.dilate(excl, np.ones((15, 15), np.uint8), iterations=2)
    excl = cv2.GaussianBlur(excl, (31, 31), 0)
    return excl


def _match_color(sam, orig, face_mask):
    """LAB uzayında renk eşleştirme — yüz merkezi istatistiği.

    L kanalı (parlaklık): sadece mean kaydır, std koruma → SAM'ın kırışık/yaşlanma
    dokusunu/kontrastını ezmiyor.  A/B kanalları: tam normalizasyon (sarı/turuncu düzeltir).
    """
    sam_lab  = cv2.cvtColor(sam,  cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask_bool = face_mask > 128
    for c in range(3):
        s = sam_lab[:, :, c][mask_bool]
        r = orig_lab[:, :, c][mask_bool]
        if len(s) < 100:
            continue
        if c == 0:
            # L: sadece mean kaydır — SAM'ın yaşlanma dokusunu/kontrastını koru
            sam_lab[:, :, c] = sam_lab[:, :, c] - s.mean() + r.mean()
        else:
            # A, B: tam normalizasyon — sarı/turuncu ton düzelt
            if s.std() < 1e-3:
                continue
            sam_lab[:, :, c] = (sam_lab[:, :, c] - s.mean()) / s.std() * r.std() + r.mean()
    return cv2.cvtColor(np.clip(sam_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _detect_hair_mask(image: np.ndarray,
                       landmarks: list[tuple[int, int]]) -> np.ndarray:
    """HSV koyu piksel tespiti — morfoloji yok, yumuşak bölge sınırı.

    Morfolojik işlem TAMAMEN kaldırıldı: CLOSE/DILATE dikdörtgen bulut yaratır.
    Bunun yerine bölge maskesi Gaussian blur ile yumuşatılır →
    saçın gerçek piksel sınırı korunur, sınır kademeli solar.
    """
    h, w = image.shape[:2]
    lms = np.array(landmarks, dtype=np.float32)

    forehead = lms[10] if 10 < len(lms) else lms.mean(axis=0)
    fy = int(np.clip(forehead[1], 0, h - 1))

    x_min = int(np.clip(lms[:, 0].min(), 0, w - 1))
    x_max = int(np.clip(lms[:, 0].max(), 0, w - 1))
    face_w = x_max - x_min
    chin_y = int(np.clip(lms[152][1] if 152 < len(lms) else h * 0.75, 0, h - 1))
    face_h = max(chin_y - fy, 60)

    temple_bottom = min(fy + int(face_h * 0.40), h - 1)
    pad   = int(face_w * 0.55)
    left  = max(x_min - pad, 0)
    right = min(x_max + pad, w - 1)

    # ELİPS tabanlı bölge — dikdörtgen köşe yok
    hair_region = np.zeros((h, w), np.uint8)

    # Üst saçlar: forehead merkezinde yarı-elips (180-360° = üst yarı)
    # top_hh minimum 100px — yakın çekimde fy küçük kalsa bile saç bölgesi yeterince büyük
    top_hw = int(face_w * 0.72)
    top_hh = max(fy + 8, 100)
    cv2.ellipse(hair_region, (int(forehead[0]), fy),
                (top_hw, top_hh), 0, 180, 360, 255, -1)

    # Yan saçlar (şakak): yüzün sağ/sol dışında küçük elipsler
    temple_h  = int(face_h * 0.38)
    side_w    = int(face_w * 0.32)
    l_center  = (max(x_min - side_w // 3, 0), fy + temple_h // 2)
    r_center  = (min(x_max + side_w // 3, w - 1), fy + temple_h // 2)
    cv2.ellipse(hair_region, l_center, (side_w, temple_h // 2 + 4), 0, 0, 360, 255, -1)
    cv2.ellipse(hair_region, r_center, (side_w, temple_h // 2 + 4), 0, 0, 360, 255, -1)

    # Yüz iç bölgesini doğru şekilde temizle: dikdörtgen yerine landmark konveks hull
    # → temple elipslerinin yanaklara taştığı bölgeyi kesin maskeler
    face_hull_pts = cv2.convexHull(
        np.clip(lms.astype(np.int32), [0, 0], [w - 1, h - 1])
    )
    cv2.fillConvexPoly(hair_region, face_hull_pts, 0)

    # Sınırları yumuşat — elips zaten köşesiz, küçük blur yeterli
    soft_region = cv2.GaussianBlur(hair_region.astype(np.float32), (21, 21), 0) / 255.0

    # HSV koyu piksel tespiti — sabit ve tutarlı eşik
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    hair_px = ((v < 85) | ((v < 120) & (s > 40))).astype(np.uint8) * 255

    # Saç kenarı yumuşatma — binary mask'in hard edge'ini giderir
    hair_px_soft = cv2.GaussianBlur(hair_px, (13, 13), 0).astype(np.float32) / 255.0

    # Maske = yumuşak saç piksel × yumuşak elips bölge
    hair_mask = (hair_px_soft * soft_region * 255.0).clip(0, 255).astype(np.uint8)
    return hair_mask


def _gray_hair_on_original(result: np.ndarray,
                            landmarks: list[tuple[int, int]],
                            gray_strength: float = 0.58) -> np.ndarray:
    """HSV V/S kanallarıyla gerçekçi saç grileştirme.

    V artırılır (parlaklık), S azaltılır (renk kaldırılır).
    Her saç teli kendi dokusunu korur — uniform gri değil, doğal gri.
    """
    hair_mask = _detect_hair_mask(result, landmarks)
    if hair_mask.max() < 10:
        return result

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

    # V kanalı: koyu saç (V≈35) → hedef 172'ye çek
    # 35 + 0.52*(172-35) = 35 + 71 = 106  (orta gri, doğal) ✓
    hsv[:, :, 2] = hsv[:, :, 2] + gray_strength * (172.0 - hsv[:, :, 2])

    # S kanalı: doygunluğu azalt (rengi kaldır)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - gray_strength * 0.85)

    grayed = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    alpha = hair_mask.astype(np.float32)[:, :, None] / 255.0
    return (grayed.astype(np.float32) * alpha + result.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)


def _paste_back(orig, result_256, M_inv, face_mask_256):
    h, w = orig.shape[:2]
    warped    = cv2.warpAffine(result_256,    M_inv, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask_orig = cv2.warpAffine(face_mask_256, M_inv, (w, h))
    alpha = mask_orig.astype(np.float32)[:, :, None] / 255.0
    return (warped.astype(np.float32) * alpha + orig.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)


def _load_sam():
    global _net, _opts
    if _net is not None:
        return _net, _opts
    if not os.path.exists(_CKPT):
        raise FileNotFoundError(f"SAM model bulunamadı: {_CKPT}")
    if not os.path.isdir(_VENDOR):
        raise RuntimeError(f"SAM repo bulunamadı: {_VENDOR}")
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)

    import torch
    from models.psp import pSp
    from argparse import Namespace

    ckpt  = torch.load(_CKPT, map_location="cpu", weights_only=False)
    opts  = Namespace(**ckpt["opts"])
    opts.device          = "cpu"
    opts.checkpoint_path = _CKPT
    net = pSp(opts)
    net.eval()
    _net, _opts = net, opts
    return _net, _opts


def apply_sam_aging(
    image_np: np.ndarray,
    landmarks: list[tuple[int, int]],
    target_age: float = 65.0,
    intensity: float = 1.0,
) -> np.ndarray:
    import torch
    import torchvision.transforms as T
    from PIL import Image as PILImage

    net, opts = _load_sam()
    from datasets.augmentations import AgeTransformer

    # 1. FFHQ hizalama
    aligned, M_fwd, M_inv = _align_ffhq(image_np, landmarks)
    orig_lms_aligned = _lms_to_aligned(landmarks, M_fwd)

    # 2. BGR → RGB PIL → tensor [-1, 1]
    pil_img   = PILImage.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.Resize((_SIZE_IN, _SIZE_IN)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_t = transform(pil_img)

    # 3. Yaş kanalı ekle
    age_transformer = AgeTransformer(target_age=int(target_age))
    input_with_age  = age_transformer(img_t).unsqueeze(0).float()

    # 4. SAM inference → 1024×1024 → 256×256
    with torch.no_grad():
        result_batch = net(input_with_age, randomize_noise=False, resize=False)

    out_np  = result_batch[0].permute(1, 2, 0).cpu().numpy()
    out_np  = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    out_256 = cv2.resize(out_bgr, (_SIZE_IN, _SIZE_IN), interpolation=cv2.INTER_LINEAR)

    # 5. Korrektif landmark hizalaması
    out_256 = _correct_sam_alignment(out_256, orig_lms_aligned)

    # 6. Maskeler
    color_mask = _build_face_mask(orig_lms_aligned)
    blend_mask = _build_blend_mask(orig_lms_aligned)
    eye_excl   = _build_eye_exclusion(orig_lms_aligned)

    # Göz/gözlük bölgesini blend maskesinden çıkar
    blend_mask = np.clip(
        blend_mask.astype(np.int32) - eye_excl.astype(np.int32), 0, 255
    ).astype(np.uint8)

    # 7. Renk eşleştirme
    out_256 = _match_color(out_256, aligned, color_mask)

    # 8. Intensity blend
    if intensity < 1.0:
        out_256 = cv2.addWeighted(aligned, 1.0 - intensity, out_256, intensity, 0)

    # 9. Orijinale yapıştır
    result = _paste_back(image_np, out_256, M_inv, blend_mask)

    # 10. Saç grileştirme — orijinal boyutta, alın üstünde kesin sınırlı
    result = _gray_hair_on_original(result, landmarks)

    return result


def is_available() -> bool:
    return os.path.exists(_CKPT) and os.path.isdir(_VENDOR)
