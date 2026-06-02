"""
apply_hair_color() birim testleri.

Çalıştır:
    cd python_service
    /opt/homebrew/bin/python3.11 -m pytest tests/test_hair_color.py -v
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock, patch

# --- Mock ağır bağımlılıkları import öncesinde ---
# mediapipe venv'de var, ama pytest sistem python ile çalışıyorsa bulamaz.
# Testler saf numpy/cv2 mantığını test eder; MediaPipe segmenter mock'lanır.
def _mock_mediapipe():
    mp_mock = MagicMock()
    mp_mock.Image = MagicMock()
    mp_mock.ImageFormat = MagicMock()
    mp_mock.ImageFormat.SRGB = 0
    mp_mock.tasks = MagicMock()
    mp_mock.tasks.python = MagicMock()
    mp_mock.tasks.python.vision = MagicMock()
    sys.modules.setdefault('mediapipe', mp_mock)
    sys.modules.setdefault('mediapipe.tasks', mp_mock.tasks)
    sys.modules.setdefault('mediapipe.tasks.python', mp_mock.tasks.python)
    sys.modules.setdefault('mediapipe.tasks.python.vision', mp_mock.tasks.python.vision)

_mock_mediapipe()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

# model_utils da mediapipe dosyaları indirir; mock'la
with patch('utils.model_utils.get_model_path', return_value='/tmp/fake.task'):
    from modules.hair_segmentation import apply_hair_color


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _solid(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    """Tek renkli test görseli oluştur."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def _solid_with_mask(
    h: int = 64,
    w: int = 64,
    hair_bgr: tuple[int, int, int] = (20, 15, 10),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Üst yarısı 'saç', alt yarısı 'deri' renginde yapay görsel ve maske döner.
    Maske 0/1 değil, 0-1 float32 — gerçek kullanıma uygun.
    """
    img = _solid(h, w, (200, 170, 150))   # deri tonu
    img[: h // 2] = hair_bgr              # üst yarı siyah-kahve saç
    mask = np.zeros((h, w), dtype=np.float32)
    mask[: h // 2] = 1.0                  # saç maskesi
    mask = mask * 0.9 + 0.05              # 0.05..0.95 — yumuşak kenar simülasyonu
    return img, mask


# ---------------------------------------------------------------------------
# 1. Çıktı şekli ve dtype
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_returns_same_shape(self):
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#6b3a2a", intensity=0.85, hair_mask=mask)
        assert out.shape == img.shape, "Çıktı şekli değişmemeli"

    def test_returns_uint8(self):
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#6b3a2a", intensity=0.85, hair_mask=mask)
        assert out.dtype == np.uint8, "Çıktı uint8 olmalı"

    def test_no_pixel_overflow(self):
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#FF69B4", intensity=1.0, hair_mask=mask)
        assert out.max() <= 255 and out.min() >= 0, "Piksel değerleri 0-255 aralığında olmalı"

    def test_larger_image(self):
        img, mask = _solid_with_mask(h=256, w=192)
        out = apply_hair_color(img, "#1565C0", intensity=0.7, hair_mask=mask)
        assert out.shape == (256, 192, 3)


# ---------------------------------------------------------------------------
# 2. Kimlik / değişmeme garantileri
# ---------------------------------------------------------------------------

class TestIdentityGuarantees:

    def test_zero_intensity_unchanged(self):
        """intensity=0 → görsel değişmemeli."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#A020F0", intensity=0.0, hair_mask=mask)
        np.testing.assert_array_equal(out, img)

    def test_near_zero_intensity_unchanged(self):
        """intensity < 0.02 → erken çıkış, kopya döner."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#A020F0", intensity=0.01, hair_mask=mask)
        np.testing.assert_array_equal(out, img)

    def test_empty_mask_unchanged(self):
        """Sıfır maske (saç yok) → görsel değişmemeli."""
        img = _solid(64, 64, (180, 140, 110))
        empty_mask = np.zeros((64, 64), dtype=np.float32)
        out = apply_hair_color(img, "#FF69B4", intensity=1.0, hair_mask=empty_mask)
        np.testing.assert_array_equal(out, img)

    def test_non_hair_region_preserved(self):
        """Maskesiz bölge (deri) değişmemeli."""
        img, mask = _solid_with_mask(h=64, w=64)
        out = apply_hair_color(img, "#1565C0", intensity=1.0, hair_mask=mask)
        # Alt yarı maske ≈ 0.05 → piksel farkı çok küçük olmalı
        bottom_orig = img[40:, :].astype(np.float32)
        bottom_out = out[40:, :].astype(np.float32)
        mean_diff = float(np.mean(np.abs(bottom_out - bottom_orig)))
        assert mean_diff < 15.0, f"Maske dışı bölge çok değişti: mean_diff={mean_diff:.2f}"


# ---------------------------------------------------------------------------
# 3. Renk değişimi gerçekleşiyor mu?
# ---------------------------------------------------------------------------

class TestColorChange:

    def test_color_applied_in_hair_region(self):
        """Saç bölgesi hedef renge doğru kaymalı."""
        img, mask = _solid_with_mask(hair_bgr=(20, 20, 20))   # çok koyu saç
        out = apply_hair_color(img, "#1565C0", intensity=1.0, hair_mask=mask)  # mavi
        # Mavi kanal üst yarıda artmalı
        blue_before = float(np.mean(img[:32, :, 0]))
        blue_after = float(np.mean(out[:32, :, 0]))
        assert blue_after > blue_before, "Saç bölgesinde mavi kanal artmadı"

    def test_natural_brown_on_dark_hair(self):
        """Kahverengi hedef → sonuç orjinal siyahtan farklı olmalı."""
        img, mask = _solid_with_mask(hair_bgr=(10, 10, 10))
        out = apply_hair_color(img, "#6b3a2a", intensity=0.85, hair_mask=mask)
        diff = float(np.mean(np.abs(out[:32].astype(float) - img[:32].astype(float))))
        assert diff > 3.0, "Kahve uygulamasında saç bölgesi değişmedi"

    def test_vivid_pink_on_dark_hair(self):
        """Canlı pembe → saç bölgesi kırmızı/yeşil oranı değişmeli."""
        img, mask = _solid_with_mask(hair_bgr=(15, 15, 15))
        out = apply_hair_color(img, "#FF69B4", intensity=1.0, hair_mask=mask)
        # Pembe: R yüksek, B yüksek, G orta
        red_after = float(np.mean(out[:32, :, 2]))
        assert red_after > 30, f"Pembe uygulamada kırmızı kanalı yeterince artmadı: {red_after:.1f}"

    def test_platinum_on_dark_hair_lightens(self):
        """Platin sarı → saç alanını belirgin şekilde aydınlatmalı."""
        img, mask = _solid_with_mask(hair_bgr=(10, 10, 10))
        out = apply_hair_color(img, "#F5EDD6", intensity=0.85, hair_mask=mask)
        mean_before = float(np.mean(img[:32]))
        mean_after = float(np.mean(out[:32]))
        assert mean_after > mean_before + 5, "Platin rengi saçı aydınlatmadı"


# ---------------------------------------------------------------------------
# 4. Highlight koruması
# ---------------------------------------------------------------------------

class TestHighlightProtection:

    def test_very_bright_pixels_mostly_preserved(self):
        """
        Neredeyse beyaz piksellar (highlight) hedef rengin A/B kanallarıyla
        çok az değişmeli — parlaklık korunuyor.
        """
        img = np.full((64, 64, 3), 240, dtype=np.uint8)   # çok parlak
        mask = np.ones((64, 64), dtype=np.float32)
        out = apply_hair_color(img, "#1a1a1a", intensity=1.0, hair_mask=mask,
                               preserve_highlights=True)
        mean_orig = float(np.mean(img))
        mean_out = float(np.mean(out))
        # Highlight koruması: parlak alanlar çok kararmamalı
        assert mean_out > mean_orig * 0.65, (
            f"Highlight koruması başarısız: {mean_out:.1f} < {mean_orig * 0.65:.1f}"
        )

    def test_highlight_off_applies_more_color(self):
        """preserve_highlights=False → daha agresif renk değişimi."""
        img = np.full((64, 64, 3), 240, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        out_protected = apply_hair_color(img, "#1a1a1a", 1.0, mask, preserve_highlights=True)
        out_raw = apply_hair_color(img, "#1a1a1a", 1.0, mask, preserve_highlights=False)
        mean_protected = float(np.mean(out_protected))
        mean_raw = float(np.mean(out_raw))
        assert mean_raw <= mean_protected, "Korumasız mod korumalıdan daha az etkilememeli"


# ---------------------------------------------------------------------------
# 5. Hex renk validasyonu
# ---------------------------------------------------------------------------

class TestHexValidation:

    def test_without_hash_prefix(self):
        """'#' prefix olmaksızın verilen hex kabul edilmeli."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "A020F0", intensity=0.5, hair_mask=mask)
        assert out.shape == img.shape

    def test_three_digit_hex(self):
        """3 haneli hex (#RGB → #RRGGBB) kabul edilmeli."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#F00", intensity=0.5, hair_mask=mask)
        assert out.shape == img.shape

    def test_invalid_hex_raises(self):
        """Geçersiz hex ValueError fırlatmalı."""
        img, mask = _solid_with_mask()
        with pytest.raises(ValueError):
            apply_hair_color(img, "#GGGGGG", intensity=0.5, hair_mask=mask)

    def test_empty_hex_raises(self):
        """Boş hex string ValueError fırlatmalı."""
        img, mask = _solid_with_mask()
        with pytest.raises(ValueError):
            apply_hair_color(img, "", intensity=0.5, hair_mask=mask)


# ---------------------------------------------------------------------------
# 6. Intensity aralığı kırpma
# ---------------------------------------------------------------------------

class TestIntensityClipping:

    def test_intensity_above_1_clamped(self):
        """intensity > 1.0 → 1.0 olarak kırpılmalı, hata fırlatmamalı."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#6b3a2a", intensity=5.0, hair_mask=mask)
        assert out.shape == img.shape

    def test_intensity_below_0_unchanged(self):
        """intensity < 0.0 → değişmemeli."""
        img, mask = _solid_with_mask()
        out = apply_hair_color(img, "#6b3a2a", intensity=-1.0, hair_mask=mask)
        np.testing.assert_array_equal(out, img)
