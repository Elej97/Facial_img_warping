from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.makeup_service import apply_makeup


def _face_image(h: int = 256, w: int = 256) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (178, 148, 126)
    return img


def _landmarks() -> list[tuple[int, int]]:
    lm = [(128, 128)] * 468

    points = {
        61: (88, 160), 185: (98, 150), 40: (112, 145), 39: (124, 144), 37: (134, 145),
        0: (144, 148), 267: (156, 145), 269: (168, 144), 270: (182, 150), 409: (194, 160),
        291: (188, 172), 375: (174, 181), 321: (156, 185), 405: (138, 186), 314: (120, 184),
        17: (104, 178), 84: (92, 170), 181: (88, 160), 91: (96, 166), 146: (102, 171),
        78: (108, 162), 191: (118, 156), 80: (132, 154), 81: (146, 154), 82: (160, 156),
        13: (174, 162), 312: (166, 168), 311: (152, 170), 310: (138, 170), 415: (124, 168),
        308: (112, 164), 324: (121, 174), 318: (137, 178), 402: (153, 178), 317: (169, 174),
        14: (178, 166), 87: (160, 178), 178: (143, 181), 88: (126, 178), 95: (110, 168),
        234: (62, 132), 93: (70, 156), 132: (82, 190), 58: (100, 205),
        454: (194, 132), 323: (186, 156), 361: (174, 190), 288: (156, 205),
        117: (82, 126), 118: (94, 120), 50: (102, 138), 205: (82, 154),
        347: (174, 126), 346: (162, 120), 280: (154, 138), 425: (174, 154),
        33: (82, 104), 7: (92, 98), 163: (106, 96), 144: (120, 98), 145: (130, 104),
        153: (120, 110), 154: (106, 113), 155: (92, 111), 133: (78, 106),
        173: (94, 96), 157: (108, 94), 158: (120, 96), 159: (128, 101), 160: (116, 100),
        161: (102, 100), 246: (88, 101),
        362: (178, 106), 382: (164, 98), 381: (150, 96), 380: (136, 98), 374: (126, 104),
        373: (136, 110), 390: (150, 113), 249: (164, 111), 263: (182, 104),
        466: (168, 96), 388: (154, 94), 387: (142, 96), 386: (130, 101), 385: (142, 100),
        384: (156, 100), 398: (172, 101),
        46: (74, 88), 53: (88, 82), 52: (102, 80), 65: (116, 82), 55: (130, 88),
        70: (84, 84), 63: (96, 81), 105: (108, 81), 66: (120, 84), 107: (132, 90),
        276: (126, 88), 283: (140, 82), 282: (154, 80), 295: (168, 82), 285: (182, 88),
        300: (124, 90), 293: (136, 84), 334: (148, 81), 296: (160, 81), 336: (172, 84),
    }
    for idx, point in points.items():
        lm[idx] = point
    return lm


@pytest.mark.parametrize("region", ["lip", "cheek", "brow", "lash", "eye", "teeth"])
def test_makeup_returns_same_shape_and_dtype(region: str):
    img = _face_image()
    out = apply_makeup(img, _landmarks(), region, "#D45A73", 0.65)["result_image"]
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_lipstick_changes_lip_area():
    img = _face_image()
    out = apply_makeup(img, _landmarks(), "lip", "#A83253", 0.9)["result_image"]
    mouth_before = img[145:186, 88:194].astype(np.float32)
    mouth_after = out[145:186, 88:194].astype(np.float32)
    assert float(np.mean(np.abs(mouth_after - mouth_before))) > 2.0


def test_zero_intensity_keeps_lip_mostly_stable():
    img = _face_image()
    out = apply_makeup(img, _landmarks(), "lip", "#A83253", 0.0)["result_image"]
    assert float(np.mean(np.abs(out.astype(np.float32) - img.astype(np.float32)))) < 2.0


def test_invalid_region_raises():
    with pytest.raises(ValueError):
        apply_makeup(_face_image(), _landmarks(), "glitter", "#FFFFFF", 0.5)


def test_invalid_hex_raises():
    with pytest.raises(ValueError):
        apply_makeup(_face_image(), _landmarks(), "lip", "#GGGGGG", 0.5)
