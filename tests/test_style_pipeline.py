import time

import numpy as np
from PIL import Image

from retrocore import ConvertSettings, convert
from retrocore.palette import PC98_DEFAULT_16


def picture(w=800, h=500):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([90 + 80 * y / h, 140 + 70 * y / h, 230 - 30 * y / h], -1)
    d = np.hypot(x - w / 2, y - h / 2)
    img[d < 150] = (240, 200, 170)
    img[(d >= 150) & (d < 155)] = (20, 10, 20)
    rng = np.random.default_rng(0)
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_default_mode_is_unchanged():
    assert ConvertSettings().mode == "normal"
    s = ConvertSettings(preset="カスタム", custom_width=80, custom_height=50)
    r = convert(picture(), s)
    assert "dither_mask" not in r.info


def test_style_mode_fast_and_valid():
    s = ConvertSettings(mode="style", style_outline=True)
    t = time.perf_counter()
    r = convert(picture(), s)
    assert time.perf_counter() - t < 5.0
    assert r.image.mode == "P" and r.image.size == (640, 400)
    assert r.used_colors <= 16 and len(r.palette) <= 16
    assert (r.palette % 17 == 0).all()
    assert 0 < r.info["dither_coverage"] < 1
    assert r.info["outline_coverage"] > 0


def test_style_mode_flat_face_is_solid():
    r = convert(picture(), ConvertSettings(mode="style", style_pattern="checker"))
    idx = np.asarray(r.image)
    face = idx[160:240, 280:360]           # inside the flat disc
    assert len(np.unique(face)) == 1


def test_style_mode_with_preset_and_forced_palette():
    s = ConvertSettings(mode="style", palette_source="preset:PC-98 標準16色",
                        preset="カスタム", custom_width=160, custom_height=100)
    r = convert(picture(), s)
    assert {tuple(c) for c in r.palette} <= {tuple(c) for c in PC98_DEFAULT_16}
    r = convert(picture(), ConvertSettings(mode="style", style_pattern="none"), palette=PC98_DEFAULT_16)
    assert r.info["dither_coverage"] == 0


def test_stage1_cache_gives_identical_results():
    cache = {}
    base = ConvertSettings(mode="style", preset="カスタム", custom_width=160, custom_height=100,
                           style_outline=True)
    for grad in (0.6, 1.5, 0.6):
        s = ConvertSettings(**{**base.__dict__, "style_grad_low": grad})
        a = np.asarray(convert(picture(), s, cache=cache).image)
        b = np.asarray(convert(picture(), s).image)
        assert np.array_equal(a, b)
    assert len(cache) == 1
