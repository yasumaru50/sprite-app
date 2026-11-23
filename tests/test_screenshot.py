import numpy as np
from PIL import Image

from retrocore.screenshot import detect_grid, normalize_screenshot, pure_pixel_mask
from synth_cg import H, PAL, W, make_cg, rgb


def _palette_matches(pal, ref, tol=18):
    d = np.sqrt(((pal[:, None].astype(float) - ref[None].astype(float)) ** 2).sum(2))
    return d.min(0) <= tol   # every reference colour found


def test_native_image_is_exact():
    idx, pal, _ = make_cg()
    n = normalize_screenshot(Image.fromarray(rgb(idx)))
    assert n.grid.method == "native"
    assert n.indices.shape == (H, W)
    assert np.array_equal(n.rgb(), rgb(idx))


def test_integer_nearest_upscale():
    img = Image.fromarray(rgb()).resize((W * 3, H * 2), Image.Resampling.NEAREST)
    g = detect_grid(img)
    assert (g.scale_x, g.scale_y) == (3, 2) and g.method == "integer"
    n = normalize_screenshot(img)
    assert np.array_equal(n.rgb(), rgb())


def busy_cg():
    """make_cg() plus random native-pixel detail (realistic edge density)."""
    idx, _, _ = make_cg()
    rng = np.random.default_rng(1)
    idx = idx.copy()
    for _ in range(400):
        y, x = rng.integers(0, H - 3), rng.integers(0, W - 3)
        idx[y: y + rng.integers(1, 4), x: x + rng.integers(1, 4)] = rng.integers(0, 8)
    return idx


def test_fractional_scaled_screenshot():
    """×1.6 area-sampled enlargement (like emulator "smooth" scaling) plus noise.

    Every output pixel averages the native pixels it covers, so pixels that
    straddle a native boundary are blends — the situation of the reference
    screenshots.
    """
    rng = np.random.default_rng(0)
    big = Image.fromarray(PAL[busy_cg()]).resize((320, 192), Image.Resampling.BOX)
    noisy = np.clip(np.asarray(big, np.int16) + rng.integers(-2, 3, (192, 320, 3)), 0, 255)
    img = Image.fromarray(noisy.astype(np.uint8))
    g = detect_grid(img)
    assert g.method == "spectral"
    assert abs(g.scale_x - 1.6) < 0.01 and abs(g.scale_y - 1.6) < 0.01
    assert abs(g.native_width - W) <= 2 and abs(g.native_height - H) <= 2
    n = normalize_screenshot(img)
    assert _palette_matches(n.palette, PAL).all()
    assert len(n.palette) <= 12
    # Most native pixels come back with their true colour.
    truth = PAL[busy_cg()].astype(int)
    h, w = n.indices.shape
    got = n.rgb().astype(int)
    oy, ox = int(round((g.offset_y - 0.3) / 1.6)), int(round((g.offset_x - 0.3) / 1.6))
    agree = (np.abs(got[: h - 2] - truth[oy: oy + h - 2, ox: ox + w]).max(2) <= 17).mean()
    assert agree > 0.8


def test_pure_pixel_mask():
    a = np.zeros((5, 5, 3), np.uint8)
    a[2, 2] = 200
    m = pure_pixel_mask(a)
    assert not m[1:4, 1:4].any() and not m[0].any()
