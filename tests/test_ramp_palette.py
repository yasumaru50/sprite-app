import numpy as np
from PIL import Image

from retrocore.color import luminance, rgb_to_hsv
from retrocore.palette import PC98_DEFAULT_16, extract_ramp_palette, infer_ramps


def three_ramp_image():
    """Left: red shading ramp, middle: blue ramp, right: grey ramp."""
    h, w = 64, 96
    t = np.linspace(0.15, 1.0, h)[:, None]
    red = np.stack([255 * t, 40 * t, 30 * t], -1).repeat(32, 1)
    blue = np.stack([30 * t, 70 * t, 255 * t], -1).repeat(32, 1)
    grey = np.stack([230 * t] * 3, -1).repeat(32, 1)
    return Image.fromarray(np.concatenate([red, blue, grey], 1).astype(np.uint8))


def test_ramps_follow_hues():
    rp = extract_ramp_palette(three_ramp_image(), 12, snap_4096=False)
    assert len(rp.ramps) == 3
    assert sorted(i for r in rp.ramps for i in r) == list(range(len(rp.palette)))
    lum = luminance(rp.palette)
    hsv = rgb_to_hsv(rp.palette)
    kinds = set()
    for r in rp.ramps:
        assert len(r) >= 2
        assert np.all(np.diff(lum[r]) > 0)           # dark -> light
        c = rp.palette[r].astype(int)
        if (c.max(1) - c.min(1)).max() < 28:
            kinds.add("grey")
        else:
            h = hsv[r, 0]
            kinds.add("red" if np.all((h < 0.1) | (h > 0.9)) else "blue" if np.all(np.abs(h - 0.63) < 0.08) else "?")
    assert kinds == {"red", "blue", "grey"}


def test_budget_steps_and_snap():
    img = three_ramp_image()
    rp = extract_ramp_palette(img, 16, snap_4096=True)
    assert len(rp.palette) <= 16
    assert (rp.palette % 17 == 0).all()
    rp = extract_ramp_palette(img, 16, n_ramps=3, steps=3, snap_4096=False)
    assert all(len(r) <= 3 for r in rp.ramps) and len(rp.palette) <= 9
    rp = extract_ramp_palette(img, 8, n_ramps=2, snap_4096=False)
    assert len(rp.ramps) <= 2 and len(rp.palette) <= 8


def test_single_colour_image():
    img = Image.new("RGB", (8, 8), (10, 200, 30))
    rp = extract_ramp_palette(img, 16)
    assert len(rp.palette) >= 1 and rp.ramps


def test_infer_ramps_covers_palette():
    ramps = infer_ramps(PC98_DEFAULT_16)
    flat = sorted(i for r in ramps for i in r)
    assert flat == list(range(16))
    lum = luminance(PC98_DEFAULT_16)
    assert all(np.all(np.diff(lum[r]) >= 0) for r in ramps)
    # Bright and dim red share a ramp.
    red = next(r for r in ramps if 2 in r)
    assert 10 in red


def small_accents_image():
    """Mostly a blue gradient, with a small face (skin) and a small vivid red tie."""
    h, w = 100, 160
    t = np.linspace(0.3, 1.0, w)[None, :, None]
    img = (np.array([40, 70, 200], float) * t).repeat(h, 0)
    img[10:25, 10:30] = (250, 214, 190)           # face light   (~1.9 %)
    img[25:30, 10:30] = (214, 160, 140)           # face shadow  (~0.6 %)
    img[60:72, 120:130] = (200, 20, 30)           # red tie      (~0.75 %)
    return Image.fromarray(img.astype(np.uint8))


def _has_color(pal, target, tol=40):
    return (np.abs(pal.astype(int) - np.array(target)).sum(1) <= tol).any()


def test_small_hues_get_ramps_only_with_new_options():
    img = small_accents_image()
    old = extract_ramp_palette(img, 16, n_ramps=1, snap_4096=False)
    new = extract_ramp_palette(img, 16, n_ramps=1, snap_4096=False,
                               hue_separation=True, reserve_skin=True)
    assert not _has_color(old.palette, (200, 20, 30))
    assert _has_color(new.palette, (200, 20, 30))          # red tie kept
    assert _has_color(new.palette, (250, 214, 190))        # face kept
    hsv = rgb_to_hsv(new.palette)
    skin = [r for r in new.ramps if all(hsv[i, 0] < 45 / 360 and 0.1 <= hsv[i, 1] <= 0.6 for i in r)]
    assert skin and max(len(r) for r in skin) >= 2
    assert len(new.palette) <= 16


def test_flat_area_is_not_split_into_close_shades():
    img = Image.new("RGB", (40, 40), (240, 200, 170))
    noise = np.random.default_rng(0).integers(-3, 4, (40, 40, 3))
    img = Image.fromarray(np.clip(np.asarray(img, int) + noise, 0, 255).astype(np.uint8))
    rp = extract_ramp_palette(img, 16, snap_4096=False, reserve_skin=True)
    lum = luminance(rp.palette)
    for r in rp.ramps:
        assert np.all(np.diff(lum[r]) >= 9.9)
