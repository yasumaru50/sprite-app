import numpy as np

from retrocore.dither import (
    gradient_mask,
    indices_to_image,
    pattern_threshold,
    selective_pattern_dither,
)


def scene(h=64, w=128):
    """Left half: smooth horizontal gradient; right half: flat colour + noise block."""
    img = np.zeros((h, w, 3), np.float32)
    ramp = np.linspace(40, 220, w // 2)
    img[:, : w // 2] = np.stack([ramp, ramp * 0.5, ramp * 0.3], -1)[None]
    img[:, w // 2:] = (90, 140, 200)
    rng = np.random.default_rng(0)
    img[: h // 2, w // 2 + 16:] += rng.normal(0, 40, (h // 2, w // 2 - 16, 3))
    return np.clip(img, 0, 255)


def test_gradient_mask_selects_gradients_only():
    img = scene()
    m = gradient_mask(img)
    h, w = m.shape
    assert m[8:-8, 8: w // 2 - 8].mean() > 0.9       # gradient
    assert m[h // 2 + 4:, w // 2 + 6:].mean() == 0  # flat
    assert m[4: h // 2 - 6, w // 2 + 24:].mean() < 0.1  # noise / texture


def test_checker_between_adjacent_ramp_shades():
    pal = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0], [0, 0, 200]], np.uint8)
    ramps = [[0, 1, 2], [3]]
    img = np.full((8, 8, 3), (150, 0, 0), np.float32)    # halfway 1..2
    idx = selective_pattern_dither(img, pal, ramps, np.ones((8, 8), bool), "checker")
    assert set(np.unique(idx)) == {1, 2}
    yy, xx = np.mgrid[0:8, 0:8]
    assert np.array_equal(idx == 2, (yy + xx) % 2 == 1) or np.array_equal(idx == 2, (yy + xx) % 2 == 0)
    # Outside the mask: solid nearest colour.
    idx = selective_pattern_dither(img, pal, ramps, np.zeros((8, 8), bool), "checker")
    assert len(np.unique(idx)) == 1


def test_tile2_levels_and_strength():
    pal = np.array([[0, 0, 0], [200, 200, 200]], np.uint8)
    img = np.full((4, 4, 3), 50, np.float32)              # 25 % of the way
    idx = selective_pattern_dither(img, pal, [[0, 1]], np.ones((4, 4), bool), "tile2")
    assert (idx == 1).sum() == 4                           # one per 2x2 tile
    idx = selective_pattern_dither(img, pal, [[0, 1]], np.ones((4, 4), bool), "tile2", strength=0)
    assert (idx == 1).sum() == 0


def test_only_ramp_neighbours_are_mixed():
    img = scene()
    pal = np.array([[30, 15, 9], [90, 45, 27], [160, 80, 48], [230, 115, 69], [90, 140, 200]], np.uint8)
    ramps = [[0, 1, 2, 3], [4]]
    m = gradient_mask(img)
    idx = selective_pattern_dither(img, pal, ramps, m, "bayer4")
    # Every horizontal neighbour pair inside the gradient differs by <= 1 ramp step.
    seg = idx[8:-8, 4:60].astype(int)
    assert np.abs(np.diff(seg, axis=1)).max() <= 1
    out = indices_to_image(idx, pal)
    assert out.mode == "P" and out.size == (128, 64)


def test_pattern_threshold_shapes():
    for p in ("checker", "tile2", "bayer4"):
        t = pattern_threshold(p, 5, 7)
        assert t.shape == (5, 7) and (t > 0).all() and (t < 1).all()


def test_one_shade_ramp_is_not_mixed_with_other_ramps():
    pal = np.array([[240, 200, 170], [60, 80, 200], [120, 160, 230]], np.uint8)
    ramps = [[0], [1, 2]]                    # skin has a single shade
    img = np.full((6, 6, 3), (230, 195, 175), np.float32)
    idx = selective_pattern_dither(img, pal, ramps, np.ones((6, 6), bool), "checker")
    assert (idx == 0).all()
