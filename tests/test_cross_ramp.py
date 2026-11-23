"""Dither pairs across ramps (opt-in, ``style_cross_ramp``)."""

import numpy as np
from PIL import Image

from retrocore import ConvertSettings, convert
from retrocore.analyzer import analyze_image, build_profile
from retrocore.dither import cross_ramp_segments, selective_pattern_dither
from retrocore.palette import _from_4bit
from retrocore.style import STYLE_PRESETS, apply_overrides, cross_ramp_settings, profile_to_settings

# 0 skin (single shade), 1-2 blue ramp, 3 white, 4 dark red, 5 yellow (light, 50° from skin)
PAL = _from_4bit([(15, 13, 11), (3, 5, 12), (8, 11, 15), (15, 15, 15), (6, 1, 1), (15, 15, 5)])
RAMPS = [[0], [1, 2], [4], [3], [5]]


def test_cross_ramp_segments_thresholds_and_order():
    lum = PAL.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    segs = cross_ramp_segments(PAL, RAMPS, max_lum=40, max_hue=180)
    pairs = {tuple(p) for p in segs.tolist()}
    assert (0, 3) in pairs                       # skin + white (white is grey: no hue limit)
    assert (1, 2) not in pairs and (2, 1) not in pairs   # same ramp
    for a, b in pairs:
        assert lum[a] <= lum[b] and lum[b] - lum[a] <= 40
    assert not any(4 in p and 3 in p for p in pairs)     # far apart in luminance
    # Hue limit only applies to two chromatic colours.
    narrow = {tuple(p) for p in cross_ramp_segments(PAL, RAMPS, 40, 20).tolist()}
    assert (0, 3) in narrow and len(narrow) < len(pairs)
    assert len(cross_ramp_segments(PAL, [[0, 1, 2, 3, 4, 5]], 255, 180)) == 0


def _dither(color, amount, pattern="checker", **kw):
    img = np.full((8, 8, 3), color, np.float32)
    segs = cross_ramp_segments(PAL, RAMPS, 40, 180)
    return selective_pattern_dither(img, PAL, RAMPS, np.ones((8, 8), bool), pattern,
                                    cross_segments=segs, cross_amount=amount, **kw)


def test_skin_and_white_are_mixed_only_when_enabled():
    mid = (PAL[0].astype(np.float32) + PAL[3]) / 2           # halfway skin -> white
    assert len(np.unique(_dither(mid, 0.0))) == 1           # default: one-shade ramp stays solid
    idx = _dither(mid, 1.0)
    assert set(np.unique(idx)) == {0, 3}
    yy, xx = np.mgrid[0:8, 0:8]
    assert np.array_equal(idx == 3, (yy + xx) % 2 == 1) or np.array_equal(idx == 3, (yy + xx) % 2 == 0)


def test_cross_pairs_need_a_real_mix():
    near = PAL[0].astype(np.float32) * 0.9 + PAL[3] * 0.1   # 10 % towards white
    assert (_dither(near, 1.0, "bayer4") == 0).all()
    assert (_dither(near, 1.0, "bayer4", cross_min_mix=0.05) == 3).any()


def test_amount_prefers_ramp_neighbours():
    # Blue ramp (0, 1) and a violet (2) of almost the same luminance as blue 1.
    pal = np.array([[0, 0, 100], [0, 0, 200], [100, 0, 200]], np.uint8)
    ramps = [[0, 1], [2]]
    segs = cross_ramp_segments(pal, ramps, 40, 180)
    assert {tuple(p) for p in segs.tolist()} == {(1, 2)}
    img = np.full((8, 8, 3), (40, 0, 190), np.float32)   # fits violet-blue (t=0.4) best
    run = lambda a: selective_pattern_dither(img, pal, ramps, np.ones((8, 8), bool), "checker",
                                             cross_segments=segs, cross_amount=a)
    assert set(np.unique(run(1.0))) == {1, 2}
    low = run(0.2)                                        # squared distance x25: ramp pair wins
    assert np.array_equal(low, run(0.0)) and 2 not in low


def test_pipeline_default_unchanged_and_opt_in_changes_output():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:120, 0:200].astype(np.float32)
    img = np.stack([200 + 50 * x / 200, 170 + 80 * x / 200, 150 + 100 * x / 200], -1)
    img[:, :60] = ((60, 90, 200) + np.stack([y, y, y * 0], -1) * 0.3)[:, :60]
    src = Image.fromarray(np.clip(img + rng.normal(0, 1, img.shape), 0, 255).astype(np.uint8))
    base = ConvertSettings(mode="style", style_pattern="checker", style_grad_low=0.2,
                           style_grad_high=12)
    a = np.asarray(convert(src, base).image)
    b = np.asarray(convert(src, apply_overrides(base, {"style_cross_ramp": 0.0,
                                                       "style_cross_ramp_max_lum": 64})).image)
    assert np.array_equal(a, b)
    r = convert(src, apply_overrides(base, {"style_cross_ramp": 1.0, "style_cross_ramp_max_lum": 64}))
    assert r.used_colors <= 16 and r.info["cross_segments"] is not None
    assert not np.array_equal(a, np.asarray(r.image))


def _mix_profile(cross_share):
    st, _ = analyze_image(Image.fromarray(PAL[np.zeros((8, 8), np.uint8)]), "x")
    prof = build_profile([st], "x")
    prof["summary"]["dither"]["mix"] = {
        "relation": {"same_ramp_adjacent": 1 - cross_share, "same_ramp_far": 0.0,
                     "cross_ramp": cross_share},
        "cross_ramp": {
            "lum_diff": {"bin_edges": [0, 16, 32, 48, 64, 96, 128, 256],
                         "hist": [0.25, 0.25, 0.25, 0.25, 0, 0, 0], "median": 32.0, "p90": 57.6},
            "hue_diff": {"bin_edges": [0, 15, 30, 60, 90, 181], "hist": [0, 0, 0.5, 0, 0.5],
                         "median": 60.0, "p90": 150.0},
        },
    }
    prof["summary"]["dither"]["ratio"] = 0.4
    return prof


def test_profile_mapping_is_opt_in():
    prof = _mix_profile(0.35)
    ov, notes = profile_to_settings(prof)
    assert ov["style_cross_ramp"] == 0.0 and any("ランプ間" in n for n in notes)
    ov, notes = profile_to_settings(prof, cross_ramp=True)
    assert ov["style_cross_ramp"] == 0.5
    assert ov["style_cross_ramp_max_lum"] == 48.0          # 75th percentile of the histogram
    assert ov["style_cross_ramp_max_hue"] == 150.0
    assert cross_ramp_settings(_mix_profile(0.05))[0]["style_cross_ramp"] == 0.0
    assert cross_ramp_settings(_mix_profile(0.9))[0]["style_cross_ramp"] == 1.0


def test_presets_carry_cross_ramp_settings_off():
    for name, ov in STYLE_PRESETS.items():
        assert ov["style_cross_ramp"] == 0.0, name
        assert 0 < ov["style_cross_ramp_max_lum"] <= 128 and 0 < ov["style_cross_ramp_max_hue"] <= 180
