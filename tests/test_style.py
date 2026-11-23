import numpy as np
from PIL import Image

from retrocore import ConvertSettings, convert
from retrocore.analyzer import analyze_image, build_profile
from retrocore.palette import PC98_DIGITAL_8, infer_ramps
from retrocore.style import (
    STYLE_PRESETS,
    apply_overrides,
    hex_to_rgb,
    profile_palette,
    profile_to_settings,
)
from synth_cg import make_cg, rgb

BANNED = ["soft", "ソフト", "Soft"]   # names must describe era / technique only


def synth_profile(idx=None):
    st, _ = analyze_image(Image.fromarray(rgb(idx)), "synth")
    return build_profile([st], "synth")


def photo(w=320, h=200):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([120 + 100 * x / w, 90 + 60 * y / h, 200 - 80 * x / w], -1)
    img[60:140, 100:220] = (230, 190, 170)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_presets_are_valid_and_named_by_era():
    for name, ov in STYLE_PRESETS.items():
        assert name[:4].isdigit() and not any(b in name for b in BANNED)
        s = apply_overrides(ConvertSettings(), ov)
        r = convert(photo(), s)
        assert r.image.mode == "P" and r.used_colors <= 16


def test_profile_to_settings_rules():
    prof = synth_profile()
    ov, notes = profile_to_settings(prof)
    assert ov["mode"] == "style" and 8 <= ov["colors"] <= 16
    assert ov["style_pattern"] in ("checker", "tile2", "bayer4")
    assert ov["style_outline"] and ov["style_outline_color"] == (0, 0, 0)
    assert ov["preset"] == "PC-98 640×400"
    assert len(notes) >= 6 and all(isinstance(n, str) for n in notes)
    s = apply_overrides(ConvertSettings(), ov)
    r = convert(photo(), s)
    # The outline colour gets a palette slot even if the source has no black.
    assert any(tuple(c) == (0, 0, 0) for c in r.palette)
    assert len(r.palette) <= 16


def test_low_dither_profile_disables_pattern():
    idx, _, _ = make_cg()
    flat = idx.copy()
    flat[flat == 6] = 5          # remove the checker / tile colours
    flat[flat == 3] = 2
    flat[flat == 7] = 1
    ov, _ = profile_to_settings(synth_profile(flat))
    assert ov["style_pattern"] == "none"


def test_profile_palette_and_hex():
    pal = profile_palette(synth_profile(), 4)
    assert pal.shape == (4, 3) and pal.dtype == np.uint8
    assert hex_to_rgb("#F80") == (255, 136, 0) and hex_to_rgb("#102030") == (16, 32, 48)


def test_ramp_extension_for_fixed_palettes():
    plain = infer_ramps(PC98_DIGITAL_8)
    ext = infer_ramps(PC98_DIGITAL_8, extend_with_gray=True)
    assert max(len(r) for r in plain) == 2            # only black-white
    red = next(r for r in ext if 2 in r)              # index 2 = red
    assert red[0] == 0 and red[-1] == 7               # black ... red ... white
