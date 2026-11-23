import numpy as np
import pytest
from PIL import Image

from retrocore import (
    ConvertSettings,
    convert,
    extract_palette,
    load_palette,
    save_image,
    save_palette,
    snap_to_pc98,
)
from retrocore.dither import DITHER_METHODS, apply_palette, bayer_matrix
from retrocore.palette import PC98_DEFAULT_16
from retrocore.pipeline import target_size


def gradient(w=160, h=100):
    y, x = np.mgrid[0:h, 0:w]
    arr = np.stack([x * 255 / w, y * 255 / h, (x + y) * 255 / (w + h)], -1)
    return Image.fromarray(arr.astype(np.uint8))


def screenshot():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 16, (20, 32))
    return Image.fromarray(PC98_DEFAULT_16[idx].repeat(4, 0).repeat(4, 1))


def test_exact_extraction_recovers_screenshot_palette():
    pal = extract_palette(screenshot(), 16, "auto")
    assert {tuple(c) for c in pal} == {tuple(c) for c in PC98_DEFAULT_16}


@pytest.mark.parametrize("method", ["auto", "exact", "kmeans", "mediancut"])
def test_extraction_respects_count(method):
    pal = extract_palette(gradient(), 8, method)
    assert 1 <= len(pal) <= 8
    assert pal.dtype == np.uint8 and pal.shape[1] == 3


def test_snap_to_pc98():
    pal = snap_to_pc98(np.array([[1, 128, 250]], np.uint8))
    assert (pal % 17 == 0).all()


def test_bayer_matrix_is_permutation():
    m = bayer_matrix(4)
    assert m.shape == (4, 4)
    assert len(np.unique(m)) == 16


@pytest.mark.parametrize("method", list(DITHER_METHODS))
def test_apply_palette_only_uses_palette_colors(method):
    out = apply_palette(gradient(), PC98_DEFAULT_16, method)
    assert out.mode == "P"
    used = {tuple(c) for c in np.asarray(out.convert("RGB")).reshape(-1, 3)}
    assert used <= {tuple(c) for c in PC98_DEFAULT_16}


@pytest.mark.parametrize("colors", [16, 256])
def test_convert_normal_mode(colors):
    s = ConvertSettings(preset="カスタム", custom_width=80, custom_height=50, colors=colors)
    r = convert(gradient(), s)
    assert r.image.size == (80, 50)
    assert len(r.palette) <= colors
    assert r.used_colors <= colors


def test_convert_transfer_mode_forces_palette():
    pal = extract_palette(screenshot(), 16)
    r = convert(gradient(), ConvertSettings(dither="floyd"), palette=pal)
    used = {tuple(c) for c in np.asarray(r.image.convert("RGB")).reshape(-1, 3)}
    assert used <= {tuple(c) for c in pal}


def test_target_size_modes():
    s = ConvertSettings(preset="PC-98 640×400", fit="fit")
    assert target_size(1600, 1200, s)[:2] == (533, 400)
    s.fit = "stretch"
    assert target_size(1600, 1200, s)[:2] == (640, 400)
    s.preset = "PC-98 640×200 (ライン倍化)"
    s.fit = "fit"
    assert target_size(1600, 1200, s) == (533, 200, 2.0)
    s.preset = "ピクセルサイズ指定"
    s.pixel_size = 4
    assert target_size(1600, 1200, s)[:2] == (400, 300)


def test_transparency_is_flattened():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    r = convert(img, ConvertSettings(preset="カスタム", custom_width=10, custom_height=10,
                                     fit="stretch", dither="none", background=(0, 0, 0)))
    assert np.asarray(r.image.convert("RGB")).max() == 0


@pytest.mark.parametrize("ext", [".gpl", ".pal", ".act"])
def test_palette_roundtrip(tmp_path, ext):
    path = tmp_path / f"p{ext}"
    save_palette(PC98_DEFAULT_16, path)
    assert np.array_equal(load_palette(path), PC98_DEFAULT_16)


@pytest.mark.parametrize("ext", ["png", "bmp", "gif"])
def test_save_image(tmp_path, ext):
    r = convert(gradient(), ConvertSettings(preset="PC-98 640×200 (ライン倍化)"))
    path = str(tmp_path / f"out.{ext}")
    save_image(r, path, scale=2, line_double=True)
    w, h = r.image.size
    assert Image.open(path).size == (w * 2, h * 4)
