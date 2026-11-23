"""retrocore.sprite（パイプライン・縮小）のテスト。入力はコードで合成したキャラクター画像。"""

import numpy as np
import pytest
from PIL import Image

from retrocore.sprite import (RESIZE_METHODS, ResizeStep, SpritePipeline, Step, load_rgba,
                              resolve_size, save_rgba, to_rgba)
from synth_sprite import BG, make_character


def test_synth_character_has_shape_gradient_and_solid_background():
    a = make_character()
    assert a.shape == (384, 256, 4) and a.dtype == np.uint8
    assert (a[0, 0, :3] == BG).all() and (a[-1, -1, :3] == BG).all()
    # 背景は単色
    bg = (a[..., :3] == BG).all(-1)
    assert 0.4 < bg.mean() < 0.9
    # グラデーション: 頭の左上は右下より明るい
    head_tl = a[70, 110, :3].astype(int).sum()
    head_br = a[120, 150, :3].astype(int).sum()
    assert head_tl > head_br + 30
    assert len(np.unique(a.reshape(-1, 4), axis=0)) > 100
    t = make_character(transparent=True)
    assert t[0, 0, 3] == 0 and t[95, 128, 3] == 255


@pytest.mark.parametrize("args, want", [
    (dict(scale=0.25), (64, 96)),
    (dict(scale=1 / 3), (85, 128)),
    (dict(width=64), (64, 96)),
    (dict(height=48), (32, 48)),
    (dict(width=50, height=50), (50, 50)),
    (dict(scale=0.001), (1, 1)),
])
def test_resolve_size(args, want):
    assert resolve_size(256, 384, **args) == want


@pytest.mark.parametrize("args", [dict(), dict(scale=0.5, width=10), dict(scale=0), dict(width=0)])
def test_resolve_size_errors(args):
    with pytest.raises(ValueError):
        resolve_size(256, 384, **args)


@pytest.mark.parametrize("method", list(RESIZE_METHODS))
def test_resize_step_size_and_alpha(method):
    src = make_character(transparent=True)
    out = ResizeStep(scale=0.25, method=method).apply(src)
    assert out.shape == (96, 64, 4) and out.dtype == np.uint8
    assert out[0, 0, 3] == 0                  # 背景の透明は残る
    assert out[24, 32, 3] == 255              # 頭の中は不透明
    assert src[0, 0, 3] == 0                  # 入力は書き換えない


def test_resize_nearest_keeps_only_source_colours():
    src = make_character()
    out = ResizeStep(width=64, method="nearest").apply(src)
    src_cols = {tuple(c) for c in src.reshape(-1, 4)}
    assert {tuple(c) for c in out.reshape(-1, 4)} <= src_cols


def test_resize_step_same_size_is_copy():
    src = make_character(64, 96)
    out = ResizeStep(width=64, height=96).apply(src)
    assert np.array_equal(out, src) and out is not src


def test_resize_step_validates():
    with pytest.raises(ValueError):
        ResizeStep(scale=0.5, method="unknown")
    with pytest.raises(ValueError):
        ResizeStep()


class _Tag(Step):
    """呼ばれた順を記録し、左上の画素に印を付けるだけの工程。"""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def apply(self, rgba):
        self.log.append((self.name, rgba.shape[:2]))
        out = rgba.copy()
        out[0, 0, 0] = len(self.log)
        return out


def test_pipeline_runs_steps_in_order():
    log = []
    pipe = SpritePipeline([_Tag("a", log)]).add(ResizeStep(scale=0.5)).add(_Tag("b", log))
    src = make_character()
    results = pipe.run_all(src)
    assert [r.name for r in results] == ["a", "resize", "b"]
    assert log == [("a", (384, 256)), ("b", (192, 128))]
    assert results[-1].image.shape == (192, 128, 4)
    assert results[-1].image[0, 0, 0] == 2
    assert all(r.seconds >= 0 for r in results)
    log.clear()
    assert np.array_equal(pipe.run(src), results[-1].image)


def test_empty_pipeline_returns_rgba_copy():
    rgb = make_character(32, 48)[..., :3]
    out = SpritePipeline().run(rgb)
    assert out.shape == (48, 32, 4) and (out[..., 3] == 255).all()


def test_to_rgba_and_io(tmp_path):
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert to_rgba(gray).shape == (3, 4, 4)
    with pytest.raises(ValueError):
        to_rgba(np.zeros((3, 4), np.float32))
    # パレット画像の透過も RGBA にする
    p = Image.new("P", (4, 3), 1)
    p.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    p.info["transparency"] = 1
    assert (to_rgba(p)[..., 3] == 0).all()

    src = make_character(32, 48, transparent=True)
    path = tmp_path / "sub" / "a.png"
    save_rgba(src, path)
    assert np.array_equal(load_rgba(path), src)
    rgb_path = tmp_path / "b.jpg"
    Image.fromarray(src[..., :3]).save(rgb_path)
    assert (load_rgba(rgb_path)[..., 3] == 255).all()


def test_large_image_is_fast():
    import time
    src = make_character(1024, 1536)
    t0 = time.perf_counter()
    ResizeStep(scale=0.125, method="box").apply(src)
    assert time.perf_counter() - t0 < 2.0


def test_scale_nearest():
    from retrocore.sprite import scale_nearest
    src = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    out = scale_nearest(src, 3)
    assert out.shape == (6, 9, 4)
    assert np.array_equal(out[::3, ::3], src) and np.array_equal(out[2::3, 1::3], src)
    one = scale_nearest(src, 1)
    assert np.array_equal(one, src) and one is not src
    for bad in (0, 1.5, -2):
        with pytest.raises(ValueError):
            scale_nearest(src, bad)
