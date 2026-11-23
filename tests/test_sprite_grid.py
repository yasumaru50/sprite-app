"""グリッド検出（retrocore/sprite/grid.py）のテスト。

テスト画像は `synth_sprite.py` でコード合成する: 小さなドット絵を、マスごとに 19〜22px のばらつきを付けて拡大し、
ぼかし・ノイズ・輪郭の半ドットのずれを加えたもの（AI が描いたドット絵風の画像のまね）。
"""

import time

import numpy as np
import pytest

from retrocore.sprite import (GridNotFoundError, GridStep, SpritePipeline, BackgroundStep, count_colors,
                              detect_grid, grid_to_pixels, remove_background)
from retrocore.sprite.grid import edge_profile, estimate_period, fit_lines
from synth_sprite import enlarge_like_ai, make_character, make_pixel_sprite

COLOR_TOL = 24


def _match(out: np.ndarray, art: np.ndarray) -> float:
    """元のドット絵と同じになったドットの割合（透明・不透明が一致し、不透明なら色の差が COLOR_TOL 以内）。"""
    assert out.shape == art.shape, (out.shape, art.shape)
    op = art[..., 3] > 0
    same_alpha = (out[..., 3] > 0) == op
    close = np.abs(out[..., :3].astype(int) - art[..., :3].astype(int)).max(-1) <= COLOR_TOL
    return float((same_alpha & (~op | close)).mean())


@pytest.mark.parametrize("seed", range(4))
def test_recovers_irregular_blurred_sprite(seed):
    art = make_pixel_sprite()
    img, xs, ys = enlarge_like_ai(art, sizes=(19, 22), blur=2, noise=6, half_shift=6, seed=seed)
    res = grid_to_pixels(remove_background(img))
    assert res.size == (art.shape[1], art.shape[0])
    assert _match(res.image, art) >= 0.98
    cw, ch = res.grid.cell_size
    assert 19 <= cw <= 22 and 19 <= ch <= 22
    assert res.colors == count_colors(res.image)
    assert set(np.unique(res.image[..., 3])) <= {0, 255}


def test_lines_follow_irregular_dots():
    """線が、ばらついた本当の境目の近く（±3px）にあること（等間隔の格子ではずれていく）。"""
    art = make_pixel_sprite()
    img, xs, ys = enlarge_like_ai(art, blur=1, seed=7)
    res = grid_to_pixels(remove_background(img))
    for found, true in zip(res.lines, (xs, ys)):
        true = true[3:-3]  # 背景の余白（3 ドット）を除いた、キャラクターの範囲の境目
        assert len(found) == len(true)
        assert np.abs(found - true).max() <= 3


def test_small_parts_survive():
    """1 ドットの目と 2 ドットの頬が残ること。"""
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, blur=2, seed=3)
    out = grid_to_pixels(remove_background(img)).image
    for color in ((70, 50, 60), (250, 180, 190)):
        want = (np.abs(art[..., :3].astype(int) - color).max(-1) == 0) & (art[..., 3] > 0)
        got = (np.abs(out[..., :3].astype(int) - color).max(-1) <= COLOR_TOL) & (out[..., 3] > 0)
        assert want.any() and (got & want).sum() == want.sum()


def test_transparent_input_without_background_step():
    """透過付きの入力（背景除去済み）もそのまま扱える。マスの過半数が透明なら透明。"""
    art = make_pixel_sprite()
    img, xs, ys = enlarge_like_ai(art, blur=0, noise=0, half_shift=0, seed=1)
    big_alpha = np.repeat(np.repeat(np.pad(art[..., 3], 3), np.diff(ys), 0), np.diff(xs), 1)
    rgba = img.copy()
    rgba[..., 3] = big_alpha
    rgba[big_alpha == 0] = 0
    res = grid_to_pixels(rgba)
    assert _match(res.image, art) == 1.0


def test_transparent_ratio():
    """半分より少し少なく透明なマスは不透明、半分より多く透明なマスは透明（既定 0.5）。"""
    cell = 20
    rgba = np.zeros((cell * 6, cell * 6, 4), np.uint8)
    rgba[cell:cell * 5, cell:cell * 5] = (200, 80, 60, 255)
    rgba[cell * 2:cell * 4, cell * 2:cell * 4] = (40, 40, 160, 255)
    # 右下のマス (4, 4) の左 9 列だけ透明に（45% 透明）→ 不透明のまま
    rgba[cell * 4:cell * 5, cell * 4:cell * 4 + 9] = 0
    # 左上のマス (1, 1) の左 11 列だけ透明に（55% 透明）→ 透明
    rgba[cell:cell * 2, cell:cell + 11] = 0
    res = grid_to_pixels(rgba, dot_size=cell, origin=(0, 0))
    assert res.size == (4, 4)
    a = res.image[..., 3]
    assert a[0, 0] == 0 and a[3, 3] == 255 and a[1:3, 1:3].all()
    assert tuple(res.image[1, 1, :3]) == (40, 40, 160)
    res2 = grid_to_pixels(rgba, dot_size=cell, origin=(0, 0), transparent_ratio=0.4)
    assert res2.image[3, 3, 3] == 0


def test_manual_dot_size_and_origin():
    """ドットの大きさ・始まりを指定すると等間隔の格子になり、判定なしで変換する。"""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (6, 8, 3)).astype(np.uint8)
    big = np.repeat(np.repeat(small, 12, 0), 10, 1)
    rgba = np.concatenate([big, np.full(big.shape[:2] + (1,), 255, np.uint8)], -1)
    res = grid_to_pixels(rgba, dot_size=(10, 12), origin=(0, 0))
    assert res.grid.manual and res.grid.period == (10.0, 12.0)
    assert res.size == (8, 6)
    assert np.array_equal(res.image[..., :3], small)
    # 大きさだけ指定（始まりは線の合わせ込みで決まる）
    res = grid_to_pixels(rgba, dot_size=(10, 12))
    assert np.array_equal(res.image[..., :3], small)


def test_manual_dot_size_for_shifted_grid():
    """格子が画像の端から始まっていない（半端なマスがある）場合も、大きさの指定だけで正しく戻る。"""
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, sizes=(20, 20), blur=1, noise=4, half_shift=0, seed=2)
    img = img[7:, 13:]  # 格子の始まりをずらす
    res = grid_to_pixels(remove_background(img), dot_size=20)
    assert _match(res.image, art) >= 0.98
    uni = grid_to_pixels(remove_background(img), dot_size=20, snap=False)
    assert _match(uni.image, art) >= 0.98


def test_uniform_grid_drifts_on_irregular_dots():
    """ばらつきのある画像では、等間隔の格子（snap=False）より線を合わせる方式の方がよく戻る。"""
    art = make_pixel_sprite(30, 32)
    img, _, _ = enlarge_like_ai(art, sizes=(18, 23), blur=1, seed=5)
    rgba = remove_background(img)
    snapped = grid_to_pixels(rgba, dot_size=20.5)
    uniform = grid_to_pixels(rgba, dot_size=20.5, snap=False)
    good = _match(snapped.image, art)
    bad = _match(uniform.image, art) if uniform.image.shape == art.shape else 0.0
    assert good >= 0.97 and good > bad


@pytest.mark.parametrize("make", [
    lambda: remove_background(make_character(512, 768)),                  # 普通のイラスト（グラデーション）
    lambda: np.dstack([np.random.default_rng(0).integers(0, 256, (300, 300, 3)).astype(np.uint8),
                       np.full((300, 300), 255, np.uint8)]),                # ノイズだけ
    lambda: np.dstack([np.tile(np.linspace(0, 255, 400).astype(np.uint8), (300, 1))[..., None].repeat(3, -1),
                       np.full((300, 400), 255, np.uint8)]),                # なめらかなグラデーション
])
def test_not_found_on_non_pixel_art(make):
    with pytest.raises(GridNotFoundError) as e:
        grid_to_pixels(make())
    assert "見つかりませんでした" in str(e.value)


def test_not_found_when_all_transparent():
    with pytest.raises(GridNotFoundError):
        grid_to_pixels(np.zeros((100, 100, 4), np.uint8))


def test_estimate_period_and_fit_lines_basic():
    prof = np.zeros(201)
    true = np.cumsum([0, 20, 19, 22, 21, 20, 19, 22, 20, 21])
    prof[true[1:-1]] = 100
    p, prom = estimate_period(prof, 0, 200)
    assert 19 <= p <= 22 and prom > 0.2
    lines = fit_lines(prof / 100, p, true[1], true[-2])
    assert set(true[1:-1]) <= set(lines.tolist())


def test_edge_profile_counts_alpha_edges():
    rgba = np.zeros((10, 30, 4), np.uint8)
    rgba[:, 10:20] = (100, 100, 100, 255)
    rgba[:, 15:20, 0] = 160
    px = edge_profile(rgba, 1)
    assert px.shape == (31,)
    assert px[10] == px[20] == 128 * 10 and px[15] == 60 * 10
    assert px[[0, 5, 12, 30]].sum() == 0


def test_grid_step_in_pipeline_and_describe():
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, seed=4)
    step = GridStep()
    pipe = SpritePipeline([BackgroundStep(), step])
    results = pipe.run_all(img)
    assert results[-1].image.shape == art.shape
    d = results[-1].description
    assert "グリッド検出" in d and "ドット" in d and "色" in d and f"{art.shape[1]}×{art.shape[0]}" in d
    with pytest.raises(ValueError):
        GridStep(transparent_ratio=0)


def test_speed_on_large_image():
    """1254×1254 相当（約 60×60 ドット × 20px）で数秒以内。"""
    art = make_pixel_sprite(52, 56)
    img, _, _ = enlarge_like_ai(art, sizes=(19, 22), seed=0)
    rgba = remove_background(img)
    t = time.perf_counter()
    res = grid_to_pixels(rgba)
    assert time.perf_counter() - t < 5
    assert _match(res.image, art) >= 0.97


def test_detect_grid_rejects_bad_dot_size():
    rgba = np.full((40, 40, 4), 255, np.uint8)
    with pytest.raises(ValueError):
        detect_grid(rgba, dot_size=0)


# ---------------------------------------------------------------- 正方形のドット（既定）と縦横別

def test_square_is_default_and_uses_one_size():
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, seed=1)
    res = grid_to_pixels(remove_background(img))
    assert res.grid.square and res.grid.period[0] == res.grid.period[1]
    assert GridStep().square
    sep = grid_to_pixels(remove_background(img), square=False)
    assert not sep.grid.square
    assert _match(sep.image, art) >= 0.98


@pytest.mark.parametrize("seed", range(3))
def test_square_keeps_aspect_of_original(seed):
    """正方形のドットで描かれた絵では、出力の縦横比が元の絵（キャラクターの範囲）の縦横比とほぼ同じになる。"""
    art = make_pixel_sprite(30, 24)
    img, xs, ys = enlarge_like_ai(art, sizes=(19, 22), blur=2, seed=seed)
    rgba = remove_background(img)
    res = grid_to_pixels(rgba)
    op = rgba[..., 3] > 0
    cols, rows = np.nonzero(op.any(0))[0], np.nonzero(op.any(1))[0]
    src_aspect = (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)
    out_aspect = res.size[0] / res.size[1]
    assert abs(out_aspect / src_aspect - 1) < 0.08
    assert _match(res.image, art) >= 0.98


def test_square_uses_both_axes_when_one_axis_has_few_edges():
    """横方向の境目が少ない絵（縦じま）でも、縦横を合わせた推定で両方向の大きさがそろう。"""
    h, w = 10, 24
    stripes = np.zeros((h, w, 4), np.uint8)
    stripes[..., 3] = 255
    colors = [(200, 60, 60), (60, 60, 200), (60, 180, 60), (220, 200, 80)]
    for j in range(w):
        stripes[:, j, :3] = colors[(j * 7 // 3) % 4]
    stripes[5:, :, :3] //= 2
    img, _, _ = enlarge_like_ai(stripes, sizes=(19, 22), seed=0, half_shift=0)
    res = grid_to_pixels(remove_background(img))
    assert res.size == (w, h)
    assert _match(res.image, stripes) >= 0.98


def test_separate_axes_for_non_square_dots():
    """横長のドット（横 21〜23px・縦 17〜19px）は縦横別の推定で大きさが分かれる。どちらの方式でも元の絵に戻る。"""
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, sizes=(21, 23), sizes_y=(17, 19), blur=1, seed=0)
    rgba = remove_background(img)
    sep = grid_to_pixels(rgba, square=False)
    px, py = sep.grid.period
    assert 21 <= px <= 23 and 17 <= py <= 19
    assert _match(sep.image, art) >= 0.98
    sq = grid_to_pixels(rgba)
    assert sq.grid.period[0] == sq.grid.period[1]
    assert _match(sq.image, art) >= 0.95


def test_estimate_square_period_combines_axes():
    from retrocore.sprite.grid import estimate_square_period
    px = np.zeros(401)
    px[np.arange(20, 400, 20)] = 100
    py = np.zeros(301)
    py[[40, 120, 200, 260]] = 100  # 横の境目は少ない（20 の倍数の位置だけ）
    p, prom, per_axis = estimate_square_period([px, py], [(0, 400), (0, 300)])
    assert abs(p - 20) < 0.6 and prom > 0.2 and len(per_axis) == 2


def test_duplicate_lines():
    from retrocore.sprite import duplicate_lines
    img = np.zeros((4, 5, 4), np.uint8)
    img[..., 3] = 255
    img[..., 0] = np.arange(5)[None, :] * 50          # 列ごとに色が違う
    img[..., 1] = np.arange(4)[:, None] * 50          # 行ごとに色が違う
    assert duplicate_lines(img) == (0, 0)
    dup = np.concatenate([img[:2], img[1:2], img[2:]], 0)          # 2 行目を重ねる
    dup = np.concatenate([dup[:, :3], dup[:, 2:3], dup[:, 3:]], 1)  # 3 列目を重ねる
    assert duplicate_lines(dup) == (1, 1)
    near = dup.copy()
    near[2, 0, :3] = (255, 255, 255)  # 重ねた行の 1 画素だけ違う
    assert duplicate_lines(near)[0] == 0
    assert duplicate_lines(near, min_ratio=0.8)[0] == 1
    empty = np.zeros((3, 3, 4), np.uint8)
    assert duplicate_lines(empty) == (0, 0)


def test_no_duplicate_lines_on_synthetic():
    """余分なグリッド線が入らなければ、元の絵に無い「同じ色が続く行・列」は増えない。"""
    from retrocore.sprite import duplicate_lines
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, blur=2, seed=2)
    res = grid_to_pixels(remove_background(img))
    want = duplicate_lines(art)
    assert res.duplicates[0] <= want[0] and res.duplicates[1] <= want[1]
