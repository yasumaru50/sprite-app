"""ドット絵向けの縮小（retrocore.sprite.downscale）のテスト。入力はコードで合成する。"""

import time

import numpy as np
import pytest

from retrocore.sprite import PIXEL_METHODS, ResizeStep, block_edges, pixel_downscale
from synth_sprite import BG, OUTLINE, add_noise, make_character


def _colours(a):
    return {tuple(c) for c in a.reshape(-1, a.shape[-1])}


def _blocks(colours, bs):
    """色の 2 次元リスト（ブロック単位）から、各ブロック bs×bs 画素の RGBA 画像を作る。"""
    rows = [np.concatenate([np.broadcast_to(np.array(c, np.uint8), (bs, bs, 4)) for c in row], 1)
            for row in colours]
    return np.ascontiguousarray(np.concatenate(rows, 0))


# --- ブロックの境界 -----------------------------------------------------------

@pytest.mark.parametrize("src, dst", [(1024, 64), (256, 64), (100, 30), (97, 31), (384, 85), (7, 3)])
def test_block_edges_are_balanced(src, dst):
    e = block_edges(src, dst)
    assert e[0] == 0 and e[-1] == src and len(e) == dst + 1
    sizes = np.diff(e)
    assert sizes.min() >= 1 and sizes.max() - sizes.min() <= 1
    # 理想位置 i*src/dst とのずれは 0.5 以内（片側に寄らない）
    assert np.abs(e - np.arange(dst + 1) * src / dst).max() <= 0.5
    # 大きいブロックが前半・後半のどちらかに偏らない
    big = sizes == sizes.max()
    if 0 < big.sum() < dst and dst >= 10:
        half = dst // 2
        assert abs(int(big[:half].sum()) - int(big[half:2 * half].sum())) <= 2


def test_block_edges_integer_ratio():
    assert list(block_edges(16, 4)) == [0, 4, 8, 12, 16]


def test_non_integer_ratio_is_symmetric():
    # 左右対称な入力を整数でない倍率で縮小しても、結果は左右対称のまま
    src = make_character(250, 375)
    src = np.concatenate([src[:, :125], src[:, :125][:, ::-1]], 1)
    for m in PIXEL_METHODS:
        out = pixel_downscale(src, (33, 48), m)        # 出力幅が奇数なら境界が .5 ちょうどにならない
        assert np.array_equal(out, out[:, ::-1])


# --- 色の選び方 ---------------------------------------------------------------

@pytest.mark.parametrize("method", PIXEL_METHODS)
def test_output_colours_exist_in_input(method):
    src = add_noise(make_character(256, 384))
    out = pixel_downscale(src, (64, 96), method, line_priority=True)
    rgb_in = _colours(src[..., :3])
    assert _colours(out[..., :3]) <= rgb_in
    assert (out[..., 3] == 255).all()


def test_mode_groups_similar_colours_and_returns_real_colour():
    # 4×4 のブロック: 似た赤が 9 画素（すべて別の色）、青が 7 画素（同じ色）。
    # 完全一致で数えると青が勝つが、まとめて数えると赤が勝つ。採用色はブロック内に実在する赤
    reds = [(200 + i, 40 + (i % 3), 50 - i) for i in range(9)]
    blue = (20, 40, 220)
    px = np.array(reds + [blue] * 7, np.uint8).reshape(4, 4, 3)
    src = np.concatenate([px, np.full((4, 4, 1), 255, np.uint8)], -1)
    out = pixel_downscale(src, (1, 1), "mode", bin_size=32)[0, 0]
    assert tuple(out[:3]) in set(reds)
    # 区間の平均色 (204, 41, 46) に最も近い実在の赤
    mean = np.array(reds, float).mean(0)
    best = min(reds, key=lambda c: (((np.array(c) - mean) ** 2).sum(), c))
    assert tuple(out[:3]) == best
    # 区間の幅 1 なら完全一致で数えるので青
    assert tuple(pixel_downscale(src, (1, 1), "mode", bin_size=1)[0, 0, :3]) == blue


def test_mode_bin_size_controls_coarseness():
    # 暗めの赤 5 画素と明るめの赤 6 画素、それとは別の緑 5 画素
    dark = [(130 + i, 10, 10) for i in range(5)]
    light = [(170 + i, 10, 10) for i in range(6)]
    green = [(10, 200, 10 + i) for i in range(5)]
    px = np.array(dark + light + green, np.uint8).reshape(4, 4, 3)
    src = np.concatenate([px, np.full((4, 4, 1), 255, np.uint8)], -1)
    fine = tuple(pixel_downscale(src, (1, 1), "mode", bin_size=16)[0, 0, :3])
    coarse = tuple(pixel_downscale(src, (1, 1), "mode", bin_size=128)[0, 0, :3])
    assert fine in set(light)                 # 細かい区間: 明るい赤 6 > 緑 5 > 暗い赤 5
    assert coarse in set(dark + light)        # 粗い区間: 赤 11 がまとまる
    assert fine != coarse                     # 代表は暗・明の赤の平均に近い色へ移る


def test_median_picks_real_colour_closest_to_median():
    vals = [10, 20, 30, 200, 250]
    px = np.array([(v, v, v) for v in vals] + [(30, 30, 30)] * 4, np.uint8).reshape(3, 3, 3)
    src = np.concatenate([px, np.full((3, 3, 1), 255, np.uint8)], -1)
    out = pixel_downscale(src, (1, 1), "median")[0, 0]
    assert tuple(out) == (30, 30, 30, 255)
    # 中央値 (100, 50, 0) の新しい色は作らず、最も近い実在の色にする
    px = np.array([(200, 100, 0), (0, 0, 0), (200, 0, 0), (0, 100, 0)], np.uint8).reshape(2, 2, 3)
    src = np.concatenate([px, np.full((2, 2, 1), 255, np.uint8)], -1)
    out = pixel_downscale(src, (1, 1), "median")[0, 0, :3]
    assert tuple(out) in _colours(px)


def test_flat_blocks_keep_colour_exactly():
    cols = [[(255, 0, 0, 255), (0, 255, 0, 255)], [(0, 0, 255, 255), (9, 9, 9, 255)]]
    src = _blocks(cols, 5)
    for m in PIXEL_METHODS:
        out = pixel_downscale(src, (2, 2), m)
        assert np.array_equal(out, np.array(cols, np.uint8))


# --- 線優先 -------------------------------------------------------------------

def test_line_priority_keeps_thin_outline():
    # 8×8 のブロックに 2px の暗い線（25%）。通常は背景に負け、線優先なら線の色
    src = np.zeros((8, 32, 4), np.uint8)
    src[...] = (*BG, 255)
    src[:, 15:17, :3] = OUTLINE                   # 縦線（ブロック 1 の右端 1px とブロック 2 の左端 1px）
    src[:, 0:2, :3] = OUTLINE                     # ブロック 0 に 2px（25%）
    for m in PIXEL_METHODS:
        plain = pixel_downscale(src, (4, 1), m)
        lined = pixel_downscale(src, (4, 1), m, line_priority=True, line_ratio=0.25)
        assert tuple(plain[0, 0, :3]) == BG
        assert tuple(lined[0, 0, :3]) == OUTLINE
        # 1px（12.5%）しかないブロックは割合が足りない
        assert tuple(lined[0, 1, :3]) == BG and tuple(lined[0, 3, :3]) == BG
        # 割合を下げれば拾う
        low = pixel_downscale(src, (4, 1), m, line_priority=True, line_ratio=0.1)
        assert tuple(low[0, 1, :3]) == OUTLINE and tuple(low[0, 2, :3]) == OUTLINE


def test_line_priority_luma_threshold():
    src = np.zeros((4, 4, 4), np.uint8)
    src[...] = (200, 200, 200, 255)
    src[:, 0, :3] = (90, 90, 90)                  # 輝度 90
    lo = pixel_downscale(src, (1, 1), "mode", line_priority=True, line_luma=64)
    hi = pixel_downscale(src, (1, 1), "mode", line_priority=True, line_luma=100)
    assert tuple(lo[0, 0, :3]) == (200, 200, 200)
    assert tuple(hi[0, 0, :3]) == (90, 90, 90)


def test_line_priority_keeps_outline_connected_on_character():
    # 細い輪郭（3px）の頭を 1/8 に縮小。線優先のほうが輪郭色の画素が多く、頭の外周が途切れにくい
    src = add_noise(make_character(256, 384), amp=3)
    outline = np.array(OUTLINE)

    def dark_ring(out):
        d = np.abs(out[..., :3].astype(int) - outline).max(-1) <= 3
        # 頭の外周（中心 (12, 16)、半径 7.5 付近）
        yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]]
        r = np.hypot(yy + 0.5 - 95 / 8, xx + 0.5 - 16)
        ring = (r > 6.0) & (r < 8.0)
        return d[ring].mean()

    for m in PIXEL_METHODS:
        plain = dark_ring(pixel_downscale(src, (32, 48), m))
        lined = dark_ring(pixel_downscale(src, (32, 48), m, line_priority=True))
        assert lined > plain + 0.15, (m, plain, lined)


# --- 透過 ---------------------------------------------------------------------

def test_transparent_ratio():
    src = np.zeros((4, 4, 4), np.uint8)
    src[...] = (255, 0, 0, 255)
    src[:2, :, 3] = 0                              # 50% 透過
    src[2, :2, 3] = 0                              # 計 62.5%
    for m in PIXEL_METHODS:
        assert tuple(pixel_downscale(src, (1, 1), m, transparent_ratio=0.5)[0, 0]) == (0, 0, 0, 0)
        assert tuple(pixel_downscale(src, (1, 1), m, transparent_ratio=0.7)[0, 0]) == (255, 0, 0, 255)
    # 半透明（α < 128）も透過として数える。色は不透明画素だけから選ぶ
    src[:, :, :3] = (0, 0, 255)
    src[3, :, :3] = (255, 0, 0)
    src[:3, :, 3] = 100
    out = pixel_downscale(src, (1, 1), "mode", transparent_ratio=1.0)[0, 0]
    assert tuple(out) == (255, 0, 0, 255)


def test_transparent_character():
    src = make_character(256, 384, transparent=True)
    for m in PIXEL_METHODS:
        out = pixel_downscale(src, (64, 96), m)
        assert set(np.unique(out[..., 3])) <= {0, 255}
        assert out[0, 0, 3] == 0 and out[24, 32, 3] == 255
        assert (out[out[..., 3] == 0] == 0).all()
        # 輪郭に沿ったブロックは線優先なら透過より線が勝つので、不透明な面積が増える
        lined = pixel_downscale(src, (64, 96), m, line_priority=True)
        assert (lined[..., 3] == 255).sum() > (out[..., 3] == 255).sum()


# --- 決定性・速度・その他 -----------------------------------------------------

@pytest.mark.parametrize("method", PIXEL_METHODS)
def test_deterministic(method):
    src = add_noise(make_character(300, 450, transparent=True), seed=3)
    a = pixel_downscale(src, (37, 55), method, line_priority=True)
    b = pixel_downscale(src.copy(), (37, 55), method, line_priority=True)
    assert np.array_equal(a, b)


def test_ties_are_resolved_deterministically():
    # 2 色が同数: 区間の平均がブロック平均に近いほう、同じなら区間番号の小さいほう
    src = _blocks([[(0, 0, 0, 255)]], 2)
    src[0, :, :3] = (250, 250, 250)
    out = pixel_downscale(src, (1, 1), "mode")
    assert tuple(out[0, 0, :3]) == (0, 0, 0)
    out = pixel_downscale(src, (1, 1), "median")   # 中央値 (125,125,125) から同距離 → 値の小さい色
    assert tuple(out[0, 0, :3]) == (0, 0, 0)


def test_does_not_modify_input_and_upscale_works():
    src = make_character(16, 24)
    keep = src.copy()
    up = pixel_downscale(src, (40, 60), "mode")
    assert np.array_equal(src, keep)
    assert up.shape == (60, 40, 4)
    assert _colours(up[..., :3]) <= _colours(src[..., :3])


@pytest.mark.parametrize("kw", [dict(method="box"), dict(bin_size=0), dict(bin_size=257),
                                dict(line_ratio=0), dict(transparent_ratio=1.5)])
def test_validation(kw):
    src = make_character(16, 24)
    method = kw.pop("method", "mode")
    with pytest.raises(ValueError):
        pixel_downscale(src, (4, 6), method, **kw)
    with pytest.raises(ValueError):
        pixel_downscale(src[..., :3].copy(), (4, 6), "mode")


@pytest.mark.parametrize("method", PIXEL_METHODS)
def test_1024_to_64_is_fast(method):
    src = add_noise(make_character(1024, 1024), seed=1)
    pixel_downscale(src[:64, :64], (4, 4), method)  # 初回の import などを除く
    t0 = time.perf_counter()
    out = pixel_downscale(src, (64, 64), method, line_priority=True)
    assert time.perf_counter() - t0 < 2.0
    assert out.shape == (64, 64, 4)


@pytest.mark.parametrize("method", PIXEL_METHODS)
def test_resize_step_pixel_methods(method):
    src = make_character(transparent=True)
    step = ResizeStep(scale=0.25, method=method, line_priority=True, bin_size=16)
    out = step.apply(src)
    assert out.shape == (96, 64, 4)
    assert out[0, 0, 3] == 0 and out[24, 32, 3] == 255
    assert "線優先" in step.describe() and method in step.describe()
    assert np.array_equal(out, pixel_downscale(src, (64, 96), method, bin_size=16, line_priority=True))
