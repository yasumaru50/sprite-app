"""背景除去（retrocore.sprite.background）のテスト。入力はコードで合成する（AI 画像は使わない）。"""

import time
from collections import deque

import numpy as np
import pytest

from retrocore.sprite import (BackgroundStep, SpritePipeline, ResizeStep, background_mask,
                              estimate_background, parse_color, pixel_downscale, remove_background)
from retrocore.sprite.background import DEFAULT_TOLERANCE, _flood_from_border, _label
from synth_sprite import BG, add_noise, character_mask, make_character

WHITE = (250, 250, 250)
OUTLINE = (90, 40, 50)


def _canvas(h, w, color):
    a = np.empty((h, w, 4), np.uint8)
    a[..., :3] = color
    a[..., 3] = 255
    return a


def _ring_with_white_body(h=80, w=80, bg=WHITE):
    """背景と同じ白の体を、暗い輪郭（幅 2）で囲んだ画像。(画像, 体のマスク, 輪郭のマスク)。"""
    a = _canvas(h, w, bg)
    yy, xx = np.mgrid[0:h, 0:w]
    outer = (np.abs(yy - h // 2) < 25) & (np.abs(xx - w // 2) < 25)
    inner = (np.abs(yy - h // 2) < 23) & (np.abs(xx - w // 2) < 23)
    a[outer & ~inner, :3] = OUTLINE
    return a, inner, outer & ~inner


# --- 外周からの塗りつぶし ----------------------------------------------------------

def test_only_background_connected_to_border_is_removed():
    a, body, line = _ring_with_white_body()
    a = add_noise(a, amp=5, seed=1)
    out = remove_background(a)
    assert (out[~(body | line), 3] == 0).all()          # 外側の背景はすべて透明
    assert (out[body | line, 3] == 255).all()           # 輪郭の中の同じ白は残る
    assert np.array_equal(out[body, :3], a[body, :3])   # 残った画素の色は変わらない
    assert (out[~(body | line)] == 0).all()             # 透明な画素は RGBA すべて 0


def test_synthetic_character_mask_matches():
    src = add_noise(make_character(256, 384), amp=6, seed=3)
    mask = background_mask(src)
    assert np.array_equal(~mask, character_mask(256, 384))


def test_output_is_two_level_alpha():
    src = add_noise(make_character(128, 192), amp=8, seed=0)
    out = remove_background(src)
    assert set(np.unique(out[..., 3])) == {0, 255}


def test_diagonal_outline_does_not_leak():
    """斜めの画素だけでできた 1 画素幅の輪郭（ひし形）の中へは漏れない（4 近傍の塗りつぶし）。"""
    n, c, r = 41, 20, 15
    a = _canvas(n, n, WHITE)
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.abs(yy - c) + np.abs(xx - c)
    a[d == r, :3] = OUTLINE
    m = background_mask(a, color=WHITE, tolerance=10)
    assert not m[d < r].any()
    assert m[d > r].all()


def test_estimate_background_ignores_character_touching_border():
    a = add_noise(_canvas(60, 90, BG), amp=4, seed=2)
    a[40:, 20:50, :3] = (200, 60, 60)          # 下の辺にキャラクターが接している
    c = estimate_background(a)
    assert max(abs(int(c[i]) - BG[i]) for i in range(3)) <= 2


def test_explicit_color_overrides_estimate():
    a = _canvas(40, 40, (0, 0, 255))
    a[:, :20, :3] = (255, 0, 0)               # 外周の半分ずつ赤と青
    m_red = background_mask(a, color=(255, 0, 0), tolerance=10)
    m_blue = background_mask(a, color=(0, 0, 255), tolerance=10)
    assert m_red[:, :20].all() and not m_red[:, 20:].any()
    assert m_blue[:, 20:].all() and not m_blue[:, :20].any()


@pytest.mark.parametrize("text, rgb", [("#d4f6bd", (212, 246, 189)), ("D4F6BD", (212, 246, 189)),
                                       ("10, 20,30", (10, 20, 30))])
def test_parse_color(text, rgb):
    assert parse_color(text) == rgb


@pytest.mark.parametrize("text", ["", "#12345", "1,2", "1,2,300", "zzzzzz", "-1,0,0"])
def test_parse_color_rejects(text):
    with pytest.raises(ValueError):
        parse_color(text)


# --- 許容量 --------------------------------------------------------------------

def test_tolerance_controls_halo():
    """背景色との差が 20 の帯（輪郭と背景の境目のにじみに相当）は、許容量 20 以上で消え、未満で残る。"""
    a, body, line = _ring_with_white_body(bg=(200, 220, 200))
    halo = np.zeros(a.shape[:2], bool)
    halo[13:67, 13:67] = True
    halo &= ~(body | line)
    a[halo, :3] = (220, 220, 200)              # 背景との差 20
    assert not background_mask(a, tolerance=19)[halo].any()
    assert background_mask(a, tolerance=20)[halo].all()
    assert background_mask(a, tolerance=DEFAULT_TOLERANCE)[halo].all()


def test_tolerance_zero_removes_only_exact_color():
    a = _canvas(30, 30, (100, 150, 200))
    a[0, 5, :3] = (101, 150, 200)
    m = background_mask(a, color=(100, 150, 200), tolerance=0)
    assert not m[0, 5] and m.sum() == 30 * 30 - 1


def test_large_tolerance_still_keeps_enclosed_white():
    a, body, line = _ring_with_white_body(bg=(230, 230, 230))
    m = background_mask(a, tolerance=100)
    assert not m[body].any()


@pytest.mark.parametrize("kw", [dict(tolerance=-1), dict(tolerance=256), dict(color=(0, 0, 300)),
                                dict(color=(1, 2)), dict(enclosed_min_area=-1)])
def test_rejects_bad_arguments(kw):
    with pytest.raises(ValueError):
        background_mask(_canvas(8, 8, WHITE), **kw)


# --- 囲まれた背景 -----------------------------------------------------------------

def _arms_with_gap():
    """腕と体のすき間（背景色、外周につながらない）と、小さな白い点（目のハイライト相当）がある画像。"""
    a = _canvas(100, 100, WHITE)
    a[20:80, 20:80, :3] = OUTLINE              # 体と腕をひとかたまりの輪郭色に
    gap = np.zeros((100, 100), bool)
    gap[30:70, 35:45] = True                   # 腕と体のすき間（400 画素）
    a[gap, :3] = WHITE
    dot = np.zeros((100, 100), bool)
    dot[25:27, 60:62] = True                   # 小さな白い点（4 画素）
    a[dot, :3] = WHITE
    return add_noise(a, amp=3, seed=5), gap, dot


def test_enclosed_background_kept_by_default():
    a, gap, dot = _arms_with_gap()
    m = background_mask(a)
    assert not m[gap].any() and not m[dot].any()
    assert m[:10].all()


def test_remove_enclosed_option():
    a, gap, dot = _arms_with_gap()
    m = background_mask(a, remove_enclosed=True)
    assert m[gap].all()                        # すき間は消える
    assert not m[dot].any()                    # 既定の最小面積（画像の 0.05% = 5 画素）より小さい点は残る
    m_all = background_mask(a, remove_enclosed=True, enclosed_min_area=1)
    assert m_all[dot].all()
    m_big = background_mask(a, remove_enclosed=True, enclosed_min_area=1000)
    assert not m_big[gap].any()


# --- 透過入力・工程 -------------------------------------------------------------

def test_transparent_input_passes_through():
    src = make_character(64, 96, transparent=True)
    out = remove_background(src)
    assert np.array_equal(out, src) and out is not src
    step = BackgroundStep()
    assert np.array_equal(step.apply(src), src)
    assert "そのまま" in step.describe()


def test_step_describe_and_color():
    step = BackgroundStep(color=(90, 170, 120), tolerance=20)
    assert step.color == (90, 170, 120)
    step.apply(make_character(64, 96))
    d = step.describe()
    assert "#5aaa78" in d and "指定" in d and "20" in d
    with pytest.raises(ValueError):
        BackgroundStep(tolerance=300)


def test_does_not_modify_input():
    src = make_character(64, 96)
    before = src.copy()
    remove_background(src)
    BackgroundStep(remove_enclosed=True).apply(src)
    assert np.array_equal(src, before)


# --- 縮小との組み合わせ -----------------------------------------------------------

def test_downscale_after_removal_majority_transparent():
    """背景除去のあと mode＋線優先で縮小: マスの過半数が透明なら透明、そうでなければ不透明な画素の色。"""
    a = _canvas(40, 40, WHITE)
    body = (180, 120, 90)
    a[0:10, 0:4, :3] = body                    # 左上のマス（10×10）: 体 40% → 透明
    a[0:10, 10:17, :3] = body                  # 2 番目のマス: 体 70% → 体の色
    a[10:20, 0:10, :3] = body                  # 2 行目の左のマス: すべて体
    a[10:20, 10:20, :3] = body                 # 体の中に背景と同じ白（外周から切り離す）
    a[12:18, 12:18, :3] = WHITE
    rgba = remove_background(a, color=WHITE, tolerance=DEFAULT_TOLERANCE)
    out = pixel_downscale(rgba, (4, 4), "mode", line_priority=True)
    assert set(np.unique(out[..., 3])) == {0, 255}
    assert out[0, 0, 3] == 0
    assert out[0, 1, 3] == 255 and tuple(out[0, 1, :3]) == body
    assert out[1, 0, 3] == 255 and tuple(out[1, 0, :3]) == body
    assert out[1, 1, 3] == 255                 # 体の中の白は透明にならない（白 36% < 50%）
    assert (out[2:, :, 3] == 0).all() and (out[:, 2:, 3] == 0).all()


def test_pipeline_background_then_mode_resize_has_no_background_colour():
    src = add_noise(make_character(256, 384), amp=6, seed=7)
    pipe = SpritePipeline.of([BackgroundStep(),
                              ResizeStep(width=64, method="mode", line_priority=True)])
    out = pipe.run(src)
    assert out.shape == (96, 64, 4)
    assert set(np.unique(out[..., 3])) == {0, 255}
    op = out[..., 3] == 255
    diff = np.abs(out[..., :3].astype(int) - np.array(BG)).max(-1)
    assert not (op & (diff <= DEFAULT_TOLERANCE)).any()   # 背景色の画素が残らない
    assert (out[0, :, 3] == 0).all() and (out[:, 0, 3] == 0).all()
    # 透明にした画素の RGB（0 = 黒）が線優先で輪郭と間違われない: 角のマスは透明のまま
    assert out[-1, 0, 3] == 0 and out[-1, -1, 3] == 0
    # キャラクターのある場所はおおむね不透明（縮小した正解マスクとの一致率）
    ref = character_mask(256, 384).reshape(96, 4, 64, 4).mean((1, 3)) >= 0.5
    assert (ref == op).mean() > 0.97


# --- 連結成分のラベル付け（内部関数）を幅優先探索と比べる -------------------------------

def _bfs_components(m):
    h, w = m.shape
    lab = np.full((h, w), -1)
    k = 0
    for y in range(h):
        for x in range(w):
            if m[y, x] and lab[y, x] < 0:
                q = deque([(y, x)])
                lab[y, x] = k
                while q:
                    cy, cx = q.popleft()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and lab[ny, nx] < 0:
                            lab[ny, nx] = k
                            q.append((ny, nx))
                k += 1
    return lab


@pytest.mark.parametrize("seed", range(6))
def test_label_and_flood_match_bfs(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((23, 31)) < 0.55
    ref = _bfs_components(m)
    lab = _label(m)
    assert ((lab >= 0) == m).all()
    # 同じ部分に同じ番号、違う部分に違う番号（番号の付け方は問わない）
    pairs = {(int(a), int(b)) for a, b in zip(ref[m], lab[m])}
    assert len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})
    border = set(ref[0]) | set(ref[-1]) | set(ref[:, 0]) | set(ref[:, -1])
    border.discard(-1)
    assert np.array_equal(_flood_from_border(m), np.isin(ref, list(border)) & m)


def test_flood_snake_needs_many_passes():
    """何度も折り返す通路（区間ごとの伝播を何回も繰り返す形）でも奥まで届く。"""
    n = 41
    m = np.zeros((n, n), bool)
    rows = list(range(1, n - 1, 2))
    for i, y in enumerate(rows):
        m[y, 1:n - 1] = True                   # 横の通路
        if i + 1 < len(rows):
            m[y + 1, n - 2 if i % 2 == 0 else 1] = True   # 端で次の段へ折り返す
    m[0, 1] = True                             # 入口（外周）
    got = _flood_from_border(m)
    assert np.array_equal(got, m)              # 通路全体が 1 つながりで、外周から届く
    m2 = m.copy()
    m2[2, n - 2] = False                       # 最初の折り返しを塞ぐと、それより奥には届かない
    got2 = _flood_from_border(m2)
    assert got2[1].sum() == m2[1].sum() and not got2[3:].any()


# --- 速さ --------------------------------------------------------------------

def test_speed_1254():
    src = add_noise(make_character(836, 1254), amp=6, seed=0)
    src = np.ascontiguousarray(np.pad(src, ((0, 0), (209, 209), (0, 0)), mode="edge"))
    assert src.shape[:2] == (1254, 1254)
    t0 = time.perf_counter()
    BackgroundStep(remove_enclosed=True).apply(src)
    assert time.perf_counter() - t0 < 3.0
