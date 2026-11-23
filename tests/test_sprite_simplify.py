"""retrocore/sprite/simplify.py（単純化: 似た色をまとめる）のテスト。テスト画像はコードで合成する。"""

import time

import numpy as np
import pytest

from retrocore.sprite import (BackgroundStep, GridStep, SimplifyStep, SpritePipeline, merge_close_colors,
                              rgb_to_oklab, simplify_colors, transition_mask)
from retrocore.sprite import simplify as simplify_mod
from retrocore.sprite.simplify import DEFAULT_COLORS, DEFAULT_MERGE_DISTANCE
from retrocore.sprite.grid import count_colors
from synth_sprite import (PIX_CHEEK, PIX_EYE, PIX_OUTLINE, enlarge_like_ai, make_pixel_sprite)

ORANGE = (240, 120, 80)
CREAM = (250, 240, 210)
DARK = (70, 40, 55)
PINK = (250, 175, 185)


def _rgba(rgb, alpha=None):
    rgb = np.asarray(rgb, np.uint8)
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8) if alpha is None else alpha[..., None].astype(np.uint8)
    return np.concatenate([rgb, a], axis=-1)


def _noisy(shape, color, amp, rng):
    return np.clip(np.array(color, int) + rng.integers(-amp, amp + 1, shape + (3,)), 0, 255)


def _dist(c1, c2):
    return float(np.linalg.norm(rgb_to_oklab(np.array(c1, float)) - rgb_to_oklab(np.array(c2, float))))


def _palette_has(pal, color, tol=0.04):
    return min(_dist(p, color) for p in pal) <= tol


def _two_areas(seed=0, amp=8):
    """左半分がオレンジ、右半分がクリーム色（どちらもノイズで色が多い）。32×32。"""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((32, 32, 3), int)
    rgb[:, :16] = _noisy((32, 16), ORANGE, amp, rng)
    rgb[:, 16:] = _noisy((32, 16), CREAM, amp, rng)
    return rgb


# --- 色数 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 24])
def test_colors_at_most_k(k):
    rng = np.random.default_rng(1)
    img = _rgba(rng.integers(0, 256, (40, 40, 3)))
    r = simplify_colors(img, k)
    assert count_colors(r.image) <= k
    assert len(r.palette) == count_colors(r.image) and r.counts.sum() == 40 * 40
    assert r.colors_before == count_colors(img)


def test_every_pixel_is_a_palette_color_no_dither():
    img = _rgba(_two_areas())
    r = simplify_colors(img, 2)
    pal = {tuple(c) for c in r.palette}
    assert {tuple(c) for c in r.image[..., :3].reshape(-1, 3)} <= pal
    # ディザなし: 面はそれぞれ 1 色の平塗り
    assert len({tuple(c) for c in r.image[:, :16, :3].reshape(-1, 3)}) == 1
    assert len({tuple(c) for c in r.image[:, 16:, :3].reshape(-1, 3)}) == 1


def test_few_colors_are_unchanged():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[:, :2], rgb[:, 2:4], rgb[:, 4:] = ORANGE, CREAM, DARK
    img = _rgba(rgb)
    assert np.array_equal(simplify_colors(img, 16).image, img)


# --- 透明 ---------------------------------------------------------------------------------

def test_transparent_pixels_kept_and_not_counted():
    rng = np.random.default_rng(2)
    rgb = _two_areas()
    alpha = np.full((32, 32), 255)
    alpha[:8] = 0
    alpha[8:10, :] = 100             # 半透明（α < 128）も透明として扱う
    img = _rgba(rgb, alpha)
    img[:8, :, :3] = rng.integers(0, 256, (8, 32, 3))   # 透明な画素の RGB は何でもよい（そのまま残す）
    r = simplify_colors(img, 2)
    tr = img[..., 3] < 128
    assert np.array_equal(r.image[tr], img[tr])
    assert np.array_equal(r.image[..., 3], img[..., 3])
    assert len(r.palette) == 2                           # 透明な画素のでたらめな色は数えない
    assert _palette_has(r.palette, ORANGE) and _palette_has(r.palette, CREAM)


def test_all_transparent():
    img = np.zeros((5, 5, 4), np.uint8)
    r = simplify_colors(img, 4)
    assert np.array_equal(r.image, img) and len(r.palette) == 0


# --- 同じ入力から同じ結果 -------------------------------------------------------------------

def test_deterministic():
    rng = np.random.default_rng(3)
    img = _rgba(rng.integers(0, 256, (30, 30, 3)))
    a, b = simplify_colors(img, 8), simplify_colors(img.copy(), 8)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.palette, b.palette)
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=1)
    pipe = SpritePipeline([BackgroundStep(), GridStep(), SimplifyStep(12)])
    assert np.array_equal(pipe.run(big), pipe.run(big))


def test_input_not_modified():
    img = _rgba(_two_areas())
    before = img.copy()
    simplify_colors(img, 2)
    assert np.array_equal(img, before)


# --- 小さくて大事な色 -----------------------------------------------------------------------

def test_small_dark_color_survives():
    """大きな面（ノイズで色が多い）2 つと、3 画素の暗い色。3 色にしても暗い色が残る。"""
    rgb = _two_areas(amp=10)
    eyes = [(10, 20), (10, 26), (11, 26)]
    for y, x in eyes:
        rgb[y, x] = DARK
    r = simplify_colors(_rgba(rgb), 3)
    assert _palette_has(r.palette, DARK, 0.02)
    for y, x in eyes:
        assert _dist(r.image[y, x, :3], DARK) < 0.02


def test_small_cheek_color_survives_against_blur_remnants():
    """面の境目のぼかしの名残（中間色の列）より、離れた小さな頬の色を残す。"""
    rgb = np.zeros((24, 24, 3), int)
    rgb[:, :12] = ORANGE
    rgb[:, 12:] = CREAM
    # 境目にぼかしの名残の中間色（何色もある）
    for y in range(24):
        t = 0.35 + 0.3 * (y % 4) / 3
        rgb[y, 12] = np.round(np.array(ORANGE) * (1 - t) + np.array(CREAM) * t)
    rgb[16, 18:20] = PINK          # 頬（2 画素）
    img = _rgba(rgb)
    r = simplify_colors(img, 3)
    assert _palette_has(r.palette, PINK, 0.02)
    assert _palette_has(r.palette, ORANGE, 0.02) and _palette_has(r.palette, CREAM, 0.02)


def test_transition_mask_marks_blur_not_outline():
    rgb = np.zeros((5, 7, 3), int)
    rgb[:, :3] = ORANGE
    rgb[:, 3] = (np.array(ORANGE) + np.array(CREAM)) // 2   # ぼかしの名残
    rgb[:, 4:] = CREAM
    rgb[2, 5] = DARK                                          # 目（周りより暗い）
    m = transition_mask(_rgba(rgb))
    assert m[:, 3].all()
    assert not m[2, 5] and not m[:, :3].any()


def test_palette_colors_are_real_not_muddy():
    """残す色は元の画像にある色の近く。中間色をいくら含んでも、平均した濁った色にならない。"""
    rgb = _two_areas(amp=6)
    rgb[:, 15] = (np.array(ORANGE) + np.array(CREAM)) // 2     # 中間色の列
    img = _rgba(rgb)
    r = simplify_colors(img, 2)
    src = img[..., :3].reshape(-1, 3)
    for p in r.palette:
        assert min(_dist(p, c) for c in src) < 0.01
    assert _palette_has(r.palette, ORANGE, 0.03) and _palette_has(r.palette, CREAM, 0.03)


# --- 近さで決める ---------------------------------------------------------------------------

def test_merge_distance():
    rgb = _two_areas(amp=6)
    rgb[20:, :8] = _noisy((12, 8), DARK, 6, np.random.default_rng(4))
    img = _rgba(rgb)
    r = simplify_colors(img, 256, merge_distance=0.1)
    assert len(r.palette) == 3
    # 既定（0.03）でも、ノイズ（±6）で分かれた色は見分けがつかないので、面ごとに 1 色にまとまる
    assert len(simplify_colors(img, 256).palette) == 3
    # 上限の色数のほうが少なければ上限で止まる
    assert len(simplify_colors(img, 2, merge_distance=0.1).palette) == 2
    # 距離 0 なら色数だけで決める（922 色 → 上限の 256 色）
    assert count_colors(img) > 256
    assert len(simplify_colors(img, 256, merge_distance=0).palette) == 256


def test_invalid_arguments():
    img = _rgba(_two_areas())
    with pytest.raises(ValueError):
        simplify_colors(img, 0)
    with pytest.raises(ValueError):
        simplify_colors(img, 4, merge_distance=-1)
    with pytest.raises(ValueError):
        simplify_colors(img[..., :3], 4)
    with pytest.raises(ValueError):
        SimplifyStep(colors=0)
    with pytest.raises(ValueError):
        SimplifyStep(merge_distance=-0.1)


# --- ドット絵風の画像を通しで -----------------------------------------------------------------

def test_pixel_art_pipeline_restores_palette():
    """合成のドット絵（9 色）を AI 風に拡大して戻すと、グリッド検出で色が増える。単純化で元の色に近い色へ戻る。"""
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=0)
    grid = SpritePipeline([BackgroundStep(), GridStep()]).run(big)
    assert grid.shape == art.shape
    assert count_colors(grid) > 30
    step = SimplifyStep(12)
    out = step.apply(grid)
    assert count_colors(out) <= 12
    assert np.array_equal(out[..., 3], grid[..., 3])
    pal = step.last.palette
    for c in (PIX_OUTLINE, PIX_EYE, PIX_CHEEK):     # 輪郭・目（1 ドット）・頬（2 ドット）が残る
        assert _palette_has(pal, c, 0.05), c
    # ほとんどの画素が元のドット絵の色に近い
    op = art[..., 3] > 0
    d = np.linalg.norm(rgb_to_oklab(out[..., :3][op].astype(float)) - rgb_to_oklab(art[..., :3][op].astype(float)), axis=1)
    assert (d < 0.05).mean() > 0.9
    assert "単純化" in step.describe() and "12" in step.describe()


def test_speed_many_colors():
    rng = np.random.default_rng(5)
    img = _rgba(rng.integers(0, 256, (96, 96, 3)))
    t = time.perf_counter()
    r = simplify_colors(img, 16)
    assert time.perf_counter() - t < 5
    assert len(r.palette) <= 16


# --- ごく近い色をまとめる（問題 3） ----------------------------------------------------------

PURPLES = [(65, 34, 81), (65, 35, 81), (67, 36, 82), (64, 32, 80)]   # ai05 の輪郭の紫に近い 4 色（差 0.003〜0.006）


def _quads(colors, size=8):
    """色ごとに size×size の面を横に並べた画像。"""
    rgb = np.zeros((size, size * len(colors), 3), np.uint8)
    for i, c in enumerate(colors):
        rgb[:, i * size:(i + 1) * size] = c
    return _rgba(rgb)


def test_defaults():
    assert DEFAULT_COLORS == 16 and DEFAULT_MERGE_DISTANCE == 0.03
    s = SimplifyStep()
    assert s.colors == 16 and s.merge_distance == 0.03


def test_near_identical_colors_merge_even_with_room():
    """ほぼ同じ紫が 4 色。上限を 16 にしても 1 色にまとまる。離れた色（クリーム）はそのまま。"""
    img = _quads(PURPLES + [CREAM])
    r = simplify_colors(img, 16)
    assert len(r.palette) == 2
    assert {tuple(c) for c in r.palette} <= {tuple(c) for c in PURPLES + [CREAM]}    # 実在の色から選ぶ
    assert r.colors_after_close == 2
    # しきい値 0 なら、上限に余裕があるので 5 色のまま（以前の動き）
    assert len(simplify_colors(img, 16, merge_distance=0).palette) == 5


DISTINCT = [(250, 240, 210), (240, 120, 80), (70, 40, 55), (250, 175, 185), (60, 140, 220), (90, 180, 70),
            (240, 220, 60), (150, 60, 160), (30, 90, 60), (200, 200, 200), (120, 70, 30), (20, 20, 120)]


def test_slots_freed_by_close_merge_are_used():
    """ほぼ同じ紫 4 色と、はっきり違う 12 色。上限 13 色なら、紫は 1 色にまとまり、12 色はすべて残る（紫が枠を使わない）。"""
    img = _quads(PURPLES + DISTINCT, size=4)
    r = simplify_colors(img, 13)
    assert len(r.palette) == 13
    for c in DISTINCT + [PURPLES[0]]:
        assert _palette_has(r.palette, c, 0.01), c
    assert r.colors_after_close == 13
    # 上限 12 色なら、はっきり違う色のどれかがまとめられるが、紫が 2 色以上残ることはない
    r12 = simplify_colors(img, 12)
    assert len(r12.palette) == 12
    assert sum(_dist(p, PURPLES[0]) < 0.03 for p in r12.palette) <= 1


def test_colors_farther_than_threshold_stay_apart():
    """しきい値より離れた 2 色（差 約 0.05。並べると分かる）は、上限に余裕があればまとまらない。"""
    a, b = (213, 94, 91), (194, 83, 89)        # ai03 のかさの 2 段の赤
    d = simplify_mod._distance(rgb_to_oklab(np.array([a], float)), rgb_to_oklab(np.array([b], float)))[0, 0]
    assert DEFAULT_MERGE_DISTANCE < d < 0.06
    r = simplify_colors(_quads([a, b, CREAM]), 16)
    assert len(r.palette) == 3
    # 近さで決める（しきい値を大きく）と 1 色になる
    assert len(simplify_colors(_quads([a, b, CREAM]), 16, merge_distance=0.06).palette) == 2


def test_chroma_counts_more_than_lightness():
    """距離は色みを 1.5 倍にして測る: 明るさだけ 0.025 違う 2 色はまとまり、色みだけ（重みなしで）0.025 違う 2 色は残る。"""
    lab0 = rgb_to_oklab(np.array([[200, 120, 120]], float))[0]

    def from_lab(lab):    # 近い RGB を探す（総当たりの近傍）
        base = np.array([200, 120, 120])
        cand = base + np.stack(np.meshgrid(*[np.arange(-12, 13)] * 3, indexing="ij"), -1).reshape(-1, 3)
        return tuple(cand[np.argmin(np.linalg.norm(rgb_to_oklab(cand.astype(float)) - lab, axis=1))])

    light = from_lab(lab0 + [0.025, 0, 0])
    chroma = from_lab(lab0 + [0, 0.025, 0])
    assert len(simplify_colors(_quads([(200, 120, 120), light]), 16).palette) == 1
    assert len(simplify_colors(_quads([(200, 120, 120), chroma]), 16).palette) == 2


def _old_cluster(lab, w, colors, merge_distance):
    """直す前の _cluster()（比べるための写し）。"""
    m = len(lab)
    dist = simplify_mod._distance(lab, lab)
    kern = np.exp(-0.5 * (dist / simplify_mod.DENSITY_SIGMA) ** 2) * w[None, :]
    members = [[i] for i in range(m)]
    rep = np.arange(m)
    gw = w.astype(np.float64).copy()
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    alive = np.ones(m, bool)
    label = np.arange(m)
    count = m
    while count > 1:
        sub = np.flatnonzero(alive)
        dd = d[np.ix_(sub, sub)]
        if count <= colors and dd.min() >= merge_distance:
            break
        cost = dd * simplify_mod._discount(np.minimum(gw[sub][:, None], gw[sub][None, :]))
        if count <= colors:
            cost = np.where(dd < merge_distance, cost, np.inf)
        k = int(np.argmin(cost))
        i, j = sub[k // len(sub)], sub[k % len(sub)]
        keep, drop = (i, j) if (gw[i], -i) >= (gw[j], -j) else (j, i)
        mem = members[keep] + members[drop]
        members[keep], members[drop] = mem, []
        gw[keep] += gw[drop]
        alive[drop] = False
        label[mem] = keep
        idx = np.array(mem)
        rep[keep] = idx[int(np.argmax(kern[np.ix_(idx, idx)].sum(1)))]
        d[drop, :] = np.inf
        d[:, drop] = np.inf
        nd = dist[rep[keep]].copy()
        nd[~alive] = np.inf
        nd[keep] = np.inf
        d[keep, :] = nd
        d[:, keep] = nd
        count -= 1
    roots = np.flatnonzero(alive)
    remap = np.full(m, -1)
    remap[roots] = np.arange(len(roots))
    return remap[label], rep[roots]


@pytest.mark.parametrize("seed, colors", [(0, 12), (1, 16), (2, 8)])
def test_threshold_zero_is_previous_behavior(seed, colors):
    """しきい値 0 のときは、直す前のクラスタリングと同じ結果になる。"""
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=seed)
    grid = SpritePipeline([BackgroundStep(), GridStep()]).run(big)
    rng = np.random.default_rng(seed)
    noise = _rgba(rng.integers(0, 256, (24, 24, 3)))
    for img in (grid, noise, _rgba(_two_areas(seed))):
        op = img[..., 3] >= 128
        trans = transition_mask(img)
        cols, _, w = simplify_mod._unique_colors(img[..., :3][op], np.where(trans[op], simplify_mod.TRANSITION_WEIGHT, 1.0))
        _, ulab, _, uw = simplify_mod._units(cols, rgb_to_oklab(cols.astype(float)), w, simplify_mod.MAX_CLUSTER_INPUT)
        g_new, r_new, _ = simplify_mod._cluster(ulab, uw, colors, 0.0)
        g_old, r_old = _old_cluster(ulab, uw, colors, 0.0)
        assert np.array_equal(g_new, g_old) and np.array_equal(r_new, r_old)


def test_small_part_colors_survive_with_default_threshold():
    """頬（2 ドット、肌との差 約 0.05）と目（1 ドット）は、既定のしきい値でまとめられない。"""
    rng = np.random.default_rng(7)
    skin = (235, 179, 157)
    cheek = (252, 186, 183)          # ai03 の頬と肌（差 0.055）
    rgb = _noisy((24, 24), skin, 2, rng)
    rgb[:4] = _noisy((4, 24), DARK, 2, rng)
    rgb[12, 6:8] = cheek
    rgb[12, 16] = PIX_EYE
    r = simplify_colors(_rgba(rgb))
    assert _palette_has(r.palette, cheek, 0.01) and _palette_has(r.palette, PIX_EYE, 0.01)
    assert tuple(r.image[12, 6, :3]) == tuple(r.image[12, 7, :3]) and _dist(r.image[12, 6, :3], cheek) < 0.01
    # ノイズで分かれた肌の色・暗い色は、それぞれ 1 色にまとまる
    assert len(r.palette) == 4


def test_close_merge_is_deterministic():
    rng = np.random.default_rng(8)
    img = _rgba(_noisy((20, 20), (120, 160, 90), 6, rng))
    a, b = simplify_colors(img, 16), simplify_colors(img.copy(), 16)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.palette, b.palette)
    cols = rng.integers(0, 256, (200, 3)).astype(np.uint8)
    w = rng.integers(1, 20, 200)
    m1, m2 = merge_close_colors(cols, w, 0.08), merge_close_colors(cols.copy(), w.copy(), 0.08)
    assert np.array_equal(m1.labels, m2.labels) and np.array_equal(m1.colors, m2.colors)


def test_pixel_art_pipeline_with_default_threshold():
    """合成のドット絵（9 色）: 既定の設定（16 色・0.03）で、輪郭・目・頬が残り、上限より少ない色で止まる。"""
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=0)
    grid = SpritePipeline([BackgroundStep(), GridStep()]).run(big)
    r = simplify_colors(grid)
    for c in (PIX_OUTLINE, PIX_EYE, PIX_CHEEK):
        assert _palette_has(r.palette, c, 0.05), c
    assert len(r.palette) <= len(simplify_colors(grid, merge_distance=0).palette)
    lab = rgb_to_oklab(r.palette.astype(float))
    d = simplify_mod._distance(lab, lab)
    np.fill_diagonal(d, 1)
    assert d.min() >= 0.02         # 残った色どうしに、見分けのつかない組がない


# --- merge_close_colors()（パレット統合などから使う） --------------------------------------

def test_merge_close_colors_basic():
    cols = np.array(PURPLES + [CREAM, ORANGE], np.uint8)
    w = np.array([53, 31, 23, 16, 100, 1])
    r = merge_close_colors(cols, w)
    assert len(r.colors) == 3
    assert r.labels[0] == r.labels[1] == r.labels[2] == r.labels[3]
    assert len({r.labels[0], r.labels[4], r.labels[5]}) == 3
    assert tuple(r.colors[r.labels[0]]) == PURPLES[0]       # 重みの大きい、実在の色を残す
    assert r.keep[r.labels[0]] == 0
    assert np.array_equal(r.mapped(cols)[:4], np.repeat(r.colors[r.labels[:1]], 4, 0))
    # 1 ドットの色（ORANGE、重み 1）でも、離れていれば残る
    assert tuple(r.colors[r.labels[5]]) == ORANGE


def test_merge_close_colors_zero_and_empty():
    cols = np.array(PURPLES, np.uint8)
    r = merge_close_colors(cols, threshold=0)
    assert np.array_equal(r.labels, np.arange(4)) and np.array_equal(r.colors, cols)
    e = merge_close_colors(np.zeros((0, 3), np.uint8))
    assert len(e.colors) == 0
    with pytest.raises(ValueError):
        merge_close_colors(cols, threshold=-1)
    with pytest.raises(ValueError):
        merge_close_colors(cols, np.ones(3))
    with pytest.raises(ValueError):
        merge_close_colors(cols[:, :2])


def test_merge_close_colors_does_not_chain_a_gradient():
    """少しずつ違う色が並ぶ階調（隣どうしの差 約 0.01、端から端まで 約 0.15）は、鎖のようにつながって 1 色にはならない。"""
    t = np.linspace(0, 1, 16)[:, None]
    cols = np.round(np.array([200, 80, 80]) * (1 - t) + np.array([250, 170, 120]) * t).astype(np.uint8)
    r = merge_close_colors(cols)
    assert len(r.colors) >= 3
    lab = rgb_to_oklab(r.colors.astype(float))
    d = simplify_mod._distance(lab, lab)
    np.fill_diagonal(d, 1)
    assert d.min() >= DEFAULT_MERGE_DISTANCE


def test_speed_close_merge():
    rng = np.random.default_rng(9)
    img = _rgba(_noisy((96, 96), (120, 160, 90), 20, rng))   # 近い色がたくさん（まとめる回数が多い）
    t = time.perf_counter()
    r = simplify_colors(img, 16)
    assert time.perf_counter() - t < 5
    assert len(r.palette) <= 16
    cols = rng.integers(0, 256, (512, 3)).astype(np.uint8)
    t = time.perf_counter()
    merge_close_colors(cols, threshold=0.1)
    assert time.perf_counter() - t < 2


# ---------------------------------------------------------------- 大きな面の色を守る（keep_large）

HAIR = (215, 238, 253)        # 淡い水色（髪）
SKIN = (252, 241, 236)        # 淡い肌色。OKLab では髪と 0.064 しか離れていない
NAVY = (22, 19, 58)
SMALL_VIVID = [(220, 30, 60), (200, 20, 160), (60, 170, 60), (250, 200, 30), (120, 70, 40), (90, 40, 140),
               (40, 120, 200), (250, 120, 20)]


def _pastel_character():
    """髪（半分強）・輪郭・肌（全体の 17%）と、20 ドット（全体の 0.6%）ずつの鮮やかな小さな色 8 つ。60×60。

    何コマも並べた AI の絵では、境目の混ざった色が 20〜100 ドット（全体の 1% 未満）の塊になり、小ささの割引が効かない。
    それを小さな鮮やかな色で真似る。
    """
    a = np.zeros((60, 60, 4), np.uint8)
    a[..., 3] = 255
    a[..., :3] = HAIR
    a[0:15, :, :3] = NAVY
    a[15:35, 0:30, :3] = SKIN
    for i, c in enumerate(SMALL_VIVID):
        y, x = 40 + (i // 4) * 8, 4 + (i % 4) * 14
        a[y:y + 4, x:x + 5, :3] = c
    return a


def test_large_pastel_area_is_not_absorbed_by_hair():
    """淡い肌の大きな面は、既定（keep_large=0.01）では髪の水色にまとめられない。0 では以前どおりまとめられる。"""
    img = _pastel_character()
    assert simplify_mod.DEFAULT_KEEP_LARGE == 0.01
    new = simplify_colors(img, 8)
    old = simplify_colors(img, 8, keep_large=0)
    assert _palette_has(new.palette, SKIN, 0.01) and _palette_has(new.palette, HAIR, 0.01)
    assert not _palette_has(old.palette, SKIN, 0.03)
    skin = np.zeros(img.shape[:2], bool)
    skin[15:35, 0:30] = True
    assert (new.image[skin][:, :3] == SKIN).all()
    assert len(new.palette) == 8
    # 守った分、小さな色が 1 つ減る（色数の上限は守る）
    assert sum(_palette_has(new.palette, c, 0.01) for c in SMALL_VIVID) == 5


def test_keep_large_does_not_split_near_identical_large_areas():
    """大きな面でも、ほとんど同じ色（距離 LARGE_DISTANCE 未満）の 2 色は守らない（結果は keep_large=0 と同じ）。"""
    img = _pastel_character()
    img[15:35, 0:30, :3] = (200, 226, 243)       # 肌の代わりに、髪とほとんど同じ水色の大きな面
    d = float(simplify_mod._distance(rgb_to_oklab(np.array([HAIR], float)),
                                     rgb_to_oklab(np.array([(200, 226, 243)], float)))[0, 0])
    assert simplify_mod.DEFAULT_MERGE_DISTANCE <= d < simplify_mod.LARGE_DISTANCE
    a = simplify_colors(img, 8)
    b = simplify_colors(img, 8, keep_large=0)
    assert np.array_equal(a.image, b.image)


def test_keep_large_without_pressure_changes_nothing():
    """色数に余裕があるとき（まとめなくても上限以下）は、keep_large の有無で結果は変わらない。"""
    img = _pastel_character()
    assert np.array_equal(simplify_colors(img, 16).image, simplify_colors(img, 16, keep_large=0).image)
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=0)
    grid = SpritePipeline([BackgroundStep(), GridStep()]).run(big)
    # 合成のキャラ（1 枚の絵）では、12 色（前の既定）でも結果が変わらない（ai03・ai05 と同じく、大きな面がもともと残る）
    assert np.array_equal(simplify_colors(grid, 12).image, simplify_colors(grid, 12, keep_large=0).image)


def test_keep_large_validation():
    img = _pastel_character()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="0〜1"):
            simplify_colors(img, 8, keep_large=bad)
        with pytest.raises(ValueError, match="0〜1"):
            SimplifyStep(keep_large=bad)
    assert SimplifyStep().keep_large == 0.01
