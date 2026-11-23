"""retrocore/sprite/cleanup.py（クリーンアップ: 余計なドットを片付ける）のテスト。テスト画像はコードで合成する。"""

import time

import numpy as np
import pytest

from retrocore.sprite import (BackgroundStep, CleanupStep, GridStep, SimplifyStep, SpritePipeline, cleanup_sprite)
from retrocore.sprite.cleanup import (KIND_ALPHA, KIND_BETWEEN, KIND_CORNER, KIND_ISOLATED, KIND_THIN, LEVELS)
from synth_sprite import PIX_CHEEK, PIX_EYE, PIX_OUTLINE, enlarge_like_ai, make_pixel_sprite

OUT = (80, 30, 40)       # 輪郭（暗い赤茶）
RED = (215, 95, 90)      # かさの赤
CREAM = (250, 243, 214)  # 体
PINK = (252, 186, 183)   # 頬
EYE = (70, 50, 60)       # 目
MID = (150, 65, 75)      # 輪郭と赤の中間の暗い赤（ぼかしの名残）
LEVEL_NAMES = ["light", "normal", "strong"]


def _sprite(h=16, w=16, fill=CREAM):
    """透明の背景に、1 ドットの輪郭で囲んだ四角（中は fill）。外側 2 ドットは透明。"""
    a = np.zeros((h, w, 4), np.uint8)
    a[2:h - 2, 2:w - 2, :3] = OUT
    a[2:h - 2, 2:w - 2, 3] = 255
    a[3:h - 3, 3:w - 3, :3] = fill
    return a


def _colors(a):
    return {tuple(int(v) for v in c) for c in a[a[..., 3] >= 128][:, :3]}


def _is(a, y, x, color):
    return a[y, x, 3] == 255 and tuple(int(v) for v in a[y, x, :3]) == tuple(color)


def _outline_connected(a, color=OUT):
    """輪郭の色のドットが（斜めも含めて）1 つにつながっているか。"""
    m = (a[..., 3] >= 128) & np.all(a[..., :3] == color, -1)
    pts = set(zip(*np.nonzero(m)))
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if q in pts and q not in seen:
                    seen.add(q)
                    stack.append(q)
    return len(seen) == len(pts)


# --- 孤立ドット ----------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_isolated_dot_becomes_surrounding_color(level):
    """面の中の、まわりに溶け込む色（少し暗いクリーム）の 1 ドットは、まわりの色になる。"""
    a = _sprite()
    a[7, 8, :3] = (232, 222, 190)
    r = cleanup_sprite(a, level)
    assert _is(r.image, 7, 8, CREAM)
    assert r.count == 1 and r.changed[7, 8] and r.kinds[7, 8] == KIND_ISOLATED
    assert r.counts()["孤立ドット"] == 1


def test_isolated_between_dot_becomes_neighbor_color():
    """2 つの面の境目にある、2 色の中間の色の 1 ドットは、どちらかの面の色になる（新しい色は作らない）。"""
    a = _sprite()
    a[3:13, 3:8, :3] = RED
    a[6, 8, :3] = np.round((np.array(RED) + np.array(CREAM)) / 2)
    r = cleanup_sprite(a, "light")
    assert tuple(r.image[6, 8, :3]) in (RED, CREAM)
    assert _colors(r.image) == {OUT, RED, CREAM}


def test_distinct_single_dot_kept_except_strong():
    """まわりと離れた色の 1 ドット（頬・ハイライト）は light / normal では残し、strong だけで消す。
    まわりよりはっきり暗い 1 ドット（1 ドットの目）は strong でも残す。"""
    a = _sprite()
    a[6, 6, :3] = PINK
    a[9, 10, :3] = EYE
    for level in ("light", "normal"):
        r = cleanup_sprite(a, level)
        assert _is(r.image, 6, 6, PINK) and _is(r.image, 9, 10, EYE) and r.count == 0
    r = cleanup_sprite(a, "strong")
    assert _is(r.image, 6, 6, CREAM) and _is(r.image, 9, 10, EYE)


@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_two_dot_parts_are_kept(level):
    """同じ色が 2 ドット以上つながっていれば部品とみなして残す（縦 2 ドットの目、斜めの 2 ドット、横 2 ドットの頬）。"""
    a = _sprite(20, 20)
    a[6:8, 6, :3] = EYE                 # 縦 2 ドットの目
    a[6, 12, :3] = EYE                  # 斜めにつながった 2 ドットの目
    a[7, 13, :3] = EYE
    a[12, 7:9, :3] = PINK               # 横 2 ドットの頬
    a[12, 12:14, :3] = (240, 225, 195)  # まわりに近い色でも 2 ドットなら孤立ドットではない
    r = cleanup_sprite(a, level)
    for y, x in ((6, 6), (7, 6), (6, 12), (7, 13)):
        assert _is(r.image, y, x, EYE)
    assert _is(r.image, 12, 7, PINK) and _is(r.image, 12, 8, PINK)
    assert r.count == 0 if level != "strong" else True


# --- 中間色の塊 ----------------------------------------------------------------------------

def _rim_sprite():
    """上半分が赤、下半分がクリーム。赤と輪郭の境目（右の輪郭の内側）に中間の暗い赤が縦に 3 ドット。"""
    a = _sprite(20, 20, CREAM)
    a[3:10, 3:17, :3] = RED
    a[4:7, 16, :3] = MID
    return a


def test_between_run_fixed_in_normal_not_light():
    a = _rim_sprite()
    assert not cleanup_sprite(a, "light").changed[4:7, 16].any()   # light は 1 ドットだけ
    for level in ("normal", "strong"):
        r = cleanup_sprite(a, level)
        assert r.changed[4:7, 16].all() and (r.kinds[4:7, 16] == KIND_BETWEEN).all()
        # 輪郭に接する内側のドットは、輪郭を太らせないよう塗りの色にする
        for y in range(4, 7):
            assert _is(r.image, y, 16, RED)
        assert MID not in _colors(r.image)


def test_widely_used_middle_color_is_kept():
    """2 色の中間の色でも、画像の中で広く使われている色（影の面など）は意味のある色として残す。"""
    a = _rim_sprite()
    a[11:16, 4:10, :3] = MID          # 同じ色の大きな面（30 ドット、不透明の 10% 以上）
    r = cleanup_sprite(a, "normal")
    assert not r.changed[4:7, 16].any() and not r.changed[11:16, 4:10].any()


def test_long_between_run_is_kept_in_normal():
    """中間色の塊が大きい（normal の上限 6 ドットより多い）ときは normal では直さない。"""
    a = _sprite(24, 24, CREAM)
    a[3:14, 3:21, :3] = RED
    a[4:12, 20, :3] = MID             # 8 ドット
    assert not cleanup_sprite(a, "normal").changed[4:12, 20].any()
    assert cleanup_sprite(a, "strong").changed[4:12, 20].all()


# --- 透明 ----------------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_alpha_speck_and_hole(level):
    a = _sprite(20, 20)
    a[0, 0] = (*CREAM, 255)          # 外側に浮いた 1 ドット
    a[8, 8, 3] = 0                   # 内側の 1 ドットの穴
    a[17, 5:7] = (*OUT, 255)         # 外側に浮いた 2 ドット（部品とみなして残す）
    r = cleanup_sprite(a, level)
    assert r.image[0, 0, 3] == 0 and r.kinds[0, 0] == KIND_ALPHA
    assert _is(r.image, 8, 8, CREAM) and r.kinds[8, 8] == KIND_ALPHA
    assert (r.image[17, 5:7, 3] == 255).all()
    # それ以外のシルエットは変わらない
    alpha = a[..., 3].copy()
    alpha[0, 0], alpha[8, 8] = 0, 255
    assert np.array_equal(r.image[..., 3], alpha)


def test_silhouette_notch_is_not_changed():
    """輪郭の外側の 1 ドットのへこみ（透明）はシルエットの形なので埋めない。"""
    a = _sprite(20, 20)
    a[2, 9, 3] = 0
    r = cleanup_sprite(a, "strong")
    assert r.image[2, 9, 3] == 0


# --- 輪郭 ----------------------------------------------------------------------------------

def _stair_sprite():
    """左上が階段になった輪郭。階段の角に L 字の余分な 1 ドット（ダブルコーナー）がある。"""
    a = np.zeros((14, 14, 4), np.uint8)
    inside = np.zeros((14, 14), bool)
    for y in range(2, 12):
        x0 = max(2, 7 - (y - 2))     # 上ほど右から始まる（左上が斜め）
        inside[y, x0:12] = True
    a[inside] = (*CREAM, 255)
    # 輪郭: 不透明の部分のうち、4 近傍に透明があるドット
    pad = np.pad(inside, 1)
    edge = inside & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    a[edge, :3] = OUT
    # 階段の角に余分な 1 ドット（内側に）: (4, 6) は上 (3, 6) と左 (4, 5) が輪郭の L 字の角になる
    a[4, 6, :3] = OUT
    return a


def test_corner_double_removed_in_normal():
    a = _stair_sprite()
    assert _outline_connected(a)
    assert not cleanup_sprite(a, "light").changed.any()
    r = cleanup_sprite(a, "normal")
    assert _is(r.image, 4, 6, CREAM) and r.kinds[4, 6] == KIND_CORNER
    assert _outline_connected(r.image)
    assert np.array_equal(r.image[..., 3], a[..., 3])


def test_thick_outline_thinned_in_strong_only():
    """右の輪郭が縦 5 ドットだけ 2 ドットの太さ。strong で内側の列を塗りの色にし、輪郭はつながったまま。"""
    a = _sprite(20, 20)
    a[6:11, 16, :3] = OUT
    # normal は両端の L 字の角だけ（太い所の途中は変えない）
    assert not cleanup_sprite(a, "normal").changed[7:10, 16].any()
    r = cleanup_sprite(a, "strong")
    assert r.changed[6:11, 16].all() and (r.kinds[7:10, 16] == KIND_THIN).all()
    assert set(r.kinds[6:11, 16].tolist()) <= {KIND_THIN, KIND_CORNER}
    assert _is(r.image, 8, 16, CREAM) and _is(r.image, 8, 17, OUT)
    assert _outline_connected(r.image)
    assert np.array_equal(r.image[..., 3], a[..., 3])


def test_outline_on_silhouette_edge_kept():
    """シルエットの縁で輪郭の代わりになっている中間色のドットは、輪郭の色にする（縁の輪郭を途切れさせない）。"""
    a = _sprite(20, 20, RED)
    a[2, 9, :3] = MID
    r = cleanup_sprite(a, "light")
    assert _is(r.image, 2, 9, OUT)


# --- 目・口・ハイライト・影を守る（ai04・ai05 の確認で見つかった問題） ---------------------------

SKIN = (254, 233, 197)      # 肌
HILITE = (242, 242, 242)    # 目のハイライト（肌に近い明るさの白）


def _face(h=20, w=24):
    """1 ドットの輪郭で囲んだ肌の面（輪郭は透明に接する）。目・口は輪郭と同じ色で描く。"""
    return _sprite(h, w, SKIN)


def _draw(a, y0, x0, rows, color=OUT):
    """rows（"#" の所）を (y0, x0) から color で描く。"""
    for dy, row in enumerate(rows):
        for dx, ch in enumerate(row):
            if ch == "#":
                a[y0 + dy, x0 + dx, :3] = color


ROUND_EYE = [".##.", "####", "####", ".##."]   # 4×4 の角を落とした丸い目
PLUS_EYE = [".#.", "###", ".#."]                # 3×3 の角を落とした目
TALL_EYE = ["##", "##", "##"]                   # 2×3 の目
WIDE_MOUTH = ["###", "###"]                     # 3×2 の口


@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_eyes_and_mouth_in_outline_color_are_kept(level):
    """輪郭と同じ色の目・口（まわりを肌で囲まれた塊）は、輪郭の角の処理で削らない（問題 1）。"""
    a = _face(22, 26)
    _draw(a, 5, 5, ROUND_EYE)
    _draw(a, 5, 12, PLUS_EYE)
    _draw(a, 5, 18, TALL_EYE)
    _draw(a, 13, 11, WIDE_MOUTH)
    r = cleanup_sprite(a, level)
    assert r.count == 0
    assert np.array_equal(r.image, a)


def test_corner_of_silhouette_outline_still_removed_next_to_eye():
    """目を守っても、シルエットの輪郭（透明に接する線）の L 字の角は今まで通り直す。"""
    a = _stair_sprite()
    big = np.zeros((14, 20, 4), np.uint8)
    big[:, :14] = a
    _draw(big, 6, 8, ["##", "##"])       # 塗りの中の 2×2 の目（輪郭の色）
    r = cleanup_sprite(big, "normal")
    assert _is(r.image, 4, 6, CREAM) and r.kinds[4, 6] == KIND_CORNER
    assert all(_is(r.image, y, x, OUT) for y in (6, 7) for x in (8, 9))


@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_highlight_in_dark_eye_is_kept(level):
    """暗い目の中の 1 ドットの白いハイライトは消さない（問題 2）。ハイライトに近い明るい色（肌）が
    斜めに接していても、上下左右がすべて目の色なら、肌に「溶け込む」とはみなさない。"""
    a = _face(20, 24)
    _draw(a, 5, 5, ROUND_EYE)
    a[6, 6, :3] = HILITE                 # 目の左上。斜め (5, 5) は肌
    _draw(a, 5, 14, ROUND_EYE)
    a[6, 15, :3] = HILITE
    r = cleanup_sprite(a, level)
    assert _is(r.image, 6, 6, HILITE) and _is(r.image, 6, 15, HILITE)
    assert not r.changed[5:9, 5:9].any()


def test_bright_dot_on_dark_fill_kept_and_blending_dot_still_fixed():
    """まわり（上下左右）よりはっきり明るい 1 ドットは strong でも残す。まわりの色に近い 1 ドットは今まで通り直す。"""
    a = _sprite(16, 16, (90, 70, 120))   # 暗い紫の面
    a[5, 5, :3] = HILITE
    a[9, 9, :3] = (98, 76, 128)          # まわりに溶け込む色
    r = cleanup_sprite(a, "strong")
    assert _is(r.image, 5, 5, HILITE)
    assert _is(r.image, 9, 9, (90, 70, 120))


WHITE = (242, 242, 242)     # 剣の白
LGRAY = (209, 208, 211)     # 剣の薄い灰色
SHADE = (169, 167, 175)     # 剣の灰色の影（白・薄い灰色より暗い。色だけ見れば輪郭と白の中間）
DARK = (70, 33, 18)         # 輪郭


@pytest.mark.parametrize("n", [2, 3, 4])
def test_gray_shadow_line_between_light_colors_is_kept(n):
    """白と薄い灰色の間に縦に並んだ灰色の影（2 ドット以上）は、中間色として片付けない（問題 4）。"""
    a = np.zeros((14, 9, 4), np.uint8)
    a[1:13, 1:8] = (*DARK, 255)
    a[2:12, 2:4, :3] = WHITE
    a[2:12, 4, :3] = WHITE
    a[2:12, 5:7, :3] = LGRAY
    a[2:2 + n, 4, :3] = SHADE            # 白 | 影 | 薄い灰色。上の端は輪郭に接する（縦に見ると輪郭と白に挟まれる）
    for level in ("light", "normal"):
        r = cleanup_sprite(a, level)
        assert all(_is(r.image, y, 4, SHADE) for y in range(2, 2 + n)), level


def test_between_run_fixed_when_squeezed_between_two_colors():
    """2 色に横から挟まれた中間色の線（ai03 の輪郭と赤の間の暗い赤のような）は今まで通り片付ける。"""
    a = _sprite(20, 20, CREAM)
    a[3:10, 3:17, :3] = RED
    a[5, 13:17, :3] = MID                # 赤 | 暗い赤 ×4 | 輪郭（横に並ぶ）
    r = cleanup_sprite(a, "normal")
    assert MID not in _colors(r.image)


# --- 全体 ----------------------------------------------------------------------------------

def _noisy_sprite(seed=0):
    """合成のドット絵（9 色）に、別の色の点（既存の色・中間の色・離れた色）をばらまいたもの。"""
    rng = np.random.default_rng(seed)
    art = make_pixel_sprite(32, 34)
    op = np.argwhere(art[..., 3] > 0)
    cols = sorted(_colors(art))
    extra = [np.round((np.array(cols[i]) + np.array(cols[j])) / 2) for i, j in ((0, 3), (2, 5), (1, 6))]
    for y, x in op[rng.choice(len(op), 40, replace=False)]:
        art[y, x, :3] = (cols + extra)[rng.integers(len(cols) + len(extra))]
    return art


@pytest.mark.parametrize("level", LEVEL_NAMES)
def test_no_new_colors_and_deterministic(level):
    a = _noisy_sprite()
    before = a.copy()
    r1 = cleanup_sprite(a, level)
    r2 = cleanup_sprite(a, level)
    assert np.array_equal(a, before)                       # 入力は書き換えない
    assert np.array_equal(r1.image, r2.image) and np.array_equal(r1.kinds, r2.kinds)
    assert _colors(r1.image) <= _colors(a)                  # 新しい色は増えない
    assert r1.count == int((np.any(r1.image != a, -1)).sum())
    assert r1.count > 0


def test_stronger_levels_change_more():
    a = _noisy_sprite(1)
    counts = [cleanup_sprite(a, lv).count for lv in LEVEL_NAMES]
    assert counts[0] <= counts[1] <= counts[2]


def test_off_changes_nothing():
    a = _noisy_sprite()
    r = cleanup_sprite(a, "off")
    assert np.array_equal(r.image, a) and r.count == 0
    step = CleanupStep("off")
    assert np.array_equal(step.apply(a), a) and "off" in step.describe()


def test_clean_sprite_is_unchanged():
    """余計なドットの無いドット絵（合成のキノコ）は、どの強さでもほとんど変わらない（1 ドットの目と、輪郭に接した
    2 ドットの頬は残る）。"""
    art = make_pixel_sprite()
    for level in LEVEL_NAMES:
        r = cleanup_sprite(art, level)
        assert PIX_EYE in _colors(r.image) and PIX_CHEEK in _colors(r.image)
        if level != "strong":
            assert r.count <= 2


def test_all_transparent_and_invalid():
    a = np.zeros((5, 5, 4), np.uint8)
    assert np.array_equal(cleanup_sprite(a).image, a)
    with pytest.raises(ValueError):
        cleanup_sprite(a, "max")
    with pytest.raises(ValueError):
        cleanup_sprite(a[..., :3])
    with pytest.raises(ValueError):
        CleanupStep("max")
    assert set(LEVELS) == {"off", "light", "normal", "strong"}


def test_step_describe():
    step = CleanupStep()
    assert step.level == "normal"
    step.apply(_rim_sprite())
    d = step.describe()
    assert "クリーンアップ" in d and "normal" in d and "中間色 3" in d


# --- ドット絵風の画像を通しで -----------------------------------------------------------------

def test_pixel_art_pipeline_keeps_parts():
    """合成のドット絵を AI 風に拡大して 背景除去 → グリッド検出 → 単純化 → クリーンアップ。
    目（1 ドット）・頬（2 ドット）・輪郭は残り、色は増えず、シルエットはほぼ変わらない。"""
    art = make_pixel_sprite()
    big, _, _ = enlarge_like_ai(art, seed=0)
    simp = SpritePipeline([BackgroundStep(), GridStep(), SimplifyStep(12)]).run(big)
    step = CleanupStep()
    out = step.apply(simp)
    assert _colors(out) <= _colors(simp)

    def near(a, color, tol=60):
        return any(np.abs(np.array(c) - color).max() <= tol for c in _colors(a))

    for c in (PIX_OUTLINE, PIX_EYE, PIX_CHEEK):
        assert near(simp, c) == near(out, c), c
    assert (out[..., 3] != simp[..., 3]).sum() <= 2
    assert step.last.count < 0.1 * (simp[..., 3] >= 128).sum()


def test_speed():
    rng = np.random.default_rng(3)
    cols = rng.integers(0, 256, (16, 3))
    a = np.zeros((200, 200, 4), np.uint8)
    a[..., :3] = cols[rng.integers(0, 16, (200, 200))]
    a[..., 3] = 255
    a[:10] = 0
    # 200×200 の 16 色ばらばらの画像（孤立ドット・塊が非常に多い、いちばん重い部類の入力）を 3 つの強さで直す時間。
    # 上限の考え方: 速くした後は手元で合計 約 1.6 秒。GitHub の Windows の作業用パソコンでは手元の約 1.7 倍
    # （速くする前の PR で 手元 6.4 秒・CI 10.2 秒）なので 約 2.7 秒の見込み。混み具合で 2 倍遅くなっても通るよう、
    # その約 2.5 倍の 7 秒を上限にする。速くする前のコード（手元 5.5〜6.4 秒、CI 約 9〜10 秒）に戻ったら落ちる。
    # 一時的な混み具合の影響を減らすため、2 回測って速いほうを使う（1 回だけ遅くても落ちない）
    best = float("inf")
    for _ in range(2):
        t = time.perf_counter()
        for level in LEVEL_NAMES:
            cleanup_sprite(a, level)
        best = min(best, time.perf_counter() - t)
    assert best < 7, f"{best:.2f} 秒"
