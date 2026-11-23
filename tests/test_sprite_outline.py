"""retrocore/sprite/outline.py（縁取り）のテスト。テスト画像はコードで合成する。"""

import numpy as np
import pytest

from retrocore.sprite import (CleanupStep, OutlineStep, SimplifyStep, SpritePipeline, detect_outline,
                              outline_sprite)
from retrocore.sprite.outline import (DEFAULT_MAX_COLORS, KIND_ADDED, KIND_GAP, KIND_THIN, darken_rgb, oklab_to_rgb,
                                      outer_ring)
from retrocore.sprite.simplify import rgb_to_oklab

OUT = (80, 30, 40)       # 輪郭（暗い赤茶）
RED = (215, 95, 90)      # かさの赤
CREAM = (250, 243, 214)  # 体
SKIN = (235, 179, 157)   # 肌
NAVY = (45, 60, 95)
COAT = (50, 45, 48)      # 黒い服


def _blank(h, w):
    return np.zeros((h, w, 4), np.uint8)


def _fill(a, mask, color):
    a[mask, :3] = color
    a[mask, 3] = 255


def _rect(h=14, w=14, box=(4, 10, 4, 10), color=CREAM):
    """透明の背景に塗りの四角（輪郭なし）。box は (上, 下, 左, 右)（右・下は含まない）。"""
    a = _blank(h, w)
    t, b, l, r = box
    m = np.zeros((h, w), bool)
    m[t:b, l:r] = True
    _fill(a, m, color)
    return a


def _disk(h=21, w=21, r=7.3, color=RED):
    yy, xx = np.mgrid[0:h, 0:w]
    m = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r * r
    a = _blank(h, w)
    _fill(a, m, color)
    return a, m


def _l_shape_outlined(fill=CREAM):
    """1 ドットの暗い輪郭で囲んだ L 字（へこんだ角がある）。"""
    h, w = 18, 18
    m = np.zeros((h, w), bool)
    m[2:16, 2:8] = True
    m[10:16, 2:16] = True
    a = _blank(h, w)
    _fill(a, m, OUT)
    inner = m.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        inner &= np.roll(np.roll(m, dy, 0), dx, 1)
    _fill(a, inner, fill)
    return a


def _is(a, y, x, color):
    return a[y, x, 3] == 255 and tuple(int(v) for v in a[y, x, :3]) == tuple(color)


def _opaque(a):
    return a[..., 3] >= 128


def _colors(a):
    return {tuple(int(v) for v in c) for c in a[_opaque(a)][:, :3]}


# --- add: 外側に 1 ドットの輪郭 ---------------------------------------------------------------

def test_add_square_thin_corners_not_thick():
    a = _rect()
    r = outline_sprite(a, "add")
    assert r.action == "add" and not r.padded and r.image.shape == a.shape
    ring = r.changed
    assert ring.sum() == 4 * 6                          # 四辺 6 ドットずつ。角（斜め）は付けない
    for y, x in ((3, 3), (3, 10), (10, 3), (10, 10)):
        assert r.image[y, x, 3] == 0
    assert (r.image[3, 4:10, 3] == 255).all() and (r.image[4:10, 10, 3] == 255).all()
    assert (r.kinds[ring] == KIND_ADDED).all()


def test_add_square_thick_fills_corners():
    r = outline_sprite(_rect(), "add", shape="thick")
    assert r.changed.sum() == 4 * 6 + 4
    for y, x in ((3, 3), (3, 10), (10, 3), (10, 10)):
        assert r.image[y, x, 3] == 255


def test_add_disk_is_one_dot_line_without_2x2_blocks():
    a, m = _disk()
    r = outline_sprite(a, "add")
    ring = r.changed
    # どの 2×2 も全部が輪郭ではない（角が 2 ドットの太さにならない）
    blocks = ring[:-1, :-1] & ring[1:, :-1] & ring[:-1, 1:] & ring[1:, 1:]
    assert not blocks.any()
    # 輪郭のどのドットも上下左右に元の絵がある（斜めだけで接する余分な角のドットが無い）
    near4 = np.zeros_like(m)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        near4 |= np.roll(np.roll(m, dy, 0), dx, 1)
    assert (near4[ring]).all()
    # 輪郭で囲まれていて、元の絵が透明に上下左右で接していない
    op = _opaque(r.image)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        assert not (m & ~np.roll(np.roll(op, dy, 0), dx, 1)).any()


def test_add_only_outside_transparent_and_inside_unchanged():
    a, m = _disk()
    a[9:12, 9:12, :3] = CREAM   # 中に別の色
    r = outline_sprite(a, "add")
    assert np.array_equal(r.image[m], a[m])            # 絵の中は変わらない
    assert not (r.changed & m).any()                   # 変わったのは元が透明だったドットだけ
    assert (a[r.changed, 3] == 0).all()


def test_add_pads_when_touching_border():
    a = _rect(10, 10, (0, 6, 3, 10))                 # 上と右の端に接する
    r = outline_sprite(a, "add")
    assert r.padding == (1, 0, 0, 1) and r.padded
    assert r.image.shape == (11, 11, 4)
    assert np.array_equal(r.image[1:, :10][_opaque(a)], a[_opaque(a)])
    assert (r.image[0, 3:10, 3] == 255).all()          # 足した余白に輪郭
    r2 = outline_sprite(a, "add", pad=False)
    assert r2.padding == (0, 0, 0, 0) and r2.image.shape == a.shape
    assert r2.changed.sum() == 6 + 7                   # 左と下だけ


def test_add_on_existing_outline_doubles_but_auto_does_not():
    a = _l_shape_outlined()
    added = outline_sprite(a, "add")
    assert added.changed.sum() > 0
    auto = outline_sprite(a, "auto")
    assert auto.detection.has_outline and auto.action == "fill"
    assert auto.count == 0 and np.array_equal(auto.image, a)   # 二重にならない（何も変えない）


def test_empty_and_fully_transparent():
    a = _blank(5, 5)
    for mode in ("auto", "add", "off"):
        r = outline_sprite(a, mode)
        assert np.array_equal(r.image, a) and r.count == 0


# --- auto: 判定と切れ目 --------------------------------------------------------------------

def test_detect_outline_criteria():
    d = detect_outline(_l_shape_outlined())
    assert d.has_outline and d.coverage >= 0.99 and d.colors[0] == OUT
    d = detect_outline(_rect(16, 16, (2, 14, 2, 14), COAT))   # 暗い広い面（黒い服）は輪郭ではない
    assert not d.has_outline and d.coverage == 0.0
    a = _rect(20, 20, (2, 18, 2, 18), NAVY)
    a[3:17, 3:17, :3] = CREAM                                # 暗い 1 ドットの縁 → 輪郭あり
    assert detect_outline(a).has_outline
    a[2:4, 2:18, :3] = NAVY
    a[2:18, 2:4, :3] = NAVY                                  # 2 ドットの太さでも輪郭あり
    assert detect_outline(a).has_outline


def test_auto_adds_when_no_outline():
    a = _rect()
    r = outline_sprite(a, "auto")
    assert not r.detection.has_outline and r.action == "add"
    assert np.array_equal(r.image, outline_sprite(a, "add").image)


@pytest.mark.parametrize("gap", [1, 2])
def test_auto_fills_gap_in_outline(gap):
    a = _l_shape_outlined()
    a[2, 4:4 + gap, :3] = CREAM                      # 上の辺の輪郭を 1〜2 ドット途切れさせる
    a[12:12 + gap, 15, :3] = SKIN                    # 右の辺も（別の色）
    r = outline_sprite(a, "auto")
    assert r.detection.has_outline and r.action == "fill"
    assert r.count == 2 * gap and (r.kinds[r.changed] == KIND_GAP).all()
    assert np.array_equal(r.image, _l_shape_outlined())   # 元の輪郭に戻る
    assert r.new_colors == [] and r.image.shape == a.shape


def test_auto_does_not_fill_three_dot_gap_or_inner_line():
    a = _l_shape_outlined()
    a[2, 3:6, :3] = CREAM                            # 3 ドットの切れ目は埋めない
    # シルエットの輪郭とつながっていない内側の線（切れ目あり）は対象外
    a[12, 5:8, :3] = OUT
    a[12, 9:12, :3] = OUT
    r = outline_sprite(a, "auto")
    assert r.detection.has_outline and r.count == 0


def test_auto_fills_inner_line_connected_to_silhouette():
    """シルエットの輪郭から伸びる線（体と腕の境目など）の端と、別の輪郭の間の 1 ドットの切れ目を埋める。"""
    a = _l_shape_outlined()
    a[12:15, 10, :3] = OUT                           # 下の辺（行 15）から上へ伸びる線。上の辺（行 10）まで 1 ドット足りない
    r = outline_sprite(a, "auto")
    assert r.count == 1 and _is(r.image, 11, 10, OUT)   # まっすぐ続くドットを埋める


GREEN = (121, 169, 60)   # マントの緑


def test_auto_does_not_fill_path_across_fill_area():
    """内側の線の端から別の輪郭まで、塗りの面を横切って 2 つに分ける道は埋めない（ai04 のマントの内側の影。問題 5）。
    シルエットの縁の切れ目は今まで通り埋める。"""
    a = _l_shape_outlined(GREEN)
    a[6, 2:5, :3] = OUT                  # 左の輪郭から右へ伸びる内側の線。右の輪郭（列 7）まで 2 ドット
    r = outline_sprite(a, "auto")
    assert r.detection.has_outline and r.count == 0
    a[2, 4, :3] = GREEN                  # シルエットの縁の切れ目（1 ドット）は埋める
    r = outline_sprite(a, "auto")
    assert r.count == 1 and _is(r.image, 2, 4, OUT)


def test_auto_does_not_fill_path_from_inner_line_across_fill_area():
    """下の辺から斜めに上がって横に伸びる内側の線の先から、上の輪郭までの 2 ドットの道も、面を分けるので埋めない。"""
    a = _l_shape_outlined(GREEN)
    a[13, 9:12, :3] = OUT                # 下の辺から離れた、横の内側の線（左端は下の辺の輪郭につながる）
    a[14, 8, :3] = OUT
    r = outline_sprite(a, "auto")
    assert not r.changed[10:13].any()


# --- 色 --------------------------------------------------------------------------------------

def test_darken_keeps_hue():
    dr = darken_rgb(RED)
    assert dr[0] > dr[1] and dr[0] > dr[2]                       # 赤の隣は暗い赤
    ds = darken_rgb(SKIN)
    assert ds[0] > ds[1] > ds[2]                                 # 肌の隣は暗い茶色
    for c in (RED, SKIN, CREAM, NAVY, COAT):
        L0 = rgb_to_oklab(np.array(c, float))[0]
        L1 = rgb_to_oklab(np.array(darken_rgb(c), float))[0]
        assert L1 < L0 - 0.01 and L1 <= 0.45


def test_oklab_roundtrip():
    rgb = np.array([[215, 95, 90], [0, 0, 0], [255, 255, 255], [45, 60, 95]], float)
    assert np.abs(oklab_to_rgb(rgb_to_oklab(rgb)).astype(int) - rgb).max() <= 1


def test_darken_colors_follow_inside_color():
    a = _rect(14, 20, (4, 10, 4, 16), RED)
    a[4:10, 10:16, :3] = SKIN
    r = outline_sprite(a, "add")
    left, right = tuple(int(v) for v in r.image[3, 5, :3]), tuple(int(v) for v in r.image[3, 14, :3])
    assert left != right
    assert left[0] > left[1] and right[0] > right[1] > right[2]
    assert len(r.new_colors) == 2


def test_darken_limits_new_colors():
    rng = np.random.default_rng(0)
    a = _blank(10, 70)
    for i in range(12):                                # 12 色の縦じま（どれも上下で透明に接する）
        a[3:7, 3 + 5 * i:8 + 5 * i, :3] = rng.integers(60, 255, 3)
        a[3:7, 3 + 5 * i:8 + 5 * i, 3] = 255
    r = outline_sprite(a, "add")
    assert 1 <= len(r.outline_colors) <= DEFAULT_MAX_COLORS
    assert len(_colors(r.image) - _colors(a)) == len(r.new_colors) <= DEFAULT_MAX_COLORS
    r1 = outline_sprite(a, "add", max_colors=1)
    assert len(r1.outline_colors) == 1


def test_single_color_auto_and_given():
    a = _rect(14, 20, (4, 10, 4, 16), RED)
    a[6, 8:10, :3] = OUT                               # いちばん暗い色
    r = outline_sprite(a, "add", color_mode="single")
    assert r.outline_colors == [OUT] and r.new_colors == []
    assert all(_is(r.image, y, x, OUT) for y, x in zip(*np.nonzero(r.changed)))
    r = outline_sprite(a, "add", color=(0x3a, 0x20, 0x30))
    assert r.outline_colors == [(0x3a, 0x20, 0x30)] and r.new_colors == [(0x3a, 0x20, 0x30)]
    # いちばん暗い色が明るいときは、それを暗くした色
    r = outline_sprite(_rect(color=CREAM), "add", color_mode="single")
    assert r.outline_colors == [darken_rgb(CREAM)]


# --- そのほか -------------------------------------------------------------------------------

def test_off_changes_nothing():
    for a in (_rect(), _l_shape_outlined()):
        r = outline_sprite(a, "off")
        assert np.array_equal(r.image, a) and r.count == 0 and r.action == "none"
        assert np.array_equal(OutlineStep(mode="off").apply(a), a)


def test_deterministic_and_input_not_modified():
    a = _l_shape_outlined()
    a[2, 4, :3] = CREAM
    b, _ = _disk()
    for img in (a, b):
        keep = img.copy()
        for mode in ("auto", "add"):
            r1, r2 = outline_sprite(img, mode), outline_sprite(img.copy(), mode)
            assert np.array_equal(r1.image, r2.image)
        assert np.array_equal(img, keep)


def test_outer_ring_shapes():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    assert outer_ring(m, "thin").sum() == 4 and outer_ring(m, "thick").sum() == 8


def test_step_in_pipeline_and_describe():
    a = _rect()
    st = OutlineStep()
    out = SpritePipeline([SimplifyStep(colors=4), CleanupStep(), st]).run(a)
    assert st.last.action == "add" and out.shape == a.shape
    text = st.describe()
    assert "縁取り" in text and "輪郭なし" in text and "余白なし" in text and "増えた色" in text
    with pytest.raises(ValueError):
        OutlineStep(mode="x")
    with pytest.raises(ValueError):
        outline_sprite(a, shape="x")


# --- thin: 太すぎる外側の輪郭を減らす ------------------------------------------------------------

def _thick_outlined(h=20, w=20, box=(3, 17, 3, 17), width=2, fill=CREAM):
    """width ドットの太さの暗い輪郭で囲んだ四角。"""
    t, b, l, r = box
    a = _blank(h, w)
    m = np.zeros((h, w), bool)
    m[t:b, l:r] = True
    _fill(a, m, OUT)
    inner = np.zeros((h, w), bool)
    inner[t + width:b - width, l + width:r - width] = True
    _fill(a, inner, fill)
    return a, m, inner


def _light_touches_clear(a):
    """透明（か画像の外）に上下左右で触れている、輪郭の色でない不透明なドットの数。"""
    op = _opaque(a)
    p = np.pad(op, 1)
    near = np.zeros_like(op)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        near |= ~p[1 + dy:1 + dy + op.shape[0], 1 + dx:1 + dx + op.shape[1]]
    dark = (a[..., :3] == OUT).all(-1)
    return int((op & near & ~dark).sum())


def test_thin_makes_two_dot_outline_one_dot():
    a, m, inner = _thick_outlined(width=2)
    r = outline_sprite(a, "thin")
    assert r.action == "thin" and r.detection.has_outline
    op = _opaque(r.image)
    # 外側の段（角も含む）だけが透明になり、中身を囲む 1 ドットの線が残る
    expect = np.zeros_like(m)
    expect[4:16, 4:16] = True
    assert np.array_equal(op, expect)
    assert (r.kinds == KIND_THIN).sum() == m.sum() - expect.sum() == r.count
    assert _light_touches_clear(r.image) == 0
    assert _colors(r.image) <= _colors(a)                 # 新しい色を作らない
    assert np.array_equal(r.image[inner], a[inner])        # 中身は変えない


def test_thin_keeps_one_dot_and_three_dot_outlines():
    for width in (1, 3):
        a, _, _ = _thick_outlined(width=width)
        r = outline_sprite(a, "thin")
        assert r.count == 0 and np.array_equal(r.image, a), width


def test_thin_keeps_dark_parts_without_light_inside():
    """中身の無い暗い部品（2 ドット幅の暗い髪の束）は削らない。明るい細い部品は残り、透明に触れない。"""
    a, _, _ = _thick_outlined(h=24, w=24, box=(8, 22, 3, 17))
    a[1:8, 6:8, :3] = OUT        # 上に伸びた 2 ドット幅の暗い束（中身なし）
    a[1:8, 6:8, 3] = 255
    a[2:8, 11:15, :3] = OUT      # 上に伸びた、1 ドット幅の明るい束（2 ドットの輪郭つき）
    a[2:8, 11:15, 3] = 255
    a[4:10, 12:13, :3] = CREAM
    r = outline_sprite(a, "thin")
    assert _opaque(r.image)[1:8, 6:8].all()                    # 暗い束はそのまま
    assert (r.image[4:10, 12, :3] == CREAM).all()              # 明るい束の中身は残る
    assert _light_touches_clear(r.image) == 0
    assert r.count > 0


def test_thin_does_not_change_auto_add_off():
    a, _, _ = _thick_outlined()
    auto = outline_sprite(a, "auto")
    assert auto.action == "fill" and auto.count == 0
    l = _l_shape_outlined()
    assert np.array_equal(outline_sprite(l, "thin").image, outline_sprite(l, "auto").image)   # 1 ドットの輪郭は同じ


def test_thin_fills_gaps_like_auto():
    a = _l_shape_outlined()
    a[15, 5, :3] = CREAM          # 下の辺の輪郭に 1 ドットの切れ目
    auto = outline_sprite(a, "auto")
    thin = outline_sprite(a, "thin")
    assert (auto.kinds == KIND_GAP).sum() >= 1
    assert np.array_equal(thin.image, auto.image) and np.array_equal(thin.kinds, auto.kinds)


def test_thin_without_outline_adds_only_outside_light_edges():
    """輪郭が無い絵: 明るい縁の外側には付け、すでに暗い縁の外側には付けない。"""
    a = _rect(h=16, w=16, box=(4, 12, 4, 12), color=CREAM)
    a[4:12, 4, :3] = OUT          # 左の辺だけ暗い（輪郭は「なし」の判定になる）
    det = detect_outline(a)
    assert not det.has_outline
    add = outline_sprite(a, "add")
    thin = outline_sprite(a, "thin")
    assert thin.action == "add"
    assert add.changed[4:12, 3].all() and not thin.changed[4:12, 3].any()   # 暗い左の辺の外には付けない
    assert np.array_equal(thin.changed[:, 5:], add.changed[:, 5:])          # ほかは add と同じ


def test_thin_deterministic_and_input_not_modified():
    a, _, _ = _thick_outlined()
    keep = a.copy()
    r1 = outline_sprite(a, "thin")
    r2 = outline_sprite(a, "thin")
    assert np.array_equal(a, keep) and np.array_equal(r1.image, r2.image)
    st = OutlineStep(mode="thin")
    st.apply(a)
    assert "2 ドットの太さの外側の輪郭を" in st.describe()
