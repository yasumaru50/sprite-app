"""32 ビットの NumPy（ブラウザ版の Pyodide は wasm32）でだけ起きる型のエラーが、Sprite App の変換に無いことを確かめる。

32 ビットでは配列の番号の型（intp）が int32 なので、`np.bincount` などに int64 の配列を渡すと
`Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'` になる。
64 ビットの CI ではこのエラーが起きないので、2 つの方法で確かめる。

1. 動かして確かめる: `tests/int32_numpy.py` の `emulate_int32()` の中で `convert_sprite()` を通す
   （32 ビットで int64 になる配列を `np.bincount` などに渡すとエラーにする）。Linux・macOS でだけ動く（Windows では飛ばす。
   CI では Linux のジョブで動かす）
2. 書いてある場所で確かめる: 上の仕組みは `//`・`%`・ビット演算を通ると int64 の印が消えて見落とすことがあるので、
   `np.bincount` などを呼んでいる場所を一覧（`REVIEWED`）と比べる。新しく呼ぶ場所を足したら、渡す配列が 32 ビットでも
   intp になるか（`np.arange`・`np.unique` の番号などか、`.astype(np.intp)` したものか）を確かめてから一覧に足す
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from int32_numpy import AVAILABLE, INT32_NUMPY, Int32CastError, emulate_int32
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import convert_for_web, encode_png, motion_for_web
from retrocore.sprite.motion import make_motion, save_sheet
from synth_sprite import add_noise, enlarge_like_ai, make_character, make_pixel_sprite

ROOT = Path(__file__).resolve().parents[1]
SPRITE = ROOT / "retrocore" / "sprite"

# 1 の仕組みは、intp と int64 の印を区別できる環境（Linux・macOS）でだけ動く。Windows の CI では 2 だけを確かめ、
# 1 は .github/workflows/tests.yml の int32-check のジョブで確かめる
needs_emulation = pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")


# ---- 仕組みそのもののテスト ----

def test_emulation_available_on_linux():
    """Linux（CI の int32-check のジョブ）では、1 の仕組みが必ず動く（飛ばされて確かめたつもりにならないように）。"""
    if sys.platform.startswith("linux"):
        assert AVAILABLE


@needs_emulation
def test_emulation_flags_int64_and_allows_intp():
    np32 = INT32_NUMPY
    i64 = np32.arange(4).astype(np32.int64)
    with pytest.raises(Int32CastError, match="int64"):
        np32.bincount(i64)
    with pytest.raises(Int32CastError):
        np32.bincount(np32.zeros(3, np32.int64) * 2 + np32.arange(3))   # int64 と intp が混ざると int64
    with pytest.raises(Int32CastError):
        np32.repeat(np.arange(4), i64)
    with pytest.raises(Int32CastError):
        np32.minimum.reduceat(np.arange(4.0), i64)
    with pytest.raises(Int32CastError):
        np32.bincount(np.arange(3, dtype=np.uint32))                  # 32 ビットでは uint32 も int32 にできない
    # intp（np.arange・np.unique の番号など）と小さい整数型、.astype(np.intp) したものは通る。値は numpy と同じ
    assert np32.bincount(np.arange(3)).tolist() == [1, 1, 1]
    assert np32.bincount(np.array([0, 2], np.uint8)).tolist() == [1, 0, 1]
    assert np32.bincount(i64.astype(np32.intp)).tolist() == [1, 1, 1, 1]
    assert np32.repeat(np.arange(2), [1, 2]).tolist() == [0, 1, 1]
    assert np32.minimum.reduceat(np.arange(4.0), np.array([0, 2])).tolist() == [0.0, 2.0]
    assert np32.minimum(1, 2) == 1


@needs_emulation
def test_emulation_reaches_retrocore_and_restores():
    from retrocore.sprite import cleanup
    with emulate_int32():
        assert cleanup.np is INT32_NUMPY
    assert cleanup.np is np


# ---- 1. convert_sprite() を 32 ビットのつもりで通す ----

def _pixel_input(seed=0):
    return enlarge_like_ai(make_pixel_sprite(), seed=seed)[0]


PALETTE = np.array([[30, 20, 40], [240, 230, 220], [200, 80, 60], [90, 160, 80], [250, 200, 60]])

CASES = [
    ("pixel", {}),
    ("pixel", {"colors": 8, "cleanup": "strong", "outline": "add"}),
    ("pixel", {"cleanup": "light", "outline": "add", "outline_color": "single", "outline_shape": "thick"}),
    ("pixel", {"cleanup": "off", "outline": "off"}),
    ("pixel", {"outline": "thin"}),
    ("pixel", {"outline": "thin", "outline_shape": "thick", "palette": PALETTE}),
    ("pixel", {"outline": "add", "outline_color": (60, 20, 40), "outline_pad": False}),
    ("pixel", {"palette": PALETTE}),
    ("pixel", {"palette": PALETTE, "outline": "add", "simplify": False}),
    ("pixel", {"dot_shape": "separate", "bg_remove_enclosed": True, "merge_distance": 0.0}),
    ("pixel", {"dot_size": (20.0, 20.0), "grid_origin": (0.0, 0.0), "grid_snap": False}),
    ("illust", {"input_type": "illust", "width": 48}),
    ("illust", {"input_type": "illust", "width": 48, "outline": "thin"}),
    ("illust", {"input_type": "illust", "width": 40, "resize_method": "median", "cleanup": "strong", "outline": "add"}),
    ("illust", {"input_type": "illust", "scale": 0.125, "resize_method": "nearest", "palette": PALETTE}),
    ("illust_noise", {"input_type": "illust", "height": 64, "colors": 256, "merge_distance": 0.0}),
    ("illust_alpha", {"input_type": "illust", "width": 32, "background": False}),
]


@pytest.fixture(scope="module")
def inputs():
    return {"pixel": _pixel_input(), "illust": make_character(), "illust_noise": add_noise(make_character(), 24),
            "illust_alpha": make_character(transparent=True)}


@needs_emulation
@pytest.mark.parametrize("name, settings", CASES, ids=[f"{n}-{i}" for i, (n, _) in enumerate(CASES)])
def test_convert_sprite_has_no_int64_index_arrays(inputs, name, settings):
    """32 ビットのつもりで変換してもエラーにならず、結果は普通に変換したものと同じ。"""
    expected = convert_sprite(inputs[name], SpriteSettings(**settings)).image
    with emulate_int32():
        got = convert_sprite(inputs[name], SpriteSettings(**settings)).image
    assert np.array_equal(got, expected)


@needs_emulation
def test_convert_for_web_has_no_int64_index_arrays(inputs):
    """画面と同じ呼び方（PNG のバイト列と設定の JSON）でも、32 ビットのつもりで通る。"""
    data = encode_png(_pixel_input(seed=5))
    expected = json.loads(convert_for_web(data, json.dumps({"colors": 6})))
    with emulate_int32():
        got = json.loads(convert_for_web(data, json.dumps({"colors": 6})))
    assert got["ok"], got
    assert got == expected


@needs_emulation
@pytest.mark.parametrize("frames, depth, split", [(2, 1, None), (4, 2, None), (2, 1, "auto"), (2, 1, 12), (4, 2, 8),
                                                  (2, 1, 1)])
def test_motion_has_no_int64_index_arrays(inputs, tmp_path, frames, depth, split):
    """動き（呼吸。分ける高さの指定を含む）も、変換の結果を材料にして 32 ビットのつもりで通る。結果は普通に作ったものと同じ。

    split=1 は足元なので付けない（警告）場合の確かめ。
    """
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    options = {} if split is None else {"split": split}
    expected = make_motion(sprite, "breathe", frames=frames, depth=depth, **options)
    with emulate_int32():
        got = make_motion(sprite, "breathe", frames=frames, depth=depth, **options)
        if split == 1:
            assert got.skipped and got.warnings == expected.warnings
            return
        save_sheet(got, tmp_path / "s_breathe_sheet.png")
    assert not got.skipped, got.warnings
    assert len(got.frames) == len(expected.frames)
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("frames, amount, split", [(4, 1, None), (8, 2, None), (3, 1, "auto"), (4, 1, 10), (6, 1, 14),
                                                   (4, 1, 1)])
def test_sway_has_no_int64_index_arrays(inputs, tmp_path, frames, amount, split):
    """動き（横揺れ。分ける高さの指定を含む）も、32 ビットのつもりで通る。結果は普通に作ったものと同じ。split=1 は足元（警告）。"""
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    options = {} if split is None else {"split": split}
    expected = make_motion(sprite, "sway", frames=frames, amount=amount, **options)
    with emulate_int32():
        got = make_motion(sprite, "sway", frames=frames, amount=amount, **options)
        if split == 1:
            assert got.skipped and got.warnings == expected.warnings
            return
        save_sheet(got, tmp_path / "s_sway_sheet.png")
    assert not got.skipped, got.warnings
    assert got.info == expected.info
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("frames, amount", [(4, 1), (8, 2), (3, 1)])
def test_float_has_no_int64_index_arrays(inputs, tmp_path, frames, amount):
    """動き（浮遊。キャラ全体を上下に動かす）も、32 ビットのつもりで通る。結果は普通に作ったものと同じ。"""
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    expected = make_motion(sprite, "float", frames=frames, amount=amount)
    with emulate_int32():
        got = make_motion(sprite, "float", frames=frames, amount=amount)
        save_sheet(got, tmp_path / "s_float_sheet.png")
    assert not got.skipped, got.warnings
    assert got.info == expected.info
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("height", [1, 2, 3])
def test_bounce_has_no_int64_index_arrays(inputs, tmp_path, height):
    """動き（弾む。キャラ全体を上にだけ動かす）も、32 ビットのつもりで通る。結果は普通に作ったものと同じ。"""
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    expected = make_motion(sprite, "bounce", height=height)
    with emulate_int32():
        got = make_motion(sprite, "bounce", height=height)
        save_sheet(got, tmp_path / "s_bounce_sheet.png")
    assert not got.skipped, got.warnings
    assert got.info == expected.info
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("depth, split", [(1, None), (2, None), (1, 10), (2, 12)])
def test_nod_has_no_int64_index_arrays(inputs, tmp_path, depth, split):
    """動き（うなずく。首を自動で探す・数で指定する）も、32 ビットのつもりで通る。結果は普通に作ったものと同じ。"""
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    options = {} if split is None else {"split": split}
    expected = make_motion(sprite, "nod", depth=depth, **options)
    with emulate_int32():
        got = make_motion(sprite, "nod", depth=depth, **options)
        if not got.skipped:
            save_sheet(got, tmp_path / "s_nod_sheet.png")
    assert got.skipped == expected.skipped and got.warnings == expected.warnings
    assert got.info == expected.info and got.spec == expected.spec
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("motion, options", [("stretch", {}), ("stretch", {"amount": 2, "split": 12}),
                                             ("stretch", {"keep_small_parts": True}), ("stretch", {"split": 3}),
                                             ("breathe", {"keep_small_parts": True, "frames": 4, "depth": 2}),
                                             ("nod", {"keep_small_parts": True, "depth": 2, "split": 10})])
def test_stretch_and_keep_have_no_int64_index_arrays(inputs, tmp_path, motion, options):
    """伸び（伸ばす）と小さな部品を守るも、32 ビットのつもりで通る。結果は普通に作ったものと同じ。"""
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    expected = make_motion(sprite, motion, **options)
    with emulate_int32():
        got = make_motion(sprite, motion, **options)
        if not got.skipped:
            save_sheet(got, tmp_path / f"s_{motion}_sheet.png")
    assert got.skipped == expected.skipped and got.warnings == expected.warnings
    assert got.info == expected.info and got.spec == expected.spec
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
def test_sample_bounce_stretch_has_no_int64_index_arrays(inputs):
    """見本の表（弾む＋伸び縮み）も、32 ビットのつもりで通る。"""
    from retrocore.sprite.motion import load_spec, make_motion_from_spec
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    spec = load_spec(ROOT / "apps" / "sprite" / "examples" / "bounce_stretch.motion.json")
    expected = make_motion_from_spec(sprite, spec)
    with emulate_int32():
        got = make_motion_from_spec(sprite, spec)
    assert not got.skipped and got.info == expected.info
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
def test_nod_hint_has_no_int64_index_arrays():
    """首が見つからないときに「試せる高さ」を探す道（スライム）も、32 ビットのつもりで通る。"""
    from test_sprite_motion import make_slime
    expected = make_motion(make_slime(), "nod")
    with emulate_int32():
        got = make_motion(make_slime(), "nod")
    assert got.skipped and got.warnings == expected.warnings and "試せる高さ" in got.warnings[0]


@needs_emulation
def test_sway_split_suggestion_has_no_int64_index_arrays():
    """横揺れで指定した高さの近くで揺らせず、近くの高さを探して知らせる道も、32 ビットのつもりで通る。"""
    from test_sprite_sway import sash
    from retrocore.sprite.motion.base import split_limits
    img = sash()
    lim = split_limits(img)
    split = next(s for s in range(lim[0], lim[1] + 1) if make_motion(img, "sway", split=s).skipped)
    expected = make_motion(img, "sway", split=split)
    with emulate_int32():
        got = make_motion(img, "sway", split=split)
    assert got.skipped and "近くで分けられる高さ" in got.warnings[0]
    assert got.warnings == expected.warnings


@needs_emulation
def test_breathe_split_suggestion_has_no_int64_index_arrays():
    """指定した高さで分けられず、近くの高さを探して知らせる道も、32 ビットのつもりで通る。"""
    from test_sprite_breathe_split import striped
    img = striped()
    expected = make_motion(img, "breathe", split=img.shape[0] - 14)
    with emulate_int32():
        got = make_motion(img, "breathe", split=img.shape[0] - 14)
    assert got.skipped and "近くで分けられる高さ" in got.warnings[0]
    assert got.warnings == expected.warnings


SPECS = Path(__file__).resolve().parent / "motion_specs"
SWAY_ROOT = {"format": "sprite-motion", "version": 1, "name": "sway_root", "loop": True,
             "regions": {"upper": {"type": "split", "at": "50.0%"}},
             "tracks": {"upper_x": {"type": "move", "region": "upper", "axis": "x"}, "root_y": {"type": "root", "axis": "y"}},
             "frames": [{"ms": 100}, {"ms": 100, "upper_x": 2, "root_y": -1}, {"ms": 100, "upper_x": -1, "root_y": 1}]}


@needs_emulation
@pytest.mark.parametrize("table", ["bounce.motion.json", "bounce_squash.motion.json", "broken.motion.json", SWAY_ROOT],
                         ids=["bounce", "bounce_squash", "broken", "sway_root"])
def test_motion_spec_has_no_int64_index_arrays(inputs, tmp_path, table):
    """動きの表（--motion-spec）から作る道も、32 ビットのつもりで通る。結果（作れない行の警告も）は普通に作ったものと同じ。"""
    from retrocore.sprite.motion import load_spec, make_motion_from_spec
    spec = load_spec(SPECS / table) if isinstance(table, str) else table
    sprite = convert_sprite(inputs["pixel"], SpriteSettings()).image
    expected = make_motion_from_spec(sprite, spec)
    with emulate_int32():
        got = make_motion_from_spec(sprite, spec)
        if not got.skipped:
            save_sheet(got, tmp_path / "s_sheet.png")
    assert got.skipped == (table == "broken.motion.json"), got.warnings
    assert got.info == expected.info and got.warnings == expected.warnings
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("options", [{}, {"split": 12, "frames": 4, "depth": 2}, {"split": 1}, {"motion": "sway"},
                                     {"motion": "sway", "frames": 8, "amount": 2, "split": 10},
                                     {"motion": "sway", "split": 1}, {"motion": "float"},
                                     {"motion": "float", "frames": 8, "amount": 2}, {"motion": "bounce"},
                                     {"motion": "bounce", "height": 3, "durations": 120}, {"motion": "nod"},
                                     {"motion": "nod", "depth": 2, "split": 12, "durations": 300},
                                     {"motion": "stretch"}, {"motion": "stretch", "amount": 2, "keep_small_parts": True},
                                     {"motion": "breathe", "keep_small_parts": True}])
def test_motion_for_web_has_no_int64_index_arrays(options):
    """画面の動き（変換の結果の PNG と設定の JSON から作る）も、32 ビットのつもりで通る。結果は普通に作ったものと同じ。"""
    data = encode_png(convert_sprite(_pixel_input(seed=5), SpriteSettings()).image)
    expected = json.loads(motion_for_web(data, json.dumps(options)))
    with emulate_int32():
        got = json.loads(motion_for_web(data, json.dumps(options)))
    assert got["ok"], got
    assert got == expected


# ---- 2. 呼んでいる場所の一覧（見落とし対策） ----

# 32 ビットで intp（int32）への「安全な」変換をする関数と、確かめる引数の位置・名前
CHECKED = {"bincount": (0, "x"), "repeat": (1, "repeats"), "put": (1, "ind"), "choose": (0, "a"), "reduceat": (1, "indices")}

# (ファイル, 関数, 呼ぶ関数, 渡している式)。どれも 32 ビットで intp（または小さい整数）になることを確かめたもの
REVIEWED = [
    ("background.py", "estimate_background", "bincount", "key.astype(np.intp)"),
    ("background.py", "_propagate", "repeat", "np.diff(np.r_[r_start, len(rid)])"),   # flatnonzero（intp）の差
    ("background.py", "_propagate", "repeat", "np.diff(np.r_[c_start, len(cs)])"),
    ("background.py", "_propagate", "reduceat", "r_start"),                            # flatnonzero
    ("background.py", "_propagate", "reduceat", "c_start"),
    ("cleanup.py", "_outline_color", "bincount", "lab[edge].astype(np.intp)"),
    ("cleanup.py", "_between_runs", "bincount", "lab[lab >= 0].astype(np.intp)"),
    ("downscale.py", "_pick_mode", "bincount", "bi"),                                  # np.nonzero
    ("edit.py", "_unscale", "repeat", "s"),                                            # Python の int（range の倍率）
    ("edit.py", "_unscale", "repeat", "s"),
    ("downscale.py", "_pick_mode", "bincount", "inv"),                                 # np.unique の番号
    ("downscale.py", "_pick_mode", "bincount", "bi"),
    ("image.py", "scale_nearest", "repeat", "factor"),                                 # Python の int
    ("motion/gif.py", "make_gif", "repeat", "scale"),                                  # Python の int（check_scale）
    ("motion/gif.py", "make_gif", "repeat", "scale"),
    ("image.py", "scale_nearest", "repeat", "factor"),
    ("outline.py", "_detect", "bincount", "lab[dark].astype(np.intp)"),
    ("palette.py", "match_palette", "bincount", "parts.labels"),                       # np.arange・np.full の番号
    ("simplify.py", "_unique_colors", "bincount", "inv"),                              # np.unique の番号
    ("simplify.py", "_units", "bincount", "grp"),                                      # np.arange / np.unique の番号
    ("simplify.py", "_units", "bincount", "grp"),
    ("simplify.py", "_units", "bincount", "grp"),
    ("simplify.py", "_units", "bincount", "grp"),
    ("simplify.py", "_merge_close_groups", "bincount", "label"),                       # np.arange・flatnonzero
    ("simplify.py", "simplify_colors", "bincount", "pix"),                             # np.argmin
]


def _call_sites():
    sites = []
    for path in sorted(SPRITE.rglob("*.py")):   # motion/ などの下のフォルダも含める
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in CHECKED):
                    continue
                pos, kw = CHECKED[node.func.attr]
                if node.func.attr == "repeat" and not (isinstance(node.func.value, ast.Name)
                                                       and node.func.value.id == "np"):
                    pos = 0  # 配列のメソッド a.repeat(repeats)
                arg = node.args[pos] if len(node.args) > pos else next(
                    (k.value for k in node.keywords if k.arg == kw), None)
                sites.append((path.relative_to(SPRITE).as_posix(), fn.name, node.func.attr, ast.unparse(arg) if arg is not None else ""))
    return sites


def test_index_casting_call_sites_are_reviewed():
    """np.bincount・np.repeat・np.put・np.choose・reduceat を呼ぶ場所が、確かめた一覧と同じ。

    失敗したら: 新しい場所で渡す配列が 32 ビット（Pyodide）でも intp になるかを確かめ（int64 なら .astype(np.intp)）、
    REVIEWED に足す。関数を書き換えて式が変わったときも同じ。
    """
    found, reviewed = Counter(_call_sites()), Counter(REVIEWED)
    assert found == reviewed, (f"確かめていない呼び出し: {sorted((found - reviewed).elements())}\n"
                               f"もう無い呼び出し: {sorted((reviewed - found).elements())}")


_REGION_TABLES = [
    # 頭の四角のうなずく・伸ばす 2・横の位置（はみ出す分の余白）・縦の位置
    ({"head": {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22}},
     {"y": {"type": "deform", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "y": -1}, {"ms": 1, "y": 2}]),
    ({"head": {"type": "rect", "left": 1, "right": 24, "bottom": "60%", "top": "100%"}},
     {"x": {"type": "move", "region": "head", "axis": "x"}}, [{"ms": 1}, {"ms": 1, "x": 1}, {"ms": 1, "x": -2}]),
    ({"head": {"type": "band", "bottom": 15, "top": 22}},
     {"v": {"type": "move", "region": "head", "axis": "y"}, "r": {"type": "root", "axis": "y"}},
     [{"ms": 1}, {"ms": 1, "v": -1, "r": -1}, {"ms": 1, "v": 1}]),
    # 作れない（空いた所の色が分からない）→ 近くで作れる範囲を探す道
    ({"head": {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22}},
     {"v": {"type": "move", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "v": -1}]),
]


@needs_emulation
@pytest.mark.parametrize("regions, tracks, frames", _REGION_TABLES)
def test_region_tables_have_no_int64_index_arrays(regions, tracks, frames):
    """帯・四角の範囲の表（作れたときと、作れずに近くの範囲を探すとき）と見本の表も、32 ビットのつもりで通る。"""
    from motion_records import make_knight
    from retrocore.sprite.motion import make_motion_from_spec
    spec = {"format": "sprite-motion", "version": 1, "name": "t", "loop": True, "regions": regions, "tracks": tracks,
            "frames": frames}
    sprite = make_knight()
    expected = make_motion_from_spec(sprite, spec)
    with emulate_int32():
        got = make_motion_from_spec(sprite, spec)
    assert got.skipped == expected.skipped and got.warnings == expected.warnings and got.info == expected.info
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))


@needs_emulation
@pytest.mark.parametrize("name", ["head_rect_nod", "bounce_head_band"])
def test_region_examples_have_no_int64_index_arrays(name):
    from retrocore.sprite.motion import load_spec, make_motion_from_spec
    sprite = make_pixel_sprite()
    spec = load_spec(ROOT / "apps" / "sprite" / "examples" / f"{name}.motion.json")
    expected = make_motion_from_spec(sprite, spec)
    with emulate_int32():
        got = make_motion_from_spec(sprite, spec)
        table = json.loads((ROOT / "apps" / "sprite" / "examples" / f"{name}.motion.json").read_text(encoding="utf-8"))
        web = json.loads(motion_for_web(encode_png(sprite), json.dumps({"spec": table})))
    assert not got.skipped and got.info == expected.info and web["ok"] and not web["skipped"]
    assert all(np.array_equal(a, b) for a, b in zip(got.frames, expected.frames))
