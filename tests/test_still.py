"""Still App（1 枚絵）の変換（retrocore/still/）のテスト。テスト画像はコードで合成する。"""

import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from int32_numpy import AVAILABLE, emulate_int32
from retrocore.still import colors as colors_mod
from retrocore.still import dots as dots_mod
from retrocore.still.convert import (MODES, StillError, StillSettings, convert_still, encode_indexed_png,
                                     info_json_text, parse_size, result_info, still_for_web)
from synth_still import LINE, POST, enlarge_scene, make_scene

ROOT = Path(__file__).resolve().parents[1]
SEED = 1   # この種では、グリッド検出が元のドットの数（48×30）をそのまま見つける


@pytest.fixture(scope="module")
def scene():
    art = make_scene()
    return art, enlarge_scene(art, seed=SEED)


def _png_bytes(rgba) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


# ---- グリッド検出 ----

def test_grid_detection_recovers_dots(scene):
    art, big = scene
    r = convert_still(big, StillSettings())
    assert r.dots.method == "grid"
    assert r.size == (48, 30)
    assert 9 <= r.dots.dot_size <= 11
    # どのドットも元の色の近く（4096 色に丸めた分くらいの差）
    diff = np.abs(r.rgb().astype(int) - art[..., :3].astype(int)).max(-1)
    assert diff.max() <= 16
    # 小さな赤いポスト（8 ドット）も消えない
    assert (np.abs(r.rgb().astype(int) - POST).max(-1) <= 16).sum() == 8


def test_grid_not_found_stops_with_width_candidates():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:200, 0:320]
    smooth = np.stack([xx * 255 // 320, yy * 255 // 200, (xx + yy) * 255 // 520], -1).astype(np.uint8)
    noisy = np.clip(smooth + rng.integers(-3, 4, smooth.shape), 0, 255).astype(np.uint8)
    with pytest.raises(StillError) as e:
        convert_still(noisy, StillSettings())
    assert e.value.kind == "grid"
    assert "--width" in str(e.value)
    cands = e.value.candidates
    assert cands and all(c["width"] <= 640 and c["height"] <= 400 for c in cands)
    assert any(c["width"] == 320 // 8 for c in cands)   # 1 ドットを 8px とした場合
    # 候補はそのまま --width に使える
    r = convert_still(noisy, StillSettings(width=cands[0]["width"]))
    assert r.size == (cands[0]["width"], cands[0]["height"])


def test_width_candidates_drop_multiple_of_other_axis():
    """縦横の推定の片方がもう片方のほぼ整数倍なら、大きい方（何ドット分かを 1 ドットと誤ったもの）は候補にしない。"""
    art = make_scene()
    big = enlarge_scene(art, seed=SEED)
    cands = dots_mod.width_candidates(big)
    est = [c for c in cands if "推定" in c["note"]]
    assert 1 <= len(est) <= 2
    assert all(abs(c["width"] - 48) <= 2 for c in est)


def test_edge_slivers_are_trimmed():
    """画像の端の、半ドットより細いマスは落とす（絵の外のはみ出し）。"""
    art = make_scene()
    big = enlarge_scene(art, seed=SEED)
    extra = np.concatenate([big, big[:, -3:]], axis=1)   # 右に 3px のはみ出し
    r = convert_still(extra, StillSettings())
    assert r.size == (48, 30)
    assert r.dots.trimmed[2] == 1


# ---- --width ----

def test_width_divides_evenly_and_keeps_aspect(scene):
    _, big = scene
    h, w = big.shape[:2]
    r = convert_still(big, StillSettings(width=40))
    assert r.dots.method == "width"
    assert r.size == (40, int(np.floor(40 * h / w + 0.5)))
    info = result_info(r)
    assert info["dots"]["width"] == 40 and info["dots"]["cell_size"][0] == pytest.approx(w / 40, abs=1e-3)


def test_width_larger_than_image_is_an_error(scene):
    _, big = scene
    with pytest.raises(StillError, match="画像の幅"):
        convert_still(big, StillSettings(width=big.shape[1] + 1))


def test_cell_color_is_the_most_common_not_the_mean():
    """マスの色は平均ではなくいちばん多い色（赤 6 割・青 4 割のマスは赤。紫にはならない）。"""
    a = np.zeros((10, 20, 3), np.uint8)
    a[:, :10] = (200, 30, 30)
    a[:6, 10:] = (200, 30, 30)
    a[6:, 10:] = (30, 30, 200)
    a[0, 0] = (201, 31, 29)   # 少しの揺れは数え方（16 段階の組）でまとまる
    out = dots_mod.mode_colors(a, np.array([0, 10, 20]), np.array([0, 10]))
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [200, 30, 30]
    assert out[0, 1].tolist() == [200, 30, 30]


# ---- 大きさの上限 ----

def test_over_max_size_stops_with_candidates(scene):
    _, big = scene
    with pytest.raises(StillError) as e:
        convert_still(big, StillSettings(max_size=(40, 40)))
    assert e.value.kind == "size"
    assert "上限 40×40" in str(e.value) and "出力しませんでした" in str(e.value)
    cands = e.value.candidates
    assert cands[0]["width"] == 40 and all(c["height"] <= 40 for c in cands)
    r = convert_still(big, StillSettings(max_size=(40, 40), width=cands[0]["width"]))
    assert r.size[0] <= 40 and r.size[1] <= 40


def test_smaller_than_max_size_is_output_as_is(scene):
    _, big = scene
    r = convert_still(big, StillSettings())
    assert r.size == (48, 30) and r.canvas_offset is None


def test_fit_width():
    assert dots_mod.fit_width(1600, 1000, 640, 400) == 640
    assert dots_mod.fit_width(1000, 1000, 640, 400) == 400
    assert dots_mod.height_for_width(640, 1600, 1000) == 400


# ---- 減色: 4 ビットの丸め ----

def test_4bit_palette_is_in_4096_colors(scene):
    _, big = scene
    r = convert_still(big, StillSettings())
    assert np.all(r.palette % 17 == 0)
    info = result_info(r)
    for p in info["palette"]:
        assert p["rgb4"] == [v // 17 for v in p["rgb"]]
        assert all(0 <= v <= 15 for v in p["rgb4"])
        assert p["hex"] == "#{:02X}{:02X}{:02X}".format(*p["rgb"])


def test_no_4bit_keeps_finer_colors(scene):
    _, big = scene
    r = convert_still(big, StillSettings(four_bit=False))
    assert np.any(r.palette % 17 != 0)
    assert any(p["rgb4"] is None for p in result_info(r)["palette"])


def test_4bit_merge_reduces_colors_and_is_recorded():
    """選んだ 2 色が 4096 色に丸めると同じになるときは、少ない色数のままにして記録する。"""
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = (129, 129, 129)
    img[:, 10:] = (143, 143, 143)
    red = colors_mod.reduce_colors(img, 4, keep_darkest=False)
    assert red.chosen == 2 and red.merged == 1
    assert red.palette.tolist() == [[136, 136, 136]]
    r = convert_still(np.repeat(np.repeat(img, 1, 0), 1, 1), StillSettings(width=20, keep_darkest=False))
    info = result_info(r)
    assert info["colors"]["merged_by_4bit"] == 1 and info["colors"]["used"] == 1
    assert any("重なり" in w for w in info["warnings"])


def test_snap_4bit_follows_perceived_brightness_and_keeps_gray():
    snap = colors_mod.snap_4bit(np.array([[8, 8, 8], [5, 5, 5], [0, 0, 0], [255, 255, 255], [136, 68, 68]]))
    # (8,8,8) は四捨五入なら 0 だが、見た目では #111111 に近い。灰色は灰色のまま
    assert snap.tolist() == [[17, 17, 17], [17, 17, 17], [0, 0, 0], [255, 255, 255], [136, 68, 68]]
    assert colors_mod.round_4bit(np.array([[8, 8, 8]])).tolist() == [[0, 0, 0]]


def test_flat_area_on_rounding_boundary_stays_one_color():
    """平らな面の色が丸めの境目にあっても（G が 127〜128）、まだらにならない（色を選んでから丸めるため）。"""
    rng = np.random.default_rng(0)
    img = np.zeros((30, 30, 3), np.uint8)
    img[:] = (138, 127, 89)
    img[..., 1] += rng.integers(0, 2, (30, 30)).astype(np.uint8)
    img[:3] = (20, 20, 20)
    red = colors_mod.reduce_colors(img, 16)
    assert len(np.unique(red.index[3:])) == 1


def test_no_new_colors_every_dot_is_a_palette_color(scene):
    _, big = scene
    r = convert_still(big, StillSettings(colors=6))
    assert len(r.palette) <= 6
    assert set(np.unique(r.index).tolist()) == set(range(len(r.palette)))
    assert int(r.counts.sum()) == r.index.size


# ---- いちばん暗い色 ----

def _line_image():
    img = np.zeros((50, 60, 3), np.uint8)
    img[:, :15] = (220, 220, 210)
    img[:, 15:30] = (70, 60, 60)
    img[:, 30:45] = (90, 140, 200)
    img[:, 45:] = (200, 120, 60)
    img[[10, 30], :] = (15, 15, 20)    # 暗い線（全体の 4%）
    return img


def test_keep_darkest_keeps_dark_lines():
    img = _line_image()
    kept = colors_mod.reduce_colors(img, 4, keep_darkest=True)
    lost = colors_mod.reduce_colors(img, 4, keep_darkest=False)
    assert kept.palette[kept.index[10, 0]].tolist() == [17, 17, 17]   # 線はいちばん暗い色のまま
    assert kept.darkest.tolist() == [17, 17, 17]
    assert lost.palette[lost.index[10, 0]].tolist() != [17, 17, 17]   # 取っておかないと中間の色に吸われる
    assert lost.darkest is None


def test_darkest_ignores_a_few_noise_dots():
    """ほんの数ドットの黒（全体の 1% 未満）は「いちばん暗い色」にしない。"""
    img = _line_image()
    img[0, 0] = (0, 0, 0)
    dark, _ = colors_mod.darkest_color(img)
    assert dark.tolist() == [15, 15, 20]


def test_darkest_json_and_palette_order(scene):
    _, big = scene
    r = convert_still(big, StillSettings())
    info = result_info(r)
    assert info["colors"]["darkest_kept"] == info["palette"][0]["hex"]
    lum = [colors_mod.rgb_to_oklab(np.array([p["rgb"]], float))[0, 0] for p in info["palette"]]
    assert lum == sorted(lum)   # 暗い順
    near = np.abs(np.array([p["rgb"] for p in info["palette"]]) - LINE).max(-1)
    assert near[0] <= 17
    assert result_info(convert_still(big, StillSettings(keep_darkest=False)))["colors"]["darkest_kept"] is None


# ---- キャンバス ----

def test_canvas_centers_with_darkest_and_no_new_colors(scene):
    _, big = scene
    plain = convert_still(big, StillSettings())
    r = convert_still(big, StillSettings(canvas=(53, 35)))
    assert r.size == (53, 35)
    assert r.canvas_offset == (2, 2)        # 余り 5・5 は左上 2、右下 3
    assert np.array_equal(r.palette, plain.palette)
    assert np.array_equal(r.index[2:32, 2:50], plain.index)
    border = np.ones((35, 53), bool)
    border[2:32, 2:50] = False
    assert np.all(r.index[border] == 0)    # 0 番 = いちばん暗い色
    assert int(r.counts.sum()) == 53 * 35
    info = result_info(r)
    assert info["canvas"] == {"width": 53, "height": 35, "offset": [2, 2], "fill": {"index": 0, "hex": info["palette"][0]["hex"]}}
    assert info["picture"] == {"width": 48, "height": 30}


@pytest.mark.parametrize("canvas, words", [((40, 30), "小さく"), ((700, 400), "上限")])
def test_canvas_errors(scene, canvas, words):
    _, big = scene
    with pytest.raises(StillError, match=words):
        convert_still(big, StillSettings(canvas=canvas))


# ---- 透明・読み込み ----

def test_transparent_is_filled_with_black():
    a = np.zeros((4, 4, 4), np.uint8)
    a[..., :3] = 200
    a[:2, :, 3] = 255
    a[2, :, 3] = 128
    rgb = dots_mod.flatten_black(a)
    assert rgb[0, 0].tolist() == [200, 200, 200]
    assert rgb[3, 0].tolist() == [0, 0, 0]
    assert rgb[2, 0].tolist() == [100, 100, 100]


# ---- インデックスカラーの PNG ----

@pytest.mark.parametrize("k, depth", [(1, 1), (2, 1), (3, 2), (4, 2), (5, 4), (16, 4), (17, 8), (256, 8)])
def test_indexed_png_round_trip(k, depth):
    rng = np.random.default_rng(k)
    w = 13 if k < 256 else 37
    idx = rng.integers(0, k, (7, w)).astype(np.uint8)
    idx[0, 0] = k - 1
    pal = rng.integers(0, 256, (k, 3)).astype(np.uint8)
    data = encode_indexed_png(idx, pal)
    assert data[24] == depth and data[25] == 3     # IHDR のビット深度・色の種類（3 = パレット）
    im = Image.open(io.BytesIO(data))
    im.load()
    assert im.mode == "P"
    assert len(im.getpalette()) // 3 == k          # パレットは渡した色だけ
    assert np.array_equal(np.asarray(im), idx)
    assert np.array_equal(np.array(im.getpalette(), np.uint8).reshape(-1, 3), pal)
    assert encode_indexed_png(idx, pal) == data    # 毎回同じバイト列


def test_output_png_has_only_used_colors(scene):
    _, big = scene
    r = convert_still(big, StillSettings())
    im = Image.open(io.BytesIO(encode_indexed_png(r.index, r.palette)))
    assert im.mode == "P" and len(im.getpalette()) // 3 == len(r.palette)
    assert len(r.palette) == len(np.unique(r.index))


# ---- 設定・JSON ----

def test_mode_preset_and_overrides():
    s = StillSettings.from_mode("pc98")
    assert (s.colors, s.four_bit, s.max_size, s.keep_darkest) == (16, True, (640, 400), True)
    assert StillSettings() == s
    s2 = StillSettings.from_mode("pc98", colors=8, four_bit=False, max_size=(320, 200))
    assert (s2.colors, s2.four_bit, s2.max_size) == (8, False, (320, 200))
    assert set(MODES) == {"pc98", "modern", "photo"}


@pytest.mark.parametrize("values, words", [
    ({"colors": 1}, "色数"), ({"colors": 257}, "色数"), ({"mode": "x"}, "--mode"),
    ({"max_size": "640"}, "幅x高さ"), ({"canvas": [0, 10]}, "1〜"), ({"width": 1.5}, "整数"),
    ({"four_bit": "yes"}, "true"), ({"colours": 3}, "知らない"),
])
def test_bad_settings(values, words):
    with pytest.raises(ValueError, match=words):
        StillSettings.from_dict(values)


def test_parse_size():
    assert parse_size("640x400") == (640, 400)
    assert parse_size("320×200") == (320, 200)
    assert parse_size([10, 20]) == (10, 20)


def test_json_has_size_palette_settings_and_grid(scene):
    _, big = scene
    r = convert_still(big, StillSettings())
    info = json.loads(info_json_text(result_info(r, "scene.png")))
    assert info["app"] == "still" and info["source"] == "scene.png"
    assert (info["width"], info["height"]) == (48, 30)
    assert info["input"] == {"width": big.shape[1], "height": big.shape[0]}
    assert info["dots"]["method"] == "grid" and 9 <= info["dots"]["dot_size"] <= 11
    assert info["settings"] == {"mode": "pc98", "colors": 16, "four_bit": True, "max_size": [640, 400],
                                "width": None, "canvas": None, "keep_darkest": True}
    assert [p["index"] for p in info["palette"]] == list(range(len(info["palette"])))
    assert info["pixel_aspect"] == "1:1"
    assert info["warnings"] == []


def test_same_input_same_bytes(scene):
    _, big = scene
    data = _png_bytes(big)
    a = json.loads(still_for_web(data, "a.png", "{}"))
    b = json.loads(still_for_web(data, "a.png", "{}"))
    assert a == b and a["ok"]


def test_still_for_web_error_has_candidates(scene):
    _, big = scene
    res = json.loads(still_for_web(_png_bytes(big), "a.png", json.dumps({"max_size": [30, 30]})))
    assert not res["ok"] and res["kind"] == "size" and res["candidates"]
    bad = json.loads(still_for_web(b"not an image", "a.png", "{}"))
    assert not bad["ok"] and "画像として読み込めません" in bad["error"] and "detail" in bad
    worse = json.loads(still_for_web(_png_bytes(big), "a.png", "[1]"))
    assert not worse["ok"]


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_convert_runs_with_int32_numpy(scene):
    _, big = scene
    expected = convert_still(big, StillSettings(canvas=(50, 32)))
    with emulate_int32():
        got = convert_still(big, StillSettings(canvas=(50, 32)))
        w = convert_still(big, StillSettings(width=30, four_bit=False))
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.palette, expected.palette)
    assert w.size[0] == 30


def test_still_does_not_call_index_casting_functions():
    """retrocore/still は np.bincount などを使わない（32 ビットで型のエラーになりやすいので、np.unique で数える）。"""
    names = {"bincount", "repeat", "put", "choose", "reduceat"}
    found = []
    for path in sorted((ROOT / "retrocore" / "still").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in names:
                found.append(f"{path.name}:{node.lineno} {node.attr}")
    assert not found, found
