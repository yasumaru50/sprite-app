"""Still App の「ざらついた範囲を模様に整える」（retrocore/still/pattern.py、--tidy-pattern）のテスト。

テスト画像はコードで合成する（tests/synth_still.py の make_noisy_scene）。
"""

import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from apps.still import cli
from int32_numpy import AVAILABLE, emulate_int32
from retrocore.still import pattern as pattern_mod
from retrocore.still.convert import StillSettings, convert_still, convert_still_bytes, still_for_web
from still_legacy import legacy_json
from synth_still import NOISY_PALETTE, enlarge_scene, make_noisy_scene, make_scene, noisy_scene_rgba, shuffled_level

PAL = NOISY_PALETTE


def _png(rgba) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def noisy():
    idx, parts = make_noisy_scene()
    return idx, parts, pattern_mod.tidy_pattern(idx, PAL, [0])


def _inner(sl, m=4):
    """範囲の端（濃さを測る窓が外にかかる所）を除いた内側。"""
    ys, xs = sl
    return slice(ys.start + m, ys.stop - m), slice(xs.start + m, xs.stop - m)


# ---- 模様の 5 段 ----

def test_five_levels_and_tile():
    assert pattern_mod.level_of([0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0]).tolist() == [0, 0, 1, 1, 2, 2, 3, 4, 4]
    for level in range(5):
        light = pattern_mod.light_dots(np.full((4, 4), level, np.int8))
        blocks = light.reshape(2, 2, 2, 2).sum(axis=(1, 3))
        assert (blocks == level).all()                     # 2×2 ごとに明るいドットがちょうど level 個
        assert np.array_equal(light[:2, :2], light[2:, 2:])  # 2×2 の決まった模様のくり返し
    half = pattern_mod.light_dots(np.full((4, 4), 2, np.int8))
    yy, xx = np.mgrid[0:4, 0:4]
    assert np.array_equal(half, (yy + xx) % 2 == 0)          # 50% は市松
    # 段は重なっていく（段が上がると明るいドットが足されるだけ）
    for level in range(4):
        a = pattern_mod.light_dots(np.full((2, 2), level, np.int8))
        b = pattern_mod.light_dots(np.full((2, 2), level + 1, np.int8))
        assert (b | ~a).all()


@pytest.mark.parametrize("part, level, colors", [("top_left", 1, (2, 3)), ("top_right", 2, (1, 2)),
                                                  ("mid_left", 3, (2, 3))])
def test_noisy_region_becomes_fixed_pattern(noisy, part, level, colors):
    idx, parts, t = noisy
    ys, xs = _inner(parts[part])
    out = t.index[ys, xs]
    a, b = colors
    expect = np.where(pattern_mod.light_dots(np.full(out.shape, level, np.int8), ys.start, xs.start), b, a)
    if part == "mid_left":   # 暗い線の行（とそのすぐ上下）は除いて比べる
        keep = np.abs(np.arange(ys.start, ys.stop) - 32) > 3
        out, expect = out[keep], expect[keep]
    assert np.array_equal(out, expect)
    assert not np.array_equal(idx[ys, xs], t.index[ys, xs])


# ---- 境目でずれない ----

def test_pattern_is_aligned_across_region_boundary(noisy):
    idx, parts, t = noisy
    # 上の左右は色の組が違う 2 つの範囲（左 25%、右 50%）。境目 x=32 のすぐそば（±2）は、どちらの範囲になるかが
    # ゆれるので除き、その外で両側とも絵全体の座標の模様（同じ表）に乗っていることを確かめる
    a, b = t.label[10, 26], t.label[10, 38]
    assert a >= 0 and b >= 0 and a != b
    yy, xx = np.mgrid[6:18, 0:64]
    tile = pattern_mod.BAYER2[yy % 2, xx % 2]
    left, right = slice(20, 30), slice(34, 44)
    assert np.array_equal(t.index[6:18, left] == 3, tile[:, left] < 1)
    assert np.array_equal(t.index[6:18, right] == 2, tile[:, right] < 2)


def test_pattern_follows_picture_coordinates_not_region_corner():
    """範囲の左上が奇数の位置でも、模様は絵全体の座標で決まる。"""
    for oy, ox in ((0, 0), (1, 0), (0, 1), (1, 1)):
        idx = np.zeros((40 + oy, 40 + ox), np.uint8)
        idx[oy:, ox:] = np.where(shuffled_level(40, 40, 2, seed=7), 2, 1)
        t = pattern_mod.tidy_pattern(idx, PAL, [0])
        yy, xx = np.mgrid[0:idx.shape[0], 0:idx.shape[1]]
        inner = (yy >= oy + 4) & (yy < idx.shape[0] - 4) & (xx >= ox + 4) & (xx < idx.shape[1] - 4)
        assert np.array_equal((t.index == 2)[inner], ((yy + xx) % 2 == 0)[inner])


# ---- 触らない所 ----

def test_outline_is_not_changed(noisy):
    idx, parts, t = noisy
    assert (t.index[parts["line"]] == 0).all()
    assert (t.index[0] == 0).all() and (t.index[:, 0] == 0).all()     # 外枠
    assert t.protected[parts["line"]].all()
    assert not (t.index[idx != 0] == 0).any()                          # 暗い色を新しく置かない


def test_flat_area_is_not_changed(noisy):
    idx, parts, t = noisy
    assert np.array_equal(t.index[parts["flat"]], idx[parts["flat"]])


def test_many_third_color_is_not_tidied(noisy):
    idx, parts, t = noisy
    ys, xs = parts["mid_right"]
    assert np.array_equal(t.index[ys, xs], idx[ys, xs])
    skipped = [r for r in t.regions if r.skipped == "third_color"]
    assert skipped and all(r.third >= pattern_mod.THIRD_MAX for r in skipped)


def test_isolated_third_color_dots_are_noise():
    """3 色目が 1 ドットずつ散らばっているだけなら（ノイズ）、整えて 2 色のどちらかにする。"""
    idx = np.where(shuffled_level(40, 40, 2, seed=3), 2, 1).astype(np.uint8)
    idx[5:35:4, 5:35:3] = 5                       # 1 ドットの 3 色目（約 7%）
    t = pattern_mod.tidy_pattern(idx, PAL, [0])
    assert len(t.tidied) == 1
    assert not (t.index[8:32, 8:32] == 5).any()


def test_small_region_is_not_tidied():
    idx = np.full((40, 40), 4, np.uint8)
    idx[10:16, 10:16] = np.where(shuffled_level(6, 6, 2, seed=1), 2, 1)   # 36 ドット（64 未満）
    t = pattern_mod.tidy_pattern(idx, PAL, [0])
    assert np.array_equal(t.index, idx) and not t.tidied


# ---- 新しい色を作らない ----

def test_no_new_colors(noisy):
    idx, parts, t = noisy
    assert set(np.unique(t.index).tolist()) <= set(np.unique(idx).tolist())
    for r in t.tidied:
        m = t.label == t.regions.index(r)
        assert set(np.unique(t.index[m]).tolist()) <= set(r.colors)


def test_convert_with_tidy_uses_only_palette_colors():
    rgba = noisy_scene_rgba()
    off = convert_still(rgba, StillSettings(width=64))
    on = convert_still(rgba, StillSettings(width=64, tidy_pattern=True))
    assert {tuple(c) for c in on.palette} <= {tuple(c) for c in off.palette}
    assert len(on.palette) == len(np.unique(on.index))       # 使った色だけ
    assert on.tidied.changed > 0
    assert int(on.counts.sum()) == on.index.size


# ---- 切のときは前と同じ ----

# この PR の前のコード（--tidy-pattern が無い版）で作った PNG・JSON の SHA-256 の先頭
BEFORE = [({}, "b6ceb690bd3e53f3", "17f26a957ee16a30"),
          ({"canvas": [52, 34], "colors": 8}, "c51c574ea29075c1", "bf4390389977c7c8"),
          ({"width": 30, "four_bit": False}, "20ba8e923d76ccbb", "bd88c323ffe1bd2b")]


@pytest.mark.parametrize("settings, png_hash, json_hash", BEFORE)
def test_off_output_is_unchanged(settings, png_hash, json_hash):
    data = _png(enlarge_scene(make_scene(), seed=1))
    for s in (settings, {**settings, "tidy_pattern": False}):
        out = convert_still_bytes(data, "scene.png", s)
        assert hashlib.sha256(out["png"]).hexdigest()[:16] == png_hash
        assert hashlib.sha256(legacy_json(out["json"]).encode("utf-8")).hexdigest()[:16] == json_hash
        assert '"pattern"' not in out["json"] and "tidy_pattern" not in legacy_json(out["json"])


def test_off_web_has_no_overlay():
    res = json.loads(still_for_web(_png(noisy_scene_rgba()), "n.png", json.dumps({"width": 64})))
    assert res["ok"] and "overlay_base64" not in res and "pattern" not in res


# ---- JSON・見える化 ----

def test_json_records_regions_changed_and_settings():
    out = convert_still_bytes(_png(noisy_scene_rgba()), "n.png", {"width": 64, "tidy_pattern": True})
    info = json.loads(out["json"])
    p = info["pattern"]
    t = out["result"].tidied
    assert p["regions"] == len(t.tidied) >= 3 and p["changed"] == t.changed > 0
    assert p["settings"]["min_region"] == pattern_mod.MIN_REGION and p["settings"]["levels"] == [0, 25, 50, 75, 100]
    assert p["skipped"]["third_color"] >= 1
    assert sum(r["changed"] for r in p["list"]) == p["changed"]
    assert info["settings"]["tidy_pattern"] is True


def test_overlay_is_display_only_and_follows_canvas():
    rgba = noisy_scene_rgba()
    s = StillSettings(width=64, tidy_pattern=True, canvas=(70, 52))
    res = convert_still(rgba, s)
    ov = res.overlay()
    assert ov.shape == (52, 70, 4)
    ox, oy = res.canvas_offset
    lab = res.tidied.label
    assert np.array_equal(ov[oy:oy + lab.shape[0], ox:ox + lab.shape[1], 3] > 0, lab >= 0)
    assert (ov[:oy, :, 3] == 0).all()
    web = json.loads(still_for_web(_png(rgba), "n.png", json.dumps({"width": 64, "tidy_pattern": True,
                                                                    "canvas": [70, 52]})))
    got = np.asarray(Image.open(io.BytesIO(base64.b64decode(web["overlay_base64"]))))
    assert np.array_equal(got, ov)
    # 重ね絵は保存する PNG に入らない
    plain = convert_still_bytes(_png(rgba), "n.png", s)
    assert base64.b64decode(web["png_base64"]) == plain["png"]


# ---- CLI とブラウザ版 ----

def test_cli_option(tmp_path):
    args = cli.build_parser().parse_args(["x.png", "--tidy-pattern"])
    assert cli.settings_from_args(args).tidy_pattern is True
    assert cli.settings_from_args(cli.build_parser().parse_args(["x.png"])).tidy_pattern is False


@pytest.mark.parametrize("extra, settings", [
    (["--width", "64", "--tidy-pattern"], {"width": 64, "tidy_pattern": True}),
    (["--width", "64", "--tidy-pattern", "--canvas", "70x52", "--colors", "5"],
     {"width": 64, "tidy_pattern": True, "canvas": [70, 52], "colors": 5}),
])
def test_web_result_equals_cli_files(tmp_path, extra, settings):
    src = tmp_path / "noisy.png"
    src.write_bytes(_png(noisy_scene_rgba()))
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", *extra]) == 0
    res = json.loads(still_for_web(src.read_bytes(), src.name, json.dumps(settings)))
    assert res["ok"] and res["pattern"]["regions"] > 0
    assert base64.b64decode(res["png_base64"]) == (tmp_path / "c.png").read_bytes()
    assert res["json_text"].encode("utf-8") == (tmp_path / "c.json").read_bytes()


def test_bad_setting():
    with pytest.raises(ValueError, match="tidy_pattern"):
        StillSettings.from_dict({"tidy_pattern": "yes"})


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_tidy_runs_with_int32_numpy(noisy):
    idx, _, expected = noisy
    with emulate_int32():
        got = pattern_mod.tidy_pattern(idx, PAL, [0])
        res = convert_still(noisy_scene_rgba(), StillSettings(width=64, tidy_pattern=True))
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.label, expected.label)
    assert res.tidied.changed > 0
