"""Still App の今風モード（--mode modern）・似た色をまとめる・孤立したドットをそろえる（--tidy-dots）のテスト。

テスト画像はコードで合成する（tests/synth_still.py の make_modern_scene）。98 モードの既定の出力が前とバイト単位で
同じことも、前の版の出力のハッシュと比べて確かめる。
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
from retrocore.still import colors as colors_mod
from retrocore.still import lone as lone_mod
from retrocore.still.convert import (MODERN_COLOR_CHOICES, MODES, StillError, StillSettings, convert_still,
                                     convert_still_bytes, spec, still_for_web)
from synth_still import (MODERN_SKY, MODERN_SPECK, MODERN_WALL, enlarge_scene, make_modern_scene, make_scene,
                         noisy_scene_rgba)


def _png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def modern():
    rgb, spots = make_modern_scene()
    return rgb, spots


def _modern(rgb, **kw):
    return convert_still(rgb, StillSettings.from_mode("modern", width=rgb.shape[1], **kw))


# ---- 98 モードの出力は前と同じ（この機能を足す前の版のハッシュ） ----

@pytest.mark.parametrize("name, settings, png_hash, json_hash", [
    ("scene", {}, "b6ceb690bd3e53f3", "17f26a957ee16a30"),
    ("scene", {"width": 30, "canvas": [52, 34], "colors": 8}, "9376a76c0ef13efc", "b2340d8e1732e335"),
    ("scene", {"four_bit": False, "keep_darkest": False}, "49286a232a4db978", "5264a0c09c43d226"),
    ("noisy", {"width": 64, "tidy_pattern": True}, "b997d539fa63e15b", "65755fe90c9d2b1a"),
    ("noisy", {"width": 64}, "f4111989eee4e15e", "c5cb7b30f3d2f64d"),
])
def test_pc98_output_is_unchanged(name, settings, png_hash, json_hash):
    data = _png(enlarge_scene(make_scene(), seed=1)) if name == "scene" else _png(noisy_scene_rgba())
    out = convert_still_bytes(data, f"{name}.png", settings)
    assert hashlib.sha256(out["png"]).hexdigest()[:16] == png_hash
    assert hashlib.sha256(out["json"].encode("utf-8")).hexdigest()[:16] == json_hash
    info = out["info"]
    assert "similar" not in info and "tidy_dots" not in info and "merged_similar" not in info["colors"]
    assert "merge_similar" not in info["settings"] and "tidy_dots" not in info["settings"]


# ---- 今風モードのプリセット ----

def test_modern_preset():
    s = StillSettings.from_mode("modern")
    assert (s.colors, s.four_bit, s.max_size, s.keep_darkest, s.merge_similar, s.tidy_pattern, s.tidy_dots) == (
        32, False, (1920, 1080), True, True, False, False)
    assert StillSettings().merge_similar is False and StillSettings().tidy_dots is False
    assert spec()["modern_colors"] == list(MODERN_COLOR_CHOICES) == [16, 24, 32, 48, 64, 96, 128]
    assert spec()["modes"]["modern"]["max_size"] == [1920, 1080]


def test_modern_json_records_mode_and_settings(modern):
    rgb, _ = modern
    info = convert_still_bytes(_png(rgb), "m.png", {"mode": "modern", "width": 64})["info"]
    assert info["settings"]["mode"] == "modern" and info["settings"]["merge_similar"] is True
    assert info["settings"]["four_bit"] is False and info["settings"]["colors"] == 32
    assert "tidy_dots" not in info["settings"] and "tidy_dots" not in info
    off = convert_still_bytes(_png(rgb), "m.png", {"mode": "modern", "width": 64, "merge_similar": False})["info"]
    assert off["settings"]["merge_similar"] is False and "similar" not in off


def test_merge_similar_is_only_for_modern():
    with pytest.raises(ValueError, match="今風モード"):
        StillSettings.from_dict({"merge_similar": True})
    with pytest.raises(ValueError, match="merge_similar"):
        StillSettings.from_dict({"mode": "modern", "merge_similar": "yes"})


def test_modern_over_max_size_stops_with_candidates():
    rgb = np.zeros((100, 2000, 3), np.uint8)
    rgb[:, ::2] = 200
    with pytest.raises(StillError) as e:
        convert_still(rgb, StillSettings.from_mode("modern", width=2000))
    assert "1920×1080" in str(e.value) and e.value.kind == "size"
    assert e.value.candidates and all(c["width"] <= 1920 for c in e.value.candidates)
    assert convert_still(rgb, StillSettings.from_mode("modern", width=2000, max_size=(2000, 200))).size == (2000, 100)


# ---- 似た色をまとめる ----

def test_merge_similar_unit_keeps_larger_and_darkest():
    pal = np.array([[10, 10, 12], [14, 12, 12], [200, 170, 130], [214, 176, 124], [90, 140, 210]], np.uint8)
    idx = np.array([[0, 1, 1, 1, 2, 2, 2, 3, 4]], np.uint8)
    counts = np.array([1, 3, 3, 1, 1])
    sim = colors_mod.merge_similar(idx, pal, counts, keep=[0])
    # 0 番（いちばん暗い色）はドット数が少なくても残す側。3 番は多い 2 番に合わせる
    assert sim.palette.tolist() == [[10, 10, 12], [200, 170, 130], [90, 140, 210]]
    assert sim.index.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1, 2]]
    assert sim.counts.tolist() == [4, 4, 1] and sim.counts.sum() == counts.sum()
    assert [(p["from"], p["to"]) for p in sim.pairs] == [((14, 12, 12), (10, 10, 12)), ((214, 176, 124), (200, 170, 130))]
    # keep どうしはまとめない
    both = colors_mod.merge_similar(idx, pal, counts, keep=[0, 1])
    assert len(both.palette) == 4 and both.pairs[0]["from"] == (214, 176, 124)


def test_merge_similar_far_colors_are_kept():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    idx = np.array([[0, 1]], np.uint8)
    sim = colors_mod.merge_similar(idx, pal, np.array([1, 1]))
    assert sim.pairs == [] and np.array_equal(sim.index, idx) and np.array_equal(sim.palette, pal)


def test_modern_merges_near_colors_without_new_colors(modern):
    rgb, _ = modern
    res = _modern(rgb)
    before = {tuple(c) for c in res.reduced.palette}
    after = {tuple(c) for c in res.palette}
    assert after < before                                     # 新しい色は無く、減っただけ
    assert len(res.similar.pairs) == 2 and len(res.palette) == len(res.reduced.palette) - 2
    assert {p["to"] for p in res.similar.pairs} == {MODERN_SKY, MODERN_WALL}
    assert all(p["distance"] < colors_mod.SIMILAR_DISTANCE for p in res.similar.pairs)
    assert tuple(res.palette[0]) == tuple(res.reduced.palette[0])   # いちばん暗い色は残る
    assert any("2 組をまとめ" in w for w in res.warnings)
    info = json.loads(convert_still_bytes(_png(rgb), "m.png", {"mode": "modern", "width": 64})["json"])
    assert info["colors"]["merged_similar"] == 2 and info["similar"]["merged"] == 2
    assert info["similar"]["distance"] == colors_mod.SIMILAR_DISTANCE
    assert sum(p["count"] for p in info["similar"]["list"]) == 18 + 112
    assert sum(p["count"] for p in info["palette"]) == 64 * 40


# ---- 孤立したドットをそろえる ----

def test_tidy_lone_dots_rules(modern):
    rgb, spots = modern
    res = _modern(rgb, tidy_dots=True)
    changed = {tuple(int(v) for v in p) for p in np.argwhere(res.lone.changed)}
    assert changed == {spots["sky"], spots["wall"]}            # 端・暗い色・2 つならんだ点は変えない
    assert res.lone.count == 2
    rgb_out = res.rgb()
    assert tuple(rgb_out[spots["sky"]]) == MODERN_SKY and tuple(rgb_out[spots["wall"]]) == MODERN_WALL
    assert tuple(rgb_out[spots["edge"]]) == MODERN_SPECK
    assert tuple(rgb_out[spots["pair"][0]]) == MODERN_SPECK
    assert tuple(rgb_out[spots["dark"]]) == tuple(res.palette[0])
    plain = _modern(rgb)
    diff = np.any(rgb_out != plain.rgb(), axis=-1)
    assert np.array_equal(diff, res.lone.changed)               # ほかのドットは変わらない


def test_tidy_lone_dots_needs_seven_same_neighbors():
    idx = np.full((5, 5), 1, np.uint8)
    idx[2, 2] = 2
    assert lone_mod.tidy_lone_dots(idx, [0]).index[2, 2] == 1
    six = idx.copy()
    six[1, 1] = 3
    six[1, 2] = 3
    assert lone_mod.tidy_lone_dots(six, [0]).count == 0        # 同じ色が 6 つでは変えない
    seven = idx.copy()
    seven[1, 1] = 3
    t = lone_mod.tidy_lone_dots(seven, [0])
    assert t.changed[2, 2] and t.index[2, 2] == 1              # 7 つなら変える（3 番の点も孤立しているので変わる）
    has_same = idx.copy()
    has_same[1, 1] = 2                                         # まわりに同じ色があれば変えない
    assert lone_mod.tidy_lone_dots(has_same, [0]).count == 0


def test_tidy_lone_dots_runs_once():
    # 市松（どのドットもまわりに同じ色がある）は変えない。1 回だけなので、変えた結果でまた変えることはない
    yy, xx = np.mgrid[0:6, 0:6]
    checker = ((yy + xx) % 2 + 1).astype(np.uint8)
    assert lone_mod.tidy_lone_dots(checker, [0]).count == 0
    idx = np.full((7, 7), 1, np.uint8)
    idx[3, 3] = 2
    idx[2, 2] = 3                                              # 3 番の点のまわりの 8 つのうち 7 つが 1 番
    t = lone_mod.tidy_lone_dots(idx, [0])
    assert t.changed[3, 3] and t.changed[2, 2] and (t.index == 1).all()
    t2 = lone_mod.tidy_lone_dots(t.index, [0])
    assert t2.count == 0


def test_tidy_dots_drops_unused_color_and_json(modern):
    rgb = modern[0].copy()
    rgb[0, 20] = MODERN_SKY                                   # 端の点を消すと、SPECK はならんだ 2 つと空・壁の点だけ
    rgb[10, 25:27] = MODERN_SKY
    res = _modern(rgb, tidy_dots=True)
    assert tuple(MODERN_SPECK) not in {tuple(c) for c in res.palette}
    assert len(res.palette) == len(res.similar.palette) - 1
    info = json.loads(convert_still_bytes(_png(rgb), "m.png", {"mode": "modern", "width": 64, "tidy_dots": True})["json"])
    assert info["tidy_dots"]["changed"] == 2 and info["tidy_dots"]["settings"]["min_same"] == lone_mod.MIN_SAME
    assert info["settings"]["tidy_dots"] is True


def test_tidy_dots_works_in_pc98_mode(modern):
    rgb, spots = modern
    res = convert_still(rgb, StillSettings(width=64, tidy_dots=True))
    assert res.similar is None and res.lone.changed[spots["sky"]]


def test_dots_overlay_is_display_only_and_follows_canvas(modern):
    rgb, spots = modern
    s = StillSettings.from_mode("modern", width=64, tidy_dots=True, canvas=(70, 44))
    res = convert_still(rgb, s)
    ov = res.dots_overlay()
    ox, oy = res.canvas_offset
    assert ov.shape == (44, 70, 4)
    y, x = spots["sky"]
    assert ov[oy + y, ox + x, 3] == 255 and 0 < ov[oy + y + 1, ox + x, 3] < 255
    assert np.array_equal(ov[..., 3] == 255, np.pad(res.lone.changed, ((oy, 44 - 40 - oy), (ox, 70 - 64 - ox))))
    web = json.loads(still_for_web(_png(rgb), "m.png", json.dumps({"mode": "modern", "width": 64, "tidy_dots": True,
                                                                   "canvas": [70, 44]})))
    got = np.asarray(Image.open(io.BytesIO(base64.b64decode(web["dots_overlay_base64"]))))
    assert np.array_equal(got, ov) and web["tidy_dots"]["changed"] == 2 and web["similar"]["merged"] == 2
    assert base64.b64decode(web["png_base64"]) == convert_still_bytes(_png(rgb), "m.png", s)["png"]
    plain = json.loads(still_for_web(_png(rgb), "m.png", json.dumps({"mode": "modern", "width": 64})))
    assert "dots_overlay_base64" not in plain and "tidy_dots" not in plain


# ---- CLI とブラウザ版 ----

def test_cli_options():
    p = cli.build_parser()
    s = cli.settings_from_args(p.parse_args(["x.png", "--mode", "modern"]))
    assert s == StillSettings.from_mode("modern")
    s = cli.settings_from_args(p.parse_args(["x.png", "--mode", "modern", "--no-merge-similar", "--tidy-dots",
                                             "--colors", "200"]))
    assert (s.merge_similar, s.tidy_dots, s.colors) == (False, True, 200)
    assert cli.settings_from_args(p.parse_args(["x.png", "--tidy-dots"])).tidy_dots is True


def test_cli_rejects_merge_option_in_pc98(tmp_path):
    src = tmp_path / "m.png"
    src.write_bytes(_png(make_modern_scene()[0]))
    # --no-merge-similar は 98 モードでも害はない（もともとまとめない）
    assert cli.main([str(src), "--out", str(tmp_path / "a"), "-q", "--width", "64", "--no-merge-similar"]) == 0


@pytest.mark.parametrize("extra, settings", [
    (["--mode", "modern"], {"mode": "modern"}),
    (["--mode", "modern", "--tidy-dots"], {"mode": "modern", "tidy_dots": True}),
    (["--mode", "modern", "--no-merge-similar", "--colors", "48"], {"mode": "modern", "merge_similar": False, "colors": 48}),
    (["--mode", "modern", "--tidy-dots", "--tidy-pattern", "--canvas", "70x44"],
     {"mode": "modern", "tidy_dots": True, "tidy_pattern": True, "canvas": [70, 44]}),
    (["--tidy-dots"], {"tidy_dots": True}),
])
def test_web_result_equals_cli_files(tmp_path, extra, settings):
    src = tmp_path / "modern.png"
    src.write_bytes(_png(make_modern_scene()[0]))
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", "--width", "64", *extra]) == 0
    res = json.loads(still_for_web(src.read_bytes(), src.name, json.dumps({"width": 64, **settings})))
    assert res["ok"]
    assert base64.b64decode(res["png_base64"]) == (tmp_path / "c.png").read_bytes()
    assert res["json_text"].encode("utf-8") == (tmp_path / "c.json").read_bytes()


def test_cli_prints_counts(tmp_path, capsys):
    src = tmp_path / "m.png"
    src.write_bytes(_png(make_modern_scene()[0]))
    assert cli.main([str(src), "--out", str(tmp_path / "a"), "--width", "64", "--mode", "modern", "--tidy-dots"]) == 0
    out = capsys.readouterr()
    assert "似た色をまとめた組 2" in out.out and "孤立したドットを置き換えた数 2" in out.out
    assert "2 組をまとめ" in out.err


def test_modes_are_listed_for_the_page():
    assert set(MODES) == {"pc98", "modern", "photo"}
    assert all("merge_similar" in m for m in MODES.values())


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_modern_runs_with_int32_numpy(modern):
    rgb, _ = modern
    expected = _modern(rgb, tidy_dots=True)
    with emulate_int32():
        got = _modern(rgb, tidy_dots=True)
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.palette, expected.palette)
    assert got.lone.count == expected.lone.count == 2
