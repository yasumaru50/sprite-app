"""Still App の「1 本の工程と初めの設定」のテスト。

3 つのモード（初めの設定）は項目（ITEMS）の値の組み合わせで、どの項目も、どの初めの設定でも変えられる。
- 初めの設定のままの出力は、この整理の前とバイト単位で同じ（前の版のハッシュと比べる。写真は PNG の入力）
- 項目を変えたときは、JSON に preset（初めの設定の名前・変えた項目・項目の値すべて）が入り、それを除くと前の形
- 新しくできる組み合わせ（マス目を探したうえで色の塊をまとめる・写真の下書きを 4096 色に丸める など）が、
  CLI とブラウザ版で同じバイト列になる
テスト画像はコードで合成する。
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
from retrocore.still import photo as photo_mod
from retrocore.still.convert import (ITEMS, MODES, SMOOTH_CHOICES, StillError, StillSettings, convert_still,
                                     convert_still_bytes, spec, still_for_web)
from still_legacy import legacy_json
from synth_still import enlarge_scene, make_modern_scene, make_scene
from test_still_photo import make_photo


def _png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_exif(rgb, orientation) -> bytes:
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90, exif=exif.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def inputs():
    return {"scene": _png(enlarge_scene(make_scene(), seed=1)), "modern": _png(make_modern_scene()[0]),
            "photo": _png(make_photo())}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _web_values(mode: str) -> dict:
    """ブラウザ版が初めの設定を選んだだけで送る値（項目の値すべて）。"""
    m = spec()["modes"][mode]
    return {"mode": mode, **{k: m[k] for k in ITEMS}}


# ---- 初めの設定のままの出力は前と同じ（この整理の前の版で作った PNG・JSON の SHA-256 の先頭） ----

# 写真の下書きは、ディザを足したときに「似た色をまとめる」の既定を切にしたので、その後の版のハッシュ。
# 前の版のハッシュ（a04756a0ff752417・d3835b004e48f60a）は、写真の下書き＋似た色をまとめる 入 で一致する
# （test_photo_with_merge_similar_matches_before）
DEFAULTS = [
    ("scene", "pc98", "b6ceb690bd3e53f3", "17f26a957ee16a30"),
    ("scene", "modern", "a9d46f4a1f461da1", "e6d08d86972bea33"),
    ("photo", "photo", "5069b3e86137a409", "1f6522442637ad79"),
]


@pytest.mark.parametrize("name, mode, png_hash, json_hash", DEFAULTS)
def test_preset_output_is_unchanged(tmp_path, inputs, name, mode, png_hash, json_hash):
    data = inputs[name]
    # 渡し方を変えても同じ: 何も渡さない（98 だけ）・mode だけ・ブラウザ版が送る項目の値すべて・CLI
    ways = [{"mode": mode}, _web_values(mode)] + ([{}] if mode == "pc98" else [])
    for values in ways:
        out = convert_still_bytes(data, f"{name}.png", values)
        assert _sha(out["png"]) == png_hash and _sha(out["json"].encode("utf-8")) == json_hash, values
        assert "preset" not in out["info"]
        web = json.loads(still_for_web(data, f"{name}.png", json.dumps(values)))
        assert base64.b64decode(web["png_base64"]) == out["png"] and web["json_text"] == out["json"]
    src = tmp_path / f"{name}.png"
    src.write_bytes(data)
    args = [] if mode == "pc98" else ["--mode", mode]
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", *args]) == 0
    assert _sha((tmp_path / "c.png").read_bytes()) == png_hash
    assert _sha((tmp_path / "c.json").read_bytes()) == json_hash


def test_photo_with_merge_similar_matches_before(tmp_path, inputs):
    """写真の下書きの前の既定（似た色をまとめる 入）は、前の版とバイト単位で同じ（preset を除く）。"""
    data = inputs["photo"]
    out = convert_still_bytes(data, "photo.png", {"mode": "photo", "merge_similar": True})
    assert _sha(out["png"]) == "a04756a0ff752417"
    assert _sha(legacy_json(out["json"]).encode("utf-8")) == "d3835b004e48f60a"
    assert list(out["info"]["preset"]["changed"]) == ["merge_similar"]
    assert out["info"]["preset"]["changed"]["merge_similar"] == {"preset": False, "used": True}
    src = tmp_path / "photo.png"
    src.write_bytes(data)
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", "--mode", "photo", "--merge-similar"]) == 0
    assert (tmp_path / "c.png").read_bytes() == out["png"]
    assert (tmp_path / "c.json").read_text(encoding="utf-8") == out["json"]
    # 既定（まとめない）では似た色の記録が無く、settings には merge_similar: false を書く
    off = convert_still_bytes(data, "photo.png", {"mode": "photo"})["info"]
    assert "similar" not in off and off["settings"]["merge_similar"] is False
    assert off["colors"]["used"] >= out["info"]["colors"]["used"]


# ---- 初めの設定 = 項目の値の組み合わせ ----

def test_every_preset_sets_every_item():
    for mode, m in MODES.items():
        assert set(ITEMS) <= set(m), mode
        s = StillSettings.from_mode(mode)
        s.validate()
        assert s.changed() == [] and s.preset_info() is None
    assert spec()["items"] == list(ITEMS)
    assert [c["value"] for c in spec()["smooth_choices"]] == list(SMOOTH_CHOICES) == ["off", "weak", "medium", "strong"]
    # 違いはすべて項目の値（写真の下書きだけの工程は無い）
    assert (MODES["photo"]["orient"], MODES["photo"]["find_grid"], MODES["photo"]["smooth"]) == (True, False, "medium")
    for mode in ("pc98", "modern"):
        assert (MODES[mode]["orient"], MODES[mode]["find_grid"], MODES[mode]["smooth"]) == (False, True, "off")


def test_direct_settings_mean_the_same_as_before():
    assert StillSettings() == StillSettings.from_mode("pc98")
    s = StillSettings(width=40)
    assert s.find_grid is False and s.smooth == "off" and s.orient is False
    p = StillSettings(mode="photo")
    p.validate()
    assert (p.width, p.find_grid, p.smooth, p.orient) == (320, False, "medium", True)


def test_changed_items_are_recorded(inputs):
    info = convert_still_bytes(inputs["modern"], "m.png", {"mode": "modern", "width": 64, "smooth": "weak"})["info"]
    pre = info["preset"]
    assert pre["name"] == "modern"
    assert list(pre["changed"]) == ["find_grid", "width", "smooth"]
    assert pre["changed"]["smooth"] == {"preset": "off", "used": "weak"}
    assert pre["changed"]["width"] == {"preset": None, "used": 64}
    # 比率に合わせて切り抜く（aspect・anchor）は、元のままのときは values に書かない（前と同じバイト列）
    assert list(pre["values"]) == [k for k in ITEMS if k not in ("aspect", "anchor")]
    assert pre["values"]["colors"] == 32 and pre["values"]["max_size"] == [1920, 1080]
    # 初めの設定と同じ値を渡しても「変えた」にはならない
    same = convert_still_bytes(inputs["scene"], "s.png", {"colors": 16, "four_bit": True, "smooth": "off"})["info"]
    assert "preset" not in same


def test_photo_can_find_grid_and_turn_off_steps(inputs):
    s = StillSettings.from_dict({"mode": "photo", "find_grid": True})
    assert s.width is None and s.find_grid is True and s.changed() == ["find_grid", "width"]
    info = convert_still_bytes(inputs["scene"], "s.png", {"mode": "photo", "find_grid": True, "smooth": "off",
                                                          "orient": False, "tidy_dots": False})["info"]
    assert info["dots"]["method"] == "grid" and info["picture"] == {"width": 48, "height": 30}
    assert "photo" not in info
    # 写真の下書きで色の塊をまとめないときは、横のマス数で分けるだけ（いちばん多い色）
    off = convert_still_bytes(inputs["photo"], "p.png", {"mode": "photo", "width": 80, "smooth": "off"})["info"]
    assert off["dots"]["method"] == "width" and off["photo"] == {"orientation": 1, "prescale": None,
                                                                   "work_size": None, "smooth": None}


# ---- 新しい組み合わせ ----

def test_grid_then_smooth(inputs):
    plain = convert_still_bytes(inputs["scene"], "s.png", {"mode": "modern"})
    soft = convert_still_bytes(inputs["scene"], "s.png", {"mode": "modern", "smooth": "weak"})
    a, b = plain["info"], soft["info"]
    assert b["dots"] == a["dots"] and b["dots"]["method"] == "grid"        # マス目は同じ
    assert b["photo"]["smooth"]["level"] == "weak" and b["photo"]["prescale"] == 4
    assert b["photo"]["work_size"] == [48 * 4, 30 * 4] and b["photo"]["orientation"] is None
    assert list(b["preset"]["changed"]) == ["smooth"]
    res = soft["result"]
    assert len(res.palette) == len(np.unique(res.index))


def test_smooth_grid_dots_without_prescale_for_small_cells():
    rgb = np.zeros((30, 30, 3), np.uint8)
    rgb[:, 15:] = 200
    xs = ys = np.arange(0, 31, 3, dtype=np.int64)
    from retrocore.still import dots as dots_mod
    d = dots_mod.Dots(dots_mod.mode_colors(rgb, xs, ys), "grid", xs, ys, 3.0)
    out, work = photo_mod.smooth_grid_dots(rgb, d, "weak")
    assert work.prescaled is False and work.work_size == (30, 30)
    assert out.size == (10, 10) and set(map(tuple, out.rgb.reshape(-1, 3))) <= {(0, 0, 0), (200, 200, 200)}


def test_sub_edges_split_each_cell():
    e = photo_mod._sub_edges(np.array([0, 9, 20], np.int64), 4)
    assert e.tolist()[0] == 0 and e.tolist()[4] == 9 and e.tolist()[-1] == 20 and len(e) == 9
    assert np.all(np.diff(e) >= 2)


def test_photo_with_16_colors_and_4bit(inputs):
    info = convert_still_bytes(inputs["photo"], "p.png", {"mode": "photo", "width": 160, "colors": 16,
                                                          "four_bit": True})["info"]
    assert info["colors"]["limit"] == 16 and info["colors"]["used"] <= 16
    assert all(p["rgb4"] is not None for p in info["palette"])
    assert list(info["preset"]["changed"]) == ["width", "colors", "four_bit"]


def test_pc98_can_use_merge_similar_and_orient():
    rgb = make_photo(240, 180)
    data = _jpeg_exif(rgb, 6)
    info = convert_still_bytes(data, "p.jpg", {"width": 60, "orient": True, "merge_similar": True})["info"]
    assert (info["input"]["width"], info["input"]["height"]) == (180, 240)   # 回した後
    assert info["photo"]["orientation"] == 6 and info["photo"]["smooth"] is None
    assert info["preset"]["changed"]["merge_similar"] == {"preset": False, "used": True}
    off = convert_still_bytes(data, "p.jpg", {"width": 60})["info"]
    assert (off["input"]["width"], off["input"]["height"]) == (240, 180) and "photo" not in off


@pytest.mark.parametrize("values, words", [
    ({"find_grid": True, "width": 50}, "一緒"),
    ({"mode": "modern", "find_grid": False}, "横のマス数"),
    ({"smooth": "max"}, "まとめの強さ"),
    ({"find_grid": "yes"}, "true"),
])
def test_bad_item_values(values, words):
    with pytest.raises(ValueError, match=words):
        StillSettings.from_dict(values)


def test_cli_new_options():
    p = cli.build_parser()
    s = cli.settings_from_args(p.parse_args(["x.png", "--mode", "photo", "--grid", "--no-orient", "--smooth", "off",
                                             "--4bit", "--merge-similar", "--keep-darkest"]))
    assert (s.find_grid, s.width, s.orient, s.smooth, s.four_bit) == (True, None, False, "off", True)
    s = cli.settings_from_args(p.parse_args(["x.png", "--mode", "modern", "--width", "600", "--smooth", "weak"]))
    assert (s.find_grid, s.width, s.smooth) == (False, 600, "weak")
    with pytest.raises(ValueError, match="一緒"):
        cli.settings_from_args(p.parse_args(["x.png", "--grid", "--width", "10"]))


@pytest.mark.parametrize("extra, values, name", [
    (["--mode", "modern", "--smooth", "weak"], {"mode": "modern", "smooth": "weak"}, "scene"),
    (["--mode", "modern", "--width", "64", "--smooth", "weak"], {"mode": "modern", "width": 64, "smooth": "weak"},
     "modern"),
    (["--mode", "photo", "--width", "120", "--colors", "16", "--4bit"],
     {"mode": "photo", "width": 120, "colors": 16, "four_bit": True}, "photo"),
    (["--mode", "photo", "--grid", "--smooth", "strong", "--outline"],
     {"mode": "photo", "find_grid": True, "smooth": "strong", "outline": True}, "scene"),
    (["--merge-similar", "--smooth", "medium", "--tidy-dots", "--canvas", "60x40"],
     {"merge_similar": True, "smooth": "medium", "tidy_dots": True, "canvas": [60, 40]}, "scene"),
])
def test_web_result_equals_cli_files(tmp_path, inputs, extra, values, name):
    src = tmp_path / f"{name}.png"
    src.write_bytes(inputs[name])
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", *extra]) == 0
    # ブラウザ版は項目の値すべてを送る（初めの設定の値に、変えた項目を重ねたもの）
    full = {**_web_values(values.get("mode", "pc98")), **values}
    if values.get("find_grid") is True:
        full["width"] = None
    elif "width" in values:
        full["find_grid"] = False
    for v in (values, full):
        res = json.loads(still_for_web(src.read_bytes(), src.name, json.dumps(v)))
        assert res["ok"], res
        assert base64.b64decode(res["png_base64"]) == (tmp_path / "c.png").read_bytes()
        assert res["json_text"].encode("utf-8") == (tmp_path / "c.json").read_bytes()
    info = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "preset" in info and json.loads(legacy_json(json.dumps(info))).get("preset") is None


def test_size_errors_with_smoothing_say_image():
    rgb = np.zeros((50, 60, 3), np.uint8)
    with pytest.raises(StillError, match="画像の幅") as e:
        convert_still(rgb, StillSettings.from_mode("modern", width=100, smooth="weak"))
    assert e.value.kind == "width"


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_grid_smooth_runs_with_int32_numpy():
    rgb = enlarge_scene(make_scene(), seed=1)
    s = StillSettings.from_mode("modern", smooth="weak")
    expected = convert_still(rgb, s)
    with emulate_int32():
        got = convert_still(rgb, s)
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.palette, expected.palette)
