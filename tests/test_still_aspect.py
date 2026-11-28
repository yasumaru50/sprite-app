"""Still App の「比率に合わせて切り抜く」（retrocore/still/aspect.py、--aspect W:H・--anchor X,Y）のテスト。

- 切り抜くだけ（拡大・縮小しない）。横のマス数で分けるときは入力の段階で、マス目を探すときはマス単位で切る
- 横 320・16:9 ならちょうど 320×180。切る位置は 0〜100 の%（50,50 は中央）
- 元のまま（既定）のときは前とバイト単位で同じ。CLI とブラウザ版で同じバイト列
テスト画像はコードで合成する。
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from apps.still import cli
from int32_numpy import AVAILABLE, emulate_int32
from retrocore.still import aspect as aspect_mod
from retrocore.still import dots as dots_mod
from retrocore.still.convert import (ITEMS, MODES, StillSettings, convert_still, convert_still_bytes, spec,
                                     still_for_web)
from synth_still import enlarge_scene, make_scene


def _png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def make_gradient(w: int, h: int) -> np.ndarray:
    """場所ごとに色が違う絵（どこを切り抜いたかが色で分かる）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    rgb[..., 1] = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    rgb[..., 2] = 80
    return rgb


@pytest.fixture(scope="module")
def ai_scene():
    """マス目を探せる、AI 風に拡大した 1 枚絵（48×30 ドットを 1 ドット 9〜11px に）。"""
    return enlarge_scene(make_scene(48, 30))


# ---- 値の読み方 ----

def test_parse_aspect_and_anchor():
    assert aspect_mod.parse_aspect("16:9") == (16, 9)
    assert aspect_mod.parse_aspect(" 1200 : 630 ") == (1200, 630)
    assert aspect_mod.parse_aspect([21, 9]) == (21, 9)
    for none in (None, "", "none", "元のまま"):
        assert aspect_mod.parse_aspect(none) is None
    for bad in ("16", "16:0", "0:9", "-16:9", "16.5:9", "a:b", "16:9:1", [16, True], "１６:９"):
        with pytest.raises(ValueError, match="比率"):
            aspect_mod.parse_aspect(bad)
    assert aspect_mod.parse_anchor("50,20") == (50, 20)
    assert aspect_mod.parse_anchor([0, 100]) == (0, 100)
    assert aspect_mod.parse_anchor(None) == (50, 50)
    for bad in ("50", "101,0", "-1,50", "a,b", "50.5,0"):
        with pytest.raises(ValueError, match="切る位置"):
            aspect_mod.parse_anchor(bad)


def test_default_is_unchanged_in_every_preset():
    for mode, m in MODES.items():
        assert m["aspect"] is None and m["anchor"] == (50, 50), mode
    assert ITEMS.index("width") < ITEMS.index("aspect") < ITEMS.index("anchor") < ITEMS.index("max_size")
    sp = spec()
    assert [16, 9] in sp["aspect_choices"] and sp["aspect_widths"] == [{"aspect": [16, 9], "widths": [320, 480, 640]}]
    # 比率が元のままなら、切る位置を変えても「変えた項目」にならない
    s = StillSettings.from_dict({"mode": "photo", "anchor": [10, 10]})
    assert s.anchor == (50, 50) and s.changed() == []


# ---- 入力の段階で切り抜く（横のマス数で分けるとき） ----

@pytest.mark.parametrize("w, h", [(300, 400), (400, 300), (600, 400), (1000, 563), (321, 181), (97, 500)])
def test_input_box_gives_the_exact_height(w, h):
    for aspect in ((16, 9), (16, 10), (4, 3), (3, 2), (1, 1), (21, 9), (1200, 630), (9, 16)):
        for width in (1, 7, 30, 64, 97, 160, 320):
            box, want_h = aspect_mod.input_box(w, h, width, aspect)
            x0, y0, x1, y1 = box
            cw, ch = x1 - x0, y1 - y0
            assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h
            assert cw == w or ch == h   # いちばん大きい四角（片方の辺はそのまま）
            if width <= cw and want_h <= ch:
                assert dots_mod.height_for_width(width, cw, ch) == want_h, (w, h, aspect, width)


def test_width_320_16_9_is_exactly_320x180():
    rgb = make_gradient(900, 1200)   # 縦長の写真
    for mode in ("photo", "modern"):
        res = convert_still(rgb, StillSettings.from_mode(mode, width=320, aspect=(16, 9), colors=8))
        assert res.size == (320, 180), mode
        a = res.aspect
        assert a["stage"] == "input" and a["ratio"] == [16, 9] and a["exact"] is True
        # 幅はそのまま、縦を上下に同じだけ切る（900×506 を残す）
        assert a["box"] == [0, 347, 900, 853] and a["cut"] == {"left": 0, "top": 347, "right": 0, "bottom": 347}
        assert res.input_size == (900, 1200)
        assert res.dots.describe()["area"] == [0, 347, 900, 853]


def test_anchor_moves_the_box_without_resizing():
    rgb = make_gradient(900, 1200)
    top = convert_still(rgb, StillSettings.from_mode("photo", aspect=(16, 9), anchor=(50, 0), smooth="off",
                                                     tidy_dots=False))
    mid = convert_still(rgb, StillSettings.from_mode("photo", aspect=(16, 9), anchor=(50, 20), smooth="off",
                                                     tidy_dots=False))
    assert top.aspect["box"] == [0, 0, 900, 506]
    assert mid.aspect["box"] == [0, 139, 900, 645]   # 694 × 0.2 = 138.8 → 139
    assert top.size == mid.size == (320, 180)
    # 横に長い絵は、左右で切る
    wide = convert_still(make_gradient(1200, 600), StillSettings.from_mode("modern", width=100, aspect=(4, 3),
                                                                            anchor=(0, 50)))
    assert wide.aspect["box"] == [0, 0, 800, 600] and wide.size == (100, 75)


def test_input_crop_is_the_same_as_converting_the_cropped_image():
    """切り抜いてから工程に通すのと、先に切り抜いた画像を通すのは同じ結果（拡大・縮小はしない）。"""
    rgb = make_gradient(640, 480)
    s = StillSettings.from_mode("photo", width=160, aspect=(16, 10), anchor=(30, 70))
    res = convert_still(rgb, s)
    x0, y0, x1, y1 = res.aspect["box"]
    ref = convert_still(np.ascontiguousarray(rgb[y0:y1, x0:x1]), StillSettings.from_mode("photo", width=160))
    assert res.size == (160, 100)
    assert np.array_equal(res.rgb(), ref.rgb())


def test_too_many_cells_for_the_cropped_image():
    with pytest.raises(Exception, match="比率 16:9 に切り抜いた"):
        convert_still(make_gradient(200, 400), StillSettings.from_mode("modern", width=250, aspect=(16, 9)))


# ---- マス単位で切る（マス目を探すとき） ----

def test_cell_box():
    assert aspect_mod.cell_box(48, 30, (16, 9)) == (0, 2, 48, 29)        # 48 × 9 ÷ 16 = 27 行。3 行を上 2・下 1（1.5 → 2）
    assert aspect_mod.cell_box(48, 30, (16, 9), (50, 0)) == (0, 0, 48, 27)
    assert aspect_mod.cell_box(48, 30, (1, 1), (100, 50)) == (18, 0, 48, 30)
    assert aspect_mod.cell_box(100, 75, (16, 9)) == (0, 10, 100, 66)      # 56.25 → 56 行。19 行を上 10（9.5 → 10）・下 9


def test_grid_crop_keeps_the_grid(ai_scene):
    full = convert_still(ai_scene, StillSettings.from_mode("modern"))
    cut = convert_still(ai_scene, StillSettings.from_mode("modern", aspect=(16, 9), anchor=(50, 0)))
    nx, ny = full.size
    k = (2 * nx * 9 + 16) // 32   # 横はそのまま、縦を 16:9 に（四捨五入）
    assert ny > k and cut.size == (nx, k)
    a = cut.aspect
    assert a["stage"] == "cells" and a["cut_cells"] == {"left": 0, "top": 0, "right": 0, "bottom": ny - k}
    # 格子はずれない: マスの境目は、切り抜く前の境目の一部
    assert np.array_equal(cut.dots.xs, full.dots.xs) and np.array_equal(cut.dots.ys, full.dots.ys[:k + 1])
    assert np.array_equal(cut.dots.rgb, full.dots.rgb[:k])
    assert a["box"] == [int(full.dots.xs[0]), int(full.dots.ys[0]), int(full.dots.xs[-1]), int(full.dots.ys[k])]
    assert a["cut"]["bottom"] == int(full.dots.ys[-1] - full.dots.ys[k]) and a["cut"]["top"] == 0
    # 下寄せ・中央でも、残したマスの色は切り抜く前と同じ
    mid = convert_still(ai_scene, StillSettings.from_mode("modern", aspect=(16, 9)))
    t = mid.aspect["cut_cells"]["top"]
    assert np.array_equal(mid.dots.rgb, full.dots.rgb[t:t + k])


def test_grid_crop_before_the_size_limit(ai_scene):
    """切り抜いてから大きさの上限を確かめる（元の大きさは上限を超えるが、16:9 に切ると収まる）。"""
    nx, ny = convert_still(ai_scene, StillSettings.from_mode("pc98")).size
    k = (2 * nx * 9 + 16) // 32
    with pytest.raises(Exception, match="上限"):
        convert_still(ai_scene, StillSettings.from_mode("pc98", max_size=(nx, k)))
    res = convert_still(ai_scene, StillSettings.from_mode("pc98", max_size=(nx, k), aspect=(16, 9)))
    assert res.size == (nx, k)


# ---- JSON・CLI・ブラウザ版 ----

def test_json_records_ratio_anchor_and_cut():
    data = _png(make_gradient(400, 300))
    off = convert_still_bytes(data, "g.png", {"mode": "modern", "width": 80})
    on = convert_still_bytes(data, "g.png", {"mode": "modern", "width": 80, "aspect": "16:9", "anchor": [50, 20]})
    assert "aspect" not in off["info"] and "aspect" not in off["info"]["settings"]
    assert "aspect" not in off["info"]["preset"]["values"]
    info = on["info"]
    assert info["width"] == 80 and info["height"] == 45 and info["input"] == {"width": 400, "height": 300}
    assert info["aspect"] == {"ratio": [16, 9], "anchor": [50, 20], "stage": "input", "box": [0, 15, 400, 240],
                              "cut": {"left": 0, "top": 15, "right": 0, "bottom": 60}, "exact": True}
    assert info["settings"]["aspect"] == [16, 9] and info["settings"]["anchor"] == [50, 20]
    pre = info["preset"]
    assert pre["changed"]["aspect"] == {"preset": None, "used": [16, 9]}
    assert pre["changed"]["anchor"] == {"preset": [50, 50], "used": [50, 20]}
    assert pre["values"]["aspect"] == [16, 9]


def test_cli_and_web_give_the_same_bytes(tmp_path, ai_scene):
    cases = [(_png(make_gradient(450, 600)), ["--mode", "photo", "--aspect", "16:9", "--anchor", "50,20"],
              {"mode": "photo", "aspect": [16, 9], "anchor": [50, 20]}),
             (_png(ai_scene[..., :3]), ["--mode", "modern", "--aspect", "1:1", "--anchor", "100,50"],
              {"mode": "modern", "aspect": "1:1", "anchor": "100,50"})]
    for i, (data, args, values) in enumerate(cases):
        src = tmp_path / f"in{i}.png"
        src.write_bytes(data)
        assert cli.main([str(src), "--out", str(tmp_path / f"o{i}"), "-q", *args]) == 0
        web = json.loads(still_for_web(data, src.name, json.dumps(values)))
        assert web["ok"], web
        assert base64.b64decode(web["png_base64"]) == (tmp_path / f"o{i}.png").read_bytes()
        assert web["json_text"] == (tmp_path / f"o{i}.json").read_text(encoding="utf-8")
        assert web["aspect"] == json.loads(web["json_text"])["aspect"]


def test_cli_errors(tmp_path, capsys):
    src = tmp_path / "in.png"
    src.write_bytes(_png(make_gradient(64, 64)))
    for args, word in ((["--aspect", "16-9"], "比率"), (["--aspect", "16:9", "--anchor", "120,0"], "切る位置"),
                       (["--anchor", "50,0"], "一緒に")):
        with pytest.raises(SystemExit):
            cli.main([str(src), "--width", "32", *args])
        assert word in capsys.readouterr().err
    web = json.loads(still_for_web(src.read_bytes(), "in.png", json.dumps({"width": 32, "aspect": "0:1"})))
    assert not web["ok"] and "比率" in web["error"]


def test_canvas_after_crop():
    """切り抜いたあとに余白を足せる（16:9 に切ってから 330×190 のキャンバスに）。"""
    res = convert_still(make_gradient(600, 600), StillSettings.from_mode("modern", width=320, aspect=(16, 9),
                                                                         canvas=(330, 190)))
    assert res.size == (330, 190) and res.dots.size == (320, 180) and res.canvas_offset == (5, 5)


@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_runs_with_32bit_intp(ai_scene):
    rgb = make_gradient(450, 600)
    want = convert_still(rgb, StillSettings.from_mode("photo", aspect=(16, 9)))
    want_g = convert_still(ai_scene, StillSettings.from_mode("modern", aspect=(16, 9), smooth="weak"))
    with emulate_int32():
        got = convert_still(rgb, StillSettings.from_mode("photo", aspect=(16, 9)))
        got_g = convert_still(ai_scene, StillSettings.from_mode("modern", aspect=(16, 9), smooth="weak"))
    assert np.array_equal(want.index, got.index) and np.array_equal(want_g.index, got_g.index)
