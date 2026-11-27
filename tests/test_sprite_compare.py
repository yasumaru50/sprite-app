"""大きさをくらべる（retrocore/sprite/compare.py、CLI の compare、画面の compare_for_web）を確かめる。"""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from apps.sprite.cli import main as cli_main
from retrocore.sprite import compare as cmp
from retrocore.sprite.convert import encode_png
from retrocore.sprite.image import load_rgba


def _sprite(h, w, color=(200, 60, 40), pad=(3, 5, 2, 4)):
    """上・右・下・左に透明な余白のある、h×w の不透明な四角の絵。"""
    top, right, bottom, left = pad
    a = np.zeros((h + top + bottom, w + left + right, 4), np.uint8)
    a[top:top + h, left:left + w] = color + (255,)
    return a


def _png(a):
    return encode_png(a)


def test_crop_to_content():
    a = _sprite(10, 6)
    c = cmp.crop_to_content(a)
    assert c.shape[:2] == (10, 6)
    empty = np.zeros((4, 5, 4), np.uint8)
    assert cmp.crop_to_content(empty).shape[:2] == (4, 5)


def test_margins_are_cropped_and_feet_aligned():
    img, info = cmp.compare_images([("a", _sprite(20, 8)), ("b", _sprite(7, 5, pad=(0, 9, 11, 1)))], scale=2, grid=16, gap=4)
    a, b = info["items"]
    assert (a["width"], a["height"], b["width"], b["height"]) == (8, 20, 5, 7)
    s, ground = 2, info["ground_y"] // 2
    for it in info["items"]:
        x = it["x"] * s + 1
        # 足元の行（地面の線のすぐ上）は絵の色、そのすぐ下は地面の線
        assert tuple(img[ground * s - 1, x, :3]) == (200, 60, 40)
        assert tuple(img[ground * s, x, :3]) == cmp.GROUND_COLOR
        # 絵のいちばん上の行のすぐ上は絵ではない
        top = (ground - it["height"]) * s
        assert tuple(img[top, x, :3]) == (200, 60, 40)
        assert tuple(img[top - 1, x, :3]) != (200, 60, 40)
    assert info["first_top_y"] == (ground - 20) * s


def test_grid_is_counted_from_the_ground():
    s, g = 3, 5
    img, info = cmp.compare_images([("a", _sprite(13, 4))], scale=s, grid=g, gap=3)
    ground = info["ground_y"] // s
    x = img.shape[1] - 2   # 絵の無い右の端の近く（縦の線を避ける）
    col = img[:info["ground_y"], x, :3]
    rows = sorted({int(r) for r in np.flatnonzero((col == cmp.GRID_COLOR).all(axis=1))})
    assert rows == [(ground - k * g) * s for k in range(ground // g, 0, -1)]
    assert ground % g == 0 and rows[0] == 0   # いちばん上の端もマス目の線（高さがマス目で数えられる）
    # 縦の線は左端から
    row = img[1, :, :3]
    cols = [int(c) for c in np.flatnonzero((row == cmp.GRID_COLOR).all(axis=1))]
    assert cols[:3] == [0, g * s, 2 * g * s]


def test_dotted_line_at_first_top():
    img, info = cmp.compare_images([("a", _sprite(10, 4)), ("b", _sprite(20, 4))], scale=4, grid=8, gap=8)
    y = info["first_top_y"] - 1
    row = img[y, :, :3]
    on = (row == cmp.TOP_COLOR).all(axis=1)
    assert on.any() and not on.all()   # 点線（横いっぱい、とびとび）
    assert on[:4].all() and not on[4:8].any()


def test_sheet_json_uses_first_frame():
    sheet = np.zeros((12, 30, 4), np.uint8)
    sheet[2:12, 1:5] = (10, 200, 10, 255)       # 1 コマ目: 4x10
    sheet[0:12, 10:20] = (10, 10, 200, 255)     # 2 コマ目: もっと大きい
    js = {"version": 2, "frame_width": 10, "frame_height": 12,
          "frames": [{"index": 0, "x": 0, "y": 0, "width": 10, "height": 12}, {"index": 1, "x": 10, "y": 0, "width": 10, "height": 12}]}
    res = cmp.compare_entries([("walk.png", _png(sheet), json.dumps(js))])
    it = res["items"][0]
    assert (it["width"], it["height"], it["sheet"]) == (4, 10, True)
    # JSON が無ければ PNG 全体
    res = cmp.compare_entries([("walk.png", _png(sheet), None)])
    assert (res["items"][0]["width"], res["items"][0]["height"]) == (19, 12)
    # frames が無い JSON は frame_width・frame_height を使う
    res = cmp.compare_entries([("walk.png", _png(sheet), json.dumps({"frame_width": 10, "frame_height": 12}))])
    assert res["items"][0]["width"] == 4
    with pytest.raises(ValueError, match="1 コマ目"):
        cmp.compare_entries([("walk.png", _png(sheet), json.dumps({"frame_width": 40, "frame_height": 12}))])


def test_pair_files():
    assert cmp.pair_files(["a.png", "w.json", "w.png", "b.png"]) == [(0, None), (2, 1), (3, None)]
    with pytest.raises(ValueError, match="x.json"):
        cmp.pair_files(["a.png", "x.json"])


def test_japanese_name_becomes_number():
    res = cmp.compare_entries([("hero.png", _png(_sprite(10, 4)), None), ("きのこ.png", _png(_sprite(5, 4)), None),
                               ("slime_01.png", _png(_sprite(5, 4)), None)])
    assert [it["label"] for it in res["items"]] == ["hero", "#2", "slime_01"]
    assert res["numbered"] == [{"label": "#2", "name": "きのこ"}]
    assert "#2 = きのこ" in cmp.summary_lines(res)
    assert not cmp.can_write("a b") and not cmp.can_write("x" * 33) and cmp.can_write("Walk-2_(L).v2+")


def test_labels_do_not_overlap():
    names = ["a_very_long_file_name_for_test", "b", "another_long_name"]
    for scale in (1, 2, 4, 16):
        img, info = cmp.compare_images([(n, _sprite(3, 2)) for n in names], scale=scale, grid=4, gap=0)
        t = cmp.text_scale(scale)
        its = info["items"]
        for a, b in zip(its, its[1:]):
            text = f"{a['label']} {a['width']}x{a['height']}"
            assert a["x"] * scale + cmp.text_width(text, t) <= b["x"] * scale
        last = its[-1]
        assert last["x"] * scale + cmp.text_width(f"{last['label']} 2x3", t) <= img.shape[1]
        # 文字の帯で、文字の色の画素は名前ごとの範囲の中だけにある
        strip = img[img.shape[0] - (cmp.GLYPH_H + 5) * t:, :, :3]
        ink = set(np.flatnonzero((strip == cmp.LABEL_COLOR).all(axis=2).any(axis=0)).tolist())
        spans = set()
        for it in its:
            x0 = it["x"] * scale
            spans |= set(range(x0, x0 + cmp.text_width(f"{it['label']} {it['width']}x{it['height']}", t)))
        assert ink and ink <= spans


def test_glyph_table():
    for c, m in cmp.GLYPHS.items():
        assert m.shape == (cmp.GLYPH_H, cmp.GLYPH_W)
    for c in "AZaz09_-.+()#x ":
        assert c in cmp.GLYPHS


def test_options_are_checked():
    a = [("a", _sprite(3, 3))]
    for kw in ({"scale": 0}, {"scale": 17}, {"grid": 0}, {"grid": 257}, {"gap": -1}):
        with pytest.raises(ValueError):
            cmp.compare_images(a, **kw)
    with pytest.raises(ValueError):
        cmp.compare_images([])


def test_cli_and_web_give_the_same_bytes(tmp_path, capsys):
    sheet = np.zeros((12, 20, 4), np.uint8)
    sheet[2:12, 1:5] = (10, 200, 10, 255)
    sheet[0:12, 11:19] = (10, 10, 200, 255)
    js = json.dumps({"version": 2, "frame_width": 10, "frame_height": 12,
                     "frames": [{"index": 0, "x": 0, "y": 0, "width": 10, "height": 12}]})
    files = {"hero.png": _png(_sprite(20, 8)), "きのこ.png": _png(_sprite(9, 7, (90, 90, 20))),
             "walk.png": _png(sheet), "walk.json": js.encode()}
    for n, d in files.items():
        (tmp_path / n).write_bytes(d)
    out = tmp_path / "o" / "compare.png"
    args = ["compare"] + [str(tmp_path / n) for n in files] + ["--scale", "3", "--grid", "8", "--gap", "5", "--out", str(out)]
    assert cli_main(args) == 0
    text = capsys.readouterr().out
    assert "#2 = きのこ" in text and "walk: 4x10（シートの 1 コマ目）" in text
    web = json.loads(cmp.compare_for_web(list(files.values()), list(files), json.dumps({"scale": 3, "grid": 8, "gap": 5})))
    assert web["ok"] and web["name"] == "compare.png"
    assert base64.b64decode(web["png_base64"]) == out.read_bytes()
    assert load_rgba(out).shape[2] == 4
    # 既定の値どうしも同じ
    out2 = tmp_path / "d.png"
    assert cli_main(["compare", str(tmp_path / "hero.png"), "-o", str(out2), "-q"]) == 0
    web = json.loads(cmp.compare_for_web([files["hero.png"]], ["hero.png"]))
    assert base64.b64decode(web["png_base64"]) == out2.read_bytes()


def test_web_reports_errors():
    r = json.loads(cmp.compare_for_web([b"x"], ["a.png"]))
    assert not r["ok"] and "読み込めません" in r["error"]
    r = json.loads(cmp.compare_for_web([_png(_sprite(3, 3)), b"{}"], ["a.png", "b.json"]))
    assert not r["ok"] and "b.json" in r["error"]


def test_cli_errors(tmp_path, capsys):
    (tmp_path / "a.png").write_bytes(_png(_sprite(3, 3)))
    with pytest.raises(SystemExit):
        cli_main(["compare", str(tmp_path / "a.png"), "--scale", "20", "-o", str(tmp_path / "c.png")])
    assert "倍率" in capsys.readouterr().err
