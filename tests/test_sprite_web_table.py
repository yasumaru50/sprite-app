"""ブラウザ版の「表で編集」（画面の表）のテスト。

画面は表を JSON にして `motion_for_web()` に {"spec": 表, "image_name": ...} で渡し、表の準備（プリセットの表・ファイルの読み込み）は
`table_for_web()` で行う。ここでは次を確かめる:

- 画面で作ったシートの PNG（バイトと画素）と JSON が、同じ表を CLI の `--motion-spec` に渡したときとバイト単位で同じ
- 画面で保存する表（数・割合）が、CLI の `--write-motion-spec --spec-height number|percent` とバイト単位で同じ
- 「表で編集」で開いたプリセットの表から作ったシートが、プリセットのシートと同じ（JSON は regions_used を除いて同じ）
- 書き間違いのあるファイルは、全部の問題をまとめて返し、表を返さない（画面は表を変えない）
- 作れない行は、行（1 から）と列の名前つきで返す（画面はそのマスを赤くする）
"""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite.convert import (TABLE_COLUMNS, convert_for_web, motion_for_web, motion_spec,
                                      spec_error_items, table_for_web)
from retrocore.sprite.motion import preset_table
from synth_sprite import enlarge_like_ai, make_pixel_sprite

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "tests" / "motion_specs"

# 呼吸の表から作った「呼吸と横揺れを交互に」（同じ行で横と縦を一緒に使うのはまだ作れないので、行を分ける）
BREATHE_SWAY = {
    "format": "sprite-motion", "version": 1, "name": "breathe_sway", "loop": True,
    "regions": {"upper": {"type": "split", "at": "auto"}},
    "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"},
               "upper_x": {"type": "move", "region": "upper", "axis": "x"}},
    "frames": [{"ms": 300}, {"ms": 300, "upper_y": -1}, {"ms": 300}, {"ms": 300, "upper_x": 1},
               {"ms": 300}, {"ms": 300, "upper_y": -1}, {"ms": 300}, {"ms": 300, "upper_x": -1}],
}


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory) -> Path:
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture(scope="module")
def sprite_png(pixel_png) -> bytes:
    res = json.loads(convert_for_web(pixel_png.read_bytes(), "{}"))
    assert res["ok"], res
    return base64.b64decode(res["png_base64"])


def _text_file_bytes(text: str) -> bytes:
    """CLI の write_text() と同じ（Windows では改行が CRLF になる）。画面は LF のまま保存する"""
    return text.replace("\n", os.linesep).encode("utf-8")


def _pixels(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as im:
        return np.asarray(im.convert("RGBA"))


def _table(spec, sprite_png: bytes, image_name: str) -> dict:
    res = json.loads(motion_for_web(sprite_png, json.dumps({"spec": spec, "image_name": image_name})))
    assert res["ok"], res
    return res


TABLES = {
    "bounce": json.loads((SPECS / "bounce.motion.json").read_text(encoding="utf-8")),
    "bounce_squash": json.loads((SPECS / "bounce_squash.motion.json").read_text(encoding="utf-8")),
    "breathe_sway": BREATHE_SWAY,
    "breathe_12": {**preset_table("breathe", 4, 400, depth=2, split=12), "name": "breathe12"},
    "sway_percent": {**preset_table("sway", 8, 150, amount=2), "regions": {"upper": {"type": "split", "at": "30%"}}},
    "float_mix": {**preset_table("float"), "tracks": {**preset_table("float")["tracks"], **TABLE_COLUMNS},
                  "regions": {"upper": {"type": "split", "at": "auto"}},
                  "frames": [{"ms": 200}, {"ms": 200, "root_y": -1, "upper_y": -1}, {"ms": 200, "upper_x": 1},
                             {"ms": 200, "root_y": 1}]},
}
TABLES["float_mix"]["name"] = "mix"


@pytest.mark.parametrize("key", list(TABLES))
def test_web_table_equals_cli_motion_spec(pixel_png, sprite_png, tmp_path, key):
    """画面の表で作ったシート（PNG と JSON）と保存する表が、CLI の --motion-spec / --write-motion-spec とバイト単位で同じ。"""
    spec = TABLES[key]
    path = tmp_path / f"{key}.motion.json"
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    for style in ("number", "percent"):
        out = tmp_path / style
        dst = out / "hero_sprite.png"
        written = out / "again.motion.json"
        assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion-spec", str(path),
                         "--write-motion-spec", str(written), "--spec-height", style]) == 0
        assert dst.read_bytes() == sprite_png
        sheet = out / f"hero_sprite_{spec['name']}_sheet.png"
        res = _table(spec, sprite_png, sheet.name)
        assert res["skipped"] is False and res["warnings"] == [] and res["problems"] == []
        png = base64.b64decode(res["sheet_png_base64"])
        assert png == sheet.read_bytes()
        assert np.array_equal(_pixels(png), _pixels(sheet.read_bytes()))
        assert _text_file_bytes(res["sheet_json"]) == sheet.with_suffix(".json").read_bytes()
        assert _text_file_bytes(res["spec_files"][style]) == written.read_bytes()
        info = json.loads(res["sheet_json"])
        assert res["durations"] == [f["duration_ms"] for f in info["frames"]]
        assert (res["frame_width"], res["frame_height"]) == (info["frame_width"], info["frame_height"])


@pytest.mark.parametrize("name, options", [
    ("breathe", {}), ("breathe", {"frames": 4, "depth": 2, "durations": [300, 200, 300, 200], "split": 12}),
    ("sway", {}), ("sway", {"frames": 8, "amount": 2, "split": 10}),
    ("float", {}), ("float", {"amount": 2, "frames": 8}),
    ("bounce", {}), ("bounce", {"height": 3}), ("bounce", {"height": 1, "durations": 150}),
])
def test_preset_opened_as_table_gives_same_sheet(sprite_png, name, options):
    """「表で編集」で開いたプリセットの表は preset_table() と同じで、作ったシートはプリセットと同じ。"""
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": name, **options}})))
    assert got["ok"], got
    opts = {k: v for k, v in options.items() if k != "durations"}
    frames = opts.pop("frames", None)
    assert got["spec"] == preset_table(name, frames, options.get("durations"), **opts)
    image = f"hero_sprite_{name}_sheet.png"
    preset = json.loads(motion_for_web(sprite_png, json.dumps({"motion": name, **options, "image_name": image})))
    table = _table(got["spec"], sprite_png, image)
    assert preset["skipped"] is False and table["skipped"] is False
    assert table["sheet_png_base64"] == preset["sheet_png_base64"]
    info = json.loads(table["sheet_json"])
    info.pop("regions_used", None)   # 表から作ったシートには、"auto" で実際に使った高さが付く（設計の 7.）
    assert json.dumps(info, ensure_ascii=False, indent=2) + "\n" == preset["sheet_json"]


def test_preset_results_are_unchanged_by_table_keys(sprite_png):
    """プリセットの返す値には、表の経路の項目（problems・spec_files など）を足していない（前の版と同じ）。"""
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "breathe"})))
    assert not ({"table", "problems", "spec_files", "spec"} & set(res))


def test_load_file_gives_table(tmp_path):
    for p in sorted(SPECS.glob("*.motion.json")):
        got = json.loads(table_for_web(json.dumps({"text": p.read_text(encoding="utf-8")})))
        assert got["ok"], got
        assert got["spec"]["name"] == json.loads(p.read_text(encoding="utf-8"))["name"]
    # 先頭の BOM（Windows のメモ帳で保存したファイル）も読める
    bom = "﻿" + (SPECS / "bounce.motion.json").read_text(encoding="utf-8")
    assert json.loads(table_for_web(json.dumps({"text": bom})))["ok"]


def test_load_broken_file_lists_every_problem():
    """書き間違いのあるファイルは、全部の問題をまとめて返す（表は返さない＝画面の表は変わらない）。"""
    text = json.dumps({"format": "sprite-motion", "version": 2, "name": "a b",
                       "regions": {"upper": {"type": "split", "at": 0}},
                       "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
                       "frames": [{"ms": 0}, {"ms": 200, "upper_y": 9, "zz": 1}]})
    got = json.loads(table_for_web(json.dumps({"text": text})))
    assert got["ok"] is False and "spec" not in got
    items = got["spec_errors"]
    assert len(items) == 6
    rows = [(i["row"], i["track"]) for i in items]
    assert (1, "ms") in rows and (2, "upper_y") in rows and (2, "zz") in rows
    for i in items:
        assert i["message"] in got["error"]


@pytest.mark.parametrize("text, words", [
    ("{", "JSON として読めません"),
    (json.dumps({"version": 1, "frame_width": 3}), "スプライトシートの JSON"),
    (json.dumps([1, 2]), "JSON のオブジェクト"),
])
def test_load_not_a_table(text, words):
    got = json.loads(table_for_web(json.dumps({"text": text})))
    assert got["ok"] is False and words in got["error"]


def test_same_row_move_and_deform_is_marked_on_the_row(sprite_png):
    spec = json.loads(json.dumps(BREATHE_SWAY))
    spec["frames"][3]["upper_y"] = -1
    res = json.loads(motion_for_web(sprite_png, json.dumps({"spec": spec})))
    assert res["ok"] is False
    assert [(i["row"], i["track"]) for i in res["spec_errors"]] == [(4, None)]
    assert "横の位置と縦の変形を同じ行で一緒に使うのは、まだ作れません" in res["error"]


def test_unbuildable_rows_are_reported_with_rows_and_tracks(sprite_png):
    """作れない行は、シートを作らず、行（1 から）・列・近くで分けられる高さを返す（画面はマスを赤くし「○○ で試す」を出す）。"""
    spec = json.loads((SPECS / "broken.motion.json").read_text(encoding="utf-8"))
    res = _table(spec, sprite_png, "x.png")
    assert res["skipped"] is True and "sheet_png_base64" not in res
    marks = {(r, q["track"]) for q in res["problems"] for r in q["rows"]}
    assert marks == {(2, "upper_y"), (3, "upper_y"), (4, "upper_y")}
    assert res["warnings"][0].startswith("動き「broken」のシートを作りませんでした")
    assert all(q["message"] in res["warnings"] for q in res["problems"])
    assert set(res["spec_files"]) == {"number", "percent"}   # 作れなくても表は保存できる（CLI も書く）
    # 分けられない高さ: 近くで分けられる高さ（near）を返す（画面の「○○ で試す」）
    seen = []
    for at in range(1, 23):
        res = _table(dict(BREATHE_SWAY, regions={"upper": {"type": "split", "at": at}}), sprite_png, "x.png")
        seen += [q for q in res["problems"] if q["rule"] == "split_not_found" and q.get("near")]
    assert seen
    for q in seen:
        assert all(isinstance(v, int) for v in q["near"]) and f"近くで分けられる高さ: " in q["message"]


def test_out_of_image_height_is_error(sprite_png):
    spec = dict(BREATHE_SWAY, regions={"upper": {"type": "split", "at": 500}})
    res = json.loads(motion_for_web(sprite_png, json.dumps({"spec": spec})))
    assert res["ok"] is False and "範囲 upper の at" in res["error"]


def test_bad_table_requests():
    assert json.loads(table_for_web("{}"))["ok"] is False
    assert json.loads(table_for_web(json.dumps({"preset": {"motion": "nope"}})))["ok"] is False
    assert json.loads(motion_for_web(b"", json.dumps({"spec": BREATHE_SWAY, "frames": 3})))["ok"] is False


def test_spec_error_items():
    items = spec_error_items(["3 行目: upper_x の値は −2〜+2 の整数で書いてください: 5", "name は…",
                              "2 行目: ms（表示時間、1 以上のミリ秒の整数）を書いてください: 0"])
    assert [(i["row"], i["track"]) for i in items] == [(3, "upper_x"), (None, None), (2, "ms")]


def test_table_spec_in_motion_spec():
    t = motion_spec()["table"]
    assert t["max_value"] == 2 and t["max_root"] == 3 and t["min_rows"] == 2 and t["max_rows"] == 32
    assert [c[0] for c in t["columns"]] == ["upper_x", "upper_y", "root_y"]


def test_web_html_has_table_ui():
    html = (ROOT / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    for words in ("表で編集", "行（1 から）", "1 枚目", "表を保存", "表を読み込む", "数（このキャラ用）",
                  "割合（別のキャラにも使う）", "table_for_web"):
        assert words in html
