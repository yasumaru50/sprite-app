"""作らなかった理由の記録（シートの JSON の `skipped`、`<出力名>_<動き>_skipped.json`。`retrocore/sprite/motion/report.py`）のテスト。

- 作れたシートの JSON は版 2 で `skipped` が空の一覧。ほかの項目は前の版と同じ
- 作れなかった動き（うなずくの首が見つからない・分け目が見つからない・四角が作れない）は、frame（0 から）・row（1 から）・
  track・code・「○○ で試す」と同じ候補（suggestions）・result = held を記録し、CLI はシートの代わりに `_skipped.json` を書く
- パレットの警告も stage = palette で記録する
- 画面（motion_for_web）の JSON は CLI のファイルとバイト単位で同じ
- 新しい判定は足さない（記録の message は今の警告の文と同じ）
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from legacy_output import legacy_sheet_text
from motion_records import make_knight
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import convert_for_web, encode_png, motion_for_web
from retrocore.sprite.motion import make_motion, make_motion_from_spec, sheet_info, sheet_json
from retrocore.sprite.motion.report import CODES, STAGES, skipped_info, skipped_json
from retrocore.sprite.motion.sheet import SHEET_VERSION
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

ROOT = Path(__file__).resolve().parents[1]
HEAD = {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22}   # make_knight() の頭（test_sprite_region.py と同じ）


def _table(regions, tracks, frames, name="t") -> dict:
    return {"format": "sprite-motion", "version": 1, "name": name, "loop": True, "regions": regions,
            "tracks": tracks, "frames": frames}


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def knight():
    return make_knight()


def _check_entry(e: dict) -> None:
    assert e["stage"] in STAGES and e["code"] in CODES and isinstance(e["message"], str) and e["message"]
    assert isinstance(e["suggestions"], list)
    if e["stage"] == "motion":
        assert e["result"] == "held"
        assert e["frames"] == [r - 1 for r in e["rows"]] and e["frame"] == e["frames"][0] and e["row"] == e["rows"][0]


# ---- 作れたとき ----

def test_made_sheet_has_empty_skipped_and_same_other_items():
    sp = make_pixel_sprite()
    res = make_motion(sp, "breathe")
    assert not res.skipped and res.skips == []
    info = sheet_info(res, "x.png")
    assert SHEET_VERSION == 2 and info["version"] == 2 and info["skipped"] == []
    assert list(info)[-1] == "skipped"   # 足すのはいちばん後ろだけ
    old = json.loads(legacy_sheet_text(sheet_json(res, "x.png")))
    assert old["version"] == 1 and "skipped" not in old
    assert {k: v for k, v in info.items() if k not in ("version", "skipped")} == {k: v for k, v in old.items() if k != "version"}


# ---- 作れなかったとき: 動き ----

def test_nod_without_neck_records_no_neck_found():
    res = make_motion(make_slime(), "nod")
    assert res.skipped and len(res.skips) == 1
    e = res.skips[0]
    _check_entry(e)
    assert e["code"] == "no_neck_found" and e["message"] == res.warnings[0]
    assert e["frames"] == [1] and e["rows"] == [2] and e["track"] == "upper_y" and e["values"] == [-1]
    # 候補は警告の「試せる高さ: …」と同じ数
    hint = re.search(r"試せる高さ: ([0-9、]+)", res.warnings[0]).group(1).split("、")
    assert e["suggestions"] == [{"nod_split": int(v)} for v in hint]
    # 候補の高さで実際に作れる
    for s in e["suggestions"]:
        assert not make_motion(make_slime(), "nod", split=s["nod_split"]).skipped
    rows_from_bottom = res.info["neck"]["band"]
    assert e["rows_from_bottom"] == rows_from_bottom


def test_preset_split_not_found_records_rows_and_suggestions(knight):
    res = make_motion(knight, "sway", split=10)
    assert res.skipped and len(res.skips) == 1
    e = res.skips[0]
    _check_entry(e)
    assert e["code"] == "split_not_found" and e["track"] == "upper_x" and e["region"] == "upper"
    assert e["rows"] == [2, 4] and e["frames"] == [1, 3] and e["values"] == [1, -1]
    near = re.search(r"近くで分けられる高さ: ([0-9、]+)", res.warnings[0]).group(1).split("、")
    assert e["suggestions"] == [{"sway_split": int(v)} for v in near]
    assert e["rows_from_bottom"] == [8, 12]


def test_table_rect_records_box_and_edges(knight):
    t = _table({"head": HEAD}, {"v": {"type": "move", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "v": -1}])
    res = make_motion_from_spec(knight, t)
    assert res.skipped
    probs = res.info["problems"]
    assert len(res.skips) == len(probs)
    for e, q in zip(res.skips, probs):
        _check_entry(e)
        # 表の問題（画面で赤くするもの）と同じ中身
        assert (e["code"], e["message"], e["rows"], e["track"], e["region"]) == \
            (q["rule"], q["message"], q["rows"], q["track"], q["region"])
        assert e["message"] in res.warnings
        assert e["suggestions"] == [{"region": "head", "edges": n} for n in q.get("near", [])]
        assert e.get("pixels", []) == q.get("pixels", [])
    e = res.skips[0]
    assert e["code"] == "hidden" and e["box"] == {k: v for k, v in HEAD.items() if k != "type"}


def test_skipped_report_has_planned_frames(knight):
    res = make_motion(knight, "sway", split=10, durations=[100, 200, 300, 400])
    d = skipped_info(res)
    assert d["image"] is None and d["made"] is False and d["version"] == SHEET_VERSION
    assert d["frame_count"] == 4 and [f["duration_ms"] for f in d["frames"]] == [100, 200, 300, 400]
    assert d["total_duration_ms"] == 1000 and d["skipped"] == res.skips
    assert all(0 <= f < d["frame_count"] for e in d["skipped"] for f in e["frames"])


def test_warnings_and_info_are_unchanged_by_the_record(knight):
    """記録は足すだけ: 警告・info・フレームは同じ（MotionResult の新しい項目は skips と planned_ms だけ）。"""
    res = make_motion(knight, "sway", split=10)
    assert "skips" not in res.info and "skipped" not in res.info
    assert len(res.frames) == 1 and res.durations == [sum(res.planned_ms)]


# ---- CLI と画面 ----

@pytest.fixture(scope="module")
def slime_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "slime.png"
    Image.fromarray(np.kron(make_slime(), np.ones((8, 8, 1), np.uint8)), "RGBA").save(path)   # 1 ドット = 8 画素
    return path


def test_cli_writes_skipped_json_instead_of_sheet(tmp_path, slime_png, capsys):
    dst = tmp_path / "o" / "slime_sprite.png"
    code = cli.main([str(slime_png), "-o", str(dst), "--dot-size", "8", "--motion", "nod"])
    out = capsys.readouterr()
    assert code == 0 and "うなずくを付けませんでした" in out.err
    names = sorted(p.name for p in dst.parent.iterdir())
    assert names == ["slime_sprite.png", "slime_sprite_nod_skipped.json"]   # シートは書かない
    assert "slime_sprite_nod_skipped.json（作らなかった理由 1 件）" in out.out
    d = json.loads(_text(dst.parent / "slime_sprite_nod_skipped.json"))
    assert d["image"] is None and d["skipped"][0]["code"] == "no_neck_found"
    # 画面: 同じ変換の結果から作ると、同じ中身
    res = json.loads(motion_for_web(dst.read_bytes(), json.dumps({"motion": "nod"})))
    assert res["ok"] and res["skipped"] and "sheet_json" not in res
    assert res["skipped_json"] == _text(dst.parent / "slime_sprite_nod_skipped.json")
    assert res["skipped_records"] == d["skipped"]


def test_cli_quiet_still_writes_skipped_json(tmp_path, slime_png, capsys):
    dst = tmp_path / "q" / "s.png"
    assert cli.main([str(slime_png), "-o", str(dst), "-q", "--dot-size", "8", "--motion", "nod"]) == 0
    assert (dst.parent / "s_nod_skipped.json").exists() and capsys.readouterr().out == ""


def test_cli_table_skipped_json_equals_screen(tmp_path, knight):
    src = tmp_path / "knight.png"
    Image.fromarray(np.kron(knight, np.ones((8, 8, 1), np.uint8)), "RGBA").save(src)
    spec = tmp_path / "rect.motion.json"
    t = _table({"head": HEAD}, {"v": {"type": "move", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "v": -1}],
               name="rect_up")
    spec.write_text(json.dumps(t), encoding="utf-8")
    dst = tmp_path / "o" / "k.png"
    assert cli.main([str(src), "-o", str(dst), "-q", "--dot-size", "8", "--no-simplify", "--cleanup", "off",
                     "--outline", "off", "--motion-spec", str(spec)]) == 0
    assert np.array_equal(np.asarray(Image.open(dst).convert("RGBA")), knight)
    path = dst.parent / "k_rect_up_skipped.json"
    assert sorted(p.name for p in dst.parent.iterdir()) == ["k.png", path.name]
    res = json.loads(motion_for_web(dst.read_bytes(), json.dumps({"spec": t})))
    assert res["skipped"] and res["skipped_json"] == _text(path)
    assert [e["code"] for e in json.loads(res["skipped_json"])["skipped"]] == [q["rule"] for q in res["problems"]]


def _palette_case(tmp_path):
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    src = tmp_path / "hero.png"
    Image.fromarray(img, "RGBA").save(src)
    pal = tmp_path / "p.hex"
    colors = ["000000", "ffffff", "c03030", "3060c0"]
    pal.write_text("\n".join(colors) + "\n", encoding="utf-8")
    return src, pal, [[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in colors]


def test_palette_warnings_are_recorded_in_sheet_json(tmp_path, capsys):
    src, pal, colors = _palette_case(tmp_path)
    dst = tmp_path / "o" / "hero_sprite.png"
    assert cli.main([str(src), "-o", str(dst), "-q", "--palette", str(pal), "--motion", "float"]) == 0
    err = capsys.readouterr().err
    js = dst.parent / "hero_sprite_float_sheet.json"
    info = json.loads(_text(js))
    pal_entries = [e for e in info["skipped"] if e["stage"] == "palette"]
    assert pal_entries and len(pal_entries) == err.count("警告（hero.png）")
    for e in pal_entries:
        _check_entry(e)
        assert e["code"] in ("palette_colors_merged", "palette_color_far") and e["message"] in err
        assert e["result"] == ("merged" if e["code"] == "palette_colors_merged" else "nearest")
    # 変換の結果の記録と同じ（並びも今の警告と同じ）
    res = convert_sprite(np.asarray(Image.open(src).convert("RGBA")),
                         SpriteSettings(palette=np.array(colors, np.uint8)))
    assert [e["message"] for e in res.skipped] == res.warnings and res.skipped == pal_entries
    # 画面: 変換の結果の "skipped" を動きに渡すと、CLI と同じバイトの JSON
    conv = json.loads(convert_for_web(src.read_bytes(), json.dumps({"palette": colors})))
    assert conv["ok"] and conv["skipped"] == pal_entries
    png = base64.b64decode(conv["png_base64"])
    assert png == dst.read_bytes()
    web = json.loads(motion_for_web(png, json.dumps({"motion": "float", "image_name": "hero_sprite_float_sheet.png",
                                                     "convert_skipped": conv["skipped"]})))
    assert web["ok"] and web["sheet_json"] == _text(js)


def test_convert_skipped_must_be_records():
    png = encode_png(make_pixel_sprite())
    res = json.loads(motion_for_web(png, json.dumps({"motion": "breathe", "convert_skipped": [{"code": "x"}]})))
    assert not res["ok"] and "stage" in res["error"]


def test_without_palette_records_are_empty():
    res = convert_sprite(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], SpriteSettings())
    assert res.skipped == [] and res.warnings == []


# ---- 文書 ----

def test_codes_are_listed_in_docs():
    plan = (ROOT / "docs" / "sprite-app-plan.md").read_text(encoding="utf-8")
    for code in CODES:
        assert f"`{code}`" in plan, code
    assert "skipped_json" in skipped_json.__name__
