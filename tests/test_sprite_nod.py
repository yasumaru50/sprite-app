"""うなずく（`retrocore/sprite/motion/nod.py`、CLI `--motion nod`）のテスト。

うなずくは新しい仕組みを作らず、呼吸と同じ範囲 `upper` と縦の変形の列 `upper_y` を埋めるプリセット。確かめること:
- 首の自動の探し方（くびれのある形で見つかる・スライムでは見つからない・同じくらいの候補が 2 つなら決めない）
- 決まり（新しい色を作らない・大きさと足元が同じ・隙間を作らない・輪郭を途切れさせない・迷ったら動かさない）を守る
- 出力が、同じ内容を手で書いた表（--motion-spec）と一致する
- 画面（motion_for_web・table_for_web）と CLI の出力が一致する
- 今あるプリセット（呼吸・横揺れ・浮遊・弾む）の出力が、この PR の前と 1 バイトも変わらない
"""

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import convert_for_web, motion_for_web, motion_spec, table_for_web
from retrocore.sprite.image import load_rgba
from retrocore.sprite.motion import (MOTIONS, check_rules, foot_center, make_motion, make_motion_from_spec, parse_spec,
                                     preset_table, sheet_info, sheet_json, spec_json, with_height_style)
from retrocore.sprite.motion.base import color_keys, split_limits
from retrocore.sprite.motion.nod import (DOWN_DURATION, NECK_SLACK, UP_DURATION, default_durations, find_neck, nod,
                                         nod_frames, nod_offsets, nod_table)
from retrocore.sprite.motion.spec import SpecError, percent_to_height
from legacy_output import legacy_bytes, legacy_files, legacy_sheet_text, legacy_stdout, legacy_web
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local_samples"


def _shape(rows: list[tuple[int, int]], w: int = 24) -> np.ndarray:
    """行ごとの (幅, 色の番号) から、左右対称の形を作る（上の行から）。輪郭は付けない（首の探し方は幅だけを見る）。"""
    cols = [(200, 120, 90, 255), (90, 160, 210, 255), (240, 220, 180, 255)]
    out = np.zeros((len(rows), w, 4), np.uint8)
    for y, (width, c) in enumerate(rows):
        x0 = (w - width) // 2
        out[y, x0:x0 + width] = cols[c]
    return out


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    # 頭と体の間に、同じくらい細い所が離れて 2 つある形（首がはっきりしない）
    twin = _shape([(10, 0)] * 2 + [(20, 0)] * 8 + [(10, 1)] + [(20, 1)] * 8 + [(10, 2)] + [(20, 2)] * 8 + [(14, 2)] * 3)
    return {"pixel": make_pixel_sprite(), "mushroom": make_pixel_sprite(28, 32), "slime": make_slime(),
            "converted": converted, "twin": twin}


# ---- 並び・既定値 ----

def test_offsets_frames_and_durations():
    assert nod_offsets(1) == nod_offsets() == [0, 1]
    assert nod_offsets(2) == [0, 1, 2, 1]
    assert (nod_frames(1), nod_frames(2)) == (2, 4)
    assert (UP_DURATION, DOWN_DURATION) == (600, 200)
    assert default_durations(1) == [600, 200]                     # 上を向いているコマを長く、下がったコマを短く
    assert default_durations(2) == [600, 200, 200, 200]
    assert "nod" in MOTIONS


def test_table_is_upper_y_only():
    """表は呼吸と同じ範囲 upper と列 upper_y だけ（新しい仕組みは無い）。値が 0 の行には書かない。"""
    t = nod_table(2, None, 14)
    assert t["name"] == "nod" and t["regions"] == {"upper": {"type": "split", "at": 14}}
    assert t["tracks"] == {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}}
    assert t["frames"] == [{"ms": 600}, {"ms": 200, "upper_y": -1}, {"ms": 200, "upper_y": -2}, {"ms": 200, "upper_y": -1}]
    assert nod_table(1, 300, "31.0%")["regions"]["upper"]["at"] == "31.0%"
    assert nod_table(1, 300, 9)["frames"] == [{"ms": 300}, {"ms": 300, "upper_y": -1}]
    parse_spec(t)   # 今の表の形のまま読める
    with pytest.raises(ValueError):
        nod_table(1, None, "auto")


# ---- 首の探し方 ----

def test_find_neck_on_chibi(sprites):
    """頭が大きく、首で細くなり、その下で体が広がる形では、いちばん細い行（首）が見つかる。"""
    got = find_neck(sprites["pixel"])
    assert got.height == 9 and got.width == 10 and got.head_width == 20
    assert got.band[0] <= 9 <= got.band[1]
    assert find_neck(sprites["mushroom"]).height == 13
    assert find_neck(sprites["converted"]).height == 13


def test_find_neck_none_on_slime(sprites):
    """足も首も無いスライムでは見つからない（下ほど広いだけで、くびれが無い）。"""
    got = find_neck(sprites["slime"])
    assert got.height is None and "見つかりませんでした" in got.reason and got.candidates == []


def test_find_neck_ambiguous(sprites):
    """同じくらい細い所が離れて 2 つあれば、どちらとも決めない。"""
    got = find_neck(sprites["twin"])
    assert got.height is None and "2 か所" in got.reason and len(got.candidates) >= 2


def test_find_neck_empty_or_tiny():
    assert find_neck(np.zeros((6, 6, 4), np.uint8)).height is None
    assert find_neck(np.zeros((40, 6, 4), np.uint8)).height is None


# ---- 決まり ----

def _check_nod(base: np.ndarray, res, depth: int) -> None:
    assert not res.skipped, res.warnings
    assert np.array_equal(res.frames[0], base)                       # 1 枚目は元のまま
    assert check_rules(base, res.frames) == []
    colors = set(np.unique(color_keys(base)).tolist())
    h = base.shape[0]
    low = min(min(s) for s in res.info["seams"])                     # 継ぎ目（元の絵の行）のいちばん上
    for k, f in zip(nod_offsets(depth), res.frames):
        assert f.shape == base.shape                                 # 大きさが同じ
        assert set(np.unique(color_keys(f)).tolist()) <= colors      # 新しい色が無い
        assert foot_center(f) == foot_center(base)                   # 足元が動かない
        bottom = max(max(s) for s in res.info["seams"])
        assert np.array_equal(f[bottom + 1:], base[bottom + 1:])     # 継ぎ目より下（体）は 1 ドットも変わらない
        if k:
            assert not np.array_equal(f[:low + 1], base[:low + 1])   # 頭は動いた
    assert res.info["seam_from_bottom"] == [[h - max(s), h - min(s)] for s in res.info["seams"]]


@pytest.mark.parametrize("name", ["pixel", "mushroom", "converted"])
@pytest.mark.parametrize("depth", [1, 2])
def test_nod_keeps_rules(sprites, name, depth):
    base = sprites[name]
    res = nod(base, depth=depth)
    _check_nod(base, res, depth)
    assert res.durations == default_durations(depth) and len(res.frames) == nod_frames(depth)
    neck = res.info["neck"]
    assert neck["from_bottom"] == find_neck(base).height
    assert abs(res.spec["regions"]["upper"]["at"] - neck["from_bottom"]) <= NECK_SLACK
    assert neck["used"] == res.spec["regions"]["upper"]["at"]
    for a, b in res.info["seam_from_bottom"]:                        # 継ぎ目は首の近く（分ける高さの上下 2 ドット）
        assert neck["used"] - 2 <= a <= b <= neck["used"] + 2


def test_nod_skips_without_neck_and_suggests_heights(sprites):
    """首が見つからなければ付けずに警告し、首を探した範囲で実際に作れた高さを「試せる高さ」に添える。"""
    base = sprites["slime"]
    res = nod(base)
    assert res.skipped and len(res.frames) == 1 and res.spec is None
    w = res.warnings[0]
    assert w.startswith("うなずくを付けませんでした: 自動では首の高さを決められませんでした") and "試せる高さ: " in w
    hints = [int(v) for v in w.split("試せる高さ: ")[1].split("、")]
    assert 1 <= len(hints) <= 3 and hints == sorted(hints)
    band = res.info["neck"]["band"]
    for s in hints:                                                  # 添えた高さは本当に作れる
        assert band[0] <= s <= band[1]
        got = nod(base, split=s)
        assert not got.skipped, got.warnings
        _check_nod(base, got, 1)


def test_nod_ambiguous_is_skipped(sprites):
    res = nod(sprites["twin"])
    assert res.skipped and "2 か所" in res.warnings[0]


def test_nod_moves_to_a_nearby_height_when_the_neck_row_cannot_be_cut(sprites):
    """首の高さちょうどでは抜けない（まわりが横向きの 1 ドットの線の重なり）なら、上下 2 ドットまでずらした高さで作る。"""
    base = sprites["pixel"].copy()
    h = base.shape[0]
    a, b = base[h - 9, 10].copy(), base[h - 12, 10].copy()          # 首の色と頭の色（どちらも元の絵にある色）
    for r in range(7, 12):                                           # 下から 7〜11 行目を 1 行ごとに色を変える
        row = h - r
        base[row, base[row, :, 3] > 0] = a if r % 2 else b
    assert find_neck(base).height == 9
    assert nod(base, split=9).skipped                                # 首の高さちょうどでは作れない
    res = nod(base)
    _check_nod(base, res, 1)
    at = res.spec["regions"]["upper"]["at"]
    assert res.info["neck"]["from_bottom"] == 9 and at != 9 and abs(at - 9) <= NECK_SLACK
    assert res.info["neck"]["used"] == at
    # 試す順は 9, 10, 8, 11, 7（頭の側が先）。使った高さより先に試した高さは、どれも作れない
    order = [9, 10, 8, 11, 7]
    assert all(nod(base, split=s).skipped for s in order[:order.index(at)])


def test_nod_with_split_number_and_percent(sprites):
    base = sprites["pixel"]
    a = nod(base, split=10)
    _check_nod(base, a, 1)
    assert a.spec["regions"]["upper"]["at"] == 10 and "neck" not in a.info and a.info["split"] == 10
    pct = with_height_style({"regions": {"upper": {"at": 10}}}, base, "percent")["regions"]["upper"]["at"]
    b = nod(base, split=pct)
    assert percent_to_height(base, pct) == 10 and b.spec["regions"]["upper"]["at"] == pct
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


def test_nod_split_out_of_reach_warns_with_nearby(sprites):
    base = sprites["pixel"]
    lim = split_limits(base)
    res = nod(base, split=lim[0] - 1)                                # 足元（動かさない行）
    assert res.skipped and "足元" in res.warnings[0]
    with pytest.raises(ValueError, match="画像の範囲の外"):
        nod(base, split=base.shape[0] + 1)


@pytest.mark.parametrize("kwargs, words", [
    ({"depth": 0}, "1〜2"), ({"depth": 3}, "1〜2"), ({"depth": True}, "1〜2"), ({"frames": 4}, "深さで決まります"),
    ({"durations": [600, 200, 200]}, "フレーム数（2）"), ({"durations": 0}, "1 ミリ秒以上"),
    ({"split": "abc"}, "割合"), ({"split": "0%"}, "割合"), ({"split": 1.5}, "整数"),
])
def test_nod_rejects_bad_settings(sprites, kwargs, words):
    with pytest.raises(ValueError, match=words):
        nod(sprites["pixel"], **kwargs)


def test_nod_durations(sprites):
    base = sprites["pixel"]
    assert nod(base, durations=250).durations == [250, 250]          # 1 つなら全部同じ（弾むと同じ）
    assert nod(base, depth=2, durations=[500, 100, 120, 140]).durations == [500, 100, 120, 140]
    assert make_motion(base, "nod", frames=2).durations == [600, 200]
    assert make_motion(base, "nod", depth=2).durations == [600, 200, 200, 200]


def test_nod_is_deterministic(sprites):
    a, b = nod(sprites["converted"], depth=2), nod(sprites["converted"], depth=2)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames)) and a.info == b.info


def test_nod_on_empty_image():
    res = nod(np.zeros((6, 6, 4), np.uint8))
    assert res.skipped and res.warnings[0].startswith("うなずくを付けませんでした: ")


# ---- 手で書いた表と一致 ----

def _same(a, b) -> bool:
    return (a.skipped == b.skipped and len(a.frames) == len(b.frames) and a.durations == b.durations
            and all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))
            and sheet_json(a, "x.png") == sheet_json(b, "x.png"))


@pytest.mark.parametrize("name", ["pixel", "mushroom", "converted"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("options", [{}, {"durations": 300}, {"split": 10}])
def test_nod_equals_handwritten_table(sprites, name, depth, options):
    """プリセットの出力（フレームの画素・シートの JSON のバイト）が、同じ内容を手で書いた表から作ったものと同じ。"""
    base = sprites[name]
    a = make_motion(base, "nod", depth=depth, **options)
    at = options.get("split", a.info.get("neck", {}).get("used"))
    ms = options.get("durations")
    rows = [600, 200] if depth == 1 else [600, 200, 200, 200]
    vals = [0, -1] if depth == 1 else [0, -1, -2, -1]
    hand = {"format": "sprite-motion", "version": 1, "name": "nod", "loop": True,
            "regions": {"upper": {"type": "split", "at": at}},
            "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
            "frames": [{"ms": ms or r, **({"upper_y": v} if v else {})} for r, v in zip(rows, vals)]}
    b = make_motion_from_spec(base, hand)
    assert _same(a, b), (a.warnings, b.warnings)
    if not a.skipped:
        assert "regions_used" not in sheet_info(b, "x.png")          # 数の高さなので、表から作っても JSON は同じ


@pytest.mark.parametrize("name", ["pixel", "converted"])
@pytest.mark.parametrize("style", ["number", "percent"])
def test_nod_table_roundtrip(sprites, name, style):
    """埋めた表（"used" つき）を JSON に書いて読み直すと、同じ出力。"""
    base = sprites[name]
    for kw in ({}, {"depth": 2}, {"split": 10}):
        a = make_motion(base, "nod", **kw)
        b = make_motion_from_spec(base, json.loads(spec_json(with_height_style(a.spec, base, style))))
        jb = sheet_info(b, "x.png")
        jb.pop("regions_used", None)
        assert not a.skipped and all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))
        assert json.dumps(jb, ensure_ascii=False, indent=2) + "\n" == sheet_json(a, "x.png")


def test_preset_table_nod(sprites):
    """preset_table() の nod: 自動は絵から首の高さを決めて数で書く（プリセットの表から "used" を除いたものと同じ）。"""
    base = sprites["pixel"]
    for kw in ({}, {"depth": 2}, {"split": 10, "durations": 300}):
        res = make_motion(base, "nod", **kw)
        spec = json.loads(json.dumps(res.spec))
        spec["regions"]["upper"].pop("used", None)
        assert preset_table("nod", rgba=base, **kw) == spec
    assert preset_table("nod", split=12) == nod_table(1, None, 12)    # 数なら絵は要らない
    with pytest.raises(ValueError, match="絵が要ります"):
        preset_table("nod")
    with pytest.raises(ValueError, match="試せる高さ"):
        preset_table("nod", rgba=sprites["slime"])
    with pytest.raises(ValueError, match="深さで決まります"):
        preset_table("nod", frames=4, split=10)
    with pytest.raises(ValueError):
        preset_table("nod", amount=1, split=10)


# ---- CLI ----

@pytest.fixture(scope="module")
def hero_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_nod(tmp_path, hero_png, capsys):
    out = tmp_path / "o" / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "--motion", "nod"]) == 0
    text = capsys.readouterr().out
    assert "hero_sprite_nod_sheet.png" in text and "600/200 ミリ秒" in text and "首を自動で下から 9 ドット目に見つけ" in text
    assert sorted(p.name for p in out.parent.iterdir()) == ["hero_sprite.png", "hero_sprite_nod_sheet.json",
                                                            "hero_sprite_nod_sheet.png"]
    info = json.loads((out.parent / "hero_sprite_nod_sheet.json").read_text(encoding="utf-8"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert info["motion"] == "nod" and info["frame_count"] == 2 and "padding" not in info
    assert [f["duration_ms"] for f in info["frames"]] == [600, 200]
    sheet = np.asarray(Image.open(out.parent / "hero_sprite_nod_sheet.png").convert("RGBA"))
    assert np.array_equal(sheet, np.concatenate(nod(sprite).frames, axis=1))


@pytest.mark.parametrize("args, frames, durations", [
    (["--nod-depth", "2"], 4, [600, 200, 200, 200]), (["--motion-duration", "300"], 2, [300, 300]),
    (["--nod-depth", "2", "--motion-duration", "700,100,150,100"], 4, [700, 100, 150, 100]),
    (["--nod-split", "10"], 2, [600, 200]), (["--nod-split", "auto", "--nod-depth", "1"], 2, [600, 200]),
])
def test_cli_nod_options(tmp_path, hero_png, args, frames, durations):
    out = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "-q", "--motion", "nod", *args]) == 0
    info = json.loads((tmp_path / "hero_sprite_nod_sheet.json").read_text(encoding="utf-8"))
    assert info["frame_count"] == frames and [f["duration_ms"] for f in info["frames"]] == durations


def test_cli_nod_split_percent(tmp_path, hero_png, capsys):
    out = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "--motion", "nod", "--nod-split", "45.0%"]) == 0
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert f"45.0%（下から {percent_to_height(sprite, '45.0%')} ドット目）で分けた" in capsys.readouterr().out


@pytest.mark.parametrize("extra, words", [
    (["--nod-depth", "2"], "動きのオプション"),
    (["--motion", "nod", "--motion-frames", "4"], "--motion-frames は --motion nod では使えません"),
    (["--motion", "nod", "--nod-depth", "3"], "1〜2"),
    (["--motion", "nod", "--nod-split", "0"], "1 以上"),
    (["--motion", "nod", "--nod-split", "abc"], "auto か"),
    (["--motion", "nod", "--nod-split", "120%"], "割合"),
    (["--motion", "nod", "--breathe-depth", "1"], "--motion nod では使えません（呼吸のオプションです）"),
    (["--motion", "nod", "--bounce-height", "1"], "--motion nod では使えません（弾むのオプションです）"),
    (["--motion", "breathe", "--nod-split", "10"], "--motion breathe では使えません（うなずくのオプションです）"),
    (["--motion", "nod", "--motion-duration", "600,200,200"], "フレーム数（2）"),
    (["--motion", "nod", "--nod-depth", "2", "--motion-duration", "600,200"], "フレーム数（4）"),
])
def test_cli_rejects_bad_nod_options(tmp_path, hero_png, capsys, extra, words):
    with pytest.raises(SystemExit):
        cli.main([str(hero_png), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_nod_without_neck_warns_and_writes_no_table(tmp_path, capsys):
    args = cli.build_parser().parse_args(["x.png", "--motion", "nod", "-q", "--write-motion-spec",
                                          str(tmp_path / "n.motion.json")])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, make_slime(), dst, dst) == 1
    err = capsys.readouterr().err
    assert "うなずくを付けませんでした" in err and "試せる高さ" in err and "n.motion.json は書きませんでした" in err
    assert not (tmp_path / "s_nod_sheet.png").exists() and not (tmp_path / "n.motion.json").exists()


@pytest.mark.parametrize("args", [["--motion", "nod"], ["--motion", "nod", "--nod-depth", "2"],
                                  ["--motion", "nod", "--nod-split", "10", "--motion-duration", "150"]])
@pytest.mark.parametrize("height", ["number", "percent"])
def test_cli_write_then_read_nod_spec_gives_same_files(tmp_path, hero_png, args, height):
    """--write-motion-spec で書いたうなずくの表を --motion-spec で読ませると、シートの PNG と JSON がバイト単位で同じ。"""
    spec = tmp_path / "n.motion.json"
    assert cli.main([str(hero_png), "-o", str(tmp_path / "a" / "s.png"), "-q", *args,
                     "--write-motion-spec", str(spec), "--spec-height", height]) == 0
    data = json.loads(spec.read_text(encoding="utf-8"))
    assert data["name"] == "nod" and list(data["tracks"]) == ["upper_y"] and "used" in data["regions"]["upper"]
    assert cli.main([str(hero_png), "-o", str(tmp_path / "b" / "s.png"), "-q", "--motion-spec", str(spec)]) == 0
    a, b = tmp_path / "a", tmp_path / "b"
    names = ["s.png", "s_nod_sheet.json", "s_nod_sheet.png"]
    assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir()) == names
    for name in names:
        if height == "percent" and name.endswith(".json"):   # 割合の表から作ると、使った高さ（regions_used）が足される
            ja = json.loads((a / name).read_text(encoding="utf-8"))
            jb = json.loads((b / name).read_text(encoding="utf-8"))
            jb.pop("regions_used")
            assert ja == jb
            continue
        assert (a / name).read_bytes() == (b / name).read_bytes()


# ---- 画面（motion_for_web・table_for_web）と CLI ----

@pytest.fixture(scope="module")
def sprite_png(hero_png) -> bytes:
    res = json.loads(convert_for_web(hero_png.read_bytes(), "{}"))
    assert res["ok"], res
    return base64.b64decode(res["png_base64"])


def _pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


WEB_CASES = [
    ({}, []),
    ({"depth": 2}, ["--nod-depth", "2"]),
    ({"split": 10}, ["--nod-split", "10"]),
    ({"split": "auto", "depth": 1}, ["--nod-split", "auto"]),
    ({"durations": 300}, ["--motion-duration", "300"]),
    ({"depth": 2, "durations": [700, 100, 150, 100]}, ["--nod-depth", "2", "--motion-duration", "700,100,150,100"]),
    ({"split": "45.0%"}, ["--nod-split", "45.0%"]),
]


@pytest.mark.parametrize("options, args", WEB_CASES)
def test_web_nod_equals_cli(hero_png, sprite_png, tmp_path, options, args):
    """画面と同じ呼び方で作ったシートの PNG と JSON が、同じ設定の CLI の出力とバイト単位で同じ（PNG は画素でも同じ）。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(dst), "-q", "--motion", "nod", *args]) == 0
    assert dst.read_bytes() == sprite_png
    sheet = tmp_path / "hero_sprite_nod_sheet.png"
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "nod", **options, "image_name": sheet.name})))
    assert res["ok"] and res["skipped"] is False and res["warnings"] == []
    png = base64.b64decode(res["sheet_png_base64"])
    assert png == sheet.read_bytes()
    assert np.array_equal(_pixels(png), _pixels(sheet.read_bytes()))
    assert res["sheet_json"].replace("\n", os.linesep).encode("utf-8") == sheet.with_suffix(".json").read_bytes()
    info = json.loads(res["sheet_json"])
    assert res["durations"] == [f["duration_ms"] for f in info["frames"]]
    assert res["seam_from_bottom"] and res["split_used"] is not None


def test_web_nod_without_neck_gives_hint_buttons(sprite_png):
    """首が見つからないときの警告は CLI と同じ言葉で、画面の「○○ で試す」のボタンになる「試せる高さ: …」を含む。"""
    slime = make_slime()
    from retrocore.sprite.convert import encode_png
    res = json.loads(motion_for_web(encode_png(slime), json.dumps({"motion": "nod"})))
    assert res["ok"] and res["skipped"] and res["neck"]["from_bottom"] is None
    assert res["warnings"] == nod(slime).warnings and "試せる高さ: " in res["warnings"][0]


@pytest.mark.parametrize("options, words", [
    ({"depth": 3}, "1〜2"), ({"depth": "1"}, "1〜2"), ({"frames": 4}, "深さで決まります"),
    ({"durations": [600, 200, 200]}, "表示時間の数（3）"), ({"durations": 0}, "1 以上"),
    ({"amount": 1}, "amount は nod の設定ではありません"), ({"height": 2}, "height は nod の設定ではありません"),
    ({"split": 0}, "分ける高さ"), ({"split": 999}, "画像の範囲の外"),
])
def test_web_nod_bad_settings(sprite_png, options, words):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "nod", **options})))
    assert res["ok"] is False and words in res["error"], res


@pytest.mark.parametrize("options, args", [({}, []), ({"depth": 2}, ["--nod-depth", "2"]),
                                           ({"split": 10, "durations": 300}, ["--nod-split", "10",
                                                                              "--motion-duration", "300"])])
def test_web_table_of_nod_equals_cli_written_table(hero_png, sprite_png, tmp_path, options, args):
    """「表で編集」で開くうなずくの表が、CLI の --write-motion-spec の表から "used" を除いたものと同じ。
    その表から作ったシートは、プリセットのシートとバイト単位で同じ（数の高さなので regions_used も付かない）。"""
    spec = tmp_path / "n.motion.json"
    assert cli.main([str(hero_png), "-o", str(tmp_path / "s.png"), "-q", "--motion", "nod", *args,
                     "--write-motion-spec", str(spec)]) == 0
    written = json.loads(spec.read_text(encoding="utf-8"))
    written["regions"]["upper"].pop("used")
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": "nod", **options}}), sprite_png))
    assert got["ok"] and got["spec"] == written
    preset = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "nod", **options, "image_name": "x.png"})))
    table = json.loads(motion_for_web(sprite_png, json.dumps({"spec": got["spec"], "image_name": "x.png"})))
    assert table["ok"] and not table["skipped"]
    assert table["sheet_png_base64"] == preset["sheet_png_base64"] and table["sheet_json"] == preset["sheet_json"]


def test_web_table_of_nod_needs_the_sprite_for_auto(sprite_png):
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": "nod"}})))
    assert got["ok"] is False and "絵が要ります" in got["error"]
    from retrocore.sprite.convert import encode_png
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": "nod"}}), encode_png(make_slime())))
    assert got["ok"] is False and "試せる高さ" in got["error"]
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": "nod", "split": 12}})))   # 数なら絵は要らない
    assert got["ok"] and got["spec"]["regions"]["upper"]["at"] == 12


def test_motion_spec_has_nod():
    n = motion_spec()["nod"]
    assert n["depth"] == {"default": 1, "min": 1, "max": 2}
    assert n["offsets"] == {"1": [0, -1], "2": [0, -1, -2, -1]}
    assert n["durations"] == {"1": [600, 200], "2": [600, 200, 200, 200]} and (n["up"], n["down"]) == (600, 200)
    assert n["split"] == {"default": "auto", "range": 2}


def test_web_html_has_nod_ui():
    html = (ROOT / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    assert 'motion: "nod"' in html and "首の高さ（分ける高さ）" in html and "試せる高さ" in html
    assert "tableForWeb(JSON.stringify(msg.request), msg.bytes" in html


# ---- 手持ちの ai03〜ai05（local_samples/ にあるときだけ） ----

def _ai(name):
    path = LOCAL / f"{name}.png"
    if not path.exists():
        pytest.skip(f"{path} がありません（手持ちの AI 画像で確かめるときだけ。preview/ai-check の ai-input/ にある）")
    return convert_sprite(load_rgba(path), SpriteSettings()).image


def test_nod_on_ai_images():
    """ai03（キノコ）・ai04（どんぐりの騎士）は首が見つかって付く。ai05（スライム）は首が無いので付けずに試せる高さを出す。"""
    ai03, ai04, ai05 = _ai("ai03"), _ai("ai04"), _ai("ai05")
    assert find_neck(ai03).height == 14 and find_neck(ai04).height == 15 and find_neck(ai05).height is None
    r = nod(ai03)
    _check_nod(ai03, r, 1)
    r = nod(ai04)                       # 首の行（襟の 1 ドットの線）では抜けないので、頭の側の 16 で分ける
    _check_nod(ai04, r, 1)
    assert r.spec["regions"]["upper"]["at"] == 16
    assert "試せる高さ: 5、9、13" in nod(ai05).warnings[0]


# ---- 今あるプリセットの出力が、この PR の前と同じ ----

def _dig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# この PR の前（PR #30 の main）で、下の `_fingerprint()` と同じ計算をしたもの（グループ → ハッシュ）。
# records|* は make_motion() の結果（フレーム・警告・表示時間・埋めた表・シートの JSON。設定の間違いはエラーの文）、
# cli|* は CLI のファイル（--write-motion-spec の表も）・終了コード・表示、web・table は画面の motion_for_web・table_for_web の結果
# （シートの PNG は画素で比べる）
BEFORE = {
    "cli|b": "c6522e2277d7222c", "cli|b_s12": "6821bf730c2edf3b", "cli|bo": "800b00649dd346b0",
    "cli|bo3": "7a49e2760ca3e4e4", "cli|f": "cefb86867114d362", "cli|s": "879d6131c4927f79",
    "records|bounce": "30f679868170b6c4", "records|breathe": "72b2bcdfad15c309", "records|float": "7b66910398c0cf69",
    "records|sway": "ede97ed4d41e326d", "table": "62ffb43804016f38", "web": "4df9e33253c44122",
}
COMBOS = {"bounce": [{}, {"height": 1}, {"height": 3}, {"durations": 150}, {"height": 1, "durations": [500, 100]},
                     {"height": 4}, {"frames": 6}],
          "breathe": [{}, {"frames": 4, "depth": 2}, {"split": 12}, {"split": 9, "durations": [600, 400]}],
          "sway": [{}, {"frames": 8, "amount": 2}, {"split": 10}],
          "float": [{}, {"frames": 8, "amount": 2}]}
CLI = {"b": ["--motion", "breathe"], "s": ["--motion", "sway"], "f": ["--motion", "float"], "bo": ["--motion", "bounce"],
       "bo3": ["--motion", "bounce", "--bounce-height", "3"], "b_s12": ["--motion", "breathe", "--breathe-split", "12"]}
WEB = [{}, {"split": 12, "frames": 4, "depth": 2}, {"motion": "sway"}, {"motion": "float"}, {"motion": "bounce"},
       {"motion": "bounce", "height": 3, "durations": 120}, {"motion": "breathe", "split": 1},
       {"motion": "bounce", "height": 9}]


def _fingerprint(tmp: Path, sprites: dict) -> dict:
    out = {}
    for name, sp in sprites.items():
        for motion, combos in COMBOS.items():
            for kw in combos:
                key = f"{name}|{motion}|{json.dumps(kw, sort_keys=True)}"
                try:
                    r = make_motion(sp, motion, **kw)
                    out[key] = _dig(json.dumps({"s": r.skipped, "w": r.warnings, "d": r.durations, "spec": r.spec,
                                                "f": _dig(b"".join(repr(f.shape).encode() + f.tobytes() for f in r.frames)),
                                                "j": legacy_sheet_text(sheet_json(r, "x.png"))},
                                               ensure_ascii=False, sort_keys=True, default=str).encode())
                except (ValueError, TypeError) as e:
                    out[key] = str(e)
    src = tmp / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(src)
    for tag, args in CLI.items():
        o = tmp / tag
        err, so = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
            code = cli.main([str(src), "-o", str(o / "s.png"), *args, "--write-motion-spec", str(o / "t.motion.json")])
        stdout = "\n".join(line for line in so.getvalue().replace(str(tmp), "<T>").replace("\\", "/").splitlines()
                           if "秒）" not in line)
        files = {}
        for f in legacy_files(sorted(o.iterdir())):
            if f.suffix == ".png":
                a = np.asarray(Image.open(f).convert("RGBA"))
                files[f.name] = _dig(repr(a.shape).encode() + a.tobytes())
            else:
                files[f.name] = _dig(legacy_bytes(f.read_bytes().replace(b"\r\n", b"\n")))
        out[f"cli|{tag}"] = {"code": code, "files": files, "stdout": _dig(legacy_stdout(stdout).encode()), "stderr": err.getvalue()}
    png = base64.b64decode(json.loads(convert_for_web(src.read_bytes(), "{}"))["png_base64"])
    for o in WEB:
        for kind, text in (("web", motion_for_web(png, json.dumps(o))), ("table", table_for_web(json.dumps({"preset": o})))):
            d = legacy_web(json.loads(text))
            d.pop("detail", None)   # エラーの詳しい情報（Python のファイルの行番号を含む）は比べない
            if "sheet_png_base64" in d:   # PNG は画素で比べる（圧縮のバイトは zlib の版で変わるため。Windows と Linux で違う）
                a = np.asarray(Image.open(io.BytesIO(base64.b64decode(d.pop("sheet_png_base64")))).convert("RGBA"))
                d["sheet_pixels"] = _dig(repr(a.shape).encode() + a.tobytes())
            out[f"{kind}|{json.dumps(o, sort_keys=True)}"] = _dig(json.dumps(d, ensure_ascii=False, sort_keys=True).encode())
    groups: dict = {}
    for k, v in out.items():
        head = k.split("|")[0]
        g = k if head == "cli" else ("records|" + k.split("|")[1] if head in sprites else head)
        groups.setdefault(g, {})[k] = v
    return {g: _dig(json.dumps(v, ensure_ascii=False, sort_keys=True).encode()) for g, v in sorted(groups.items())}


def test_existing_presets_are_unchanged(tmp_path):
    """呼吸・横揺れ・浮遊・弾むの出力（make_motion・CLI のファイルと表示・画面の結果）が、この PR の前と 1 バイトも変わらない。"""
    sprites = {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "pixel": make_pixel_sprite(),
               "converted": convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image}
    assert _fingerprint(tmp_path, sprites) == BEFORE
