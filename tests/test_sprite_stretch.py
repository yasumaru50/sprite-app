"""伸ばす（動きの表の縦の変形の +）・伸び（`retrocore/sprite/motion/stretch.py`、CLI `--motion stretch`）・
小さな部品を守る（`keep_small_parts`、CLI `--keep-small-parts`）のテスト。

確かめること:
- 伸ばす: 継ぎ目のドットを複製して上を上げる。新しい色を作らない・大きさと足元が同じ（上に透明の行を足し、pivot はフレームの下端）・
  隙間を作らない・輪郭を途切れさせない。縮めて伸ばすと元に戻る。作れないときは動かさずに警告し、近くで伸ばせる高さを知らせる
- 伸びの出力が、同じ内容を手で書いた表（--motion-spec）と一致する。表を書き出して読み直しても同じ
- 画面（motion_for_web・table_for_web）と CLI の出力が一致する
- 小さな部品を守る: 入にすると、小さな部品（縦 3 ドット以下・面積 24 ドット以下の同じ色の塊）の形が変わらない。切（既定）は前の版と同じ
- 見本の表（弾む＋伸び縮み）が作れる
- 今あるプリセット（うなずく）の出力が、この PR の前と同じ（呼吸・横揺れ・浮遊・弾むは test_sprite_timeline.py と test_sprite_nod.py）
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
from motion_fingerprint import nod_fingerprint
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import convert_for_web, motion_for_web, motion_spec, table_for_web
from retrocore.sprite.image import load_rgba
from retrocore.sprite.motion import (MOTIONS, check_rules, foot_center, load_spec, make_motion, make_motion_from_spec,
                                     parse_spec, preset_table, sheet_info, sheet_json)
from retrocore.sprite.motion.base import color_keys
from retrocore.sprite.motion.breathe import (find_seam, lower_above_seam, parts_kept, raise_above_seam, small_parts)
from retrocore.sprite.motion.spec import SpecError, spec_to_dict
from retrocore.sprite.motion.stretch import stretch, stretch_frames, stretch_offsets, stretch_table
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local_samples"
SAMPLE = ROOT / "apps" / "sprite" / "examples" / "bounce_stretch.motion.json"


def _cheeks() -> np.ndarray:
    """丸い顔に、縦 3・横 3 の頬（小さな部品）が左右にある絵。頬の高さで分けると、継ぎ目が頬を通りやすい。"""
    h, w = 30, 26
    out = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    body = ((yy - 15) / 13.0) ** 2 + ((xx - 12.5) / 11.0) ** 2 <= 1
    out[body] = (240, 220, 180, 255)
    edge = body & ~(np.roll(body, 1, 0) & np.roll(body, -1, 0) & np.roll(body, 1, 1) & np.roll(body, -1, 1))
    out[edge] = (60, 40, 40, 255)
    out[12:15, 8] = out[12:15, 17] = (60, 40, 40, 255)        # 目（縦 3 の線）
    out[15:18, 4:7] = out[15:18, 19:22] = (250, 150, 140, 255)   # 頬（縦 3・横 3）
    return out


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"pixel": make_pixel_sprite(), "mushroom": make_pixel_sprite(28, 32), "slime": make_slime(),
            "converted": converted, "cheeks": _cheeks()}


# ---- 並び・表 ----

def test_offsets_frames_and_table():
    assert stretch_offsets(1) == stretch_offsets() == [0, 1]
    assert stretch_offsets(2) == [0, 1, 2, 1]
    assert (stretch_frames(1), stretch_frames(2)) == (2, 4)
    assert "stretch" in MOTIONS
    t = stretch_table(2)
    assert t["name"] == "stretch" and t["regions"] == {"upper": {"type": "split", "at": "auto"}}
    assert t["tracks"] == {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}}
    assert t["frames"] == [{"ms": 500}, {"ms": 500, "upper_y": 1}, {"ms": 500, "upper_y": 2}, {"ms": 500, "upper_y": 1}]
    assert stretch_table(1, 300, 12)["frames"] == [{"ms": 300}, {"ms": 300, "upper_y": 1}]
    assert stretch_table(1, None, "40.0%")["regions"]["upper"]["at"] == "40.0%"
    assert stretch_table(1, keep_small_parts=True)["regions"]["upper"] == {"type": "split", "at": "auto",
                                                                        "keep_small_parts": True}
    parse_spec(t)
    for bad in (0, 3, True, "1"):
        with pytest.raises(ValueError, match="伸びる量"):
            stretch_table(bad)


def test_raise_is_the_reverse_of_lower(sprites):
    """同じ継ぎ目で伸ばして縮めると元に戻る（伸ばすは、縮めるで抜いてよい所と同じ所に足す）。"""
    for sp in sprites.values():
        found = find_seam(sp)
        if found is None:
            continue
        base = np.pad(sp, ((1, 0), (0, 0), (0, 0)))
        seam = found[0] + 1
        up = raise_above_seam(base, seam)
        assert np.array_equal(lower_above_seam(up, seam), base)
        # 継ぎ目より下は 1 ドットも変わらない
        for x, s in enumerate(seam):
            assert np.array_equal(up[s:, x], base[s:, x])


# ---- 伸び（プリセット） ----

def _check_stretch(base, res, amount):
    """決まり: 上に足した透明の行のほかは元の大きさ、新しい色なし、足元が同じ、部品と輪郭のつながりが同じ。"""
    assert not res.skipped, res.warnings
    top = res.info["padding_y"][0]
    assert res.info["padding_y"][1] == 0 and top >= 0
    padded = np.pad(base, ((top, 0), (0, 0), (0, 0)))
    assert np.array_equal(res.frames[0], padded)                   # 1 枚目は元の絵（上に余白を足しただけ）
    assert check_rules(padded, res.frames) == []
    for f, k in zip(res.frames, stretch_offsets(amount)):
        assert f.shape == padded.shape
        assert foot_center(f) == foot_center(padded)               # 足元（pivot）は動かない
        rows = np.flatnonzero((f[..., 3] > 0).any(axis=1))
        rows0 = np.flatnonzero((padded[..., 3] > 0).any(axis=1))
        assert rows[0] == rows0[0] - k                             # 上の端がちょうど k ドット上がる
        assert set(np.unique(color_keys(f)).tolist()) <= set(np.unique(color_keys(base)).tolist())
    info = sheet_info(res, "x.png")
    h = res.frames[0].shape[0]
    assert info["pivot"] == {"x": base.shape[1] / 2, "y": h}       # pivot はフレームの下端（= 元の足元の下）
    assert info["padding"] == {"left": 0, "right": 0, "top": top, "bottom": 0}
    assert all("offset_y" not in f for f in info["frames"])        # 体全体は動かないので offset_y は書かない


@pytest.mark.parametrize("name", ["pixel", "mushroom", "slime", "converted"])
@pytest.mark.parametrize("amount", [1, 2])
def test_stretch_keeps_rules(sprites, name, amount):
    base = sprites[name]
    res = stretch(base, amount)
    _check_stretch(base, res, amount)
    assert res.durations == [500] * stretch_frames(amount)
    assert len(res.info["seams"]) == amount


def test_padding_is_only_what_is_needed(sprites):
    """上に透明の行がもともとあれば、その分は足さない。"""
    sp = sprites["mushroom"]
    room = np.pad(sp, ((3, 0), (0, 0), (0, 0)))
    res = stretch(room, 2)
    assert res.info["padding_y"] == [0, 0] and res.size == (room.shape[1], room.shape[0])
    assert json.loads(sheet_json(res, "x.png"))["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}


def test_stretch_with_split_number_and_percent(sprites):
    base = sprites["mushroom"]
    res = stretch(base, split=12)
    _check_stretch(base, res, 1)
    lo, hi = res.info["seam_from_bottom"][0]
    assert 10 <= lo <= hi <= 14
    res = stretch(base, split="40.0%")
    assert not res.skipped and res.spec["regions"]["upper"]["at"] == "40.0%"


def test_stretch_failure_warns_with_nearby_heights(sprites):
    """伸ばせない高さ（足元の中）は動かさずに警告し、近くで伸ばせる高さを知らせる（画面の「○○ で試す」）。"""
    base = sprites["mushroom"]
    res = stretch(base, split=3)
    assert res.skipped and len(res.frames) == 1 and np.array_equal(res.frames[0], base)
    assert res.warnings[0].startswith("伸びを付けませんでした: 下から 3 ドット目では分けられません")
    # 1 ドットの横線が全部の列を横切る絵: どこでも伸ばせない
    band = base.copy()
    rows = np.flatnonzero((band[..., 3] > 0).any(axis=1))
    for y in range(rows[0], rows[-1] + 1):
        if y % 2 == 0:
            band[y, :] = (10, 10, 10, 255)
    res = stretch(band)
    assert res.skipped and "伸ばせる高さが見つかりませんでした" in res.warnings[0]
    with pytest.raises(ValueError, match="画像の範囲の外"):
        stretch(base, split=base.shape[0] + 1)


def test_stretch_nearby_hint_is_workable():
    """近くで伸ばせる高さとして知らせた数は、その数で実際に作れる。"""
    sp = _cheeks()
    for split in range(1, sp.shape[0] + 1):
        res = stretch(sp, split=split)
        if not res.skipped or "近くで分けられる高さ: " not in res.warnings[0]:
            continue
        near = [int(v) for v in res.warnings[0].split("近くで分けられる高さ: ")[1].split("、")]
        assert near and all(not stretch(sp, split=v).skipped for v in near)


@pytest.mark.parametrize("kwargs, words", [
    ({"amount": 3}, "1〜2"), ({"frames": 4}, "量で決まります"), ({"durations": [500, 500, 500]}, "表示時間の数（3）"),
    ({"durations": 0}, "1 ミリ秒以上"), ({"split": "abc"}, "分ける高さ"), ({"keep_small_parts": 1}, "true か false"),
])
def test_stretch_rejects_bad_settings(sprites, kwargs, words):
    with pytest.raises(ValueError, match=words):
        stretch(sprites["mushroom"], **kwargs)


def test_stretch_is_deterministic(sprites):
    a, b = stretch(sprites["converted"], 2), stretch(sprites["converted"], 2)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


@pytest.mark.parametrize("name", ["pixel", "mushroom", "converted"])
@pytest.mark.parametrize("options", [{}, {"amount": 2}, {"split": 12}, {"amount": 2, "durations": [400, 100, 200, 100]},
                                     {"keep_small_parts": True}])
def test_stretch_equals_handwritten_table(sprites, name, options):
    """伸びの出力が、同じ内容を手で書いた表から作ったものと同じ（PNG の画素・JSON は regions_used を除いてバイト単位）。"""
    base = sprites[name]
    amount = options.get("amount", 1)
    durations = options.get("durations", 500)
    rows = [{"ms": d} for d in ([durations] * stretch_frames(amount) if isinstance(durations, int) else durations)]
    for r, k in zip(rows, stretch_offsets(amount)):
        if k:
            r["upper_y"] = k
    region = {"type": "split", "at": options.get("split", "auto")}
    if options.get("keep_small_parts"):
        region["keep_small_parts"] = True
    hand = {"format": "sprite-motion", "version": 1, "name": "stretch", "loop": True, "regions": {"upper": region},
            "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}}, "frames": rows}
    preset = make_motion(base, "stretch", **options)
    table = make_motion_from_spec(base, hand)
    assert preset.skipped == table.skipped
    if preset.skipped:
        assert table.warnings
        return
    assert all(np.array_equal(a, b) for a, b in zip(preset.frames, table.frames))
    assert preset.durations == table.durations
    pj = json.loads(sheet_json(preset, "x.png"))
    tj = json.loads(sheet_json(table, "x.png"))
    tj.pop("regions_used", None)
    assert pj == tj
    assert preset.spec == table.spec                                # 埋めた表（"used" つき）も同じ


@pytest.mark.parametrize("style", ["number", "percent"])
def test_stretch_table_roundtrip(sprites, style):
    from retrocore.sprite.motion import spec_json, with_height_style
    base = sprites["mushroom"]
    for options in ({}, {"amount": 2, "split": 12}, {"keep_small_parts": True}):
        res = make_motion(base, "stretch", **options)
        again = make_motion_from_spec(base, spec_json(with_height_style(res.spec, base, style)))
        assert all(np.array_equal(a, b) for a, b in zip(res.frames, again.frames))


def test_preset_table_stretch(sprites):
    assert preset_table("stretch") == stretch_table()
    assert preset_table("stretch", amount=2, split=10, keep_small_parts=True) == stretch_table(2, None, 10, True)
    with pytest.raises(ValueError, match="量で決まります"):
        preset_table("stretch", frames=4)
    with pytest.raises(ValueError):
        preset_table("stretch", depth=1)
    with pytest.raises(ValueError, match="画像の範囲の外"):
        preset_table("stretch", split=999, rgba=sprites["mushroom"])


# ---- 表の伸ばす（+）と組み合わせ ----

def _table(frames, at="auto", keep=None, extra_tracks=None, name="t"):
    region = {"type": "split", "at": at}
    if keep is not None:
        region["keep_small_parts"] = keep
    tracks = {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}, **(extra_tracks or {})}
    return {"format": "sprite-motion", "version": 1, "name": name, "loop": True, "regions": {"upper": region},
            "tracks": tracks, "frames": frames}


def test_shrink_and_stretch_in_one_table(sprites):
    """1 つの列に − と + があってもよい。上の余白は + の分だけ、縮めるポーズにも同じ余白が付く。"""
    base = sprites["mushroom"]
    res = make_motion_from_spec(base, _table([{"ms": 100}, {"ms": 100, "upper_y": -1}, {"ms": 100, "upper_y": 2}]))
    assert not res.skipped, res.warnings
    top = res.info["padding_y"][0]
    padded = np.pad(base, ((top, 0), (0, 0), (0, 0)))
    assert check_rules(padded, res.frames) == []
    shrink = make_motion(base, "breathe", depth=1)
    assert np.array_equal(res.frames[1][top:], shrink.frames[1])     # 縮めるポーズは呼吸と同じ（上に余白を足しただけ）
    used = res.spec["regions"]["upper"]["used"]["lines"]
    assert set(used) == {"upper_y", "upper_y+"} and len(used["upper_y+"]) == 2


def test_stretch_lines_reproduce_the_same_frames(sprites):
    """"used" の線（伸ばすは「列の名前+」）を lines に書くと、同じフレームになる。"""
    base = sprites["converted"]
    res = make_motion_from_spec(base, _table([{"ms": 100}, {"ms": 100, "upper_y": 1}, {"ms": 100, "upper_y": 2}]))
    used = res.spec["regions"]["upper"]["used"]
    t = _table([{"ms": 100}, {"ms": 100, "upper_y": 1}, {"ms": 100, "upper_y": 2}], at=used["at"])
    t["regions"]["upper"]["lines"] = used["lines"]
    again = make_motion_from_spec(base, t)
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, again.frames))


def test_sample_bounce_stretch(sprites):
    """見本の表（弾む＋着地で縮む・空中で伸びる）: 上の余白は弾む高さ 2 と同じ 2 行で足りる。offset_y は弾むと同じ。"""
    spec = load_spec(SAMPLE)
    for name in ("pixel", "mushroom", "converted", "slime"):
        base = sprites[name]
        res = make_motion_from_spec(base, spec)
        assert not res.skipped, res.warnings
        info = sheet_info(res, "x.png")
        assert info["padding"] == {"left": 0, "right": 0, "top": 2, "bottom": 0}
        assert [f["offset_y"] for f in info["frames"]] == [0, -1, -2, -1]
        assert [f["duration_ms"] for f in info["frames"]] == [300, 150, 150, 150]
        assert info["pivot"] == {"x": base.shape[1] / 2, "y": base.shape[0] + 2}
        padded = np.pad(base, ((2, 0), (0, 0), (0, 0)))
        assert check_rules(padded, res.frames, shifts=[0, -1, -2, -1]) != []   # 形は変わる（伸び縮み）が
        for f, dy in zip(res.frames, [0, -1, -2, -1]):                         # 足元はずれの分だけ動く
            fx, fy = foot_center(f)
            assert (fx, fy) == (foot_center(padded)[0], foot_center(padded)[1] + dy)
    bounce = make_motion(sprites["mushroom"], "bounce")
    res = make_motion_from_spec(sprites["mushroom"], spec)
    assert np.array_equal(res.frames[2], bounce.frames[2])             # 頂点（伸び縮み無し）は弾むと同じ


# ---- 小さな部品を守る ----

def test_small_parts_found():
    sp = _cheeks()
    parts = small_parts(sp)
    cheek = [(c, ys, xs) for c, ys, xs in parts if len(ys) == 9]
    assert len(cheek) == 2 and all(ys.max() - ys.min() == 2 for _, ys, _ in cheek)
    assert parts_kept(sp, sp, parts)


@pytest.mark.parametrize("motion, kw, off_changes", [
    ("breathe", {"split": 14}, True), ("breathe", {"split": 14, "frames": 4, "depth": 2}, True),
    ("nod", {"split": 14, "depth": 2}, True),
    ("stretch", {"split": 14}, True), ("stretch", {"split": 14, "amount": 2}, True)])
def test_keep_small_parts_keeps_the_cheeks(motion, kw, off_changes):
    """切だと継ぎ目が頬を通って頬の長さが変わる。入だと頬は形のまま（全部のドットが同じ量だけ動く）。"""
    sp = _cheeks()
    off = make_motion(sp, motion, **kw)
    on = make_motion(sp, motion, keep_small_parts=True, **kw)
    assert not off.skipped and not on.skipped, (off.warnings, on.warnings)
    top = (on.info.get("padding_y") or [0, 0])[0]
    padded = np.pad(sp, ((top, 0), (0, 0), (0, 0)))
    assert all(parts_kept(padded, f) for f in on.frames)
    assert all(parts_kept(padded, f) for f in off.frames) is not off_changes   # 切だと頬が変わるか
    assert check_rules(padded, on.frames) == []
    assert on.spec["regions"]["upper"]["keep_small_parts"] is True
    assert "keep_small_parts" not in off.spec["regions"]["upper"]


def test_keep_small_parts_auto_and_failure(sprites):
    for name in ("pixel", "mushroom", "converted"):
        base = sprites[name]
        res = make_motion(base, "breathe", keep_small_parts=True)
        if not res.skipped:
            assert all(parts_kept(base, f) for f in res.frames)
    # 足元の中は守っても守らなくても作れない
    res = make_motion(sprites["mushroom"], "breathe", split=3, keep_small_parts=True)
    assert res.skipped


def test_keep_small_parts_in_spec():
    t = _table([{"ms": 100}, {"ms": 100, "upper_y": -1}], keep=True)
    assert spec_to_dict(parse_spec(t))["regions"]["upper"]["keep_small_parts"] is True
    t = _table([{"ms": 100}, {"ms": 100, "upper_y": -1}], keep=False)
    assert "keep_small_parts" not in spec_to_dict(parse_spec(t))["regions"]["upper"]   # 切は書かない
    with pytest.raises(SpecError, match="keep_small_parts は true か false"):
        parse_spec(_table([{"ms": 100}], keep="yes"))


@pytest.mark.parametrize("motion", ["sway", "float", "bounce"])
def test_keep_small_parts_only_for_deform_motions(sprites, motion):
    with pytest.raises(TypeError):
        make_motion(sprites["mushroom"], motion, keep_small_parts=True)


# ---- CLI ----

@pytest.fixture(scope="module")
def hero_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_stretch(tmp_path, hero_png, capsys):
    out = tmp_path / "o" / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "--motion", "stretch", "--stretch-amount", "2"]) == 0
    text = capsys.readouterr().out
    assert "hero_sprite_stretch_sheet.png" in text and "500/500/500/500 ミリ秒" in text and "上に 2 行の余白" in text
    info = json.loads((out.parent / "hero_sprite_stretch_sheet.json").read_text(encoding="utf-8"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert info["motion"] == "stretch" and info["frame_count"] == 4
    assert info["padding"] == {"left": 0, "right": 0, "top": 2, "bottom": 0}
    assert info["pivot"] == {"x": sprite.shape[1] / 2, "y": sprite.shape[0] + 2}
    sheet = np.asarray(Image.open(out.parent / "hero_sprite_stretch_sheet.png").convert("RGBA"))
    assert np.array_equal(sheet, np.concatenate(stretch(sprite, 2).frames, axis=1))


@pytest.mark.parametrize("extra, words", [
    (["--motion", "stretch", "--motion-frames", "4"], "伸びのフレーム数は量で決まります"),
    (["--motion", "stretch", "--stretch-amount", "3"], "1〜2"),
    (["--motion", "breathe", "--stretch-amount", "2"], "伸びのオプションです"),
    (["--motion", "stretch", "--nod-depth", "2"], "うなずくのオプションです"),
    (["--motion", "sway", "--keep-small-parts"], "--keep-small-parts は --motion sway では使えません"),
    (["--keep-small-parts"], "動きのオプションです"),
    (["--motion", "stretch", "--motion-duration", "1,2,3"], "--motion-duration の数（3）"),
])
def test_cli_rejects_bad_stretch_options(tmp_path, hero_png, capsys, extra, words):
    with pytest.raises(SystemExit):
        cli.main([str(hero_png), "-o", str(tmp_path / "s.png"), *extra])
    assert words in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--motion", "stretch"], ["--motion", "stretch", "--stretch-amount", "2"],
                                  ["--motion", "stretch", "--stretch-split", "12", "--keep-small-parts"],
                                  ["--motion", "nod", "--nod-depth", "2", "--nod-split", "10", "--keep-small-parts"],
                                  ["--motion", "breathe", "--keep-small-parts"]])
def test_cli_write_then_read_spec_gives_same_files(tmp_path, hero_png, args):
    """プリセットの表を書き出して --motion-spec で読ませると、シートの PNG は同じ（JSON は regions_used を除いて同じ）。"""
    a, b = tmp_path / "a", tmp_path / "b"
    spec = tmp_path / "t.motion.json"
    assert cli.main([str(hero_png), "-o", str(a / "s.png"), "-q", *args, "--write-motion-spec", str(spec)]) == 0
    written = json.loads(spec.read_text(encoding="utf-8"))
    assert ("keep_small_parts" in written["regions"]["upper"]) == ("--keep-small-parts" in args)
    assert cli.main([str(hero_png), "-o", str(b / "s.png"), "-q", "--motion-spec", str(spec)]) == 0
    name = written["name"]
    pa, pb = a / f"s_{name}_sheet.png", b / f"s_{name}_sheet.png"
    assert pa.read_bytes() == pb.read_bytes()
    ja = json.loads((a / f"s_{name}_sheet.json").read_text(encoding="utf-8"))
    jb = json.loads((b / f"s_{name}_sheet.json").read_text(encoding="utf-8"))
    jb.pop("regions_used", None)
    assert ja == jb


def test_cli_sample_spec(tmp_path, hero_png):
    assert cli.main([str(hero_png), "-o", str(tmp_path / "s.png"), "-q", "--motion-spec", str(SAMPLE)]) == 0
    info = json.loads((tmp_path / "s_bounce_stretch_sheet.json").read_text(encoding="utf-8"))
    assert info["padding"]["top"] == 2 and [f["offset_y"] for f in info["frames"]] == [0, -1, -2, -1]


# ---- 画面（motion_for_web・table_for_web）と CLI ----

@pytest.fixture(scope="module")
def sprite_png(hero_png) -> bytes:
    res = json.loads(convert_for_web(hero_png.read_bytes(), "{}"))
    assert res["ok"], res
    return base64.b64decode(res["png_base64"])


def _pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


WEB_CASES = [
    ("stretch", {}, []),
    ("stretch", {"amount": 2}, ["--stretch-amount", "2"]),
    ("stretch", {"split": 12}, ["--stretch-split", "12"]),
    ("stretch", {"split": "40.0%", "durations": 300}, ["--stretch-split", "40.0%", "--motion-duration", "300"]),
    ("stretch", {"amount": 2, "keep_small_parts": True}, ["--stretch-amount", "2", "--keep-small-parts"]),
    ("breathe", {"keep_small_parts": True}, ["--keep-small-parts"]),
    ("nod", {"depth": 2, "split": 10, "keep_small_parts": True}, ["--nod-depth", "2", "--nod-split", "10",
                                                                  "--keep-small-parts"]),
]


@pytest.mark.parametrize("motion, options, args", WEB_CASES)
def test_web_equals_cli(hero_png, sprite_png, tmp_path, motion, options, args):
    """画面と同じ呼び方で作ったシートの PNG と JSON が、同じ設定の CLI の出力とバイト単位で同じ。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(dst), "-q", "--motion", motion, *args]) == 0
    assert dst.read_bytes() == sprite_png
    sheet = tmp_path / f"hero_sprite_{motion}_sheet.png"
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": motion, **options, "image_name": sheet.name})))
    assert res["ok"] and res["skipped"] is False and res["warnings"] == [], res
    png = base64.b64decode(res["sheet_png_base64"])
    assert png == sheet.read_bytes()
    assert np.array_equal(_pixels(png), _pixels(sheet.read_bytes()))
    assert res["sheet_json"].replace("\n", os.linesep).encode("utf-8") == sheet.with_suffix(".json").read_bytes()
    if motion == "stretch":
        assert res["padding_y"][0] == json.loads(res["sheet_json"])["padding"]["top"] and res["seam_from_bottom"]


@pytest.mark.parametrize("options, words", [
    ({"amount": 3}, "1〜2"), ({"frames": 4}, "量で決まります"), ({"durations": [500, 500, 500]}, "表示時間の数（3）"),
    ({"depth": 1}, "depth は stretch の設定ではありません"), ({"keep_small_parts": "yes"}, "true か false"),
    ({"split": 999}, "画像の範囲の外"),
])
def test_web_stretch_bad_settings(sprite_png, options, words):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "stretch", **options})))
    assert res["ok"] is False and words in res["error"], res


def test_web_keep_only_for_deform_motions(sprite_png):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "sway", "keep_small_parts": True})))
    assert res["ok"] is False and "keep_small_parts は sway の設定ではありません" in res["error"]
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "sway", "keep_small_parts": False})))
    assert res["ok"] is True


def test_web_skipped_stretch_gives_hint(sprite_png):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "stretch", "split": 3})))
    assert res["ok"] and res["skipped"] and res["warnings"][0].startswith("伸びを付けませんでした")


@pytest.mark.parametrize("options, args", [({}, []), ({"amount": 2, "keep_small_parts": True},
                                                      ["--stretch-amount", "2", "--keep-small-parts"]),
                                           ({"split": 12}, ["--stretch-split", "12"])])
def test_web_table_of_stretch_equals_cli_written_table(hero_png, sprite_png, tmp_path, options, args):
    """「表で編集」で開く伸びの表が、CLI の --write-motion-spec の表から "used" を除いたものと同じ。
    その表から作ったシートの PNG はプリセットと同じ。"""
    spec = tmp_path / "s.motion.json"
    assert cli.main([str(hero_png), "-o", str(tmp_path / "s.png"), "-q", "--motion", "stretch", *args,
                     "--write-motion-spec", str(spec)]) == 0
    written = json.loads(spec.read_text(encoding="utf-8"))
    written["regions"]["upper"].pop("used")
    got = json.loads(table_for_web(json.dumps({"preset": {"motion": "stretch", **options}}), sprite_png))
    assert got["ok"] and got["spec"] == written
    preset = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "stretch", **options, "image_name": "x.png"})))
    table = json.loads(motion_for_web(sprite_png, json.dumps({"spec": got["spec"], "image_name": "x.png"})))
    assert table["ok"] and not table["skipped"]
    assert table["sheet_png_base64"] == preset["sheet_png_base64"]


def test_web_table_reads_sample(sprite_png):
    got = json.loads(table_for_web(json.dumps({"text": SAMPLE.read_text(encoding="utf-8")})))
    assert got["ok"] and got["spec"]["name"] == "bounce_stretch"
    res = json.loads(motion_for_web(sprite_png, json.dumps({"spec": got["spec"]})))
    assert res["ok"] and not res["skipped"] and res["padding_y"] == [2, 0] and res["offset_y"] == [0, -1, -2, -1]


def test_motion_spec_has_stretch_and_keep():
    s = motion_spec()
    assert s["stretch"]["amount"] == {"default": 1, "min": 1, "max": 2}
    assert s["stretch"]["offsets"] == {"1": [0, 1], "2": [0, 1, 2, 1]}
    assert s["stretch"]["duration"]["default"] == 500 and s["stretch"]["split"] == {"default": "auto", "range": 2}
    assert s["keep_small_parts"] == {"motions": ["breathe", "nod", "stretch"], "default": False}


def test_web_html_has_stretch_ui():
    html = (ROOT / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    assert 'motion: "stretch"' in html and 'id="keep-small-parts"' in html and "小さな部品を守る" in html
    assert "keep_small_parts" in html and "updateStretchFrames" in html


# ---- うなずくの出力が、この PR の前と同じ ----

# この PR の前（PR #31 の main）で `tests/motion_fingerprint.py` の `nod_fingerprint()` を計算したもの。
# ai03〜ai05 は local_samples/ にあるときだけ比べる
NOD_BEFORE = {
    "cli|n": "1dc97ec4d3e67eec", "cli|n_d2": "21789ea4c086aeac", "cli|n_p": "d339692a46d2fe15",
    "cli|n_s10": "0dd7ae75c90da1cb", "records|ai03": "22b073d5dfab363d", "records|ai04": "26cbd1b5e03bb9d0",
    "records|ai05": "f3e690f8a6745626", "records|converted": "ad5e227aabd19956", "records|mushroom": "69ff18e5ce8b3902",
    "records|pixel": "bbd1831c714ef126", "records|slime": "985a7b6a03d706d4", "web": "f57a24c1105cbe58",
}


def test_nod_is_unchanged(tmp_path, sprites):
    """うなずくの出力（make_motion・CLI のファイルと表示・画面の結果）が、この PR の前と 1 バイトも変わらない（PNG は画素）。"""
    sp = {k: sprites[k] for k in ("mushroom", "slime", "pixel", "converted")}
    for name in ("ai03", "ai04", "ai05"):
        if (LOCAL / f"{name}.png").exists():
            sp[name] = convert_sprite(load_rgba(LOCAL / f"{name}.png"), SpriteSettings(keep_large=0)).image
    got = nod_fingerprint(tmp_path, sp, enlarge_like_ai(make_pixel_sprite(), seed=3)[0])
    assert got == {k: v for k, v in NOD_BEFORE.items() if k in got}


# ---- 手持ちの ai03〜ai05（local_samples/ にあるときだけ） ----

def _ai(name):
    path = LOCAL / f"{name}.png"
    if not path.exists():
        pytest.skip(f"{path} がありません（手持ちの AI 画像で確かめるときだけ。preview/ai-check の ai-input/ にある）")
    return convert_sprite(load_rgba(path), SpriteSettings()).image


def test_stretch_on_ai_images():
    """ai03〜ai05 は自動で量 1・2 とも伸ばせる。ai04 の頬（縦 3・横 3）は、小さな部品を守ると長さが変わらない。"""
    for name in ("ai03", "ai04", "ai05"):
        base = _ai(name)
        for amount in (1, 2):
            _check_stretch(base, stretch(base, amount), amount)
    ai04 = _ai("ai04")
    cheek = [p for p in small_parts(ai04) if len(p[1]) == 9]
    assert len(cheek) == 2
    for motion, kw in (("breathe", {}), ("stretch", {}), ("nod", {"depth": 2, "split": 19})):   # 切だと頬が変わる
        res = make_motion(ai04, motion, **kw)
        top = (res.info.get("padding_y") or [0, 0])[0]
        padded = np.pad(ai04, ((top, 0), (0, 0), (0, 0)))
        assert not res.skipped and not all(parts_kept(padded, f, cheek) for f in res.frames)
    for motion, kw in (("breathe", {}), ("stretch", {}), ("nod", {"depth": 2, "split": 20})):
        res = make_motion(ai04, motion, keep_small_parts=True, **kw)
        top = (res.info.get("padding_y") or [0, 0])[0]
        padded = np.pad(ai04, ((top, 0), (0, 0), (0, 0)))
        assert not res.skipped and all(parts_kept(padded, f, cheek) for f in res.frames)
