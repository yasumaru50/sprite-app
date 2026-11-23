"""弾む（`retrocore/sprite/motion/bouncing.py`、CLI `--motion bounce`）のテスト。

弾むは浮遊と同じく「体全体のずれ」（表の `root_y` の列）だけを使い、地面から上にだけ動いて戻る。
足元が上に決まった量だけ動くこと以外は、今までの決まり（新しい色を作らない・全フレームの大きさが同じ・隙間を作らない・
輪郭を途切れさせない・迷ったら動かさない）を守ることを確かめる。あわせて:
- 余白は上にだけ足し、pivot はフレームの下端のまま（JSON は浮遊と同じ形）
- --write-motion-spec で書いた表を --motion-spec で読み直すと、同じファイルになる
- 画面（motion_for_web）と CLI の出力がバイト単位で同じ（PNG は画素でも比べる）
- 浮遊・--motion-spec の CLI の出力が、この PR の前（PR #27 の main）と同じ
"""

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import convert_for_web, motion_for_web, motion_spec
from retrocore.sprite.motion import (MOTIONS, check_rules, foot_center, load_spec, make_motion, make_motion_from_spec,
                                     parse_spec, preset_table, sheet_info)
from retrocore.sprite.motion.base import color_keys, shape_signature
from retrocore.sprite.motion.bouncing import (AIR_DURATION, DEFAULT_HEIGHT, LANDING_DURATION, MOVES_FOOT, bounce,
                                              bounce_frames, bounce_offsets, bounce_table, default_durations)
from retrocore.sprite.motion.spec import SpecError
from retrocore.sprite.motion.sway import euler_numbers
from legacy_output import legacy_bytes, legacy_files
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime
from test_sprite_sway import diamond, sash

SPECS = Path(__file__).resolve().parent / "motion_specs"


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "converted": converted,
            "diamond": diamond(), "sash": sash()}


NAMES = ["mushroom", "slime", "converted", "diamond", "sash"]


def _check_bounce(base: np.ndarray, res, height: int) -> None:
    """弾むの結果が決まり（足元が上に動く例外のほかは今までどおり）を守っていることを、check_rules() とは別の書き方でも確かめる。"""
    assert not res.skipped, res.warnings
    top, bottom = res.info["padding_y"]
    assert bottom == 0                                                  # 下には動かないので、下の余白は無い
    padded = np.pad(base, ((top, 0), (0, 0), (0, 0)))
    assert np.array_equal(res.frames[0], padded)                       # 1 枚目（着地）は元のまま（上に余白を足しただけ）
    assert check_rules(padded, res.frames, shifts=res.info["offsets"]) == []
    base_colors = set(np.unique(color_keys(base)).tolist())
    fx, fy = foot_center(padded)
    assert (fx, fy) == (foot_center(base)[0], foot_center(base)[1] + top)   # 着地の足元は元の絵と同じ（上に足した分だけ下）
    opaque = int((base[..., 3] > 0).sum())
    for k, f in zip(res.info["offsets"], res.frames):
        assert k <= 0                                                   # 上にだけ動く
        assert f.shape == padded.shape and f.dtype == np.uint8          # 全フレームの大きさが同じ
        assert set(np.unique(color_keys(f)).tolist()) <= base_colors   # 新しい色が無い
        assert foot_center(f) == (fx, fy + k)                          # 例外: 足元は上にちょうど −k だけ動く（横は動かない）
        assert shape_signature(f) == shape_signature(padded)           # 部品・輪郭のつながりが同じ
        assert euler_numbers(f) == euler_numbers(padded)               # 隙間ができない・輪郭が途切れない
        assert int((f[..., 3] > 0).sum()) == opaque                    # はみ出して欠けたドットが無い
        assert np.array_equal(f[:f.shape[0] + k], padded[-k:])         # 形はまったく同じで、位置だけが上に −k
    assert min(res.info["offsets"]) == -height


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("height", [1, 2, 3])
def test_bounce_keeps_rules(sprites, name, height):
    """どんな形でも（横揺れでは揺らせないひし形・たすきの絵も）、丸ごと上に動かすだけなので決まりを守って付く。"""
    base = sprites[name]
    res = bounce(base, height=height)
    _check_bounce(base, res, height)
    assert len(res.frames) == bounce_frames(height) and res.warnings == []
    assert res.info["height"] == height and res.info["moves_foot"] is True


def test_bounce_offsets_and_defaults():
    assert bounce_offsets(1) == [0, -1]
    assert bounce_offsets(2) == bounce_offsets() == [0, -1, -2, -1]     # 既定（決めたこと）
    assert bounce_offsets(3) == [0, -1, -2, -3, -2, -1]
    for h in (1, 2, 3):   # 1 フレームに 1 ドットずつしか動かない（最後のフレームから 1 枚目に戻るときも）
        offs = bounce_offsets(h)
        assert all(abs(a - b) == 1 for a, b in zip(offs, offs[1:] + offs[:1]))
    assert (DEFAULT_HEIGHT, AIR_DURATION, LANDING_DURATION, MOVES_FOOT) == (2, 150, 300, True)
    assert default_durations(1) == [300, 150]
    assert default_durations(2) == [300, 150, 150, 150]                # 着地（ずれ 0 の行）だけ長め
    assert default_durations(3) == [300, 150, 150, 150, 150, 150]
    assert "bounce" in MOTIONS


def test_bounce_padding_top_only_and_json(sprites):
    """変換の結果（上下が切り詰めてある）には上にだけ高さの分の行を足す。JSON は浮遊と同じ形で、pivot はフレームの下端。"""
    base = sprites["converted"]
    for h in (1, 2, 3):
        res = make_motion(base, "bounce", height=h)
        assert res.info["padding_y"] == [h, 0]
        info = sheet_info(res, "x.png")
        assert info["motion"] == "bounce"
        assert info["frame_height"] == base.shape[0] + h and info["frame_width"] == base.shape[1]
        assert info["padding"] == {"left": 0, "right": 0, "top": h, "bottom": 0}
        assert info["pivot"] == {"x": base.shape[1] / 2, "y": info["frame_height"]}
        assert [f["offset_y"] for f in info["frames"]] == bounce_offsets(h)
        assert [f["duration_ms"] for f in info["frames"]] == default_durations(h)
        assert info["total_duration_ms"] == sum(default_durations(h))
        assert "regions_used" not in info
    # 上に透明の行が余っていれば、その分は足さない
    roomy = np.pad(base, ((2, 0), (0, 0), (0, 0)))
    assert make_motion(roomy, "bounce", height=3).info["padding_y"] == [1, 0]


def test_bounce_durations(sprites):
    base = sprites["slime"]
    assert bounce(base).durations == [300, 150, 150, 150]
    assert bounce(base, durations=150).durations == [150] * 4              # 1 つなら全フレーム同じ（着地も）
    assert bounce(base, height=3, durations=[400, 100, 100, 100, 100, 100]).durations[0] == 400
    assert make_motion(base, "bounce", durations=200).durations == [200] * 4
    assert make_motion(base, "bounce", frames=4).durations == [300, 150, 150, 150]   # 高さと同じ数なら受ける


@pytest.mark.parametrize("kwargs, words", [
    ({"height": 0}, "1〜3"), ({"height": 4}, "1〜3"), ({"height": True}, "1〜3"), ({"height": 2.0}, "1〜3"),
    ({"frames": 6}, "高さで決まります"), ({"durations": [300, 150]}, "フレーム数（4）"),
    ({"durations": 0}, "1 ミリ秒以上"),
])
def test_bounce_rejects_bad_settings(sprites, kwargs, words):
    with pytest.raises(ValueError, match=words):
        bounce(sprites["slime"], **kwargs)


def test_bounce_has_no_other_options(sprites):
    for kw in ({"split": 10}, {"amount": 1}, {"depth": 1}):
        with pytest.raises(TypeError):
            make_motion(sprites["slime"], "bounce", **kw)


def test_bounce_skips_empty_image():
    res = bounce(np.zeros((6, 6, 4), np.uint8))
    assert res.skipped and res.warnings[0].startswith("弾むを付けませんでした: ")
    assert len(res.frames) == 1 and res.spec is not None


def test_bounce_is_deterministic(sprites):
    a, b = bounce(sprites["converted"], height=3), bounce(sprites["converted"], height=3)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


def test_bounce_table_is_root_only():
    """表は体全体のずれ（root_y）の列だけ。値が 0 の行には書かない。範囲は無い。"""
    t = bounce_table(3)
    assert t["name"] == "bounce" and "regions" not in t
    assert t["tracks"] == {"root_y": {"type": "root", "axis": "y"}}
    assert t["frames"] == [{"ms": 300}, {"ms": 150, "root_y": -1}, {"ms": 150, "root_y": -2}, {"ms": 150, "root_y": -3},
                           {"ms": 150, "root_y": -2}, {"ms": 150, "root_y": -1}]
    assert preset_table("bounce") == bounce_table()
    assert preset_table("bounce", durations=150, height=1) == bounce_table(1, 150)
    with pytest.raises(ValueError):
        preset_table("bounce", frames=6)
    with pytest.raises(ValueError):
        preset_table("bounce", amount=1)


def test_bounce_equals_test_spec_file(sprites):
    """PR #27 のテスト用の表（tests/motion_specs/bounce.motion.json。全部 150 ミリ秒）と、高さ 2・150 ミリ秒の弾むが同じ。"""
    base = sprites["converted"]
    a = make_motion(base, "bounce", durations=150)
    b = make_motion_from_spec(base, load_spec(SPECS / "bounce.motion.json"))
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames)) and len(a.frames) == len(b.frames)
    assert json.dumps(sheet_info(a, "x.png")) == json.dumps(sheet_info(b, "x.png"))


def test_root_value_3_is_allowed_only_for_root():
    """高さ 3 のため、体全体のずれ（root）の列だけ値を ±3 まで書ける。ほかの列は今までどおり ±2。"""
    spec = parse_spec(bounce_table(3))
    assert min(spec.column("root_y")) == -3
    bad = {"format": "sprite-motion", "version": 1, "name": "x",
           "tracks": {"root_y": {"type": "root", "axis": "y"}},
           "frames": [{"ms": 100}, {"ms": 100, "root_y": -4}]}
    with pytest.raises(SpecError, match="−3〜\\+3"):
        parse_spec(bad)
    bad = {"format": "sprite-motion", "version": 1, "name": "x",
           "regions": {"upper": {"type": "split", "at": "auto"}},
           "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
           "frames": [{"ms": 100}, {"ms": 100, "upper_y": -3}]}
    with pytest.raises(SpecError, match="−2〜\\+2"):
        parse_spec(bad)


# ---- CLI ----

@pytest.fixture(scope="module")
def hero_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_bounce(tmp_path, hero_png, capsys):
    out = tmp_path / "o" / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "--motion", "bounce"]) == 0
    text = capsys.readouterr().out
    assert "hero_sprite_bounce_sheet.png" in text and "上下のずれ +0/-1/-2/-1" in text
    assert "300/150/150/150 ミリ秒" in text and "上に 2 行・下に 0 行" in text
    assert sorted(p.name for p in out.parent.iterdir()) == [
        "hero_sprite.png", "hero_sprite_bounce_sheet.json", "hero_sprite_bounce_sheet.png"]
    info = json.loads((out.parent / "hero_sprite_bounce_sheet.json").read_text(encoding="utf-8"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert info["motion"] == "bounce" and info["frame_count"] == 4
    assert [f["offset_y"] for f in info["frames"]] == [0, -1, -2, -1]
    assert info["padding"] == {"left": 0, "right": 0, "top": 2, "bottom": 0}
    assert info["pivot"]["y"] == info["frame_height"] == sprite.shape[0] + 2
    sheet = np.asarray(Image.open(out.parent / "hero_sprite_bounce_sheet.png").convert("RGBA"))
    assert np.array_equal(sheet, np.concatenate(bounce(sprite).frames, axis=1))


@pytest.mark.parametrize("height, frames", [(1, 2), (2, 4), (3, 6)])
def test_cli_bounce_height(tmp_path, hero_png, height, frames):
    out = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "-q", "--motion", "bounce", "--bounce-height", str(height)]) == 0
    info = json.loads((tmp_path / "hero_sprite_bounce_sheet.json").read_text(encoding="utf-8"))
    assert info["frame_count"] == frames and info["padding"]["top"] == height
    assert [f["duration_ms"] for f in info["frames"]] == default_durations(height)


def test_cli_bounce_duration(tmp_path, hero_png):
    out = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "-q", "--motion", "bounce", "--motion-duration", "120"]) == 0
    info = json.loads((tmp_path / "hero_sprite_bounce_sheet.json").read_text(encoding="utf-8"))
    assert [f["duration_ms"] for f in info["frames"]] == [120] * 4
    assert cli.main([str(hero_png), "-o", str(out), "-q", "--motion", "bounce", "--bounce-height", "1",
                     "--motion-duration", "500,100"]) == 0
    info = json.loads((tmp_path / "hero_sprite_bounce_sheet.json").read_text(encoding="utf-8"))
    assert [f["duration_ms"] for f in info["frames"]] == [500, 100]


@pytest.mark.parametrize("extra, words", [
    (["--bounce-height", "2"], "動きのオプション"),
    (["--motion", "bounce", "--motion-frames", "4"], "--motion-frames は --motion bounce では使えません"),
    (["--motion", "bounce", "--bounce-height", "4"], "1〜3"),
    (["--motion", "bounce", "--bounce-height", "0"], "1 以上"),
    (["--motion", "bounce", "--float-amount", "1"], "--motion bounce では使えません（浮遊のオプションです）"),
    (["--motion", "bounce", "--sway-split", "10"], "--motion bounce では使えません（横揺れのオプションです）"),
    (["--motion", "bounce", "--breathe-depth", "1"], "--motion bounce では使えません（呼吸のオプションです）"),
    (["--motion", "float", "--bounce-height", "1"], "--motion float では使えません（弾むのオプションです）"),
    (["--motion", "bounce", "--motion-duration", "300,150"], "フレーム数（4）"),
    (["--motion", "bounce", "--bounce-height", "3", "--motion-duration", "300,150,150,150"], "フレーム数（6）"),
    (["--motion-spec", str(SPECS / "bounce.motion.json"), "--bounce-height", "2"], "--motion-spec と一緒に使えません"),
])
def test_cli_rejects_bad_bounce_options(tmp_path, hero_png, capsys, extra, words):
    with pytest.raises(SystemExit):
        cli.main([str(hero_png), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_bounce_skipped_keeps_sprite(tmp_path, capsys):
    args = cli.build_parser().parse_args(["x.png", "--motion", "bounce", "-q"])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, np.zeros((6, 6, 4), np.uint8), dst, dst) == 1
    assert "弾むを付けませんでした" in capsys.readouterr().err
    assert not (tmp_path / "s_bounce_sheet.png").exists()


@pytest.mark.parametrize("args", [["--motion", "bounce"], ["--motion", "bounce", "--bounce-height", "1"],
                                  ["--motion", "bounce", "--bounce-height", "3"],
                                  ["--motion", "bounce", "--motion-duration", "150"],
                                  ["--motion", "bounce", "--bounce-height", "3", "--motion-duration", "100,90,80,70,60,50"]])
@pytest.mark.parametrize("height", ["number", "percent"])
def test_cli_write_then_read_bounce_spec_gives_same_files(tmp_path, hero_png, args, height):
    """--write-motion-spec で書いた弾むの表を --motion-spec で読ませると、シートの PNG と JSON がバイト単位で同じ。"""
    spec = tmp_path / "b.motion.json"
    assert cli.main([str(hero_png), "-o", str(tmp_path / "a" / "s.png"), "-q", *args,
                     "--write-motion-spec", str(spec), "--spec-height", height]) == 0
    data = json.loads(spec.read_text(encoding="utf-8"))
    assert data["format"] == "sprite-motion" and data["name"] == "bounce" and "regions" not in data
    assert cli.main([str(hero_png), "-o", str(tmp_path / "b" / "s.png"), "-q", "--motion-spec", str(spec)]) == 0
    a, b = tmp_path / "a", tmp_path / "b"
    assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir()) == [
        "s.png", "s_bounce_sheet.json", "s_bounce_sheet.png"]
    for name in ("s.png", "s_bounce_sheet.png", "s_bounce_sheet.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


# ---- 画面（motion_for_web）と CLI ----

@pytest.fixture(scope="module")
def sprite_png(hero_png) -> bytes:
    res = json.loads(convert_for_web(hero_png.read_bytes(), "{}"))
    assert res["ok"], res
    return base64.b64decode(res["png_base64"])


def _pixels(data: bytes) -> np.ndarray:
    import io
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


WEB_CASES = [
    ({}, []),
    ({"height": 1}, ["--bounce-height", "1"]),
    ({"height": 2}, ["--bounce-height", "2"]),
    ({"height": 3}, ["--bounce-height", "3"]),
    ({"durations": 150}, ["--motion-duration", "150"]),
    ({"height": 3, "durations": [400, 100, 100, 100, 100, 100]}, ["--bounce-height", "3", "--motion-duration",
                                                                 "400,100,100,100,100,100"]),
    ({"height": 1, "durations": [300, 150]}, ["--bounce-height", "1", "--motion-duration", "300,150"]),
]


@pytest.mark.parametrize("options, args", WEB_CASES)
def test_web_bounce_equals_cli(hero_png, sprite_png, tmp_path, options, args):
    """画面と同じ呼び方で作ったシートの PNG と JSON が、同じ設定の CLI の出力とバイト単位で同じ（PNG は画素でも同じ）。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(dst), "-q", "--motion", "bounce", *args]) == 0
    assert dst.read_bytes() == sprite_png
    sheet = tmp_path / "hero_sprite_bounce_sheet.png"
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "bounce", **options, "image_name": sheet.name})))
    assert res["ok"] and res["skipped"] is False and res["warnings"] == []
    png = base64.b64decode(res["sheet_png_base64"])
    assert png == sheet.read_bytes()
    assert np.array_equal(_pixels(png), _pixels(sheet.read_bytes()))
    assert res["sheet_json"].replace("\n", os.linesep).encode("utf-8") == sheet.with_suffix(".json").read_bytes()
    info = json.loads(res["sheet_json"])
    assert res["padding_y"] == [info["padding"]["top"], info["padding"]["bottom"]]
    assert res["offset_y"] == [f["offset_y"] for f in info["frames"]]
    assert res["durations"] == [f["duration_ms"] for f in info["frames"]]


@pytest.mark.parametrize("options, words", [
    ({"height": 4}, ["弾む高さは 1〜3"]), ({"height": "2"}, ["弾む高さは 1〜3"]),
    ({"frames": 6}, ["高さで決まります"]), ({"durations": [300, 150]}, ["表示時間の数（2）"]),
    ({"durations": 0}, ["1 以上"]), ({"amount": 1}, ["amount は bounce の設定ではありません"]),
    ({"split": 10}, ["split は bounce の設定ではありません"]),
])
def test_web_bounce_bad_settings(sprite_png, options, words):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "bounce", **options})))
    assert res["ok"] is False
    for w in words:
        assert w in res["error"]


def test_web_height_only_for_bounce(sprite_png):
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "float", "height": 2})))
    assert res["ok"] is False and "height は float の設定ではありません" in res["error"]


def test_motion_spec_has_bounce():
    b = motion_spec()["bounce"]
    assert b["height"] == {"default": 2, "min": 1, "max": 3}
    assert b["offsets"] == {"1": [0, -1], "2": [0, -1, -2, -1], "3": [0, -1, -2, -3, -2, -1]}
    assert b["durations"]["2"] == [300, 150, 150, 150] and (b["air"], b["landing"]) == (150, 300)


def test_web_html_has_bounce_ui():
    html = (Path(__file__).resolve().parents[1] / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    assert 'id="bounce-height"' in html and "弾む高さ" in html and 'motion: "bounce"' in html


# ---- 今までの結果が変わっていないこと ----

def _digest(path) -> str:
    """ファイルの中身のハッシュ（test_sprite_float.py と同じ。PNG は画素で、JSON は改行を LF にそろえて）。"""
    if path.suffix == ".png":
        a = np.asarray(Image.open(path).convert("RGBA"))
        data = repr(a.shape).encode() + a.tobytes()
    else:
        data = legacy_bytes(path.read_bytes().replace(b"\r\n", b"\n"))   # 版 2 で足した skipped を除く
    return hashlib.sha256(data).hexdigest()[:16]


# PR #27 の main（この PR の前）の CLI で作ったファイルのハッシュ。入力は enlarge_like_ai(make_pixel_sprite(), seed=3)
BEFORE = [
    ("float", "hero_sprite.png", "616c81fd24a30eac"),
    ("float", "hero_sprite_float_sheet.json", "43a2336fb12a1cfb"),
    ("float", "hero_sprite_float_sheet.png", "486db911eaea8085"),
    ("spec_bounce", "hero_sprite.png", "616c81fd24a30eac"),
    ("spec_bounce", "hero_sprite_bounce_sheet.json", "a9e660fe710b0388"),
    ("spec_bounce", "hero_sprite_bounce_sheet.png", "4a8641e38d6e8b83"),
    ("spec_squash", "hero_sprite.png", "616c81fd24a30eac"),
    ("spec_squash", "hero_sprite_bounce_squash_sheet.json", "090d2936eab71154"),
    ("spec_squash", "hero_sprite_bounce_squash_sheet.png", "0addd314806aef6c"),
]
ARGS = {"float": ["--motion", "float"],
        "spec_bounce": ["--motion-spec", str(SPECS / "bounce.motion.json")],
        "spec_squash": ["--motion-spec", str(SPECS / "bounce_squash.motion.json")]}


@pytest.mark.parametrize("tag", list(ARGS))
def test_cli_float_and_motion_spec_are_unchanged(tmp_path, hero_png, tag):
    out = tmp_path / tag
    assert cli.main([str(hero_png), "-o", str(out / "hero_sprite.png"), "-q", *ARGS[tag]]) == 0
    got = sorted((f.name, _digest(f)) for f in legacy_files(out.iterdir()))
    assert got == sorted((name, h) for t, name, h in BEFORE if t == tag)


def test_preset_150_equals_spec_file_on_cli(tmp_path, hero_png):
    """--motion bounce --motion-duration 150 と --motion-spec bounce.motion.json（この PR の前から同じハッシュ）が同じファイル。"""
    out = tmp_path / "p"
    assert cli.main([str(hero_png), "-o", str(out / "hero_sprite.png"), "-q", "--motion", "bounce",
                     "--motion-duration", "150"]) == 0
    got = sorted((f.name, _digest(f)) for f in legacy_files(out.iterdir()))
    assert got == sorted((name, h) for t, name, h in BEFORE if t == "spec_bounce")
