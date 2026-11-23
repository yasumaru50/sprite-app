"""動きの表（タイムライン、retrocore/sprite/motion/spec.py・timeline.py）と CLI の --motion-spec / --write-motion-spec のテスト。

設計は docs/sprite-app-plan.md の「動きの表（タイムライン）の設計」。この PR は「中身を作り直すけれど、出力は今と 1 バイトも
変わらない」作業なので、次の 3 つを確かめる:

1. 今の呼吸・横揺れ・浮遊（表を埋めるプリセットに作り直した）の結果が、この PR の前とまったく同じ
   （合成の絵で `make_motion()` のたくさんの組み合わせと CLI。手持ちの ai03〜ai05 が local_samples/ にあればそれも）
2. プリセットの表を書き出して読み直す（--write-motion-spec → --motion-spec）と、元のプリセットと同じ出力になる
3. プリセット以外の表（tests/motion_specs/ の弾む相当・着地でつぶれる・わざと作れない表）が、設計どおりに作られる・警告される
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.image import load_rgba
from retrocore.sprite.motion import (SpecError, check_rules, foot_center, load_spec, make_motion, make_motion_from_spec,
                                     parse_spec, preset_table, sheet_info, sheet_json, spec_json, with_height_style)
from retrocore.sprite.motion import timeline
from retrocore.sprite.motion.base import split_limits
from retrocore.sprite.motion.floating import shift_rows
from retrocore.sprite.motion.spec import height_to_percent, percent_to_height
from legacy_output import legacy_bytes, legacy_files, legacy_sheet_text, legacy_stdout
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

ROOT = Path(__file__).resolve().parents[1]
SPECS = Path(__file__).resolve().parent / "motion_specs"
LOCAL = ROOT / "local_samples"


@pytest.fixture(scope="module")
def sprites():
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "pixel": make_pixel_sprite(),
            "converted": convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image,
            "hero": convert_sprite(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], SpriteSettings()).image}


def _table(**kw) -> dict:
    """テスト用の表（書いたものだけを上書き）。"""
    d = {"format": "sprite-motion", "version": 1, "name": "t", "loop": True,
         "regions": {"upper": {"type": "split", "at": "auto"}},
         "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"},
                    "upper_x": {"type": "move", "region": "upper", "axis": "x"},
                    "root_y": {"type": "root", "axis": "y"}},
         "frames": [{"ms": 100}, {"ms": 100, "upper_y": -1}]}
    d.update(kw)
    return d


# ---- 1. 今の動きの結果が、この PR の前とまったく同じ ----

def _dig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _combos(sp, motion):
    """比べる組み合わせ（フレーム数・量・分ける高さ。高さは指定できる範囲の端・中と、その外・画像の外）。"""
    lim = split_limits(sp)
    h = sp.shape[0]
    splits = ["auto"]
    if lim:
        lo, hi = lim
        splits += sorted({max(1, lo - 1), lo, lo + (hi - lo) // 4, (lo + hi) // 2, lo + 3 * (hi - lo) // 4, hi,
                          min(h, hi + 1)})
    splits.append(h + 1)
    if motion == "breathe":
        for f in (2, 3, 4, 8):
            for d in (1, 2):
                for s in splits:
                    yield dict(frames=f, depth=d, split=s)
        yield dict(frames=None, durations=[600, 400])
        yield dict()
    elif motion == "sway":
        for f in (3, 4, 8):
            for a in (1, 2):
                for s in splits:
                    yield dict(frames=f, amount=a, split=s)
        yield dict()
        yield dict(frames=None, durations=[100, 200, 300, 400])
    else:
        for f in (3, 4, 5, 8):
            for a in (1, 2):
                yield dict(frames=f, amount=a)
        yield dict()
        yield dict(durations=[1, 2, 3, 4])


def _records(sp, motion) -> str:
    """`make_motion()` の結果（フレーム・表示時間・警告・info・シートの JSON。設定の間違いはエラーの文）のハッシュ。"""
    recs = {}
    for kw in _combos(sp, motion):
        key = f"{motion}|{json.dumps(kw, sort_keys=True)}"
        try:
            r = make_motion(sp, motion, **kw)
            recs[key] = {"skipped": r.skipped, "warnings": r.warnings, "durations": r.durations,
                         "info": _jsonable(r.info), "n": len(r.frames),
                         "frames": _dig(b"".join(repr(f.shape).encode() + f.tobytes() for f in r.frames)),
                         "json": _dig(legacy_sheet_text(sheet_json(r, "x.png")).encode())}
        except ValueError as e:
            recs[key] = {"error": str(e)}
    recs = json.loads(json.dumps(recs))
    return _dig(json.dumps(recs, sort_keys=True, ensure_ascii=False).encode())


# この PR の前（PR #26 の main）の `_records()`。(絵, 動き) → ハッシュ。ai03〜ai05 は変換の既定（SpriteSettings()）の結果
BEFORE = {
    ("ai03", "breathe"): "7622940ebb760f97", ("ai03", "float"): "800b0e96e331e86c", ("ai03", "sway"): "5a5cc5c00ce7d912",
    ("ai04", "breathe"): "d378382f1fd09120", ("ai04", "float"): "b60bb3b206897d10", ("ai04", "sway"): "ca070ed4453a10c1",
    ("ai05", "breathe"): "0c67ecf8bce6b4ad", ("ai05", "float"): "7a9e53d3b68c479c", ("ai05", "sway"): "3dfdb0c22ba81670",
    ("converted", "breathe"): "5810c998c984d6dd", ("converted", "float"): "a6f7c47b4baf1b10",
    ("converted", "sway"): "5faab27da08f1eb7",
    ("hero", "breathe"): "cc6bd9e4ae28b7c6", ("hero", "float"): "238fdc10f3594956", ("hero", "sway"): "a745f924a4140cbb",
    ("mushroom", "breathe"): "b09817c1cff949f4", ("mushroom", "float"): "bb468937594bb693",
    ("mushroom", "sway"): "4ceb4439b4f20bf0",
    ("pixel", "breathe"): "f8d4066d59beaf6f", ("pixel", "float"): "465cccb86380580c", ("pixel", "sway"): "1c3cd07476a687b1",
    ("slime", "breathe"): "45632acf995fd3bd", ("slime", "float"): "dc34b5f14711c404", ("slime", "sway"): "38d507fd88a8991a",
}
SYNTH = ["mushroom", "slime", "pixel", "converted", "hero"]


@pytest.mark.parametrize("name", SYNTH)
@pytest.mark.parametrize("motion", ["breathe", "sway", "float"])
def test_presets_are_unchanged(sprites, name, motion):
    """呼吸・横揺れ・浮遊の結果（フレーム・警告・info・シートの JSON・エラーの文）が、この PR の前と同じ（合成の絵）。"""
    assert _records(sprites[name], motion) == BEFORE[(name, motion)]


# ai03〜ai05 の記録は、大きな面の色を守る（keep_large、単純化）を足す前の変換の結果。動きの出力を比べるテストなので、
# 変換はその前の設定に固定する（ai04 は 12 色で 28 ドット変わる。ai03・ai05 は keep_large があっても同じ）
AI_SETTINGS = SpriteSettings(keep_large=0)
AI_CLI_EXTRA = ["--keep-large", "0"]


def _ai(name):
    path = LOCAL / f"{name}.png"
    if not path.exists():
        pytest.skip(f"{path} がありません（手持ちの AI 画像で確かめるときだけ。preview/ai-check の ai-input/ にある）")
    return path


@pytest.fixture(scope="module")
def ai_sprites():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = convert_sprite(load_rgba(_ai(name)), AI_SETTINGS).image
        return cache[name]
    return get


@pytest.mark.parametrize("name", ["ai03", "ai04", "ai05"])
@pytest.mark.parametrize("motion", ["breathe", "sway", "float"])
def test_presets_are_unchanged_on_ai_images(ai_sprites, name, motion):
    """手持ちの ai03〜ai05（local_samples/ にあるときだけ）でも、この PR の前と同じ。"""
    assert _records(ai_sprites(name), motion) == BEFORE[(name, motion)]


def _file_digest(path: Path) -> str:
    """PNG は画素（大きさと RGBA）、JSON は改行を LF にそろえたバイト列のハッシュ。"""
    if path.suffix == ".png":
        a = np.asarray(Image.open(path).convert("RGBA"))
        return _dig(repr(a.shape).encode() + a.tobytes())
    return _dig(legacy_bytes(path.read_bytes().replace(b"\r\n", b"\n")))


# この PR の前の CLI の出力（ファイル名 → `_file_digest()`）。ai03〜ai05（local_samples/ にあるときだけ確かめる）
AI_CLI_BEFORE = {
    ("ai03", "b"): {"s.png": "af1ab0484e2237e8", "s_breathe_sheet.json": "b937e6ec3e3a3791", "s_breathe_sheet.png": "2c846bb0264550dc"},
    ("ai03", "b_s12"): {"s.png": "af1ab0484e2237e8", "s_breathe_sheet.json": "b937e6ec3e3a3791", "s_breathe_sheet.png": "33bbbceee54e13ec"},
    ("ai03", "f"): {"s.png": "af1ab0484e2237e8", "s_float_sheet.json": "ae06dc05568d616b", "s_float_sheet.png": "2037f46dd7f7e503"},
    ("ai03", "f_f8a2"): {"s.png": "af1ab0484e2237e8", "s_float_sheet.json": "35797199d02122c2", "s_float_sheet.png": "72db605b54c4e4ff"},
    ("ai03", "s"): {"s.png": "af1ab0484e2237e8", "s_sway_sheet.json": "0ce99706c69d009d", "s_sway_sheet.png": "e54c7be4d0d1c0ea"},
    ("ai03", "s_f8a2"): {"s.png": "af1ab0484e2237e8", "s_sway_sheet.json": "743fe1d731ef22d3", "s_sway_sheet.png": "88abc332e8237570"},
    ("ai04", "b"): {"s.png": "d988a3df890d49a9", "s_breathe_sheet.json": "619000543f6439c9", "s_breathe_sheet.png": "ba01d99baeda495a"},
    ("ai04", "b_s12"): {"s.png": "d988a3df890d49a9", "s_breathe_sheet.json": "619000543f6439c9", "s_breathe_sheet.png": "f87e8ccc7bed8acf"},
    ("ai04", "f"): {"s.png": "d988a3df890d49a9", "s_float_sheet.json": "23c10cef8239cb44", "s_float_sheet.png": "88056cce72e4a8c3"},
    ("ai04", "f_f8a2"): {"s.png": "d988a3df890d49a9", "s_float_sheet.json": "c8ec3eda2f337d7e", "s_float_sheet.png": "c63c2325d7875c13"},
    ("ai04", "s"): {"s.png": "d988a3df890d49a9", "s_sway_sheet.json": "6cf53e195ec3c4c9", "s_sway_sheet.png": "aa799d405e9e884d"},
    ("ai04", "s_f8a2"): {"s.png": "d988a3df890d49a9", "s_sway_sheet.json": "e931b538037ce9f9", "s_sway_sheet.png": "85581dab17bbc5c3"},
    ("ai05", "b"): {"s.png": "ef7ab41bede43b69", "s_breathe_sheet.json": "879ccf636ebeaac1", "s_breathe_sheet.png": "e6fe61f6fa30d391"},
    ("ai05", "b_s12"): {"s.png": "ef7ab41bede43b69", "s_breathe_sheet.json": "879ccf636ebeaac1", "s_breathe_sheet.png": "c63921fec30ff1ea"},
    ("ai05", "f"): {"s.png": "ef7ab41bede43b69", "s_float_sheet.json": "61520f2f4c6e1ee5", "s_float_sheet.png": "4d4bf7280cbf832b"},
    ("ai05", "f_f8a2"): {"s.png": "ef7ab41bede43b69", "s_float_sheet.json": "06d8a3e2d728b12a", "s_float_sheet.png": "a2b75c4caa393160"},
    ("ai05", "s"): {"s.png": "ef7ab41bede43b69", "s_sway_sheet.json": "47f52bf4ca61e9e7", "s_sway_sheet.png": "7f0ac02a5802bd21"},
    ("ai05", "s_f8a2"): {"s.png": "ef7ab41bede43b69", "s_sway_sheet.json": "f205f3aeb9ec94e9", "s_sway_sheet.png": "57d9e378cd0b28bd"},
}
CLI_ARGS = {
    "none": [], "b": ["--motion", "breathe"],
    "b_f4d2": ["--motion", "breathe", "--motion-frames", "4", "--breathe-depth", "2"],
    "b_f8": ["--motion", "breathe", "--motion-frames", "8"], "b_s12": ["--motion", "breathe", "--breathe-split", "12"],
    "b_s8": ["--motion", "breathe", "--breathe-split", "8"], "b_s20": ["--motion", "breathe", "--breathe-split", "20"],
    "b_s3": ["--motion", "breathe", "--breathe-split", "3"], "b_s999": ["--motion", "breathe", "--breathe-split", "999"],
    "b_d": ["--motion", "breathe", "--motion-duration", "600,400"],
    "b_auto": ["--motion", "breathe", "--breathe-split", "auto", "--motion-frames", "4"],
    "s": ["--motion", "sway"], "s_f8a2": ["--motion", "sway", "--motion-frames", "8", "--sway-amount", "2"],
    "s_f3": ["--motion", "sway", "--motion-frames", "3"], "s_f4a2": ["--motion", "sway", "--sway-amount", "2"],
    "s_s10": ["--motion", "sway", "--sway-split", "10"], "s_s15": ["--motion", "sway", "--sway-split", "15"],
    "s_s25": ["--motion", "sway", "--sway-split", "25"], "s_s3": ["--motion", "sway", "--sway-split", "3"],
    "s_d": ["--motion", "sway", "--motion-duration", "100,200,300,400"],
    "f": ["--motion", "float"], "f_f8a2": ["--motion", "float", "--motion-frames", "8", "--float-amount", "2"],
    "f_f3": ["--motion", "float", "--motion-frames", "3"], "f_a2": ["--motion", "float", "--float-amount", "2"],
    "f_d": ["--motion", "float", "--motion-duration", "100,200,300,400"],
}
# この PR の前の、合成の hero（enlarge_like_ai(make_pixel_sprite(), seed=3)）の CLI_ARGS 全部の
# （終了コード・ファイル名とハッシュ・警告とエラーの文・進捗の表示（時間の行は除く））のハッシュ
HERO_CLI_BEFORE = "39a85de5d77f03de"


def _run_cli(src: Path, out: Path, args: list[str], tmp: Path) -> dict:
    err, so = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
        code = cli.main([str(src), "-o", str(out / "s.png"), *args])
    stdout = so.getvalue().replace(str(tmp), "<TMP>").replace(str(ROOT), "<ROOT>").replace("\\", "/")
    stdout = legacy_stdout("\n".join(line for line in stdout.splitlines() if "秒）" not in line))
    return {"code": code, "files": {f.name: _file_digest(f) for f in legacy_files(sorted(out.iterdir()))},
            "stderr": err.getvalue(), "stdout": stdout}


def test_cli_presets_are_unchanged(tmp_path):
    """--motion なし・呼吸・横揺れ・浮遊の CLI の出力（ファイル・警告・エラー・終了コード・表示）が、この PR の前と同じ。"""
    src = tmp_path / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(src)
    recs = {f"hero|{tag}": _run_cli(src, tmp_path / "hero" / tag, args, tmp_path) for tag, args in CLI_ARGS.items()}
    assert _dig(json.dumps(recs, sort_keys=True, ensure_ascii=False).encode()) == HERO_CLI_BEFORE


@pytest.mark.parametrize("key", sorted(AI_CLI_BEFORE))
def test_cli_presets_are_unchanged_on_ai_images(tmp_path, key):
    """手持ちの ai03〜ai05（local_samples/ にあるときだけ）の CLI の出力が、この PR の前と同じ。"""
    name, tag = key
    got = _run_cli(_ai(name), tmp_path / tag, CLI_ARGS[tag] + AI_CLI_EXTRA, tmp_path)
    assert got["code"] == 0 and got["files"] == AI_CLI_BEFORE[key]


# ---- 2. プリセットの表を書き出して読み直すと、同じ出力 ----

ROUNDTRIP = [("breathe", {}), ("breathe", {"frames": 4, "depth": 2}), ("breathe", {"frames": 8}),
             ("breathe", {"split": 12}), ("breathe", {"split": 9, "frames": 4, "depth": 2}),
             ("breathe", {"durations": [600, 400]}), ("breathe", {"split": 3}),
             ("sway", {}), ("sway", {"frames": 8, "amount": 2}), ("sway", {"frames": 3}), ("sway", {"split": 10}),
             ("sway", {"split": 15, "amount": 2}), ("sway", {"split": 2}),
             ("float", {}), ("float", {"frames": 8, "amount": 2}), ("float", {"frames": 3}),
             ("float", {"durations": [100, 200, 300, 400]})]


def _same_output(a, b) -> bool:
    """同じ出力: 作れたかどうかが同じで、作れたときはフレームが画素まで同じ、シートの JSON がバイト単位で同じ
    （表から作ったときだけ足す regions_used を除く）。"""
    if a.skipped != b.skipped:
        return False
    if a.skipped:
        return True
    jb = sheet_info(b, "x.png")
    jb.pop("regions_used", None)
    return (len(a.frames) == len(b.frames) and all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))
            and a.durations == b.durations
            and json.dumps(jb, ensure_ascii=False, indent=2) + "\n" == sheet_json(a, "x.png"))


@pytest.mark.parametrize("name", SYNTH)
@pytest.mark.parametrize("motion, options", ROUNDTRIP, ids=[f"{m}-{i}" for i, (m, _) in enumerate(ROUNDTRIP)])
@pytest.mark.parametrize("style", ["number", "percent"])
def test_preset_table_roundtrip(sprites, name, motion, options, style):
    """プリセットが埋めた表（"used" つき）を JSON に書いて読み直すと、元のプリセットと同じ出力（高さは数でも割合でも）。"""
    sp = sprites[name]
    a = make_motion(sp, motion, **options)
    assert a.spec is not None and a.spec["name"] == motion
    text = spec_json(with_height_style(a.spec, sp, style))
    b = make_motion_from_spec(sp, json.loads(text))
    assert _same_output(a, b), (a.warnings, b.warnings)
    if not a.skipped:
        assert ("used" in json.dumps(a.spec)) == (motion != "float")   # 範囲のある動きは、実際に使った高さと線が付く


def test_preset_table_matches_design():
    """プリセットの表は設計（docs/sprite-app-plan.md の 2.）の表と同じ形。"""
    assert preset_table("breathe")["frames"] == [{"ms": 500}, {"ms": 500, "upper_y": -1}]
    assert [f.get("upper_y", 0) for f in preset_table("breathe", frames=4, depth=2)["frames"]] == [0, -1, -2, -1]
    assert preset_table("breathe", split=12)["regions"]["upper"]["at"] == 12
    assert [f.get("upper_x", 0) for f in preset_table("sway")["frames"]] == [0, 1, 0, -1]
    assert [f.get("upper_x", 0) for f in preset_table("sway", frames=8, amount=2)["frames"]] == [0, 1, 2, 1, 0, -1, -2, -1]
    t = preset_table("float")
    assert "regions" not in t and t["tracks"] == {"root_y": {"type": "root", "axis": "y"}}
    assert [(f["ms"], f.get("root_y", 0)) for f in t["frames"]] == [(400, 0), (400, -1), (400, 0), (400, 1)]
    for name in ("breathe", "sway", "float"):
        parse_spec(preset_table(name))   # 書き間違いが無い


def test_lines_reproduce_the_same_frames(sprites):
    """"used" の線を "lines" に書くと、継ぎ目・分け目を選ばずにその線で同じフレームができる（縮める 2・横 2 も）。"""
    for name in ("mushroom", "hero"):
        sp = sprites[name]
        for motion, options, track in (("breathe", {"frames": 4, "depth": 2}, "upper_y"),
                                       ("sway", {"frames": 8, "amount": 2}, "upper_x")):
            a = make_motion(sp, motion, **options)
            assert not a.skipped
            t = json.loads(json.dumps(a.spec))
            t["regions"]["upper"]["lines"] = {track: t["regions"]["upper"]["used"]["lines"][track]}
            t["regions"]["upper"]["at"] = 1      # lines があれば at では選ばない（画像の中の数なら何でもよい）
            b = make_motion_from_spec(sp, t)
            assert _same_output(a, b)
            assert "regions_used" not in b.info   # 線をそのまま使ったときは書かない
            # 幅の違う線はエラー（ValueError。何も作らない）
            t["regions"]["upper"]["lines"][track] = [line[:-1] for line in t["regions"]["upper"]["lines"][track]]
            with pytest.raises(ValueError, match="幅"):
                make_motion_from_spec(sp, t)


# ---- 3. プリセット以外の表 ----

def test_bounce_table(sprites):
    """弾む相当（体全体のずれ 0, −1, −2, −1）: 上にだけ動くので余白は上だけ、offset_y、pivot はフレームの下端。"""
    spec = load_spec(SPECS / "bounce.motion.json")
    for name in ("mushroom", "slime", "hero"):
        sp = sprites[name]
        res = make_motion_from_spec(sp, spec)
        assert not res.skipped, res.warnings
        assert res.name == "bounce" and res.durations == [150] * 4
        top_blank = int(np.flatnonzero((sp[..., 3] > 0).any(axis=1))[0])
        assert res.info["padding_y"] == [max(0, 2 - top_blank), 0] and res.info["offset_y"] == [0, -1, -2, -1]
        padded = np.pad(sp, ((res.info["padding_y"][0], 0), (0, 0), (0, 0)))
        assert all(np.array_equal(f, shift_rows(padded, k)) for f, k in zip(res.frames, [0, -1, -2, -1]))
        assert check_rules(padded, res.frames, shifts=[0, -1, -2, -1]) == []
        info = sheet_info(res, "x_bounce_sheet.png")
        assert info["motion"] == "bounce" and info["padding"] == {"left": 0, "right": 0, "top": res.info["padding_y"][0],
                                                                  "bottom": 0}
        assert info["pivot"] == {"x": sp.shape[1] / 2, "y": padded.shape[0]}
        assert [f["offset_y"] for f in info["frames"]] == [0, -1, -2, -1]
        assert "regions_used" not in info


def test_bounce_squash_table(sprites):
    """着地でつぶれる（1 行目だけ縮める −1 ＋ 体全体のずれ）: 1 行目はポーズ（足元が動かない）、ほかはずらしただけ。"""
    spec = load_spec(SPECS / "bounce_squash.motion.json")
    sp = sprites["mushroom"]
    res = make_motion_from_spec(sp, spec)
    assert not res.skipped, res.warnings
    breath = make_motion(sp, "breathe")            # 縮める −1 のポーズは呼吸の 2 枚目と同じ
    top = res.info["padding_y"][0]
    padded = np.pad(sp, ((top, 0), (0, 0), (0, 0)))
    assert np.array_equal(res.frames[0], np.pad(breath.frames[1], ((top, 0), (0, 0), (0, 0))))
    assert foot_center(res.frames[0]) == foot_center(padded)                       # ポーズでは足元が動かない
    assert check_rules(padded, res.frames[1:], shifts=[-1, -2, -1]) == []         # ずれは決まった量だけ
    assert res.info["poses"] == 1 and res.info["offset_y"] == [0, -1, -2, -1]
    used = res.info["regions_used"]["upper"]
    assert used["at"] == breath.info["seam_from_bottom"][0][0] and used["lines"]["upper_y"][0] == \
        [sp.shape[0] - v for v in breath.info["seams"][0]]
    assert sheet_info(res, "x.png")["regions_used"] == res.info["regions_used"]


def test_poses_are_made_once(sprites, monkeypatch):
    """同じ値のポーズは 1 回だけ作る（8 フレームの呼吸でもポーズは −1 の 1 つ。体全体のずれと組み合わせても同じ）。"""
    calls = []
    real = timeline.shrink_poses

    def spy(*a, **k):
        calls.append(a[2])
        return real(*a, **k)
    monkeypatch.setattr(timeline, "shrink_poses", spy)
    res = make_motion(sprites["mushroom"], "breathe", frames=8)
    assert not res.skipped and res.info["offsets"] == [0, 0, 0, 1, 1, 1, 0, 0] and calls == [1]
    t = _table(frames=[{"ms": 100}] + [{"ms": 100, "upper_y": -1, "root_y": v} for v in (0, -1, -1, 0, 1)])
    res = make_motion_from_spec(sprites["mushroom"], t)
    assert not res.skipped, res.warnings
    assert res.info["poses"] == 1 and calls == [1, 1]
    assert np.array_equal(res.frames[2], res.frames[3]) and res.frames[2] is not res.frames[3]


def test_broken_table_warns_every_row(sprites):
    """わざと作れない表: シートは作らず、全部の行の問題を「N 行目（列 = 値、範囲 名前）: …」でまとめて知らせる（行は 1 から）。"""
    sp = sprites["mushroom"]
    res = make_motion_from_spec(sp, load_spec(SPECS / "broken.motion.json"))
    assert res.skipped and len(res.frames) == 1 and np.array_equal(res.frames[0], sp)
    assert res.durations == [1000]
    lim = split_limits(sp)
    # 伸ばす（+1）も作れるようになったので、3 行目も縮めると同じ「足元の中の高さ」の問題になり、1 つにまとまる
    assert res.warnings == [
        "動き「broken」のシートを作りませんでした（作れない行が 3 行あります。問題 1 件）",
        f"2・3・4 行目（upper_y = −1・+1、範囲 upper）: 下から 3 ドット目では分けられません（足元の {lim[0] - 1} 行は動かさないため、"
        f"この絵で指定できるのは {lim[0]}〜{lim[1]}）",
    ]
    probs = res.info["problems"]
    assert [p["rows"] for p in probs] == [[2, 3, 4]] and [p["values"] for p in probs] == [[-1, 1]]
    assert [p["rule"] for p in probs] == ["split_not_found"]
    assert probs[0]["track"] == "upper_y" and probs[0]["region"] == "upper" and probs[0]["rows_from_bottom"] == [3, 3]


def test_problem_reports_nearby_heights(sprites):
    """指定した高さの近くで縮められないときは、近くで分けられる高さ（実際に作って確かめたもの）を知らせる。"""
    from test_sprite_breathe_split import striped
    img = striped()
    split = img.shape[0] - 14
    old = make_motion(img, "breathe", split=split)
    res = make_motion_from_spec(img, _table(regions={"upper": {"type": "split", "at": split}}))
    assert old.skipped and res.skipped
    p = res.info["problems"][0]
    assert p["rule"] == "split_not_found" and p["row"] == 2 and p["near"]
    hint = old.warnings[0].split("。")[-1]
    assert hint.startswith("近くで分けられる高さ: ") and res.warnings[1].endswith(hint)
    assert res.warnings[1].startswith(f"2 行目（upper_y = −1、範囲 upper）: 下から {split - 2}〜{split + 2} ドット目では、")


def test_second_seam_failure_marks_deeper_rows_only(sprites):
    """縮める 2 の 2 本目の継ぎ目だけが作れないときは、−2 の行だけを知らせる（−1 の行は作れる）。"""
    sp = sprites["converted"]
    assert make_motion(sp, "breathe", depth=2, split=29).skipped and not make_motion(sp, "breathe", split=29).skipped
    t = _table(regions={"upper": {"type": "split", "at": 29}},
               frames=[{"ms": 1}, {"ms": 1, "upper_y": -1}, {"ms": 1, "upper_y": -2}, {"ms": 1, "upper_y": -1}])
    res = make_motion_from_spec(sp, t)
    assert res.skipped and [p["rows"] for p in res.info["problems"]] == [[3]]
    assert "（2 本目の継ぎ目）" in res.warnings[1] and res.warnings[1].startswith("3 行目（upper_y = −2、範囲 upper）")


def test_move_failure_marks_all_moving_rows():
    """横に動かせない絵（1 ドットの輪郭だけのひし形）では、横の位置を使う全部の行を知らせる。"""
    from test_sprite_sway import diamond
    res = make_motion_from_spec(diamond(), _table(frames=[{"ms": 100}, {"ms": 100, "upper_x": 1}, {"ms": 100},
                                                          {"ms": 100, "upper_x": -1}]))
    assert res.skipped
    assert res.warnings[1].startswith("2・4 行目（upper_x = +1・−1、範囲 upper）: 自動では、")
    assert make_motion(diamond(), "sway").skipped


def test_root_on_empty_image():
    res = make_motion_from_spec(np.zeros((8, 8, 4), np.uint8), load_spec(SPECS / "bounce.motion.json"))
    assert res.skipped and res.info["problems"][0]["rule"] == "no_image"
    assert res.warnings[1] == "2・3・4 行目（root_y = −1・−2）: 絵が無いので、体全体を動かせません（不透明なドットが無い）"


def test_move_and_root_combination(sprites):
    """横の位置と体全体のずれを別々の列で一緒に使うと、余白は左右と上下の両方（左右は同じ数）。"""
    sp = sprites["mushroom"]
    t = _table(frames=[{"ms": 100}, {"ms": 100, "upper_x": 1, "root_y": -1}, {"ms": 100, "root_y": -1},
                       {"ms": 100, "upper_x": -1}])
    res = make_motion_from_spec(sp, t)
    assert not res.skipped, res.warnings
    sway = make_motion(sp, "sway")
    p = sway.info["padding"]
    info = sheet_info(res, "x.png")
    assert info["padding"]["left"] == info["padding"]["right"] == p and info["padding"]["bottom"] == 0
    assert np.array_equal(res.frames[1], shift_rows(np.pad(sway.frames[1], ((info["padding"]["top"], 0), (0, 0), (0, 0))), -1))
    assert [f["offset_y"] for f in info["frames"]] == [0, -1, -1, 0]


def test_percent_heights_roundtrip(sprites):
    """割合の高さ: 数 → 割合 → 数で同じ数に戻る（キャラの下に透明の行がある絵でも）。基準は透明を除いたキャラの高さ。"""
    for sp in list(sprites.values()) + [np.pad(sprites["mushroom"], ((5, 7), (0, 0), (0, 0)))]:
        h = sp.shape[0]
        blank = h - 1 - int(np.flatnonzero((sp[..., 3] > 0).any(axis=1))[-1])
        for n in range(blank + 1, h + 1):
            p = height_to_percent(sp, n)
            assert p is not None and p.endswith("%") and percent_to_height(sp, p) == n
    sp = np.pad(sprites["mushroom"], ((0, 4), (0, 0), (0, 0)))   # 下に透明の行が 4 行
    rows = np.flatnonzero((sprites["mushroom"][..., 3] > 0).any(axis=1))
    hh = int(rows[-1] - rows[0] + 1)
    blank = sprites["mushroom"].shape[0] - 1 - int(rows[-1])
    assert percent_to_height(sp, "50%") == 4 + blank + int(np.floor(hh * 0.5 + 0.5))
    assert percent_to_height(sp, "0.1%") == 4 + blank + 1        # 1 より小さければ 1
    assert percent_to_height(np.zeros((4, 4, 4), np.uint8), "50%") is None


def test_percent_table_uses_the_same_height(sprites):
    """at を割合で書いた表は、その割合の数で書いた表と同じ結果。作ると regions_used に実際に使った高さが付く。"""
    sp = sprites["hero"]
    n = make_motion(sp, "breathe").info["seam_from_bottom"][0][0]
    p = height_to_percent(sp, n)
    a = make_motion(sp, "breathe", split=n)
    b = make_motion_from_spec(sp, _table(name="breathe", regions={"upper": {"type": "split", "at": p}},
                                         frames=[{"ms": 500}, {"ms": 500, "upper_y": -1}]))
    assert _same_output(a, b)
    assert b.info["regions_used"]["upper"]["at"] == n


def test_split_outside_image_is_error(sprites):
    sp = sprites["hero"]
    with pytest.raises(ValueError, match="範囲 upper の at: 分ける高さ 99 は画像の範囲の外です"):
        make_motion_from_spec(sp, _table(regions={"upper": {"type": "split", "at": 99}}))


def test_zero_rows_only(sprites):
    """値が全部 0 の表は、元の絵が並ぶだけ（何も動かない。決まりは守れている）。"""
    sp = sprites["mushroom"]
    res = make_motion_from_spec(sp, _table(frames=[{"ms": 10}, {"ms": 20}]))
    assert not res.skipped and all(np.array_equal(f, sp) for f in res.frames) and res.durations == [10, 20]
    assert "padding" not in sheet_info(res, "x.png")


@pytest.mark.parametrize("change, words", [
    ({"format": None}, "これは動きの表ではありません"),
    ({"version": 2}, "version は 1"),
    ({"name": "a b"}, "name は英数字"),
    ({"loop": False}, "loop は今は true だけ"),
    ({"regions": {"upper": {"type": "mask", "at": "auto"}}}, "type は \"split\""),   # 帯・四角は読める（test_sprite_region.py）
    ({"regions": {"upper": {"type": "split", "at": 0}}}, "at は \"auto\""),
    ({"regions": {"upper": {"type": "split", "at": "120%"}}}, "at は \"auto\""),
    ({"regions": {"upper": {"type": "split", "at": "auto", "aim": 0.4}}}, "aim（自動で狙う高さ）はまだ"),
    ({"regions": {"upper": {"type": "split", "at": "auto", "range": 99}}}, "range は 0〜8"),
    ({"regions": {"upper": {"type": "split", "at": "auto", "lines": {"upper_y": 3}}}}, "lines は"),
    ({"regions": {"upper": {"type": "split", "at": "auto", "color": 1}}}, "知らない項目です: color"),
    ({"tracks": {"spin": {"type": "rotate", "axis": "z"}}}, "知らない種類です"),
    ({"tracks": {"up": {"type": "move", "region": "upper", "axis": "y"}}}, "縦の位置（move の axis \"y\"）は使えません"),
    ({"tracks": {"r": {"type": "root", "region": "upper", "axis": "y"}}}, "範囲を持ちません"),
    ({"tracks": {"d": {"type": "deform", "region": "head", "axis": "y"}}}, "範囲 \"head\" が regions にありません"),
    ({"tracks": {"a": {"type": "deform", "region": "upper", "axis": "y"},
                 "b": {"type": "deform", "region": "upper", "axis": "y"}}}, "2 つあります（a・b）"),
    ({"frames": [{"ms": 100}]}, "2〜32 行"),
    ({"frames": [{"ms": 100}] * 33}, "2〜32 行"),
    ({"frames": [{"ms": 100}, {"ms": 0}]}, "2 行目: ms"),
    ({"frames": [{"ms": 100}, {"ms": 100, "upper_y": 3}]}, "2 行目: upper_y の値は −2〜+2"),
    ({"frames": [{"ms": 100}, {"ms": 100, "upper_y": True}]}, "2 行目: upper_y の値は"),
    ({"frames": [{"ms": 100}, {"ms": 100, "head_y": -1}]}, "2 行目: 知らない列です: head_y"),
    ({"frames": [{"ms": 100}, {"ms": 100, "upper_y": -1, "upper_x": 1}]}, "横の位置と縦の変形を同じ行で一緒に使うのは、まだ作れません"),
])
def test_spec_mistakes_are_errors(change, words):
    t = _table()
    t.update(change)
    if t["format"] is None:
        del t["format"]
    with pytest.raises(SpecError, match=words.replace("(", r"\(").replace(")", r"\)").replace("+", r"\+")):
        parse_spec(t)


def test_spec_two_regions_in_one_row_is_error():
    t = _table(regions={"upper": {"type": "split", "at": "auto"}, "head": {"type": "split", "at": 20}},
               tracks={"upper_y": {"type": "deform", "region": "upper", "axis": "y"},
                       "head_y": {"type": "deform", "region": "head", "axis": "y"}},
               frames=[{"ms": 1}, {"ms": 1, "upper_y": -1, "head_y": -1}])
    with pytest.raises(SpecError, match="2 つの範囲（head・upper）を動かすのは、まだ作れません"):
        parse_spec(t)


def test_spec_collects_all_mistakes():
    """書き間違いは 1 つ見つけても止めず、全部まとめて知らせる。"""
    with pytest.raises(SpecError) as e:
        parse_spec({"format": "sprite-motion", "version": 2, "name": "a b", "tracks": {"x": {"type": "spin"}},
                    "frames": [{"ms": 0, "zz": 1}]})
    assert len(e.value.problems) == 6


def test_sheet_json_is_not_a_table(tmp_path, sprites):
    """シートの JSON をうっかり --motion-spec に渡したら「これは動きの表ではありません」。"""
    res = make_motion(sprites["mushroom"], "breathe")
    with pytest.raises(SpecError, match="スプライトシートの JSON です"):
        parse_spec(sheet_json(res, "x.png"))
    with pytest.raises(SpecError, match="JSON として読めません"):
        parse_spec("{")


def test_spec_files_in_tests_are_valid():
    for path in sorted(SPECS.glob("*.motion.json")):
        spec = load_spec(path)
        assert spec.name == path.name.split(".")[0]


# ---- CLI ----

@pytest.fixture(scope="module")
def hero_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(path)
    return path


@pytest.mark.parametrize("args", [["--motion", "breathe"], ["--motion", "breathe", "--breathe-split", "12"],
                                  ["--motion", "breathe", "--motion-frames", "4", "--breathe-depth", "2"],
                                  ["--motion", "sway"], ["--motion", "sway", "--motion-frames", "8", "--sway-amount", "2"],
                                  ["--motion", "sway", "--sway-split", "10"],
                                  ["--motion", "float"], ["--motion", "float", "--motion-duration", "100,200,300,400"]])
@pytest.mark.parametrize("height", ["number", "percent"])
def test_cli_write_then_read_spec_gives_same_files(tmp_path, hero_png, args, height):
    """--write-motion-spec で書いた表を --motion-spec で読ませると、シートの PNG は同じ、JSON も regions_used を除いて同じ。"""
    spec = tmp_path / "x.motion.json"
    assert cli.main([str(hero_png), "-o", str(tmp_path / "a" / "s.png"), "-q", *args,
                     "--write-motion-spec", str(spec), "--spec-height", height]) == 0
    data = json.loads(spec.read_text(encoding="utf-8"))
    assert data["format"] == "sprite-motion" and data["name"] == args[1]
    if args[1] != "float":
        assert "used" in data["regions"]["upper"]
        at = data["regions"]["upper"]["at"]
        assert at == "auto" or (isinstance(at, int) if height == "number" else at.endswith("%"))
    assert cli.main([str(hero_png), "-o", str(tmp_path / "b" / "s.png"), "-q", "--motion-spec", str(spec)]) == 0
    name = args[1]
    a, b = tmp_path / "a", tmp_path / "b"
    assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir())
    assert (a / f"s_{name}_sheet.png").read_bytes() == (b / f"s_{name}_sheet.png").read_bytes()
    ja = (a / f"s_{name}_sheet.json").read_text(encoding="utf-8")
    jb = json.loads((b / f"s_{name}_sheet.json").read_text(encoding="utf-8"))
    jb.pop("regions_used", None)
    assert json.dumps(jb, ensure_ascii=False, indent=2) + "\n" == ja.replace("\r\n", "\n")


def test_cli_motion_spec_bounce(tmp_path, hero_png, capsys):
    out = tmp_path / "hero_sprite.png"
    assert cli.main([str(hero_png), "-o", str(out), "--motion-spec", str(SPECS / "bounce.motion.json")]) == 0
    text = capsys.readouterr().out
    assert "動きの表 bounce.motion.json（bounce）: 4 フレーム" in text and "上下のずれ +0/-1/-2/-1" in text
    info = json.loads((tmp_path / "hero_sprite_bounce_sheet.json").read_text(encoding="utf-8"))
    assert info["motion"] == "bounce" and info["padding"]["bottom"] == 0 and info["frame_count"] == 4
    assert (tmp_path / "hero_sprite_bounce_sheet.png").exists()


def test_cli_motion_spec_broken_warns_and_keeps_sprite(tmp_path, hero_png, capsys):
    out = tmp_path / "s.png"
    spec_out = tmp_path / "again.motion.json"
    assert cli.main([str(hero_png), "-o", str(out), "-q", "--motion-spec", str(SPECS / "broken.motion.json"),
                     "--write-motion-spec", str(spec_out)]) == 0
    err = capsys.readouterr().err
    assert "警告（hero.png）: 動き「broken」のシートを作りませんでした" in err
    assert "警告（hero.png）: 2・3・4 行目（upper_y = −1・+1、範囲 upper）: 下から 3 ドット目では分けられません" in err
    assert out.exists() and not (tmp_path / "s_broken_sheet.png").exists()
    assert load_spec(spec_out).name == "broken"     # 作れなくても表は書く（直して使えるように）


def test_cli_motion_spec_image_error(tmp_path, hero_png, capsys):
    t = tmp_path / "far.motion.json"
    t.write_text(json.dumps(_table(regions={"upper": {"type": "split", "at": 999}})), encoding="utf-8")
    assert cli.main([str(hero_png), "-o", str(tmp_path / "s.png"), "-q", "--motion-spec", str(t)]) == 1
    assert "エラー（hero.png）: 範囲 upper の at: 分ける高さ 999 は画像の範囲の外です" in capsys.readouterr().err


@pytest.mark.parametrize("extra, words", [
    (["--motion", "breathe", "--motion-spec", "x.json"], "--motion-spec と --motion は一緒に使えません"),
    (["--motion-spec", "x.json", "--motion-frames", "4"], "--motion-frames は --motion-spec と一緒に使えません"),
    (["--motion-spec", "x.json", "--breathe-split", "4"], "--breathe-split は --motion-spec と一緒に使えません"),
    (["--write-motion-spec", "x.json"], "--write-motion-spec は --motion breathe などか --motion-spec と一緒に"),
    (["--motion", "sway", "--spec-height", "percent"], "--spec-height は --write-motion-spec と一緒に"),
])
def test_cli_rejects_bad_spec_options(tmp_path, capsys, extra, words):
    with pytest.raises(SystemExit) as e:
        cli.main([str(tmp_path / "x.png"), *extra])
    assert e.value.code == 2
    assert words in capsys.readouterr().err


def test_cli_rejects_broken_spec_file_before_converting(tmp_path, hero_png, capsys):
    """表の書き間違いは変換の前に止める（何も書かない）。"""
    t = tmp_path / "bad.motion.json"
    t.write_text(json.dumps(_table(version=3)), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main([str(hero_png), "-o", str(tmp_path / "out" / "s.png"), "--motion-spec", str(t)])
    assert e.value.code == 2 and "version は 1" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_write_spec_for_many_inputs(tmp_path, hero_png):
    d = tmp_path / "in"
    d.mkdir()
    for n in ("a", "b"):
        (d / f"{n}.png").write_bytes(hero_png.read_bytes())
    assert cli.main([str(d), "-o", str(tmp_path / "out"), "-q", "--motion", "breathe",
                     "--write-motion-spec", str(tmp_path / "t.json")]) == 0
    assert (tmp_path / "t_a.json").exists() and (tmp_path / "t_b.json").exists()
