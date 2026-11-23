"""浮遊（`retrocore/sprite/motion/floating.py`、CLI `--motion float`）のテスト。

浮遊は「足元が動く」はじめての動き（決まりの例外。docs/sprite-app-plan.md の「浮遊」）。キャラクターは分けずに丸ごと上下に動かす。
足元が縦に決まった量だけ動くこと以外は、今までの決まり（新しい色を作らない・全フレームの大きさが同じ・隙間を作らない・
輪郭を途切れさせない・迷ったら動かさない）を守ることを確かめる。
あわせて、--motion なし・呼吸・横揺れの CLI の出力が、この PR の前（PR #24 の main）と同じことをハッシュで確かめる。
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.motion import MOTIONS, check_rules, foot_center, make_motion, save_sheet, sheet_info
from retrocore.sprite.motion.base import color_keys, shape_signature
from retrocore.sprite.motion.floating import (DEFAULT_AMOUNT, DEFAULT_DURATION, DEFAULT_FRAMES, MOVES_FOOT,
                                              float_motion, float_offsets, shift_rows)
from retrocore.sprite.motion.sway import euler_numbers
from legacy_output import legacy_bytes, legacy_files
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime
from test_sprite_sway import diamond, sash


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "converted": converted,
            "diamond": diamond(), "sash": sash()}


NAMES = ["mushroom", "slime", "converted", "diamond", "sash"]


def _check_float(base: np.ndarray, res) -> None:
    """浮遊の結果が決まり（足元の例外のほかは今までどおり）を守っていることを、check_rules() とは別の書き方でも確かめる。"""
    assert not res.skipped, res.warnings
    top, bottom = res.info["padding_y"]
    padded = np.pad(base, ((top, bottom), (0, 0), (0, 0)))
    assert np.array_equal(res.frames[0], padded)                       # 1 枚目は元のまま（余白を足しただけ）
    assert check_rules(padded, res.frames, shifts=res.info["offsets"]) == []
    base_colors = set(np.unique(color_keys(base)).tolist())
    fx, fy = foot_center(padded)
    opaque = int((base[..., 3] > 0).sum())
    for k, f in zip(res.info["offsets"], res.frames):
        assert f.shape == padded.shape and f.dtype == np.uint8          # 全フレームの大きさが同じ
        assert set(np.unique(color_keys(f)).tolist()) <= base_colors   # 新しい色が無い
        assert foot_center(f) == (fx, fy + k)                          # 例外: 足元は縦にちょうど k だけ動く（横は動かない）
        assert shape_signature(f) == shape_signature(padded)           # 部品・輪郭のつながりが同じ
        assert euler_numbers(f) == euler_numbers(padded)               # 穴の数も同じ（隙間ができない・輪郭が途切れない）
        assert int((f[..., 3] > 0).sum()) == opaque                    # はみ出して欠けたドットが無い
        rows = np.flatnonzero((f[..., 3] > 0).any(axis=1))
        assert rows[0] >= 0 and rows[-1] < f.shape[0]
        # 形はまったく同じで、位置だけが縦に k ずれている
        assert np.array_equal(f[max(0, k):f.shape[0] + min(0, k)], padded[max(0, -k):padded.shape[0] - max(0, k)])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("frames, amount", [(4, 1), (8, 2), (3, 1), (6, 2), (32, 1)])
def test_float_keeps_rules(sprites, name, frames, amount):
    """どんな形でも（横揺れでは揺らせないひし形・たすきの絵も）、丸ごと動かすだけなので決まりを守って付く。"""
    base = sprites[name]
    res = float_motion(base, frames=frames, amount=amount)
    _check_float(base, res)
    assert len(res.frames) == frames and res.warnings == []


def test_float_offsets():
    assert float_offsets() == [0, -1, 0, 1]                           # 中央 → 上 → 中央 → 下（− が上）
    assert float_offsets(8, 2) == [0, -1, -2, -1, 0, 1, 2, 1]
    assert float_offsets(8, 1) == [0, -1, -1, -1, 0, 1, 1, 1]
    assert float_offsets(3, 1) == [0, -1, 1]
    assert (DEFAULT_FRAMES, DEFAULT_DURATION, DEFAULT_AMOUNT, MOVES_FOOT) == (4, 400, 1, True)


def test_float_padding_only_where_needed(sprites):
    """変換の結果（上下が切り詰めてある）には上下に揺れ幅の分の行を足す。上下に透明の行が余っていれば、その分は足さない。"""
    tight = sprites["converted"]
    res = float_motion(tight)
    assert res.info["padding_y"] == [1, 1] and res.size == (tight.shape[1], tight.shape[0] + 2)
    res2 = float_motion(tight, frames=8, amount=2)
    assert res2.info["padding_y"] == [2, 2] and res2.size == (tight.shape[1], tight.shape[0] + 4)
    roomy = np.pad(tight, ((1, 3), (0, 0), (0, 0)))
    res3 = float_motion(roomy, frames=8, amount=2)
    assert res3.info["padding_y"] == [1, 0] and res3.size == (roomy.shape[1], roomy.shape[0] + 1)
    _check_float(roomy, res3)


def test_shift_rows():
    img = np.zeros((4, 2, 4), np.uint8)
    img[1, 0] = (1, 2, 3, 255)
    assert shift_rows(img, -1)[0, 0].tolist() == [1, 2, 3, 255] and shift_rows(img, -1)[1, 0, 3] == 0
    assert shift_rows(img, 2)[3, 0].tolist() == [1, 2, 3, 255]
    assert np.array_equal(shift_rows(img, 0), img)


def test_float_skips_empty_image():
    res = float_motion(np.zeros((10, 10, 4), np.uint8))
    assert res.skipped and "浮遊を付けませんでした" in res.warnings[0] and len(res.frames) == 1


def test_float_defaults_and_durations(sprites):
    res = make_motion(sprites["slime"], "float")
    assert res.name == "float" and len(res.frames) == 4 and res.durations == [400] * 4
    res = make_motion(sprites["slime"], "float", frames=3, durations=[500, 200, 200])
    assert res.durations == [500, 200, 200]
    assert "float" in MOTIONS


@pytest.mark.parametrize("kwargs, words", [
    ({"frames": 2}, "3〜32"), ({"frames": 33}, "2〜32"), ({"amount": 3}, "1〜2"), ({"amount": 0}, "1〜2"),
    ({"amount": True}, "揺れ幅"), ({"durations": [400, 400]}, "表示時間の数"), ({"durations": 0}, "1 ミリ秒以上"),
])
def test_float_rejects_bad_settings(sprites, kwargs, words):
    with pytest.raises(ValueError, match=words):
        float_motion(sprites["slime"], **kwargs)


def test_float_has_no_split_option(sprites):
    """浮遊は分けないので、分ける高さの指定は無い（間違えて渡したらエラー）。"""
    with pytest.raises(TypeError):
        make_motion(sprites["slime"], "float", split=10)


def test_check_rules_with_shifts_finds_problems(sprites):
    """足元の例外（shifts）でも、縦に決まった量ずらしただけでないフレームは見つける。省いたときは今までどおり足元が動けばだめ。"""
    base = np.pad(sprites["mushroom"], ((1, 1), (0, 0), (0, 0)))
    up = shift_rows(base, -1)
    assert check_rules(base, [base, up], shifts=[0, -1]) == []
    assert any("足元の位置が変わりました" in p for p in check_rules(base, [base, up]))       # 例外なし（呼吸・横揺れ）
    assert any("決まった量" in p for p in check_rules(base, [base, up], shifts=[0, 1]))       # 向きが違う
    side = np.roll(up, 1, axis=1)
    assert check_rules(base, [side], shifts=[-1])                                           # 横にも動いた
    broken = up.copy()
    ys, xs = np.nonzero(broken[..., 3] > 0)
    broken[ys[len(ys) // 2], xs[len(xs) // 2]] = (1, 2, 3, 255)
    assert any("新しい" in p or "無い色" in p for p in check_rules(base, [broken], shifts=[-1]))
    assert any("ずらしただけ" in p for p in check_rules(base, [broken], shifts=[-1]))
    tight = sprites["mushroom"]
    cut = shift_rows(tight, -1)                                                             # 上の行が欠ける
    assert any("ずらしただけ" in p for p in check_rules(tight, [cut], shifts=[-1]))
    assert check_rules(base, [base], shifts=[0, 1])                                         # 数が合わない


def test_float_is_deterministic(sprites):
    a, b = float_motion(sprites["converted"]), float_motion(sprites["converted"])
    assert a.info == b.info and all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


# ---- シートと JSON ----

def test_float_sheet_json(tmp_path, sprites):
    """JSON には上下の余白（padding）と、フレームごとの上下のずれ（offset_y。下向きが +）を書く。pivot は 1 枚目の足元。"""
    sprite = sprites["converted"]
    res = float_motion(sprite, frames=8, amount=2)
    png, js = save_sheet(res, tmp_path / "s_float_sheet.png")
    info = json.loads(js.read_text(encoding="utf-8"))
    h0, w0 = sprite.shape[:2]
    assert info["motion"] == "float" and info["image"] == "s_float_sheet.png" and info["frame_count"] == 8
    assert info["padding"] == {"left": 0, "right": 0, "top": 2, "bottom": 2}
    assert info["frame_width"] == w0 and info["frame_height"] == h0 + 4
    assert [f["offset_y"] for f in info["frames"]] == [0, -1, -2, -1, 0, 1, 2, 1]
    assert info["pivot"] == {"x": w0 / 2, "y": h0 + 2}                 # 1 枚目の足元（元の絵の下端）。下の余白 2 行の上
    # 1 枚目の絵のいちばん下の行は pivot.y のすぐ上、各フレームでは offset_y だけずれる
    sheet = np.asarray(Image.open(png).convert("RGBA"))
    fw = info["frame_width"]
    for f in info["frames"]:
        cell = sheet[:, f["x"]:f["x"] + fw]
        bottom_row = int(np.flatnonzero((cell[..., 3] > 0).any(axis=1))[-1])
        assert bottom_row == info["pivot"]["y"] - 1 + f["offset_y"]
    assert np.array_equal(sheet, np.concatenate(res.frames, axis=1))


def test_other_motions_json_has_no_offset_y(sprites):
    """呼吸・横揺れの JSON には offset_y を書かない（前と同じ）。"""
    for name in ("breathe", "sway"):
        info = sheet_info(make_motion(sprites["mushroom"], name), "x.png")
        assert all("offset_y" not in f for f in info["frames"])
        assert info["pivot"]["y"] == info["frame_height"]


# ---- CLI ----

def _ai_input(tmp_path):
    path = tmp_path / "in" / "chara.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_float(tmp_path, capsys):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "--motion", "float"]) == 0
    text = capsys.readouterr().out
    assert "chara_float_sheet.png" in text and "上下のずれ +0/-1/+0/+1" in text
    assert sorted(p.name for p in out.parent.iterdir()) == ["chara.png", "chara_float_sheet.json", "chara_float_sheet.png"]
    info = json.loads((out.parent / "chara_float_sheet.json").read_text(encoding="utf-8"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert info["motion"] == "float" and info["frame_count"] == 4
    assert [f["duration_ms"] for f in info["frames"]] == [400] * 4
    assert [f["offset_y"] for f in info["frames"]] == [0, -1, 0, 1]
    assert info["padding"] == {"left": 0, "right": 0, "top": 1, "bottom": 1}
    sheet = np.asarray(Image.open(out.parent / "chara_float_sheet.png").convert("RGBA"))
    assert np.array_equal(sheet, np.concatenate(float_motion(sprite).frames, axis=1))


def test_cli_float_options(tmp_path):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "float", "--motion-frames", "8", "--float-amount", "2",
                     "--motion-duration", "150"]) == 0
    info = json.loads((out.parent / "chara_float_sheet.json").read_text(encoding="utf-8"))
    assert info["frame_count"] == 8 and info["total_duration_ms"] == 1200 and info["padding"]["top"] == 2


@pytest.mark.parametrize("extra, words", [
    (["--float-amount", "2"], "動きのオプション"),
    (["--motion", "float", "--sway-amount", "1"], "--motion float では使えません（横揺れのオプションです）"),
    (["--motion", "float", "--sway-split", "10"], "--motion float では使えません"),
    (["--motion", "float", "--breathe-split", "10"], "--motion float では使えません（呼吸のオプションです）"),
    (["--motion", "float", "--breathe-depth", "1"], "--motion float では使えません"),
    (["--motion", "sway", "--float-amount", "1"], "--motion sway では使えません（浮遊のオプションです）"),
    (["--motion", "breathe", "--float-amount", "1"], "--motion breathe では使えません"),
    (["--motion", "float", "--motion-frames", "2"], "3〜32"),
    (["--motion", "float", "--float-amount", "3"], "1〜2"),
    (["--motion", "float", "--motion-duration", "300,200"], "フレーム数（4）"),
])
def test_cli_rejects_bad_float_options(tmp_path, capsys, extra, words):
    src = _ai_input(tmp_path)
    with pytest.raises(SystemExit):
        cli.main([str(src), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_float_skipped_keeps_sprite(tmp_path, capsys):
    """付けられない絵（空）は警告してシートを書かない（変換の結果はそのまま）。"""
    args = cli.build_parser().parse_args(["x.png", "--motion", "float", "-q"])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, np.zeros((6, 6, 4), np.uint8), dst, dst) == 1
    assert "浮遊を付けませんでした" in capsys.readouterr().err
    assert not (tmp_path / "s_float_sheet.png").exists()


# ---- 今までの結果が変わっていないこと ----

def _digest(path) -> str:
    """ファイルの中身のハッシュ。PNG は画素（大きさと RGBA）で比べる（test_sprite_sway.py と同じ）。JSON は改行を LF にそろえる。"""
    if path.suffix == ".png":
        a = np.asarray(Image.open(path).convert("RGBA"))
        data = repr(a.shape).encode() + a.tobytes()
    else:
        data = legacy_bytes(path.read_bytes().replace(b"\r\n", b"\n"))   # 版 2 で足した skipped を除く
    return hashlib.sha256(data).hexdigest()[:16]


# PR #24 の main（この PR の前）の CLI で作ったファイルのハッシュ（`_digest()`。SHA-256 の先頭 16 文字）。
# 入力は enlarge_like_ai(make_pixel_sprite(), seed=3)。(設定の名前, ファイル名, ハッシュ)
BEFORE = [
    ("none", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe", "hero_sprite_breathe_sheet.json", "736eef558dfcfa5e"),
    ("breathe", "hero_sprite_breathe_sheet.png", "a3bffd484ccafac1"),
    ("breathe_f4d2", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe_f4d2", "hero_sprite_breathe_sheet.json", "7272ab6f606082e3"),
    ("breathe_f4d2", "hero_sprite_breathe_sheet.png", "69ac46a9a88e37aa"),
    ("sway", "hero_sprite.png", "616c81fd24a30eac"),
    ("sway", "hero_sprite_sway_sheet.json", "a8292adc0cbec055"),
    ("sway", "hero_sprite_sway_sheet.png", "30001d5280694ce4"),
    ("sway_f8a2", "hero_sprite.png", "616c81fd24a30eac"),
    ("sway_f8a2", "hero_sprite_sway_sheet.json", "347d8395fa3bd4f7"),
    ("sway_f8a2", "hero_sprite_sway_sheet.png", "a1d4cc536475c913"),
    ("sway_s10", "hero_sprite.png", "616c81fd24a30eac"),
    ("sway_s10", "hero_sprite_sway_sheet.json", "a8292adc0cbec055"),
    ("sway_s10", "hero_sprite_sway_sheet.png", "30001d5280694ce4"),
]
ARGS = {"none": [], "breathe": ["--motion", "breathe"],
        "breathe_f4d2": ["--motion", "breathe", "--motion-frames", "4", "--breathe-depth", "2"],
        "sway": ["--motion", "sway"],
        "sway_f8a2": ["--motion", "sway", "--motion-frames", "8", "--sway-amount", "2"],
        "sway_s10": ["--motion", "sway", "--sway-split", "10"]}


@pytest.mark.parametrize("tag", list(ARGS))
def test_cli_without_motion_breathe_and_sway_are_unchanged(tmp_path, tag):
    """--motion なし・呼吸・横揺れの CLI の出力（ファイルの名前と中身）が、この PR の前とまったく同じ。"""
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    src = tmp_path / "hero.png"
    Image.fromarray(img, "RGBA").save(src)
    out = tmp_path / tag
    assert cli.main([str(src), "-o", str(out / "hero_sprite.png"), "-q", *ARGS[tag]]) == 0
    got = sorted((f.name, _digest(f)) for f in legacy_files(out.iterdir()))
    assert got == sorted((name, h) for t, name, h in BEFORE if t == tag)
