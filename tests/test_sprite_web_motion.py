"""ブラウザ版の「動き」（呼吸・横揺れ・浮遊の再生とスプライトシートの保存）のテスト。

画面は Pyodide の中で `convert_for_web()`（変換）→ `motion_for_web()`（変換の結果の PNG から動きを作る）の順に呼ぶ。
ここでは同じ呼び方で作ったシートの PNG と JSON が、同じ設定の CLI（`--motion breathe ...` / `--motion sway ...`）の出力と
バイト単位で同じことと、
呼吸を付けないときの結果が今までと同じことを確かめる。HTML が古くなっていないかは test_sprite_convert.py で確かめる。
"""

import base64
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite.convert import convert_for_web, motion_for_web, motion_image_bytes, motion_spec
from retrocore.sprite.motion import breathe as breathe_mod
from retrocore.sprite.motion import floating as float_mod
from retrocore.sprite.motion import sway as sway_mod
from synth_sprite import enlarge_like_ai, make_pixel_sprite

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory) -> Path:
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture(scope="module")
def sprite_png(pixel_png) -> bytes:
    """画面と同じ呼び方で変換した結果（PNG のバイト列）。画面はこれを動きの材料として渡す。"""
    res = json.loads(convert_for_web(pixel_png.read_bytes(), "{}"))
    assert res["ok"], res
    return base64.b64decode(res["png_base64"])


def _web_motion(sprite_png: bytes, options: dict) -> dict:
    res = json.loads(motion_for_web(sprite_png, json.dumps(options)))
    assert res["ok"], res
    return res


def _text_file_bytes(text: str) -> bytes:
    """CLI の write_text() と同じ（Windows では改行が CRLF になる）。画面は LF のまま保存する"""
    return text.replace("\n", os.linesep).encode("utf-8")


# 画面の設定（JSON で送るもの）と、同じ意味の CLI の引数
CASES = [
    ("breathe", {}, []),
    ("breathe", {"split": "auto"}, ["--breathe-split", "auto"]),
    ("breathe", {"split": 12}, ["--breathe-split", "12"]),
    ("breathe", {"frames": 4, "depth": 2, "durations": 250},
     ["--motion-frames", "4", "--breathe-depth", "2", "--motion-duration", "250"]),
    ("breathe", {"frames": 2, "durations": [600, 400]}, ["--motion-duration", "600,400"]),
    ("breathe", {"frames": 8, "split": 8}, ["--motion-frames", "8", "--breathe-split", "8"]),
    # 横揺れ（画面は「なし／呼吸／横揺れ」で横揺れを選んだとき、この形で送る）
    ("sway", {}, []),
    ("sway", {"split": "auto"}, ["--sway-split", "auto"]),
    ("sway", {"split": 10}, ["--sway-split", "10"]),
    ("sway", {"frames": 8, "amount": 2, "durations": 150},
     ["--motion-frames", "8", "--sway-amount", "2", "--motion-duration", "150"]),
    ("sway", {"frames": 3, "durations": [400, 200, 200]}, ["--motion-frames", "3", "--motion-duration", "400,200,200"]),
    ("sway", {"frames": 6, "split": 14}, ["--motion-frames", "6", "--sway-split", "14"]),
    # 浮遊（キャラ全体が上下。分ける高さは無いので送らない）
    ("float", {}, []),
    ("float", {"amount": 2}, ["--float-amount", "2"]),
    ("float", {"frames": 8, "amount": 2, "durations": 200},
     ["--motion-frames", "8", "--float-amount", "2", "--motion-duration", "200"]),
    ("float", {"frames": 3, "durations": [500, 300, 300]}, ["--motion-frames", "3", "--motion-duration", "500,300,300"]),
]


@pytest.mark.parametrize("name, options, args", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_web_sheet_equals_cli_sheet(pixel_png, sprite_png, tmp_path, name, options, args):
    """画面と同じ呼び方で作ったシートの PNG と JSON が、同じ設定の CLI の出力とバイト単位で同じ。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", name, *args]) == 0
    assert dst.read_bytes() == sprite_png       # 材料（変換の結果）も同じ
    sheet = tmp_path / f"hero_sprite_{name}_sheet.png"
    res = _web_motion(sprite_png, {"motion": name, **options, "image_name": sheet.name})
    assert res["skipped"] is False and res["warnings"] == []
    assert base64.b64decode(res["sheet_png_base64"]) == sheet.read_bytes()
    assert _text_file_bytes(res["sheet_json"]) == sheet.with_suffix(".json").read_bytes()
    info = json.loads(res["sheet_json"])
    assert info["image"] == sheet.name
    assert res["durations"] == [f["duration_ms"] for f in info["frames"]]   # 画面はこの値どおりに再生する
    assert (res["frame_width"], res["frame_height"]) == (info["frame_width"], info["frame_height"])


def test_skipped_matches_cli_warning(pixel_png, sprite_png, tmp_path, capsys):
    """付けられない高さ（足元）では、画面にも CLI と同じ警告の文言を返し、シートは作らない（CLI も書かない）。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "breathe", "--breathe-split", "1"]) == 0
    err = capsys.readouterr().err
    assert not (tmp_path / "hero_sprite_breathe_sheet.png").exists()
    res = _web_motion(sprite_png, {"split": 1})
    assert res["skipped"] is True and "sheet_png_base64" not in res and "sheet_json" not in res
    assert res["warnings"] and all(f"警告（{pixel_png.name}）: {w}" in err for w in res["warnings"])
    assert res["split_limits"] is not None and res["height"] == Image.open(dst).height


def test_out_of_image_split_is_error_like_cli(pixel_png, sprite_png, tmp_path, capsys):
    """画像の外の高さはエラー（CLI は「エラー（ファイル名）: …」）。画面には同じ文言を出す。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "breathe", "--breathe-split", "999"]) == 1
    err = capsys.readouterr().err
    res = json.loads(motion_for_web(sprite_png, json.dumps({"split": 999})))
    assert res["ok"] is False and "Traceback" in res["detail"]
    message = err.split(f"エラー（{pixel_png.name}）: ", 1)[1].strip()
    assert message in res["error"]


@pytest.mark.parametrize("options, words", [
    ({"frames": 1}, ["フレーム数は 2〜32"]),
    ({"frames": True}, ["フレーム数"]),
    ({"depth": 3}, ["下がる量は 1〜2"]),
    ({"durations": 0}, ["表示時間"]),
    ({"durations": [500, 500, 500]}, ["表示時間の数（3）はフレーム数（2）"]),
    ({"durations": "500"}, ["表示時間"]),
    ({"split": 0}, ["分ける高さ"]),
    ({"split": "middle"}, ["分ける高さ"]),
    ({"motion": "wobble"}, ["動きは breathe / sway"]),
    ({"speed": 2}, ["知らない動きの設定です: speed"]),
    ({"motion": "sway", "frames": 2}, ["フレーム数は 3〜32"]),
    ({"motion": "sway", "amount": 3}, ["揺れ幅は 1〜2"]),
    ({"motion": "sway", "amount": 1.5}, ["揺れ幅"]),
    ({"motion": "sway", "depth": 1}, ["depth は sway の設定ではありません"]),
    ({"motion": "breathe", "amount": 1}, ["amount は breathe の設定ではありません"]),
    ({"motion": "sway", "durations": [300, 300]}, ["表示時間の数（2）はフレーム数（4）"]),
    ({"motion": "sway", "split": 0}, ["分ける高さ"]),
    ({"motion": "float", "frames": 2}, ["フレーム数は 3〜32"]),
    ({"motion": "float", "amount": 3}, ["揺れ幅は 1〜2"]),
    ({"motion": "float", "split": 10}, ["split は float の設定ではありません"]),
    ({"motion": "float", "split": "auto"}, ["split は float の設定ではありません"]),
    ({"motion": "float", "depth": 1}, ["depth は float の設定ではありません"]),
    ({"motion": "float", "durations": [400]*3}, ["表示時間の数（3）はフレーム数（4）"]),
])
def test_bad_motion_settings_give_messages(sprite_png, options, words):
    res = json.loads(motion_for_web(sprite_png, json.dumps(options)))
    assert res["ok"] is False
    for w in words:
        assert w in res["error"]


def test_bad_json_and_bad_image():
    for text in ("{", "[1]"):
        res = json.loads(motion_for_web(b"", text))
        assert res["ok"] is False and "動きの設定" in res["error"]
    res = motion_image_bytes(b"not an image", {})
    assert res["ok"] is False and "画像として読み込めません" in res["error"]


def test_sway_skipped_and_error_match_cli(pixel_png, sprite_png, tmp_path, capsys):
    """横揺れでも、付けられない高さの警告と、画像の外の高さのエラーは CLI と同じ文言。"""
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "sway", "--sway-split", "1"]) == 0
    err = capsys.readouterr().err
    assert not (tmp_path / "hero_sprite_sway_sheet.png").exists()
    res = _web_motion(sprite_png, {"motion": "sway", "split": 1})
    assert res["skipped"] is True and "sheet_png_base64" not in res
    assert res["warnings"] and all(f"警告（{pixel_png.name}）: {w}" in err for w in res["warnings"])
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "sway", "--sway-split", "999"]) == 1
    message = capsys.readouterr().err.split(f"エラー（{pixel_png.name}）: ", 1)[1].strip()
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "sway", "split": 999})))
    assert res["ok"] is False and message in res["error"]


def test_sway_result_fields(sprite_png):
    """画面が使う横揺れの情報（分け目の高さ・余白）と、JSON の padding。呼吸の返す値には足さない（前の版と同じ）。"""
    res = _web_motion(sprite_png, {"motion": "sway"})
    info = json.loads(res["sheet_json"])
    assert res["padding"] == info["padding"]["left"] and len(res["split_from_bottom"]) == 1
    assert res["frame_width"] == info["frame_width"]
    br = _web_motion(sprite_png, {"motion": "breathe"})
    assert "padding" not in br and "split_from_bottom" not in br and "padding" not in json.loads(br["sheet_json"])


def test_float_result_fields(sprite_png):
    """画面が使う浮遊の情報（上下の余白・各フレームのずれ）が JSON と同じ。呼吸・横揺れの返す値には足さない。"""
    res = _web_motion(sprite_png, {"motion": "float"})
    info = json.loads(res["sheet_json"])
    assert res["padding_y"] == [info["padding"]["top"], info["padding"]["bottom"]] == [1, 1]
    assert res["offset_y"] == [f["offset_y"] for f in info["frames"]] == [0, -1, 0, 1]
    assert res["frame_height"] == info["frame_height"] == res["height"] + 2
    assert "split_from_bottom" not in res and "padding" not in res
    for name in ("breathe", "sway"):
        other = _web_motion(sprite_png, {"motion": name})
        assert "padding_y" not in other and "offset_y" not in other


def test_motion_off_result_unchanged(pixel_png, sprite_png, tmp_path):
    """呼吸を付けないとき: 画面が送る設定（色数・クリーンアップ・縁取り）と返す結果は今までと同じ形で、CLI の出力と同じ。

    動きは変換とは別の呼び出しなので、変換の結果に動きの情報は入らない。
    """
    dst = tmp_path / "hero_sprite.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q"]) == 0
    res = json.loads(convert_for_web(pixel_png.read_bytes(), json.dumps({"colors": 12, "cleanup": "normal", "outline": "auto"})))
    # "skipped" は作らなかった理由の記録（変換の工程の分。パレットを使わなければ空）
    assert set(res) == {"ok", "png_base64", "width", "height", "colors", "steps", "warnings", "skipped"}
    assert res["skipped"] == []
    assert base64.b64decode(res["png_base64"]) == dst.read_bytes() == sprite_png
    assert not list(tmp_path.glob("*_sheet.*"))


def test_motion_spec_matches_cli_defaults():
    spec = motion_spec()
    assert spec["frames"] == {"default": breathe_mod.DEFAULT_FRAMES, "min": 2, "max": breathe_mod.MAX_FRAMES}
    assert spec["duration"]["default"] == breathe_mod.DEFAULT_DURATION == 500
    assert spec["depth"] == {"default": 1, "min": 1, "max": breathe_mod.MAX_DEPTH}
    assert spec["split"]["default"] == breathe_mod.AUTO
    args = cli.build_parser().parse_args(["x.png", "--motion", "breathe"])
    assert args.motion_frames is None and args.breathe_depth is None and args.breathe_split is None  # CLI も既定（上と同じ値）
    sw = spec["sway"]
    assert sw["frames"] == {"default": sway_mod.DEFAULT_FRAMES, "min": 3, "max": 32}
    assert sw["duration"]["default"] == sway_mod.DEFAULT_DURATION == 300
    assert sw["amount"] == {"default": 1, "min": 1, "max": sway_mod.MAX_AMOUNT}
    assert sw["split"]["default"] == "auto"
    assert [c[0] for c in spec["choices"]] == ["breathe", "sway", "float", "bounce", "nod", "stretch"]
    fl = spec["float"]
    assert fl["frames"] == {"default": float_mod.DEFAULT_FRAMES, "min": 3, "max": 32} and fl["frames"]["default"] == 4
    assert fl["duration"]["default"] == float_mod.DEFAULT_DURATION == 400
    assert fl["amount"] == {"default": 1, "min": 1, "max": 2} and "split" not in fl


def test_web_html_has_motion_ui():
    """sprite-app.html（作り直したもの）に動きの欄（なし／呼吸／横揺れ）が入り、Python の motion_for_web を呼ぶ。"""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('name="motion"', 'id="motion-none"', 'id="amount"', 'id="split-mode"', 'id="show-rows"', 'id="dl-sheet"',
                'id="dl-json"', "motion_for_web", 'id="split-field"', "_float_sheet.png"):
        assert key in html
    assert build_sprite_web.settings_spec()["motion"] == motion_spec()
