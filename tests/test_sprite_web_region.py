"""ブラウザ版の「表で編集」で、動かす範囲（帯・四角）を画面で作ったときのテスト。

`tests/web_region/` の中身は、ヘッドレスの Chromium で実際の画面（apps/sprite/web/sprite-app.html）を動かして保存したもの
（作り方は `tests/web_region/README.md`）。合成の騎士（`make_knight()` を 20 倍に拡大した画像）を画面で変換し、

- `web_rect`: 呼吸の表を開き、絵の上でドラッグして頭の四角（左 5〜右 24・下 10〜上 22）を描き、upper_y をその四角で動かした（表示時間 600・200）
- `web_band`: さらに「＋ 帯」で帯を足し、入力欄で端を下 10〜上 22 にし、名前を head_band に変え、列の見出しで upper_y を帯にし、
  3 行目に root_y −1 を足した（四角 head は使わないまま残る）

確かめること:
1. 画面で保存した表（数・割合）が、同じ内容を手で書いた表と同じ（`used` を除く）
2. 画面で保存した表が、手で書いた表を CLI の `--write-motion-spec --spec-height number|percent` に通したものとバイト単位で同じ
3. 画面で保存した表を CLI の `--motion-spec` に通すと、画面のシートと同じシートになる（PNG は画素、JSON はバイト）
4. 手で書いた表を画面の呼び方（`motion_for_web()`）に渡しても、同じシート
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from legacy_output import legacy_sheet_text
from motion_records import make_knight
from retrocore.sprite.convert import motion_for_web
from retrocore.sprite.image import load_rgba
from retrocore.sprite.motion import parse_spec, spec_json
from retrocore.sprite.motion.spec import spec_to_dict

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "tests" / "web_region"

ROOT_Y = {"type": "root", "axis": "y"}
HAND = {
    "web_rect": {
        "format": "sprite-motion", "version": 1, "name": "web_rect", "loop": True,
        "regions": {"head": {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22}},
        "tracks": {"upper_y": {"type": "deform", "region": "head", "axis": "y"}},
        "frames": [{"ms": 600}, {"ms": 200, "upper_y": -1}],
    },
    "web_band": {
        "format": "sprite-motion", "version": 1, "name": "web_band", "loop": True,
        "regions": {"head": {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22},
                    "head_band": {"type": "band", "bottom": 10, "top": 22}},
        "tracks": {"upper_y": {"type": "deform", "region": "head_band", "axis": "y"}, "root_y": ROOT_Y},
        "frames": [{"ms": 600}, {"ms": 200, "upper_y": -1}, {"ms": 200, "root_y": -1}],
    },
}


def _without_used(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    for r in d.get("regions", {}).values():
        r.pop("used", None)
    return d


def _text(path: Path) -> str:
    return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


@pytest.fixture(scope="module")
def knight_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("in") / "knight.png"
    Image.fromarray(np.kron(make_knight(), np.ones((20, 20, 1), np.uint8)), "RGBA").save(path)
    return path


@pytest.fixture(scope="module")
def cli_out(knight_png, tmp_path_factory):
    """手で書いた表と、画面で保存した表を CLI に通した結果（名前 → フォルダ）。画面の出力名（knight_sprite）にそろえる。"""
    base = tmp_path_factory.mktemp("cli")
    out = {}
    for name, table in HAND.items():
        hand = base / f"{name}.hand.motion.json"
        hand.write_text(spec_json(table), encoding="utf-8")
        for style in ("number", "percent"):
            d = base / f"hand_{name}_{style}"
            code = cli.main([str(knight_png), "-o", str(d / "knight_sprite.png"), "--motion-spec", str(hand),
                             "--write-motion-spec", str(d / "w.motion.json"), "--spec-height", style, "--quiet"])
            assert code == 0
            out[(name, "hand", style)] = d
        d = base / f"saved_{name}"
        code = cli.main([str(knight_png), "-o", str(d / "knight_sprite.png"),
                         "--motion-spec", str(SAVED / f"{name}.number.motion.json"), "--quiet"])
        assert code == 0
        out[(name, "saved")] = d
    return out


@pytest.mark.parametrize("name", sorted(HAND))
@pytest.mark.parametrize("style", ["number", "percent"])
def test_saved_table_is_the_hand_written_table(name, style, knight_png):
    """画面で保存した表を読むと、手で書いた表と同じ（数の表はそのまま、割合の表は数に戻すと同じ）。"""
    from retrocore.sprite.motion import with_height_style
    saved = json.loads(_text(SAVED / f"{name}.{style}.motion.json"))
    sprite = load_rgba(SAVED / f"{name}_sheet.png")[:, :24]   # 1 枚目（変換結果のまま）
    got = _without_used(spec_to_dict(parse_spec(saved)))
    if style == "percent":
        assert any(isinstance(v, str) for r in got["regions"].values() for v in r.values() if v not in ("rect", "band"))
        got = _without_used(with_height_style(got, sprite, "number"))
    assert got == spec_to_dict(parse_spec(HAND[name]))


@pytest.mark.parametrize("name", sorted(HAND))
@pytest.mark.parametrize("style", ["number", "percent"])
def test_saved_table_equals_cli_write_motion_spec(name, style, cli_out):
    assert _text(SAVED / f"{name}.{style}.motion.json") == _text(cli_out[(name, "hand", style)] / "w.motion.json")


@pytest.mark.parametrize("name", sorted(HAND))
@pytest.mark.parametrize("how", ["hand", "saved"])
def test_cli_gives_the_screen_sheet(name, how, cli_out):
    """画面のシート ＝ 手で書いた表の CLI のシート ＝ 画面で保存した表を CLI に通したシート。"""
    d = cli_out[(name, "hand", "number")] if how == "hand" else cli_out[(name, "saved")]
    sheet = d / f"knight_sprite_{name}_sheet.png"
    assert np.array_equal(_pixels(sheet.read_bytes()), _pixels((SAVED / f"{name}_sheet.png").read_bytes()))
    # 画面で保存した JSON は版 1（作らなかった理由の記録 skipped を足す前のビルドで保存）。足した分を除くと同じ
    assert legacy_sheet_text(_text(d / f"knight_sprite_{name}_sheet.json")) == _text(SAVED / f"{name}_sheet.json")


@pytest.mark.parametrize("name", sorted(HAND))
def test_web_call_with_hand_table_gives_the_screen_sheet(name, cli_out):
    sprite_png = (cli_out[(name, "hand", "number")] / "knight_sprite.png").read_bytes()
    res = json.loads(motion_for_web(sprite_png, json.dumps({"spec": HAND[name], "image_name": f"knight_sprite_{name}_sheet.png"})))
    assert res["ok"] and not res["skipped"], res
    assert np.array_equal(_pixels(base64.b64decode(res["sheet_png_base64"])), _pixels((SAVED / f"{name}_sheet.png").read_bytes()))
    assert legacy_sheet_text(res["sheet_json"]) == _text(SAVED / f"{name}_sheet.json")
    assert res["spec_files"]["number"] == _text(SAVED / f"{name}.number.motion.json")


def test_band_sheet_keeps_outside_still():
    """画面で作った帯の表: 帯（下 10〜上 22）より下の行は、縮めたフレームでも動かない（剣も動かない）。"""
    sheet = load_rgba(SAVED / "web_band_sheet.png")
    info = json.loads(_text(SAVED / "web_band_sheet.json"))
    w, h = info["frame_width"], info["frame_height"]
    f0, f1 = sheet[:h, :w], sheet[:h, w:2 * w]
    below = h - 9   # 下から 9 行（帯より下）
    assert np.array_equal(f0[below:], f1[below:])
    assert not np.array_equal(f0, f1)


def test_web_html_has_region_editor():
    html = (ROOT / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    for words in ("動かす範囲", "＋ 四角", "＋ 帯", "＋ 分け目", "部品を", "足元（動かせない）", "region-canvas",
                  "dataset.colRegion", "をこの範囲", "案 ${s.n}", "edge_number", "範囲の中を縮める・伸ばす"):
        assert words in html, words
