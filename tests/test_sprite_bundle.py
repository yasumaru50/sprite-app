"""画面の「まとめて保存（ZIP）」（`retrocore/sprite/bundle.py`）のテスト。

- ZIP の中のファイルは、渡したバイト列とバイト単位で同じ（圧縮しない）
- 中の日時は固定。同じファイルを同じ順に渡せば、同じ ZIP（バイト単位）になる
- 画面の経路（frames_convert_for_web → frames_sheets_for_web → gif_for_web）で作ったファイルを ZIP にすると、
  中身は 1 つずつ保存したもの（= CLI が書くファイル）と同じ
- 画面の HTML に、まとめて保存・折りたたむ欄があり、今までの保存のボタンも全部残っている
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from retrocore.sprite import frames as F
from retrocore.sprite.bundle import FIXED_DATE_TIME, zip_files, zip_for_web
from test_sprite_frames import _cli_files, _text_bytes, make_strip

ROOT = Path(__file__).resolve().parents[1]


def _members(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        return {i.filename: (z.read(i), i) for i in z.infolist()}


def test_zip_keeps_bytes_and_fixed_time():
    files = [("a.png", b"\x89PNG\r\n\x1a\n" + bytes(range(256))), ("a.json", b'{"x": 1}\n'), ("a.gif", b"GIF89a")]
    data = zip_files(files)
    got = _members(data)
    assert list(got) == ["a.png", "a.json", "a.gif"]   # 渡した順のまま
    for name, raw in files:
        body, info = got[name]
        assert body == raw
        assert info.date_time == FIXED_DATE_TIME
        assert info.compress_type == zipfile.ZIP_STORED
        assert (info.external_attr >> 16) & 0o777 == 0o644
    assert zip_files(files) == data   # 同じ入力なら同じ ZIP


def test_zip_skips_second_same_name_and_rejects_bad_names():
    data = zip_files([("a.png", b"1"), ("a.png", b"2")])
    assert {k: v[0] for k, v in _members(data).items()} == {"a.png": b"1"}
    for bad in ("", "dir/a.png", "..\\a.png", ".."):
        with pytest.raises(ValueError):
            zip_files([(bad, b"x")])


def test_zip_for_web_result():
    res = json.loads(zip_for_web([b"1", b"22", b"1"], ["キャラ_sprite.png", "b.json", "キャラ_sprite.png"]))
    assert res["ok"] and res["names"] == ["キャラ_sprite.png", "b.json"]
    got = _members(base64.b64decode(res["zip_base64"]))
    assert {k: v[0] for k, v in got.items()} == {"キャラ_sprite.png": b"1", "b.json": b"22"}
    bad = json.loads(zip_for_web([b"1"], ["a", "b"]))
    assert not bad["ok"] and "ZIP" in bad["error"]


def test_web_zip_members_equal_each_saved_file(tmp_path):
    """画面と同じ呼び方で作ったシート・JSON・GIF（左右反転も）を ZIP にすると、1 つずつ保存したもの・CLI のファイルと同じ。"""
    strip = make_strip(3)
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web, gif_for_web
    cli = _cli_files(tmp_path, strip, ["--frame-count", "3", "--frame-name", "walk"])
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}, "name": "walk"})))
    assert conv["ok"] and conv["made"], conv
    each = {"hero_sprite.png": base64.b64decode(conv["png_base64"])}   # 1 つずつ保存（画面のリンクが指すバイト列）
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite",
                                                          "mirror": True, "name": "walk"})))
    assert sheets["ok"], sheets
    for s in sheets["sheets"]:
        png = base64.b64decode(s["png_base64"])
        each[s["name"]] = png
        each[s["name"][:-4] + ".json"] = s["json"].encode("utf-8")
        g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
        assert g["ok"], g
        each[g["name"]] = base64.b64decode(g["gif_base64"])
    names = list(each)
    res = json.loads(zip_for_web([each[n] for n in names], names))
    assert res["ok"] and res["names"] == names
    data = base64.b64decode(res["zip_base64"])
    got = {k: v[0] for k, v in _members(data).items()}
    assert got == each
    # 1 つずつ保存したもの（画面は改行 LF のまま）は CLI のファイルと同じ（ちらつきの見える化は ZIP に入れない）
    for k, v in got.items():
        assert (_text_bytes(v.decode("utf-8")) if k.endswith(".json") else v) == cli[k], k
    assert {k for k in cli if "_flicker" not in k} == set(got)
    # 同じ入力なら同じ ZIP
    again = json.loads(zip_for_web([each[n] for n in names], names))
    assert base64.b64decode(again["zip_base64"]) == data
    assert F.DEFAULT_NAME   # （名前の既定は frames のもの。ここでは walk を使った）


def test_web_html_has_layout_and_every_save_button():
    html = (ROOT / "apps" / "sprite" / "web" / "sprite-app.html").read_text(encoding="utf-8")
    for s in ('id="dl-zip"', "まとめて保存（ZIP）", 'id="fold-need"', "必要なときに開く", 'id="fold-fine"', "細かい設定",
              'id="fold-editor"', "外のエディタで直す", 'id="fold-each"', "1 つずつ保存", "（変更あり）", "sprite_app_zip",
              "from retrocore.sprite.bundle import zip_for_web"):
        assert s in html, s
    # 今までの保存のボタン・読み戻しは 1 つも消さない
    for i in ("download", "dl-sheet", "dl-json", "dl-gif", "dl-piskel", "dl-anims-config", "dl-mirror", "dl-mirror-json",
              "dl-mirror-gif", "dl-mirror-piskel", "dl-flicker", "dl-flicker-json", "dl-fx", "dl-fx-json", "dl-fx-gif",
              "dl-fx-piskel", "dl-fx-mirror", "dl-fx-mirror-json", "dl-fx-mirror-gif", "dl-fx-mirror-piskel",
              "dl-talk-sheet", "dl-talk-json", "dl-talk-gif", "dl-talk-piskel", "dl-talk-mirror", "dl-talk-mirror-json",
              "dl-talk-mirror-gif", "dl-talk-mirror-piskel", "dl-same", "dl-skipped", "touch-save", "swap-save",
              "table-save", "edit-file", "edit-json", "edit-orig"):
        assert html.count(f'id="{i}"') == 1, i
    # 折りたたむ欄は最初は閉じている
    for i in ("fold-need", "fold-fine", "fold-editor", "fold-each"):
        tag = html[html.index(f'id="{i}"') - 40: html.index(f'id="{i}"') + 20]
        assert " open" not in tag, i
