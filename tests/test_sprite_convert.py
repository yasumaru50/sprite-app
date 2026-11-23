"""retrocore/sprite/convert.py（CLI とブラウザ UI で共通の変換の入口）と、ブラウザ版の HTML の組み立てのテスト。"""

import base64
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import (GridNotFoundError, SpriteSettings, convert_sprite, load_palette, load_rgba,
                              save_palette, save_rgba)
from retrocore.sprite.convert import convert_for_web, convert_image_bytes, decode_image, encode_png, friendly_error
from synth_sprite import enlarge_like_ai, make_character, make_pixel_sprite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_sprite_web  # noqa: E402


def _png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory) -> Path:
    """ドット絵風の入力（AI が描いたような、ドット 1 個が約 20px の画像）。"""
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    path = tmp_path_factory.mktemp("in") / "pixel.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


def _cli_output(src: Path, tmp_path: Path, *extra: str) -> bytes:
    dst = tmp_path / f"cli_{len(list(tmp_path.iterdir()))}.png"
    assert cli.main([str(src), "-o", str(dst), "-q", *extra]) == 0
    return dst.read_bytes()


def test_default_settings_match_cli_defaults():
    """SpriteSettings() の既定値は、CLI でオプションを何も指定しないときと同じ。"""
    from_cli = cli.settings_from_args(cli.build_parser().parse_args(["x.png"]))
    assert dataclasses.asdict(from_cli) == dataclasses.asdict(SpriteSettings())


@pytest.mark.parametrize("extra, settings", [
    ((), {}),
    (("--colors", "6"), {"colors": 6}),
    (("--cleanup", "strong", "--outline", "add"), {"cleanup": "strong", "outline": "add"}),
    (("--no-cleanup", "--no-outline"), {"cleanup": "off", "outline": "off"}),
    (("--outline", "thin"), {"outline": "thin"}),
    (("--cleanup", "light", "--outline", "off", "--colors", "16"), {"cleanup": "light", "outline": "off", "colors": 16}),
])
def test_web_result_equals_cli_result(pixel_png, tmp_path, extra, settings):
    """画面と同じ呼び方（PNG のバイト列 + 設定の JSON）の結果が、CLI の出力ファイルとバイト単位で一致する。"""
    expected = _cli_output(pixel_png, tmp_path, *extra)
    res = json.loads(convert_for_web(pixel_png.read_bytes(), json.dumps(settings)))
    assert res["ok"], res
    assert base64.b64decode(res["png_base64"]) == expected
    out = np.asarray(Image.open(io.BytesIO(expected)))
    assert (res["width"], res["height"]) == (out.shape[1], out.shape[0])
    assert res["colors"] == len(np.unique(out[out[..., 3] > 0][:, :3], axis=0))
    assert len(res["steps"]) == 5 - (settings.get("cleanup") == "off") - (settings.get("outline") == "off")
    assert res["warnings"] == []


def test_convert_sprite_equals_cli_for_illust(tmp_path):
    """普通のイラスト（縮小）でも、convert_sprite() と CLI の結果が一致する。"""
    src = tmp_path / "chara.png"
    Image.fromarray(make_character(256, 384), "RGBA").save(src)
    expected = np.asarray(Image.open(io.BytesIO(_cli_output(src, tmp_path, "--input-type", "illust", "--width", "48"))))
    res = convert_sprite(load_rgba(src), SpriteSettings(input_type="illust", width=48))
    assert np.array_equal(res.image, expected)
    assert [s.name for s in res.steps] == ["background", "resize", "simplify", "cleanup", "outline"]


def test_convert_sprite_with_palette_returns_warnings_like_cli(pixel_png, tmp_path, capsys):
    """パレットに合わせたときの警告は result.warnings に入り、CLI と同じものが表示される。"""
    pal = tmp_path / "two.hex"
    save_palette(np.array([[0, 0, 0], [255, 255, 255]]), pal)
    expected = _cli_output(pixel_png, tmp_path, "--palette", str(pal))
    cli_err = capsys.readouterr().err
    res = convert_sprite(load_rgba(pixel_png), SpriteSettings(palette=load_palette(pal)))
    assert encode_png(res.image) == expected
    assert res.warnings
    assert all(f"警告（{pixel_png.name}）: {w}" in cli_err for w in res.warnings)


def test_convert_sprite_does_not_change_input(pixel_png):
    rgba = load_rgba(pixel_png)
    before = rgba.copy()
    convert_sprite(rgba)
    assert np.array_equal(rgba, before)


def test_decode_encode_match_file_helpers(pixel_png, tmp_path):
    """バイト列の読み書きは load_rgba() / save_rgba() と同じ結果になる。"""
    a = decode_image(pixel_png.read_bytes())
    assert np.array_equal(a, load_rgba(pixel_png))
    save_rgba(a, tmp_path / "a.png")
    assert encode_png(a) == (tmp_path / "a.png").read_bytes()


def test_grid_not_found_gives_friendly_message():
    """ドット絵風でない画像は、利用者に分かる言葉のエラーを返す（例外は投げない）。"""
    res = convert_image_bytes(_png_bytes(make_character(256, 384)), {})
    assert res["ok"] is False
    assert "マス目が見つからなかった" in res["error"] and "ドット絵風の画像" in res["error"]


@pytest.mark.parametrize("data, settings, words", [
    (b"not an image", {}, ["画像として読み込めませんでした"]),
    (b"", {}, ["画像として読み込めませんでした"]),
    (None, {"colors": 0}, ["色数は 1〜256"]),
    (None, {"colors": 300}, ["色数は 1〜256"]),
    (None, {"colors": 2.5}, ["色数は 1〜256"]),
    (None, {"cleanup": "max"}, ["クリーンアップの強さ"]),
    (None, {"outline": "inner"}, ["縁取りは"]),
    (None, {"colour": 8}, ["知らない設定です: colour"]),
])
def test_bad_input_or_settings_give_messages(pixel_png, data, settings, words):
    res = convert_image_bytes(pixel_png.read_bytes() if data is None else data, settings)
    assert res["ok"] is False
    for w in words:
        assert w in res["error"]


def test_convert_for_web_bad_json(pixel_png):
    for text in ("{", "[1, 2]"):
        res = json.loads(convert_for_web(pixel_png.read_bytes(), text))
        assert res["ok"] is False and "設定" in res["error"]


def test_friendly_error_unexpected():
    assert "思いがけないエラー" in friendly_error(RuntimeError("x"))
    assert "マス目" in friendly_error(GridNotFoundError("y"))


def test_web_html_is_up_to_date():
    """apps/sprite/web/sprite-app.html は、ひな形と今の retrocore から作り直したものと同じ（古くなっていない）。"""
    assert build_sprite_web.main(["--check"]) == 0, "python tools/build_sprite_web.py で作り直してください"


def test_web_html_embeds_sources_and_cli_defaults():
    html = build_sprite_web.build()
    for key in ("__PY_SOURCES__", "__SETTINGS_SPEC__", "__PYODIDE_VERSION__", "__BUILD_ID__"):
        assert key not in html
    spec = build_sprite_web.settings_spec()
    defaults = SpriteSettings()
    assert spec["colors"]["default"] == defaults.colors == 16
    assert spec["cleanup"]["default"] == defaults.cleanup == "normal"
    assert spec["outline"]["default"] == defaults.outline == "auto"
    sources = build_sprite_web.python_sources()
    assert "retrocore/sprite/convert.py" in sources and "retrocore/__init__.py" in sources
    # 埋め込んだ JSON は </script> などで途切れない
    start = html.index('<script type="application/json" id="py-sources">') + len('<script type="application/json" id="py-sources">')
    body = html[start:html.index("</script>", start)]
    assert json.loads(body) == sources
    assert build_sprite_web.build() == html  # 毎回同じものができる


def test_error_has_detail_with_traceback():
    """エラーのときは、画面の「詳細」に出す情報（環境とトレースバック）も返す。"""
    res = convert_image_bytes(_png_bytes(make_character(256, 384)), {})
    assert res["ok"] is False
    assert "Traceback" in res["detail"] and "GridNotFoundError" in res["detail"]
    assert "NumPy" in res["detail"] and "intp" in res["detail"]
    res = json.loads(convert_for_web(b"", "{"))
    assert res["ok"] is False and "Traceback" in res["detail"]
