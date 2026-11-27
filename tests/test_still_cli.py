"""Still App の CLI（apps/still）とブラウザ版（still-app.html）のテスト。

ブラウザ版は CLI と同じ retrocore.still の関数（`still_for_web()`）を Pyodide で呼ぶので、ここでは同じ関数を
Python で呼び、CLI が書いたファイルとバイト単位で同じかを確かめる（Sprite App と同じやり方）。
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.still import cli
from retrocore.still.convert import MODES, StillSettings, spec, still_for_web
from synth_still import enlarge_scene, make_scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_still_web  # noqa: E402
import build_sprite_web  # noqa: E402

SEED = 1


@pytest.fixture(scope="module")
def scene_png(tmp_path_factory):
    path = tmp_path_factory.mktemp("still") / "scene.png"
    Image.fromarray(enlarge_scene(make_scene(), seed=SEED)).save(path)
    return path


def _run(src: Path, out: Path, *extra: str) -> int:
    return cli.main([str(src), "--out", str(out), "-q", *extra])


def test_cli_writes_indexed_png_and_json(scene_png, tmp_path):
    assert _run(scene_png, tmp_path / "a", "--mode", "pc98") == 0
    im = Image.open(tmp_path / "a.png")
    assert im.mode == "P" and im.size == (48, 30)
    info = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert info["width"] == 48 and info["source"] == "scene.png"
    assert len(im.getpalette()) // 3 == info["colors"]["used"] == len(info["palette"])


def test_cli_defaults_match_settings_defaults():
    args = cli.build_parser().parse_args(["x.png"])
    assert cli.settings_from_args(args) == StillSettings()


def test_cli_options_override_preset():
    args = cli.build_parser().parse_args(["x.png", "--colors", "8", "--no-4bit", "--max-size", "320x200",
                                          "--width", "100", "--canvas", "320x200", "--no-keep-darkest"])
    s = cli.settings_from_args(args)
    assert (s.colors, s.four_bit, s.max_size, s.width, s.canvas, s.keep_darkest) == (8, False, (320, 200), 100,
                                                                                    (320, 200), False)


@pytest.mark.parametrize("extra", [["--colors", "1"], ["--colors", "300"], ["--max-size", "abc"], ["--canvas", "10"]])
def test_cli_rejects_bad_options(scene_png, tmp_path, extra):
    with pytest.raises(SystemExit):
        _run(scene_png, tmp_path / "a", *extra)


def test_cli_over_max_size_writes_nothing(scene_png, tmp_path, capsys):
    assert _run(scene_png, tmp_path / "a", "--max-size", "40x40") == 1
    err = capsys.readouterr().err
    assert "上限 40×40" in err and "--width 40" in err
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "a.json").exists()


def test_cli_grid_not_found_lists_width_candidates(tmp_path, capsys):
    yy, xx = np.mgrid[0:120, 0:200]
    smooth = np.stack([xx * 255 // 200, yy * 255 // 120, (xx + yy) * 255 // 320], -1).astype(np.uint8)
    src = tmp_path / "smooth.png"
    Image.fromarray(smooth).save(src)
    assert _run(src, tmp_path / "a") == 1
    err = capsys.readouterr().err
    assert "マス目が見つかりませんでした" in err and "--width 25" in err   # 1 ドットを 8px とした場合
    assert _run(src, tmp_path / "a", "--width", "25") == 0


def test_cli_missing_input(tmp_path, capsys):
    assert _run(tmp_path / "none.png", tmp_path / "a") == 1


def test_cli_default_out_name(scene_png, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(scene_png), "-q"]) == 0
    assert (tmp_path / "out" / "still" / "scene.png").exists() and (tmp_path / "out" / "still" / "scene.json").exists()


def test_python_m_entry_point(scene_png, tmp_path):
    r = subprocess.run([sys.executable, "-m", "apps.still", str(scene_png), "--out", str(tmp_path / "m"), "--width", "24"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "24×15 ドット" in r.stdout


# ---- ブラウザ版と CLI がバイト単位で同じ ----

@pytest.mark.parametrize("extra, settings", [
    ([], {}),
    (["--width", "30"], {"width": 30}),
    (["--canvas", "52x34", "--colors", "8"], {"canvas": [52, 34], "colors": 8}),
    (["--no-4bit", "--no-keep-darkest"], {"four_bit": False, "keep_darkest": False}),
    (["--mode", "pc98", "--max-size", "100x60"], {"mode": "pc98", "max_size": [100, 60]}),
])
def test_web_result_equals_cli_files(scene_png, tmp_path, extra, settings):
    assert _run(scene_png, tmp_path / "c", *extra) == 0
    res = json.loads(still_for_web(scene_png.read_bytes(), scene_png.name, json.dumps(settings)))
    assert res["ok"]
    assert base64.b64decode(res["png_base64"]) == (tmp_path / "c.png").read_bytes()
    assert res["json_text"].encode("utf-8") == (tmp_path / "c.json").read_bytes()


def test_web_error_equals_cli_error(scene_png, tmp_path, capsys):
    assert _run(scene_png, tmp_path / "e", "--max-size", "30x30") == 1
    err = capsys.readouterr().err
    res = json.loads(still_for_web(scene_png.read_bytes(), scene_png.name, json.dumps({"max_size": [30, 30]})))
    assert not res["ok"] and res["error"] in err
    assert all(f"--width {c['width']}" in err for c in res["candidates"])


# ---- ブラウザ版の HTML ----

def test_still_web_html_is_up_to_date():
    assert build_still_web.main(["--check"]) == 0, "python tools/build_still_web.py で作り直してください"


def test_still_web_html_embeds_sources_spec_and_build_id():
    html = build_still_web.build()
    for key in ("__SETTINGS_SPEC__", "__PY_SOURCES__", "__PYODIDE_VERSION__", "__BUILD_ID__"):
        assert key not in html
    sources = build_still_web.python_sources()
    assert "retrocore/still/convert.py" in sources and "retrocore/sprite/grid.py" in sources
    assert build_still_web.settings_spec() == spec()
    assert spec()["modes"]["pc98"]["max_size"] == list(MODES["pc98"]["max_size"])
    assert f"Pyodide {build_still_web.PYODIDE_VERSION} ／ ビルド " in html
    assert build_still_web.PYODIDE_VERSION == build_sprite_web.PYODIDE_VERSION
    assert build_still_web.build() == html   # 毎回同じものができる


def test_sprite_web_html_does_not_include_still():
    """Sprite App の HTML には retrocore/still を入れない（Sprite App の HTML を変えないため）。"""
    assert not any(p.startswith("retrocore/still/") for p in build_sprite_web.python_sources())
    assert build_sprite_web.main(["--check"]) == 0
