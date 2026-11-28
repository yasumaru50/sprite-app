"""ドットの大きさの欄（画面）・--dot-size（CLI）と、格子が見つからなかったときの候補（`retrocore/sprite/dotsize.py`）。

- 候補は、自動の推定でいちばん点が高かった大きさを 0.5 刻みに丸めた所を真ん中に、0.5 刻みで最大 5 つ
- 画面の設定（数・[横, 縦]）は CLI の --dot-size と同じ (横, 縦) になり、出力はバイト単位で同じ
- 動きをまとめる設定（.anims.json）の settings に書いた大きさは、CLI でも同じに効く
- 格子が見つからないときは、画面の結果に候補が入り、CLI は「次に試せる大きさ」を添える

自動で見つからない絵をコードで合成するのは難しい（ノイズを強くすると、推定そのものが外れる）ので、ここでは見つかる絵で
見つからない判定の基準だけを上げて（monkeypatch）、候補の作り方と、その大きさで変換した結果を確かめる。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import anims as A
from retrocore.sprite import convert as C
from retrocore.sprite import dotsize as D
from retrocore.sprite import grid as G
from retrocore.sprite.image import save_rgba
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_frames import make_strip


@pytest.fixture(scope="module")
def one():
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), sizes=(19, 22), half_shift=0)
    return img


@pytest.fixture(scope="module")
def strip():
    return make_strip(3)


@pytest.fixture
def strict(monkeypatch):
    """格子が見つからない判定の基準を上げる（1 枚の絵の detect_grid と、コマの絵の detect_frame_grids）。"""
    kw = dict(G.detect_grid.__kwdefaults__)
    monkeypatch.setattr(G.detect_grid, "__kwdefaults__", dict(kw, min_prominence=5.0))
    monkeypatch.setattr(G, "DEFAULT_MIN_PROMINENCE", 5.0)


def test_candidate_sizes():
    assert D.candidate_sizes(6.4) == [5.5, 6.0, 6.5, 7.0, 7.5]
    assert D.candidate_sizes(6.6) == [5.5, 6.0, 6.5, 7.0, 7.5]
    assert D.candidate_sizes(6.76) == [6.0, 6.5, 7.0, 7.5, 8.0]
    assert D.candidate_sizes(20.0) == [19.0, 19.5, 20.0, 20.5, 21.0]
    assert D.candidate_sizes(2.3) == [2.0, 2.5, 3.0, 3.5]         # 下限（2）より小さいものは除く
    assert D.candidate_sizes(None) == [] and D.candidate_sizes(0.0) == []
    assert D.cli_hint([6.0, 6.5, 7.0]) == "次に試せる大きさ: --dot-size 6 / 6.5 / 7"
    assert D.cli_hint([]) == ""


def test_parse_dot_size():
    assert D.parse_dot_size(6.5) == (6.5, 6.5)
    assert D.parse_dot_size(7) == (7.0, 7.0)
    assert D.parse_dot_size([6, 7.5]) == (6.0, 7.5)
    assert D.parse_dot_size("6.5") == (6.5, 6.5)
    assert D.parse_dot_size("6x7") == (6.0, 7.0)
    assert D.parse_dot_size(None) is None and D.parse_dot_size(" ") is None
    for bad in (1.5, 300, "abc", [1, 2, 3], True, float("nan"), {"a": 1}):
        with pytest.raises(ValueError):
            D.parse_dot_size(bad)


def test_settings_from_dict():
    assert SpriteSettings.from_dict({"dot_size": 6.5}).dot_size == (6.5, 6.5)
    assert SpriteSettings.from_dict({"dot_size": [6, 7]}).dot_size == (6.0, 7.0)
    assert SpriteSettings.from_dict({"dot_size": None}).dot_size is None
    assert SpriteSettings.from_dict({}) == SpriteSettings()   # 欄が空のとき（送らない）は今までと同じ
    with pytest.raises(ValueError):
        SpriteSettings.from_dict({"dot_size": "x"})
    r = C.convert_image_bytes(b"", {"dot_size": 1})
    assert not r["ok"] and "ドットの大きさ" in r["error"]


def test_error_keeps_estimate(one, strict):
    from retrocore.sprite.background import BackgroundStep
    a = BackgroundStep().apply(one)
    with pytest.raises(G.GridNotFoundError) as e:
        G.detect_grid(a)
    assert 19 <= e.value.estimate <= 22
    assert str(e.value).startswith("ドットの格子が見つかりませんでした")   # 文は前と同じ


def test_single_candidates(one, strict):
    data = C.encode_png(one)
    r = C.convert_image_bytes(data, {})
    assert not r["ok"]
    sizes = r["dot_candidates"]
    assert len(sizes) == 5 and sizes == sorted(sizes) and all(v * 2 == int(v * 2) for v in sizes)
    assert sizes[2] - 1 == sizes[0] and 19 <= sizes[2] <= 22
    got = C.dot_candidates_bytes([("a.png", data)], {"kind": "single", "sizes": sizes})
    assert got["ok"] and [c["dot_size"] for c in got["candidates"]] == sizes
    for c in got["candidates"]:
        assert c["ok"] and len(c["heights"]) == 1
        # 見本は、欄にその数を入れて変換したときと同じ絵
        same = C.convert_image_bytes(data, {"dot_size": c["dot_size"]})
        assert same["ok"] and same["png"] == c["png"] and same["height"] == c["heights"][0]
    # 画面から呼ぶ形（JSON・base64）
    w = json.loads(C.dot_candidates_for_web([data], ["a.png"], json.dumps({"kind": "single", "sizes": sizes[:1]})))
    assert w["ok"] and "png_base64" in w["candidates"][0] and "png" not in w["candidates"][0]


def test_found_has_no_candidates(one):
    r = C.convert_image_bytes(C.encode_png(one), {})
    assert r["ok"] and "dot_candidates" not in r


def test_frames_candidates(strip, strict):
    data = C.encode_png(strip)
    r = C.frames_convert_bytes(data, {})
    assert r["ok"] and not r["made"] and r["code"] == "grid_not_found"
    sizes = r["dot_candidates"]
    assert len(sizes) == 5 and 19 <= sizes[2] <= 22
    got = C.dot_candidates_bytes([("a.png", data)], {"kind": "frames", "sizes": sizes[1:4], "frames": {"frame_count": 3}})
    assert got["ok"] and len(got["candidates"]) == 3
    for c in got["candidates"]:
        assert c["ok"] and len(c["heights"]) == 3 and len(c["widths"]) == 3
    # 候補の高さは、欄にその数を入れて変換したときのコマ（そろえる前）の高さ
    c = got["candidates"][1]
    w = C.frames_convert_bytes(data, {"settings": {"dot_size": c["dot_size"]}})
    assert w["made"] and w["png"] == c["png"]
    res = C._FRAMES_CACHE["result"]
    assert [f.shape[0] for f in res.raw] == c["heights"]
    # まとめる（anims）: 動きの設定のファイルの名前で渡す
    cfg = {"anims": [{"name": "walk", "files": ["a.png"]}]}
    r = C.anims_convert_bytes([("a.png", data)], {"anims": cfg})
    assert not r["made"] and r["dot_candidates"] == sizes
    got = C.dot_candidates_bytes([("a.png", data)], {"kind": "anims", "sizes": sizes[2:3], "anims": cfg})
    assert got["ok"] and got["candidates"][0]["ok"] and len(got["candidates"][0]["heights"]) == 3


def test_candidates_bad_request():
    assert not C.dot_candidates_bytes([("a.png", b"")], {"kind": "x"})["ok"]
    assert not C.dot_candidates_bytes([("a.png", b"")], {"kind": "single", "sizes": [1]})["ok"]
    assert not C.dot_candidates_bytes([("a.png", b"")], {"kind": "single", "what": 1})["ok"]
    # 変換できない大きさは、その候補だけ ok: False
    img = np.zeros((20, 20, 4), np.uint8)
    img[5:15, 5:15] = (200, 50, 50, 255)
    got = C.dot_candidates_bytes([("a.png", C.encode_png(img))], {"kind": "frames", "sizes": [200]})
    assert got["ok"] and not got["candidates"][0]["ok"] and got["candidates"][0]["error"]


# ---- CLI ----

def test_cli_hint(one, strip, strict, tmp_path, capsys):
    from apps.sprite.cli import main
    a, b = tmp_path / "one.png", tmp_path / "strip.png"
    save_rgba(one, a)
    save_rgba(strip, b)
    assert main([str(a), "-o", str(tmp_path / "o" / "one.png"), "-q"]) == 1
    err = capsys.readouterr().err
    sizes = C.convert_image_bytes(a.read_bytes(), {})["dot_candidates"]
    assert D.cli_hint(sizes) in err and "--dot-size" in D.cli_hint(sizes)
    assert main([str(b), "--frames", "-o", str(tmp_path / "o" / "strip.png"), "-q"]) == 1
    err = capsys.readouterr().err
    sizes = C.frames_convert_bytes(b.read_bytes(), {})["dot_candidates"]
    assert D.cli_hint(sizes) in err
    assert err.count("コマを作りませんでした") == 1


def test_cli_and_web_same_bytes(one, strip, tmp_path):
    from apps.sprite.cli import main
    a, b = tmp_path / "one.png", tmp_path / "strip.png"
    save_rgba(one, a)
    save_rgba(strip, b)
    out = tmp_path / "out"
    # 1 枚の絵
    assert main([str(a), "--dot-size", "20.5", "-o", str(out / "one.png"), "-q"]) == 0
    w = C.convert_image_bytes(a.read_bytes(), {"dot_size": 20.5})
    assert (out / "one.png").read_bytes() == w["png"]
    auto = C.convert_image_bytes(a.read_bytes(), {})
    assert auto["png"] != w["png"] or auto["width"] == w["width"]
    # コマが並んだ絵（シートと JSON まで）
    assert main([str(b), "--frames", "--dot-size", "21", "-o", str(out / "walk.png"), "-q", "--mirror"]) == 0
    w = C.frames_convert_bytes(b.read_bytes(), {"settings": {"dot_size": 21}})
    assert w["made"] and w["dot_size"] == 21.0
    assert (out / "walk.png").read_bytes() == w["png"]
    sh = C.frames_sheets({"token": w["token"], "base": "walk", "mirror": True})
    assert sh["ok"]
    for s in sh["sheets"]:
        assert (out / s["name"]).read_bytes() == s["png"]
        assert (out / s["name"]).with_suffix(".json").read_text(encoding="utf-8") == s["json"]
        assert json.loads(s["json"])["source"]["dot_size"] == 21.0


def test_anims_config_keeps_dot_size(strip, tmp_path):
    """画面で保存した .anims.json の settings に入れた大きさで、CLI が画面と同じシートを作る（無い古いファイルは自動のまま）。"""
    from apps.sprite.cli import main
    save_rgba(strip, tmp_path / "walk.png")
    settings = {"colors": 16, "cleanup": "normal", "outline": "auto", "dot_size": 20.5}
    anims = {"name": "hero", "anims": [{"name": "walk", "files": ["walk.png"]}]}
    files = [("walk.png", (tmp_path / "walk.png").read_bytes())]
    conv = C.anims_convert_bytes(files, {"settings": settings, "anims": anims})
    assert conv["ok"] and conv["made"] and conv["dot_size"] == 20.5
    sh = C.anims_sheets({"token": conv["token"], "base": "hero", "anims": anims, "settings": settings})
    assert sh["ok"]
    cfg = json.loads(sh["config_json"])
    assert cfg["settings"]["dot_size"] == 20.5            # 保存する設定のファイルに入る
    p = tmp_path / sh["config_name"]
    p.write_text(sh["config_json"], encoding="utf-8")
    out = tmp_path / "out"
    assert main(["--anims", str(p), "-o", str(out / "hero.png"), "-q"]) == 0
    for s in sh["sheets"]:
        assert (out / s["name"]).read_bytes() == s["png"]
        assert (out / s["name"].replace(".png", ".json")).read_text(encoding="utf-8") == s["json"]
    # 大きさの無い設定は自動（推定した大きさ）
    old = dict(cfg, settings={k: v for k, v in cfg["settings"].items() if k != "dot_size"})
    c2 = A.config_from_dict(old)
    assert "dot_size" not in c2.settings
    conv2 = C.anims_convert_bytes(files, {"settings": old["settings"], "anims": anims})
    assert conv2["made"] and conv2["dot_size"] != 20.5
