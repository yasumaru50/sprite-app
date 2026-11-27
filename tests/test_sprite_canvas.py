"""キャンバスを倍数にそろえる（`retrocore/sprite/canvas.py`、CLI の `--canvas-multiple`・`--canvas-align`、画面の「キャンバスを倍数にそろえる」）のテスト。

- 余白の足し方（45×37・倍数 16・下ぞろえ → 48×48、左 1・右 2・上 11・下 0。上下左右中央・もう倍数・そろえない）
- 1 枚の絵（<出力名>.png と .json）・動きのシート・コマのシート・まとめたシートで、絵は同じ・pivot と ground_y は余白の分ずれる
- エフェクトのシートは同じ大きさで、position が余白の分ずれる
- 直したドット（edits.json）は余白を足す前の座標のまま、同じ絵の所に付く
- そろえないときの出力は前とバイト単位で同じ。CLI と画面の出力はバイト単位で同じ
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import canvas as CV
from retrocore.sprite import convert as C
from retrocore.sprite import fx as X
from retrocore.sprite import touchup as T
from retrocore.sprite.image import load_rgba, save_rgba
from retrocore.sprite.motion.gif import frames_from_sheet
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_frames import make_strip


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory):
    p = tmp_path_factory.mktemp("canvas") / "hero.png"
    save_rgba(enlarge_like_ai(make_pixel_sprite())[0], p)
    return p


@pytest.fixture(scope="module")
def walk_png(tmp_path_factory):
    p = tmp_path_factory.mktemp("canvas_walk") / "walk.png"
    save_rgba(make_strip(4), p)
    return p


def _run(*argv):
    from apps.sprite.cli import main
    return main([str(a) for a in argv])


def _frames(png_path, json_path):
    info = json.loads(json_path.read_text(encoding="utf-8"))
    return frames_from_sheet(load_rgba(png_path), info)[0], info


# ---- 余白の足し方 ----

def test_example_45x37_multiple_16_bottom():
    assert CV.pads(45, 37, 16, "bottom") == {"left": 1, "right": 2, "top": 11, "bottom": 0}
    assert CV.pads(45, 37, 16, "center") == {"left": 1, "right": 2, "top": 5, "bottom": 6}
    assert CV.pads(48, 32, 16) == {"left": 0, "right": 0, "top": 0, "bottom": 0}   # もう倍数なら足さない
    assert CV.pads(45, 37, 0) == {"left": 0, "right": 0, "top": 0, "bottom": 0}    # そろえない
    img = np.zeros((37, 45, 4), np.uint8)
    img[36, 22] = (10, 20, 30, 255)
    out, text = CV.pad_still(img, "x.png", 16)
    assert out.shape == (48, 48, 4) and (out[11:, 1:46] == img).all()
    info = json.loads(text)
    assert info["width"] == 48 and info["height"] == 48
    assert info["pivot"] == {"x": 23.5, "y": 48} and info["ground_y"] == 48
    assert info["canvas"] == {"multiple": 16, "align": "bottom", "width": 45, "height": 37,
                              "left": 1, "right": 2, "top": 11, "bottom": 0}
    same, none = CV.pad_still(img, "x.png", 0)
    assert same is img and none is None
    for bad in (-1, 257, 1.5, True, "16"):
        with pytest.raises(ValueError):
            CV.check(bad)
    with pytest.raises(ValueError):
        CV.check(16, "top")


# ---- 1 枚の絵と動きのシート ----

def test_default_output_is_unchanged(tmp_path, pixel_png):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--motion", "breathe", "--gif", "--fx", "surprise", "-q"]
    assert _run(pixel_png, "-o", a / "hero.png", *common) == 0
    assert _run(pixel_png, "-o", b / "hero.png", *common, "--canvas-multiple", "0") == 0
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir()) and "hero.json" not in names
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_still_and_motion_sheet(tmp_path, pixel_png):
    base, pad = tmp_path / "base", tmp_path / "pad"
    assert _run(pixel_png, "-o", base / "hero.png", "--motion", "float", "-q") == 0
    assert _run(pixel_png, "-o", pad / "hero.png", "--motion", "float", "-q", "--canvas-multiple", "16") == 0
    still0, still = load_rgba(base / "hero.png"), load_rgba(pad / "hero.png")
    h0, w0 = still0.shape[:2]
    rec = CV.record(w0, h0, 16, "bottom")
    assert still.shape[0] % 16 == 0 and still.shape[1] % 16 == 0
    assert (CV.crop_image(still, rec) == still0).all()
    info = json.loads((pad / "hero.json").read_text(encoding="utf-8"))
    assert info["image"] == "hero.png" and info["canvas"] == rec and info["ground_y"] == h0 + rec["top"]
    f0, i0 = _frames(base / "hero_float_sheet.png", base / "hero_float_sheet.json")
    f1, i1 = _frames(pad / "hero_float_sheet.png", pad / "hero_float_sheet.json")
    c = i1["canvas"]
    assert i1["frame_width"] % 16 == 0 and i1["frame_height"] % 16 == 0
    assert all((CV.crop_image(b, c) == a).all() for a, b in zip(f0, f1))
    assert i1["pivot"] == {"x": i0["pivot"]["x"] + c["left"], "y": i0["pivot"]["y"] + c["top"]}
    assert i1["ground_y"] == i1["pivot"]["y"] and "ground_y" not in i0
    assert [f["offset_y"] for f in i1["frames"]] == [f["offset_y"] for f in i0["frames"]]


def test_motion_cli_matches_web(tmp_path, pixel_png):
    d = tmp_path / "o"
    assert _run(pixel_png, "-o", d / "hero.png", "--motion", "sway", "--gif", "-q",
                "--canvas-multiple", "32", "--canvas-align", "center") == 0
    conv = C.convert_image_bytes(pixel_png.read_bytes(), {})
    raw = base64.b64decode(conv["png_base64"]) if "png_base64" in conv else conv["png"]
    got = C.motion_image_bytes(raw, {"motion": "sway", "image_name": "hero_sway_sheet.png",
                                     "convert_skipped": conv.get("skipped", []),
                                     "canvas": {"multiple": 32, "align": "center"}})
    assert got["ok"] and not got["skipped"]
    assert got["sheet_png"] == (d / "hero_sway_sheet.png").read_bytes()
    assert got["sheet_json"] == (d / "hero_sway_sheet.json").read_text(encoding="utf-8")
    assert got["canvas"]["align"] == "center"
    gif = json.loads(C.gif_for_web(got["sheet_png"], got["sheet_json"], json.dumps({})))
    assert base64.b64decode(gif["gif_base64"]) == (d / "hero_sway.gif").read_bytes()
    still = json.loads(C.canvas_still_for_web(raw, json.dumps({"multiple": 32, "align": "center", "image_name": "hero.png"})))
    assert still["ok"] and base64.b64decode(still["png_base64"]) == (d / "hero.png").read_bytes()
    assert still["json"] == (d / "hero.json").read_text(encoding="utf-8")
    off = json.loads(C.canvas_still_for_web(raw, json.dumps({"multiple": 0})))
    assert off["ok"] and off["json"] is None and base64.b64decode(off["png_base64"]) == raw


def test_fx_keeps_size_and_moves_position(tmp_path, pixel_png):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--motion", "breathe", "--fx", "surprise", "-q"]
    assert _run(pixel_png, "-o", a / "hero.png", *common) == 0
    assert _run(pixel_png, "-o", b / "hero.png", *common, "--canvas-multiple", "16") == 0
    fa = json.loads((a / "hero_breathe_fx_surprise.json").read_text(encoding="utf-8"))
    fb = json.loads((b / "hero_breathe_fx_surprise.json").read_text(encoding="utf-8"))
    sheet = json.loads((b / "hero_breathe_sheet.json").read_text(encoding="utf-8"))
    c = sheet["canvas"]
    assert (a / "hero_breathe_fx_surprise.png").read_bytes() == (b / "hero_breathe_fx_surprise.png").read_bytes()
    assert (fb["frame_width"], fb["frame_height"]) == (fa["frame_width"], fa["frame_height"])
    assert fb["position"] == {"x": fa["position"]["x"] + c["left"], "y": fa["position"]["y"] + c["top"]}
    assert fb["character_frame"] == {"width": sheet["frame_width"], "height": sheet["frame_height"]}
    # 画面（fx_for_web）も同じ
    r = json.loads(X.fx_for_web((b / "hero_breathe_sheet.png").read_bytes(),
                                (b / "hero_breathe_sheet.json").read_text(encoding="utf-8"), json.dumps({"fx": "surprise"})))
    assert r["ok"] and r["json"] == (b / "hero_breathe_fx_surprise.json").read_text(encoding="utf-8")


# ---- コマの絵 ----

def test_frames_sheet_pads_every_frame_the_same(tmp_path, walk_png):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--frames", "--frame-name", "walk", "--mirror", "--piskel", "--gif", "-q"]
    assert _run(walk_png, "-o", a / "walk.png", *common) == 0
    assert _run(walk_png, "-o", b / "walk.png", *common, "--canvas-multiple", "16") == 0
    assert (a / "walk.png").read_bytes() == (b / "walk.png").read_bytes()   # そろえたコマを並べた絵は変えない
    (sa,) = [p for p in a.glob("walk_walk_*x*_4f.png") if "mirror" not in p.name]
    (sb,) = [p for p in b.glob("walk_walk_*x*_4f.png") if "mirror" not in p.name]
    f0, i0 = _frames(sa, sa.with_suffix(".json"))
    f1, i1 = _frames(sb, sb.with_suffix(".json"))
    c = i1["canvas"]
    assert sb.name == f"walk_walk_{i1['frame_width']}x{i1['frame_height']}_4f.png"
    assert i1["frame_width"] % 16 == 0 and i1["frame_height"] % 16 == 0
    assert (c["width"], c["height"]) == (i0["frame_width"], i0["frame_height"])
    assert all((CV.crop_image(y, c) == x).all() for x, y in zip(f0, f1))   # 全部のコマに同じ余白
    assert i1["pivot"] == {"x": i0["pivot"]["x"] + c["left"], "y": i0["pivot"]["y"] + c["top"]}
    assert i1["ground_y"] == i1["pivot"]["y"]
    for p in (sb.with_suffix(".gif"), sb.with_suffix(".piskel")):
        assert p.exists()
    (mb,) = b.glob("walk_walk_mirror_*x*_4f.json")
    assert json.loads(mb.read_text(encoding="utf-8"))["canvas"]["multiple"] == 16


def test_frames_cli_matches_web(tmp_path, walk_png):
    d = tmp_path / "o"
    assert _run(walk_png, "-o", d / "walk.png", "--frames", "--frame-name", "walk", "--mirror", "-q",
                "--canvas-multiple", "24", "--canvas-align", "center") == 0
    w = C.frames_convert_bytes(walk_png.read_bytes(), {"name": "walk"})
    assert w["ok"] and w["made"]
    sh = C.frames_sheets({"token": w["token"], "base": "walk", "name": "walk", "mirror": True,
                          "canvas": {"multiple": 24, "align": "center"}})
    assert sh["ok"] and sh["canvas"]["multiple"] == 24
    for s in sh["sheets"]:
        assert (d / s["name"]).read_bytes() == s["png"]
        assert (d / s["name"]).with_suffix(".json").read_text(encoding="utf-8") == s["json"]
    off = C.frames_sheets({"token": w["token"], "base": "walk", "name": "walk"})
    assert off["ok"] and "canvas" not in off and "canvas" not in json.loads(off["sheets"][0]["json"])


def test_edits_stay_on_the_same_picture(tmp_path, walk_png):
    from retrocore.sprite import frames as F
    res = F.convert_frames(load_rgba(walk_png), SpriteSettings(), F.FrameSettings())
    f = res.frames[1]
    ys, xs = np.nonzero(f[..., 3])
    x, y = int(xs[np.argmin(ys)]), int(ys.min())
    pal = T.palette_of(res.frames)
    cur = "#%02x%02x%02x" % tuple(int(v) for v in f[y, x, :3])
    color = next(c for c in pal if c != cur)
    w, h = res.size
    t = T.Touchup(frames={2: [T.Dot(x, y, color, cur)]}, frame_width=w, frame_height=h, frame_count=len(res.frames))
    ep = tmp_path / "walk_edits.json"
    ep.write_text(T.touchup_json(t, pal), encoding="utf-8")
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--frames", "--frame-name", "walk", "--edits", ep, "-q"]
    assert _run(walk_png, "-o", a / "walk.png", *common) == 0
    assert _run(walk_png, "-o", b / "walk.png", *common, "--canvas-multiple", "32") == 0
    (sa,) = a.glob("walk_walk_*_4f.png")
    (sb,) = b.glob("walk_walk_*_4f.png")
    f0, i0 = _frames(sa, sa.with_suffix(".json"))
    f1, i1 = _frames(sb, sb.with_suffix(".json"))
    c = i1["canvas"]
    rgb = tuple(int(color[k:k + 2], 16) for k in (1, 3, 5))
    assert tuple(f0[1][y, x, :3]) == rgb                                   # 直しが入っている
    assert tuple(f1[1][y + c["top"], x + c["left"], :3]) == rgb          # 余白の分ずれた所（同じ絵の所）
    assert all((CV.crop_image(q, c) == p).all() for p, q in zip(f0, f1))
    assert i1["edits"] == i0["edits"]                                      # 記録は余白を足す前の座標のまま


# ---- まとめたシート ----

def test_anims_sheet_from_config_and_cli(tmp_path):
    from test_sprite_jump_spin import make_jump_strip
    save_rgba(make_strip(4), tmp_path / "walk.png")
    save_rgba(make_jump_strip(seed=5), tmp_path / "jump.png")
    d = {"anims": [{"name": "walk", "files": ["walk.png"]},
                   {"name": "jump", "files": ["jump.png"], "align_y": "keep", "loop": False}],
         "common": {"mirror": True}}
    (tmp_path / "plain.anims.json").write_text(json.dumps(d), encoding="utf-8")
    d2 = json.loads(json.dumps(d))
    d2["common"].update({"canvas_multiple": 16, "canvas_align": "bottom"})
    (tmp_path / "pad.anims.json").write_text(json.dumps(d2), encoding="utf-8")
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    assert _run("--anims", tmp_path / "plain.anims.json", "-o", a / "hero.png", "--gif", "--piskel", "-q") == 0
    assert _run("--anims", tmp_path / "pad.anims.json", "-o", b / "hero.png", "--gif", "--piskel", "-q") == 0
    assert _run("--anims", tmp_path / "plain.anims.json", "-o", c / "hero.png", "--gif", "--piskel", "-q",
                "--canvas-multiple", "16") == 0
    assert sorted(p.name for p in b.iterdir()) == sorted(p.name for p in c.iterdir())
    for p in b.iterdir():
        assert p.read_bytes() == (c / p.name).read_bytes(), p.name   # 設定のファイルでも引数でも同じ
    (ja,) = [p for p in a.glob("hero_anims_*.json") if "mirror" not in p.name]
    (jb,) = [p for p in b.glob("hero_anims_*.json") if "mirror" not in p.name]
    ia, ib = json.loads(ja.read_text(encoding="utf-8")), json.loads(jb.read_text(encoding="utf-8"))
    cv = ib["canvas"]
    assert ib["frame_width"] % 16 == 0 and ib["frame_height"] % 16 == 0
    assert ib["pivot"] == {"x": ia["pivot"]["x"] + cv["left"], "y": ia["pivot"]["y"] + cv["top"]}
    assert ib["ground_y"] == ia["ground_y"] + cv["top"]
    assert [f["air"] for f in ib["frames"]] == [f["air"] for f in ia["frames"]]
    assert ib["animations"] == ia["animations"]
    # 画面と同じバイト
    files = [(n, (tmp_path / n).read_bytes()) for n in ("walk.png", "jump.png")]
    conv = C.anims_convert_bytes(files, {"anims": d2})
    assert conv["ok"] and conv["made"]
    sh = C.anims_sheets({"token": conv["token"], "base": "hero", "anims": d2})
    assert sh["ok"], sh
    for s in sh["sheets"]:
        assert (b / s["name"]).read_bytes() == s["png"]
        assert (b / s["name"].replace(".png", ".json")).read_text(encoding="utf-8") == s["json"]
    for p in sh["piskels"]:
        assert (b / p["name"]).read_text(encoding="ascii") == p["text"]
    cfg = json.loads(sh["config_json"])
    assert cfg["common"]["canvas_multiple"] == 16 and cfg["common"]["canvas_align"] == "bottom"
    # そろえない設定のファイルは前と同じ形（canvas の項目を書かない）
    plain = C.anims_sheets({"token": conv["token"], "base": "hero", "anims": d})
    assert "canvas_multiple" not in json.loads(plain["config_json"])["common"]


def test_cli_rejects_bad_canvas_options(tmp_path, pixel_png, capsys):
    for argv in (["--canvas-multiple", "300"], ["--canvas-multiple", "x"], ["--canvas-align", "center"]):
        with pytest.raises(SystemExit):
            _run(pixel_png, "-o", tmp_path / "o.png", *argv)
    assert "--canvas" in capsys.readouterr().err


def test_web_html_has_canvas_ui():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "apps/sprite/web/template.html").read_text(encoding="utf-8")
    for key in ('id="canvas-multiple"', 'id="canvas-align"', 'id="canvas-custom"', 'id="dl-still-json"',
                "canvas_still_for_web", "canvas_multiple"):
        assert key in html, key
