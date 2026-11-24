"""動きに名前を付けて 1 つのシートにまとめる（`retrocore/sprite/anims.py`、CLI の `--anims`、画面の「動きに名前を付けてまとめる」）のテスト。

- 設定: 同じ名前が続くファイルは 1 つの動き、離れていればエラー。名前の決まり・重なり・1 つのファイルは 1 つの動きだけ
- シート: 1 つの動き = 1 行、横はいちばんコマの多い動き（余りは透明）。全部の動きで同じ大きさ・同じ地面（ground_y = 下端）・同じパレット
- JSON の animations（frames はシート全体の番号・duration_ms・loop・row）。くり返さない動きの GIF は 1 回だけ再生
- 直しの一覧（edits.json）: 動きの並びを入れ替えても、直したドットは同じ絵の同じ所に付く
- 読み戻し: 動きごと（ファイルの名前から決める・選ぶ）。直したドットはまとめたシートのその動きの行に入る
- CLI と画面（anims_convert_bytes → anims_sheets・anims_gif_bytes）の出力はバイト単位で同じ
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import anims as A
from retrocore.sprite import convert as C
from retrocore.sprite import edit as E
from retrocore.sprite import touchup as T
from retrocore.sprite.image import save_rgba
from retrocore.sprite.motion import gif as G
from retrocore.sprite.piskel import piskel_text
from test_sprite_frames import make_strip
from test_sprite_jump_spin import make_jump_strip


@pytest.fixture(scope="module")
def walk():
    return make_strip(4)


@pytest.fixture(scope="module")
def jump():
    # 2 コマ目が上に浮いた絵（元の絵の高さを保つで、宙に浮くコマになる）
    return make_jump_strip(seed=5)


@pytest.fixture(scope="module")
def idle():
    return make_strip(1, seed=9)


def _cfg(**over):
    d = {"anims": [{"name": "idle", "files": ["idle.png"]},
                   {"name": "walk", "files": ["walk.png"], "order": "1,2,3,2r", "duration_ms": [150, 100, 150, 100]},
                   {"name": "jump", "files": ["jump.png"], "align_y": "keep", "loop": False, "duration_ms": 120}],
         "common": {"mirror": True}}
    d.update(over)
    return d


@pytest.fixture(scope="module")
def converted(walk, jump, idle):
    cfg = A.config_from_dict(_cfg())
    res = A.convert_anims([("walk.png", walk), ("jump.png", jump), ("idle.png", idle)], SpriteSettings(), cfg)
    return cfg, res


# ---- 設定 ----

def test_group_rows_rules():
    assert A.group_rows(["a", "b", "c"], ["walk", "walk", "jump"]) == [("walk", ["a", "b"]), ("jump", ["c"])]
    with pytest.raises(ValueError, match="離れています"):
        A.group_rows(["a", "b", "c"], ["walk", "jump", "walk"])
    with pytest.raises(ValueError, match="名前がありません"):
        A.group_rows(["a"], [""])
    for bad in ("Walk", "1run", "run-1", "x" * 40):
        with pytest.raises(ValueError, match="英小文字"):
            A.group_rows(["a"], [bad])


@pytest.mark.parametrize("d, words", [
    ({"anims": [{"name": "walk", "files": ["a.png"]}, {"name": "walk", "files": ["b.png"]}]}, "2 つあります"),
    ({"anims": [{"name": "walk", "files": ["a.png"]}, {"name": "run", "files": ["a.png"]}]}, "両方"),
    ({"anims": [{"name": "walk", "files": ["a.png"], "align_y": "up"}]}, "縦のそろえ方"),
    ({"anims": [{"name": "walk", "files": ["a.png"], "duration_ms": 0}]}, "1 以上"),
    ({"anims": [{"name": "walk", "files": ["a.png"], "speed": 2}]}, "知らない設定"),
    ({"kind": "sprite_edits", "anims": [{"name": "walk", "files": ["a.png"]}]}, "kind"),
    ({"anims": []}, "空"),
])
def test_config_errors(d, words):
    with pytest.raises(ValueError, match=words):
        A.config_from_dict(d)


def test_config_json_round_trip():
    cfg = A.config_from_dict(dict(_cfg(), settings={"colors": 8}))
    again = A.load_config(A.config_json(cfg))
    assert A.config_dict(again) == A.config_dict(cfg)
    assert again.anims[1].order == "1,2,3,2r" and again.anims[2].loop is False and again.settings == {"colors": 8}


# ---- 変換とシート ----

def test_one_canvas_one_palette_one_ground(converted):
    cfg, res = converted
    assert [(r["name"], r["first"], r["count"]) for r in res.anims] == [("idle", 1, 1), ("walk", 2, 4), ("jump", 6, 3)]
    assert len({f.shape for f in res.frames}) == 1
    sheet = A.assemble_anims(res, cfg, "hero")
    info = sheet.info
    w, h = sheet.size
    assert sheet.image_name == f"hero_anims_{w}x{h}_8f.png"
    img = sheet.image()
    assert img.shape == (3 * h, 4 * w, 4)
    # 1 つの動き = 1 行。idle の行の 2 列目から先は透明
    assert not img[0:h, w:, 3].any()
    assert info["columns"] == 4 and info["rows"] == 3 and info["frame_count"] == 8
    assert info["animations"] == {"idle": {"frames": [0], "duration_ms": 150, "loop": True, "row": 0},
                                  "walk": {"frames": [1, 2, 3, 4], "duration_ms": 150, "loop": True, "row": 1},
                                  "jump": {"frames": [5, 6, 7], "duration_ms": 120, "loop": False, "row": 2}}
    assert [(f["x"], f["y"]) for f in info["frames"]][:3] == [(0, 0), (0, h), (w, h)]
    assert [f["animation"] for f in info["frames"]] == ["idle"] + ["walk"] * 4 + ["jump"] * 3
    assert [f["duration_ms"] for f in info["frames"]] == [150, 150, 100, 150, 100, 120, 120, 120]
    # 地面の線は全部の動きでフレームの下端。足元で合わせた動きは浮かない、元の絵の高さを保つ jump は浮くコマがある
    assert info["ground_y"] == info["pivot"]["y"] == h
    air = [f["air"] for f in info["frames"]]
    assert air[:5] == [0] * 5 and air[5] == 0 and air[6] > air[7] > 0
    # 順番の r（裏返したコマ）と通し番号の source_frame
    assert [f["source_frame"] for f in info["frames"]] == [1, 2, 3, 4, 3, 6, 7, 8]
    assert info["frames"][4].get("flipped") is True
    # 新しい色を作らない（シートの色は変換したコマの色だけ）
    assert set(info["palette"]) == set(T.palette_of(res.frames))
    assert [a["name"] for a in info["source"]["anims"]] == ["idle", "walk", "jump"]
    assert info["source"]["anims"][2]["align_y"] == "keep" and info["source"]["frame_count"] == 8


def test_mirror_sheet(converted):
    cfg, res = converted
    a, m = A.assemble_anims(res, cfg, "hero"), A.assemble_anims(res, cfg, "hero", mirror=True)
    assert m.image_name.startswith("hero_anims_mirror_")
    ia, im = a.image(), m.image()
    w, h = a.size
    assert np.array_equal(ia[h:2 * h, :w][:, ::-1], im[h:2 * h, :w])
    assert m.info["source"]["mirrored"] is True and a.info["source"]["mirrored"] is False


def test_gif_plays_once_when_not_looping(converted):
    cfg, res = converted
    sheet = A.assemble_anims(res, cfg, "hero")
    name, g = A.anim_gif(sheet.image(), sheet.info, "jump")
    assert name == "hero_jump.gif" and b"NETSCAPE2.0" not in g.data
    name, g = A.anim_gif(sheet.image(), sheet.info, "walk")
    assert name == "hero_walk.gif" and b"NETSCAPE2.0" in g.data
    # make_gif の既定（くり返す）は今までと同じ
    fr, du = A.anim_frames(sheet.image(), sheet.info, "jump")
    assert G.make_gif(fr, du).data == G.make_gif(fr, du, loop=True).data


def test_missing_or_extra_file(walk, jump):
    cfg = A.config_from_dict({"anims": [{"name": "walk", "files": ["walk.png"]}, {"name": "jump", "files": ["jump.png"]}]})
    with pytest.raises(ValueError, match="ありません"):
        A.convert_anims([("walk.png", walk)], SpriteSettings(), cfg)
    with pytest.raises(ValueError, match="どの動きにも"):
        A.convert_anims([("walk.png", walk), ("jump.png", jump), ("x.png", walk)], SpriteSettings(), cfg)


# ---- 直しの一覧: 動きの並びを入れ替えても同じ絵に付く ----

def test_edits_follow_the_picture_when_anims_are_reordered(walk, jump):
    files = [("walk.png", C.encode_png(walk)), ("jump.png", C.encode_png(jump))]
    d1 = {"anims": [{"name": "walk", "files": ["walk.png"]}, {"name": "jump", "files": ["jump.png"], "align_y": "keep"}]}
    out = C.anims_convert_bytes(files, {"anims": d1})
    assert out["ok"] and out["made"] and out["edits_origin"]
    res = C._FRAMES_CACHE["result"]
    # jump の 2 コマ目（通し番号 6）の、いちばん上の不透明なドットを別の色に直す
    k = 6
    f = res.frames[k - 1]
    ys, xs = np.nonzero(f[..., 3] > 0)
    y, x = int(ys[0]), int(xs[ys == ys[0]][0])
    pal = T.palette_of(res.frames)
    under = "#%02x%02x%02x" % tuple(int(v) for v in f[y, x, :3])
    color = next(c for c in pal if c != under)
    edits = {"version": 1, "kind": "sprite_edits", "name": "anims", "frame_width": res.size[0], "frame_height": res.size[1],
             "frame_count": len(res.frames), "palette": pal, "frames": [{"source_frame": k, "dots": [[x, y, pal.index(color)]]}]}
    got = C.frames_edits({"token": out["token"], "edits": edits})
    assert got["ok"] and got["applied"], got
    saved = json.loads(got["edits_json"])
    assert saved["frames"][0]["file"] == "jump.png" and saved["frames"][0]["file_frame"] == 2
    # 動きの並びを入れ替えて（jump が先）、保存した一覧を当てる
    d2 = {"anims": [d1["anims"][1], d1["anims"][0]]}
    cfg2 = A.config_from_dict(d2)
    imgs = [(n, C.decode_image(b)) for n, b in files]
    res2 = A.convert_anims(imgs, SpriteSettings(), cfg2, T.load_touchup(got["edits_json"]))
    k2 = res2.sources.index(("jump.png", 2))
    ox, oy = res.offsets[k - 1]
    nx, ny = res2.offsets[k2]
    assert "#%02x%02x%02x" % tuple(int(v) for v in res2.frames[k2][y - oy + ny, x - ox + nx, :3]) == color
    assert res2.touchup.dot_count == 1


# ---- 読み戻し ----

def _sheet_files(converted, tmp_path):
    cfg, res = converted
    made = A.anims_outputs(res, cfg, "hero")
    sheet = made["sheets"][0]
    return sheet, made["files"][sheet.image_name], made["files"][sheet.image_name[:-4] + ".json"].decode("utf-8")


def test_readback_one_anim(converted, tmp_path):
    sheet, png, js = _sheet_files(converted, tmp_path)
    info = json.loads(js)
    frames, _ = A.anim_frames(sheet.image(), info, "walk")
    edited = [f.copy() for f in frames]
    ys, xs = np.nonzero(edited[0][..., 3] > 0)
    edited[0][ys[0], xs[0], 3] = 0   # 1 ドット消す（新しい色は作らない）
    pk = piskel_text(edited, "hero_walk").encode("ascii")
    r = E.readback(pk, "hero_walk (1).piskel", js, png, E.ReadbackOptions())
    assert r.made, r.errors
    assert r.report["animation"] == "walk"
    out = C.decode_image(r.files[r.names["sheet"]])
    w, h = sheet.size
    assert r.names["sheet"] == sheet.image_name[:-4] + "_edited.png"
    assert out[h + ys[0], xs[0], 3] == 0
    before = sheet.image()
    diff = np.argwhere((out != before).any(-1))
    assert len(diff) == 1
    assert "hero_walk_edited.gif" in r.files and "hero_walk_edited.piskel" in r.files
    assert r.names.get("mirror") and json.loads(r.files[r.names["json"]])["edited"]["animation"] == "walk"
    # 直していなければ、まとめたシートは同じ
    same = E.readback(piskel_text(frames, "x").encode("ascii"), "hero_walk.piskel", js, png, E.ReadbackOptions())
    assert same.files[same.names["sheet"]] == png


def test_readback_needs_anim_and_original(converted, tmp_path):
    sheet, png, js = _sheet_files(converted, tmp_path)
    frames, _ = A.anim_frames(sheet.image(), json.loads(js), "jump")
    pk = piskel_text(frames, "x").encode("ascii")
    r = E.readback(pk, "naoshita.piskel", js, png, E.ReadbackOptions())
    assert not r.made and r.report["code"] == "anim_unknown"
    r = E.readback(pk, "naoshita.piskel", js, png, E.ReadbackOptions(anim="jump"))
    assert r.made and r.report["animation"] == "jump"
    r = E.readback(pk, "hero_jump.piskel", js, None, E.ReadbackOptions())
    assert not r.made and r.report["code"] == "need_original"
    assert A.guess_anim("hero_jump_edited.piskel", ["idle", "jump"]) == "jump"
    assert A.guess_anim("other.png", ["idle", "jump"]) is None


def test_plain_sheet_rejects_anim_option(walk):
    from retrocore.sprite import frames as F
    res = F.convert_frames(walk, SpriteSettings(), F.FrameSettings())
    m = F.assemble_frames(res)
    js = F.frames_sheet_json(m, "w.png", res.skipped, res.dot_size)
    r = E.readback(C.encode_png(np.concatenate(m.frames, 1)), "w.png", js, None, E.ReadbackOptions(anim="walk"))
    assert not r.made and "--anim" in r.errors[0]


# ---- CLI と画面 ----

def test_cli_and_web_give_the_same_bytes(walk, jump, idle, tmp_path, capsys):
    from apps.sprite.cli import main
    d = _cfg(settings={"colors": 12, "cleanup": "normal", "outline": "auto"})
    for n, a in (("walk.png", walk), ("jump.png", jump), ("idle.png", idle)):
        save_rgba(a, tmp_path / n)
    (tmp_path / "hero.anims.json").write_text(json.dumps(d), encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(["--anims", str(tmp_path / "hero.anims.json"), "-o", str(out_dir / "hero.png"),
                 "--gif", "--piskel", "-q"]) == 0
    files = [(n, (tmp_path / n).read_bytes()) for n in ("idle.png", "walk.png", "jump.png")]
    conv = C.anims_convert_bytes(files, {"settings": d["settings"], "anims": {k: v for k, v in d.items() if k != "settings"}})
    assert conv["ok"] and conv["made"]
    sh = C.anims_sheets({"token": conv["token"], "base": "hero", "anims": {k: v for k, v in d.items() if k != "settings"},
                         "settings": d["settings"]})
    assert sh["ok"], sh
    assert len(sh["sheets"]) == 2
    for s in sh["sheets"]:
        assert (out_dir / s["name"]).read_bytes() == s["png"]
        assert (out_dir / s["name"].replace(".png", ".json")).read_text(encoding="utf-8") == s["json"]
    for p in sh["piskels"]:
        assert (out_dir / p["name"]).read_text(encoding="ascii") == p["text"]
    for s in sh["sheets"]:
        for anim in ("idle", "walk", "jump"):
            g = C.anims_gif_bytes(s["png"], s["json"], anim, {"scale": G.DEFAULT_SCALE})
            assert g["ok"] and (out_dir / g["name"]).read_bytes() == g["gif"]
    # 保存する設定のファイルは CLI でそのまま読める
    assert A.config_dict(A.load_config(sh["config_json"]))["anims"] == A.config_dict(A.config_from_dict(d))["anims"]
    assert sh["config_name"] == "hero_anims.anims.json"


def test_cli_ignores_unsupported_and_rejects_per_anim_options(walk, jump, tmp_path, capsys):
    from apps.sprite.cli import main
    for n, a in (("walk.png", walk), ("jump.png", jump)):
        save_rgba(a, tmp_path / n)
    cfg = tmp_path / "a.anims.json"
    cfg.write_text(json.dumps({"anims": [{"name": "walk", "files": ["walk.png"]}, {"name": "jump", "files": ["jump.png"]}]}),
                   encoding="utf-8")
    assert main(["--anims", str(cfg), "-o", str(tmp_path / "o" / "x.png"), "--steady", "on", "--flicker", "-q"]) == 0
    err = capsys.readouterr().err
    assert "ちらつきを減らす" in err and "無視します" in err
    assert not list((tmp_path / "o").glob("*flicker*"))
    with pytest.raises(SystemExit):
        main(["--anims", str(cfg), "--frame-order", "1,2"])
    assert "動きごと" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--anims", str(cfg), str(tmp_path / "walk.png")])   # jump.png が入力に無い
    assert "jump.png" in capsys.readouterr().err


def test_web_html_has_anims_ui():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "apps" / "sprite" / "web" / "template.html").read_text(encoding="utf-8")
    for key in ('id="anims-on"', 'id="anims-list"', 'id="anims-play"', 'id="dl-anims-config"', 'id="edit-anim"',
                "anims_convert_files_for_web", "anims_sheets_for_web", "anims_gif_for_web",
                # 1 回で終わる動きは、見本の再生で最後のコマに止まる（info.once）
                "once: a.loop === false", "ONCE_HOLD_MS", "onceEnd(info)",
                # まとめているときの ⑤ の名前（どれに animations が入るか分かるように）・印を変えたら前のシートを保存できない
                "⑤ まとめたシートの情報を保存（JSON。animations 入り）", "animsConfigOff()"):
        assert key in html
