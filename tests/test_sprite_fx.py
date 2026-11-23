"""重ね（漫画的エフェクト。retrocore/sprite/fx/）のテスト: 絵の表、置く場所、左右反転、終わり方、重ねた見た目、
CLI と画面のバイト単位の一致、エフェクトを選ばないときに出力が変わらないこと。"""

import json

import numpy as np
import pytest

from retrocore.sprite import fx as X
from retrocore.sprite import save_rgba
from retrocore.sprite.fx import art
from retrocore.sprite.motion import make_motion
from retrocore.sprite.piskel import read_piskel
from synth_sprite import enlarge_like_ai, make_pixel_sprite

DARK = (20, 10, 30, 255)
BODY = (200, 120, 80, 255)


def _char(w=24, h=30):
    """頭（上の 8 行、x 6〜13）と体（x 3〜20）の、輪郭の色がいちばん暗い四角いキャラ。"""
    a = np.zeros((h, w, 4), np.uint8)
    a[2:10, 6:14] = DARK
    a[3:9, 7:13] = BODY
    a[10:28, 3:21] = DARK
    a[11:27, 4:20] = BODY
    return a


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory):
    p = tmp_path_factory.mktemp("fx") / "hero.png"
    save_rgba(enlarge_like_ai(make_pixel_sprite())[0], p)
    return p


# ---- 絵の表 ----

def test_all_effects_load_and_use_only_known_colors():
    assert X.effect_names()[:4] == ["surprise", "sweat", "heart", "sleep"]
    for n in X.effect_names():
        e = X.load_effect(n)
        assert 4 <= e.width <= 20 and 4 <= e.height <= 20
        assert all(d % 50 == 0 for d in e.durations)   # .piskel にそろえやすい
        frames = X.render(e, (1, 2, 3))
        assert all(f.shape == (e.height, e.width, 4) for f in frames)
        allowed = {(*c, 255) for c in art.COLORS.values()} | {(1, 2, 3, 255)}
        for f in frames:
            used = {tuple(int(v) for v in px) for px in f[f[..., 3] > 0]}
            assert used <= allowed
            assert set(np.unique(f[..., 3]).tolist()) <= {0, 255}   # 半透明なし
    spec = X.fx_spec()
    assert [e["loop"] for e in spec["effects"]] == [False, False, True, True]


def test_bad_tables_are_rejected():
    with pytest.raises(X.FxError):
        X.parse_part(".o.\n.oo.")
    with pytest.raises(X.FxError):
        X.parse_part(".q.")
    bad = {"x": {"anchor": "top", "loop": True, "parts": {"a": "o"}, "frames": [([("b", 0, 0)], 100)]}}
    with pytest.raises(X.FxError):
        X.load_effect("x", bad)
    with pytest.raises(X.FxError):
        X.load_effect("nothing")


# ---- 置く場所・色 ----

def test_outline_is_darkest_edge_color():
    assert X.outline_color(_char()) == DARK[:3]
    assert X.outline_color(np.zeros((4, 4, 4), np.uint8)) == art.OUTLINE_FALLBACK


def test_top_and_side_positions():
    c = _char()
    head = X.head_box(c)
    assert head == {"top": 2, "rows": 6, "left": 6, "right": 13}   # 高さ 26 の 1/4 ≒ 6 行
    s = X.make_fx("surprise", [c])
    w, h = s.size
    assert s.position == ((6 + 13 + 1 - w) // 2, 2 - X.GAP - h)
    assert s.position[1] + h < head["top"]                        # 頭に重ならない
    t = X.make_fx("sweat", [c])
    assert t.position[0] == 13 + 1 + X.GAP
    o = X.make_fx("sweat", [c], offset=(2, -3))
    assert o.position == (t.position[0] + 2, t.position[1] - 3)


def test_mirror_flips_position_but_not_art():
    c = _char()
    m = c[:, ::-1].copy()
    for n in X.effect_names():
        a = X.make_fx(n, [c], offset=(2, 1))
        b = X.make_fx(n, [m], mirror=True, offset=(2, 1))
        w, _ = a.size
        assert b.position == (c.shape[1] - a.position[0] - w, a.position[1])
        assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))   # 絵は反転しない
        assert b.mirrored and not a.mirrored


# ---- 終わり方 ----

def test_once_keep_and_hide():
    c = _char()
    keep = X.make_fx("surprise", [c])
    hide = X.make_fx("surprise", [c], end="hide")
    assert keep.end == "keep" and hide.end == "hide"
    assert len(hide.frames) == len(keep.frames) + 1 and hide.durations[-1] == X.HIDE_MS
    assert (keep.frames[-1][..., 3] > 0).any() and not (hide.frames[-1][..., 3] > 0).any()
    info = X.fx_info(hide, "a.png", "b.png")
    assert info["loop"] is False and info["frames"][-1]["empty"] is True
    assert "empty" not in X.fx_info(keep, "a.png", "b.png")["frames"][-1]
    # 重ねた見た目: 残すなら最後まで見え、消すなら最後は見えない
    ch = [c, c]
    fk, _, vk = X.composite(ch, [500, 500], keep)
    fh, _, vh = X.composite(ch, [500, 500], hide)
    region = lambda f, v, fx: f[v["fx_y"]:v["fx_y"] + fx.size[1], v["fx_x"]:v["fx_x"] + fx.size[0], 3]
    assert (region(fk[-1], vk, keep) > 0).any()
    assert not (region(fh[-1], vh, hide) > 0).any()
    assert vk["length_ms"] % 1000 == 0 and vk["length_ms"] >= sum(keep.durations) + X.HOLD_MS
    # くり返すものは終わり方を持たない
    assert X.make_fx("heart", [c], end="hide").end is None
    with pytest.raises(X.FxError):
        X.make_fx("surprise", [c], end="later")
    with pytest.raises(X.FxError):
        X.make_fx("surprise", [c], offset=(X.MAX_OFFSET + 1, 0))


def test_composite_keeps_character_pixels_and_loops():
    c = _char()
    res = make_motion(make_pixel_sprite(), "breathe")
    fx = X.make_fx("heart", res.frames)
    frames, durations, v = X.composite(res.frames, res.durations, fx)
    assert v["length_ms"] == 1000 and sum(durations) == 1000
    ch, cw = res.frames[0].shape[:2]
    assert np.array_equal(frames[0][v["char_y"]:v["char_y"] + ch, v["char_x"]:v["char_x"] + cw], res.frames[0])
    gif = X.combined_gif(res.frames, res.durations, fx, 2)
    assert gif.data[:6] == b"GIF89a"
    text, notes = X.combined_piskel(res.frames, res.durations, fx, "x")
    pk = read_piskel(text)
    assert [l["name"] for l in pk.layers] == ["キャラ", "エフェクト"] and pk.fps == 20 and len(pk.frames) == 20
    # 時間をそろえられないときは作らない
    t2, n2 = X.combined_piskel([c], [37], X.make_fx("heart", [c]), "x")
    assert t2 is None and n2


def test_file_names():
    fx = X.make_fx("sleep", [_char()])
    w, h = fx.size
    assert X.fx_file_name("hero_breathe_sheet.png", fx) == "hero_breathe_fx_sleep.png"
    assert X.fx_file_name("hero_walk_20x22_3f.png", fx) == f"hero_walk_fx_sleep_{w}x{h}_4f.png"
    assert X.fx_stem("hero_walk_mirror_20x22_3f.png", "sleep") == "hero_walk_mirror_fx_sleep"


# ---- CLI と画面 ----

def _files(d):
    return {p.name: p.read_bytes() for p in d.iterdir()}


def test_cli_no_fx_output_is_unchanged(tmp_path, pixel_png):
    from apps.sprite import cli
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main([str(pixel_png), "-o", str(a / "hero.png"), "-q", "--motion", "breathe", "--gif"]) == 0
    assert cli.main([str(pixel_png), "-o", str(b / "hero.png"), "-q", "--motion", "breathe", "--gif",
                     "--fx", "surprise"]) == 0
    fa, fb = _files(a), _files(b)
    assert not any("_fx_" in n for n in fa)
    assert {n: v for n, v in fb.items() if "_fx_" not in n} == fa   # キャラのシートは変わらない
    assert {"hero_breathe_fx_surprise.png", "hero_breathe_fx_surprise.json", "hero_breathe_fx_surprise.gif"} <= set(fb)


def test_cli_matches_web_bytes(tmp_path, pixel_png):
    from apps.sprite import cli
    import base64
    d = tmp_path / "o"
    assert cli.main([str(pixel_png), "-o", str(d / "hero.png"), "-q", "--motion", "sway", "--fx", "sweat",
                     "--fx-offset=1,-2", "--fx-end", "hide"]) == 0
    sheet = (d / "hero_sway_sheet.png").read_bytes()
    sj = (d / "hero_sway_sheet.json").read_text(encoding="utf-8")
    r = json.loads(X.fx_for_web(sheet, sj, json.dumps({"fx": "sweat", "offset": [1, -2], "end": "hide"})))
    assert r["ok"], r
    assert r["name"] == "hero_sway_fx_sweat.png"
    assert base64.b64decode(r["png_base64"]) == (d / r["name"]).read_bytes()
    assert r["json"].encode("utf-8") == (d / "hero_sway_fx_sweat.json").read_bytes()
    info = json.loads(r["json"])
    assert info["position"] == {"x": r["position"][0], "y": r["position"][1]} and info["offset"] == {"x": 1, "y": -2}
    bad = json.loads(X.fx_for_web(sheet, sj, json.dumps({"fx": "nothing"})))
    assert not bad["ok"]


def test_cli_frames_mirror_piskel_and_web(tmp_path):
    from apps.sprite import cli
    from test_sprite_frames import make_strip
    import base64
    src = tmp_path / "walk.png"
    save_rgba(make_strip(), src)
    d = tmp_path / "o"
    assert cli.main([str(src), "-o", str(d / "hero.png"), "-q", "--frames", "--mirror", "--piskel", "--gif",
                     "--fx", "sleep", "--fx-offset=2,0"]) == 0
    names = set(_files(d))
    main = next(n for n in names if n.startswith("hero_frames_") and "_fx_" not in n and "mirror" not in n
                and n.endswith(".png"))
    mirror = main.replace("hero_frames_", "hero_frames_mirror_")
    fxs = sorted(n for n in names if "_fx_sleep" in n and n.endswith(".png"))
    assert len(fxs) == 2
    fxs = [next(n for n in fxs if "mirror" in n), next(n for n in fxs if "mirror" not in n)]   # [反転, 反転なし]
    assert {"hero_frames_fx_sleep.gif", "hero_frames_fx_sleep.piskel", "hero_frames_mirror_fx_sleep.gif",
            "hero_frames_mirror_fx_sleep.piskel"} <= names
    pk = read_piskel((d / "hero_frames_fx_sleep.piskel").read_bytes())
    assert [l["name"] for l in pk.layers] == ["キャラ", "エフェクト"]
    a = json.loads((d / fxs[1].replace(".png", ".json")).read_text(encoding="utf-8"))
    b = json.loads((d / fxs[0].replace(".png", ".json")).read_text(encoding="utf-8"))
    assert a["mirrored"] is False and b["mirrored"] is True
    cw = a["character_frame"]["width"]
    assert b["position"]["x"] == cw - a["position"]["x"] - a["frame_width"]
    for sheet_name, fx_name in ((main, fxs[1]), (mirror, fxs[0])):
        r = json.loads(X.fx_for_web((d / sheet_name).read_bytes(),
                                    (d / sheet_name.replace(".png", ".json")).read_text(encoding="utf-8"),
                                    json.dumps({"fx": "sleep", "offset": [2, 0], "piskel": True})))
        assert r["ok"] and r["name"] == fx_name
        assert base64.b64decode(r["png_base64"]) == (d / fx_name).read_bytes()
        assert r["json"].encode("utf-8") == (d / fx_name.replace(".png", ".json")).read_bytes()
        assert r["piskel"].encode("ascii") == (d / r["piskel_name"]).read_bytes()


@pytest.mark.parametrize("argv", [
    ["--fx", "surprise"],                                    # 動きもコマも無い
    ["--motion", "breathe", "--fx-offset", "1,1"],           # --fx が無い
    ["--motion", "breathe", "--fx", "heart", "--fx-end", "hide"],   # くり返すものに終わり方
    ["--motion", "breathe", "--fx", "surprise", "--fx-offset", "1"],
])
def test_cli_fx_option_errors(tmp_path, pixel_png, argv):
    from apps.sprite import cli
    with pytest.raises(SystemExit) as e:
        cli.main([str(pixel_png), "-o", str(tmp_path / "x.png"), "-q"] + argv)
    assert e.value.code == 2
