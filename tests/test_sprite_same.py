"""1 コマ目と同じにする範囲（`retrocore/sprite/same.py`、`--frames --same-as-first`）のテスト。

テスト画像はコードで合成する（`test_sprite_steady.make_frame()`: 暗い輪郭・髪・顔・2 ドットの目の頭と、体・腕）。
2 コマ目以降の頭を描き直したり（輪郭・点の違い）、ずらしたり、腕を頭の範囲に入れたりして、範囲の中身が 1 コマ目と同じになること、
写せないときは何もしないことを確かめる。
"""

import base64
import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import same as Sm
from test_sprite_steady import BODY, EYE, HAIR, OUT, SPOT, H, W, make_frame

HEAD = Sm.SameRange(0, 0, W, 21)   # 頭（2〜19 行）を囲む四角。下の端は体の上


def colors(frames):
    return {tuple(c) for f in frames for c in f[f[..., 3] > 0][:, :3].tolist()}


def redraw_head(f, dy=0):
    """AI の描き直しのような頭の違い: 輪郭が 1 ドット外に出た所・髪の中の点・ハイライトが消えた所。"""
    f = f.copy()
    f[3 + dy:9 + dy, 4] = OUT          # 左の輪郭が 1 ドット外
    f[2 + dy:9 + dy, 5] = HAIR
    f[5 + dy, 14] = SPOT              # 髪の中の点
    f[6 + dy, 16:19] = SPOT
    f[4 + dy, 10] = HAIR              # ハイライトが消えた
    return f


def test_head_becomes_same_as_first_and_body_stays():
    ref = make_frame()
    f = redraw_head(make_frame(arm=4))   # 腕が頭の右の空いた所（範囲の中）まで上がってきた
    out, rep = Sm.same_frames([ref, f], [HEAD])
    g = out[1]
    rr = rep.frames[0].ranges[0]
    assert rr.copied and rr.code is None and rr.shift == (0, 0) and rr.match > 0.8, (rr.code, rr.seam)
    # 頭（顔・目・描き直した所）は 1 コマ目と同じ。範囲の外の体・脚は元のまま
    assert np.array_equal(g[0:15, 0:22], ref[0:15, 0:22])
    assert np.array_equal(g[24:], f[24:])
    # 腕（範囲の外から入ってきた太い部品）は消さずに残す
    assert rr.kept > 0
    assert np.array_equal(g[14:26, 26:30], f[14:26, 26:30])
    assert np.array_equal(out[0], ref)
    assert colors(out) <= colors([ref, f])
    assert rr.changed == int((Sm._keys(g) != Sm._keys(f)).sum())


def test_moved_head_is_copied_at_found_position():
    """頭が 1 ドット下・1 ドット右にずれたコマ: そのずれた位置に 1 コマ目の頭を写す（目は 1 コマ目の形のまま、ずれた所）。"""
    ref = make_frame()
    f = make_frame()
    f[0:20] = 0
    f[20:] = ref[20:]
    head = ref[2:20, 5:25]
    f[3:21, 6:26] = np.where(head[..., 3:4] > 0, head, f[3:21, 6:26])
    f = redraw_head(f, dy=1)
    out, rep = Sm.same_frames([ref, f], [Sm.SameRange(0, 0, W, 20)], search=2)
    rr = rep.frames[0].ranges[0]
    assert rr.shift == (1, 1) and rr.copied
    g = out[1]
    assert np.array_equal(g[3:15, 6:26], ref[2:14, 5:25])
    assert (g[14:16, 12] == EYE).all() and (g[14:16, 19] == EYE).all()


def test_low_match_is_not_copied():
    ref = make_frame()
    f = make_frame()
    f[0:20] = 0
    f[4:20, 2:28] = BODY             # 頭が全く違うもの（合い方が低い）
    out, rep = Sm.same_frames([ref, f], [Sm.SameRange(0, 0, W, 20)])
    rr = rep.frames[0].ranges[0]
    assert not rr.copied and rr.code == Sm.CODE_LOW and rr.match < Sm.MIN_MATCH
    assert np.array_equal(out[1], f)
    assert any("合い方" in w for w in rep.warnings)
    assert any("次に試せること" in w for w in rep.warnings)


def test_seam_problem_is_not_copied_and_marked():
    """範囲の下の端が頭の途中を横切り、そのコマの頭が 3 ドット横にずれている（探す幅 0）: 境目で輪郭が切れるので写さない。"""
    ref = make_frame()
    f = np.zeros_like(ref)
    f[20:] = ref[20:]
    head = ref[2:20, 5:25]
    f[2:20, 8:28] = np.where(head[..., 3:4] > 0, head, f[2:20, 8:28])
    out, rep = Sm.same_frames([ref, f], [Sm.SameRange(0, 0, W, 11)], search=0)
    rr = rep.frames[0].ranges[0]
    assert not rr.copied and rr.code == Sm.CODE_SEAM and rr.seam
    assert np.array_equal(out[1], f)
    img = Sm.same_image([ref, f], out, rep, 4)
    assert (np.all(img[..., :3] == (255, 100, 0), axis=-1)).any()     # 境目の問題の印（橙）
    sugg = rep.suggestions[0]
    assert sugg["range"] == 1 and "seam" in sugg["codes"] and sugg["tips"]


def test_intruding_part_touching_eye_is_not_copied():
    ref = make_frame()
    f = make_frame()
    f[11:14, 12:18] = BODY           # 目のすぐ横まで入ってきた腕（範囲の外とつながる）
    f[11:14, 12:30] = BODY
    out, rep = Sm.same_frames([ref, f], [Sm.SameRange(0, 0, 26, 21)])
    rr = rep.frames[0].ranges[0]
    # 範囲の中で 1 コマ目は髪・顔、コマは腕: 腕は 1 コマ目の不透明な所に重なるので「入ってきた部品」ではない。
    # 写すと腕が切れるので、境目か部品の決まりで写さない
    assert not rr.copied and rr.code in (Sm.CODE_SEAM, Sm.CODE_PART, Sm.CODE_HOLE)
    assert np.array_equal(out[1], f)


def test_foot_warning_and_feet_do_not_move():
    ref = make_frame()
    f = make_frame(arm=1)
    out, rep = Sm.same_frames([ref, f], [Sm.SameRange(0, 30, W, 10)])
    rr = rep.frames[0].ranges[0]
    assert rr.foot_warning
    assert any("足元" in w for w in rep.warnings)
    bottom = lambda a: int(np.flatnonzero((a[..., 3] > 0).any(1))[-1])   # noqa: E731
    assert bottom(out[1]) == bottom(f)


def test_several_ranges_and_reference_frame():
    ref = make_frame()
    f2 = redraw_head(make_frame())
    f3 = redraw_head(make_frame())
    ranges = [Sm.SameRange(0, 0, 15, 21), Sm.SameRange(15, 0, 15, 21)]
    out, rep = Sm.same_frames([ref, f2, f3], ranges)
    assert [len(f.ranges) for f in rep.frames] == [2, 2]
    assert np.array_equal(out[1], out[2])
    # 元にするコマを 3 コマ目に（1 コマ目が 3 コマ目に合わせられる）
    out2, rep2 = Sm.same_frames([ref, f2, f3], [HEAD], reference=3)
    assert rep2.reference == 3 and [f.frame for f in rep2.frames] == [1, 2]
    assert np.array_equal(out2[2], f3)
    with pytest.raises(ValueError):
        Sm.same_frames([ref, f2], [HEAD], reference=3)


def test_parse_and_fit_ranges():
    assert Sm.parse_rect("1, 2,3,4") == Sm.SameRange(1, 2, 3, 4)
    b = Sm.parse_band("5,9")
    assert b.kind == "band" and b.y == 5 and b.h == 5
    for bad in ("1,2,3", "a,b,c,d", "-1,0,3,3", "0,0,0,3"):
        with pytest.raises(ValueError):
            Sm.parse_rect(bad)
    with pytest.raises(ValueError):
        Sm.parse_band("9,5")
    fit = Sm.fit_ranges([b, Sm.SameRange(25, 35, 10, 10)], W, H)
    assert fit[0] == Sm.SameRange(0, 5, W, 5, "band") and fit[1] == Sm.SameRange(25, 35, 5, 5)
    with pytest.raises(ValueError, match="外"):
        Sm.fit_ranges([Sm.SameRange(W, 0, 3, 3)], W, H)
    assert Sm.range_from_dict({"type": "band", "y": 2, "h": 3}).kind == "band"
    for bad in ({"x": 0, "y": 0, "w": "3", "h": 3}, {"x": 0, "y": 0, "w": 3, "h": 3, "z": 1}, {"type": "circle", "y": 0, "h": 1}):
        with pytest.raises(ValueError):
            Sm.range_from_dict(bad)
    with pytest.raises(ValueError):
        Sm.validate_search(9)


# ---- コマの変換（convert_frames）・シート・CLI・画面 ----

@pytest.fixture(scope="module")
def strip():
    from test_sprite_frames import make_strip
    return make_strip()


@pytest.fixture(scope="module")
def plain(strip):
    return F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))


def head_range(res):
    w, h = res.size
    return Sm.SameRange(0, 0, w, h // 2)


def test_no_range_is_unchanged(strip, plain):
    assert plain.same is None and plain.aligned is not None
    info = F.frames_sheet_info(F.assemble_frames(plain, [1, 2, 3, 2]), "x.png")
    assert "same_as_first" not in info
    assert all(e["stage"] != "same_as_first" for e in info["skipped"])


def test_convert_with_range_records_json_and_pingpong(strip, plain):
    fs = F.FrameSettings(frame_count=3, same_ranges=(head_range(plain),))
    res = F.convert_frames(strip, SpriteSettings(), fs)
    assert res.same is not None and res.size == plain.size and res.pivot == plain.pivot
    for a, b in zip(res.aligned, plain.frames):
        assert np.array_equal(a, b)
    anim = F.assemble_frames(res, [1, 2, 3, 2], mirror=True)
    info = json.loads(F.frames_sheet_json(anim, "x.png", res.skipped))
    keys = list(info)
    assert keys.index("same_as_first") == keys.index("source") + 1
    sa = info["same_as_first"]
    assert sa["reference_frame"] == 1 and sa["search"] == 3 and sa["ranges"][0]["type"] == "rect"
    assert [f["source_frame"] for f in sa["frames"]] == [2, 3]
    for f in sa["frames"]:
        r = f["ranges"][0]
        assert set(r) >= {"range", "shift", "match", "copied", "changed_dots", "kept_dots"}
        if not r["copied"]:
            assert r["code"] in Sm.CODE_TEXT
    assert np.array_equal(anim.frames[1], anim.frames[3])       # 行って戻るの同じコマは同じ
    assert colors(res.frames) <= colors(plain.frames)
    # 写さなかった範囲は skipped にも入る
    held = [e for e in info["skipped"] if e["stage"] == "same_as_first"]
    assert len(held) == sum(not f["ranges"][0]["copied"] for f in sa["frames"])
    # 足元は動かない
    for a, b in zip(res.frames, plain.frames):
        assert int(np.flatnonzero((a[..., 3] > 0).any(1))[-1]) == int(np.flatnonzero((b[..., 3] > 0).any(1))[-1])


@pytest.mark.parametrize("outline,steady", [("auto", "on"), ("thin", "off"), ("thin", "on")])
def test_range_with_outline_and_steady(strip, plain, outline, steady):
    rng = head_range(plain)
    fs = F.FrameSettings(frame_count=3, steady=steady, same_ranges=(rng,))
    res = F.convert_frames(strip, SpriteSettings(outline=outline), fs)
    assert res.same is not None
    assert (res.steady is not None) == (steady == "on")
    if steady == "on":
        # 範囲を写した後にちらつきを減らす（ちらつきを減らす前のコマ = 範囲を写した結果）
        alone = F.convert_frames(strip, SpriteSettings(outline=outline),
                                 F.FrameSettings(frame_count=3, same_ranges=(rng,)))
        for a, b in zip(res.before_steady, alone.frames):
            assert np.array_equal(a, b)
        info = F.frames_sheet_info(F.assemble_frames(res), "x.png")
        assert list(info).index("steady") == list(info).index("same_as_first") + 1


def test_refinish_matches_convert(strip, plain):
    fs = F.FrameSettings(frame_count=3, steady="on", same_ranges=(head_range(plain), Sm.SameRange(0, 0, 4, 4)))
    want = F.convert_frames(strip, SpriteSettings(), fs)
    base = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, steady="on"))
    got = F.refinish_frames(base, fs)
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
    assert got.same.info() == want.same.info() and got.skipped == want.skipped and got.warnings == want.warnings
    # 範囲を消すと元に戻る
    back = F.refinish_frames(got, F.FrameSettings(frame_count=3, steady="on"))
    assert back.same is None and all(np.array_equal(a, b) for a, b in zip(back.frames, base.frames))
    assert back.warnings == base.warnings and back.skipped == base.skipped


def test_frame_settings_check_same():
    with pytest.raises(ValueError):
        F.FrameSettings(same_search=9).validate()
    with pytest.raises(ValueError):
        F.FrameSettings(same_ranges=((0, 0, 3, 3),)).validate()
    with pytest.raises(ValueError):
        F.FrameSettings(same_from=0).validate()


@pytest.mark.parametrize("argv", [["--frames", "--same-search", "2"], ["--frames", "--same-as-first", "1,2,3"],
                                  ["--frames", "--same-band", "5,1"], ["--same-as-first", "0,0,3,3"],
                                  ["--frames", "--same-as-first", "0,0,3,3", "--same-search", "9"],
                                  ["--frames", "--frame-count", "2", "--same-as-first", "0,0,3,3", "--same-from", "3"]])
def test_cli_same_option_errors(tmp_path, argv):
    from apps.sprite.cli import main
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.png")] + argv)


def test_cli_same_writes_json_and_image(tmp_path, strip, plain):
    from apps.sprite.cli import main
    from retrocore.sprite import load_rgba, save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    out = tmp_path / "o" / "hero.png"
    w, h = plain.size
    assert main([str(src), "-o", str(out), "-q", "--frames", "--frame-count", "3", "--same-as-first", f"0,0,{w},{h // 2}",
                 "--same-band", "0,3", "--same-search", "2", "--frame-order", "1,2,3,2"]) == 0
    js = json.loads(next(out.parent.glob("hero_frames_*.json")).read_text(encoding="utf-8"))
    assert js["same_as_first"]["search"] == 2 and [r["type"] for r in js["same_as_first"]["ranges"]] == ["rect", "band"]
    assert load_rgba(out.parent / "hero_frames_same.png").shape[2] == 4
    # 範囲がコマの外ならエラー（何も書かない）
    bad = tmp_path / "bad" / "hero.png"
    assert main([str(src), "-o", str(bad), "-q", "--frames", "--same-as-first", f"{w + 5},0,3,3"]) == 1
    assert not bad.exists()


def test_web_same_match_cli_bytes(tmp_path, strip, plain):
    """画面で範囲を描いた結果（変換 → 範囲だけやり直す → シート）と、CLI の --same-as-first がバイト単位で一致する。"""
    from retrocore.sprite.convert import (encode_png, frames_convert_for_web, frames_same_for_web, frames_sheets_for_web,
                                          gif_for_web)
    from test_sprite_frames import _cli_files, _text_bytes
    w, h = plain.size
    for outline, steady in (("auto", "off"), ("thin", "on")):
        argv = ["--frame-count", "3", "--frame-name", "walk", "--steady", steady, "--frame-order", "1,2,3,2",
                "--same-as-first", f"0,0,{w},{h // 2}", "--same-band", "0,2", "--piskel"]
        if outline == "thin":
            argv += ["--outline", "thin"]
        d = tmp_path / outline
        d.mkdir()
        cli = _cli_files(d, strip, argv)
        conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps(
            {"settings": {"outline": outline}, "frames": {"frame_count": 3, "steady": steady}, "name": "walk"})))
        assert conv["ok"] and conv["made"] and conv["same"] is None and "aligned_png_base64" in conv
        res = json.loads(frames_same_for_web(json.dumps({"token": conv["token"], "same_search": 3, "same_ranges": [
            {"type": "rect", "x": 0, "y": 0, "w": w, "h": h // 2}, {"type": "band", "y": 0, "h": 3}]})))
        assert res["ok"], res
        assert res["same"]["ranges"][1] == {"type": "band", "x": 0, "y": 0, "w": w, "h": 3}
        web = {"hero_sprite.png": base64.b64decode(res["png_base64"]),
               "hero_sprite_walk_flicker.png": base64.b64decode(res["flicker_png_base64"]),
               "hero_sprite_walk_flicker.json": _text_bytes(res["flicker_json"]),
               "hero_sprite_walk_same.png": base64.b64decode(res["same_png_base64"])}
        sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite", "name": "walk",
                                                              "order": [1, 2, 3, 2], "mirror": True})))
        assert sheets["ok"], sheets
        for s in sheets["sheets"]:
            png = base64.b64decode(s["png_base64"])
            web[s["name"]] = png
            web[s["name"][:-4] + ".json"] = _text_bytes(s["json"])
            web[s["piskel_name"]] = s["piskel"].encode("ascii")
            g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
            web[g["name"]] = base64.b64decode(g["gif_base64"])
        assert sorted(web) == sorted(cli)
        for k in cli:
            assert web[k] == cli[k], (outline, k)
        assert '"same_as_first"' in sheets["sheets"][0]["json"]


def test_web_same_errors(strip):
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_same_for_web
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    for req in ({"token": -1, "same_ranges": []}, {"token": conv["token"], "same_ranges": [{"x": 0}]},
                {"token": conv["token"], "same_ranges": [{"x": 999, "y": 0, "w": 3, "h": 3}]},
                {"token": conv["token"], "same_search": 99}, {"token": conv["token"], "what": 1},
                {"token": conv["token"], "same_from": 9}):
        res = json.loads(frames_same_for_web(json.dumps(req)))
        assert not res["ok"] and res["error"], req
    # 範囲が空なら、変換したときと同じ
    res = json.loads(frames_same_for_web(json.dumps({"token": conv["token"], "same_ranges": []})))
    assert res["ok"] and res["same"] is None and res["png_base64"] == conv["png_base64"]
    assert not json.loads(frames_same_for_web("[1]"))["ok"]


def test_web_page_has_same_ranges():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('id="same-canvas"', 'id="same-search"', 'id="same-clear"', 'id="same-undo"', "frames_same_for_web",
                "--same-as-first", 'id="dl-same"',
                # 範囲を直す（辺・角・内側のドラッグ、選ぶ、矢印キー、元に戻す、計算中の表示）
                'id="same-back"', 'id="same-help"', "function sameHit", "function sameEdited", "SAME_CURSOR", "SAME_GRIP",
                "SAME_KEY_WAIT", "Shift＋矢印", "計算中"):
        assert key in html, key


def test_same_runs_with_int32_numpy(strip, plain):
    """32 ビットの NumPy のつもりで通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_same_for_web
    fs = F.FrameSettings(frame_count=3, steady="on", same_ranges=(head_range(plain),))
    want = F.convert_frames(strip, SpriteSettings(), fs)
    ref = make_frame()
    f = redraw_head(make_frame(arm=2))
    want_small = Sm.same_frames([ref, f], [HEAD])[0]
    seam_f = np.zeros_like(ref)
    seam_f[20:] = ref[20:]
    seam_f[2:20, 8:28] = np.where(ref[2:20, 5:25, 3:4] > 0, ref[2:20, 5:25], seam_f[2:20, 8:28])
    with emulate_int32():
        got = F.convert_frames(strip, SpriteSettings(), fs)
        got_small, rep = Sm.same_frames([ref, f], [HEAD])
        Sm.same_image([ref, f], got_small, rep, 2)
        _, rep2 = Sm.same_frames([ref, seam_f], [Sm.SameRange(0, 0, W, 11)], search=0)
        conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
        res = json.loads(frames_same_for_web(json.dumps({"token": conv["token"], "same_ranges": [
            {"x": 0, "y": 0, "w": plain.size[0], "h": plain.size[1] // 2}]})))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
    assert all(np.array_equal(a, b) for a, b in zip(want_small, got_small))
    assert rep2.frames[0].ranges[0].code == Sm.CODE_SEAM
    assert res["ok"] and res["same"] is not None
