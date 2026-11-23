"""手の直し（retrocore/sprite/touchup.py、CLI の --edits、ブラウザ版の「ドットを直す」）のテスト。テスト画像はコードで合成する。"""

import base64
import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import same as Sm
from retrocore.sprite import touchup as T


@pytest.fixture(scope="module")
def strip():
    from test_sprite_frames import make_strip
    return make_strip()


@pytest.fixture(scope="module")
def plain(strip):
    return F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))


def _hex(p) -> str:
    return "#%02x%02x%02x" % (int(p[0]), int(p[1]), int(p[2]))


def _opaque_and_clear(frame):
    """不透明なドット 1 つと、透明なドット 1 つの位置。"""
    ys, xs = np.nonzero(frame[..., 3] > 0)
    tys, txs = np.nonzero(frame[..., 3] == 0)
    return (int(xs[len(xs) // 2]), int(ys[len(ys) // 2])), (int(txs[0]), int(tys[0]))


def _edits_for(res, k=2):
    """元のコマ k に、透明の所へ色を 1 つ・不透明の所を透明に 1 つ直す一覧（下の色つき）。"""
    f = res.frames[k - 1]
    (ox, oy), (tx, ty) = _opaque_and_clear(f)
    pal = T.palette_of(res.frames)
    color = _hex(f[oy, ox])
    d = {"version": 1, "kind": "sprite_edits", "name": "walk", "frame_width": res.size[0], "frame_height": res.size[1],
         "frame_count": len(res.frames), "palette": pal,
         "frames": [{"source_frame": k, "dots": [[tx, ty, pal.index(color), None], [ox, oy, None, pal.index(color)]]}]}
    return d, (tx, ty), (ox, oy), color


# ---- 当てる ----

def test_apply_paints_and_erases_without_new_colors(plain):
    d, (tx, ty), (ox, oy), color = _edits_for(plain)
    t = T.touchup_from_dict(d)
    frames, rep = T.apply_touchup(plain.frames, t)
    assert _hex(frames[1][ty, tx]) == color and frames[1][ty, tx, 3] == 255
    assert tuple(frames[1][oy, ox]) == (0, 0, 0, 0)
    # ほかのコマ・ほかのドットは変わらない。元のコマも変えない
    assert np.array_equal(frames[0], plain.frames[0]) and np.array_equal(frames[2], plain.frames[2])
    diff = np.any(frames[1] != plain.frames[1], axis=2)
    assert int(diff.sum()) == 2
    assert plain.frames[1][ty, tx, 3] == 0
    # 新しい色は無い
    assert set(T.palette_of(frames)) <= set(T.palette_of(plain.frames))
    assert rep.dot_count == 2 and rep.changed_under_count == 0 and not rep.warnings


def test_pingpong_same_source_frame_gets_edit_twice_and_mirror(plain):
    d, (tx, ty), _, color = _edits_for(plain, k=2)
    res = T.touchup_result(plain, T.touchup_from_dict(d))
    anim = F.assemble_frames(res, [1, 2, 3, 2])
    assert _hex(anim.frames[1][ty, tx]) == color and _hex(anim.frames[3][ty, tx]) == color
    assert anim.frames[0][ty, tx, 3] == plain.frames[0][ty, tx, 3]
    mir = F.assemble_frames(res, [1, 2, 3, 2], mirror=True)
    w = res.size[0]
    assert _hex(mir.frames[3][ty, w - 1 - tx]) == color
    info = json.loads(F.frames_sheet_json(anim, "x.png", res.skipped))
    keys = list(info)
    assert keys.index("edits") == keys.index("source") + 1 and keys.index("palette") == keys.index("edits") + 1
    e = info["edits"]
    assert e["dot_count"] == 2 and e["frames"][0]["source_frame"] == 2
    assert e["frames"][0]["dots"][0][2] is None or isinstance(e["frames"][0]["dots"][0][2], int)
    assert e["palette"] == T.palette_of(plain.frames)


def test_no_edits_is_byte_identical(strip, plain):
    a = F.frames_sheet_json(F.assemble_frames(plain, [1, 2, 3, 2]), "x.png", plain.skipped)
    res = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, edits=T.Touchup()))
    assert res.touchup is None and res.before_touchup is None
    b = F.frames_sheet_json(F.assemble_frames(res, [1, 2, 3, 2]), "x.png", res.skipped)
    assert a == b and '"edits"' not in a
    assert T.touchup_result(plain, None) is plain


def test_convert_with_edits_matches_touchup_result(strip, plain):
    d, *_ = _edits_for(plain, k=3)
    t = T.touchup_from_dict(d)
    res = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, edits=t))
    want = T.touchup_result(plain, t)
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, want.frames))
    assert res.touchup.info() == want.touchup.info()
    assert any(s.name == "touchup" for s in res.steps)
    # 範囲をやり直しても直しは上に重ねる
    fs = F.FrameSettings(frame_count=3, edits=t, same_ranges=(Sm.SameRange(0, 0, res.size[0], 4),))
    again = F.refinish_frames(plain, fs)
    assert again.touchup is not None and again.touchup.dot_count == 2


def test_changed_under_is_counted_and_warned(plain):
    d, (tx, ty), _, _ = _edits_for(plain)
    pal = d["palette"]
    other = next(i for i, c in enumerate(pal))
    d["frames"][0]["dots"][0][3] = other   # 直したときは色があった（いまは透明）
    frames, rep = T.apply_touchup(plain.frames, T.touchup_from_dict(d))
    assert rep.changed_under_count == 1 and rep.frames[0].changed_under == [(tx, ty)]
    assert rep.warnings and "1 個" in rep.warnings[0] and "2 コマ目 1" in rep.warnings[0]


@pytest.mark.parametrize("change, code", [
    (lambda d: d.update(frame_width=d["frame_width"] + 1), T.CODE_SIZE),
    (lambda d: d["frames"][0].update(source_frame=9), T.CODE_FRAME),
    (lambda d: d.update(palette=["#123456"] + d["palette"][1:]), T.CODE_PALETTE),
    (lambda d: d["frames"][0]["dots"].append([999, 0, None]), T.CODE_OUTSIDE),
])
def test_not_applicable(plain, change, code):
    d, *_ = _edits_for(plain)
    d["frames"][0]["dots"][0][2] = 0
    change(d)
    with pytest.raises(T.TouchupError) as e:
        T.apply_touchup(plain.frames, T.touchup_from_dict(d))
    assert e.value.code == code and str(e.value)


@pytest.mark.parametrize("bad", [
    [], {"kind": "other"}, {"version": 99}, {"what": 1}, {"palette": ["red"]}, {"frames": [{"dots": []}]},
    {"frames": [{"source_frame": 0, "dots": []}]}, {"frames": [{"source_frame": 1, "dots": [[0, 0]]}]},
    {"palette": ["#000000"], "frames": [{"source_frame": 1, "dots": [[0, 0, 1]]}]},
    {"frames": [{"source_frame": 1, "dots": [[-1, 0, None]]}]}, {"frame_width": 0}, {"name": "a b"},
    {"frames": [{"source_frame": True, "dots": []}]},
])
def test_bad_files(bad):
    with pytest.raises(ValueError):
        T.touchup_from_dict(bad)
    with pytest.raises(ValueError):
        T.load_touchup("{")


def test_file_roundtrip_and_duplicates(plain):
    d, *_ = _edits_for(plain)
    d["frames"].append({"source_frame": 2, "dots": [[1, 1, None], [1, 1, 0]]})   # 同じ場所は後のものを使う
    t = T.touchup_from_dict(d)
    assert t.dot_count == 3
    text = T.touchup_json(t, T.palette_of(plain.frames))
    assert text.endswith("}\n")
    t2 = T.load_touchup(text)
    assert t2.frames == t.frames and t2.frame_width == t.frame_width and t2.name == "walk"
    assert json.loads(text)["dot_count"] == 3
    assert T.edits_file_name("hero_sprite", "walk") == "hero_sprite_walk_edits.json"
    assert all([p.y, p.x] == sorted([[q.y, q.x] for q in t.frames[2]])[i] for i, p in enumerate(t.frames[2]))


# ---- CLI と画面 ----

def test_cli_edits_errors(tmp_path, strip, plain):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    with pytest.raises(SystemExit):
        main([str(src), "--edits", "x.json"])    # --frames が無い
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert main([str(src), "-o", str(tmp_path / "o.png"), "-q", "--frames", "--frame-count", "3", "--edits", str(bad)]) == 1
    d, *_ = _edits_for(plain)
    d["frame_width"] += 1
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(d), encoding="utf-8")
    assert main([str(src), "-o", str(tmp_path / "o.png"), "-q", "--frames", "--frame-count", "3", "--edits", str(wrong)]) == 1


def test_web_edits_match_cli_bytes(tmp_path, strip, plain):
    """画面で直した結果（変換 → 範囲 → 直す → シート）と、CLI の --edits がバイト単位で一致する。"""
    from retrocore.sprite.convert import (encode_png, frames_convert_for_web, frames_edits_for_web, frames_same_for_web,
                                          frames_sheets_for_web, gif_for_web)
    from test_sprite_frames import _cli_files, _text_bytes
    w, h = plain.size
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps(
        {"frames": {"frame_count": 3, "steady": "on"}, "name": "walk"})))
    assert conv["ok"] and conv["touchup"] is None and conv["touchup_palette"] and "touchup_base_png_base64" in conv
    same = json.loads(frames_same_for_web(json.dumps({"token": conv["token"], "same_ranges": [
        {"type": "rect", "x": 0, "y": 0, "w": w, "h": h // 2}]})))
    assert same["ok"] and same["touchup_error"] is None
    from retrocore.sprite.convert import _FRAMES_CACHE
    d, *_ = _edits_for(_FRAMES_CACHE["finished"], k=2)
    ed = json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": d})))
    assert ed["ok"] and ed["applied"], ed
    assert ed["touchup"]["dot_count"] == 2 and ed["edits"]["dot_count"] == 2 and ed["edits_json"]
    ef = tmp_path / "hero_sprite_walk_edits.json"
    ef.write_text(ed["edits_json"], encoding="utf-8")
    argv = ["--frame-count", "3", "--frame-name", "walk", "--steady", "on", "--frame-order", "1,2,3,2",
            "--same-as-first", f"0,0,{w},{h // 2}", "--piskel", "--edits", str(ef)]
    cli = _cli_files(tmp_path, strip, argv)
    web = {"hero_sprite.png": base64.b64decode(ed["png_base64"]),
           "hero_sprite_walk_flicker.png": base64.b64decode(ed["flicker_png_base64"]),
           "hero_sprite_walk_flicker.json": _text_bytes(ed["flicker_json"]),
           "hero_sprite_walk_same.png": base64.b64decode(ed["same_png_base64"])}
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
        assert web[k] == cli[k], k
    assert '"edits"' in sheets["sheets"][0]["json"] and '"edits"' in sheets["sheets"][1]["json"]
    # 範囲の図は直しの前のコマで描く（直しを足しても変わらない）
    assert ed["same_png_base64"] == same["same_png_base64"]


def test_web_edits_reapplied_and_cleared(strip, plain):
    from retrocore.sprite.convert import (_FRAMES_CACHE, encode_png, frames_convert_for_web, frames_edits_for_web,
                                          frames_same_for_web)
    w, h = plain.size
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    d, (tx, ty), _, _ = _edits_for(_FRAMES_CACHE["finished"], k=2)
    ed = json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": d})))
    assert ed["applied"] and not ed["touchup_warnings"]
    # 前の段（範囲）を変えると、同じ直しを重ね直す。下の色が変わったドットは数えて知らせる
    same = json.loads(frames_same_for_web(json.dumps({"token": conv["token"], "same_ranges": [
        {"type": "rect", "x": 0, "y": 0, "w": w, "h": h}]})))
    assert same["ok"] and same["touchup"]["dot_count"] == 2
    changed = sum(len(f["dots"]) for f in same["touchup_changed"])
    assert len(same["touchup_warnings"]) == (1 if changed else 0)
    # 当てられない直し: 覚えている直しは変えない
    bad = dict(d, frame_height=h + 5)
    r = json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": bad})))
    assert r["ok"] and not r["applied"] and r["code"] == T.CODE_SIZE
    assert _FRAMES_CACHE["edits"] is not None
    # 全部消す
    r = json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": None})))
    assert r["applied"] and r["touchup"] is None and r["edits_json"] is None
    assert r["png_base64"] == same["touchup_base_png_base64"]
    for req in ({"token": -1}, {"token": conv["token"], "what": 1}, {"token": conv["token"], "edits": {"kind": "x"}}):
        assert not json.loads(frames_edits_for_web(json.dumps(req)))["ok"]
    assert not json.loads(frames_edits_for_web("[1]"))["ok"]


def test_web_restore_previous_conversion(strip, plain):
    """直しを当てられない確認の「やめる」: 1 つ前の変換（範囲・直しも）に戻せる。戻せるのは 1 回だけ。"""
    from retrocore.sprite.convert import (_FRAMES_CACHE, encode_png, frames_convert_for_web, frames_edits_for_web,
                                          frames_restore_for_web, frames_sheets_for_web)
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    d, *_ = _edits_for(_FRAMES_CACHE["finished"], k=2)
    assert json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": d})))["applied"]
    req = {"base": "hero_sprite", "name": "walk", "order": [1, 2, 3, 2]}
    before = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], **req})))["sheets"][0]
    # 縁取り add で変換し直す（大きさが変わり、直しは当てられない）
    conv2 = json.loads(frames_convert_for_web(encode_png(strip), json.dumps(
        {"settings": {"outline": "add"}, "frames": {"frame_count": 3}})))
    assert conv2["token"] == conv["token"] + 1
    r = json.loads(frames_edits_for_web(json.dumps({"token": conv2["token"], "edits": d})))
    assert r["ok"] and not r["applied"]
    # 戻す
    assert not json.loads(frames_restore_for_web(json.dumps({"token": conv2["token"]})))["ok"]
    back = json.loads(frames_restore_for_web(json.dumps({"token": conv["token"]})))
    assert back == {"ok": True, "token": conv["token"]}
    after = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], **req})))["sheets"][0]
    assert after["png_base64"] == before["png_base64"] and after["json"] == before["json"] and '"edits"' in after["json"]
    assert not json.loads(frames_restore_for_web(json.dumps({"token": conv["token"]})))["ok"]   # 2 回目は戻せない
    for bad in ("[1]", json.dumps({"token": 1, "x": 2})):
        assert not json.loads(frames_restore_for_web(bad))["ok"]


def test_web_page_has_editor():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('id="touch-box"', 'id="touch-canvas"', 'id="touch-zoom"', 'id="touch-grid"', 'id="touch-onion"',
                'id="touch-palette"', 'id="touch-preview"', 'id="touch-clear"', 'id="touch-save"', 'id="touch-load"',
                "frames_edits_for_web", "TOUCH_UNDO_MAX", "contextmenu", "--edits", "touchLine",
                # 直しを当てられないときの確認（ボタンの文字で中身が分かる 3 つ。「やめる」は前の結果に戻す）
                'id="touch-confirm"', 'id="touch-confirm-save"', 'id="touch-confirm-discard"', 'id="touch-confirm-cancel"',
                "直しを保存してから捨てる", "保存せずに捨てる", "やめる（変換し直しを取り消す）", "frames_restore_for_web",
                "function framesRestore"):
        assert key in html, key


def test_touchup_runs_with_int32_numpy(strip, plain):
    """32 ビットの NumPy のつもりで通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_edits_for_web
    d, *_ = _edits_for(plain, k=2)
    t = T.touchup_from_dict(d)
    want = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, edits=t))
    with emulate_int32():
        got = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, edits=T.touchup_from_dict(d)))
        conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
        ed = json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": d})))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
    assert ed["ok"] and ed["applied"]


def test_web_reselect_image_keeps_previous_and_counts_changed(strip, plain):
    """画面で直しがあるときに画像を選び直したときの流れ（Python の側は今までの関数だけで足りることを確かめる）:
    新しい画像を変換 → 大きさが同じなら同じ直しを当てられ、下の色が変わったドットの数が分かる（確認の窓に出す数）。
    「やめる（前の画像に戻す）」は 1 つ前の変換（前の画像の結果と直し）に戻す。大きさが違えば当てられない。"""
    from test_sprite_frames import make_strip
    from retrocore.sprite.convert import (_FRAMES_CACHE, encode_png, frames_convert_for_web, frames_edits_for_web,
                                          frames_restore_for_web, frames_sheets_for_web)
    req = {"base": "hero_sprite", "name": "walk", "order": [1, 2, 3, 2]}
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    finished = _FRAMES_CACHE["finished"]
    d, (tx, ty), (ox, oy), color = _edits_for(finished, k=2)
    assert json.loads(frames_edits_for_web(json.dumps({"token": conv["token"], "edits": d})))["applied"]
    before = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], **req})))["sheets"][0]

    # 「描き直した」画像（AI 風のにじみが違う。コマの大きさは同じ）
    other = make_strip(seed=1)
    conv2 = json.loads(frames_convert_for_web(encode_png(other), json.dumps({"frames": {"frame_count": 3}})))
    assert (conv2["frame_width"], conv2["frame_height"]) == (conv["frame_width"], conv["frame_height"])
    new = _FRAMES_CACHE["finished"].frames[1]
    r = json.loads(frames_edits_for_web(json.dumps({"token": conv2["token"], "edits": d})))
    assert r["ok"] and r["applied"]
    under = {(tx, ty): None, (ox, oy): color}
    want = sorted([x, y] for (x, y), was in under.items()
                  if (None if new[y, x, 3] == 0 else _hex(new[y, x])) != was)
    got = sorted(p for f in (r["touchup_changed"] or []) for p in f["dots"])
    assert [list(p) for p in got] == want
    assert len(r["touchup_warnings"]) == (1 if want else 0)

    # 「やめる（前の画像に戻す）」: 前の画像の結果と直しに戻る（シートはバイト単位で同じ）
    back = json.loads(frames_restore_for_web(json.dumps({"token": conv["token"]})))
    assert back == {"ok": True, "token": conv["token"]}
    after = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], **req})))["sheets"][0]
    assert after["png_base64"] == before["png_base64"] and after["json"] == before["json"]

    # 大きさの違う画像には当てられない（確認の窓に「新しい画像にも当てる」を出さない）
    # （ここでは縁取り add で 1 ドットずつ大きくした絵を「大きさの違う画像」の代わりにする）
    conv3 = json.loads(frames_convert_for_web(encode_png(other), json.dumps(
        {"settings": {"outline": "add"}, "frames": {"frame_count": 3}})))
    assert (conv3["frame_width"], conv3["frame_height"]) != (conv["frame_width"], conv["frame_height"])
    r = json.loads(frames_edits_for_web(json.dumps({"token": conv3["token"], "edits": d})))
    assert r["ok"] and not r["applied"] and r["code"] == T.CODE_SIZE


def test_web_page_asks_when_image_is_reselected():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('id="touch-confirm-apply"', "新しい画像にも当てる", "やめる（前の画像に戻す）", "やめる（入力の種類を戻す）",
                "画像を選び直しました", "入力の種類を変えます", "function touchCarry", "function touchCarryBack",
                "function touchSaveFile", "touch.carry"):
        assert key in html, key
