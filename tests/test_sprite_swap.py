"""差し替え（retrocore/sprite/swap.py、CLI の --swaps、ブラウザ版の「差し替えを描く」）のテスト。テスト画像はコードで合成する。"""

import base64
import json

import numpy as np
import pytest
from PIL import Image

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import swap as S
from retrocore.sprite.motion import make_motion, sheet_json
from retrocore.sprite.motion.base import MotionResult
from retrocore.sprite.touchup import palette_of
from synth_sprite import PIX_EYE, enlarge_like_ai, make_pixel_sprite


def _hex(p) -> str:
    return "#%02x%02x%02x" % (int(p[0]), int(p[1]), int(p[2]))


def _face():
    """目（2 つの 2×2）のある 20×24 の顔。輪郭・肌・目の 3 色。"""
    img = np.zeros((24, 20, 4), np.uint8)
    img[2:23, 2:18] = (200, 40, 40, 255)          # 輪郭
    img[3:22, 3:17] = (250, 220, 190, 255)        # 肌
    img[8:10, 6:8] = (20, 20, 60, 255)            # 目
    img[8:10, 12:14] = (20, 20, 60, 255)
    return img


def _blink_dict(img, half=True, name="walk"):
    """目を肌の色にして、下の行に輪郭の色の線を引いた「閉じた目」（と、上の行だけ肌の色にした「半分」）。"""
    eye = np.all(img[..., :3] == img[8, 6, :3], axis=-1) & (img[..., 3] > 0)
    skin = img[4, 4].copy()
    line = img[2, 2].copy()
    closed = img.copy()
    closed[eye] = skin
    ys, xs = np.nonzero(eye)
    closed[ys.max(), xs] = line
    drawn = {"closed": closed}
    if half:
        h = img.copy()
        h[ys.min(), xs[ys == ys.min()]] = skin
        drawn["half"] = h
    sw = S.swap_from_image(img, drawn)
    s = S.SwapSet([sw], img.shape[1], img.shape[0], name=name)
    return S.swaps_dict(s, palette_of([img])), closed


def _anim(frames, durations, **info):
    return MotionResult(name="walk", frames=list(frames), durations=list(durations), info=dict(info))


# ---- 当てる ----

def test_blink_repeats_and_splits_without_changing_total():
    img = _face()
    d, closed = _blink_dict(img, half=False)
    anim = _anim([img, img, img, img], [150] * 4, source_frames=[1, 2, 3, 2])
    out, rep = S.apply_swaps(anim, img, S.swaps_from_dict(d))
    assert rep.repeats == 5 and rep.cycle_ms == 600 and rep.applied
    assert sum(out.durations) == 3000
    # 真ん中（1500 ミリ秒）のフレームを 120 + 30 に分ける
    assert len(out.frames) == 21 and out.durations[10:12] == [120, 30]
    assert np.array_equal(out.frames[10], closed) and np.array_equal(out.frames[11], img)
    assert out.info["source_frames"][10:12] == [3, 3]
    assert out.info["swap_frames"][10] == {"type": "blink", "shape": "closed"}
    assert sum(1 for m in out.info["swap_frames"] if m) == 1
    # 新しい色を作らない・大きさを変えない
    have = set(palette_of([img]))
    for f in out.frames:
        assert f.shape == img.shape and set(palette_of([f])) <= have
    # 元の結果は変えない
    assert len(anim.frames) == 4


def test_half_shape_goes_half_closed_half():
    img = _face()
    d, closed = _blink_dict(img, half=True)
    anim = _anim([img, img], [500, 500])
    out, rep = S.apply_swaps(anim, img, S.swaps_from_dict(d))
    assert rep.repeats == 3 and sum(out.durations) == 3000
    shapes = [m and m["shape"] for m in out.info["swap_frames"]]
    i = shapes.index("half")
    assert shapes[i:i + 3] == ["half", "closed", "half"]
    assert out.durations[i:i + 4] == [60, 120, 60, 260]
    assert np.array_equal(out.frames[i + 1], closed)


def test_moved_frame_is_found_and_foot_kept():
    img = _face()
    moved = np.zeros_like(img)
    moved[:-1] = img[1:]           # 頭が 1 ドット上に（足元も上がるが、当てても変わらない）
    d, closed = _blink_dict(img, half=False)
    anim = _anim([img, moved], [1500, 1500])
    out, rep = S.apply_swaps(anim, img, S.swaps_from_dict(d))
    pl = rep.items[0]["placements"]
    assert pl[1]["shift"] == {"x": 0, "y": -1} and pl[1]["match"] == 1.0
    assert rep.repeats == 1 and out.durations == [1500, 120, 1380]
    assert out.info["swap_frames"] == [None, {"type": "blink", "shape": "closed"}, None]
    want = moved.copy()
    want[:-1] = closed[1:]
    assert np.array_equal(out.frames[1], want) and np.array_equal(out.frames[2], moved)


def test_low_match_frame_is_not_used_and_warned():
    img = _face()
    other = np.zeros_like(img)
    other[5:20, 1:19] = (200, 40, 40, 255)    # まったく違う絵
    d, closed = _blink_dict(img, half=False)
    anim = _anim([img, other], [150, 150])
    out, rep = S.apply_swaps(anim, img, S.swaps_from_dict(d))
    item = rep.items[0]
    assert item["applied"] and item["frames_not_used"][0]["frame"] == 1
    assert item["frames_not_used"][0]["code"] == S.SKIP_LOW
    assert any("2 枚目" in w for w in rep.warnings)
    for k in item["frames"]:
        assert out.info.get("swap_frames")[k] and np.array_equal(out.frames[k], closed)
    # どのフレームにも当てられないときは、元のまま（周も増やさない）で理由だけ書く
    out2, rep2 = S.apply_swaps(_anim([other], [150]), img, S.swaps_from_dict(d), palette_of([img, other]))
    assert out2.frames == [other] and rep2.repeats == 1 and not rep2.applied
    assert out2.info["swaps"]["items"][0]["applied"] is False and rep2.warnings


HAIR = (90, 60, 30, 255)


def _with_strand(img, x, y0=3, y1=8):
    """前髪の 1 本（x 列の y0〜y1 行）を描いた顔。"""
    out = img.copy()
    out[y0:y1 + 1, x] = HAIR
    return out


def test_rects_are_split_per_eye_and_near_dots_are_merged():
    img = _face()
    d, _ = _blink_dict(img, half=False)
    sw = S.swaps_from_dict(d).swaps[0]
    assert sw.rects() == [(6, 8, 2, 2), (12, 8, 2, 2)]
    # 2 ドット以内のドットは同じ四角
    near = S.Swap("blink", {"closed": [S.Dot(1, 1, None), S.Dot(3, 1, None), S.Dot(9, 1, None)]})
    assert near.rects() == [(1, 1, 3, 1), (9, 1, 1, 1)]


def test_moving_bangs_between_eyes_are_not_reverted():
    """目と目の間の前髪がコマごとに動いても、まばたきのコマで 1 コマ目の形に戻らない（四角を目ごとに分けるので入らない）。"""
    img = _face()
    f1, f2 = _with_strand(img, 10), _with_strand(img, 9)
    d, _ = _blink_dict(f1, half=False)
    out, rep = S.apply_swaps(_anim([f1, f2], [1500, 1500]), f1, S.swaps_from_dict(d), palette_of([f1, f2]))
    assert rep.items[0]["placements"][1]["used"] and out.info["swap_frames"][1]
    got = out.frames[1]
    assert tuple(got[8, 9]) == HAIR and tuple(got[8, 10]) != HAIR   # 2 コマ目の前髪のまま
    assert np.array_equal(got[:, 8:12], f2[:, 8:12])


def test_bangs_crossing_eye_rect_frame_is_not_used():
    """前髪が目の四角の境目をまたいで動くコマには入れない（入れると四角の中だけ 1 コマ目の形に戻る）。"""
    img = _face()
    f1, f2 = img, _with_strand(img, 7, 3, 8)     # 2 コマ目だけ、前髪が左の目の四角に入る
    d, closed = _blink_dict(f1, half=False)
    out, rep = S.apply_swaps(_anim([f1, f2], [1500, 1500]), f1, S.swaps_from_dict(d), palette_of([f1, f2]))
    item = rep.items[0]
    assert item["frames_not_used"] == [{"frame": 1, "code": S.SKIP_CROSS, "match": item["placements"][1]["match"]}]
    assert any("2 枚目" in w and "前髪" in w for w in rep.warnings)
    # 1 コマ目に入れ、2 コマ目の絵はどのフレームでもそのまま
    for k, m in enumerate(out.info["swap_frames"]):
        if m:
            assert np.array_equal(out.frames[k], closed)
    assert any(np.array_equal(f, f2) for f in out.frames) and not any(
        (f[3:9, 7] == np.array(HAIR, np.uint8)).all(-1).any() and not np.array_equal(f, f2) for f in out.frames)


def test_seam_frame_is_not_used():
    """置き換えると四角の境目で輪郭が途切れるコマには入れない（same.py と同じ確かめ）。
    目を輪郭の色で描いて顔の縁に付けた絵で、2 コマ目だけ目の隣の輪郭が欠けている。閉じた目（肌の色）を置くと、肌が外に出る。"""
    line = (200, 40, 40, 255)
    img = np.zeros((16, 16, 4), np.uint8)
    img[2:14, 2:14] = line
    img[3:13, 3:13] = (250, 220, 190, 255)
    img[7, 3:5] = line                                # 目（輪郭の色。左の縁に付いている）
    img[10, 9:12] = line                              # 口
    f2 = img.copy()
    f2[7, 2] = 0                                      # 2 コマ目は目の隣の輪郭が欠けている（目が縁になっている）
    closed = img.copy()
    closed[7, 3:5] = (250, 220, 190, 255)
    sw = S.swap_from_image(img, {"closed": closed})
    assert sw.rects() == [(3, 7, 2, 1)]
    d = S.swaps_dict(S.SwapSet([sw], 16, 16), palette_of([img]))
    out, rep = S.apply_swaps(_anim([img, f2], [1500, 1500]), img, S.swaps_from_dict(d), palette_of([img, f2]))
    nu = rep.items[0]["frames_not_used"]
    assert [(e["frame"], e["code"]) for e in nu] == [(1, S.SKIP_SEAM)]
    assert any("段差・輪郭の途切れ" in w for w in rep.warnings)


def test_short_pieces_are_merged():
    img = _face()
    d, _ = _blink_dict(img, half=False)
    s = S.swaps_from_dict(dict(d, close_ms=135))
    out, rep = S.apply_swaps(_anim([img] * 4, [150] * 4), img, s)
    # 150 − 135 = 15 ミリ秒の切れ端は作らず、閉じた目を 150 ミリ秒にする
    assert min(out.durations) >= S.MIN_PIECE and sum(out.durations) == 3000
    assert 150 in [d for d, m in zip(out.durations, out.info["swap_frames"]) if m]


@pytest.mark.parametrize("change, code", [
    ({"frame_width": 99}, S.CODE_SIZE),
    ({"palette": ["#010203"] * 8}, S.CODE_PALETTE),
])
def test_not_applicable(change, code):
    img = _face()
    d, _ = _blink_dict(img)
    d.update(change)
    with pytest.raises(S.SwapError) as e:
        S.apply_swaps(_anim([img], [150]), img, S.swaps_from_dict(d))
    assert e.value.code == code


@pytest.mark.parametrize("bad", [
    {"kind": "sprite_edits"},
    {"swaps": [{"type": "mouth", "closed": []}]},
    {"swaps": [{"type": "blink", "half": [[0, 0, None]]}]},
    {"swaps": [{"type": "blink", "closed": [[0, 0, 5]]}]},
    {"swaps": [{"type": "blink", "closed": [[0, 0, None]], "open": []}]},
    {"interval_ms": 10},
    {"close_ms": "120"},
    {"extra": 1},
])
def test_bad_files(bad):
    with pytest.raises(ValueError):
        S.swaps_from_dict(dict({"version": 1, "kind": "sprite_swaps", "palette": []}, **bad))


def test_file_roundtrip():
    img = _face()
    d, _ = _blink_dict(img)
    s = S.swaps_from_dict(d)
    text = S.swaps_json(s, palette_of([img]))
    again = S.load_swaps(text)
    assert S.swaps_dict(again, palette_of([img])) == S.swaps_dict(s, palette_of([img]))
    # 左右の目は別々の四角（目と目の間は入れない）
    assert json.loads(text)["swaps"][0]["rects"] == [{"x": 6, "y": 8, "w": 2, "h": 2}, {"x": 12, "y": 8, "w": 2, "h": 2}]
    assert S.swaps_file_name("hero", "walk") == "hero_walk_swaps.json"


def test_no_swaps_is_byte_identical():
    """差し替えが無いときのシートの JSON は今までと同じ（swaps も swap も無い）。"""
    img = _face()
    res = make_motion(img, "breathe")
    a = sheet_json(res, "x.png")
    assert '"swap' not in a
    d, _ = _blink_dict(img)
    out, _ = S.apply_swaps(res, img, S.swaps_from_dict(d))
    b = sheet_json(out, "x.png")
    assert '"swaps"' in b and '"swap": {' in b


def test_mirror_is_flipped_normal():
    img = _face()
    img[12, 4] = (20, 20, 60, 255)    # 左右で違う絵にする
    d, _ = _blink_dict(img, half=False)
    normal = _anim([img, img], [150, 150], pivot_x=10.0)
    mirror = MotionResult(name="walk_mirror", frames=[f[:, ::-1].copy() for f in normal.frames],
                          durations=[150, 150], info={"pivot_x": 10.0})
    out, _ = S.apply_swaps(normal, img, S.swaps_from_dict(d))
    m = S.mirror_swapped(out, mirror)
    assert m.name == "walk_mirror" and m.durations == out.durations
    assert all(np.array_equal(a[:, ::-1], b) for a, b in zip(out.frames, m.frames))
    rs = m.info["swaps"]["items"][0]["rects"]
    want = [img.shape[1] - r["x"] - r["w"] for r in out.info["swaps"]["items"][0]["rects"]]
    assert [r["x"] for r in rs] == want and len(rs) >= 2


# ---- CLI と画面 ----

@pytest.fixture(scope="module")
def strip():
    from test_sprite_frames import make_strip
    return make_strip()


@pytest.fixture(scope="module")
def plain(strip):
    return F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))


def _eye_swaps(ref, name):
    """合成のキャラの目（PIX_EYE）を、すぐ左の色にした「閉じた目」の差し替え。"""
    eye = np.all(ref[..., :3] == np.array(PIX_EYE, np.uint8), axis=-1) & (ref[..., 3] > 0)
    assert eye.any()
    closed = ref.copy()
    for y, x in zip(*np.nonzero(eye)):
        closed[y, x] = ref[y, x - 1]
    sw = S.swap_from_image(ref, {"closed": closed})
    return S.swaps_dict(S.SwapSet([sw], ref.shape[1], ref.shape[0], name=name), palette_of([ref]))


def test_cli_swaps_errors(tmp_path, strip, plain):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    with pytest.raises(SystemExit):
        main([str(src), "--swaps", "x.json"])    # 動きが無い
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert main([str(src), "-o", str(tmp_path / "o.png"), "-q", "--frames", "--frame-count", "3", "--swaps", str(bad)]) == 1
    d = _eye_swaps(plain.frames[0], "frames")
    d["frame_width"] += 1
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(d), encoding="utf-8")
    assert main([str(src), "-o", str(tmp_path / "w" / "o.png"), "-q", "--frames", "--frame-count", "3",
                 "--swaps", str(wrong)]) == 1
    assert not any(p.name.endswith("f.json") for p in (tmp_path / "w").glob("*"))


def test_web_frames_swaps_match_cli_bytes(tmp_path, strip, plain):
    """画面でまばたきを入れたシート（JSON・GIF・反転・.piskel）と、CLI の --swaps がバイト単位で一致する。"""
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web, gif_for_web
    from test_sprite_frames import _cli_files, _text_bytes
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}, "name": "walk"})))
    assert conv["ok"]
    d = _eye_swaps(plain.frames[0], "walk")
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite", "name": "walk",
                                                          "order": [1, 2, 3, 2], "mirror": True, "swaps": d})))
    assert sheets["ok"] and sheets["swaps_error"] is None, sheets
    assert sheets["swaps"]["repeats"] == 5 and sheets["swaps"]["items"][0]["applied"]
    sf = tmp_path / "hero_sprite_walk_swaps.json"
    sf.write_text(sheets["swaps_json"], encoding="utf-8")
    argv = ["--frame-count", "3", "--frame-name", "walk", "--frame-order", "1,2,3,2", "--piskel", "--swaps", str(sf)]
    cli = _cli_files(tmp_path, strip, argv)
    web = {"hero_sprite.png": base64.b64decode(conv["png_base64"]),
           "hero_sprite_walk_flicker.png": base64.b64decode(conv["flicker_png_base64"]),
           "hero_sprite_walk_flicker.json": _text_bytes(conv["flicker_json"])}
    for s in sheets["sheets"]:
        png = base64.b64decode(s["png_base64"])
        web[s["name"]] = png
        web[s["name"][:-4] + ".json"] = _text_bytes(s["json"])
        web[s["piskel_name"]] = s["piskel"].encode("ascii")
        g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
        web[g["name"]] = base64.b64decode(g["gif_base64"])
        assert '"swaps"' in s["json"]
    assert sorted(web) == sorted(cli)
    for k in cli:
        assert web[k] == cli[k], k
    # 保存した一覧を読み込んで、もう一度同じ一覧になる
    assert S.swaps_json(S.load_swaps(sheets["swaps_json"]), palette_of(plain.frames)) == sheets["swaps_json"]


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory):
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


def test_web_motion_swaps_match_cli_bytes(tmp_path, pixel_png):
    """1 枚の絵の動き（呼吸）にまばたきを入れたシートと GIF が、CLI の --motion breathe --swaps とバイト単位で一致する。"""
    from apps.sprite import cli
    from retrocore.sprite.convert import convert_for_web, decode_image, gif_for_web, motion_for_web
    conv = json.loads(convert_for_web(pixel_png.read_bytes(), "{}"))
    sprite_png = base64.b64decode(conv["png_base64"])
    d = _eye_swaps(decode_image(sprite_png), "breathe")
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "breathe", "frames": 2, "swaps": d,
                                                            "image_name": "hero_breathe_sheet.png"})))
    assert res["ok"] and not res["skipped"] and res["swaps"]["repeats"] == 3, res
    assert sum(res["sheet_durations"]) == 3000 and res["durations"] == [500, 500]
    sf = tmp_path / "hero_breathe_swaps.json"
    sf.write_text(res["swaps_json"], encoding="utf-8")
    dst = tmp_path / "cli" / "hero.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "breathe", "--gif", "--swaps", str(sf)]) == 0
    from test_sprite_frames import _text_bytes
    assert (dst.parent / "hero_breathe_sheet.png").read_bytes() == base64.b64decode(res["sheet_png_base64"])
    assert (dst.parent / "hero_breathe_sheet.json").read_bytes() == _text_bytes(res["sheet_json"])
    g = json.loads(gif_for_web(base64.b64decode(res["sheet_png_base64"]), res["sheet_json"], json.dumps({"scale": 4})))
    assert (dst.parent / "hero_breathe.gif").read_bytes() == base64.b64decode(g["gif_base64"])
    # 差し替えが無いときの返す値は今までと同じ（swaps の項目も足さない）
    plain = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "breathe", "frames": 2})))
    assert "swaps" not in plain and "swaps_json" not in plain


def test_readback_does_not_flag_blink_frames(tmp_path, strip, plain):
    from retrocore.sprite.edit import ReadbackOptions, readback
    from retrocore.sprite.motion import make_sheet
    d = _eye_swaps(plain.frames[0], "walk")
    anim = F.assemble_frames(plain, [1, 2, 3, 2], None, None, "walk")
    out, _ = S.apply_swaps(anim, plain.frames[0], S.swaps_from_dict(d), palette_of(plain.frames))
    from retrocore.sprite.convert import encode_png
    png = encode_png(make_sheet(out.frames))
    info = F.frames_sheet_json(out, "x.png", plain.skipped)
    res = readback(png, "x.png", info, None, ReadbackOptions())
    assert res.made and not any("同じコマ" in w for w in res.warnings), res.warnings


def test_swaps_run_with_int32_numpy(strip, plain):
    """32 ビットの NumPy のつもりで通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    d = _eye_swaps(plain.frames[0], "walk")
    anim = F.assemble_frames(plain, [1, 2, 3, 2], None, [0, -1, 0, -1], "walk")
    want, _ = S.apply_swaps(anim, plain.frames[0], S.swaps_from_dict(d), palette_of(plain.frames))
    with emulate_int32():
        got, _ = S.apply_swaps(anim, plain.frames[0], S.swaps_from_dict(d), palette_of(plain.frames))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames)) and want.durations == got.durations


def test_web_page_has_swap_editor():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('id="swap-mode"', 'id="swap-box"', 'id="swap-shape"', 'id="swap-interval"', 'id="swap-close"',
                'id="swap-half"', 'id="swap-save"', 'id="swap-load"', 'id="swap-status"', "差し替えを描く",
                "function swapDict", "function touchSingleSync", "--swaps", "sprite_swaps",
                # 大きさが変わって当てられないときの確認の窓（「ドットを直す」と同じ窓）
                "function swapCannotApply", "function swapHold", "function swapRestoreHeld", 'id="touch-confirm-save-label"',
                "差し替えを保存してから捨てる", "直しと差し替えを保存してから捨てる", "1 枚の絵では、前の変換に戻す仕組みがまだ無いため"):
        assert key in html, key
    # 説明の文は「？」の中（class="help"）に入れる
    assert 'class="note help" data-help-at="touch-title">「差し替えを描く」' in html
    assert 'class="note help" data-help-at="swap-io">' in html
