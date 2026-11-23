"""口パク（しゃべりのシート。retrocore/sprite/swap.py の talk と apply_talk、CLI の --swaps、ブラウザ版の「差し替えを描く」）のテスト。
テスト画像はコードで合成する。"""

import base64
import json

import numpy as np
import pytest
from PIL import Image

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import swap as S
from retrocore.sprite.motion.base import MotionResult
from retrocore.sprite.touchup import palette_of
from synth_sprite import PIX_EYE, enlarge_like_ai, make_pixel_sprite
from test_sprite_swap import _blink_dict, _face

DARK = (20, 20, 60, 255)    # 目と同じ色（パレットにある色）


def _talk(img, half=False, at=(8, 11), size=(4, 2)):
    """口（`at` の左上から `size` の四角を暗い色にした「開いた口」と、上の行だけの「半分」）の差し替え。"""
    x, y = at
    w, h = size
    op = img.copy()
    op[y:y + h, x:x + w] = DARK
    drawn = {"open": op}
    if half:
        hm = img.copy()
        hm[y, x:x + w] = DARK
        drawn["half"] = hm
    return S.swap_from_image(img, drawn, "talk"), op


def _set(img, *swaps, **kw):
    return S.SwapSet(list(swaps), img.shape[1], img.shape[0], name="walk", **kw)


def _anim(frames, durations, **info):
    return MotionResult(name="walk", frames=list(frames), durations=list(durations), info=dict(info))


def _marks(out):
    return [(m or {}).get("talk") for m in out.info["swap_frames"]]


# ---- 口の順番と時間 ----

def test_talk_without_half_is_closed_open_150():
    img = _face()
    sw, op = _talk(img)
    anim = _anim([img, img], [300, 300])
    out, rep = S.apply_talk(anim, img, _set(img, sw))
    assert out.name == "walk_talk" and rep.repeats == 1
    assert out.durations == [150] * 4 and _marks(out) == [None, "open", None, "open"]
    assert np.array_equal(out.frames[0], img) and np.array_equal(out.frames[1], op)
    it = rep.items[0]
    assert it["applied"] and it["step_ms"] == 150 and it["pattern"] == ["closed", "open"] and it["steps"] == 4
    # 新しい色を作らない・大きさを変えない・元の結果は変えない
    for f in out.frames:
        assert f.shape == img.shape and set(palette_of([f])) <= set(palette_of([img]))
    assert len(anim.frames) == 2


def test_talk_with_half_is_closed_half_open_half_100():
    img = _face()
    sw, op = _talk(img, half=True)
    out, rep = S.apply_talk(_anim([img], [400]), img, _set(img, sw))
    assert out.durations == [100] * 4 and _marks(out) == [None, "half", "open", "half"]
    assert rep.items[0]["step_ms"] == 100
    # 速さを変える
    out, rep = S.apply_talk(_anim([img], [400]), img, _set(img, sw, talk_ms=200))
    assert out.durations == [200] * 4 and sum(out.durations) == 800 and rep.repeats == 2


def test_talk_splits_motion_frames_by_time():
    img = _face()
    moved = np.zeros_like(img)
    moved[:-1] = img[1:]
    sw, op = _talk(img)
    out, rep = S.apply_talk(_anim([img, moved], [500, 500]), img, _set(img, sw))
    # 1 周 1000 は 300 で割り切れない → 3 周（3000）にする。境目で時間を分ける
    assert rep.repeats == 3 and sum(out.durations) == 3000
    assert out.durations[:5] == [150, 150, 150, 50, 100]
    assert out.info["swaps"]["items"][0]["placements"][1]["shift"] == {"x": 0, "y": -1}
    want = moved.copy()
    want[:-1] = op[1:]
    assert _marks(out)[4] == "open"   # 500〜600 ミリ秒（2 コマ目）の開いた口
    assert np.array_equal(out.frames[4], want)


# ---- ループのつながり ----

@pytest.mark.parametrize("durations, half", [([500, 500], False), ([500, 500], True), ([170, 170, 171], False),
                                             ([333], True), ([1001], False)])
def test_loop_joins_and_no_short_pieces(durations, half):
    img = _face()
    sw, _ = _talk(img, half=half)
    s = _set(img, sw)
    out, rep = S.apply_talk(_anim([img] * len(durations), durations), img, s)
    total = sum(out.durations)
    assert total == sum(durations) * rep.repeats
    assert min(out.durations) >= S.MIN_PIECE
    marks = _marks(out)
    # 始まりは閉じた口、終わりは 1 回りの最後の形（半分あり: 半分、なし: 開く）→ くり返すと 閉じる につながる
    assert marks[0] is None and marks[-1] == ("half" if half else "open")
    # 口の 1 回りの数は、形の数の倍数（つながる）
    it = rep.items[0]
    assert it["steps"] % len(it["pattern"]) == 0
    step = s.talk_step()
    assert all(abs(v - step) <= max(1, step * 3 // 100) for v in it["step_ms_used"])


def test_talk_repeats_follow_blink_interval():
    img = _face()
    d, closed = _blink_dict(img, half=False)
    d["close_ms"] = 200     # 1500〜1700: 閉じた口（〜1650）と開いた口（1650〜）にかかる
    s = S.swaps_from_dict(d)
    sw, op = _talk(img)
    s.swaps.append(sw)
    out, rep = S.apply_talk(_anim([img, img], [500, 500]), img, s)
    assert rep.repeats == 3 and sum(out.durations) == 3000
    blink = rep.items[1]
    assert blink["type"] == "blink" and blink["applied"] and blink["start_ms"] == 1500
    # まばたきと口が同じフレームに入る（目と口の両方を当てた絵）
    both = [i for i, m in enumerate(out.info["swap_frames"]) if m and "talk" in m and "blink" in m]
    assert both
    want = op.copy()
    eye = np.any(closed != img, axis=-1)
    want[eye] = closed[eye]
    assert np.array_equal(out.frames[both[0]], want)
    # ふつうのシート（apply_swaps）は口パクを入れない（まばたきだけ）
    plain, prep = S.apply_swaps(_anim([img, img], [500, 500]), img, s)
    assert all(m is None or m["type"] == "blink" for m in plain.info["swap_frames"])


# ---- 目と口が重なるとき ----

def test_overlapping_eyes_and_mouth_prefer_mouth():
    img = _face()
    d, closed = _blink_dict(img, half=False)
    s = S.swaps_from_dict(d)
    sw, op = _talk(img, at=(5, 9), size=(10, 2))    # 目の下の行と重なる口
    s.swaps.append(sw)
    out, rep = S.apply_talk(_anim([img, img], [500, 500]), img, s)
    assert out is not None and rep.items[0]["applied"]
    blink = rep.items[1]
    assert blink["applied"] is False
    assert {e["code"] for e in blink["frames_not_used"]} == {S.SKIP_OVERLAP}
    assert any("重なる" in w for w in rep.warnings)
    assert not any(m and "blink" in m for m in out.info["swap_frames"])


# ---- 入らないコマが多いとき ----

def test_too_many_frames_without_mouth_makes_no_sheet():
    img = _face()
    other = np.zeros_like(img)
    other[5:20, 1:19] = (200, 40, 40, 255)
    sw, _ = _talk(img)
    out, rep = S.apply_talk(_anim([img, other], [300, 300]), img, _set(img, sw), palette_of([img, other]))
    assert out is None and rep.items[0]["applied"] is False
    assert any("次に試せること" in w and "2 枚目" in w for w in rep.warnings)
    # 3 枚のうち 1 枚だけなら作る（そのフレームは閉じた口のまま。警告）
    out, rep = S.apply_talk(_anim([img, other, img], [300, 300, 300]), img, _set(img, sw), palette_of([img, other]))
    assert out is not None and any("口を入れなかったフレーム" in w for w in rep.warnings)
    srcs = [i for i, f in enumerate(out.frames) if f is other or np.array_equal(f, other)]
    assert srcs and all(out.info["swap_frames"][i] is None for i in srcs)


# ---- ファイル ----

def test_talk_file_roundtrip_and_blink_only_unchanged():
    img = _face()
    d, _ = _blink_dict(img, half=False)
    text = S.swaps_json(S.swaps_from_dict(d), palette_of([img]))
    assert "talk_ms" not in text                     # まばたきだけの一覧は前と同じ
    sw, _ = _talk(img, half=True)
    s = S.swaps_from_dict(d)
    s.swaps.append(sw)
    text = S.swaps_json(s, palette_of([img]))
    assert '"talk_ms": null' in text and '"type": "talk"' in text and '"open": [' in text
    back = S.load_swaps(text)
    assert back.find("talk") is not None and back.talk_ms is None and back.talk_step() == 100
    assert S.swaps_json(back, palette_of([img])) == text
    s.talk_ms = 120
    assert S.load_swaps(S.swaps_json(s, palette_of([img]))).talk_ms == 120
    # 口パクだけの一覧: ふつうのシートは何も変えない
    t = _set(img, sw)
    anim = _anim([img], [500])
    same, rep = S.apply_swaps(anim, img, t)
    assert same is anim and rep is None


@pytest.mark.parametrize("bad", [{"talk_ms": 10}, {"talk_ms": "x"}, {"swaps": [{"type": "talk", "half": [[1, 1, 0]]}]},
                                 {"swaps": [{"type": "talk", "closed": [[1, 1, 0]]}]}])
def test_bad_talk_files(bad):
    d = {"version": 1, "palette": ["#000000"], "swaps": [{"type": "talk", "open": [[1, 1, 0]]}]}
    d.update(bad)
    with pytest.raises(ValueError):
        S.swaps_from_dict(d)


# ---- CLI と画面 ----

def _mouth(ref):
    """合成のキャラの目（PIX_EYE）の 2 行下に、目の左端から 3 ドットの「開いた口」（目の色）を描いた口パク。"""
    eye = np.all(ref[..., :3] == np.array(PIX_EYE, np.uint8), axis=-1) & (ref[..., 3] > 0)
    ys, xs = np.nonzero(eye)
    y, x = int(ys.max()) + 3, int(xs.min()) + 1
    op = ref.copy()
    op[y, x:x + 3] = ref[ys[0], xs[0]]
    half = ref.copy()
    half[y, x + 1] = ref[ys[0], xs[0]]
    return S.swap_from_image(ref, {"open": op, "half": half}, "talk")


def _eyes_and_mouth(ref, name, blink=True):
    from test_sprite_swap import _eye_swaps
    s = S.swaps_from_dict(_eye_swaps(ref, name))
    if not blink:
        s.swaps = []
    s.swaps.append(_mouth(ref))
    return S.swaps_dict(s, palette_of([ref]))


@pytest.fixture(scope="module")
def strip():
    from test_sprite_frames import make_strip
    return make_strip()


@pytest.fixture(scope="module")
def plain(strip):
    return F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))


@pytest.fixture(scope="module")
def pixel_png(tmp_path_factory):
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


def test_web_motion_talk_matches_cli_bytes(tmp_path, pixel_png):
    """1 枚の絵の呼吸に、まばたきと口パクを入れた。ふつうのシートとしゃべりのシート（PNG・JSON・GIF）が CLI と同じ。"""
    from apps.sprite import cli
    from retrocore.sprite.convert import convert_for_web, decode_image, gif_for_web, motion_for_web
    from test_sprite_frames import _text_bytes
    conv = json.loads(convert_for_web(pixel_png.read_bytes(), "{}"))
    sprite_png = base64.b64decode(conv["png_base64"])
    d = _eyes_and_mouth(decode_image(sprite_png), "breathe")
    res = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "breathe", "frames": 2, "swaps": d,
                                                            "image_name": "hero_breathe_sheet.png"})))
    assert res["ok"] and res["talk"] and res["talk_error"] is None, res.get("talk_warnings")
    assert res["talk"]["name"] == "hero_breathe_talk_sheet.png"
    assert res["talk_swaps"]["items"][0]["type"] == "talk" and res["talk_swaps"]["items"][1]["applied"]
    sf = tmp_path / "hero_breathe_swaps.json"
    sf.write_text(res["swaps_json"], encoding="utf-8")
    dst = tmp_path / "cli" / "hero.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--motion", "breathe", "--gif", "--swaps", str(sf)]) == 0
    files = {p.name for p in dst.parent.iterdir()}
    assert {"hero_breathe_talk_sheet.png", "hero_breathe_talk_sheet.json", "hero_breathe_talk.gif"} <= files
    assert (dst.parent / "hero_breathe_sheet.png").read_bytes() == base64.b64decode(res["sheet_png_base64"])
    assert (dst.parent / "hero_breathe_sheet.json").read_bytes() == _text_bytes(res["sheet_json"])
    t = res["talk"]
    png = base64.b64decode(t["sheet_png_base64"])
    assert (dst.parent / "hero_breathe_talk_sheet.png").read_bytes() == png
    assert (dst.parent / "hero_breathe_talk_sheet.json").read_bytes() == _text_bytes(t["sheet_json"])
    g = json.loads(gif_for_web(png, t["sheet_json"], json.dumps({"scale": 4})))
    assert g["name"] == "hero_breathe_talk.gif"
    assert (dst.parent / "hero_breathe_talk.gif").read_bytes() == base64.b64decode(g["gif_base64"])
    # 口パクの無い一覧（まばたきだけ）: しゃべりの項目は足さない（前と同じ）
    from test_sprite_swap import _eye_swaps
    old = json.loads(motion_for_web(sprite_png, json.dumps({"motion": "breathe", "frames": 2,
                                                            "swaps": _eye_swaps(decode_image(sprite_png), "breathe")})))
    assert not any(k.startswith("talk") for k in old)


def test_cli_talk_without_motion(tmp_path, pixel_png):
    """口パクだけの一覧なら、--motion なしで口だけ動くしゃべりのシートを書く。"""
    from apps.sprite import cli
    from retrocore.sprite.convert import convert_for_web, decode_image
    conv = json.loads(convert_for_web(pixel_png.read_bytes(), "{}"))
    ref = decode_image(base64.b64decode(conv["png_base64"]))
    sf = tmp_path / "talk.json"
    sf.write_text(json.dumps(_eyes_and_mouth(ref, "talk", blink=False)), encoding="utf-8")
    dst = tmp_path / "o" / "hero.png"
    assert cli.main([str(pixel_png), "-o", str(dst), "-q", "--gif", "--swaps", str(sf)]) == 0
    info = json.loads((dst.parent / "hero_talk_sheet.json").read_text(encoding="utf-8"))
    assert info["motion"] == "talk" and info["frame_count"] == 4 and info["total_duration_ms"] == 400
    assert [f["duration_ms"] for f in info["frames"]] == [100] * 4
    assert (dst.parent / "hero_talk.gif").exists()
    # まばたきだけの一覧は、今までどおり動きが要る
    sf.write_text(json.dumps(S.swaps_dict(S.SwapSet([S.swaps_from_dict(_eyes_and_mouth(ref, "x")).find("blink")],
                                                    ref.shape[1], ref.shape[0]), palette_of([ref]))), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main([str(pixel_png), "-o", str(dst), "-q", "--swaps", str(sf)])


def test_web_frames_talk_matches_cli_bytes(tmp_path, strip, plain):
    """コマが並んだ絵: しゃべりのシート（反転・.piskel・GIF も）が CLI の --frames --swaps とバイト単位で同じ。"""
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web, gif_for_web
    from test_sprite_frames import _cli_files, _text_bytes
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}, "name": "walk"})))
    d = _eyes_and_mouth(plain.frames[0], "walk")
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite", "name": "walk",
                                                          "order": [1, 2, 3, 2], "mirror": True, "swaps": d})))
    assert sheets["ok"] and sheets["talk_error"] is None, sheets.get("talk_warnings")
    assert len(sheets["talk_sheets"]) == 2 and sheets["talk_sheets"][1]["mirror"]
    assert "_walk_talk_" in sheets["talk_sheets"][0]["name"] and "_walk_talk_mirror_" in sheets["talk_sheets"][1]["name"]
    sf = tmp_path / "hero_sprite_walk_swaps.json"
    sf.write_text(sheets["swaps_json"], encoding="utf-8")
    argv = ["--frame-count", "3", "--frame-name", "walk", "--frame-order", "1,2,3,2", "--piskel", "--swaps", str(sf)]
    cli = _cli_files(tmp_path, strip, argv)
    web = {"hero_sprite.png": base64.b64decode(conv["png_base64"]),
           "hero_sprite_walk_flicker.png": base64.b64decode(conv["flicker_png_base64"]),
           "hero_sprite_walk_flicker.json": _text_bytes(conv["flicker_json"])}
    for s in sheets["sheets"] + sheets["talk_sheets"]:
        png = base64.b64decode(s["png_base64"])
        web[s["name"]] = png
        web[s["name"][:-4] + ".json"] = _text_bytes(s["json"])
        web[s["piskel_name"]] = s["piskel"].encode("ascii")
        g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
        web[g["name"]] = base64.b64decode(g["gif_base64"])
    assert sorted(web) == sorted(cli)
    for k in cli:
        assert web[k] == cli[k], k
    # 反転したしゃべりのシートは、反転していない方を左右反転したもの
    a = json.loads(sheets["talk_sheets"][0]["json"])
    b = json.loads(sheets["talk_sheets"][1]["json"])
    assert [f["duration_ms"] for f in a["frames"]] == [f["duration_ms"] for f in b["frames"]]
    assert b["source"]["mirrored"] and [f.get("swap") for f in a["frames"]] == [f.get("swap") for f in b["frames"]]


def test_readback_does_not_flag_talk_frames(strip, plain):
    from retrocore.sprite.convert import encode_png
    from retrocore.sprite.edit import ReadbackOptions, readback
    from retrocore.sprite.motion import make_sheet
    s = S.swaps_from_dict(_eyes_and_mouth(plain.frames[0], "walk"))
    anim = F.assemble_frames(plain, [1, 2, 3, 2], None, None, "walk")
    out, _ = S.apply_talk(anim, plain.frames[0], s, palette_of(plain.frames))
    png = encode_png(make_sheet(out.frames))
    res = readback(png, "x.png", F.frames_sheet_json(out, "x.png", plain.skipped), None, ReadbackOptions())
    assert res.made and not any("同じコマ" in w for w in res.warnings), res.warnings


def test_talk_runs_with_int32_numpy(strip, plain):
    """32 ビットの NumPy のつもりで通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    s = S.swaps_from_dict(_eyes_and_mouth(plain.frames[0], "walk"))
    anim = F.assemble_frames(plain, [1, 2, 3, 2], None, [0, -1, 0, -1], "walk")
    want, _ = S.apply_talk(anim, plain.frames[0], s, palette_of(plain.frames))
    with emulate_int32():
        got, _ = S.apply_talk(anim, plain.frames[0], s, palette_of(plain.frames))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames)) and want.durations == got.durations


def test_web_page_has_talk_editor():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('<option value="open">開いた口（口パク）</option>', '<option value="mouth_half">', 'id="swap-talk"',
                'id="talk-box"', 'id="talk-canvas"', 'id="dl-talk-sheet"', 'id="dl-talk-json"', 'id="dl-talk-gif"',
                'id="dl-talk-piskel"', 'id="dl-talk-mirror"', 'id="dl-talk-mirror-gif"', "function talkShow", "SWAP_KEYS",
                "差し替えを描く（まばたき・口パク）", "差し替え（まばたき・口パク）のドット"):
        assert key in html, key
    assert 'class="note help" data-help-at="talk-title">' in html
