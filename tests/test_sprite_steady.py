"""ちらつきを減らす（`retrocore/sprite/steady.py`、`--frames --steady on`）のテスト。

テスト画像はコードで合成する（暗い輪郭・塗り・目のある頭と体）。2 コマ目以降に、動きと関係のない小さな違い（点）と、
腕の動き（大きな違い）、頭のずれなどを入れて、小さな違いだけが 1 コマ目の色に戻ることを確かめる。
"""

import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import steady as St

OUT = (20, 18, 50, 255)       # 輪郭
HAIR = (150, 200, 240, 255)
SKIN = (250, 225, 210, 255)
EYE = (40, 30, 80, 255)
BODY = (200, 90, 90, 255)
SPOT = (255, 255, 255, 255)   # 髪のハイライト
H, W = 40, 30


def _box(img, y0, y1, x0, x1, color, outline=True):
    img[y0:y1, x0:x1] = color
    if outline:
        img[y0, x0:x1] = OUT
        img[y1 - 1, x0:x1] = OUT
        img[y0:y1, x0] = OUT
        img[y0:y1, x1 - 1] = OUT


def make_frame(arm: int = 0, head_dy: int = 0) -> np.ndarray:
    """1 コマ: 上の方に頭（髪・顔・2 ドットの目）、下の方に体と腕。arm は腕の横のずれ、head_dy は頭だけの縦のずれ。"""
    img = np.zeros((H, W, 4), np.uint8)
    # 体（下の方）と脚
    _box(img, 20, 36, 8, 22, BODY)
    _box(img, 36, 40, 9, 13, BODY)
    _box(img, 36, 40, 17, 21, BODY)
    # 腕（体の右。上の方に届く）
    _box(img, 14, 26, 22 + arm, 26 + arm, BODY)
    # 頭（上の方）
    hy = 2 + head_dy
    _box(img, hy, hy + 18, 5, 25, HAIR)
    img[hy + 8:hy + 16, 8:22] = SKIN
    img[hy + 11:hy + 13, 11] = EYE
    img[hy + 11:hy + 13, 18] = EYE
    img[hy + 4, 10] = SPOT
    img[hy + 4, 19] = SPOT
    return img


def test_small_specks_go_back_and_large_motion_stays():
    ref = make_frame()
    f = make_frame(arm=2)                 # 腕が 2 ドット動いた（大きな違い・下の方）
    f[5, 14] = SPOT                       # 髪の中の 1 ドット（ちらつき）
    f[6, 10] = HAIR                       # ハイライトが消えた（ちらつき）
    f[8, 13:16] = SKIN                    # 細い線（3 ドット）
    out, rep = St.steady_frames([ref, f])
    g = out[1]
    for y, x in [(5, 14), (6, 10), (8, 13), (8, 14), (8, 15)]:
        assert (g[y, x] == ref[y, x]).all(), (y, x)
    # 腕は動いたまま（下の方は戻さない。上の方に届く部分も大きな違い）
    assert np.array_equal(g[20:], f[20:])
    assert np.array_equal(g[14:26, 22:], f[14:26, 22:])
    sf = rep.frames[0]
    assert sf.restored_count == 5
    assert sf.kept_count == int(F._diff_mask(ref, f, H).sum()) - 5
    assert np.array_equal(out[0], ref)


def test_no_new_colors_same_size_and_feet():
    ref = make_frame()
    f = make_frame(arm=1)
    f[5, 14] = SPOT
    out, rep = St.steady_frames([ref, f, f])
    colors = lambda a: {tuple(c) for c in a[a[..., 3] > 0][:, :3].tolist()}   # noqa: E731
    for g in out:
        assert g.shape == ref.shape
        assert colors(g) <= colors(ref) | colors(f)
        assert np.flatnonzero((g[..., 3] > 0).any(1))[-1] == H - 1
    # 同じ元のコマは同じ結果（行って戻る 1,2,3,2 など）
    assert np.array_equal(out[1], out[2])


def test_head_shift_is_undone_when_comparing():
    """頭（上の方）全体が 1 ドット下がったコマは、ずれを戻した位置で比べ、点だけを戻す（目が二重にならない）。"""
    ref = make_frame()
    f = make_frame(head_dy=1)
    f[1:22, :] = ref[0:21, :]             # 上の方を丸ごと 1 ドット下げる（体も含めて同じずれ）
    f[0] = 0
    f[3 + 5, 14] = SPOT                   # ずれた頭の髪の中の点
    out, rep = St.steady_frames([ref, f])
    sf = rep.frames[0]
    assert sf.shift == (0, 1)
    assert (out[1][8, 14] == HAIR).all()
    # 目は 1 コマ目と同じ形のまま、1 ドット下にある
    assert (out[1][14:16, 11] == EYE).all() and (out[1][14:16, 18] == EYE).all()
    assert int(((out[1][..., :3] == EYE[:3]).all(-1)).sum()) == 4


def test_eye_is_all_or_nothing():
    """目が 1 ドット動いたとき、片方だけが戻ってちぐはぐにならない（全部戻って 1 コマ目と同じ形か、そのまま）。"""
    ref = make_frame()
    f = ref.copy()
    f[13:15, 11] = SKIN
    f[13:15, 12] = EYE                    # 左の目が右へ 1 ドット
    out, _ = St.steady_frames([ref, f])
    eye = out[1][12:16, 9:15]
    assert np.array_equal(eye, ref[12:16, 9:15]) or np.array_equal(eye, f[12:16, 9:15])
    assert int(((out[1][..., :3] == EYE[:3]).all(-1)).sum()) == 4


def test_eye_next_to_large_change_is_kept():
    """目のすぐそばで大きく形が変わった所があれば、目の違いも戻さない（迷ったら何もしない）。"""
    ref = make_frame()
    f = ref.copy()
    f[13:15, 11] = SKIN
    f[13:15, 12] = EYE
    f[9:12, 9:15] = BODY                  # 目の上に大きな違い（18 ドット）
    out, rep = St.steady_frames([ref, f])
    assert np.array_equal(out[1], f)
    assert rep.frames[0].restored_count == 0


def test_hole_and_outline_are_protected():
    """戻すと体の中に透明の穴ができる所・暗い輪郭が途切れる所は戻さない。"""
    ref = make_frame()
    ref[6, 14] = (0, 0, 0, 0)             # 1 コマ目の髪の中に透明の穴（ゴミ）
    f = make_frame()
    out, rep = St.steady_frames([ref, f])
    assert (out[1][6, 14] == HAIR).all()  # 穴を写さない
    assert rep.frames[0].reasons.get(St.REASON_HOLE) == 1
    ref2 = make_frame()
    ref2[2, 15] = HAIR                    # 1 コマ目の頭の上の輪郭に切れ目
    f2 = make_frame()
    out2, rep2 = St.steady_frames([ref2, f2])
    assert (out2[1][2, 15] == OUT).all()  # 切れ目を写さない
    assert rep2.frames[0].reasons.get(St.REASON_OUTLINE) == 1


def test_size_and_rows():
    ref = make_frame()
    f = ref.copy()
    f[5:7, 13:16] = SPOT                  # 6 ドットのかたまり
    _, rep4 = St.steady_frames([ref, f], size=4)
    _, rep8 = St.steady_frames([ref, f], size=8)
    assert rep4.frames[0].restored_count == 0 and rep8.frames[0].restored_count == 6
    _, rep_rows = St.steady_frames([ref, f], size=8, rows=4)   # 戻す範囲の外
    assert rep_rows.frames[0].restored_count == 0
    assert rep_rows.frames[0].reasons.get(St.REASON_LOWER) == 6
    info = rep8.info()
    assert info["size"] == 8 and info["frames"][0]["restored_dots"] == 6 and rep8.counts() == [0, 6]
    with pytest.raises(ValueError):
        St.validate("maybe", 4)
    with pytest.raises(ValueError):
        St.validate("on", 0)


def test_image_draws_restored_and_kept():
    ref = make_frame()
    f = make_frame(arm=2)
    f[5, 14] = SPOT
    out, rep = St.steady_frames([ref, f])
    img = St.steady_image(out, rep, 4)
    assert img.shape[2] == 4
    red = ((img[..., 0] == 255) & (img[..., 1] == 30)).sum()
    blue = ((img[..., 0] == 40) & (img[..., 2] == 255)).sum()
    assert red > 0 and blue > 0


# ---- コマの変換（convert_frames）・シート・CLI・画面 ----

@pytest.fixture(scope="module")
def strip():
    from test_sprite_frames import make_strip
    return make_strip()


def test_off_is_unchanged_and_on_records_json(strip):
    off = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))
    on = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, steady="on"))
    assert off.steady is None and on.steady is not None
    assert on.size == off.size and on.pivot == off.pivot
    anim_off = F.assemble_frames(off, [1, 2, 3, 2])
    assert "steady" not in F.frames_sheet_info(anim_off, "x.png")
    anim = F.assemble_frames(on, [1, 2, 3, 2], mirror=True)
    info = json.loads(F.frames_sheet_json(anim, "x.png"))
    assert list(info).index("steady") == list(info).index("source") + 1
    assert info["steady"]["mode"] == "on" and [f["source_frame"] for f in info["steady"]["frames"]] == [2, 3]
    assert np.array_equal(anim.frames[1], anim.frames[3])
    for a, b in zip(on.before_steady, off.frames):
        assert np.array_equal(a, b)
    # 新しい色は増えない
    cols = lambda fs: {tuple(c) for f in fs for c in f[f[..., 3] > 0][:, :3].tolist()}   # noqa: E731
    assert cols(on.frames) <= cols(off.frames)


def test_on_with_outline_thin(strip):
    on = F.convert_frames(strip, SpriteSettings(outline="thin"), F.FrameSettings(frame_count=3, steady="on"))
    assert on.steady is not None and len(on.steady.frames) == 2


def test_frame_settings_check_steady():
    with pytest.raises(ValueError, match="ちらつき"):
        F.FrameSettings(steady="yes").validate()
    with pytest.raises(ValueError):
        F.FrameSettings(steady="on", steady_size=99).validate()


@pytest.mark.parametrize("argv", [["--steady-size", "3"], ["--steady", "on", "--steady-size", "0"],
                                  ["--steady", "on", "--steady-size", "x"], ["--steady", "on"]])
def test_cli_steady_option_errors(tmp_path, argv):
    from apps.sprite.cli import main
    extra = [] if argv == ["--steady", "on"] else ["--frames"]
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.png")] + extra + argv)


def test_cli_steady_writes_json(tmp_path, strip):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    out = tmp_path / "o" / "hero.png"
    assert main([str(src), "-o", str(out), "-q", "--frames", "--frame-count", "3", "--steady", "on",
                 "--steady-size", "6", "--frame-order", "1,2,3,2"]) == 0
    js = json.loads(next(out.parent.glob("hero_frames_*.json")).read_text(encoding="utf-8"))
    assert js["steady"]["size"] == 6


def test_web_frames_steady_match_cli_bytes(tmp_path, strip):
    """「ちらつきを減らす: あり」でも、画面と CLI のファイルがバイト単位で一致する（縁取り thin との組み合わせも）。"""
    import base64

    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web, gif_for_web
    from test_sprite_frames import _cli_files, _text_bytes
    for outline in ("auto", "thin"):
        argv = ["--frame-count", "3", "--frame-name", "walk", "--steady", "on", "--frame-order", "1,2,3,2"]
        if outline == "thin":
            argv += ["--outline", "thin"]
        d = tmp_path / outline
        d.mkdir()
        cli = _cli_files(d, strip, argv)
        conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps(
            {"settings": {"outline": outline}, "frames": {"frame_count": 3, "steady": "on"}, "name": "walk"})))
        assert conv["ok"] and conv["made"], conv
        assert conv["steady"]["mode"] == "on" and conv["steady_counts"][0] == 0
        assert len(conv["steady_counts"]) == 3
        web = {"hero_sprite.png": base64.b64decode(conv["png_base64"]),
               "hero_sprite_walk_flicker.png": base64.b64decode(conv["flicker_png_base64"]),
               "hero_sprite_walk_flicker.json": _text_bytes(conv["flicker_json"])}
        sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite", "name": "walk",
                                                              "order": [1, 2, 3, 2], "mirror": True})))
        assert sheets["ok"], sheets
        for s in sheets["sheets"]:
            png = base64.b64decode(s["png_base64"])
            web[s["name"]] = png
            web[s["name"][:-4] + ".json"] = _text_bytes(s["json"])
            g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
            web[g["name"]] = base64.b64decode(g["gif_base64"])
        assert sorted(web) == sorted(cli)
        for k in cli:
            assert web[k] == cli[k], (outline, k)
        assert '"steady"' in sheets["sheets"][0]["json"]


def test_web_frames_steady_off_has_no_counts(strip):
    from retrocore.sprite.convert import encode_png, frames_convert_for_web
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    assert conv["steady"] is None and conv["steady_counts"] is None
    bad = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"steady": "maybe"}})))
    assert not bad["ok"] and "ちらつき" in bad["error"]


def test_web_page_has_steady_setting():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    assert 'id="frame-steady"' in html and "steadyText" in html
    spec = build_sprite_web.settings_spec()["frames"]["steady"]
    assert spec["default"] == "off" and [c[0] for c in spec["choices"]] == ["off", "on"]


def test_steady_runs_with_int32_numpy(strip):
    """32 ビットの NumPy のつもりで通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    from retrocore.sprite.convert import encode_png, frames_convert_for_web
    fs = F.FrameSettings(frame_count=3, steady="on")
    want = F.convert_frames(strip, SpriteSettings(), fs)
    ref = make_frame()
    f = make_frame(arm=2)
    f[5, 14] = SPOT
    want_small = St.steady_frames([ref, f])[0]
    with emulate_int32():
        got = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3, steady="on"))
        got_small, rep = St.steady_frames([ref, f])
        St.steady_image(got_small, rep, 2)
        conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"steady": "on"}})))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
    assert all(np.array_equal(a, b) for a, b in zip(want_small, got_small))
    assert conv["ok"] and conv["steady"] is not None
