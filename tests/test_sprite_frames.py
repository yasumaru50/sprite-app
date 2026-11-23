"""コマを切り分けてそろえる（`retrocore/sprite/frames.py`、CLI の --frames）のテスト。

テスト画像は合成: キノコのドット絵（`make_pixel_sprite()`）の脚のあたりだけをコマごとに変え、コマごとに上下・左右の位置をずらし、
コマごとに別の乱数で AI 風に拡大して（ドットの大きさ 19〜22px のばらつき、ぼかし、ノイズ。コマごとに線の位相が違う）、
背景色のすき間を空けて横に並べる。
"""

import json

import numpy as np
import pytest
from PIL import Image

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite.grid import GridNotFoundError
from retrocore.sprite.motion.gif import make_gif
from retrocore.sprite.motion.report import CODES, STAGES
from synth_sprite import PIX_BG, enlarge_like_ai, make_pixel_sprite

FRAMES = 3


def _variant(k: int) -> np.ndarray:
    """k 番目のコマ: 上（かさ）は同じで、いちばん下の 2 行（足）だけを左右にずらす。"""
    art = make_pixel_sprite().copy()
    h = art.shape[0]
    feet = art[h - 2:].copy()
    art[h - 2:] = 0
    shift = (-1, 0, 1)[k % 3]
    art[h - 2:] = np.roll(feet, shift, axis=1)
    return art


def make_strip(n: int = FRAMES, offsets=((0, 0), (1, 1), (0, 2)), gap: int = 90, seed: int = 0):
    """n コマを横に並べた AI 風の絵（不透明な RGB の RGBA）。offsets はコマごとの (横, 縦) のずれ（ドット）。"""
    parts = []
    for k in range(n):
        art = _variant(k)
        dx, dy = offsets[k % len(offsets)]
        canvas = np.zeros((art.shape[0] + 3, art.shape[1] + 2, 4), np.uint8)
        canvas[dy:dy + art.shape[0], dx:dx + art.shape[1]] = art
        img, _, _ = enlarge_like_ai(canvas, seed=seed + 10 * k, half_shift=0)
        parts.append(img)
    H = max(p.shape[0] for p in parts)
    gap_img = np.zeros((H, gap, 4), np.uint8)
    gap_img[..., :3] = PIX_BG
    gap_img[..., 3] = 255
    row = [gap_img]
    for p in parts:
        q = np.zeros((H, p.shape[1], 4), np.uint8)
        q[..., :3] = PIX_BG
        q[..., 3] = 255
        q[:p.shape[0]] = p
        row += [q, gap_img]
    return np.concatenate(row, 1)


@pytest.fixture(scope="module")
def strip():
    return make_strip()


@pytest.fixture(scope="module")
def result(strip):
    return F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES))


# ---- 切り分け ----

def _mask_image(mask):
    a = np.zeros(mask.shape + (4,), np.uint8)
    a[mask] = (200, 100, 50, 255)
    return a


def test_find_frames_splits_on_transparent_columns_and_attaches_small_parts():
    m = np.zeros((60, 200), bool)
    m[10:50, 10:40] = True        # 1 コマ目
    m[10:50, 100:130] = True      # 2 コマ目
    m[5:8, 45:48] = True          # 1 コマ目から離れた小さな部品（髪の先）→ 近い 1 コマ目
    m[30, 190] = True             # 背景の消し残し（ごく小さい）→ 捨てる
    boxes, notes = F.find_frames(_mask_image(m))
    assert [(b.x0, b.x1) for b in boxes] == [(10, 48), (100, 130)]
    assert boxes[0].parts == [(45, 48)] and (boxes[0].y0, boxes[0].y1) == (5, 50)
    assert any("1 コマ目に入れました" in n for n in notes) and any("消し残し" in n for n in notes)


def test_find_frames_count_mismatch_is_not_made():
    m = np.zeros((40, 120), bool)
    m[5:35, 5:30] = True
    m[5:35, 60:90] = True
    with pytest.raises(F.FramesError) as e:
        F.find_frames(_mask_image(m), frame_count=3)
    assert e.value.code == "frame_count" and e.value.extra["found"] == 2 and "指定 3" in str(e.value)
    with pytest.raises(F.FramesError) as e:
        F.find_frames(np.zeros((10, 10, 4), np.uint8))
    assert e.value.code == "no_frames"


# ---- 変換とそろえ方 ----

def test_frames_share_size_grid_and_palette(result):
    assert len(result.frames) == FRAMES
    assert len({f.shape for f in result.frames}) == 1
    # ドットの大きさは全部のコマで 1 つ（元は 19〜22px）
    assert 19 <= result.dot_size[0] <= 22 and result.dot_size[0] == result.dot_size[1]
    assert all(np.allclose(g.period, result.dot_size) for g in result.grids)
    # 色は全部のコマをまとめて単純化した共通のパレット（12 色以下）
    allcols = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in result.frames]), axis=0)
    assert len(allcols) <= 12
    # 1 ドット＝1 画素に戻ったコマの大きさは、元のドット絵（22×24 ＋ 縁取り）に近い
    w, h = result.size
    assert 20 <= w <= 28 and 22 <= h <= 30


def test_feet_are_aligned_and_top_matches(result):
    H = result.size[1]
    for f in result.frames:
        assert (f[H - 1, :, 3] > 0).any()          # どのコマも足元がいちばん下の行
    rep = F.flicker_report(result.frames, rows=result.size[1] // 2)
    # 上（かさ）は同じ絵なので、そろえたあとは違うドットがほとんど無い（ずらしも 0）
    assert all(fr.count <= 6 for fr in rep.frames), [fr.count for fr in rep.frames]
    assert all(fr.shift == (0, 0) for fr in rep.frames)


def test_align_none_keeps_source_offsets(strip):
    res = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(align="none", align_y="none"))
    # 元の絵では 2・3 コマ目が下に 1・2 ドットずれている → そろえないと足元の高さが違う
    bottoms = [int(np.flatnonzero((f[..., 3] > 0).any(1))[-1]) for f in res.frames]
    assert len(set(bottoms)) > 1
    # ちらつきは小さなずれを重ね直して数えるので、違いは「ずれ」に出る（数には出ない）
    rep = F.flicker_report(res.frames, rows=res.size[1] // 2)
    assert any(fr.shift != (0, 0) for fr in rep.frames)


@pytest.mark.parametrize("align", ["match", "head", "center"])
def test_each_horizontal_align_keeps_top_together(strip, align):
    res = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(align=align))
    rep = F.flicker_report(res.frames, rows=res.size[1] // 2)
    assert all(fr.count <= 6 for fr in rep.frames)


def test_align_y_match_follows_the_top():
    # 2 コマ目だけ足を 3 行分持ち上げた（空中のコマ）: 足元でそろえると頭が沈む。match なら頭の高さのまま
    a = make_pixel_sprite()
    air = np.zeros((a.shape[0] - 3, a.shape[1], 4), np.uint8)
    air[:] = a[:a.shape[0] - 3]
    raw = [a, air]
    foot, _, _, _ = F.align_frames(raw, "center", "foot")
    match, _, _, _ = F.align_frames(raw, "center", "match")
    tops = lambda fs: [int(np.flatnonzero((f[..., 3] > 0).any(1))[0]) for f in fs]
    assert len(set(tops(foot))) == 2
    assert len(set(tops(match))) == 1


def test_frame_settings_and_settings_are_checked(strip):
    with pytest.raises(ValueError):
        F.convert_frames(strip, SpriteSettings(), F.FrameSettings(align="left"))
    with pytest.raises(ValueError):
        F.convert_frames(strip, SpriteSettings(input_type="illust", width=32), F.FrameSettings())
    with pytest.raises(ValueError):
        F.convert_frames(strip, SpriteSettings(grid_origin=(0, 0)), F.FrameSettings())
    with pytest.raises(ValueError, match="線の並び"):
        F.convert_frames(strip, SpriteSettings(), F.FrameSettings(grid="middle"))


def test_grid_not_found_for_flat_frames():
    m = np.zeros((80, 300), bool)
    m[10:70, 10:100] = True
    m[10:70, 180:270] = True
    with pytest.raises(GridNotFoundError):
        F.convert_frames(_mask_image(m), SpriteSettings(background=False), F.FrameSettings())


def test_palette_is_used_for_all_frames(strip, result):
    pal = np.unique(result.frames[0][result.frames[0][..., 3] > 0][:, :3], axis=0)
    res = F.convert_frames(strip, SpriteSettings(palette=pal), F.FrameSettings())
    used = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in res.frames]), axis=0)
    assert {tuple(c) for c in used} <= {tuple(c) for c in pal}


# ---- 並べ方・シート ----

def test_assemble_order_durations_root_and_mirror(result):
    anim = F.assemble_frames(result, [1, 2, 3, 2], [150], [0, -1, 0, -1], "walk")
    w, h = result.size
    assert len(anim.frames) == 4 and anim.durations == [150] * 4
    assert anim.frames[0].shape == (h + 1, w, 4)
    assert np.array_equal(anim.frames[1][0:h], result.frames[1])       # 上に 1 ドット
    assert np.array_equal(anim.frames[0][1:], result.frames[0])
    info = F.frames_sheet_info(anim, "x_walk_sheet.png", [], result.dot_size)
    assert [f["source_frame"] for f in info["frames"]] == [1, 2, 3, 2]
    assert [f["offset_y"] for f in info["frames"]] == [0, -1, 0, -1]
    assert info["padding"] == {"left": 0, "right": 0, "top": 1, "bottom": 0}
    assert info["pivot"] == {"x": result.pivot[0], "y": h + 1}   # 上に足した 1 行の分だけ下（足元はフレームの下端）
    assert info["source"]["frame_count"] == 3 and info["source"]["mirrored"] is False
    assert info["skipped"] == [] and list(info)[-1] == "skipped" and info["version"] == 2
    m = F.assemble_frames(result, [1, 2], [100, 120], name="walk", mirror=True)
    assert m.name == "walk_mirror" and np.array_equal(m.frames[0], result.frames[0][:, ::-1])
    mi = F.frames_sheet_info(m, "x.png")
    assert mi["pivot"]["x"] == w - result.pivot[0] and mi["source"]["mirrored"] is True
    with pytest.raises(ValueError):
        F.assemble_frames(result, [1, 4])
    with pytest.raises(ValueError):
        F.assemble_frames(result, [1, 2], [100, 100, 100])
    with pytest.raises(ValueError):
        F.assemble_frames(result, [1, 2], root_y=[0])


def test_gif_matches_sheet_frames(result):
    anim = F.assemble_frames(result, [1, 2, 3, 2])
    gif = make_gif(anim.frames, anim.durations, 1)
    im = Image.open(__import__("io").BytesIO(gif.data))
    assert im.n_frames == 4


def test_flicker_report_marks_changed_dots(result):
    frames = [f.copy() for f in result.frames]
    ys, xs = np.nonzero(frames[1][:3, :, 3] > 0)
    frames[1][ys[0], xs[0], :3] = 255 - frames[1][ys[0], xs[0], :3]   # 上の方の 1 ドットだけ色を変える
    rep = F.flicker_report(frames, rows=result.size[1] // 2)
    base = F.flicker_report(result.frames, rows=result.size[1] // 2)
    assert rep.frames[1].count == base.frames[1].count + 1
    assert rep.frames[1].diff[ys[0], xs[0]]
    info = json.loads(F.flicker_json(rep))
    assert info["reference_frame"] == 1 and info["frames"][1]["diff_dots"] == rep.frames[1].count
    img = F.flicker_image(frames, rep, scale=4)
    assert img.shape[2] == 4 and img.shape[1] > 4 * 3 * result.size[0]
    with pytest.raises(ValueError):
        F.flicker_report(frames, rows=10_000)


def test_skipped_codes_are_registered():
    assert "frames" in STAGES
    for code in ("frame_count", "no_frames", "empty_frame", "grid_not_found"):
        assert code in CODES
    info = F.frames_skipped_info(F.FramesError("frame_count", "合いません", expected=3, found=2), "walk")
    e = info["skipped"][0]
    assert info["made"] is False and info["image"] is None and info["motion"] == "walk"
    assert (e["stage"], e["code"], e["expected"], e["result"]) == ("frames", "frame_count", 3, "held")
    g = F.frames_skipped_info(GridNotFoundError("無い"))["skipped"][0]
    assert (g["stage"], g["code"]) == ("grid", "grid_not_found")


# ---- CLI ----

def test_cli_frames(tmp_path, strip):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    out = tmp_path / "o" / "hero.png"
    rc = main([str(src), "-o", str(out), "--frames", "--frame-count", "3", "--frame-order", "1,2,3,2",
               "--frame-name", "walk", "--gif", "--mirror", "--flicker", "--frame-root-y=0,-1,0,-1", "-q"])
    assert rc == 0
    res = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=3))
    w, h = res.size
    size = f"{w}x{h + 1}_4f"   # 体の上下（0,-1,0,-1）で上に 1 行足す。並べたのは 4 フレーム
    for name in ("hero.png", f"hero_walk_{size}.png", f"hero_walk_{size}.json", f"hero_walk_{size}.gif",
                 f"hero_walk_mirror_{size}.png", f"hero_walk_mirror_{size}.json", f"hero_walk_mirror_{size}.gif",
                 "hero_walk_flicker.png", "hero_walk_flicker.json"):
        assert (out.parent / name).exists(), name
    info = json.loads((out.parent / f"hero_walk_{size}.json").read_text(encoding="utf-8"))
    assert info["image"] == f"hero_walk_{size}.png"
    assert (info["frame_width"], info["frame_height"]) == (w, h + 1)
    assert info["motion"] == "walk" and info["frame_count"] == 4
    assert [f["source_frame"] for f in info["frames"]] == [1, 2, 3, 2]
    # 数が合わない: 作らずに理由を書き、終了コード 1
    rc = main([str(src), "-o", str(tmp_path / "bad.png"), "--frames", "--frame-count", "5", "-q"])
    assert rc == 1
    sk = json.loads((tmp_path / "bad_frames_skipped.json").read_text(encoding="utf-8"))
    assert sk["skipped"][0]["code"] == "frame_count" and not (tmp_path / "bad.png").exists()


@pytest.mark.parametrize("argv", [
    ["--frame-count", "3"],                                   # --frames が無い
    ["--frames", "--motion", "breathe"],
    ["--frames", "--input-type", "illust", "--width", "32"],
    ["--frames", "--grid-origin", "0,0"],
    ["--frames", "--frame-count", "2", "--frame-order", "1,3"],
    ["--frames", "--frame-order", "1,2", "--frame-duration", "100,100,100"],
    ["--frames", "--frame-order", "1,2", "--frame-root-y=0"],
    ["--frames", "--flicker-rows", "10"],
    ["--frames", "--frame-order", "0,1"],
    ["--frame-grid", "shared"],                               # --frames が無い
    ["--frames", "--frame-grid", "middle"],
])
def test_cli_frames_option_errors(tmp_path, argv):
    from apps.sprite.cli import main
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "x.png")] + argv)
    assert e.value.code == 2


def test_single_image_conversion_is_unchanged_by_frames_options():
    """--frames を付けない 1 枚の絵の変換は、convert_sprite() のまま（frames.py を通らない）。"""
    from apps.sprite.cli import build_parser, check_args, settings_from_args
    from retrocore.sprite import convert_sprite
    p = build_parser()
    args = p.parse_args(["x.png"])
    check_args(p, args)
    assert args.frames is None
    img = enlarge_like_ai(make_pixel_sprite(), seed=3)[0]
    a = convert_sprite(img, settings_from_args(args)).image
    b = convert_sprite(img, SpriteSettings()).image
    assert np.array_equal(a, b)


# ---- 32 ビット（Pyodide）----

def test_frames_run_with_int32_numpy(strip):
    """32 ビットの NumPy のつもりで通す（np.bincount などに int64 を渡していない）。Linux・macOS でだけ動く。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    want = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES))
    with emulate_int32():
        got = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES))
        rep = F.flicker_report(got.frames)
        F.flicker_image(got.frames, rep, 2)
        anim = F.assemble_frames(got, [1, 2, 3, 2], root_y=[0, -1, 0, -1], mirror=True)
        make_gif(anim.frames, anim.durations, 2)
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))


# ---- グリッドの線の並びを全部のコマで共通にする（--frame-grid） ----

def same_drawing_strip(seed: int = 0, n: int = FRAMES, gap: int = 90, amp: int = 10) -> np.ndarray:
    """同じ絵を同じ拡大（ドットの大きさのばらつきも同じ）で n 回並べ、コマごとに別のノイズを足したもの。

    AI が同じ形に描いた正面歩きの上半身（ai08）を真似る。
    """
    art = make_pixel_sprite()
    canvas = np.zeros((art.shape[0] + 3, art.shape[1] + 2, 4), np.uint8)
    canvas[1:1 + art.shape[0], 1:1 + art.shape[1]] = art
    img, _, _ = enlarge_like_ai(canvas, seed=seed, half_shift=0)
    rng = np.random.default_rng(seed + 100)
    g = np.zeros((img.shape[0], gap, 4), np.uint8)
    g[..., :3] = PIX_BG
    g[..., 3] = 255
    row = [g]
    for _ in range(n):
        q = img.copy()
        q[..., :3] = np.clip(q[..., :3].astype(int) + rng.integers(-amp, amp + 1, q[..., :3].shape), 0, 255)
        row += [q, g]
    return np.concatenate(row, 1)


def _same_pattern(grids):
    """どのコマの線も、ずらせば同じ並び（間隔の列が同じ）か。"""
    for axis in ("xs", "ys"):
        gaps = [np.diff(getattr(g, axis))[1:-1] for g in grids]   # 端の欠けたマスは除く
        n = min(len(d) for d in gaps)
        if any(not np.array_equal(d[:n], gaps[0][:n]) for d in gaps[1:]):
            return False
    return True


def test_auto_shares_lines_for_frames_drawn_the_same():
    st = same_drawing_strip()
    res = F.convert_frames(st, SpriteSettings(), F.FrameSettings(frame_count=FRAMES))
    assert any("共通" in m for m in res.notes)
    assert _same_pattern(res.grids)
    assert len({f.shape for f in res.raw}) == 1
    each = F.convert_frames(st, SpriteSettings(), F.FrameSettings(frame_count=FRAMES, grid="each"))
    assert not any("共通" in m for m in each.notes)
    # 共通にしても、どのコマも境目の揃い方は大きく下がらない
    for a, b in zip(res.grids, each.grids):
        assert all(x > y - F.SHARED_MAX_DROP for x, y in zip(a.alignment, b.alignment))


def test_auto_keeps_each_frame_lines_when_frames_are_drawn_differently(strip, result):
    """コマごとにドットの大きさのばらつきが違う絵（make_strip）では、共通の線は境目に乗らないので使わない（PR #37 と同じ結果）。"""
    assert F.DEFAULT_GRID_MODE == "auto" and result.notes == F.convert_frames(
        strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES, grid="each")).notes
    each = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES, grid="each"))
    assert all(np.array_equal(a, b) for a, b in zip(result.frames, each.frames))
    shared = F.convert_frames(strip, SpriteSettings(), F.FrameSettings(frame_count=FRAMES, grid="shared"))
    assert any("共通" in m for m in shared.notes)


def test_cli_frame_grid(tmp_path):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(same_drawing_strip(), src)
    outs = {}
    for mode in ("each", "shared"):
        out = tmp_path / mode / "hero.png"
        assert main([str(src), "-o", str(out), "--frames", "--frame-count", "3", "--frame-grid", mode, "-q"]) == 0
        (js,) = out.parent.glob("hero_frames_*x*_3f.json")   # シートの名前に 1 コマの大きさとコマの数が入る
        outs[mode] = json.loads(js.read_text(encoding="utf-8"))
    assert outs["each"]["frame_count"] == outs["shared"]["frame_count"] == 3


# ---- シートの名前（エディタで読み込むときに入れる数）と、Piskel の Import as spritesheet でそのまま切れる形 ----

def test_sheet_file_name_has_frame_size_and_count(result):
    anim = F.assemble_frames(result, [1, 2, 3, 2], name="walk")
    w, h = result.size
    assert F.sheet_file_name("ai07", anim) == f"ai07_walk_{w}x{h}_4f.png"
    m = F.assemble_frames(result, [1, 2, 3], root_y=[0, -1, 0], name="run", mirror=True)
    assert F.sheet_file_name("x_sprite", m) == f"x_sprite_run_mirror_{w}x{h + 1}_3f.png"


def test_sheet_slices_like_piskel_import(result):
    """Piskel の Import as spritesheet は、幅・高さを指定して左から右へ（次に上から下へ）切り、余った画素は捨てる。
    シートは横 1 列・すき間なし・拡大なしなので、名前の幅・高さで切ると、そのまま元のフレームに戻る。"""
    from retrocore.sprite.motion import make_sheet
    anim = F.assemble_frames(result, [1, 2, 3, 2], root_y=[0, -1, 0, -1], name="walk")
    sheet = make_sheet(anim.frames)
    w, h = anim.size
    assert sheet.shape[:2] == (h, w * len(anim.frames))   # 余りの画素が無い
    cut = [sheet[0:h, x:x + w] for x in range(0, sheet.shape[1] - w + 1, w)]
    assert len(cut) == len(anim.frames)
    for a, b in zip(cut, anim.frames):
        assert np.array_equal(a, b)
        assert (a[..., 3] > 0).any()   # 透明だけのコマは Piskel が捨てる（番号がずれる）ので、無いこと


def test_flicker_labels_do_not_depend_on_fonts(result):
    """ちらつきの画像の文字は自前の字（Pillow のフォントは Pyodide とパソコンで字形が違う）。同じ画素になる。"""
    from PIL import ImageFont
    rep = F.flicker_report(result.frames)
    a = F.flicker_image(result.frames, rep)
    orig = ImageFont.load_default
    try:
        ImageFont.load_default = lambda *a, **k: (_ for _ in ()).throw(AssertionError("フォントを使った"))
        b = F.flicker_image(result.frames, rep)
    finally:
        ImageFont.load_default = orig
    assert np.array_equal(a, b)


# ---- ブラウザ版（Pyodide の中で呼ぶ関数）: CLI と同じバイト ----

def _cli_files(tmp_path, strip, argv):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    out = tmp_path / "cli" / "hero_sprite.png"
    assert main([str(src), "-o", str(out), "-q", "--frames", "--mirror", "--flicker", "--gif", "--gif-scale", "8"] + argv) == 0
    return {p.name: p.read_bytes() for p in out.parent.iterdir()}


def _text_bytes(text: str) -> bytes:
    import os
    return text.replace("\n", os.linesep).encode("utf-8")   # CLI の write_text()（Windows では CRLF）


@pytest.mark.parametrize("frames_req, sheet_req, argv", [
    ({"frame_count": 3}, {"name": "walk"}, ["--frame-count", "3", "--frame-name", "walk"]),
    ({"frame_count": 3, "align": "head", "align_y": "match"},
     {"name": "run", "order": [1, 2, 3, 2], "durations": [100], "root_y": [0, -1, 0, -1]},
     ["--frame-count", "3", "--align", "head", "--align-y", "match", "--frame-order", "1,2,3,2",
      "--frame-duration", "100", "--frame-root-y=0,-1,0,-1", "--frame-name", "run"]),
    ({}, {"order": [1, 2], "durations": [120, 80]}, ["--frame-order", "1,2", "--frame-duration", "120,80"]),
])
def test_web_frames_match_cli_bytes(tmp_path, strip, frames_req, sheet_req, argv):
    import base64
    from retrocore.sprite.convert import (encode_png, frames_convert_for_web, frames_sheets_for_web,
                                          gif_for_web)
    cli = _cli_files(tmp_path, strip, argv)
    name = sheet_req.get("name", F.DEFAULT_NAME)
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": frames_req, "name": name})))
    assert conv["ok"] and conv["made"], conv
    web = {"hero_sprite.png": base64.b64decode(conv["png_base64"]),
           f"hero_sprite_{name}_flicker.png": base64.b64decode(conv["flicker_png_base64"]),
           f"hero_sprite_{name}_flicker.json": _text_bytes(conv["flicker_json"])}
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite",
                                                          "mirror": True, **sheet_req})))
    assert sheets["ok"], sheets
    assert [s["mirror"] for s in sheets["sheets"]] == [False, True]
    for s in sheets["sheets"]:
        png = base64.b64decode(s["png_base64"])
        web[s["name"]] = png
        web[s["name"][:-4] + ".json"] = _text_bytes(s["json"])
        g = json.loads(gif_for_web(png, s["json"], json.dumps({"scale": 8})))
        assert g["ok"], g
        web[g["name"]] = base64.b64decode(g["gif_base64"])
    assert sorted(web) == sorted(cli)
    for k in cli:
        assert web[k] == cli[k], k
    assert len(conv["boxes"]) == conv["frame_count"]


def test_web_frames_outline_thin_match_cli_bytes(tmp_path, strip):
    """縁取り thin（太い外側の輪郭を 1 ドットにする）でも、画面と CLI のファイルがバイト単位で一致する。"""
    import base64
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web
    cli = _cli_files(tmp_path, strip, ["--frame-count", "3", "--frame-name", "walk", "--outline", "thin"])
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps(
        {"settings": {"outline": "thin"}, "frames": {"frame_count": 3}, "name": "walk"})))
    assert conv["ok"] and conv["made"], conv
    assert any("縁取り（thin" in s for s in conv["steps"])
    assert base64.b64decode(conv["png_base64"]) == cli["hero_sprite.png"]
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite",
                                                          "mirror": True, "name": "walk"})))
    assert sheets["ok"], sheets
    for sh in sheets["sheets"]:
        assert base64.b64decode(sh["png_base64"]) == cli[sh["name"]], sh["name"]
        assert _text_bytes(sh["json"]) == cli[sh["name"][:-4] + ".json"]


def test_web_frames_not_made_gives_skipped_json(tmp_path, strip):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    from retrocore.sprite.convert import encode_png, frames_convert_for_web
    res = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 5}, "name": "walk"})))
    assert res["ok"] and not res["made"] and "コマの数が合いません" in res["error"]
    assert len(res["boxes"]) == 3
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    assert main([str(src), "-o", str(tmp_path / "hero_sprite.png"), "-q", "--frames", "--frame-count", "5",
                 "--frame-name", "walk"]) == 1
    assert _text_bytes(res["skipped_json"]) == (tmp_path / "hero_sprite_walk_skipped.json").read_bytes()


def test_web_frames_errors(strip):
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web
    bad = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"align": "left"}})))
    assert not bad["ok"] and "横のそろえ方" in bad["error"]
    conv = json.loads(frames_convert_for_web(encode_png(strip), "{}"))
    for req, word in (({"order": [1, 9]}, "順番"), ({"name": "歩き"}, "名前"), ({"token": -1}, "もう一度変換"),
                      ({"order": [1, 2], "root_y": [0]}, "上下")):
        r = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], **req})))
        assert not r["ok"] and word in r["error"], (req, r["error"])


def test_web_frames_run_with_int32_numpy(strip):
    """画面の経路（frames_convert_for_web → frames_sheets_for_web）も 32 ビットの NumPy のつもりで通す。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web
    data = encode_png(strip)
    want = json.loads(frames_convert_for_web(data, "{}"))
    with emulate_int32():
        got = json.loads(frames_convert_for_web(data, "{}"))
        sheets = json.loads(frames_sheets_for_web(json.dumps({"token": got["token"], "order": [1, 2, 3, 2],
                                                              "root_y": [0, -1, 0, -1], "mirror": True})))
    assert got["ok"] and got["png_base64"] == want["png_base64"] and got["flicker_png_base64"] == want["flicker_png_base64"]
    assert sheets["ok"] and len(sheets["sheets"]) == 2


def test_web_page_has_frames_mode():
    """ブラウザ版の HTML に「コマが並んだ絵」の欄があり、Python の関数を呼んでいる（中身の一致は上のテスト）。"""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    import build_sprite_web
    html = build_sprite_web.build()
    for key in ('id="kind-frames"', 'id="frame-count"', 'id="frame-order"', 'id="frame-root"', 'id="dl-mirror"',
                'id="dl-flicker"', 'id="dl-skipped"', 'id="piskel-hint"', "frames_convert_for_web", "frames_sheets_for_web",
                "Import as spritesheet"):
        assert key in html, key
    spec = build_sprite_web.settings_spec()["frames"]
    assert spec["align"]["default"] == F.DEFAULT_ALIGN and spec["align_y"]["default"] == F.DEFAULT_ALIGN_Y
    assert spec["duration"] == F.DEFAULT_DURATION and spec["name"] == F.DEFAULT_NAME
    assert {c[0] for c in spec["align"]["choices"]} <= set(F.ALIGNS)
    assert {c[0] for c in spec["align_y"]["choices"]} <= set(F.ALIGNS_Y)
