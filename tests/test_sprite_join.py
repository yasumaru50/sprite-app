"""別々のファイルの絵をつなげる（`retrocore/sprite/join.py`、CLI の `--frames --join`、画面の「つなげるファイル」）のテスト。

- ファイルが 1 つのときは今までの `convert_frames()` とバイト単位で同じ（シートの JSON にファイルの記録は付かない）
- 1 枚の絵を 2 つのファイルに切り分けてつなげると、1 枚のまま変換したときと同じコマになる
- 並び順を入れ替えても、各ファイルのコマの中身は同じで、番号（sources）だけが変わる。パレットは全部のファイルで 1 つ
- キャラの高さ: 既定は合わせずに違いを警告する（skipped の join_height。しゃがみ・ジャンプのファイルでほかを小さくしないため）。
  合わせる指定（match_height・CLI の --join-match-height）のときは 1 割までの違いを格子を変えて合わせ、それより大きい違いは警告だけ
- 直しの一覧（edits.json）: つなげたときはファイルの何コマ目か・置いた位置を書き、並び順を入れ替えても同じ絵の同じ所に付く。
  前の形（版 1）は番号で当てる。外したファイルの直しは当てない
- CLI と画面（frames_convert_bytes にファイルの並びを渡す）の出力はバイト単位で同じ
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import join as J
from retrocore.sprite import touchup as T
from retrocore.sprite.motion.report import CODES
from synth_sprite import PIX_BG, enlarge_like_ai, make_pixel_sprite
from test_sprite_frames import _variant, make_strip


def _split(strip: np.ndarray, at: int) -> list[np.ndarray]:
    """背景を除いた絵のコマの間（at コマ目と次のコマの間の真ん中）で 2 つに切る。"""
    cur, _ = F._remove_background(strip, SpriteSettings())
    boxes, _ = F.find_frames(cur)
    cut = (boxes[at - 1].x1 + boxes[at].x0) // 2
    return [strip[:, :cut].copy(), strip[:, cut:].copy()]


def _row(arts: list[np.ndarray], seed: int = 0, sizes=(19, 22), gap: int = 90) -> np.ndarray:
    """ドット絵の並びを AI 風に拡大して、背景のすき間を空けて横に並べた絵。"""
    parts = []
    for k, art in enumerate(arts):
        canvas = np.zeros((art.shape[0] + 3, art.shape[1] + 2, 4), np.uint8)
        canvas[:art.shape[0], :art.shape[1]] = art
        img, _, _ = enlarge_like_ai(canvas, sizes=sizes, seed=seed + 10 * k, half_shift=0)
        parts.append(img)
    H = max(p.shape[0] for p in parts)
    bg = np.zeros((H, gap, 4), np.uint8)
    bg[..., :3] = PIX_BG
    bg[..., 3] = 255
    row = [bg]
    for p in parts:
        q = np.zeros((H, p.shape[1], 4), np.uint8)
        q[..., :3] = PIX_BG
        q[..., 3] = 255
        q[:p.shape[0]] = p
        row += [q, bg]
    return np.concatenate(row, 1)


@pytest.fixture(scope="module")
def strip4():
    return make_strip(4)


@pytest.fixture(scope="module")
def halves(strip4):
    return _split(strip4, 2)


# ---- 1 つのとき・切り分けてつなげたとき ----

def test_single_file_is_the_same_as_convert_frames(strip4):
    want = F.convert_frames(strip4, SpriteSettings(), F.FrameSettings())
    got = J.convert_frame_files([("walk.png", strip4)], SpriteSettings(), F.FrameSettings())
    assert not got.joined and got.sources == [("walk.png", k) for k in range(1, 5)]
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
    assert [s.description for s in want.steps] == [s.description for s in got.steps]
    a, b = F.assemble_frames(want), F.assemble_frames(got)
    assert F.frames_sheet_json(a, "x.png", want.skipped, want.dot_size) == \
        F.frames_sheet_json(b, "x.png", got.skipped, got.dot_size)
    assert np.array_equal(F.flicker_image(got.frames, F.flicker_report(got.frames), 2, file_starts=J.file_starts(got)),
                          F.flicker_image(want.frames, F.flicker_report(want.frames), 2))


@pytest.mark.parametrize("grid", ["each", "auto"])
def test_split_into_two_files_gives_the_same_frames(strip4, halves, grid):
    fs = F.FrameSettings(grid=grid)
    one = F.convert_frames(strip4, SpriteSettings(), fs)
    two = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), fs)
    assert two.joined and [f.frame_count for f in two.files] == [2, 2] and [f.first for f in two.files] == [1, 3]
    assert two.sources == [("a.png", 1), ("a.png", 2), ("b.png", 1), ("b.png", 2)]
    assert all(np.array_equal(a, b) for a, b in zip(one.frames, two.frames))
    assert J.file_starts(two) == [3]
    assert all(not j["large"] for j in two.joins) and [j["loop"] for j in two.joins] == [False, True]


def test_reorder_keeps_each_file_frames_and_shares_one_palette(halves):
    s = SpriteSettings(colors=8)
    ab = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], s, F.FrameSettings())
    ba = J.convert_frame_files([("b.png", halves[1]), ("a.png", halves[0])], s, F.FrameSettings())
    assert ba.sources == [("b.png", 1), ("b.png", 2), ("a.png", 1), ("a.png", 2)]
    raw_ab, raw_ba = dict(zip(ab.sources, ab.raw)), dict(zip(ba.sources, ba.raw))
    assert all(np.array_equal(raw_ab[k], raw_ba[k]) for k in raw_ab)
    # パレットは全部のファイルで 1 つ（どのコマの色も、全部のコマを合わせた色の中。色数は設定以下）
    pal = T.palette_of(ab.frames)
    assert len(pal) <= 8 and set(pal) == set(T.palette_of(ba.frames))
    assert len(pal) < len(T.palette_of(ab.frames[:2])) + len(T.palette_of(ab.frames[2:]))


def test_frame_count_is_the_total_and_sheet_json_records_files(halves):
    with pytest.raises(F.FramesError) as e:
        J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings(frame_count=3))
    assert e.value.code == "frame_count" and e.value.extra["files"] == [["a.png", 2], ["b.png", 2]]
    res = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(),
                                F.FrameSettings(frame_count=4))
    anim = F.assemble_frames(res, [4, 3, 1])
    info = json.loads(F.frames_sheet_json(anim, "x.png", res.skipped, res.dot_size))
    assert [(f["source_frame"], f["source_file"], f["source_file_frame"]) for f in info["frames"]] == \
        [(4, "b.png", 2), (3, "b.png", 1), (1, "a.png", 1)]
    assert [f["name"] for f in info["source"]["files"]] == ["a.png", "b.png"] and info["source"]["frame_count"] == 4
    assert info["source"]["joins"][0]["from_file"] == "a.png"


def test_file_keys_make_same_names_unique():
    assert J.file_keys(["a/walk.png", "b/walk.png", "run.png", "c\\walk.png"]) == ["walk.png", "walk.png#2", "run.png",
                                                                                     "walk.png#3"]


# ---- キャラの高さ ----

def _tall(h: int) -> list[np.ndarray]:
    """足だけ違う 2 コマ（高さ h ドットのキノコ）。"""
    out = []
    for k in range(2):
        art = make_pixel_sprite(h=h).copy()
        feet = art[-2:].copy()
        art[-2:] = np.roll(feet, (-1, 1)[k], axis=1)
        out.append(art)
    return out


@pytest.fixture(scope="module")
def base_file():
    return _row([_variant(0), _variant(1)])


def test_height_is_not_matched_by_default_but_warned(base_file):
    tall = _row(_tall(26), seed=5)        # 24 → 26 ドット（約 8%）
    res = J.convert_frame_files([("a.png", base_file), ("b.png", tall)], SpriteSettings(), F.FrameSettings())
    b = res.files[1]
    assert b.status == "off" and b.scale == 1.0 and b.height_used == b.height > res.files[0].height
    e = [e for e in res.skipped if e["code"] == "join_height"]
    assert len(e) == 1 and e[0]["file"] == "b.png" and e[0]["result"] == "held"
    assert any("b.png" in w and "印を付けると" in w and "--join-match-height" in w for w in res.warnings)
    assert not any("倍" in n for n in res.notes)


def test_height_within_ten_percent_is_matched_by_the_grid(base_file):
    tall = _row(_tall(26), seed=5)        # 24 → 26 ドット（約 8%）
    res = J.convert_frame_files([("a.png", base_file), ("b.png", tall)], SpriteSettings(), F.FrameSettings(),
                                match_height=True)
    a, b = res.files
    assert (a.status, b.status) == ("base", "adjusted") and b.height > a.height and b.height_used == a.height
    assert 1.03 < b.scale < 1.15 and b.dot_size[0] > b.detected[0]
    assert any("b.png" in n and "倍" in n for n in res.notes)
    assert max(f.shape[0] for f in res.raw[2:]) == max(f.shape[0] for f in res.raw[:2])
    assert not [e for e in res.skipped if e["code"] == "join_height"]
    info = b.info()
    assert info["height_match"] == "adjusted" and info["grid_scale"] == round(b.scale, 4)


def test_height_far_apart_is_warned_and_not_matched(base_file):
    big = _row(_tall(30), seed=7)         # 24 → 30 ドット（25%）
    res = J.convert_frame_files([("a.png", base_file), ("b.png", big)], SpriteSettings(), F.FrameSettings(),
                                match_height=True)
    b = res.files[1]
    assert b.status == "too_far" and b.scale == 1.0 and b.height_used == b.height
    e = [e for e in res.skipped if e["code"] == "join_height"]
    assert len(e) == 1 and e[0]["file"] == "b.png" and e[0]["result"] == "held" and e[0]["target"] == res.files[0].height
    assert any("b.png" in w and "描き直" in w for w in res.warnings)
    assert "join_height" in CODES and "join_shift" in CODES


def test_join_report_warns_large_shift_at_the_seam():
    f = np.zeros((20, 20, 4), np.uint8)
    f[5:20, 4:10] = (10, 20, 30, 255)
    g = np.zeros_like(f)
    g[5:20, 12:18] = (10, 20, 30, 255)     # 体が 8 ドット右
    files = [J.JoinFile("a.png", 2, 1, (6, 6), (6, 6), 15, 15), J.JoinFile("b.png", 1, 3, (6, 6), (6, 6), 15, 15)]
    joins, warns, skipped = J.join_report([f, f, g], files, 0.3)
    assert joins[0]["large"] and joins[0]["shift"]["x"] == 8 and joins[1]["loop"] and joins[1]["shift"]["x"] == -8
    assert len(warns) == 2 and "a.png の最後（2 コマ目）→ b.png の最初（3 コマ目）" in warns[0]
    assert skipped[0]["code"] == "join_shift" and skipped[0]["frames"] == [1, 2]
    joins, warns, _ = J.join_report([f, f, f], files, 0.3)
    assert not warns and not any(j["large"] for j in joins)


def test_flicker_image_marks_file_seams(halves):
    res = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings())
    rep = F.flicker_report(res.frames)
    plain = F.flicker_image(res.frames, rep, 2)
    marked = F.flicker_image(res.frames, rep, 2, file_starts=J.file_starts(res))
    assert marked.shape[0] == plain.shape[0] + 12
    w = res.frames[0].shape[1] * 2
    seam_x = 2 + 2 * (w + 2) - 1        # 3 コマ目の前のすき間
    assert tuple(marked[marked.shape[0] - 1, seam_x]) == F.JOIN_COLOR


# ---- 直しの一覧 ----

def _edit_one(res, k: int) -> tuple[T.Touchup, int, int, str]:
    """k コマ目のいちばん上の不透明なドットを、パレットの別の色にする直し。"""
    f = res.frames[k - 1]
    ys, xs = np.nonzero(f[..., 3])
    x, y = int(xs[np.argmin(ys)]), int(ys.min())
    pal = T.palette_of(res.frames)
    cur = "#%02x%02x%02x" % tuple(int(v) for v in f[y, x, :3])
    color = next(c for c in pal if c != cur)
    w, h = res.size
    return T.Touchup(frames={k: [T.Dot(x, y, color, cur)]}, frame_width=w, frame_height=h, frame_count=len(res.frames)), x, y, color


def test_edits_follow_the_file_after_reorder(halves):
    ab = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings())
    t, x, y, color = _edit_one(ab, 3)                     # b.png の 1 コマ目
    pal = T.palette_of(ab.frames)
    text = T.touchup_json(t, pal, ab.sources, ab.offsets)
    d = json.loads(text)
    assert d["version"] == 2 and d["files"] == ["a.png", "b.png"]
    assert d["frames"][0]["file"] == "b.png" and d["frames"][0]["file_frame"] == 1 and d["frames"][0]["origin"] == list(ab.offsets[2])
    assert T.touchup_dict(t, pal, ab.sources, ab.offsets) == d
    loaded = T.load_touchup(text)
    assert loaded.frame_files == {3: ("b.png", 1)}
    # 並び順を入れ替えて変換し直す: b.png の 1 コマ目は 1 コマ目になり、置いた位置の違いだけずらして同じ絵の所に付く
    ba = J.convert_frame_files([("b.png", halves[1]), ("a.png", halves[0])], SpriteSettings(),
                               F.FrameSettings(edits=loaded))
    assert ba.touchup is not None and [f.frame for f in ba.touchup.frames] == [1]
    dx, dy = ba.offsets[0][0] - ab.offsets[2][0], ba.offsets[0][1] - ab.offsets[2][1]
    px = ba.frames[0][y + dy, x + dx]
    assert "#%02x%02x%02x" % tuple(int(v) for v in px[:3]) == color and px[3] == 255
    assert ba.touchup.changed_under_count == 0
    # シートの JSON の edits にもファイルの記録
    info = json.loads(F.frames_sheet_json(F.assemble_frames(ba), "x.png", ba.skipped, ba.dot_size))
    assert info["edits"]["frames"][0]["file"] == "b.png" and info["edits"]["frames"][0]["file_frame"] == 1
    # b.png を外すと当てない
    with pytest.raises(T.TouchupError) as e:
        J.convert_frame_files([("a.png", halves[0])], SpriteSettings(), F.FrameSettings(edits=loaded))
    assert e.value.code == T.CODE_FILE and "b.png の 1 コマ目" in str(e.value)


def test_old_edits_without_files_are_applied_by_number(halves):
    ab = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings())
    t, x, y, color = _edit_one(ab, 3)
    text = T.touchup_json(t, T.palette_of(ab.frames))     # 前の形（版 1。ファイルの記録なし）
    d = json.loads(text)
    assert d["version"] == 1 and "files" not in d and "file" not in d["frames"][0]
    res = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(),
                                F.FrameSettings(edits=T.load_touchup(text)))
    assert [f.frame for f in res.touchup.frames] == [3]
    assert "#%02x%02x%02x" % tuple(int(v) for v in res.frames[2][y, x, :3]) == color


def test_single_file_edits_file_is_unchanged(strip4):
    res = J.convert_frame_files([("walk.png", strip4)], SpriteSettings(), F.FrameSettings())
    t, *_ = _edit_one(res, 2)
    pal = T.palette_of(res.frames)
    assert T.touchup_json(t, pal, res.sources, res.offsets) == T.touchup_json(t, pal)


def test_edits_file_checks():
    with pytest.raises(ValueError, match="file と file_frame"):
        T.touchup_from_dict({"frames": [{"source_frame": 1, "file": "a.png", "dots": []}]})
    with pytest.raises(ValueError, match="新しい版"):
        T.touchup_from_dict({"version": 4, "frames": []})


# ---- 差し替え ----

def test_swaps_record_the_base_frame_and_warn_after_reorder():
    from retrocore.sprite.swap import Swap, SwapSet, base_warning, load_swaps, swaps_json
    s = SwapSet(swaps=[Swap("blink", {"closed": [T.Dot(1, 1, "#000000", "#ffffff")]})], frame_width=10, frame_height=10)
    assert '"base"' not in swaps_json(s, ["#000000", "#ffffff"])     # つなげていないときは前と同じ形
    text = swaps_json(s, ["#000000", "#ffffff"], ("a.png", 1))
    got = load_swaps(text)
    assert got.base == ("a.png", 1)
    assert base_warning(got, [("a.png", 1), ("b.png", 1)]) is None
    assert "a.png の 1 コマ目" in base_warning(got, [("b.png", 1), ("a.png", 1)])


# ---- CLI と画面 ----

def test_cli_join_and_web_write_the_same_bytes(tmp_path, halves):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    from retrocore.sprite.convert import frames_convert_bytes, frames_sheets
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    save_rgba(halves[0], a)
    save_rgba(halves[1], b)
    out = tmp_path / "out"
    assert main([str(a), str(b), "--frames", "--join", "-o", str(out / "dance.png"), "-q", "--mirror", "--flicker",
                 "--frame-name", "dance"]) == 0
    w = frames_convert_bytes([("a.png", a.read_bytes()), ("b.png", b.read_bytes())], {"name": "dance"})
    assert w["ok"] and w["made"] and [f["name"] for f in w["files"]] == ["a.png", "b.png"] and w["file_starts"] == [3]
    assert w["sources"][2] == ["b.png", 1] and len(w["offsets"]) == 4 and w["files"][1]["boxes"]
    assert (out / "dance.png").read_bytes() == w["png"]
    assert (out / "dance_dance_flicker.png").read_bytes() == w["flicker_png"]
    sh = frames_sheets({"token": w["token"], "base": "dance", "name": "dance", "mirror": True})
    assert sh["ok"]
    for s in sh["sheets"]:
        assert (out / s["name"]).read_bytes() == s["png"]
        assert (out / s["name"]).with_suffix(".json").read_text(encoding="utf-8") == s["json"]
    info = json.loads(sh["sheets"][0]["json"])
    assert [f["name"] for f in info["source"]["files"]] == ["a.png", "b.png"]
    # 1 枚のとき（--join なし）の画面の出力は、ファイルの記録の無い前の形
    one = frames_convert_bytes(a.read_bytes(), {"name": "dance"})
    s1 = frames_sheets({"token": one["token"], "base": "a", "name": "dance"})
    assert "files" not in json.loads(s1["sheets"][0]["json"])["source"]
    assert one["files"][0]["name"] == "image" and one["file_starts"] == []


def test_web_edits_follow_reorder(halves):
    from retrocore.sprite.convert import encode_png, frames_convert_bytes, frames_edits
    da, db = encode_png(halves[0]), encode_png(halves[1])
    res = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings())
    t, x, y, color = _edit_one(res, 3)
    d = T.touchup_dict(t, T.palette_of(res.frames), res.sources, res.offsets)
    w2 = frames_convert_bytes([("b.png", db), ("a.png", da)], {})
    got = frames_edits({"token": w2["token"], "edits": d})
    assert got["ok"] and got["applied"]
    assert [f["source_frame"] for f in got["edits"]["frames"]] == [1] and got["edits"]["frames"][0]["file"] == "b.png"
    assert '"file": "b.png", "file_frame": 1' in got["edits_json"]


def test_cli_join_option_errors(tmp_path, halves):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    a = tmp_path / "a.png"
    save_rgba(halves[0], a)
    for argv in ([str(a), "--join"], [str(a), "--frames", "--join-match-height"]):
        with pytest.raises(SystemExit):
            main(argv)


def test_join_runs_with_int32_numpy(halves):
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    tall = _row(_tall(26), seed=5)
    want = J.convert_frame_files([("a.png", halves[0]), ("b.png", tall)], SpriteSettings(), F.FrameSettings())
    with emulate_int32():
        got = J.convert_frame_files([("a.png", halves[0]), ("b.png", tall)], SpriteSettings(), F.FrameSettings())
        F.flicker_image(got.frames, F.flicker_report(got.frames), 2, file_starts=J.file_starts(got))
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))


def test_cli_join_match_height_and_web_write_the_same_bytes(tmp_path, base_file):
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    from retrocore.sprite.convert import frames_convert_bytes, frames_sheets
    tall = _row(_tall(26), seed=5)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    save_rgba(base_file, a)
    save_rgba(tall, b)
    for flag, status in (([], "off"), (["--join-match-height"], "adjusted")):
        out = tmp_path / ("out" + "".join(flag))
        assert main([str(a), str(b), "--frames", "--join", "-o", str(out / "j.png"), "-q"] + flag) == 0
        w = frames_convert_bytes([("a.png", a.read_bytes()), ("b.png", b.read_bytes())], {"match_height": bool(flag)})
        assert w["files"][1]["height_match"] == status
        sh = frames_sheets({"token": w["token"], "base": "j"})
        s = sh["sheets"][0]
        assert (out / s["name"]).read_bytes() == s["png"]
        assert (out / s["name"]).with_suffix(".json").read_text(encoding="utf-8") == s["json"]


def test_web_swaps_warn_when_first_frame_changes(halves):
    """画面で描いた差し替え（1 コマ目の記録 base 付き）は、並び順を変えて 1 コマ目が別の絵になると注意を返す（CLI の --swaps と同じ文）。"""
    from retrocore.sprite.convert import encode_png, frames_convert_bytes, frames_sheets
    from retrocore.sprite.swap import base_warning, load_swaps
    da, db = encode_png(halves[0]), encode_png(halves[1])
    w = frames_convert_bytes([("a.png", da), ("b.png", db)], {})
    res = J.convert_frame_files([("a.png", halves[0]), ("b.png", halves[1])], SpriteSettings(), F.FrameSettings())
    f = res.frames[0]
    ys, xs = np.nonzero(f[..., 3])
    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    pal = T.palette_of(res.frames)
    cur = "#%02x%02x%02x" % tuple(int(v) for v in f[y, x, :3])
    other = next(c for c in pal if c != cur)
    w0, h0 = res.size
    d = {"version": 1, "kind": "sprite_swaps", "name": "frames", "frame_width": w0, "frame_height": h0, "palette": pal,
         "base": {"file": "a.png", "file_frame": 1},
         "swaps": [{"type": "blink", "closed": [[x, y, pal.index(other), pal.index(cur)]]}]}
    same = frames_sheets({"token": w["token"], "base": "j", "swaps": d})
    assert same["ok"] and "swaps_base_warning" not in same
    assert '"base": {"file": "a.png", "file_frame": 1}' in same["swaps_json"]
    w2 = frames_convert_bytes([("b.png", db), ("a.png", da)], {})
    moved = frames_sheets({"token": w2["token"], "base": "j", "swaps": d})
    assert moved["ok"] and moved["swaps_base_warning"] == base_warning(load_swaps(json.dumps(d)), [("b.png", 1)])
    assert "a.png の 1 コマ目" in moved["swaps_base_warning"]
    assert '"base": {"file": "a.png", "file_frame": 1}' in moved["swaps_json"]   # 保存する一覧は描いたときの記録のまま
