"""宙に浮くコマ（縦のそろえ方 keep・コマごとの上げ下げ）と、コマごとの左右反転（順番の r）のテスト。

`retrocore/sprite/frames.py`（そろえ方・順番・シート）、`touchup.py`（裏返したコマだけの直し・版 3 の edits.json）、
`swap.py`（裏返したコマへの差し替え）、`edit.py`（読み戻し）、CLI の --align-y keep・--frame-shift-y・--frame-order 1,2,3,2r。
テスト画像はコードで合成する（`test_sprite_frames.make_strip()`）。
"""

import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import frames as F
from retrocore.sprite import swap as S
from retrocore.sprite import touchup as T
from retrocore.sprite.convert import encode_png

JUMP = ((0, 6), (0, 0), (0, 3))     # 元の絵での上下のずれ（ドット）: 2 コマ目がいちばん高い（跳んだコマ）


def _hex(p) -> str:
    return "#%02x%02x%02x" % (int(p[0]), int(p[1]), int(p[2]))


def _bottom(f) -> int:
    rows = np.flatnonzero((f[..., 3] > 0).any(1))
    return int(rows[-1]) + 1


def make_jump_strip(n: int = 3, offsets=JUMP, seed: int = 0, gap: int = 90):
    """`test_sprite_frames.make_strip()` と同じ作り方で、コマごとの上下のずれ（ドット）を大きくできるもの。"""
    from synth_sprite import PIX_BG, enlarge_like_ai
    from test_sprite_frames import _variant
    pad = max(dy for _, dy in offsets) + 1
    parts = []
    for k in range(n):
        art = _variant(k)
        dx, dy = offsets[k % len(offsets)]
        canvas = np.zeros((art.shape[0] + pad, art.shape[1] + 2, 4), np.uint8)
        canvas[dy:dy + art.shape[0], dx:dx + art.shape[1]] = art
        img, _, _ = enlarge_like_ai(canvas, seed=seed + 10 * k, half_shift=0)
        parts.append(img)
    H = max(p.shape[0] for p in parts)
    row = []
    for p in [None] + [q for part in parts for q in (part, None)]:
        q = np.zeros((H, gap if p is None else p.shape[1], 4), np.uint8)
        q[..., :3] = PIX_BG
        q[..., 3] = 255
        if p is not None:
            q[:p.shape[0]] = p
        row.append(q)
    return np.concatenate(row, 1)


@pytest.fixture(scope="module")
def jump_strip():
    return make_jump_strip()


@pytest.fixture(scope="module")
def foot(jump_strip):
    return F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3))


@pytest.fixture(scope="module")
def keep(jump_strip):
    return F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, align_y="keep"))


@pytest.fixture(scope="module")
def plain():
    from test_sprite_frames import make_strip
    return F.convert_frames(make_strip(), SpriteSettings(), F.FrameSettings(frame_count=3))


# ---- 元の絵の高さを保つ（keep） ----

def test_keep_uses_source_height(foot, keep):
    # 足元でそろえると、どのコマも足元がキャンバスの下端
    assert {_bottom(f) for f in foot.frames} == {foot.size[1]}
    # keep は、いちばん低い足元（1 コマ目）が下端で、跳んだコマ（2 コマ目）は元の絵のずれ（6 ドット）くらい上
    air = [keep.size[1] - _bottom(f) for f in keep.frames]
    assert air[0] == 0
    assert 5 <= air[1] <= 7 and 2 <= air[2] <= 4
    # キャンバスは跳んだ分だけ高い（全部のコマが入る。全コマ同じ大きさ）
    assert keep.size[1] >= foot.size[1] + 5
    assert len({f.shape for f in keep.frames}) == 1


def test_keep_sheet_json_has_ground_line(keep, foot):
    anim = F.assemble_frames(keep, name="jump")
    info = F.frames_sheet_info(anim, "x.png")
    keys = list(info)
    assert keys[keys.index("pivot") + 1] == "ground_y"
    assert info["ground_y"] == info["pivot"]["y"] == keep.size[1]
    assert [f["air"] for f in info["frames"]] == [keep.size[1] - _bottom(f) for f in keep.frames]
    assert info["source"]["align_y"] == "keep"
    # 足元のときは今までどおり（ground_y・air は書かない）
    info0 = F.frames_sheet_info(F.assemble_frames(foot), "x.png")
    assert "ground_y" not in info0 and all("air" not in f for f in info0["frames"])


def test_keep_with_root_y_counts_air_from_ground(keep):
    anim = F.assemble_frames(keep, root_y=[0, -1, 0])
    info = F.frames_sheet_info(anim, "x.png")
    assert info["ground_y"] == info["pivot"]["y"] == anim.size[1] - info["padding"]["bottom"]
    assert info["frames"][1]["air"] == keep.size[1] - _bottom(keep.frames[1]) + 1


def test_keep_join_aligns_files_by_lowest_foot():
    from retrocore.sprite.join import convert_frame_files
    from test_sprite_frames import make_strip
    a = make_jump_strip(n=2, offsets=((0, 0), (0, 4)), seed=1)     # 2 コマ目が足元（1 コマ目が 4 ドット上）
    b = make_jump_strip(n=2, offsets=((0, 9), (0, 9)), seed=7)     # 元の絵では全部低い所にあるファイル
    fs = F.FrameSettings(align_y="keep")
    res = convert_frame_files([("a.png", a), ("b.png", b)], SpriteSettings(), fs)
    air = [res.size[1] - _bottom(f) for f in res.frames]
    # ファイルの中では元の絵の上下のまま、ファイルどうしはいちばん低い足元でそろう
    assert air[1] == 0 and 3 <= air[0] <= 5
    assert min(air[2:]) == 0 and max(air[2:]) <= 1


def test_none_is_unchanged(jump_strip):
    # 前からある none は変わらない（keep を足しても）
    r = F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, align="none", align_y="none"))
    assert r.align_y == "none" and r.shift_y is None


# ---- コマごとの上げ下げ ----

def test_shift_y_moves_frames_and_grows_canvas(foot, jump_strip):
    r = F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, shift_y=(0, -5, 2)))
    air = [r.size[1] - _bottom(f) for f in r.frames]
    assert air == [2, 7, 0]
    assert r.size[1] == foot.size[1] + 7
    assert r.shift_y == [0, -5, 2]
    info = F.frames_sheet_info(F.assemble_frames(r), "x.png")
    assert info["source"]["shift_y"] == [0, -5, 2]
    assert [f["air"] for f in info["frames"]] == [2, 7, 0]
    # 上げ下げしたコマの絵そのものは変わらない（置いた位置だけ）
    for a, b in zip(foot.raw, r.raw):
        assert np.array_equal(a, b)


def test_shift_y_with_keep_adds_to_source_height(keep, jump_strip):
    r = F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, align_y="keep", shift_y=(0, -2, 0)))
    a0 = [keep.size[1] - _bottom(f) for f in keep.frames]
    a1 = [r.size[1] - _bottom(f) for f in r.frames]
    assert a1 == [a0[0], a0[1] + 2, a0[2]]


def test_shift_y_checks(jump_strip):
    with pytest.raises(ValueError, match="上げ下げの数"):
        F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, shift_y=(0, -1)))
    for bad in ((), (0, True, 0), (0, 1.5, 0), (0, -999, 0)):
        with pytest.raises(ValueError, match="上げ下げ"):
            F.FrameSettings(shift_y=bad).validate()


# ---- 順番の r ----

@pytest.mark.parametrize("text, nums, flips", [
    ("1,2,3,2r", [1, 2, 3, 2], [False, False, False, True]),
    (" 1, 2R ,3 ", [1, 2, 3], [False, True, False]),
    ([1, "2r", 3], [1, 2, 3], [False, True, False]),
    ([1, 2, 3], [1, 2, 3], [False, False, False]),
])
def test_parse_order_flips(text, nums, flips):
    assert F.parse_order_flips(text, 3) == (nums, flips)
    assert F.order_text(nums, flips) == ",".join(f"{k}{'r' if f else ''}" for k, f in zip(nums, flips))


@pytest.mark.parametrize("bad", ["1,2x", "r", "1,,r2", "0r", "4r", [1, 2.5], [True]])
def test_parse_order_flips_errors(bad):
    with pytest.raises(ValueError):
        F.parse_order_flips(bad, 3)


def test_flip_warning():
    assert F.flip_warning("1,2,3,2") is None and F.flip_warning(None) is None and F.flip_warning([1, 2]) is None
    w = F.flip_warning("1,2r,3,2r")
    assert "2r" in w and "反対側に移ります" in w and "裏返したコマを" in w


def test_flipped_frame_is_mirrored_around_pivot(plain):
    a, b, fx = plain.flip_layout()
    w, h = plain.size
    assert a - b == w - int(round(2 * plain.pivot[0]))
    anim = F.assemble_frames(plain, "1,2,3,2r", name="spin")
    assert anim.size == (w + a + b, h)
    px = anim.info["pivot_x"]
    assert px == plain.pivot[0] + a and px == anim.size[0] / 2   # 基準の左右で同じ幅
    # r のコマは、2 コマ目（左右に足した列の中）を基準の線で左右に裏返したもの（キャラは横にずれない）
    assert np.array_equal(anim.frames[3], anim.frames[1][:, ::-1])
    assert np.array_equal(anim.frames[1][:, a:a + w], plain.frames[1])
    info = F.frames_sheet_info(anim, "x.png")
    assert [f.get("flipped", False) for f in info["frames"]] == [False, False, False, True]
    assert info["padding"] == {"left": a, "right": b, "top": 0, "bottom": 0}
    assert info["pivot"]["x"] == px


def test_whole_mirror_with_r_flips_everything(plain):
    anim = F.assemble_frames(plain, "1,2r")
    mir = F.assemble_frames(plain, "1,2r", mirror=True)
    for f, g in zip(anim.frames, mir.frames):
        assert np.array_equal(f[:, ::-1], g)
    info = F.frames_sheet_info(mir, "x.png")
    a, b, _ = plain.flip_layout()
    assert info["padding"]["left"] == b and info["padding"]["right"] == a
    assert info["source"]["mirrored"] is True and info["frames"][1]["flipped"] is True


def test_no_r_output_is_unchanged(plain):
    # r を使わないときは今までと同じ（padding・flipped を書かない）
    a = F.frames_sheet_json(F.assemble_frames(plain, [1, 2, 3, 2]), "x.png")
    b = F.frames_sheet_json(F.assemble_frames(plain, "1,2,3,2"), "x.png")
    assert a == b and "flipped" not in a and "padding" not in a and "ground_y" not in a


# ---- 裏返したコマだけの直し ----

def _spots(frame):
    ys, xs = np.nonzero(frame[..., 3] > 0)
    tys, txs = np.nonzero(frame[..., 3] == 0)
    return (int(xs[len(xs) // 2]), int(ys[len(ys) // 2])), (int(txs[0]), int(tys[0]))


def _mirror_edits(res, k=2):
    """元の k コマ目に 1 ドット、裏返した k コマ目だけに 1 ドットの直し（版 3）。"""
    f = res.frames[k - 1]
    pal = T.palette_of(res.frames)
    (ox, oy), (tx, ty) = _spots(f)
    color = _hex(f[oy, ox])
    g = f[:, ::-1]
    (mx, my), (mtx, mty) = _spots(g)
    d = {"version": 3, "kind": "sprite_edits", "name": "spin", "frame_width": res.size[0], "frame_height": res.size[1],
         "frame_count": len(res.frames), "palette": pal,
         "frames": [{"source_frame": k, "dots": [[tx, ty, pal.index(color)]]},
                    {"source_frame": k, "mirrored": True, "dots": [[mtx, mty, pal.index(color)], [mx, my, None]]}]}
    return d, (tx, ty), (mtx, mty), (mx, my), color


def test_mirror_only_edits_go_to_flipped_frame_only(plain):
    d, (tx, ty), (mtx, mty), (mx, my), color = _mirror_edits(plain)
    t = T.touchup_from_dict(d)
    assert set(t.frames) == {2} and set(t.mirror_frames) == {2} and t.dot_count == 3
    res = T.touchup_result(plain, t)
    w = plain.size[0]
    # 元のコマには元のコマの直しだけ
    assert _hex(res.frames[1][ty, tx]) == color
    assert res.frames[1][my, w - 1 - mx, 3] > 0 and res.frames[1][mty, w - 1 - mtx, 3] == 0
    # 裏返したコマには、元のコマの直し（裏返して）と、裏返したコマだけの直し
    g = res.flipped_frame(2)
    assert _hex(g[ty, w - 1 - tx]) == color
    assert _hex(g[mty, mtx]) == color and g[my, mx, 3] == 0
    # 裏返していないコマ 1・3 の裏返しには何も足されない
    assert np.array_equal(res.flipped_frame(1), res.frames[0][:, ::-1])
    # シートでは r のコマだけに入る
    anim = F.assemble_frames(res, "1,2,3,2r")
    a, b, fx = res.flip_layout()
    assert np.array_equal(anim.frames[3][:, fx:fx + w], g)
    assert np.array_equal(anim.frames[1][:, a:a + w], res.frames[1])
    rep = res.touchup
    assert [(f.label, len(f.dots)) for f in rep.frames] == [("2", 1), ("2r", 2)]
    assert rep.info()["frames"][1]["mirrored"] is True


def test_mirror_edits_file_roundtrip_version3(plain):
    d, *_ = _mirror_edits(plain)
    t = T.touchup_from_dict(d)
    pal = T.palette_of(plain.frames)
    out = T.touchup_dict(t, pal)
    assert out["version"] == 3
    assert [(f["source_frame"], f.get("mirrored", False)) for f in out["frames"]] == [(2, False), (2, True)]
    text = T.touchup_json(t, pal)
    assert '"mirrored": true' in text
    t2 = T.load_touchup(text)
    assert t2.frames == t.frames and t2.mirror_frames == t.mirror_frames


def test_old_edits_files_still_read(plain):
    # 前の形（版 1・mirrored なし）: 元のコマの直しとして読み、書き出しも版 1 のまま
    f = plain.frames[0]
    (ox, oy), (tx, ty) = _spots(f)
    pal = T.palette_of(plain.frames)
    old = {"version": 1, "kind": "sprite_edits", "name": "walk", "frame_width": plain.size[0],
           "frame_height": plain.size[1], "frame_count": 3, "palette": pal,
           "frames": [{"source_frame": 1, "dots": [[tx, ty, 0, None]]}]}
    t = T.touchup_from_dict(old)
    assert t.mirror_frames == {} and list(t.frames) == [1]
    assert T.touchup_dict(t, pal)["version"] == 1
    assert "mirrored" not in T.touchup_json(t, pal)
    res = T.touchup_result(plain, t)
    assert res.flipped is None
    # 版がなくても読める
    del old["version"]
    assert T.touchup_from_dict(old).frames == t.frames


def test_mirror_edits_errors(plain):
    d, *_ = _mirror_edits(plain)
    d["frames"][1]["mirrored"] = "yes"
    with pytest.raises(ValueError, match="mirrored"):
        T.touchup_from_dict(d)
    d, *_ = _mirror_edits(plain)
    d["frames"][1]["dots"].append([plain.size[0], 0, 0])
    with pytest.raises(T.TouchupError, match="2r"):
        T.touchup_result(plain, T.touchup_from_dict(d))


def test_convert_with_mirror_edits_matches(plain):
    from test_sprite_frames import make_strip
    d, *_ = _mirror_edits(plain)
    t = T.touchup_from_dict(d)
    r = F.convert_frames(make_strip(), SpriteSettings(), F.FrameSettings(frame_count=3, edits=t))
    ref = T.touchup_result(plain, t)
    assert np.array_equal(r.flipped_frame(2), ref.flipped_frame(2))
    js = F.frames_sheet_json(F.assemble_frames(r, "1,2,3,2r"), "x.png")
    assert '"mirrored": true' in js


def test_join_mirror_edits_follow_file_order():
    from retrocore.sprite.join import convert_frame_files
    from test_sprite_frames import make_strip
    a = make_strip(n=1, seed=1)
    b = make_strip(n=2, offsets=((1, 0), (0, 1)), seed=5)
    fs = F.FrameSettings()
    r1 = convert_frame_files([("a.png", a), ("b.png", b)], SpriteSettings(), fs)
    d, *_ = _mirror_edits(r1, k=3)
    t = T.touchup_from_dict(d)
    pal = T.palette_of(r1.frames)
    saved = T.touchup_dict(t, pal, r1.sources, r1.offsets, r1.mirror_offsets())
    assert saved["version"] == 3 and saved["frames"][1]["file"] == "b.png" and "origin" in saved["frames"][1]
    want = T.touchup_result(r1, t).flipped_frame(3)
    # ファイルの順を入れ替えても、裏返したコマの直しは同じ絵（b.png の 2 コマ目）に付く
    r2 = convert_frame_files([("b.png", b), ("a.png", a)], SpriteSettings(), fs)
    got = T.touchup_result(r2, T.touchup_from_dict(saved))
    assert list(got.touchup.frames[1:2])[0].mirrored and got.touchup.frames[1].frame == 2

    def body(img):
        ys, xs = np.nonzero(img[..., 3] > 0)
        return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert np.array_equal(body(got.flipped_frame(2)), body(want))


# ---- 差し替え（まばたき）も裏返して当てる ----

def _eye_swaps(ref):
    from synth_sprite import PIX_EYE
    eye = np.all(ref[..., :3] == np.array(PIX_EYE, np.uint8), axis=-1) & (ref[..., 3] > 0)
    closed = ref.copy()
    for y, x in zip(*np.nonzero(eye)):
        closed[y, x] = ref[y, x - 1]
    return S.SwapSet([S.swap_from_image(ref, {"closed": closed})], ref.shape[1], ref.shape[0], name="spin"), closed


def test_swaps_are_flipped_on_flipped_frames(plain):
    s, closed = _eye_swaps(plain.frames[0])
    anim = F.assemble_frames(plain, "1,1r", durations=[400])
    sw = s.find("blink")
    from retrocore.sprite.steady import outline_keys
    lines = outline_keys(plain.frames[0])
    origins = S._origins(anim.info, 2)
    p0 = S._place_frame(anim.frames[0], 0, plain.frames[0], sw, origins[0], s.search, lines)
    p1 = S._place_frame(anim.frames[1], 1, plain.frames[0], sw, origins[1], s.search, lines, flipped=True)
    assert p0.ok and p1.ok and p1.flipped and p1.info()["flipped"] is True
    assert np.array_equal(p1.images["closed"], p0.images["closed"][:, ::-1])
    out, rep = S.apply_swaps(anim, plain.frames[0], s)
    assert rep.applied
    info = F.frames_sheet_info(out, "x.png")
    # 時間で分けたフレームにも、裏返したかどうかが付いていく
    assert len(out.info["flipped"]) == len(out.frames)
    assert [f.get("flipped", False) for f in info["frames"]] == list(out.info["flipped"])


# ---- 読み戻し ----

def test_readback_keeps_flipped_frames_apart(plain):
    from retrocore.sprite.edit import ReadbackOptions, readback
    anim = F.assemble_frames(plain, "1,2,1r", name="spin")
    sheet = np.concatenate(anim.frames, 1)
    js = F.frames_sheet_json(anim, "s.png")
    res = readback(encode_png(sheet), "s.png", js, None,
                   ReadbackOptions(gif=False, mirror=False, piskel=False, same_frames="first"))
    assert res.made and res.report["repeats"] == []
    assert np.array_equal(res.frames[2], anim.frames[2])          # 1r は 1 に合わせない
    assert json.loads(res.files[res.names["json"]])["frames"][2]["flipped"] is True


# ---- CLI ----

def test_cli_order_r_and_shift_y(tmp_path, capsys):
    from PIL import Image

    from apps.sprite.cli import main
    from test_sprite_frames import make_strip
    src = tmp_path / "jump.png"
    Image.fromarray(make_jump_strip()[..., :3]).save(src)
    out = tmp_path / "out"
    rc = main([str(src), "-o", str(out / "jump.png"), "--frames", "--frame-count", "3", "--align-y", "keep",
               "--frame-shift-y=0,-1,0", "--frame-order", "1,2,3,2r", "--frame-name", "jump", "--mirror", "-q"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "反対側に移ります" in err
    js = [p for p in sorted(out.glob("jump_jump_*_4f.json")) if "_mirror_" not in p.name]
    assert len(js) == 1
    info = json.loads(js[0].read_text(encoding="utf-8"))
    assert info["source"]["align_y"] == "keep" and info["source"]["shift_y"] == [0, -1, 0]
    assert "ground_y" in info and info["frames"][3]["flipped"] is True and info["padding"]
    assert (out / js[0].name.replace("jump_jump_", "jump_jump_mirror_")).exists()


@pytest.mark.parametrize("args, msg", [
    (["--frame-order", "1,x"], "1,2,3,2r"),
    (["--frame-order", "0r"], "1 から"),
    (["--frame-count", "3", "--frame-order", "1,4r"], "--frame-count"),
    (["--frame-count", "3", "--frame-shift-y=0,1"], "--frame-shift-y の数"),
    (["--frame-shift-y=0,999"], "--frame-shift-y は"),
])
def test_cli_errors(tmp_path, capsys, args, msg):
    from apps.sprite.cli import main
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.png"), "--frames"] + args)
    assert msg in capsys.readouterr().err


def test_cli_shift_y_needs_frames(tmp_path, capsys):
    from apps.sprite.cli import main
    with pytest.raises(SystemExit):
        main([str(tmp_path / "x.png"), "--frame-shift-y=0,1"])
    assert "--frames と一緒に" in capsys.readouterr().err


# ---- 画面（Pyodide）と CLI のバイト単位の一致 ----

def test_web_matches_cli_bytes(tmp_path):
    import base64

    from PIL import Image

    from apps.sprite.cli import main
    from retrocore.sprite.convert import frames_convert_bytes, frames_edits, frames_sheets
    from test_sprite_frames import make_strip
    strip = make_jump_strip()
    src = tmp_path / "jump.png"
    Image.fromarray(strip[..., :3]).save(src)
    data = src.read_bytes()
    conv = frames_convert_bytes(data, {"frames": {"frame_count": 3, "align_y": "keep", "shift_y": [0, -1, 0]},
                                       "name": "jump"})
    assert conv["ok"] and conv["made"] and conv["shift_y"] == [0, -1, 0]
    assert len(conv["mirror_offsets"]) == 3 and len(conv["flip_layout"]) == 3
    from retrocore.sprite.convert import _FRAMES_CACHE
    res = _FRAMES_CACHE["result"]
    d, *_ = _mirror_edits(res)
    ed = frames_edits({"token": conv["token"], "edits": d})
    assert ed["applied"] and ed["edits"]["version"] == 3
    (tmp_path / "e.json").write_text(ed["edits_json"], encoding="utf-8")
    sh = frames_sheets({"token": conv["token"], "base": "jump", "name": "jump", "order": [1, 2, 3, "2r"],
                        "durations": [150], "mirror": True})
    assert sh["ok"] and "反対側" in sh["flip_warning"]
    out = tmp_path / "cli"
    rc = main([str(src), "-o", str(out / "jump.png"), "--frames", "--frame-count", "3", "--align-y", "keep",
               "--frame-shift-y=0,-1,0", "--frame-order", "1,2,3,2r", "--frame-name", "jump", "--mirror",
               "--edits", str(tmp_path / "e.json"), "-q"])
    assert rc == 0
    for s in sh["sheets"]:
        assert (out / s["name"]).read_bytes() == s["png"]
        assert (out / s["name"]).with_suffix(".json").read_text(encoding="utf-8") == s["json"]
    assert base64  # (画面は base64 で受け取るが、中身は同じバイト)


def test_runs_with_int32_numpy(jump_strip, plain):
    """32 ビットの NumPy のつもりで、keep・上げ下げ・r・裏返したコマの直し・差し替えを通す（Linux・macOS でだけ動く）。"""
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    d, *_ = _mirror_edits(plain)
    s, _ = _eye_swaps(plain.frames[0])

    def run():
        r = F.convert_frames(jump_strip, SpriteSettings(), F.FrameSettings(frame_count=3, align_y="keep", shift_y=(0, -1, 0)))
        t = T.touchup_result(plain, T.touchup_from_dict(d))
        anim = F.assemble_frames(t, "1,2,3,2r", durations=[400])
        out, _ = S.apply_swaps(anim, plain.frames[0], s)
        return r.frames + list(out.frames), F.frames_sheet_json(F.assemble_frames(r, "1,3r"), "x.png")
    want = run()
    with emulate_int32():
        got = run()
    assert all(np.array_equal(a, b) for a, b in zip(want[0], got[0])) and want[1] == got[1]
