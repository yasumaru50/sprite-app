"""手直ししたシートの読み戻し（retrocore/sprite/edit.py）と .piskel の読み書き（retrocore/sprite/piskel.py）。

テストの絵はコードで合成する（synth_sprite のキノコ）。Piskel が書く形（レイヤーが JSON の文字列、layout は列ごと、
hiddenFrames が [""] のこともある）も、Piskel のソースと本物の .piskel に合わせて合成する。
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from retrocore.sprite import edit as E
from retrocore.sprite import frames as F
from retrocore.sprite.motion.base import MotionResult
from retrocore.sprite.motion.sheet import make_sheet
from retrocore.sprite.piskel import PiskelError, fps_for_durations, piskel_text, read_piskel
from synth_sprite import make_pixel_sprite

ORDER = [1, 2, 3, 2]


def _frames3():
    base = make_pixel_sprite()
    out = []
    for k in range(3):
        f = np.zeros((base.shape[0] + 2, base.shape[1] + 2, 4), np.uint8)
        f[1 + (k == 1):1 + (k == 1) + base.shape[0], 1:1 + base.shape[1]] = base
        if k == 2:
            f[0, 0] = f[3, 5]   # 3 コマ目だけ違うドット
            f[0, 0, 3] = 255 if f[3, 5, 3] else 0
        out.append(f)
    return out


def _anim(order=ORDER, durations=(150,), mirror=False, root_y=None):
    src = _frames3()
    frames = [src[k - 1] for k in order]
    ds = list(durations) * (len(order) if len(durations) == 1 else 1)
    info = {"source_frames": list(order), "align": "match", "align_y": "foot", "pivot_x": frames[0].shape[1] / 2}
    if root_y is not None:
        h, w = frames[0].shape[:2]
        top, bottom = max(0, -min(root_y)), max(0, max(root_y))
        sh = []
        for f, dy in zip(frames, root_y):
            c = np.zeros((h + top + bottom, w, 4), np.uint8)
            c[top + dy:top + dy + h] = f
            sh.append(c)
        frames = sh
        info.update({"padding": 0, "padding_y": (top, bottom), "offset_y": list(root_y)})
    return MotionResult(name="walk", frames=frames, durations=ds, info=info)


def _sheet(anim=None):
    anim = anim or _anim()
    name = F.sheet_file_name("hero_sprite", anim)
    return make_sheet(anim.frames), F.frames_sheet_json(anim, name), anim


def _png(rgba) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _grid(frames, cols):
    h, w = frames[0].shape[:2]
    rows = -(-len(frames) // cols)
    img = np.zeros((rows * h, cols * w, 4), np.uint8)
    for i, f in enumerate(frames):
        img[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = f
    return img


def _decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


# ---------------------------------------------------------------- .piskel

def test_fps_from_most_common_duration():
    assert fps_for_durations([150, 150, 150, 150])[0] == 7
    assert fps_for_durations([100])[0] == 10
    assert fps_for_durations([100])[1] == []            # 割り切れれば説明なし
    fps, notes = fps_for_durations([100, 150, 100])
    assert fps == 10 and "いちばん多い 100 ミリ秒" in notes[0]
    assert fps_for_durations([2000])[0] == 1 and fps_for_durations([10])[0] == 24   # Piskel の 1〜24 に収める


def test_piskel_text_has_piskel_shape_and_round_trips():
    anim = _anim()
    text = piskel_text(anim.frames, "日本語_walk", 7)
    assert text.isascii()                               # Piskel は 1 バイトずつ読むので \\uXXXX で書く
    doc = json.loads(text)
    assert doc["modelVersion"] == 2
    p = doc["piskel"]
    h, w = anim.frames[0].shape[:2]
    assert (p["name"], p["fps"], p["width"], p["height"], p["hiddenFrames"]) == ("日本語_walk", 7, w, h, [])
    assert isinstance(p["layers"][0], str)              # レイヤーは JSON の文字列
    layer = json.loads(p["layers"][0])
    assert layer["frameCount"] == 4 and layer["opacity"] == 1
    assert layer["chunks"][0]["layout"] == [[0], [1], [2], [3]]
    png = _decode(base64.b64decode(layer["chunks"][0]["base64PNG"].split(",", 1)[1]))
    assert png.shape == (h, 4 * w, 4) and np.array_equal(png, make_sheet(anim.frames))
    back = read_piskel(text.encode("ascii"))
    assert all(np.array_equal(a, b) for a, b in zip(back.frames, anim.frames))
    assert piskel_text(anim.frames, "日本語_walk", 7) == text   # 毎回同じ


def _piskel_doc(layers, w, h, hidden=None, version=2):
    return json.dumps({"modelVersion": version, "piskel": {
        "name": "x", "description": "", "fps": 12, "height": h, "width": w,
        "layers": [json.dumps(l) for l in layers], "hiddenFrames": [""] if hidden is None else hidden}})


def _layer(frames, name="Layer 1", opacity=1, cols=None):
    """Piskel の形のレイヤー。cols を渡すと chunk を 2 つに分け、2 つ目は縦に並べる（layout が列ごとなのを確かめる）。"""
    url = lambda a: "data:image/png;base64," + base64.b64encode(_png(a)).decode()   # noqa: E731
    if cols is None:
        return {"name": name, "opacity": opacity, "frameCount": len(frames),
                "chunks": [{"layout": [[i] for i in range(len(frames))], "base64PNG": url(np.concatenate(frames, 1))}]}
    a, b = frames[:cols], frames[cols:]
    return {"name": name, "opacity": opacity, "frameCount": len(frames),
            "chunks": [{"layout": [[i] for i in range(cols)], "base64PNG": url(np.concatenate(a, 1))},
                       {"layout": [[cols + i for i in range(len(b))]], "base64PNG": url(np.concatenate(b, 0))}]}


def test_read_piskel_chunks_layers_and_hidden_frames():
    fr = _anim().frames
    h, w = fr[0].shape[:2]
    pk = read_piskel(_piskel_doc([_layer(fr, cols=2)], w, h))
    assert all(np.array_equal(a, b) for a, b in zip(pk.frames, fr))
    assert pk.warnings == [] and pk.hidden_frames == []   # 本物の .piskel の hiddenFrames [""] は無視
    # 見えるレイヤーを下から重ねる。不透明度 0 のレイヤーは入れない
    top = [np.zeros_like(f) for f in fr]
    top[1][0, 0] = (10, 20, 30, 255)
    hidden = [np.full_like(f, 200) for f in fr]
    pk = read_piskel(_piskel_doc([_layer(fr), _layer(top, "上"), _layer(hidden, "消した", 0)], w, h, hidden=[2]))
    assert pk.frames[1][0, 0].tolist() == [10, 20, 30, 255]
    assert np.array_equal(pk.frames[0], fr[0]) and np.array_equal(pk.frames[1][1:], fr[1][1:])
    assert any("見えるレイヤー 2 枚を重ねて" in m and "消した" in m for m in pk.warnings)
    assert pk.hidden_frames == [3] and any("飛ばす" in m for m in pk.warnings)
    # 不透明度 0.5 は半透明になる
    pk = read_piskel(_piskel_doc([_layer(fr, opacity=0.5)], w, h))
    a = pk.frames[0][..., 3]
    assert set(np.unique(a).tolist()) <= {0, 128} and any("不透明度" in m for m in pk.warnings)


def test_read_piskel_old_layer_without_chunks():
    fr = _anim().frames
    h, w = fr[0].shape[:2]
    layer = {"name": "L", "opacity": 1, "frameCount": 4,
             "base64PNG": "data:image/png;base64," + base64.b64encode(_png(np.concatenate(fr, 1))).decode()}
    pk = read_piskel(_piskel_doc([layer], w, h, version=1))
    assert all(np.array_equal(a, b) for a, b in zip(pk.frames, fr))


@pytest.mark.parametrize("doc, word", [
    (b"not json", "読めません"),
    (json.dumps({"modelVersion": 0, "piskel": {}}), "古い形"),
    (json.dumps({"modelVersion": 2, "piskel": {"width": 4, "height": 4, "layers": []}}), "レイヤーがありません"),
    (_piskel_doc([{"name": "L", "opacity": 1, "frameCount": 2,
                   "chunks": [{"layout": [[0], [1]], "base64PNG": "data:image/png;base64,"
                               + base64.b64encode(_png(np.zeros((4, 4, 4), np.uint8))).decode()}]}], 4, 4), "合いません"),
])
def test_read_piskel_errors(doc, word):
    with pytest.raises(PiskelError, match=word):
        read_piskel(doc)


# ---------------------------------------------------------------- 読み戻し

def test_frames_json_has_palette():
    sheet, text, _ = _sheet()
    info = json.loads(text)
    assert info["palette"] == E.palette_hex([sheet])
    assert list(info)[-2:] == ["palette", "skipped"]


def test_readback_unchanged_sheet_keeps_everything():
    sheet, text, anim = _sheet(_anim(durations=(150, 100, 150, 100), root_y=[0, -1, 0, -1]))
    res = E.readback(_png(sheet), "walk.png", text)
    assert res.made, res.errors
    assert res.warnings == []
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, anim.frames))
    src, new = json.loads(text), res.info
    for k in ("frame_width", "frame_height", "pivot", "padding", "total_duration_ms", "source", "palette", "skipped"):
        assert new[k] == src[k], k
    assert [f["duration_ms"] for f in new["frames"]] == [150, 100, 150, 100]
    assert [f["source_frame"] for f in new["frames"]] == ORDER
    assert [f["offset_y"] for f in new["frames"]] == [0, -1, 0, -1]
    w, h = new["frame_width"], new["frame_height"]
    assert new["image"] == f"hero_sprite_walk_{w}x{h}_4f_edited.png"
    assert new["edited"]["from"] == "walk.png" and new["edited"]["read_as"] == "png"
    names = res.names
    assert names["mirror"] == f"hero_sprite_walk_mirror_{w}x{h}_4f_edited.png"
    assert set(res.files) == {names[k] for k in ("sheet", "json", "mirror", "mirror_json", "gif", "mirror_gif",
                                                 "piskel", "mirror_piskel", "check", "check_json")}
    assert np.array_equal(_decode(res.files[names["sheet"]]), sheet)


def test_readback_mirror_flips_pivot_padding_and_name():
    sheet, text, anim = _sheet(_anim(root_y=[0, -1, 0, -1]))
    info = json.loads(text)
    info["pivot"]["x"] = 10.0
    info["padding"] = {"left": 1, "right": 3, "top": 1, "bottom": 0}
    res = E.readback(_png(sheet), "walk.png", json.dumps(info))
    m = json.loads(res.files[res.names["mirror_json"]])
    w = info["frame_width"]
    assert m["pivot"]["x"] == w - 10.0 and m["padding"]["left"] == 3 and m["padding"]["right"] == 1
    assert m["motion"] == "walk_mirror" and m["source"]["mirrored"] is True
    mimg = _decode(res.files[res.names["mirror"]])
    assert np.array_equal(mimg[:, :w], anim.frames[0][:, ::-1])
    # 反転のシートを読み戻すと、反転の反転は元の名前
    res2 = E.readback(res.files[res.names["mirror"]], "m.png", res.files[res.names["mirror_json"]])
    assert res2.names["mirror"] == f"hero_sprite_walk_{w}x{info['frame_height']}_4f_edited.png"


@pytest.mark.parametrize("cols, scale", [(2, 1), (3, 1), (1, 1), (4, 2), (3, 3)])
def test_readback_reads_grid_and_scaled_sheets(cols, scale):
    sheet, text, anim = _sheet()
    img = _grid(anim.frames, cols)
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)
    res = E.readback(_png(img), "e.png", text)
    assert res.made, res.errors
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, anim.frames))
    lay = res.report["layout"]
    assert (lay["columns"], lay["scale"]) == (cols, scale)
    assert np.array_equal(_decode(res.files[res.names["sheet"]]), sheet)   # 横 1 列に並べ直す
    if cols != 4:
        assert any("列 ×" in m for m in res.notes)


@pytest.mark.parametrize("make, code", [
    (lambda fr: np.concatenate(fr, 1)[:, :-1], "size_mismatch"),                   # キャンバスの幅を変えた
    (lambda fr: np.concatenate(fr[:3], 1), "frame_count"),                         # コマを消した
    (lambda fr: np.concatenate(fr + [fr[0]], 1), "frame_count"),                   # コマを足した
    (lambda fr: np.concatenate(fr[:3] + [np.zeros_like(fr[0])], 1), "empty_frame"),   # 透明だけのコマ
])
def test_readback_not_made(make, code):
    sheet, text, anim = _sheet()
    res = E.readback(_png(make(anim.frames)), "e.png", text)
    assert not res.made and res.files == {}
    assert res.report["code"] == code and res.errors


def test_readback_bad_json_and_options():
    sheet, text, _ = _sheet()
    for bad, word in ((b"{", "読めません"), (json.dumps({"made": False}), "作らなかった"),
                      (json.dumps({"tracks": []}), "動きの表")):
        res = E.readback(_png(sheet), "e.png", bad)
        assert not res.made and word in res.errors[0]
    res = E.readback(_png(sheet), "e.png", text, options=E.ReadbackOptions(same_frames="last"))
    assert not res.made
    res = E.readback(b"GIF89a", "e.png", text)
    assert not res.made and res.report["code"] == "bad_image"


def test_readback_warns_new_colors_semi_and_repeats_but_keeps_dots():
    sheet, text, anim = _sheet()
    fr = [f.copy() for f in anim.frames]
    ys, xs = np.nonzero(fr[0][..., 3])
    fr[0][ys[0], xs[0]] = (1, 2, 3, 255)          # 新しい色
    fr[2][ys[5], xs[5], 3] = 100                  # 半透明
    fr[3][ys[9], xs[9]] = fr[3][ys[10], xs[10]] if not np.array_equal(fr[3][ys[9], xs[9]], fr[3][ys[10], xs[10]]) \
        else (0, 0, 0, 0)                          # 4 コマ目だけ直した（2 コマ目と同じ元のコマ）
    res = E.readback(_png(np.concatenate(fr, 1)), "e.png", text)
    assert res.made
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, fr))      # 直したドットはそのまま
    r = res.report
    assert r["palette_source"] == "json"
    assert [c["color"] for c in r["new_colors"]] == ["#010203"] and r["new_colors"][0]["pixels"] == [[1, xs[0], ys[0]]]
    assert r["semi_transparent"]["dots"] == 1 and r["semi_transparent"]["pixels"] == [[3, xs[5], ys[5]]]
    assert r["repeats"] == [{"source_frame": 2, "frames": [2, 4], "dots": 1, "pixels": [[xs[9], ys[9]]]}]
    text_all = "\n".join(res.warnings)
    assert "#010203" in text_all and "半透明" in text_all and "2 コマ目と 4 コマ目" in text_all
    chk = _decode(res.files[res.names["check"]])
    assert (chk[..., :3] == E.MARK_NEW[:3]).all(-1).any() and (chk[..., :3] == E.MARK_SEMI[:3]).all(-1).any()
    assert (chk[..., :3] == E.MARK_REPEAT[:3]).all(-1).any()
    # 選んだときだけ直す
    res2 = E.readback(_png(np.concatenate(fr, 1)), "e.png", text,
                      options=E.ReadbackOptions(fix_colors=True, same_frames="first"))
    assert res2.made and res2.report["new_colors"] == [] and res2.report["semi_transparent"]["dots"] == 0
    assert res2.report["repeats"] == [] and np.array_equal(res2.frames[3], res2.frames[1])
    assert res2.info["edited"]["fixed"] == {"colors": 1, "semi_transparent": 1}
    assert set(res2.info["palette"]) <= set(json.loads(text)["palette"])


def test_readback_old_json_uses_original_sheet_for_colors_and_changes():
    sheet, text, anim = _sheet()
    info = json.loads(text)
    del info["palette"]                            # 前の版の JSON（palette が無い）
    fr = [f.copy() for f in anim.frames]
    fr[1][2, 3] = (9, 9, 9, 255)
    res = E.readback(_png(np.concatenate(fr, 1)), "e.png", json.dumps(info))
    assert res.made and res.report["palette_source"] is None and any("調べていません" in m for m in res.notes)
    res = E.readback(_png(np.concatenate(fr, 1)), "e.png", json.dumps(info), original=_png(sheet))
    assert res.report["palette_source"] == "original" and res.report["new_colors"][0]["color"] == "#090909"
    assert [f["changed_dots"] for f in res.report["frames"]] == [0, 1, 0, 0]
    assert "changes" in res.names and res.names["changes"] in res.files
    assert res.info["edited"]["changed_dots"] == [0, 1, 0, 0]


def test_readback_from_piskel_and_gif_times():
    sheet, text, anim = _sheet(_anim(durations=(150, 100, 150, 100)))
    pk = piskel_text(anim.frames, "x", 3)
    a = E.readback(pk.encode(), "直した.piskel", text)
    b = E.readback(_png(sheet), "直した.png", text)
    assert a.made and a.report["layout"]["kind"] == "piskel"
    for k in ("sheet", "gif", "mirror", "mirror_gif", "piskel", "check"):
        assert a.files[a.names[k]] == b.files[b.names[k]], k
    g = Image.open(io.BytesIO(a.files[a.names["gif"]]))
    times = []
    for i in range(g.n_frames):
        g.seek(i)
        times.append(g.info["duration"])
    assert times == [150, 100, 150, 100]
    # .piskel のコマの数・大きさが違えば作らない
    bad = piskel_text(anim.frames[:3], "x", 3)
    assert E.readback(bad.encode(), "e.piskel", text).report["code"] == "frame_count"
    bad = piskel_text([f[:-1] for f in anim.frames], "x", 3)
    assert E.readback(bad.encode(), "e.piskel", text).report["code"] == "size_mismatch"


def test_readback_motion_sheet_without_source():
    """1 枚の絵の動き（呼吸など）のシートも読める（名前は <元の名前>_edited、反転は _mirror_edited）。"""
    from retrocore.sprite.motion import make_motion, sheet_json
    res = make_motion(make_pixel_sprite(), "breathe")
    text = sheet_json(res, "hero_breathe_sheet.png")
    rb = E.readback(_png(make_sheet(res.frames)), "e.png", text)
    assert rb.made and rb.names["sheet"] == "hero_breathe_sheet_edited.png"
    assert rb.names["mirror"] == "hero_breathe_sheet_mirror_edited.png"


# ---------------------------------------------------------------- CLI と画面

def _cli_readback(tmp_path, edited_bytes, edited_name, info_text, extra=(), original=None):
    from apps.sprite.cli import main
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    (d / edited_name).write_bytes(edited_bytes)
    (d / "info.json").write_text(info_text, encoding="utf-8")
    argv = [str(d / edited_name), "--readback", str(d / "info.json"), "-o", str(tmp_path / "out"), "-q"] + list(extra)
    if original is not None:
        (d / "orig.png").write_bytes(original)
        argv += ["--original", str(d / "orig.png")]
    return main(argv)


@pytest.mark.parametrize("opts, argv", [
    ({}, ["--gif", "--mirror", "--piskel"]),
    ({"gif_scale": 8, "gif_background": "#203040", "same_frames": "first", "fix_colors": True},
     ["--gif", "--gif-scale", "8", "--gif-background", "#203040", "--mirror", "--piskel", "--same-frames", "first",
      "--fix-colors"]),
])
def test_web_readback_matches_cli_bytes(tmp_path, opts, argv):
    from retrocore.sprite.convert import readback_for_web
    sheet, text, anim = _sheet()
    fr = [f.copy() for f in anim.frames]
    fr[3][3, 4] = (1, 2, 3, 255)
    edited = _png(_grid(fr, 2))
    assert _cli_readback(tmp_path, edited, "直した.png", text, argv, original=_png(sheet)) == 0
    cli = {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()}
    web = json.loads(readback_for_web(edited, "直した.png", text, _png(sheet), json.dumps(opts)))
    assert web["ok"] and web["made"]
    files = {f["name"]: base64.b64decode(f["base64"]) for f in web["files"]}
    assert sorted(files) == sorted(cli)
    for k in cli:
        assert files[k] == cli[k], k


def test_cli_readback_not_made_and_option_errors(tmp_path, capsys):
    from apps.sprite.cli import main
    sheet, text, anim = _sheet()
    assert _cli_readback(tmp_path, _png(np.concatenate(anim.frames[:3], 1)), "e.png", text) == 1
    assert "作り直しませんでした" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    for argv in (["--colors", "8"], ["--frames"], ["--motion", "breathe"], ["--gif-scale", "2"]):
        with pytest.raises(SystemExit):
            main([str(tmp_path / "in" / "e.png"), "--readback", str(tmp_path / "in" / "info.json")] + argv)
    for argv in (["--fix-colors"], ["--same-frames", "first"], ["--piskel"]):
        with pytest.raises(SystemExit):
            main([str(tmp_path / "in" / "e.png")] + argv)


def test_web_readback_errors():
    from retrocore.sprite.convert import readback_for_web
    sheet, text, _ = _sheet()
    r = json.loads(readback_for_web(_png(sheet), "e.png", text, None, json.dumps({"colors": 3})))
    assert not r["ok"]
    r = json.loads(readback_for_web(_png(sheet[:, :-2]), "e.png", text, None, "{}"))
    assert r["ok"] and not r["made"] and r["errors"] and r["files"] == []


def test_frames_piskel_cli_matches_web(tmp_path):
    """--frames --piskel の .piskel は、画面のシートの "piskel" と同じ。"""
    from apps.sprite.cli import main
    from retrocore.sprite import save_rgba
    from retrocore.sprite.convert import encode_png, frames_convert_for_web, frames_sheets_for_web
    from test_sprite_frames import make_strip
    strip = make_strip()
    src = tmp_path / "walk.png"
    save_rgba(strip, src)
    out = tmp_path / "cli" / "hero_sprite.png"
    assert main([str(src), "-o", str(out), "-q", "--frames", "--frame-count", "3", "--frame-order", "1,2,3,2",
                 "--mirror", "--piskel"]) == 0
    conv = json.loads(frames_convert_for_web(encode_png(strip), json.dumps({"frames": {"frame_count": 3}})))
    sheets = json.loads(frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "hero_sprite",
                                                          "order": [1, 2, 3, 2], "durations": [150], "mirror": True})))
    for s in sheets["sheets"]:
        assert (out.parent / s["piskel_name"]).read_text(encoding="ascii") == s["piskel"]
        pk = read_piskel(s["piskel"])
        png = _decode(base64.b64decode(s["png_base64"]))
        assert np.array_equal(np.concatenate(pk.frames, 1), png) and pk.fps == 7
        # 書き出した .piskel を読み戻すと、元のシートと同じ
        rb = E.readback(s["piskel"].encode(), s["piskel_name"], s["json"])
        assert rb.made and rb.warnings == [] and np.array_equal(_decode(rb.files[rb.names["sheet"]]), png)


def test_readback_runs_with_int32_numpy():
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    sheet, text, anim = _sheet()
    fr = [f.copy() for f in anim.frames]
    fr[0][2, 3] = (1, 2, 3, 128)
    img = np.repeat(np.repeat(_grid(fr, 3), 2, 0), 2, 1)
    want = E.readback(_png(img), "e.png", text, _png(sheet), E.ReadbackOptions(fix_colors=True, same_frames="first"))
    with emulate_int32():
        got = E.readback(_png(img), "e.png", text, _png(sheet), E.ReadbackOptions(fix_colors=True, same_frames="first"))
        pk = read_piskel(piskel_text(anim.frames, "x", 7))
    assert got.made and got.files == want.files and len(pk.frames) == 4
