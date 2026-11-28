"""縦にも何段も並んだ絵の切り分け（`frames.find_rows()`）。

- 背景だけの行で段に分け、段ごとに背景だけの列でコマに分ける。コマは左から右、上の段から下の段の順
- 迷ったら分けない: 小さなかけら（離れた手・エフェクト・影）だけの段は作らず近い段に入れる。段どうしが縦に重なっていない
  （横 1 列の絵で跳んだコマだけが上にある）ときも分けない。コマの数を入れたときは段に分けない（今までどおり）
- 1 段の絵の結果は前と同じ（段の切り分けを通しても通さなくても同じコマ）
- 画面と CLI の出力はバイト単位で同じ。つなげる・動きをまとめるはファイルごとに段に分ける
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from retrocore.sprite import SpriteSettings
from retrocore.sprite import convert as C
from retrocore.sprite import frames as F
from retrocore.sprite import join as J
from retrocore.sprite.image import save_rgba
from test_sprite_frames import make_strip

ROW_GAP = 60   # 段の間の背景だけの行（画素）


def _mask_image(mask):
    a = np.zeros(mask.shape + (4,), np.uint8)
    a[mask] = (200, 100, 50, 255)
    return a


def make_rows(n_rows: int = 2, per_row: int = 3, seed: int = 0) -> np.ndarray:
    """`make_strip()` の 1 段を縦に `n_rows` 段重ねた絵（段の間は背景だけの行）。段ごとにノイズを変える。"""
    rows = [make_strip(per_row, seed=seed + 100 * r) for r in range(n_rows)]
    W = max(r.shape[1] for r in rows)
    bg = rows[0][0, 0]
    parts = []
    for k, r in enumerate(rows):
        q = np.empty((r.shape[0], W, 4), np.uint8)
        q[:] = bg
        q[:, :r.shape[1]] = r
        parts.append(q)
        if k < len(rows) - 1:
            g = np.empty((ROW_GAP, W, 4), np.uint8)
            g[:] = bg
            parts.append(g)
    return np.concatenate(parts, 0)


@pytest.fixture(scope="module")
def sheet():
    return make_rows()


@pytest.fixture(scope="module")
def result(sheet):
    return F.convert_frames(sheet, SpriteSettings(), F.FrameSettings())


# ---- 切り分け ----

def test_rows_are_split_then_columns_in_reading_order():
    m = np.zeros((200, 200), bool)
    m[10:80, 10:40] = True        # 1 段目 1 コマ目
    m[10:80, 100:180] = True      # 1 段目 2 コマ目（腕を広げた。2 段目の列と重なる）
    m[110:190, 20:60] = True      # 2 段目 1 コマ目（1 段目の 2 つの列にまたがらない位置）
    m[110:190, 120:150] = True    # 2 段目 2 コマ目
    m[112:117, 75:80] = True      # 2 段目の小さな部品 → 2 段目のいちばん近いコマ
    boxes, notes = F.find_frames(_mask_image(m))
    assert [(b.row, b.x0, b.x1, b.y0, b.y1) for b in boxes] == [
        (0, 10, 40, 10, 80), (0, 100, 180, 10, 80), (1, 20, 80, 110, 190), (1, 120, 150, 110, 190)]
    assert boxes[2].parts == [(75, 80)]
    assert "2 段" in notes[0] and any("2 段目、列 75〜79" in n and "3 コマ目" in n for n in notes)
    assert boxes[0].band == (10, 80) and boxes[3].band == (110, 190)
    assert "2 段" in F.frames_text(boxes) and "／2 段目 20〜79" in F.frames_text(boxes)


def test_small_piece_band_goes_to_the_nearest_row():
    """離れた小さなかけら（エフェクト・影）だけの段は作らない。近い段に入れる。"""
    m = np.zeros((220, 120), bool)
    m[40:120, 10:40] = True
    m[40:120, 70:100] = True
    m[5:15, 20:30] = True         # 頭の上のエフェクト（小さい）→ 1 段目
    m[130:136, 10:40] = True      # 足の下の影（低い）→ 1 段目
    m[130:136, 70:100] = True
    assert F.find_rows(m) is None
    boxes, _ = F.find_frames(_mask_image(m))
    assert [(b.x0, b.x1, b.y0, b.y1) for b in boxes] == [(10, 40, 5, 136), (70, 100, 40, 136)]
    assert all(b.band is None and b.row == 0 for b in boxes)
    # 低い段（いちばん高い段の 3 割より低い）だけを足した 2 段は、低い方を近い段に入れる
    m2 = np.zeros((200, 120), bool)
    m2[10:110, 10:40] = True
    m2[140:160, 10:40] = True     # 高さ 20（1 段目の 2 割）。画素数は多くても段にしない
    assert F.find_rows(m2) is None


def test_frames_not_stacked_are_not_split():
    """横 1 列の絵で、跳んだコマだけが上にある（背景だけの行で分かれて見える）ときは、段に分けない。"""
    m = np.zeros((200, 200), bool)
    m[100:190, 10:40] = True
    m[5:90, 80:110] = True        # 跳んだコマ（ほかのコマと列が重ならない）
    m[100:190, 150:180] = True
    assert F.find_rows(m) is None
    boxes, _ = F.find_frames(_mask_image(m))
    assert [b.x0 for b in boxes] == [10, 80, 150]


def test_frame_count_keeps_columns_and_hints_rows():
    m = np.zeros((200, 120), bool)
    for y in (10, 110):
        m[y:y + 80, 10:40] = True
        m[y:y + 80, 70:100] = True
    boxes, _ = F.find_frames(_mask_image(m), frame_count=2)   # 数を入れたら今までどおり（段に分けない）
    assert [(b.y0, b.y1) for b in boxes] == [(10, 190), (10, 190)]
    with pytest.raises(F.FramesError) as e:
        F.find_frames(_mask_image(m), frame_count=4)
    assert "2 段" in str(e.value) and "段に分けると 4 コマ" in str(e.value) and e.value.extra["found"] == 2
    assert len(F.find_frames(_mask_image(m))[0]) == 4
    assert len(F.find_frames(_mask_image(m), rows=False)[0]) == 2


def test_one_row_strip_is_unchanged():
    """1 段の絵は、段の切り分けを通しても通さなくても同じコマ（前と同じ）。"""
    strip = make_strip(3)
    op = strip[..., 3] > 0
    a, na = F.find_frames(F._remove_background(strip, SpriteSettings())[0])
    b, nb = F.find_frames(F._remove_background(strip, SpriteSettings())[0], rows=False)
    assert a == b and na == nb
    assert op.any()


# ---- 変換 ----

def test_sheet_converts_to_one_animation(result):
    assert len(result.frames) == 6 and len({f.shape for f in result.frames}) == 1
    assert [b.row for b in result.boxes] == [0, 0, 0, 1, 1, 1]
    # 同じ形のコマは段が違っても同じ高さ（縦の 2 体が 1 コマに入らない）
    hs = [f.shape[0] for f in result.raw]
    assert max(hs) - min(hs) <= 2 and max(hs) < 40
    # 段の下のコマも、上の段の同じ列のコマとほとんど同じ形
    one = F.convert_frames(make_strip(3), SpriteSettings(), F.FrameSettings())
    assert [f.shape[0] for f in one.raw] == hs[:3]


def test_keep_and_none_use_row_relative_positions(sheet):
    """元の絵の高さを保つ・そろえないは、段ごとの位置（段の最初のコマ・段のいちばん低い足元）から数える。"""
    for align, align_y in (("match", "keep"), ("none", "none")):
        res = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings(align=align, align_y=align_y))
        h = res.size[1]
        assert h < 40, (align, align_y, res.size)   # 2 段目のコマが段の高さの分だけ下に置かれない
        # 同じ形のコマは、段が違っても同じくらいの高さに置く（格子の合い方で 1 ドットまでずれる）
        assert all(abs(a[1] - b[1]) <= 1 for a, b in zip(res.offsets[3:], res.offsets[:3])), res.offsets


def test_cli_and_web_bytes_match(tmp_path, sheet):
    from apps.sprite.cli import main
    src = tmp_path / "spin.png"
    save_rgba(sheet, src)
    out = tmp_path / "cli" / "spin_sprite.png"
    assert main([str(src), "-o", str(out), "-q", "--frames", "--frame-order", "1,2,3,4,5,6,5r",
                 "--frame-name", "spin", "--gif", "--gif-scale", "4"]) == 0
    cli = {p.name: p.read_bytes() for p in out.parent.iterdir()}
    conv = json.loads(C.frames_convert_for_web(C.encode_png(sheet), json.dumps({"frames": {}, "name": "spin"})))
    assert conv["ok"] and conv["made"], conv
    assert conv["frame_count"] == 6 and len(conv["boxes"]) == 6
    web = {"spin_sprite.png": base64.b64decode(conv["png_base64"])}
    sheets = json.loads(C.frames_sheets_for_web(json.dumps({"token": conv["token"], "base": "spin_sprite", "name": "spin",
                                                            "order": "1,2,3,4,5,6,5r"})))
    assert sheets["ok"], sheets
    for s in sheets["sheets"]:
        png = base64.b64decode(s["png_base64"])
        web[s["name"]] = png
        web[s["name"][:-4] + ".json"] = s["json"].encode("utf-8")
        g = json.loads(C.gif_for_web(png, s["json"], json.dumps({"scale": 4})))
        web[g["name"]] = base64.b64decode(g["gif_base64"])
    assert sorted(web) == sorted(cli)
    for k in cli:
        assert web[k] == cli[k], k


def test_join_splits_rows_per_file(sheet):
    """つなげる: ファイルごとに段に分ける（1 ファイル = 1 つの動きは変わらない。コマはファイルの順に続く）。"""
    strip = make_strip(3, seed=7)
    res = J.convert_frame_files([("spin.png", sheet), ("walk.png", strip)], SpriteSettings(), F.FrameSettings())
    assert len(res.frames) == 9
    assert [f.frame_count for f in res.files] == [6, 3]
    assert res.sources[3] == ("spin.png", 4) and res.sources[6] == ("walk.png", 1)
    # コマの数（合計）を入れたときは段に分けない（今までどおり）: 3 + 3
    with pytest.raises(F.FramesError) as e:
        J.convert_frame_files([("spin.png", sheet), ("walk.png", strip)], SpriteSettings(), F.FrameSettings(frame_count=9))
    assert e.value.extra["found"] == 6


def test_anims_split_rows_per_file(sheet):
    from retrocore.sprite import anims as A
    cfg = A.config_from_dict({"anims": [{"name": "spin", "files": ["spin.png"]},
                                          {"name": "walk", "files": ["walk.png"]}]})
    res = A.convert_anims([("spin.png", sheet), ("walk.png", make_strip(3, seed=7))], SpriteSettings(), cfg)
    assert [(a["name"], a["count"]) for a in res.anims] == [("spin", 6), ("walk", 3)]


def test_dot_candidates_give_height_per_frame(sheet):
    """ドットの大きさの候補: 段に分けたので、高さはコマごと（縦の 2 体の「段ごとの高さ」は出ない）。"""
    got = C.dot_candidates_bytes([("a.png", C.encode_png(sheet))], {"kind": "frames", "sizes": [20]})
    c = got["candidates"][0]
    assert c["ok"] and len(c["heights"]) == 6 and "rows" not in c
    # コマの数を入れた（段に分けない）ときは今までどおり、縦の 2 体が 1 コマ → 段ごとの高さ
    got = C.dot_candidates_bytes([("a.png", C.encode_png(sheet))],
                                 {"kind": "frames", "sizes": [20], "frames": {"frame_count": 3}})
    c = got["candidates"][0]
    assert c["ok"] and len(c["heights"]) == 3 and [len(r) for r in c["rows"]] == [2, 2, 2]


def test_edits_from_old_cut_are_not_applied(tmp_path, sheet):
    """前の切り分け（段に分けない 3 コマ）で直した一覧は、コマの大きさが違うので当てない（作らずに理由を出す）。"""
    from apps.sprite.cli import main
    from retrocore.sprite import touchup as T
    old = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings(frame_count=3))
    w, h = old.size
    t = T.Touchup(frames={1: [T.Dot(0, 0, None)]}, frame_width=w, frame_height=h, frame_count=3)
    new = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings())
    with pytest.raises(T.TouchupError) as e:
        T.check_touchup(new.frames, t)
    assert e.value.code == T.CODE_SIZE
    src = tmp_path / "spin.png"
    save_rgba(sheet, src)
    p = tmp_path / "edits.json"
    p.write_text(T.touchup_json(t, T.palette_of(old.frames)), encoding="utf-8")
    assert main([str(src), "-o", str(tmp_path / "o.png"), "-q", "--frames", "--edits", str(p)]) != 0
    assert main([str(src), "-o", str(tmp_path / "o.png"), "-q", "--frames", "--frame-count", "3", "--edits", str(p)]) == 0


def test_rows_run_with_int32_numpy(sheet):
    from int32_numpy import AVAILABLE, emulate_int32
    if not AVAILABLE:
        pytest.skip("intp と int64 を区別できない環境（Windows など）")
    want = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings())
    with emulate_int32():
        got = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings(align_y="keep"))
        got = F.convert_frames(sheet, SpriteSettings(), F.FrameSettings())
    assert all(np.array_equal(a, b) for a, b in zip(want.frames, got.frames))
