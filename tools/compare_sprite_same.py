"""Sprite App の「1 コマ目と同じにする範囲」（--frames --same-as-first）の比較画像を作る。

    python tools/compare_sprite_same.py 歩き.png --frame-count 4 --range 0,0,53,53 -o out/compare/same
    python tools/compare_sprite_same.py 正面.png --frame-count 3 --frame-order 1,2,3,2 --range 0,0,66,56 -o out/compare/same
    python tools/compare_sprite_same.py 走り.png --frame-count 4 --align-y match --root-y=0,-1,0,-1 --duration 100 \\
        --range 0,0,62,54 -o out/compare/same

書くもの（<名前> は入力のファイル名。変換は範囲のほかは同じ設定。範囲は何回でも --range で足せる）:
- <名前>_range_x8.png: 1 コマ目に範囲の線を引いたもの（8 倍）
- <名前>_off_on_x8.gif: 左 = 範囲なし、右 = 範囲あり の 8 倍の GIF を横に並べたもの
- <名前>_lineup_x8.png: フレームの並び（8 倍）。上の段 = 範囲なし、下の段 = 範囲あり。下の段に、写した位置の枠
  （緑 = 写した、赤 = 写さなかった）と、見つけたずれ（+横,+縦）・合い方を書く
- <名前>_seam_x8.png: 境目の図。範囲ありの各コマに、1 コマ目のドットを写した所の境目（紫の線）と、境目の問題の所（橙の四角。
  写さなかったコマだけにある）、残した部品（青）
- <名前>_face_x16.png: 顔のあたり（--face で指定。省くと範囲の下の方）を 16 倍にし、上の段 = 範囲なし、下の段 = 範囲ありを全コマ並べたもの
- <名前>_counts.json: コマごとのちらつきのドット数（範囲なし → 範囲あり → 範囲あり＋--steady on。上から半分の行と、コマ全体）、
  見つけたずれ・合い方・写した／写さなかった
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrocore.sprite import SpriteSettings, load_rgba, save_rgba, scale_nearest  # noqa: E402
from retrocore.sprite import frames as F  # noqa: E402
from retrocore.sprite import same as Sm  # noqa: E402
from retrocore.sprite.motion.gif import make_gif  # noqa: E402

BG = (236, 236, 236)


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]


def _flat(frame: np.ndarray) -> np.ndarray:
    a = frame[..., 3:4].astype(np.float64) / 255
    return (frame[..., :3] * a + np.array(BG) * (1 - a)).astype(np.uint8)


def side_by_side_gif(a, b, durations, scale: int) -> bytes:
    frames = []
    for fa, fb in zip(a, b):
        h, w = fa.shape[:2]
        c = np.zeros((h, 2 * w + 2, 4), np.uint8)
        c[:, :w] = fa
        c[:, w + 2:] = fb
        c[:, w:w + 2] = (170, 170, 170, 255)
        frames.append(c)
    return make_gif(frames, durations, scale, BG).data


def range_image(frame: np.ndarray, ranges, scale: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    im = Image.fromarray(scale_nearest(_flat(frame), scale))
    d = ImageDraw.Draw(im)
    for r in ranges:
        d.rectangle([r.x * scale, r.y * scale, (r.x + r.w) * scale - 1, (r.y + r.h) * scale - 1],
                    outline=(255, 30, 30), width=max(2, scale // 3))
    return np.asarray(im)


def lineup(off: list[np.ndarray], on: list[np.ndarray], rep, scale: int = 8) -> np.ndarray:
    """上の段 = 範囲なし、下の段 = 範囲あり（写した位置の枠と、ずれ・合い方）。"""
    h, w = off[0].shape[:2]
    gap = 1
    n = len(off)
    out = np.full(((h + gap) * 2 + gap, (w + gap) * n + gap, 3), 170, np.uint8)
    for j, row in enumerate((off, on)):
        for i, f in enumerate(row):
            y, x = gap + j * (h + gap), gap + i * (w + gap)
            out[y:y + h, x:x + w] = _flat(f)
    im = Image.fromarray(scale_nearest(out, scale))
    d = ImageDraw.Draw(im)
    by = {f.frame: f for f in rep.frames}
    y0 = (gap + h + gap) * scale
    for i in range(n):
        x0 = (gap + i * (w + gap)) * scale
        sf = by.get(i + 1)
        if sf is None:
            for r in rep.ranges:
                d.rectangle([x0 + r.x * scale, y0 + r.y * scale, x0 + (r.x + r.w) * scale - 1, y0 + (r.y + r.h) * scale - 1],
                            outline=(0, 170, 255), width=2)
            F._draw_label(d, x0 + 2, y0 + 2, "#1 (ref)", scale=2)
            continue
        for k, rr in enumerate(sf.ranges):
            r = rep.ranges[k]
            tx, ty = r.x + rr.shift[0], r.y + rr.shift[1]
            col = (0, 190, 60) if rr.copied else (255, 30, 30)
            d.rectangle([x0 + tx * scale, y0 + ty * scale, x0 + (tx + r.w) * scale - 1, y0 + (ty + r.h) * scale - 1],
                        outline=col, width=2)
            # ずれの向きを、範囲の左上から写した位置の左上への線で示す（ずれが無ければ点）
            ax, ay = x0 + r.x * scale + 6, y0 + r.y * scale + 6
            bx, by_ = x0 + tx * scale + 6, y0 + ty * scale + 6
            d.line([ax, ay, bx, by_], fill=(255, 0, 200), width=2)
            d.ellipse([bx - 3, by_ - 3, bx + 3, by_ + 3], fill=(255, 0, 200))
            F._draw_label(d, x0 + 2, y0 + 2 + 12 * k,
                          f"#{i + 1} {rr.shift[0]:+d},{rr.shift[1]:+d} {int(round(rr.match * 100))}%" + ("" if rr.copied else " x"),
                          scale=2, fill=(0, 0, 0, 255))
    return np.asarray(im)


def seam_image(on: list[np.ndarray], rep, scale: int = 8) -> np.ndarray:
    """範囲ありの各コマに、写した所の境目（紫の線）・境目の問題（橙）・残した部品（青）。"""
    h, w = on[0].shape[:2]
    gap = scale
    n = len(on)
    W = n * w * scale + (n + 1) * gap
    im = Image.new("RGB", (W, h * scale + 2 * gap + 12), (200, 200, 200))
    d = ImageDraw.Draw(im)
    by = {f.frame: f for f in rep.frames}
    for i, f in enumerate(on):
        x0, y0 = gap + i * (w * scale + gap), gap + 12
        im.paste(Image.fromarray(scale_nearest(_flat(f), scale)), (x0, y0))
        sf = by.get(i + 1)
        if sf is None:
            F._draw_label(d, x0, 1, f"#{i + 1} (ref)")
            continue
        n_bad = 0
        for rr in sf.ranges:
            m = rr.pasted_mask
            if m is not None:
                # 写した所と写していない所の境目に線
                for y in range(h):
                    for x in range(w):
                        if not m[y, x]:
                            continue
                        X, Y = x0 + x * scale, y0 + y * scale
                        if y == 0 or not m[y - 1, x]:
                            d.line([X, Y, X + scale - 1, Y], fill=(150, 0, 220), width=2)
                        if y == h - 1 or not m[y + 1, x]:
                            d.line([X, Y + scale - 1, X + scale - 1, Y + scale - 1], fill=(150, 0, 220), width=2)
                        if x == 0 or not m[y, x - 1]:
                            d.line([X, Y, X, Y + scale - 1], fill=(150, 0, 220), width=2)
                        if x == w - 1 or not m[y, x + 1]:
                            d.line([X + scale - 1, Y, X + scale - 1, Y + scale - 1], fill=(150, 0, 220), width=2)
            if rr.kept_mask is not None and rr.copied:
                for y, x in zip(*np.nonzero(rr.kept_mask)):
                    d.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                                outline=(40, 90, 255), width=2)
            for x, y in list(rr.seam) + list(rr.holes):
                n_bad += 1
                d.rectangle([x0 + x * scale - 2, y0 + y * scale - 2, x0 + (x + 1) * scale + 1, y0 + (y + 1) * scale + 1],
                            outline=(255, 120, 0), width=3)
        F._draw_label(d, x0, 1, f"#{i + 1} seam {n_bad}" + ("" if all(r.copied for r in sf.ranges) else " x"))
    return np.asarray(im)


def face_image(off, on, box, scale: int = 16) -> np.ndarray:
    x0, y0, x1, y1 = box
    gap = 1
    n = len(off)
    h, w = y1 - y0, x1 - x0
    out = np.full(((h + gap) * 2 + gap, (w + gap) * n + gap, 3), 170, np.uint8)
    for j, row in enumerate((off, on)):
        for i, f in enumerate(row):
            y, x = gap + j * (h + gap), gap + i * (w + gap)
            out[y:y + h, x:x + w] = _flat(f[y0:y1, x0:x1])
    return scale_nearest(out, scale)


def counts(frames, rows=None) -> list[int]:
    return [f.count for f in F.flicker_report(frames, rows).frames[1:]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("--frame-count", type=int, default=None)
    p.add_argument("--align-y", default="foot")
    p.add_argument("--frame-order", type=_ints, default=None)
    p.add_argument("--root-y", type=_ints, default=None)
    p.add_argument("--duration", type=int, default=150)
    p.add_argument("--range", dest="ranges", action="append", type=Sm.parse_rect, required=True, metavar="X,Y,W,H")
    p.add_argument("--search", type=int, default=Sm.DEFAULT_SEARCH)
    p.add_argument("--face", type=_ints, default=None, metavar="X0,Y0,X1,Y1", help="顔の拡大の範囲（終わりは含まない）")
    p.add_argument("-o", "--out", type=Path, default=Path("out/compare/same"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.input.stem
    rgba = load_rgba(a.input)
    s = SpriteSettings()
    fs0 = F.FrameSettings(frame_count=a.frame_count, align_y=a.align_y)
    off = F.convert_frames(rgba, s, fs0)
    on = F.refinish_frames(off, F.FrameSettings(frame_count=a.frame_count, align_y=a.align_y,
                                                same_ranges=tuple(a.ranges), same_search=a.search))
    on_st = F.refinish_frames(off, F.FrameSettings(frame_count=a.frame_count, align_y=a.align_y, steady="on",
                                                   same_ranges=tuple(a.ranges), same_search=a.search))
    rep = on.same
    save_rgba(range_image(off.frames[0], rep.ranges), a.out / f"{name}_range_x8.png")
    order = a.frame_order or list(range(1, len(off.frames) + 1))
    anims = [F.assemble_frames(r, order, [a.duration], a.root_y) for r in (off, on)]
    (a.out / f"{name}_off_on_x8.gif").write_bytes(side_by_side_gif(anims[0].frames, anims[1].frames, anims[0].durations, 8))
    save_rgba(lineup(off.frames, on.frames, rep), a.out / f"{name}_lineup_x8.png")
    save_rgba(seam_image(on.frames, rep), a.out / f"{name}_seam_x8.png")
    if a.face:
        box = tuple(a.face)
    else:
        r = rep.ranges[0]
        box = (r.x, max(r.y, r.y + r.h - 24), r.x + r.w, min(off.frames[0].shape[0], r.y + r.h + 2))
    save_rgba(face_image(off.frames, on.frames, box), a.out / f"{name}_face_x16.png")
    h = off.frames[0].shape[0]
    info = {
        "input": a.input.name, "ranges": [r.info() for r in rep.ranges], "search": rep.search,
        "flicker_rows": F.flicker_report(off.frames).rows,
        "flicker_top_half": {"none": counts(off.frames), "range": counts(on.frames), "range_steady": counts(on_st.frames)},
        "flicker_whole_frame": {"none": counts(off.frames, h), "range": counts(on.frames, h),
                                "range_steady": counts(on_st.frames, h)},
        "steady_restored": [f.restored_count for f in on_st.steady.frames],
        "same_as_first": rep.info(),
        "warnings": rep.warnings,
    }
    (a.out / f"{name}_counts.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("flicker_top_half", "flicker_whole_frame", "steady_restored")}, ensure_ascii=False))
    for f in rep.frames:
        for rr in f.ranges:
            print(f"  {f.frame} コマ目 範囲 {rr.range_index}: ずれ {rr.shift} 合い方 {rr.match:.3f} "
                  + ("写した" if rr.copied else f"写さなかった（{rr.code}）") + f" 変えた {rr.changed} 残した {rr.kept}")
    for w in rep.warnings:
        print("  警告:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
