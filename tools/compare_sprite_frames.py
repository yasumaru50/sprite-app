"""Sprite App のコマの切り分け・そろえ方（--frames）の比較画像を作る。

    python tools/compare_sprite_frames.py 歩き.png --frame-count 4 -o out/compare/frames
    python tools/compare_sprite_frames.py 正面.png --frame-order 1,2,3,2 -o out/compare/frames
    python tools/compare_sprite_frames.py 走り.png --align-y match --root-y=0,-1,0,-1 --duration 100 --mirror -o out/compare/frames

書くもの（<名前> は入力のファイル名）:
- <名前>_boxes.png: 元の絵に、見つけたコマの枠（赤）と、コマごとに合わせたグリッド線（水色）を重ねた画像
- <名前>_before_x8.gif / <名前>_after_x8.gif: 変換後の 8 倍の GIF。そろえる前（横・縦とも none = 元の絵の位置のまま）と、そろえた後
- <名前>_lineup.png: そろえたコマの並び（8 倍）。足元の線（赤）と、横をそろえた基準の線（緑）入り
- <名前>_flicker.png / .json: ちらつきの見える化（`frames.flicker_image()`）と、コマごとの違うドットの数
- <名前>_root_x8.gif: --root-y を付けたとき、体全体の上下を重ねた GIF
- <名前>_mirror_sheet.png: --mirror を付けたとき、左右反転のシート（1 倍。8 倍は _mirror_sheet_x8.png）
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
from retrocore.sprite.motion.gif import make_gif  # noqa: E402
from retrocore.sprite.motion.sheet import make_sheet  # noqa: E402

SCALE = 8


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]


def boxes_image(src: np.ndarray, res: F.FramesResult) -> np.ndarray:
    """元の絵にコマの枠とグリッド線を重ねる。"""
    im = Image.fromarray(src[..., :3].copy(), "RGB").convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for b, g in zip(res.boxes, res.grids):
        x0, x1 = int(g.xs[0]), int(g.xs[-1])
        y0, y1 = int(g.ys[0]), int(g.ys[-1])
        for x in g.xs:
            d.line([int(x), y0, int(x), y1], fill=(0, 200, 255, 110))
        for y in g.ys:
            d.line([x0, int(y), x1, int(y)], fill=(0, 200, 255, 110))
        d.rectangle([b.x0, b.y0, b.x1 - 1, b.y1 - 1], outline=(255, 30, 30, 255), width=3)
        for s, e in b.parts:
            d.rectangle([s, b.y0, e - 1, b.y1 - 1], outline=(255, 160, 0, 255), width=2)
    for i, b in enumerate(res.boxes):
        d.text((b.x0 + 4, max(0, b.y0 - 14)), f"#{i + 1}", fill=(255, 30, 30, 255))
    return np.asarray(Image.alpha_composite(im, over))


def lineup_image(res: F.FramesResult, scale: int = SCALE) -> np.ndarray:
    """そろえたコマを並べ、足元の線（赤）と横の基準の線（緑）を引く。背景は市松模様。"""
    w, h = res.size
    gap = scale
    W = len(res.frames) * (w * scale + gap) + gap
    H = h * scale + 2 * gap + 12
    canvas = Image.new("RGBA", (W, H), (200, 200, 200, 255))
    d = ImageDraw.Draw(canvas)
    yy, xx = np.mgrid[0:h, 0:w]
    for i, f in enumerate(res.frames):
        x0, y0 = gap + i * (w * scale + gap), gap + 12
        bg = np.full((h, w, 3), 240, np.uint8)
        bg[(yy + xx) % 2 == 1] = 225
        a = (f[..., 3:4] > 0)
        img = np.where(a, f[..., :3], bg)
        canvas.paste(Image.fromarray(img, "RGB").resize((w * scale, h * scale), Image.NEAREST), (x0, y0))
        foot = int(np.flatnonzero((f[..., 3] > 0).any(1))[-1]) + 1
        d.line([x0, y0 + foot * scale - 1, x0 + w * scale - 1, y0 + foot * scale - 1], fill=(255, 0, 0, 255), width=2)
        ax = x0 + int(round(res.pivot[0] * scale))
        d.line([ax, y0, ax, y0 + h * scale - 1], fill=(0, 170, 0, 255), width=2)
        d.text((x0, 2), f"#{i + 1} offset {res.offsets[i][0]},{res.offsets[i][1]}", fill=(0, 0, 0, 255))
    return np.asarray(canvas)


def write_gif(frames, durations, path: Path) -> None:
    gif = make_gif(frames, durations, SCALE)
    path.write_bytes(gif.data)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--out", type=Path, default=ROOT / "out" / "compare" / "frames")
    p.add_argument("--frame-count", type=int)
    p.add_argument("--frame-order", type=_ints)
    p.add_argument("--align", default=F.DEFAULT_ALIGN, choices=list(F.ALIGNS))
    p.add_argument("--align-y", default=F.DEFAULT_ALIGN_Y, choices=list(F.ALIGNS_Y))
    p.add_argument("--duration", type=int, default=F.DEFAULT_DURATION)
    p.add_argument("--root-y", type=_ints)
    p.add_argument("--mirror", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    src = load_rgba(args.input)
    fs = F.FrameSettings(frame_count=args.frame_count, align=args.align, align_y=args.align_y)
    res = F.convert_frames(src, SpriteSettings(), fs)
    before = F.convert_frames(src, SpriteSettings(), F.FrameSettings(frame_count=args.frame_count,
                                                                      align="none", align_y="none"))
    order = args.frame_order or list(range(1, len(res.frames) + 1))
    save_rgba(boxes_image(src, res), args.out / f"{stem}_boxes.png")
    for tag, r in (("before", before), ("after", res)):
        anim = F.assemble_frames(r, order, [args.duration])
        write_gif(anim.frames, anim.durations, args.out / f"{stem}_{tag}_x{SCALE}.gif")
    save_rgba(lineup_image(res), args.out / f"{stem}_lineup.png")
    rep = F.flicker_report(res.frames)
    save_rgba(F.flicker_image(res.frames, rep), args.out / f"{stem}_flicker.png")
    (args.out / f"{stem}_flicker.json").write_text(F.flicker_json(rep), encoding="utf-8")
    if args.root_y:
        anim = F.assemble_frames(res, order, [args.duration], args.root_y)
        write_gif(anim.frames, anim.durations, args.out / f"{stem}_root_x{SCALE}.gif")
    if args.mirror:
        anim = F.assemble_frames(res, order, [args.duration], args.root_y, mirror=True)
        sheet = make_sheet(anim.frames)
        save_rgba(sheet, args.out / f"{stem}_mirror_sheet.png")
        save_rgba(scale_nearest(sheet, SCALE), args.out / f"{stem}_mirror_sheet_x{SCALE}.png")
    summary = {"input": args.input.name, "frames": len(res.frames), "dot_size": round(res.dot_size[0], 3),
               "frame_size": list(res.size), "align": res.align, "align_y": res.align_y,
               "offsets": [list(o) for o in res.offsets], "offsets_before": [list(o) for o in before.offsets],
               "order": order, "flicker": [f.count for f in rep.frames],
               "flicker_before": [f.count for f in F.flicker_report(before.frames).frames],
               "notes": res.notes, "steps": [s.description for s in res.steps]}
    (args.out / f"{stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
