"""Sprite App の「ちらつきを減らす」（--frames --steady on）の比較画像を作る。

    python tools/compare_sprite_steady.py 歩き.png --frame-count 4 -o out/compare/steady
    python tools/compare_sprite_steady.py 正面.png --frame-count 3 --frame-order 1,2,3,2 -o out/compare/steady
    python tools/compare_sprite_steady.py 走り.png --frame-count 4 --align-y match --root-y=0,-1,0,-1 --duration 100 -o out/compare/steady

書くもの（<名前> は入力のファイル名。変換は --steady のほかは同じ設定）:
- <名前>_off_on_x8.gif: 左 = off（今まで）、右 = on の 8 倍の GIF を横に並べたもの（--frame-order・--root-y・--duration のとおり）
- <名前>_lineup_x8.png: フレームの並び（8 倍）。上の段 = off、下の段 = on
- <名前>_diff_x8.png: 違いの図。直した後のコマに、赤 = 1 コマ目に戻したドット、青 = 動きとして残したドット（水色の線から上が戻す範囲）
- <名前>_face_x16.png: 頭のあたり（戻す範囲とその 8 行下までの不透明な所）だけを切り出して 16 倍にし、上の段 = off、下の段 = on を全コマ並べたもの
- <名前>_counts.json: コマごとのちらつきのドット数（off → on）と、戻した・残したドットの数（理由ごと）。--size で複数の強さも数える
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrocore.sprite import SpriteSettings, load_rgba, save_rgba, scale_nearest  # noqa: E402
from retrocore.sprite import frames as F  # noqa: E402
from retrocore.sprite import steady as St  # noqa: E402
from retrocore.sprite.motion.gif import make_gif  # noqa: E402

BG = (236, 236, 236)


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",")]


def _flat(frame: np.ndarray) -> np.ndarray:
    a = frame[..., 3:4].astype(np.float64) / 255
    return (frame[..., :3] * a + np.array(BG) * (1 - a)).astype(np.uint8)


def lineup(rows: list[list[np.ndarray]], scale: int, gap: int = 1) -> np.ndarray:
    """段ごとのコマを並べた画像（RGB。コマの間・段の間は灰色）。"""
    h, w = rows[0][0].shape[:2]
    n = max(len(r) for r in rows)
    out = np.full(((h + gap) * len(rows) + gap, (w + gap) * n + gap, 3), 170, np.uint8)
    for j, r in enumerate(rows):
        for i, f in enumerate(r):
            y, x = gap + j * (h + gap), gap + i * (w + gap)
            out[y:y + h, x:x + w] = _flat(f)
    return scale_nearest(out, scale)


def face_box(frames: list[np.ndarray], rows: int) -> tuple[int, int, int, int]:
    """全部のコマの、戻す範囲（上から rows 行）の不透明な所を囲む四角 (x0, y0, x1, y1)。"""
    m = np.zeros(frames[0].shape[:2], bool)
    for f in frames:
        m |= f[..., 3] > 0
    m[rows:] = False
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--out", type=Path, default=ROOT / "out" / "compare" / "steady")
    p.add_argument("--frame-count", type=int, default=None)
    p.add_argument("--align-y", choices=list(F.ALIGNS_Y), default=F.DEFAULT_ALIGN_Y)
    p.add_argument("--frame-grid", choices=list(F.GRID_MODES), default=F.DEFAULT_GRID_MODE)
    p.add_argument("--frame-order", type=_ints, default=None)
    p.add_argument("--root-y", type=_ints, default=None)
    p.add_argument("--duration", type=int, default=F.DEFAULT_DURATION)
    p.add_argument("--outline", default="auto", help="縁取り（auto / thin など）")
    p.add_argument("--steady-size", type=int, default=St.DEFAULT_SIZE, help="画像に使う強さ")
    p.add_argument("--sizes", type=_ints, default=[2, 4, 8], help="一覧表で数える強さ")
    p.add_argument("--also-outline", default="thin", help="一覧表で参考に数える縁取り（空なら数えない）")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.input.stem
    rgba = load_rgba(args.input)

    def run(outline, steady="off", size=St.DEFAULT_SIZE):
        fs = F.FrameSettings(frame_count=args.frame_count, align_y=args.align_y, grid=args.frame_grid,
                             steady=steady, steady_size=size)
        return F.convert_frames(rgba, SpriteSettings(outline=outline), fs)

    off = run(args.outline)
    on = run(args.outline, "on", args.steady_size)
    order = args.frame_order or list(range(1, len(off.frames) + 1))
    a_off = F.assemble_frames(off, order, [args.duration], args.root_y)
    a_on = F.assemble_frames(on, order, [args.duration], args.root_y)
    (args.out / f"{name}_off_on_x8.gif").write_bytes(side_by_side_gif(a_off.frames, a_on.frames, a_off.durations, 8))
    save_rgba(lineup([a_off.frames, a_on.frames], 8), args.out / f"{name}_lineup_x8.png")
    save_rgba(St.steady_image(on.frames, on.steady, 8), args.out / f"{name}_diff_x8.png")
    # 顔が戻す範囲の境目にかかることがあるので、境目の 8 行下まで切り出す
    x0, y0, x1, y1 = face_box(off.frames, min(on.steady.rows + 8, off.frames[0].shape[0]))
    crop = lambda fs: [f[y0:y1, x0:x1] for f in fs]   # noqa: E731
    save_rgba(lineup([crop(off.frames), crop(on.frames)], 16), args.out / f"{name}_face_x16.png")

    def flicker(res):
        return [f.count for f in F.flicker_report(res.frames).frames[1:]]

    counts = {"input": args.input.name, "outline": args.outline, "rows": on.steady.rows,
              "off": flicker(off), "on": {}}
    for size in args.sizes:
        r = on if size == args.steady_size else run(args.outline, "on", size)
        counts["on"][str(size)] = {"flicker": flicker(r), "steady": r.steady.info()["frames"]}
    if args.also_outline:
        o2 = run(args.also_outline)
        r2 = run(args.also_outline, "on", args.steady_size)
        counts["also"] = {"outline": args.also_outline, "off": flicker(o2), "on": flicker(r2),
                          "size": args.steady_size}
    (args.out / f"{name}_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: ちらつき off {counts['off']} → on（{args.steady_size}）{counts['on'][str(args.steady_size)]['flicker']}"
          + (f" ／ {args.also_outline}: {counts['also']['off']} → {counts['also']['on']}" if args.also_outline else ""))
    for size, v in counts["on"].items():
        print(f"  強さ {size}: {v['flicker']}（戻した " + "・".join(str(f["restored_dots"]) for f in v["steady"]) + "）")
    print(f"  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
