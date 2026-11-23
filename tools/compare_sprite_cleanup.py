"""Sprite App のクリーンアップ（余計なドットを片付ける）の確認用の比較画像を作る。

    python tools/compare_sprite_cleanup.py 入力.png [入力 ...] [-o 出力フォルダ] [--colors 12 16]
                                           [--levels light normal strong] [--zoom 8]

入力ごとに、背景除去（既定の設定）→ グリッド検出 → 単純化（`--colors` の色数ごと）→ クリーンアップ（`--levels` の強さごと）
を行い、出力フォルダ（既定 `out/sprite/cleanup/`）に次の画像を保存する。

- `<名前>_c<色数>_cleanup.png`: **比較画像**。上の段は「クリーンアップ前 ｜ 強さごとの結果」、
  下の段は「強さごとに、変わったドットに印を付けた画像」（結果を薄くして、変わったドットだけを理由ごとの目立つ色で塗る）。
  どれも最近傍で `--zoom` 倍（既定 8）、透明の部分は市松模様
- `<名前>_c<色数>_<強さ>.png`: 強さごとの結果（1 ドット＝1 画素、透過 PNG）と、その拡大 `_x<倍率>.png`

印の色: 孤立ドット = マゼンタ、中間色 = シアン、輪郭の角 = 黄、輪郭の太さ = 緑、透明の点・穴 = 赤
（透明にしたドットは赤い枠）。標準出力に、強さごとに変わったドットの数（理由ごと）と色数を表示する。

確認に使う AI 画像（ai01〜ai03）は preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    mkdir -p out/ai && git show origin/preview/ai-check:ai-input/ai03.png > out/ai/ai03.png
    python tools/compare_sprite_cleanup.py out/ai/ai03.png -o out/sprite/cleanup
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrocore.sprite import (BackgroundStep, GridNotFoundError, GridStep, SpritePipeline,  # noqa: E402
                              cleanup_sprite, load_rgba, save_rgba, scale_nearest, simplify_colors)
from retrocore.sprite.cleanup import (KIND_ALPHA, KIND_BETWEEN, KIND_CORNER, KIND_ISOLATED,  # noqa: E402
                                      KIND_THIN, LEVELS)
from retrocore.sprite.grid import count_colors  # noqa: E402

DEFAULT_OUT = ROOT / "out" / "sprite" / "cleanup"
DEFAULT_COLORS = (12, 16)
DEFAULT_LEVELS = ("light", "normal", "strong")
GAP = 16
LABEL_H = 16
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
CHECK = ((200, 200, 200), (150, 150, 150))
MARK = {KIND_ISOLATED: (255, 0, 255), KIND_BETWEEN: (0, 230, 255), KIND_CORNER: (255, 230, 0),
        KIND_THIN: (0, 230, 0), KIND_ALPHA: (255, 40, 40)}
MARK_NAMES = {KIND_ISOLATED: "isolated", KIND_BETWEEN: "between", KIND_CORNER: "corner",
              KIND_THIN: "thick line", KIND_ALPHA: "alpha"}


def on_checker(rgba: np.ndarray, cell: int) -> np.ndarray:
    """透明の部分を市松模様にした RGB。"""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    k = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[~k], out[k] = CHECK
    op = rgba[..., 3] >= 128
    out[op] = rgba[..., :3][op]
    return out


def marked(rgba: np.ndarray, kinds: np.ndarray, changed: np.ndarray, zoom: int) -> np.ndarray:
    """結果を薄く（白に寄せて）表示し、変わったドットだけを理由ごとの色で塗った RGB（zoom 倍）。"""
    base = on_checker(scale_nearest(rgba, zoom), max(2, zoom // 2)).astype(np.float64)
    out = (base * 0.35 + 255 * 0.65).astype(np.uint8)
    for y, x in zip(*np.nonzero(changed)):
        col = MARK.get(int(kinds[y, x]), (255, 0, 255))
        ys, xs = slice(y * zoom, (y + 1) * zoom), slice(x * zoom, (x + 1) * zoom)
        if rgba[y, x, 3] < 128:  # 透明にしたドットは枠だけ
            b = max(1, zoom // 4)
            out[ys, xs][:b], out[ys, xs][-b:], out[ys, xs][:, :b], out[ys, xs][:, -b:] = col, col, col, col
        else:
            out[ys, xs] = col
    return out


def panel(label: str, rgb: np.ndarray) -> Image.Image:
    img = Image.new("RGB", (rgb.shape[1], LABEL_H + rgb.shape[0]), SHEET_BG)
    ImageDraw.Draw(img).text((0, 2), label, fill=FG)
    img.paste(Image.fromarray(rgb), (0, LABEL_H))
    return img


def legend(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), SHEET_BG)
    d = ImageDraw.Draw(img)
    d.text((0, 2), "marks (changed dots):", fill=FG)
    for i, (k, col) in enumerate(MARK.items()):
        y = LABEL_H + 6 + i * 20
        d.rectangle((0, y, 13, y + 13), fill=col)
        d.text((20, y + 1), MARK_NAMES[k], fill=FG)
    return img


def grid_sheet(rows: list[list[Image.Image]]) -> Image.Image:
    cw = max(c.width for r in rows for c in r)
    rh = [max(c.height for c in r) for r in rows]
    ncol = max(len(r) for r in rows)
    sheet = Image.new("RGB", (GAP + ncol * (cw + GAP), GAP + sum(h + GAP for h in rh)), SHEET_BG)
    y = GAP
    for r, h in zip(rows, rh):
        x = GAP
        for c in r:
            sheet.paste(c, (x, y))
            x += cw + GAP
        y += h + GAP
    return sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="クリーンアップ（余計なドットを片付ける）の比較画像を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--colors", type=int, nargs="+", default=list(DEFAULT_COLORS), help="単純化の色数（既定 12 16）")
    p.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS), choices=[k for k in LEVELS if k != "off"],
                   help="比べる強さ（既定 light normal strong）")
    p.add_argument("--zoom", type=int, default=8, help="拡大の倍率（最近傍、既定 8）")
    args = p.parse_args(argv)

    failed = 0
    for src in args.inputs:
        name = src.stem
        try:
            base = SpritePipeline([BackgroundStep(), GridStep()]).run(load_rgba(src))
        except GridNotFoundError as e:
            print(f"{src}: グリッドが見つかりません（{e}）", file=sys.stderr)
            failed += 1
            continue
        h, w = base.shape[:2]
        print(f"{src}: {w}×{h} ドット")
        for k in args.colors:
            simp = simplify_colors(base, k).image
            n0 = count_colors(simp)
            print(f"  単純化 {k} 色（{n0} 色が残った）")
            top = [panel(f"c{k} before cleanup: {n0} colors", on_checker(scale_nearest(simp, args.zoom),
                                                                         max(2, args.zoom // 2)))]
            bottom = []
            for lv in args.levels:
                t0 = time.perf_counter()
                r = cleanup_sprite(simp, lv)
                sec = time.perf_counter() - t0
                n1 = count_colors(r.image)
                detail = "、".join(f"{kk} {v}" for kk, v in r.counts().items() if v)
                print(f"    {lv}: {r.count} ドットを変更（{detail}）、{n0} 色 → {n1} 色、{sec:.3f} 秒")
                save_rgba(r.image, args.output / f"{name}_c{k}_{lv}.png")
                save_rgba(scale_nearest(r.image, args.zoom), args.output / f"{name}_c{k}_{lv}_x{args.zoom}.png")
                top.append(panel(f"--cleanup {lv}: {n1} colors",
                                 on_checker(scale_nearest(r.image, args.zoom), max(2, args.zoom // 2))))
                bottom.append(panel(f"{lv}: {r.count} dots changed", marked(r.image, r.kinds, r.changed, args.zoom)))
            bottom.insert(0, legend(top[0].width, top[0].height))
            out = args.output / f"{name}_c{k}_cleanup.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            grid_sheet([top, bottom]).save(out)
            print(f"    → {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
