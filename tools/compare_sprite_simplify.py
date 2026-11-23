"""Sprite App の単純化（似た色をまとめる）の確認用の比較画像を作る。

    python tools/compare_sprite_simplify.py 入力.png [入力 ...] [-o 出力フォルダ] [--colors 8 12 16 24] [--zoom 8]
                                            [--input-type pixel|illust] [--width W]

入力ごとに、背景除去（既定の設定）→ グリッド検出（`--input-type illust` なら縮小）→ 単純化 を行い、
出力フォルダ（既定 `out/sprite/simplify/`）に次の画像を保存する。

- `<名前>_simplify.png`: 単純化の前 ｜ 色数ごとの結果 を横に並べた比較画像。どれも最近傍で `--zoom` 倍（既定 8）、
  透明の部分は市松模様。それぞれの下に色の見本（パレット。使われた画素の多い順。単純化の前は多い順に 32 色）を並べる
- `<名前>_c<色数>.png`: 色数ごとの結果（1 ドット＝1 画素、透過 PNG）
- `<名前>_c<色数>_x<倍率>.png`: その拡大（最近傍）

標準出力に、色数ごとの残った色（RGB と画素数）・中間色と判定した画素の数・時間を表示する。

確認に使う AI 画像（ai01〜ai03）は preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    mkdir -p out/ai && git show origin/preview/ai-check:ai-input/ai03.png > out/ai/ai03.png
    python tools/compare_sprite_simplify.py out/ai/ai03.png -o out/sprite/simplify
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

from retrocore.sprite import (BackgroundStep, GridNotFoundError, GridStep, ResizeStep, SpritePipeline,  # noqa: E402
                              load_rgba, save_rgba, scale_nearest, simplify_colors)
from retrocore.sprite.grid import count_colors  # noqa: E402

DEFAULT_OUT = ROOT / "out" / "sprite" / "simplify"
DEFAULT_COLORS = (8, 12, 16, 24)
GAP = 16
LABEL_H = 16
SWATCH = 16          # 色の見本 1 つの大きさ（画素）
SWATCH_GAP = 2
BEFORE_SWATCHES = 32
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
CHECK = ((200, 200, 200), (150, 150, 150))


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


def palette_of(rgba: np.ndarray, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """不透明な画素の色と画素数（多い順。同点は RGB の小さい順）。"""
    rgb = rgba[..., :3][rgba[..., 3] >= 128]
    cols, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    order = np.lexsort((cols[:, 2], cols[:, 1], cols[:, 0], -counts))
    cols, counts = cols[order], counts[order]
    if limit is not None:
        cols, counts = cols[:limit], counts[:limit]
    return cols, counts


def swatches(colors: np.ndarray, width: int) -> np.ndarray:
    """色の見本を幅 width に折り返して並べた RGB。"""
    per_row = max(1, (width + SWATCH_GAP) // (SWATCH + SWATCH_GAP))
    rows = max(1, -(-len(colors) // per_row))
    h = rows * (SWATCH + SWATCH_GAP)
    out = np.empty((h, width, 3), np.uint8)
    out[:] = SHEET_BG
    for i, c in enumerate(colors):
        r, q = divmod(i, per_row)
        y, x = r * (SWATCH + SWATCH_GAP), q * (SWATCH + SWATCH_GAP)
        out[y:y + SWATCH, x:x + SWATCH] = c
    return out


def column(label: str, rgba: np.ndarray, colors: np.ndarray, zoom: int, sw_label: str) -> Image.Image:
    big = on_checker(scale_nearest(rgba, zoom), max(2, zoom // 2))
    w = big.shape[1]
    sw = swatches(colors, w)
    img = Image.new("RGB", (w, LABEL_H + big.shape[0] + 6 + LABEL_H + sw.shape[0]), SHEET_BG)
    d = ImageDraw.Draw(img)
    d.text((0, 2), label, fill=FG)
    img.paste(Image.fromarray(big), (0, LABEL_H))
    y = LABEL_H + big.shape[0] + 6
    d.text((0, y + 2), sw_label, fill=FG)
    img.paste(Image.fromarray(sw), (0, y + LABEL_H))
    return img


def hstack(cols: list[Image.Image]) -> Image.Image:
    h = max(c.height for c in cols)
    w = sum(c.width for c in cols) + GAP * (len(cols) + 1)
    sheet = Image.new("RGB", (w, h + 2 * GAP), SHEET_BG)
    x = GAP
    for c in cols:
        sheet.paste(c, (x, GAP))
        x += c.width + GAP
    return sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="単純化（似た色をまとめる）の比較画像を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--colors", type=int, nargs="+", default=list(DEFAULT_COLORS), help="比べる色数（既定 8 12 16 24）")
    p.add_argument("--zoom", type=int, default=8, help="拡大の倍率（最近傍、既定 8）")
    p.add_argument("--input-type", choices=["pixel", "illust"], default="pixel")
    p.add_argument("--width", type=int, default=64, help="--input-type illust のときの縮小後の幅（既定 64）")
    args = p.parse_args(argv)

    failed = 0
    for src in args.inputs:
        name = src.stem
        pre = SpritePipeline([BackgroundStep()])
        pre.add(GridStep() if args.input_type == "pixel"
                else ResizeStep(width=args.width, method="mode", line_priority=True))
        try:
            base = pre.run(load_rgba(src))
        except GridNotFoundError as e:
            print(f"{src}: グリッドが見つかりません（{e}）", file=sys.stderr)
            failed += 1
            continue
        h, w = base.shape[:2]
        n0 = count_colors(base)
        print(f"{src}: {w}×{h} ドット、単純化の前 {n0} 色")
        before_cols, _ = palette_of(base, BEFORE_SWATCHES)
        cols = [column(f"before: {n0} colors", base, before_cols, args.zoom, f"top {len(before_cols)} colors")]
        for k in args.colors:
            t0 = time.perf_counter()
            r = simplify_colors(base, k)
            sec = time.perf_counter() - t0
            print(f"  {k} 色: {len(r.palette)} 色が残った（中間色 {r.transitions} 画素、{sec:.2f} 秒）")
            for c, n in zip(r.palette, r.counts):
                print(f"    #{c[0]:02x}{c[1]:02x}{c[2]:02x} ({c[0]:3d},{c[1]:3d},{c[2]:3d}) {n:4d} 画素")
            save_rgba(r.image, args.output / f"{name}_c{k}.png")
            save_rgba(scale_nearest(r.image, args.zoom), args.output / f"{name}_c{k}_x{args.zoom}.png")
            cols.append(column(f"--colors {k}: {len(r.palette)} colors", r.image, r.palette, args.zoom,
                               "palette"))
        sheet = hstack(cols)
        out = args.output / f"{name}_simplify.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
        print(f"  → {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
