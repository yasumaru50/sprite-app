"""Sprite App の縮小方法の比較画像を作る。

    python tools/compare_sprite_downscale.py            # → docs/compare/sprite/downscale.png

入力は `tests/synth_sprite.py` で合成したキャラクター（512×768、AI 画像風に色を揺らしたもの）。
幅 32px・64px に nearest / box / mode / median / mode＋線優先 で縮小し、4 倍（最近傍）に拡大して横に並べる。
上段が 32px、下段が 64px。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from retrocore.sprite import ResizeStep  # noqa: E402
from synth_sprite import add_noise, make_character  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "compare" / "sprite" / "downscale.png"
WIDTHS = (32, 64)
ZOOM = 4
GAP = 8
LABEL_H = 14
BG = (40, 40, 48)
FG = (230, 230, 230)

# (ラベル, ResizeStep の引数)。ラベルは Pillow の既定フォントで描くため ASCII
COLUMNS = [
    ("nearest", dict(method="nearest")),
    ("box", dict(method="box")),
    ("mode", dict(method="mode")),
    ("median", dict(method="median")),
    ("mode+line", dict(method="mode", line_priority=True)),
]


def make_input() -> np.ndarray:
    return add_noise(make_character(512, 768), amp=8, seed=0)


def build_sheet(src: np.ndarray) -> Image.Image:
    rows = []
    for w in WIDTHS:
        tiles = [(label, ResizeStep(width=w, **kw).apply(src)) for label, kw in COLUMNS]
        rows.append((w, tiles))
    col_w = max(t.shape[1] for _, ts in rows for _, t in ts) * ZOOM
    sheet_w = GAP + len(COLUMNS) * (col_w + GAP)
    sheet_h = GAP + sum(LABEL_H + ts[0][1].shape[0] * ZOOM + GAP for _, ts in rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    y = GAP
    for w, tiles in rows:
        for i, (label, tile) in enumerate(tiles):
            x = GAP + i * (col_w + GAP)
            draw.text((x, y), f"{label} {w}px", fill=FG)
            big = Image.fromarray(tile, "RGBA").resize(
                (tile.shape[1] * ZOOM, tile.shape[0] * ZOOM), Image.Resampling.NEAREST)
            sheet.paste(big.convert("RGB"), (x, y + LABEL_H), big)
        y += LABEL_H + tiles[0][1].shape[0] * ZOOM + GAP
    return sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="出力 PNG")
    args = p.parse_args(argv)
    sheet = build_sheet(make_input())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG", optimize=True)
    print(f"→ {args.output} ({sheet.width}×{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
