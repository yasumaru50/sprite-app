"""Sprite App の背景除去の確認用の比較画像を作る。

    python tools/compare_sprite_background.py 入力.png [入力 ...] [-o 出力フォルダ] [--tolerance T]

入力ごとに次の画像を出力フォルダ（既定 `out/sprite/bg/`）に保存する。

- `<名前>_bg.png`: 元画像 ｜ 背景除去の結果をチェック柄の上に置いたもの ｜ マゼンタの上に置いたもの
  （縦 `--height` 画素に縮小して横に並べる。消し残し・にじみはマゼンタの上で見やすい）
- `<名前>_edge.png`: 輪郭の周りの拡大図。上端・左端・右端・下端の輪郭の 4 か所と、外周につながっていない
  背景に近い色の最大の部分（enclosed。囲まれた背景の消し残し、または背景と同じ色の服など）を切り出し、
  元画像 ｜ マゼンタの上の結果 を `--zoom` 倍（最近傍）で並べる（`--edge` で名前を指定した画像、または `--edge-all`）

標準出力に、画像ごとの背景色・透明の割合・処理時間と、輪郭に接する「背景に近い明るさの残り画素」
（背景色との差が `--halo` 以下の不透明画素で、透明な画素に上下左右で接するもの）の数を表示する。

確認に使う AI 画像（ai01〜ai03）は preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    git show origin/preview/ai-check:ai-input/ai03.png > out/ai03.png
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

from retrocore.sprite import load_rgba  # noqa: E402
from retrocore.sprite.background import (DEFAULT_TOLERANCE, _label, background_mask,  # noqa: E402
                                         estimate_background, parse_color)

DEFAULT_OUT = ROOT / "out" / "sprite" / "bg"
GAP = 8
LABEL_H = 14
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
MAGENTA = (255, 0, 255)
CHECK = ((200, 200, 200), (150, 150, 150))


def checker(h: int, w: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    k = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[~k], out[k] = CHECK
    return out


def composite(rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """2 段階の α（0 / 255）の画像を背景の上に置く。"""
    out = bg.copy()
    op = rgba[..., 3] >= 128
    out[op] = rgba[..., :3][op]
    return out


def resize_rgb(rgb: np.ndarray, height: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == height:
        return rgb
    nw = max(1, round(w * height / h))
    return np.asarray(Image.fromarray(rgb).resize((nw, height), Image.Resampling.LANCZOS))


def zoom(rgb: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, k, 0), k, 1)


def row_sheet(tiles: list[tuple[str, np.ndarray]]) -> Image.Image:
    """ラベル付きのタイルを横に並べる（ラベルは Pillow の既定フォントで描くため ASCII）。"""
    h = max(t.shape[0] for _, t in tiles)
    w = GAP + sum(t.shape[1] + GAP for _, t in tiles)
    sheet = Image.new("RGB", (w, GAP + LABEL_H + h + GAP), SHEET_BG)
    draw = ImageDraw.Draw(sheet)
    x = GAP
    for label, t in tiles:
        draw.text((x, GAP), label, fill=FG)
        sheet.paste(Image.fromarray(t), (x, GAP + LABEL_H))
        x += t.shape[1] + GAP
    return sheet


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    w = max(r.width for r in rows)
    sheet = Image.new("RGB", (w, sum(r.height for r in rows)), SHEET_BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    return sheet


def edge_pixels(mask: np.ndarray) -> np.ndarray:
    """透明な画素に上下左右で接する不透明な画素。"""
    t = mask
    nb = np.zeros_like(t)
    nb[1:] |= t[:-1]
    nb[:-1] |= t[1:]
    nb[:, 1:] |= t[:, :-1]
    nb[:, :-1] |= t[:, 1:]
    return ~t & nb


def edge_points(mask: np.ndarray, near_bg: np.ndarray) -> list[tuple[str, int, int]]:
    """拡大図の中心 (名前, y, x): 輪郭の上端・左端・右端・下端と、残った背景に近い色の最大の部分。

    最後のもの（enclosed）は、外周につながっていない背景に近い色の部分のうち最も大きいものの中で、
    その重心に最も近い画素。囲まれた背景の消し残しか、背景と同じ色の服などが残っているかを見るため。
    """
    ys, xs = np.nonzero(edge_pixels(mask))
    if len(ys) == 0:
        return []
    pts = []
    for name, i in (("top", np.argmin(ys)), ("left", np.argmin(xs)),
                    ("right", np.argmax(xs)), ("bottom", np.argmax(ys))):
        pts.append((name, int(ys[i]), int(xs[i])))
    rest = near_bg & ~mask
    if rest.any():
        lab = _label(rest)
        ids, cnt = np.unique(lab[rest], return_counts=True)
        ys, xs = np.nonzero(lab == ids[np.argmax(cnt)])
        i = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
        pts.append(("enclosed", int(ys[i]), int(xs[i])))
    return pts


def crop(rgb: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    y0 = int(np.clip(y - size // 2, 0, max(0, h - size)))
    x0 = int(np.clip(x - size // 2, 0, max(0, w - size)))
    return rgb[y0:y0 + size, x0:x0 + size]


def process(path: Path, out_dir: Path, args) -> dict:
    rgba = load_rgba(path)
    h, w = rgba.shape[:2]
    color = args.color if args.color is not None else estimate_background(rgba)
    t0 = time.perf_counter()
    mask = background_mask(rgba, color, args.tolerance, remove_enclosed=args.remove_enclosed)
    sec = time.perf_counter() - t0
    res = rgba.copy()
    res[mask] = 0
    res[~mask, 3] = 255
    src = rgba[..., :3]
    on_check = composite(res, checker(h, w, max(8, h // 64)))
    on_mag = composite(res, np.broadcast_to(np.array(MAGENTA, np.uint8), (h, w, 3)).copy())

    edge = edge_pixels(mask)
    diff = np.abs(src.astype(np.int16) - np.array(color, np.int16)).max(-1)
    halo = int((edge & (diff <= args.halo)).sum())
    stats = dict(name=path.stem, size=(w, h), color=color, transparent=float(mask.mean()),
                 seconds=sec, edge=int(edge.sum()), halo=halo)

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = [("original", resize_rgb(src, args.height)),
             (f"removed (tol {args.tolerance:g}) on checker", resize_rgb(on_check, args.height)),
             ("on magenta", resize_rgb(on_mag, args.height))]
    row_sheet(tiles).save(out_dir / f"{path.stem}_bg.png")

    if args.edge_all or path.stem in args.edge:
        rows = []
        for name, y, x in edge_points(mask, diff <= args.tolerance):
            a = zoom(crop(src, y, x, args.crop), args.zoom)
            b = zoom(crop(on_mag, y, x, args.crop), args.zoom)
            rows.append(row_sheet([(f"{name} ({x},{y}) original", a), ("on magenta", b)]))
        if rows:
            stack_rows(rows).save(out_dir / f"{path.stem}_edge.png")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="背景除去の比較画像を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help=f"出力フォルダ（既定 {DEFAULT_OUT}）")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"許容量（既定 {DEFAULT_TOLERANCE}）")
    p.add_argument("--color", type=parse_color, help="背景色（#RRGGBB または R,G,B）。省略時は外周から推定")
    p.add_argument("--remove-enclosed", action="store_true", help="囲まれた背景も消す")
    p.add_argument("--height", type=int, default=600, help="比較画像の 1 枚の高さ（既定 600）")
    p.add_argument("--edge", nargs="*", default=[], metavar="名前",
                   help="輪郭の拡大図を作る画像の名前（拡張子なし。例: ai03）")
    p.add_argument("--edge-all", action="store_true", help="すべての画像で輪郭の拡大図を作る")
    p.add_argument("--crop", type=int, default=96, help="拡大図の切り出しの大きさ（既定 96）")
    p.add_argument("--zoom", type=int, default=4, help="拡大図の倍率（既定 4）")
    p.add_argument("--halo", type=int, default=64,
                   help="にじみの数え方: 輪郭に接する画素のうち背景色との差がこの値以下のもの（既定 64）")
    args = p.parse_args(argv)
    for path in args.inputs:
        s = process(path, args.output, args)
        c = s["color"]
        print(f"{s['name']}: {s['size'][0]}×{s['size'][1]} 背景色 #{c[0]:02x}{c[1]:02x}{c[2]:02x} "
              f"透明 {s['transparent']:.1%} {s['seconds']:.2f} 秒 / 輪郭 {s['edge']} 画素のうち "
              f"背景に近い色（差 ≤ {args.halo}）{s['halo']} 画素")
    return 0


if __name__ == "__main__":
    sys.exit(main())
