"""Sprite App のグリッド検出の確認用の比較画像を作る。

    python tools/compare_sprite_grid.py 入力.png [入力 ...] [-o 出力フォルダ] [--dot-size W[xH]] [--no-snap]
                                        [--dot-shape square|separate]

入力ごとに、背景除去（既定の設定）→ グリッド検出 を行い、次の画像を出力フォルダ（既定 `out/sprite/grid/`）に保存する。

- `<名前>_grid.png`: 元画像 ｜ 検出したグリッド線を重ねた図 ｜ 結果を 1 倍 ｜ 結果を最近傍で `--zoom` 倍（既定 10）。
  元画像と重ねた図はキャラクターの範囲（切り詰めた範囲）を切り出し、`--zoom` 倍の結果と同じ高さにそろえる
- `<名前>_lines.png`: 輪郭のあたりの拡大図（`--crop` 画素四方を `--scale` 倍、最近傍）。上端・左端・右端・下端の輪郭と、
  キャラクターの中央（顔のあたり）の 5 か所で、グリッド線を重ねた元画像 ｜ 各マスを結果の色で塗った図 を並べる
- `<名前>_shapes.png`: ドットの形の比べ: 元画像 ｜ 縦横別に推定（separate）の結果 ｜ 正方形（square、既定）の結果
  （どちらも最近傍で `--zoom` 倍。3 枚の高さをそろえる。元画像はキャラクターの範囲を切り出す）
- グリッドが見つからない画像は画像を作らず、理由を表示する

標準出力に、画像ごとの推定したドットの大きさ・線の間隔の範囲・周期の目立ち方・境目の揃い方・出力の大きさ・縦横比
（元画像のキャラクターの範囲の縦横比も）・色数・同じ色が続く行と列の組の数（すべて同じ／9 割以上同じ）・時間を、
正方形（square）と縦横別（separate）の両方について表示する。画像は `--dot-shape` の方式（既定 square）で作る。

確認に使う AI 画像（ai01〜ai03）は preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    mkdir -p out/ai && for f in ai01 ai02 ai03; do git show origin/preview/ai-check:ai-input/$f.png > out/ai/$f.png; done
    python tools/compare_sprite_grid.py out/ai/ai01.png out/ai/ai02.png out/ai/ai03.png -o out/sprite/grid
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

from retrocore.sprite import (GridNotFoundError, duplicate_lines, grid_to_pixels, load_rgba,  # noqa: E402
                              remove_background)

DEFAULT_OUT = ROOT / "out" / "sprite" / "grid"
GAP = 8
LABEL_H = 14
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
LINE = (0, 90, 255)
CHECK = ((200, 200, 200), (150, 150, 150))


def checker(h: int, w: int, cell: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    k = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[~k], out[k] = CHECK
    return out


def on_checker(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    out = checker(*rgba.shape[:2], cell)
    op = rgba[..., 3] >= 128
    out[op] = rgba[..., :3][op]
    return out


def zoom(rgb: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, k, 0), k, 1)


def resize_to_height(rgb: np.ndarray, height: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == height:
        return rgb
    nw = max(1, round(w * height / h))
    return np.asarray(Image.fromarray(rgb).resize((nw, height), Image.Resampling.LANCZOS))


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


def draw_lines(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray, x0: int = 0, y0: int = 0,
               scale: int = 1, width: int = 1) -> np.ndarray:
    """拡大済みの画像（入力の (x0, y0) から scale 倍）にグリッド線を描く。線は境目の位置（画素の左・上の端）。"""
    out = rgb.copy()
    h, w = out.shape[:2]
    for x in xs:
        p = (int(x) - x0) * scale
        if 0 <= p < w:
            out[:, max(0, p - width // 2):p - width // 2 + width] = LINE
    for y in ys:
        p = (int(y) - y0) * scale
        if 0 <= p < h:
            out[max(0, p - width // 2):p - width // 2 + width, :] = LINE
    return out


def paint_cells(shape: tuple[int, int], xs: np.ndarray, ys: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """各マスを結果の色で塗った、元画像と同じ大きさの RGBA（グリッドの外は透明）。"""
    out = np.zeros(shape + (4,), np.uint8)
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            out[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = cells[r, c]
    return out


def outline_points(alpha: np.ndarray) -> list[tuple[str, int, int]]:
    """拡大図の中心 (名前, y, x): 輪郭の上端・左端・右端・下端と、キャラクターの中央。"""
    ys, xs = np.nonzero(alpha)
    pts = []
    for name, i in (("top", np.argmin(ys)), ("left", np.argmin(xs)),
                    ("right", np.argmax(xs)), ("bottom", np.argmax(ys))):
        pts.append((name, int(ys[i]), int(xs[i])))
    pts.append(("center", int(ys.mean()), int(xs.mean())))
    return pts


def crop_box(h: int, w: int, y: int, x: int, size: int) -> tuple[int, int]:
    y0 = int(np.clip(y - size // 2, 0, max(0, h - size)))
    x0 = int(np.clip(x - size // 2, 0, max(0, w - size)))
    return y0, x0


def _stats(res, sec: float) -> dict:
    g = res.grid
    xs, ys = res.lines
    return dict(found=True, seconds=sec, period=g.period, cell=g.cell_size,
                gaps=(int(np.diff(xs).min()), int(np.diff(xs).max()), int(np.diff(ys).min()), int(np.diff(ys).max())),
                prominence=g.prominence, alignment=g.alignment, out=res.size, colors=res.colors,
                aspect=res.size[0] / res.size[1], dup=res.duplicates,
                dup90=duplicate_lines(res.image, min_ratio=0.9))


def process(path: Path, out_dir: Path, args) -> dict:
    src = load_rgba(path)
    rgba = remove_background(src)
    op = rgba[..., 3] >= 128
    cols, rows = np.nonzero(op.any(0))[0], np.nonzero(op.any(1))[0]
    src_aspect = (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1) if len(cols) else 0.0
    info = dict(name=path.stem, size=src.shape[1::-1], src_aspect=src_aspect, modes={})
    results = {}
    for shape in ("square", "separate"):
        t0 = time.perf_counter()
        try:
            res = grid_to_pixels(rgba, args.dot_size, snap=not args.no_snap, square=shape == "square")
        except GridNotFoundError as e:
            info["modes"][shape] = dict(found=False, message=str(e), seconds=time.perf_counter() - t0)
            continue
        results[shape] = res
        info["modes"][shape] = _stats(res, time.perf_counter() - t0)
    res = results.get(args.dot_shape)
    if res is None:
        return info
    g = res.grid
    xs, ys = res.lines
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 枚目: 元画像 ｜ 線を重ねた図 ｜ 1 倍 ｜ N 倍（キャラクターの範囲を切り出してそろえる）
    big = zoom(on_checker(res.image, 1), args.zoom)
    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
    orig = src[y0:y1, x0:x1, :3]
    lined = draw_lines(orig, xs, ys, x0, y0, 1, max(1, round((y1 - y0) / big.shape[0])))
    one = on_checker(res.image, 1)
    pad = np.full((big.shape[0], one.shape[1], 3), SHEET_BG, np.uint8)
    pad[:one.shape[0]] = one
    tiles = [("original", resize_to_height(orig, big.shape[0])),
             ("grid lines", resize_to_height(lined, big.shape[0])),
             ("x1", pad),
             (f"x{args.zoom} nearest", big)]
    row_sheet(tiles).save(out_dir / f"{path.stem}_grid.png")

    # 2 枚目: 輪郭のあたりの拡大図
    full_cells = np.zeros((len(g.ys) - 1, len(g.xs) - 1, 4), np.uint8)
    c0, r0, c1, r1 = res.crop
    full_cells[r0:r1, c0:c1] = res.image
    painted = on_checker(paint_cells(src.shape[:2], g.xs, g.ys, full_cells), 8)
    rows_img = []
    h, w = src.shape[:2]
    for name, y, x in outline_points(op):
        cy, cx = crop_box(h, w, y, x, args.crop)
        a = draw_lines(zoom(src[cy:cy + args.crop, cx:cx + args.crop, :3], args.scale), g.xs, g.ys, cx, cy,
                       args.scale)
        b = draw_lines(zoom(painted[cy:cy + args.crop, cx:cx + args.crop], args.scale), g.xs, g.ys, cx, cy,
                       args.scale)
        rows_img.append(row_sheet([(f"{name} ({cx},{cy}) original + grid", a), ("result cells", b)]))
    stack_rows(rows_img).save(out_dir / f"{path.stem}_lines.png")

    # 3 枚目: ドットの形の比べ（元画像 ｜ 縦横別 ｜ 正方形）
    if len(results) == 2:
        bigs = {k: zoom(on_checker(r.image, 1), args.zoom) for k, r in results.items()}
        hh = max(b.shape[0] for b in bigs.values())
        crop = src[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1, :3]
        tiles = [(f"original {crop.shape[1]}x{crop.shape[0]} (aspect {src_aspect:.3f})", resize_to_height(crop, hh))]
        for k, label in (("separate", "before: separate"), ("square", "after: square (default)")):
            r = results[k]
            b = bigs[k]
            padded = np.full((hh, b.shape[1], 3), SHEET_BG, np.uint8)
            padded[:b.shape[0]] = b
            tiles.append((f"{label} {r.size[0]}x{r.size[1]} (aspect {r.size[0] / r.size[1]:.3f})", padded))
        row_sheet(tiles).save(out_dir / f"{path.stem}_shapes.png")
    return info


def _print(info: dict) -> None:
    head = f"{info['name']}: {info['size'][0]}×{info['size'][1]}（キャラクターの縦横比 {info['src_aspect']:.3f}）"
    print(head)
    for shape, s in info["modes"].items():
        if not s["found"]:
            print(f"  {shape}: グリッドなし（{s['seconds']:.2f} 秒）: {s['message']}")
            continue
        print(f"  {shape}: ドット {s['period'][0]:.2f}×{s['period'][1]:.2f}px"
              f" 間隔 横 {s['gaps'][0]}〜{s['gaps'][1]}・縦 {s['gaps'][2]}〜{s['gaps'][3]}px"
              f" 目立ち方 {s['prominence'][0]:.2f}/{s['prominence'][1]:.2f}"
              f" 揃い方 {s['alignment'][0]:.2f}/{s['alignment'][1]:.2f}"
              f" → {s['out'][0]}×{s['out'][1]} ドット（縦横比 {s['aspect']:.3f}） {s['colors']} 色"
              f" 同じ色が続く 行 {s['dup'][0]}・列 {s['dup'][1]}（9 割以上同じ: 行 {s['dup90'][0]}・列 {s['dup90'][1]}）"
              f"（{s['seconds']:.2f} 秒）")


def _dot_size(text: str):
    parts = [float(v) for v in text.lower().split("x")]
    return parts[0] if len(parts) == 1 else tuple(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="グリッド検出の比較画像を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help=f"出力フォルダ（既定 {DEFAULT_OUT}）")
    p.add_argument("--dot-size", type=_dot_size, help="ドットの大きさ（例: 20 または 20x18）。省略時は推定")
    p.add_argument("--no-snap", action="store_true", help="線を境目に合わせず等間隔の格子にする")
    p.add_argument("--dot-shape", choices=["square", "separate"], default="square",
                   help="比較画像を作るときのドットの形（既定 square。標準出力には両方の結果を出す）")
    p.add_argument("--zoom", type=int, default=10, help="結果の拡大倍率（既定 10）")
    p.add_argument("--crop", type=int, default=160, help="拡大図の切り出しの大きさ（既定 160）")
    p.add_argument("--scale", type=int, default=3, help="拡大図の倍率（既定 3）")
    args = p.parse_args(argv)
    for path in args.inputs:
        _print(process(path, args.output, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
