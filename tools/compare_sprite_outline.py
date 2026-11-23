"""Sprite App の縁取りの確認用の比較画像を作る。

    python tools/compare_sprite_outline.py 入力.png [入力 ...] [-o 出力フォルダ] [--input-type pixel|illust]
                                           [--height 48] [--colors 12] [--cleanup normal] [--zoom 8]

入力ごとに、CLI（`python -m apps.sprite`）と同じ工程で縁取りの前まで（背景除去 → グリッド検出または縮小 → 単純化 →
クリーンアップ）を行い、縁取りを 3 通り（auto / add・darken / add・single）かけて、出力フォルダ（既定 `out/sprite/outline/`）に
次の画像を保存する。

- `<名前>_outline.png`: **比較画像**。上の段は「縁取り前 ｜ auto ｜ add（darken）｜ add（single）」、
  下の段は、それぞれの下に「縁取りで変わったドットに印を付けた画像」（結果を薄くして、変わったドットだけを目立つ色で塗る）。
  どれも最近傍で `--zoom` 倍（既定 8）、透明の部分は市松模様。余白を足した結果と位置がそろうように、
  余白を足していない画像も同じ大きさの枠の中に置く
- `<名前>_<before|auto|add_darken|add_single>.png`: それぞれの結果（1 ドット＝1 画素、透過 PNG）と、その拡大 `_x<倍率>.png`

印の色: 外側に付けた輪郭 = マゼンタ、切れ目を埋めた = シアン。標準出力に、判定の結果・変わったドットの数・増えた色の数を表示する。

確認に使う AI 画像（ai01〜ai03）は preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    mkdir -p out/ai && git show origin/preview/ai-check:ai-input/ai03.png > out/ai/ai03.png
    python tools/compare_sprite_outline.py out/ai/ai03.png --input-type pixel --colors 12
    python tools/compare_sprite_outline.py out/ai/ai01.png out/ai/ai02.png --input-type illust --height 48 --colors 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apps.sprite import cli  # noqa: E402
from retrocore.sprite import GridNotFoundError, load_rgba, outline_sprite, save_rgba, scale_nearest  # noqa: E402
from retrocore.sprite.grid import count_colors  # noqa: E402
from retrocore.sprite.outline import KIND_ADDED, KIND_GAP  # noqa: E402

DEFAULT_OUT = ROOT / "out" / "sprite" / "outline"
GAP = 16
LABEL_H = 16
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
CHECK = ((200, 200, 200), (150, 150, 150))
MARK = {KIND_ADDED: (255, 0, 255), KIND_GAP: (0, 230, 255)}
MARK_NAMES = {KIND_ADDED: "added outline", KIND_GAP: "gap filled"}
VARIANTS = (("auto", dict(mode="auto")), ("add_darken", dict(mode="add", color_mode="darken")),
            ("add_single", dict(mode="add", color_mode="single")))


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


def place(rgba: np.ndarray, shape: tuple, offset: tuple) -> np.ndarray:
    """rgba を大きさ shape（高さ, 幅）の透明の枠の offset（上, 左）に置く。"""
    out = np.zeros((shape[0], shape[1], 4), np.uint8)
    t, l = offset
    out[t:t + rgba.shape[0], l:l + rgba.shape[1]] = rgba
    return out


def marked(rgba: np.ndarray, kinds: np.ndarray, changed: np.ndarray, zoom: int) -> np.ndarray:
    """結果を薄く（白に寄せて）表示し、変わったドットだけを理由ごとの色で塗った RGB（zoom 倍）。"""
    base = on_checker(scale_nearest(rgba, zoom), max(2, zoom // 2)).astype(np.float64)
    out = (base * 0.35 + 255 * 0.65).astype(np.uint8)
    for y, x in zip(*np.nonzero(changed)):
        out[y * zoom:(y + 1) * zoom, x * zoom:(x + 1) * zoom] = MARK.get(int(kinds[y, x]), (255, 0, 255))
    return out


def panel(label: str, rgb: np.ndarray) -> Image.Image:
    img = Image.new("RGB", (max(rgb.shape[1], 8 * len(label)), LABEL_H + rgb.shape[0]), SHEET_BG)
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


def before_outline(src: Path, args) -> np.ndarray:
    """CLI と同じ工程で、縁取りの前までを行った結果。"""
    argv = [str(src), "--input-type", args.input_type, "--colors", str(args.colors), "--cleanup", args.cleanup,
            "--no-outline"]
    if args.input_type == "illust":
        argv += ["--height", str(args.height)]
    pa = cli.build_parser().parse_args(argv)
    return cli.build_pipeline(pa).run(load_rgba(src))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="縁取りの比較画像を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--input-type", choices=["pixel", "illust"], default="pixel")
    p.add_argument("--height", type=int, default=48, help="illust のときの縮小後の高さ（既定 48）")
    p.add_argument("--colors", type=int, default=12, help="単純化の色数（既定 12）")
    p.add_argument("--cleanup", default="normal", help="クリーンアップの強さ（既定 normal）")
    p.add_argument("--zoom", type=int, default=8, help="拡大の倍率（最近傍、既定 8）")
    args = p.parse_args(argv)
    z = args.zoom
    cell = max(2, z // 2)

    failed = 0
    for src in args.inputs:
        name = src.stem
        try:
            base = before_outline(src, args)
        except GridNotFoundError as e:
            print(f"{src}: グリッドが見つかりません（{e}）", file=sys.stderr)
            failed += 1
            continue
        h, w = base.shape[:2]
        n0 = count_colors(base)
        print(f"{src}: {w}×{h} ドット、{n0} 色")
        results = [(key, outline_sprite(base, **kw)) for key, kw in VARIANTS]
        # 位置をそろえる枠: 余白をいちばん多く足した結果に合わせる
        pads = [r.padding for _, r in results]
        t, b, l, r_ = (max(v[i] for v in pads) for i in range(4))
        shape = (h + t + b, w + l + r_)
        save_rgba(base, args.output / f"{name}_before.png")
        save_rgba(scale_nearest(base, z), args.output / f"{name}_before_x{z}.png")
        framed = place(base, shape, (t, l))
        top = [panel(f"before: {w}x{h}, {n0} colors", on_checker(scale_nearest(framed, z), cell))]
        bottom = [legend(top[0].width, top[0].height)]
        for key, res in results:
            n1 = count_colors(res.image)
            det = res.detection
            judge = ""
            if det is not None:
                judge = (f"、判定: 輪郭{'あり' if det.has_outline else 'なし'}（暗い縁 {det.dark_edges}/{det.judged_edges}"
                         f" = {det.coverage:.0%}、縁 {det.edges}、輪郭の色 "
                         + " ".join("#%02x%02x%02x" % c for c in det.colors) + "）")
            pad = "・".join(nm for nm, v in zip(("上", "下", "左", "右"), res.padding) if v) or "なし"
            print(f"  {key}: {res.action}、{res.count} ドット変更、余白 {pad}、色 {n0} → {n1}"
                  f"（増えた色 {len(res.new_colors)}: " + " ".join("#%02x%02x%02x" % c for c in res.new_colors)
                  + "、使った輪郭の色: " + " ".join("#%02x%02x%02x" % c for c in res.outline_colors) + "）" + judge)
            save_rgba(res.image, args.output / f"{name}_{key}.png")
            save_rgba(scale_nearest(res.image, z), args.output / f"{name}_{key}_x{z}.png")
            off = (t - res.padding[0], l - res.padding[2])
            img = place(res.image, shape, off)
            ch = place(np.dstack([res.changed.astype(np.uint8)] * 4), shape, off)[..., 0].astype(bool)
            kinds = place(np.dstack([res.kinds.astype(np.uint8)] * 4), shape, off)[..., 0]
            label = f"{key.replace('_', ' ')}: {res.action}, {n1} colors (+{len(res.new_colors)})"
            top.append(panel(label, on_checker(scale_nearest(img, z), cell)))
            bottom.append(panel(f"{key.replace('_', ' ')}: {res.count} dots changed", marked(img, kinds, ch, z)))
        out = args.output / f"{name}_outline.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        grid_sheet([top, bottom]).save(out)
        print(f"  → {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
