"""Still App の「ざらついた範囲を模様に整える」（--tidy-pattern）の比較画像を作る。

    python tools/compare_still_pattern.py 入力.png --width 600 --out out/still/比較 [--crop 左,上,右,下]

- <out>_overview.png: 整えた後（左）と、整えた範囲を色で重ねたもの（右）を 2 倍で横に並べる
- <out>_crop.png（--crop のとき）: 切り出した所の、整える前（左）・整えた後（右）を 4 倍で横に並べる
  （--crop はドットの座標。どちらもニアレストネイバー）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrocore.still.convert import StillSettings, convert_still, decode_image  # noqa: E402


def _zoom(rgb: np.ndarray, z: int) -> Image.Image:
    return Image.fromarray(rgb).resize((rgb.shape[1] * z, rgb.shape[0] * z), Image.NEAREST)


def _side(a: Image.Image, b: Image.Image, gap: int = 8) -> Image.Image:
    out = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--crop", default=None, help="左,上,右,下（ドット）")
    args = p.parse_args(argv)
    rgba = decode_image(args.input.read_bytes())
    before = convert_still(rgba, StillSettings(width=args.width))
    after = convert_still(rgba, StillSettings(width=args.width, tidy_pattern=True))
    a_rgb = after.rgb()
    ov = after.overlay().astype(np.float64)
    al = ov[..., 3:] / 255.0
    shown = (a_rgb * (1 - al) + ov[..., :3] * al).round().astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _side(_zoom(a_rgb, 2), _zoom(shown, 2)).save(f"{args.out}_overview.png")
    if args.crop:
        x0, y0, x1, y1 = (int(v) for v in args.crop.split(","))
        _side(_zoom(before.rgb()[y0:y1, x0:x1], 4), _zoom(a_rgb[y0:y1, x0:x1], 4)).save(f"{args.out}_crop.png")
    t = after.tidied
    print(f"{args.input.name}: 整えた範囲 {len(t.tidied)} か所、変えたドット {t.changed}、"
          f"3 色目が多くて整えなかった範囲 {len(t.regions) - len(t.tidied)}、小さくて整えなかった範囲 {t.small}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
