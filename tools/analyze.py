"""Build style profiles (JSON) from reference CG.

    python tools/analyze.py                         # every samples/reference/<group>/
    python tools/analyze.py path/to/dir --name mygroup
    python tools/analyze.py a.png b.png --name test

Only exact references are used by default: at most 16 colours, every channel
on the PC-98 16-level grid (e.g. emulator BMP saves converted to PNG).  Other
images are skipped with the reason, and listed under ``excluded`` in the
profile; ``--allow-screenshots`` analyses them with screenshot recovery
(grid detection + palette reconstruction) instead.  ``--exclude NAME`` skips
files explicitly.

For each group ``out/profiles/<group>.json`` is written, plus one debug sheet
per image (``out/profiles/<group>/<image>.png``): the recovered native image,
the dither classes overlaid (red = checker, green = 2×2 tile, blue = stripe,
yellow = 4×4 tile, cyan = ordered mix, magenta = irregular) and the outline
pixels.
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

from retrocore.analyzer import (  # noqa: E402
    analyze_native,
    build_profile,
    check_reference,
    classify_dither,
    profile_summary_text,
    save_profile,
)
from retrocore.color import LUMA  # noqa: E402
from retrocore.screenshot import normalize_screenshot  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
CLASS_COLORS = np.array([[0, 0, 0], [255, 40, 40], [40, 230, 40], [60, 60, 255],
                         [255, 230, 0], [0, 230, 230], [255, 0, 255]], np.uint8)


def debug_sheet(native, scale: int = 2) -> Image.Image:
    rgb = native.rgb()
    lum = native.palette.astype(np.float32) @ LUMA
    cls = classify_dither(native.indices, lum)
    over = rgb.astype(np.float32) * 0.35 + CLASS_COLORS[cls] * 0.65
    over = np.where((cls > 0)[..., None], over, rgb * 0.6).astype(np.uint8)
    h, w = native.indices.shape
    sheet = Image.new("RGB", (2 * w * scale + 8, h * scale + 40), (40, 40, 48))
    sheet.paste(Image.fromarray(rgb).resize((w * scale, h * scale), Image.Resampling.NEAREST), (0, 40))
    sheet.paste(Image.fromarray(over).resize((w * scale, h * scale), Image.Resampling.NEAREST),
                (w * scale + 8, 40))
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(native.palette):
        d.rectangle([i * 18, 4, i * 18 + 16, 20], fill=tuple(int(v) for v in c))
    d.text((w * scale + 8, 10), "checker / tile2x2 / stripe / tile4x4 / ordered / irregular",
           fill=(220, 220, 220))
    return sheet


def collect(paths: list[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not paths:
        base = ROOT / "samples" / "reference"
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if files:
                groups[d.name] = files
        return groups
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            files.append(p)
    groups["profile"] = files
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="画像ファイルまたはフォルダ (既定: samples/reference/*/)")
    ap.add_argument("--name", help="プロファイル名 (paths 指定時)")
    ap.add_argument("--allow-screenshots", action="store_true",
                    help="16色・16段階でない画像も、拡大された画像として格子を推定して使う")
    ap.add_argument("--exclude", action="append", default=[], help="除外するファイル名 (複数可)")
    ap.add_argument("--max-colors", type=int, default=64,
                    help="格子を推定するときのパレットの上限色数")
    ap.add_argument("--no-debug", action="store_true", help="デバッグ画像を出力しない")
    ap.add_argument("--out", default=str(ROOT / "out" / "profiles"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for group, files in collect(args.paths).items():
        name = args.name or group
        stats, excluded = [], {}
        for f in files:
            t0 = time.perf_counter()
            if f.name in args.exclude or f.stem in args.exclude:
                excluded[f.name] = "--exclude で指定"
                print(f"  {name}: {f.name}  除外 (--exclude)")
                continue
            img = Image.open(f)
            img.load()
            reason = check_reference(img)
            if reason and not args.allow_screenshots:
                excluded[f.name] = reason
                print(f"  {name}: {f.name}  除外 ({reason})")
                continue
            native = normalize_screenshot(img, max_colors=args.max_colors)
            stats.append(analyze_native(native, f.name, img.size))
            print(f"  {name}: {f.name}  {img.width}×{img.height} -> "
                  f"{native.indices.shape[1]}×{native.indices.shape[0]}, "
                  f"{len(native.palette)}色  {time.perf_counter() - t0:.2f}s")
            if not args.no_debug:
                (out / name).mkdir(exist_ok=True)
                debug_sheet(native).save(out / name / f"{f.stem}.png")
        if not stats:
            print(f"  {name}: 使える画像がありません\n")
            continue
        prof = build_profile(stats, name, excluded)
        save_profile(prof, out / f"{name}.json")
        print(profile_summary_text(prof))
        print(f"  -> {out / (name + '.json')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
