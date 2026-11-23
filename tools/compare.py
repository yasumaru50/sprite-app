"""Side-by-side comparison sheets for conversion settings.

Converts every input image with several settings and writes one PNG sheet per
input (plus an overview sheet) to ``out/compare/``.  Used to compare the
existing conversion with the style-conversion features added in later phases.

Examples::

    python tools/compare.py                       # samples/input/*, default configs
    python tools/compare.py --max-inputs 2
    python tools/compare.py -c "k16 floyd:dither=floyd" -c "k16 none:dither=none"
    python tools/compare.py --configs-json my_configs.json --inputs a.png b.jpg
    python tools/compare.py --synthetic           # ignore samples, use generated images
    python tools/compare.py --presets             # existing method vs. every style preset
    python tools/compare.py --profile out/profiles/group_a.json --fit 16
                                                  # settings derived from a profile (+ search)
    python tools/compare.py --profile out/profiles/group_a.json --cross-ramp
                                                  # profile without / with cross-ramp dither

A config is ``label:key=value,key=value`` where keys are ``ConvertSettings``
fields; values are parsed as Python literals when possible (``16``, ``0.5``,
``True``), otherwise taken as strings.  ``--configs-json`` takes a JSON object
``{"label": {"field": value, ...}, ...}``.

Each sheet has, per image, a row at native resolution (pixel aspect applied by
line doubling, as on the real machine) and a row of nearest-neighbour enlarged
centre crops so that dither patterns are visible.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrocore.pipeline import ConvertSettings, convert, flatten, load_image  # noqa: E402
from retrocore.preview import PC98_PIXEL_ASPECT, aspect_preview, crt_preview  # noqa: E402
from retrocore.style import STYLE_PRESETS, apply_overrides, profile_to_settings  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# Default comparison set: the existing pipeline in a few typical settings.
# Later phases append the style-conversion configs here.
DEFAULT_CONFIGS: dict[str, dict] = {
    "既定 (k-means16 / Bayer4)": {},
    "k-means16 / ディザなし": {"dither": "none"},
    "k-means16 / Floyd-Steinberg": {"dither": "floyd", "dither_strength": 1.0},
    "スタイル: 市松": {"mode": "style", "style_pattern": "checker"},
    "スタイル: 2×2タイル＋輪郭": {"mode": "style", "style_pattern": "tile2",
                                 "style_outline": True},
    "スタイル: Kuwahara＋Bayer4": {"mode": "style", "style_smooth": "kuwahara",
                                  "style_smooth_strength": 0.4, "style_pattern": "bayer4"},
}

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
]


def _font(size: int = 14):
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def synthetic_images() -> list[tuple[str, Image.Image]]:
    """Generated test pictures: smooth gradients, flat shapes and outlines."""
    h, w = 400, 640
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    # 1) "anime cel" style: sky gradient, flat disc with shading and outline.
    sky = np.stack([90 + 80 * y / h, 140 + 70 * y / h, 230 - 30 * y / h], -1)
    img = sky.copy()
    d = np.hypot(x - 320, y - 220)
    face = d < 140
    shade = np.clip((x - 320 + y - 220) / 200, 0, 1)
    skin = np.stack([255 - 40 * shade, 220 - 60 * shade, 190 - 60 * shade], -1)
    img[face] = skin[face]
    img[(d >= 140) & (d < 144)] = (30, 20, 30)
    hair = (y < 160) & (np.abs(x - 320) < 170) & (d < 175)
    img[hair] = np.stack([200 - 0.5 * y, 60 + 0 * y, 80 + 0 * y], -1)[hair]
    a = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    # 2) Colour ramps: hue across, value down.
    hue = x / w
    val = 1 - y / h
    hsv = np.stack([hue * 255, np.full_like(hue, 200), val * 255], -1).astype(np.uint8)
    b = Image.fromarray(hsv, "HSV").convert("RGB")
    return [("synthetic_cel", a), ("synthetic_ramps", b)]


def collect_inputs(paths: list[str] | None, use_synthetic: bool, limit: int | None):
    if use_synthetic:
        return synthetic_images()
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = sorted(p for p in (ROOT / "samples" / "input").glob("*")
                       if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print("samples/input に画像が無いため合成画像を使います")
        return synthetic_images()
    if limit:
        files = files[:limit]
    return [(f.stem, load_image(str(f))) for f in files]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

_FIELD_NAMES = {f.name for f in fields(ConvertSettings)}


def _parse_value(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def parse_config(spec: str) -> tuple[str, dict]:
    label, _, body = spec.partition(":")
    over = {}
    for item in filter(None, (s.strip() for s in body.split(","))):
        k, _, v = item.partition("=")
        if k not in _FIELD_NAMES:
            raise SystemExit(f"不明な設定項目: {k}  (使用可能: {sorted(_FIELD_NAMES)})")
        over[k] = _parse_value(v)
    return label.strip() or spec, over


def make_settings(base: dict, over: dict) -> ConvertSettings:
    vals = {k: v for k, v in {**base, **over}.items() if not k.startswith("__")}
    return apply_overrides(ConvertSettings(), vals)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fit_to(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height)
    size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
    return img.resize(size, Image.Resampling.LANCZOS if s < 1 else Image.Resampling.NEAREST)


def _crop_zoom(img: Image.Image, frac: float, zoom: int, cell: tuple[int, int]) -> Image.Image:
    """Centre crop covering ``cell / zoom`` pixels, enlarged by ``zoom``."""
    cw, ch = cell[0] // zoom, cell[1] // zoom
    cx, cy = int(img.width * 0.5), int(img.height * frac)
    left = max(0, min(img.width - cw, cx - cw // 2))
    top = max(0, min(img.height - ch, cy - ch // 2))
    crop = img.crop((left, top, left + cw, top + ch))
    return crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.NEAREST)


def build_sheet(name: str, src: Image.Image, results: list[tuple[str, Image.Image, str]],
                cell=(640, 400), zoom: int = 4, crop_y: float = 0.4) -> Image.Image:
    font, small = _font(16), _font(13)
    pad, label_h = 8, 44
    cols = 1 + len(results)
    W = pad + cols * (cell[0] + pad)
    H = pad + 2 * (label_h + cell[1] + pad)
    sheet = Image.new("RGB", (W, H), (40, 40, 48))
    draw = ImageDraw.Draw(sheet)

    # Original, resized to the same display size as the converted images.
    ref_size = results[0][1].size if results else cell
    orig = flatten(src).resize(ref_size, Image.Resampling.LANCZOS)
    tiles = [(f"元画像: {name}", f"{src.width}×{src.height}", orig)]
    tiles += [(label, info, disp) for label, disp, info in results]

    for i, (label, info, disp) in enumerate(tiles):
        x = pad + i * (cell[0] + pad)
        y = pad
        draw.text((x, y), label, fill=(240, 240, 240), font=font)
        draw.text((x, y + 22), info, fill=(170, 170, 190), font=small)
        y += label_h
        sheet.paste(_fit_to(disp, *cell), (x, y))
        y += cell[1] + pad
        draw.text((x, y + 4), f"中央付近 ×{zoom} 拡大", fill=(170, 170, 190), font=small)
        y += label_h
        sheet.paste(_crop_zoom(disp, crop_y, zoom, cell), (x, y))
    return sheet


def run(args) -> list[Path]:
    configs: dict[str, dict] = {}
    if args.configs_json:
        configs.update(json.loads(Path(args.configs_json).read_text(encoding="utf-8")))
    for spec in args.config or []:
        label, over = parse_config(spec)
        configs[label] = over
    if args.presets:
        configs.setdefault("既定 (k-means16 / Bayer4)", {})
        configs.update(STYLE_PRESETS)
    for path in args.profile or []:
        from retrocore.analyzer import load_profile
        prof = load_profile(path)
        ov, _ = profile_to_settings(prof)
        name = prof.get("name") or Path(path).stem
        configs[f"プロファイル: {name}"] = ov
        if args.cross_ramp:
            cov, _ = profile_to_settings(prof, cross_ramp=True)
            configs[f"プロファイル: {name} + ランプ間混色 {cov['style_cross_ramp']}"] = cov
        if args.fit:
            configs[f"プロファイル: {name} + 自動調整{args.fit}回"] = {**ov, "__fit__": (prof, args.fit)}
    if not configs:
        configs = dict(DEFAULT_CONFIGS)
    base = dict(parse_config("base:" + args.base)[1]) if args.base else {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args.inputs, args.synthetic, args.max_inputs)
    written: list[Path] = []
    overview_rows: list[Image.Image] = []

    for name, src in inputs:
        results = []
        for label, over in configs.items():
            s = make_settings(base, over)
            t0 = time.perf_counter()
            if "__fit__" in over:
                from retrocore.optimize import fit_to_profile
                prof, n = over["__fit__"]
                fr = fit_to_profile(src, prof, s, max_evals=n)
                s = fr.settings
            r = convert(src, s)
            dt = time.perf_counter() - t0
            if args.display == "aspect":
                disp = aspect_preview(r.image, PC98_PIXEL_ASPECT * r.pixel_aspect, scale=1)
            elif args.display == "crt":
                disp = crt_preview(r.image, PC98_PIXEL_ASPECT * r.pixel_aspect, scale=2)
            else:
                disp = r.display_image(1, line_double=True).convert("RGB")
            info = f"{r.image.width}×{r.image.height}  {r.used_colors}色  {dt:.2f}s"
            print(f"  {name:>40s} | {label:<32s} | {info}")
            results.append((label, disp, info))
        sheet = build_sheet(name, src, results, zoom=args.zoom)
        path = out_dir / f"{name}.png"
        sheet.save(path, optimize=True)
        written.append(path)
        overview_rows.append(sheet.resize((sheet.width // 2, sheet.height // 2),
                                          Image.Resampling.LANCZOS))

    if overview_rows and not args.no_overview:
        ow = max(r.width for r in overview_rows)
        ov = Image.new("RGB", (ow, sum(r.height for r in overview_rows)), (40, 40, 48))
        y = 0
        for r in overview_rows:
            ov.paste(r, (0, y))
            y += r.height
        path = out_dir / "_overview.png"
        ov.save(path, optimize=True)
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", help="入力画像 (既定: samples/input/*)")
    ap.add_argument("--synthetic", action="store_true", help="合成画像を使う")
    ap.add_argument("--max-inputs", type=int, default=None)
    ap.add_argument("-c", "--config", action="append", help="label:key=value,... (複数指定可)")
    ap.add_argument("--configs-json", help='{"label": {field: value}} の JSON ファイル')
    ap.add_argument("--no-overview", action="store_true", help="_overview.png を出力しない")
    ap.add_argument("--presets", action="store_true", help="スタイル変換プリセットをすべて並べる")
    ap.add_argument("--profile", action="append", help="プロファイル JSON から設定を作って並べる (複数可)")
    ap.add_argument("--fit", type=int, default=0, help="--profile と併用: 自動調整の評価回数")
    ap.add_argument("--cross-ramp", action="store_true",
                    help="--profile と併用: ランプ間の混色を反映した列も並べる")
    ap.add_argument("--base", default="", help="全設定に共通の上書き key=value,...")
    ap.add_argument("--zoom", type=int, default=4, help="拡大クロップの倍率")
    ap.add_argument("--display", choices=["raw", "aspect", "crt"], default="raw",
                    help="表示方法: raw=画素そのまま / aspect=4:3縦長補正 / crt=簡易CRT (表示専用)")
    ap.add_argument("--out", default=str(ROOT / "out" / "compare"))
    args = ap.parse_args(argv)
    for p in run(args):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
