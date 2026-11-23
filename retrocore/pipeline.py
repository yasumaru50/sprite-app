"""High level conversion pipeline: resize -> adjust -> palette -> dither."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from . import postprocess
from .dither import (
    apply_palette,
    cross_ramp_segments,
    gradient_mask,
    indices_to_image,
    selective_pattern_dither,
)
from .palette import (
    PRESET_PALETTES,
    RampPalette,
    extract_palette,
    extract_ramp_palette,
    infer_ramps,
    snap_to_pc98,
)
from .preprocess import smooth

# name -> (width, height, pixel aspect ratio (height of one pixel / width))
RESOLUTION_PRESETS: dict[str, tuple[int, int, float] | None] = {
    "PC-98 640×400": (640, 400, 1.0),
    "PC-98 640×200 (ライン倍化)": (640, 200, 2.0),
    "320×200": (320, 200, 1.0),
    "320×240": (320, 240, 1.0),
    "256×192": (256, 192, 1.0),
    "160×100": (160, 100, 1.0),
    "ピクセルサイズ指定": None,
    "カスタム": None,
}

FIT_MODES = {
    "fit": "全体を収める (余白なし)",
    "fill": "切り抜いて埋める",
    "stretch": "引き伸ばす",
}

RESAMPLE_MODES = {
    "box": ("平均 (Box)", Image.Resampling.BOX),
    "lanczos": ("Lanczos", Image.Resampling.LANCZOS),
    "bilinear": ("Bilinear", Image.Resampling.BILINEAR),
    "nearest": ("ニアレストネイバー", Image.Resampling.NEAREST),
}

PALETTE_SOURCES = {
    "kmeans": "画像から生成 (k-means)",
    "mediancut": "画像から生成 (メディアンカット)",
    **{f"preset:{k}": f"プリセット: {k}" for k in PRESET_PALETTES},
}

CONVERT_MODES = {
    "normal": "通常 (減色＋ディザ)",
    "style": "スタイル変換 (平滑化→ランプ→選択ディザ→後処理)",
}


@dataclass
class ConvertSettings:
    # Resolution
    preset: str = "PC-98 640×400"
    custom_width: int = 320
    custom_height: int = 200
    pixel_size: int = 4
    pixel_aspect: float = 1.0
    fit: str = "fit"
    resample: str = "box"
    # Adjustments (-100 .. 100)
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    # Colours
    colors: int = 16
    palette_source: str = "kmeans"
    snap_4096: bool = True
    # Dithering
    dither: str = "bayer4"
    dither_strength: float = 0.75
    perceptual: bool = False
    # Background used for transparent pixels
    background: tuple[int, int, int] = (0, 0, 0)
    # --- Style conversion (used only when mode == "style") -------------------
    mode: str = "normal"
    style_smooth: str = "bilateral"          # none / bilateral / kuwahara
    style_smooth_strength: float = 0.5       # 0-1
    style_ramps: int = 0                     # number of ramps (0 = auto)
    style_steps: int = 0                     # shades per ramp (0 = auto)
    style_pattern: str = "checker"           # checker / tile2 / bayer4 / none
    style_dither_strength: float = 1.0       # 0-1
    style_grad_low: float = 0.6              # gradation mask: min gradient
    style_grad_high: float = 7.0             # gradation mask: max gradient
    style_mask_curvature: float = 2.5        # gradation mask: tolerated curvature (higher = wider)
    style_despeckle: int = 1                 # majority filter passes (0 = off)
    style_outline: bool = False
    style_outline_threshold: float = 40.0    # luminance gradient / pixel
    style_outline_thickness: int = 1
    style_outline_color: tuple[int, int, int] | None = None  # None = darkest palette colour
    style_ramp_extend: bool = False          # fixed palettes: extend hue ramps to black / white
    style_hue_separation: bool = True        # extra ramps for distant / vivid small hues
    style_reserve_skin: bool = True          # keep a skin-tone ramp (≥ 3 shades)
    # Dither pairs across ramps (opt-in): 0 = only adjacent shades of one ramp
    # (default), up to 1 = pairs of different ramps compete equally.
    style_cross_ramp: float = 0.0
    style_cross_ramp_max_lum: float = 32.0   # max luminance difference of a cross pair (0-255)
    style_cross_ramp_max_hue: float = 180.0  # max hue difference (degrees) of two chromatic colours


@dataclass
class ConvertResult:
    image: Image.Image          # indexed (P) image at native retro resolution
    palette: np.ndarray         # palette actually used
    pixel_aspect: float = 1.0
    elapsed: float = 0.0
    used_colors: int = 0
    info: dict = field(default_factory=dict)

    def display_image(self, scale: int = 1, line_double: bool = True) -> Image.Image:
        """Return the image enlarged for display / saving (nearest neighbour)."""
        w, h = self.image.size
        sy = scale * (int(round(self.pixel_aspect)) if line_double else 1)
        if scale == 1 and sy == 1:
            return self.image.copy()
        return self.image.resize((w * scale, h * max(1, sy)), Image.Resampling.NEAREST)


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.load()
    return img


def flatten(img: Image.Image, background=(0, 0, 0)) -> Image.Image:
    """Convert to RGB, compositing any transparency onto ``background``."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, tuple(background) + (255,))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")


def target_size(src_w: int, src_h: int, s: ConvertSettings) -> tuple[int, int, float]:
    """Compute (width, height, pixel_aspect) of the retro image."""
    preset = RESOLUTION_PRESETS.get(s.preset)
    if s.preset == "ピクセルサイズ指定":
        px = max(1, s.pixel_size)
        aspect = max(1.0, s.pixel_aspect)
        return max(1, round(src_w / px)), max(1, round(src_h / (px * aspect))), aspect
    if preset is None:
        tw, th, aspect = s.custom_width, s.custom_height, max(1.0, s.pixel_aspect)
    else:
        tw, th, aspect = preset
    tw, th = max(1, tw), max(1, th)
    if s.fit == "fit":
        # Work in "display" space where each pixel is `aspect` tall.
        scale = min(tw / src_w, th * aspect / src_h)
        return max(1, round(src_w * scale)), max(1, round(src_h * scale / aspect)), aspect
    return tw, th, aspect


def resize(img: Image.Image, s: ConvertSettings) -> tuple[Image.Image, float]:
    w, h, aspect = target_size(img.width, img.height, s)
    method = RESAMPLE_MODES.get(s.resample, RESAMPLE_MODES["box"])[1]
    if s.fit == "fill" and s.preset != "ピクセルサイズ指定":
        # Crop the source to the target aspect (in display space) first.
        want = w / (h * aspect)
        have = img.width / img.height
        if have > want:
            nw = max(1, round(img.height * want))
            left = (img.width - nw) // 2
            img = img.crop((left, 0, left + nw, img.height))
        elif have < want:
            nh = max(1, round(img.width / want))
            top = (img.height - nh) // 2
            img = img.crop((0, top, img.width, top + nh))
    if (w, h) != img.size:
        img = img.resize((w, h), method)
    return img, aspect


def adjust(img: Image.Image, s: ConvertSettings) -> Image.Image:
    if s.brightness:
        img = ImageEnhance.Brightness(img).enhance(1.0 + s.brightness / 100.0)
    if s.contrast:
        img = ImageEnhance.Contrast(img).enhance(1.0 + s.contrast / 100.0)
    if s.saturation:
        img = ImageEnhance.Color(img).enhance(1.0 + s.saturation / 100.0)
    return img


def build_palette(img: Image.Image, s: ConvertSettings) -> np.ndarray:
    if s.palette_source.startswith("preset:"):
        pal = PRESET_PALETTES[s.palette_source[len("preset:"):]].copy()
        return snap_to_pc98(pal) if s.snap_4096 else pal
    method = "mediancut" if s.palette_source == "mediancut" else "kmeans"
    return extract_palette(img, s.colors, method=method, snap_4096=s.snap_4096)


def build_ramp_palette(img: Image.Image, s: ConvertSettings,
                       palette: np.ndarray | None = None) -> RampPalette:
    """Ramp palette for style mode (generated, preset or forced)."""
    if palette is not None:
        pal = np.asarray(palette, np.uint8)
        return RampPalette(pal, infer_ramps(pal, extend_with_gray=s.style_ramp_extend))
    if s.palette_source.startswith("preset:"):
        pal = build_palette(img, s)
        return RampPalette(pal, infer_ramps(pal, extend_with_gray=s.style_ramp_extend))
    reserve = s.style_outline and s.style_outline_color is not None and s.colors > 2
    if reserve:
        # No slot needed when the source already contains the outline colour.
        px = np.asarray(img.convert("RGB"), dtype=np.int16).reshape(-1, 3)
        near = np.abs(px - np.array(s.style_outline_color, np.int16)).sum(1) <= 40
        reserve = near.mean() < 0.005
    rp = extract_ramp_palette(img, s.colors - int(reserve), n_ramps=s.style_ramps,
                              steps=s.style_steps, snap_4096=s.snap_4096,
                              hue_separation=s.style_hue_separation,
                              reserve_skin=s.style_reserve_skin)
    if reserve:
        # The outline colour (e.g. black from a reference profile) often does
        # not occur in the source; give it its own palette slot.
        col = np.array([s.style_outline_color], np.uint8)
        if s.snap_4096:
            col = snap_to_pc98(col)
        d = np.abs(rp.palette.astype(int) - col.astype(int)).sum(1)
        if d.min() > 24:
            rp = RampPalette(np.vstack([rp.palette, col]), rp.ramps + [[len(rp.palette)]])
    return rp


# Settings that influence the style pipeline up to (and including) the ramp
# palette.  Searches that only vary the later stages (mask, dither,
# despeckle, outline threshold) can reuse that work via ``convert(cache=...)``.
_STAGE1_FIELDS = (
    "preset", "custom_width", "custom_height", "pixel_size", "pixel_aspect", "fit", "resample",
    "brightness", "contrast", "saturation", "background", "colors", "palette_source", "snap_4096",
    "style_smooth", "style_smooth_strength", "style_ramps", "style_steps", "style_ramp_extend",
    "style_hue_separation", "style_reserve_skin", "style_outline", "style_outline_color",
)


def _style_stage1(src: Image.Image, s: ConvertSettings, palette: np.ndarray | None):
    """Resize → adjust → smoothing → ramp palette.  Returns (aspect, rgb, ramp palette)."""
    img = flatten(src, s.background)
    img, aspect = resize(img, s)
    img = adjust(img, s)
    rgb = smooth(img, s.style_smooth, s.style_smooth_strength)
    smoothed = Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8))
    return aspect, rgb, build_ramp_palette(smoothed, s, palette)


def _convert_style(rgb: np.ndarray, rp: RampPalette, s: ConvertSettings):
    """(前処理・ランプパレット済み) → 選択ディザ → 後処理."""
    if s.style_pattern != "none" and s.style_dither_strength > 0:
        mask = gradient_mask(rgb, s.style_grad_low, s.style_grad_high, s.style_mask_curvature)
    else:
        mask = np.zeros(rgb.shape[:2], bool)
    pattern = s.style_pattern if s.style_pattern != "none" else "checker"
    cross = None
    if s.style_cross_ramp > 0:
        cross = cross_ramp_segments(rp.palette, rp.ramps, s.style_cross_ramp_max_lum,
                                    s.style_cross_ramp_max_hue)
    idx = selective_pattern_dither(rgb, rp.palette, rp.ramps, mask, pattern,
                                   s.style_dither_strength, s.perceptual,
                                   cross_segments=cross, cross_amount=s.style_cross_ramp)
    if s.style_despeckle > 0:
        idx = postprocess.remove_isolated(idx, s.style_despeckle, protect=mask)
    outline = None
    if s.style_outline:
        outline = postprocess.outline_mask(rgb, s.style_outline_threshold, s.style_outline_thickness)
        if s.style_outline_color is None:
            ci = postprocess.darkest_index(rp.palette)
        else:
            from .dither import NearestColor
            ci = int(NearestColor(rp.palette).query(np.array([s.style_outline_color]))[0])
        idx = postprocess.apply_outline(idx, outline, ci)
    info = {
        "ramps": rp.ramps,
        "cross_segments": cross,
        "dither_mask": mask,
        "dither_coverage": float(mask.mean()),
        "outline_coverage": float(outline.mean()) if outline is not None else 0.0,
    }
    return indices_to_image(idx, rp.palette), rp.palette, info


def convert(src: Image.Image, s: ConvertSettings, palette: np.ndarray | None = None,
            cache: dict | None = None) -> ConvertResult:
    """Run the full pipeline.

    If ``palette`` is given (palette-transfer mode) it is forced onto the image;
    otherwise a palette is generated according to ``s.palette_source``.
    ``s.mode == "style"`` selects the style-conversion pipeline.

    ``cache`` (style mode): a dict owned by the caller, reused across calls
    with the *same* ``src`` and ``palette``; the expensive first stage
    (resize, smoothing, ramp palette) is computed once per distinct value of
    the settings it depends on.  The result is identical to an uncached call.
    """
    t0 = time.perf_counter()
    if s.mode == "style":
        key = tuple(getattr(s, f) for f in _STAGE1_FIELDS)
        if cache is not None and key in cache:
            aspect, rgb, rp = cache[key]
        else:
            aspect, rgb, rp = _style_stage1(src, s, palette)
            if cache is not None:
                cache[key] = (aspect, rgb, rp)
        out, pal, info = _convert_style(rgb, rp, s)
        return ConvertResult(
            image=out,
            palette=pal,
            pixel_aspect=aspect,
            elapsed=time.perf_counter() - t0,
            used_colors=len(out.getcolors(256) or []),
            info=info,
        )
    img = flatten(src, s.background)
    img, aspect = resize(img, s)
    img = adjust(img, s)
    pal = build_palette(img, s) if palette is None else np.asarray(palette, np.uint8)
    out = apply_palette(img, pal, s.dither, s.dither_strength, s.perceptual)
    used = len(out.getcolors(256) or [])
    return ConvertResult(
        image=out,
        palette=pal,
        pixel_aspect=aspect,
        elapsed=time.perf_counter() - t0,
        used_colors=used,
    )


def save_image(result: ConvertResult, path: str, scale: int = 1, line_double: bool = True) -> None:
    """Save the result; indexed formats keep their palette."""
    img = result.display_image(scale, line_double)
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("jpg", "jpeg", "webp"):
        img.convert("RGB").save(path, quality=95)
    elif ext == "png":
        img.save(path, optimize=True)
    else:
        img.save(path)
