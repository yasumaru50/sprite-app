"""Display-only previews (never used for saved data).

* ``aspect_preview``: PC-98 shows 640x400 on a 4:3 monitor, so every pixel is
  1.2x taller than wide (640/400 ÷ 4/3).  200-line modes double that again.
* ``crt_preview``: rough CRT look — horizontal beam blur, scanline gaps
  between rasters and a little bloom on bright areas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

DISPLAY_ASPECT = 4 / 3


def screen_pixel_aspect(width: int = 640, height: int = 400,
                        display_aspect: float = DISPLAY_ASPECT) -> float:
    """Pixel height / width when ``width``x``height`` fills a ``display_aspect`` screen."""
    return (width / height) / display_aspect


PC98_PIXEL_ASPECT = screen_pixel_aspect(640, 400)  # 1.2


def aspect_preview(img: Image.Image, pixel_aspect: float = PC98_PIXEL_ASPECT,
                   scale: int = 2, smooth: bool = False) -> Image.Image:
    """Enlarge ``scale`` times horizontally and ``scale * pixel_aspect`` vertically."""
    w, h = img.size
    size = (w * scale, max(1, round(h * scale * pixel_aspect)))
    method = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    return img.convert("RGB").resize(size, method)


def crt_preview(img: Image.Image, pixel_aspect: float = PC98_PIXEL_ASPECT, scale: int = 3,
                scanline: float = 0.35, blur: float = 0.6, bloom: float = 0.25) -> Image.Image:
    """Simple CRT simulation.

    ``scanline`` 0-1 darkens the gap between source rasters, ``blur`` is the
    horizontal beam spread in source pixels, ``bloom`` adds a blurred copy of
    the bright parts.
    """
    src = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = src.shape[:2]
    sy = max(2, int(round(scale * pixel_aspect)))
    # Horizontal: nearest upscale then blur along x only (beam spread).
    big = np.repeat(np.repeat(src, scale, axis=1), sy, axis=0)
    if blur > 0:
        rad = blur * scale
        k = np.exp(-0.5 * (np.arange(-int(3 * rad) - 1, int(3 * rad) + 2) / max(rad, 1e-3)) ** 2)
        k /= k.sum()
        r = len(k) // 2
        p = np.pad(big, ((0, 0), (r, r), (0, 0)), mode="edge")
        acc = np.zeros_like(big)
        for i, kv in enumerate(k):
            acc += p[:, i: i + big.shape[1]] * kv
        big = acc
    if scanline > 0:
        # Raised-cosine brightness profile within each raster.
        pos = (np.arange(sy) + 0.5) / sy
        prof = 1.0 - scanline * (1 - np.sin(np.pi * pos)) ** 2
        big *= np.tile(prof, h)[:, None, None].astype(np.float32)
    out = Image.fromarray(np.clip(big, 0, 255).astype(np.uint8))
    if bloom > 0:
        glow = out.filter(ImageFilter.GaussianBlur(radius=scale * 1.5))
        g = np.asarray(glow, np.float32)
        lum = g @ np.array([0.299, 0.587, 0.114], np.float32)
        g *= np.clip((lum - 128) / 127, 0, 1)[:, :, None]
        out = Image.fromarray(np.clip(np.asarray(out, np.float32) + bloom * g, 0, 255).astype(np.uint8))
    return out
