"""Clean-up passes on the quantised (palette index) image.

* ``remove_isolated``: hand-drawn 16-colour CG has almost no stray single
  pixels outside dither patterns.  A pixel whose 8 neighbours share its colour
  at most ``max_same`` times is replaced by the most common neighbour colour
  (a majority filter), optionally sparing a protected mask (dither regions,
  where isolated pixels are intentional).
* ``outline_mask`` / ``apply_outline``: dark contour lines on the darker side
  of strong edges of the (smoothed) source image.

All functions work on ``HxW`` integer index arrays; NumPy only.
"""

from __future__ import annotations

import numpy as np

from .color import box_mean, luminance, sobel

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``a`` shifted so that out[y, x] = a[y+dy, x+dx] (edge replicated)."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    return p[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]


def neighbour_counts(idx: np.ndarray, n_colors: int | None = None) -> np.ndarray:
    """``(K, H, W)`` counts of each palette index among the 8 neighbours."""
    idx = np.asarray(idx)
    k = int(n_colors or idx.max() + 1)
    counts = np.zeros((k,) + idx.shape, dtype=np.uint8)
    flat = counts.reshape(k, -1)
    pos = np.arange(idx.size)
    for dy, dx in _N8:
        nb = _shifted(idx, dy, dx).ravel()
        flat[nb, pos] += 1
    return counts


def remove_isolated(idx: np.ndarray, iterations: int = 1, max_same: int = 0,
                    protect: np.ndarray | None = None) -> np.ndarray:
    """Replace isolated pixels by the majority colour of their neighbours."""
    out = np.asarray(idx).copy()
    for _ in range(max(0, int(iterations))):
        counts = neighbour_counts(out)
        h, w = out.shape
        same = np.take_along_axis(counts, out[None].astype(np.int64), axis=0)[0]
        lonely = same <= max_same
        if protect is not None:
            lonely &= ~protect
        if not lonely.any():
            break
        majority = counts.argmax(axis=0).astype(out.dtype)
        out = np.where(lonely, majority, out)
    return out


def outline_mask(rgb: np.ndarray, threshold: float = 40.0, thickness: int = 1) -> np.ndarray:
    """Pixels on the darker side of luminance edges stronger than ``threshold``.

    ``threshold`` is the Sobel gradient in luminance levels / pixel
    (0-255 scale).  ``thickness`` > 1 grows the line towards the dark side.
    """
    lum = luminance(np.asarray(rgb, dtype=np.float32))
    gx, gy = sobel(lum)
    mag = np.hypot(gx, gy)
    edge = mag >= threshold
    m = edge & (lum <= box_mean(lum, 1))
    t = max(1, int(thickness))
    if t > 1:
        # Grow towards the dark side, staying within ``t`` pixels of an edge.
        dark_wide = lum <= box_mean(lum, t)
        near = box_mean(edge.astype(np.float32), t - 1) > 0
        for _ in range(t - 1):
            grown = box_mean(m.astype(np.float32), 1) > 0
            m = m | (grown & dark_wide & near)
    return m


def darkest_index(palette: np.ndarray) -> int:
    return int(np.argmin(luminance(np.asarray(palette, np.float32))))


def apply_outline(idx: np.ndarray, mask: np.ndarray, color_index: int) -> np.ndarray:
    out = np.asarray(idx).copy()
    out[mask] = color_index
    return out
