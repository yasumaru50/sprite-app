"""Edge-preserving smoothing used to flatten shading before quantisation.

Hand-painted PC-98 CG consists of flat cel-shaded areas; photos and modern
illustrations carry fine texture and noise that turn into speckles after
reducing to 16 colours.  Both filters below remove that texture while keeping
outlines sharp.  NumPy only (no SciPy / OpenCV), vectorised over the image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .color import box_sum

SMOOTH_METHODS = {
    "none": "なし",
    "bilateral": "バイラテラル",
    "kuwahara": "Kuwahara",
}


def _as_float(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.float32)
    return np.asarray(img, dtype=np.float32)


def bilateral(img, radius: int = 2, sigma_color: float = 25.0,
              sigma_space: float | None = None, iterations: int = 1) -> np.ndarray:
    """Bilateral filter on an RGB image; returns ``float32`` HxWx3.

    The weight of a neighbour is ``exp(-d²/2σs²) * exp(-|Δrgb|²/2σc²)``, so
    pixels across a strong colour edge barely contribute.  Implemented as a
    loop over the (2r+1)² offsets, each a whole-image array operation.
    """
    x = _as_float(img)
    radius = max(0, int(radius))
    if radius == 0 or iterations <= 0:
        return x.copy()
    sigma_space = sigma_space or max(1.0, radius / 1.5)
    inv_c = -0.5 / float(sigma_color) ** 2
    h, w = x.shape[:2]
    for _ in range(int(iterations)):
        p = np.pad(x, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        acc = np.zeros_like(x)
        wsum = np.zeros((h, w), dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ws = np.float32(np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space ** 2)))
                nb = p[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
                d2 = ((nb - x) ** 2).sum(axis=2)
                wt = ws * np.exp(d2 * inv_c)
                acc += nb * wt[:, :, None]
                wsum += wt
        x = acc / wsum[:, :, None]
    return x


def kuwahara(img, radius: int = 2) -> np.ndarray:
    """Classic Kuwahara filter; returns ``float32`` HxWx3.

    For each pixel the four overlapping (r+1)x(r+1) quadrants around it are
    examined and the mean of the one with the lowest variance (summed over
    the channels) is taken.  Means / variances come from integral images.
    """
    x = _as_float(img)
    r = max(0, int(radius))
    if r == 0:
        return x.copy()
    h, w = x.shape[:2]
    # Box statistics of size (r+1)x(r+1) whose top-left corner is at (y, x),
    # computed on an edge-padded image so every quadrant is in range.
    p = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge").astype(np.float64)
    n = r + 1
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1, 3))
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    ii2 = np.zeros_like(ii)
    ii2[1:, 1:] = (p * p).cumsum(0).cumsum(1)

    def stats(oy, ox):
        # quadrant rows [y+oy, y+oy+n) in padded coords
        ys, xs = slice(oy, oy + h), slice(ox, ox + w)
        ye, xe = slice(oy + n, oy + n + h), slice(ox + n, ox + n + w)
        s = ii[ye, xe] - ii[ys, xe] - ii[ye, xs] + ii[ys, xs]
        s2 = ii2[ye, xe] - ii2[ys, xe] - ii2[ye, xs] + ii2[ys, xs]
        mean = s / (n * n)
        var = (s2 / (n * n) - mean ** 2).sum(axis=2)
        return mean, var

    # In padded coordinates the pixel (y, x) is at (y+r, x+r).
    quads = [stats(0, 0), stats(0, r), stats(r, 0), stats(r, r)]
    var = np.stack([q[1] for q in quads], axis=0)
    best = var.argmin(axis=0)
    out = np.zeros_like(x, dtype=np.float64)
    for i, (mean, _) in enumerate(quads):
        out += mean * (best == i)[:, :, None]
    return out.astype(np.float32)


def smooth(img, method: str = "bilateral", strength: float = 0.5) -> np.ndarray:
    """Edge-preserving smoothing with a single 0-1 ``strength`` knob.

    Returns ``float32`` HxWx3.  ``strength`` 0 (or ``method == "none"``) is a
    no-op.  Mapping:

    * bilateral: radius 1-3, colour sigma 12-60, 1-2 passes
    * kuwahara: radius 1-4
    """
    x = _as_float(img)
    s = float(np.clip(strength, 0.0, 1.0))
    if method == "none" or s <= 0:
        return x
    if method == "kuwahara":
        return kuwahara(x, radius=1 + int(round(3 * s)))
    if method == "bilateral":
        return bilateral(x, radius=1 + int(round(2 * s)), sigma_color=12 + 48 * s,
                         iterations=2 if s > 0.6 else 1)
    raise ValueError(f"unknown smoothing method: {method}")


def local_variance(gray: np.ndarray, r: int = 1) -> np.ndarray:
    """Helper for tests / analysis: variance in a (2r+1)² window."""
    g = np.asarray(gray, dtype=np.float64)
    k = float((2 * r + 1) ** 2)
    m = box_sum(g, r) / k
    return box_sum(g * g, r) / k - m * m
