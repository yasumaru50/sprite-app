"""Small vectorised colour helpers shared by the style-conversion modules.

All functions accept arrays whose last axis is RGB (0-255, any numeric dtype)
and return ``float32`` arrays.  Only NumPy is used.
"""

from __future__ import annotations

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma, 0-255."""
    return np.asarray(rgb, dtype=np.float32) @ LUMA


def chroma(rgb: np.ndarray) -> np.ndarray:
    """max(R,G,B) - min(R,G,B), 0-255 (more robust than HSV saturation on dark colours)."""
    a = np.asarray(rgb, dtype=np.float32)
    return a.max(axis=-1) - a.min(axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Return H (0-1, cyclic), S (0-1), V (0-1) stacked on the last axis."""
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0)
    h = np.where(mx == r, ((g - b) / safe) % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = np.where(d > 0, h / 6.0, 0.0)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1).astype(np.float32)


def hue_distance(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Cyclic distance between hues in 0-1 (result in 0-0.5)."""
    d = np.abs(np.asarray(h1, np.float32) - np.asarray(h2, np.float32)) % 1.0
    return np.minimum(d, 1.0 - d)


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) window (edge padded) via an integral image."""
    if r <= 0:
        return np.asarray(a, dtype=np.float64).copy()
    a = np.asarray(a, dtype=np.float64)
    pad = [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(a, pad, mode="edge")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1) + p.shape[2:], dtype=np.float64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    h, w = a.shape[:2]
    return ii[k:k + h, k:k + w] - ii[:h, k:k + w] - ii[k:k + h, :w] + ii[:h, :w]


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    return box_sum(a, r) / float((2 * r + 1) ** 2)


def sobel(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel derivatives (normalised to units per pixel), edge padded."""
    p = np.pad(np.asarray(gray, dtype=np.float32), 1, mode="edge")
    gx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
          - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / 8.0
    gy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
          - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / 8.0
    return gx, gy
