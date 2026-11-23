"""Palette mapping with ordered / error-diffusion dithering."""

from __future__ import annotations

import numpy as np
from PIL import Image

DITHER_METHODS = {
    "none": "なし",
    "bayer2": "ordered (Bayer 2×2)",
    "bayer4": "ordered (Bayer 4×4)",
    "bayer8": "ordered (Bayer 8×8)",
    "floyd": "誤差拡散 (Floyd-Steinberg)",
    "atkinson": "誤差拡散 (Atkinson)",
    "sierra_lite": "誤差拡散 (Sierra Lite)",
    "jjn": "誤差拡散 (Jarvis-Judice-Ninke)",
}

# (dx, dy, weight) entries; weights already normalised.
_KERNELS: dict[str, list[tuple[int, int, float]]] = {
    "floyd": [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)],
    "atkinson": [(1, 0, 1 / 8), (2, 0, 1 / 8), (-1, 1, 1 / 8), (0, 1, 1 / 8),
                 (1, 1, 1 / 8), (0, 2, 1 / 8)],
    "sierra_lite": [(1, 0, 2 / 4), (-1, 1, 1 / 4), (0, 1, 1 / 4)],
    "jjn": [(1, 0, 7 / 48), (2, 0, 5 / 48),
            (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
            (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48)],
}

# Channel weights used for the "perceptual" colour distance.
_PERCEPTUAL = np.array([0.30, 0.59, 0.11], dtype=np.float32) * 3.0


def bayer_matrix(n: int) -> np.ndarray:
    """Normalised Bayer threshold matrix in the range (-0.5, 0.5)."""
    m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size - 0.5


class NearestColor:
    """Fast nearest-palette-colour lookup."""

    def __init__(self, palette: np.ndarray, perceptual: bool = False):
        self.palette = np.asarray(palette, dtype=np.uint8)
        self.weights = _PERCEPTUAL if perceptual else np.ones(3, np.float32)
        self._sw = np.sqrt(self.weights)
        p = self.palette.astype(np.float32) * self._sw
        self._p = p
        self._p_sq = (p ** 2).sum(axis=1)
        self._lut: list[int] | None = None

    def query(self, pixels: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """Return palette indices for an ``(N, 3)`` float/uint8 array."""
        x = np.asarray(pixels, dtype=np.float32).reshape(-1, 3) * self._sw
        out = np.empty(len(x), dtype=np.uint8 if len(self.palette) <= 256 else np.int32)
        for s in range(0, len(x), chunk):
            xs = x[s: s + chunk]
            dist = self._p_sq[None, :] - 2.0 * xs @ self._p.T
            out[s: s + chunk] = dist.argmin(axis=1)
        return out

    def lut(self) -> list[int]:
        """Lookup table indexed by 6-bit-per-channel colour keys."""
        if self._lut is None:
            levels = np.arange(64, dtype=np.float32) * (255.0 / 63.0)
            grid = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1)
            self._lut = self.query(grid.reshape(-1, 3)).tolist()
        return self._lut


def palette_spread(palette: np.ndarray) -> float:
    """Typical distance between neighbouring palette colours."""
    p = palette.astype(np.float32)
    if len(p) < 2:
        return 0.0
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def _ordered(rgb: np.ndarray, nc: NearestColor, size: int, strength: float) -> np.ndarray:
    h, w, _ = rgb.shape
    m = bayer_matrix(size)
    tile = np.tile(m, (h // size + 1, w // size + 1))[:h, :w]
    spread = palette_spread(nc.palette) * strength
    shifted = np.clip(rgb + tile[:, :, None] * spread, 0, 255)
    return nc.query(shifted.reshape(-1, 3)).reshape(h, w)


def _error_diffusion(
    rgb: np.ndarray, nc: NearestColor, kernel: list[tuple[int, int, float]],
    strength: float, serpentine: bool = True,
) -> np.ndarray:
    h, w, _ = rgb.shape
    lut = nc.lut()
    pal = nc.palette.astype(np.float64).tolist()
    r = rgb[:, :, 0].astype(np.float64).ravel().tolist()
    g = rgb[:, :, 1].astype(np.float64).ravel().tolist()
    b = rgb[:, :, 2].astype(np.float64).ravel().tolist()
    out = [0] * (w * h)
    ks = [(dx, dy, wt * strength) for dx, dy, wt in kernel]
    ks_mirror = [(-dx, dy, wt) for dx, dy, wt in ks]

    for y in range(h):
        if serpentine and y & 1:
            xs = range(w - 1, -1, -1)
            kern = ks_mirror
        else:
            xs = range(w)
            kern = ks
        row = y * w
        for x in xs:
            i = row + x
            rv, gv, bv = r[i], g[i], b[i]
            ri = 0 if rv < 0 else (255 if rv > 255 else int(rv + 0.5))
            gi = 0 if gv < 0 else (255 if gv > 255 else int(gv + 0.5))
            bi = 0 if bv < 0 else (255 if bv > 255 else int(bv + 0.5))
            k = lut[((ri >> 2) << 12) | ((gi >> 2) << 6) | (bi >> 2)]
            out[i] = k
            pr, pg, pb = pal[k]
            er, eg, eb = rv - pr, gv - pg, bv - pb
            for dx, dy, wt in kern:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and ny < h:
                    j = ny * w + nx
                    r[j] += er * wt
                    g[j] += eg * wt
                    b[j] += eb * wt
    return np.array(out, dtype=np.uint8).reshape(h, w)


def apply_palette(
    img: Image.Image,
    palette: np.ndarray,
    method: str = "none",
    strength: float = 1.0,
    perceptual: bool = False,
) -> Image.Image:
    """Map ``img`` onto ``palette`` and return an indexed (mode ``P``) image."""
    palette = np.asarray(palette, dtype=np.uint8)
    if len(palette) == 0:
        raise ValueError("パレットが空です")
    palette = palette[:256]
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    nc = NearestColor(palette, perceptual)

    if method.startswith("bayer") and strength > 0:
        idx = _ordered(rgb, nc, int(method[5:]), strength)
    elif method in _KERNELS and strength > 0:
        idx = _error_diffusion(rgb, nc, _KERNELS[method], strength)
    else:
        idx = nc.query(rgb.reshape(-1, 3)).reshape(rgb.shape[:2])

    h, w = idx.shape
    out = Image.frombytes("P", (w, h), np.ascontiguousarray(idx, dtype=np.uint8).tobytes())
    out.putpalette(palette.ravel().tolist())
    return out


# ---------------------------------------------------------------------------
# Selective pattern dither (style conversion)
# ---------------------------------------------------------------------------
#
# Era CG used dithering sparingly: flat cel areas and faces are solid colour,
# only soft gradations (skies, metal, hair highlights, shading ramps) are
# rendered with a regular pattern mixing two *neighbouring shades of the same
# ramp*.  We therefore (1) find gradation regions and (2) inside them pick the
# bracketing pair of ramp shades and choose between them with a small fixed
# threshold pattern anchored to the screen grid.

from .color import box_mean, luminance, sobel  # noqa: E402

PATTERN_METHODS = {
    "checker": "市松 (2色 50%)",
    "tile2": "2×2 タイル (25% 刻み)",
    "bayer4": "4×4 Bayer",
}

# Threshold matrices in (0, 1); a pixel takes the lighter shade when t > thr.
_PATTERNS = {
    # Checkerboard: t < 1/4 -> dark, 1/4..3/4 -> 50% checker, > 3/4 -> light.
    "checker": np.array([[0.25, 0.75], [0.75, 0.25]], np.float32),
    # 2x2 ordered tile: 0 / 25 / 50 / 75 / 100 % mixes.
    "tile2": (np.array([[0, 2], [3, 1]], np.float32) + 0.5) / 4.0,
    "bayer4": bayer_matrix(4) + 0.5,
}


def gradient_mask(
    rgb: np.ndarray,
    grad_low: float = 0.6,
    grad_high: float = 7.0,
    curvature: float = 2.5,
    scale: int = 2,
    min_fill: float = 0.55,
    open_radius: int = 1,
) -> np.ndarray:
    """Boolean mask of "gradation" regions.

    A pixel belongs to a gradation when, measured on a slightly blurred
    image (box radius ``scale``):

    * the colour gradient (max over channels, units / pixel) lies in
      ``[grad_low, grad_high]`` — not flat paint, not an edge, and
    * the change is continuous: the Laplacian is small compared to the
      gradient (``|∇²| <= curvature + 0.5 * |∇|``), which rejects texture
      and noise.

    The raw mask is cleaned with a majority (box) filter and a morphological
    opening of ``open_radius`` so that only coherent regions remain, and
    pixels next to strong edges are excluded.
    """
    x = np.asarray(rgb, dtype=np.float32)
    blur = box_mean(x, scale).astype(np.float32)
    g = np.zeros(x.shape[:2], np.float32)
    lap = np.zeros(x.shape[:2], np.float32)
    edge = np.zeros(x.shape[:2], np.float32)
    for c in range(3):
        gx, gy = sobel(blur[:, :, c])
        g = np.maximum(g, np.hypot(gx, gy))
        p = np.pad(blur[:, :, c], 1, mode="edge")
        l4 = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
        lap = np.maximum(lap, np.abs(l4))
        ex, ey = sobel(x[:, :, c])
        edge = np.maximum(edge, np.hypot(ex, ey))
    raw = (g >= grad_low) & (g <= grad_high) & (lap <= curvature + 0.5 * g)
    near_edge = box_mean((edge > grad_high * 3).astype(np.float32), 1) > 0
    raw &= ~near_edge
    m = box_mean(raw.astype(np.float32), 2) >= min_fill
    if open_radius > 0:
        m = box_mean(m.astype(np.float32), open_radius) > 0.999   # erode
        m = box_mean(m.astype(np.float32), open_radius) > 0.001   # dilate
    return m


def _segments(palette: np.ndarray, ramps: list[list[int]]) -> np.ndarray:
    segs = [(r[i], r[i + 1]) for r in ramps for i in range(len(r) - 1)]
    return np.array(segs, dtype=np.int64).reshape(-1, 2)


def cross_ramp_segments(palette: np.ndarray, ramps: list[list[int]], max_lum: float = 32.0,
                        max_hue: float = 180.0, gray_chroma: float = 28.0) -> np.ndarray:
    """Pairs of colours from *different* ramps that may be mixed by dither.

    Reference CG often mixes colours of different hues whose luminance is
    close (skin + white, a warm and a cool shade of similar brightness) rather
    than two shades of one ramp.  A pair qualifies when the two colours never
    share a ramp, their luminance differs by at most ``max_lum`` (0-255) and,
    if both are chromatic (max - min channel ≥ ``gray_chroma``), their hue
    differs by at most ``max_hue`` degrees.  Rows are ``(darker, lighter)``.
    """
    from .color import rgb_to_hsv

    pal = np.asarray(palette, dtype=np.float32)
    k = len(pal)
    if k < 2:
        return np.zeros((0, 2), np.int64)
    together = np.zeros((k, k), bool)
    for r in ramps:
        rr = np.asarray(r, dtype=np.int64)
        together[np.ix_(rr, rr)] = True
    lum = pal @ np.array([0.299, 0.587, 0.114], np.float32)
    chroma = pal.max(1) - pal.min(1)
    hue = rgb_to_hsv(pal)[:, 0] * 360.0
    dh = np.abs(hue[:, None] - hue[None, :])
    dh = np.minimum(dh, 360.0 - dh)
    chrom2 = (chroma[:, None] >= gray_chroma) & (chroma[None, :] >= gray_chroma)
    ok = ~together & (np.abs(lum[:, None] - lum[None, :]) <= max_lum) & \
        (~chrom2 | (dh <= max_hue))
    i, j = np.nonzero(np.triu(ok, 1))
    dark_first = lum[i] <= lum[j]
    return np.stack([np.where(dark_first, i, j), np.where(dark_first, j, i)], 1).astype(np.int64)


def pattern_threshold(pattern: str, h: int, w: int) -> np.ndarray:
    m = _PATTERNS[pattern]
    n = m.shape[0]
    return np.tile(m, (h // n + 1, w // n + 1))[:h, :w]


def selective_pattern_dither(
    rgb: np.ndarray,
    palette: np.ndarray,
    ramps: list[list[int]],
    mask: np.ndarray | None = None,
    pattern: str = "checker",
    strength: float = 1.0,
    perceptual: bool = False,
    chunk: int = 32768,
    cross_segments: np.ndarray | None = None,
    cross_amount: float = 0.0,
    cross_min_mix: float = 0.2,
) -> np.ndarray:
    """Quantise ``rgb`` (HxWx3) to palette indices (HxW uint8).

    Outside ``mask`` every pixel gets its nearest colour (solid paint).  Inside,
    the pixel is projected onto every segment between adjacent shades of a ramp;
    the closest segment containing the pixel's nearest colour gives the pair
    (dark, light) and the mix ratio ``t``,
    and the screen-anchored ``pattern`` threshold picks one of the two.
    ``strength`` < 1 pulls ``t`` towards 0/1 (fewer dithered pixels).

    ``cross_segments`` (see ``cross_ramp_segments``) adds pairs of colours of
    different ramps as candidates when ``cross_amount`` > 0 (opt-in; 0 keeps
    the same-ramp-only behaviour).  Their squared projection distance is
    divided by ``cross_amount²``, so at 1 they compete equally with ramp
    neighbours and at lower values they are only used when they fit clearly
    better (e.g. beyond the end of a ramp).  The rule that a pair must contain
    the pixel's nearest colour applies to them as well, and a cross pair is
    only a candidate when each colour gets at least ``cross_min_mix`` of the
    mix: a few dots of a different hue in an almost solid area read as noise
    (tinted bands along soft gradients), not as an intermediate colour.
    """
    x = np.asarray(rgb, dtype=np.float32)
    h, w = x.shape[:2]
    pal = np.asarray(palette, dtype=np.uint8)
    nc = NearestColor(pal, perceptual)
    idx = nc.query(x.reshape(-1, 3)).reshape(h, w)
    segs = _segments(pal, ramps)
    factor = None
    amount = float(np.clip(cross_amount, 0.0, 1.0))
    if amount > 0 and cross_segments is not None and len(cross_segments):
        cross = np.asarray(cross_segments, dtype=np.int64).reshape(-1, 2)
        factor = np.concatenate([np.ones(len(segs), np.float32),
                                 np.full(len(cross), 1.0 / amount ** 2, np.float32)])
        is_cross = np.arange(len(factor)) >= len(segs)
        segs = np.concatenate([segs, cross])
        lo_mix = float(np.clip(cross_min_mix, 0.0, 0.5))
    if mask is None or not mask.any() or len(segs) == 0 or strength <= 0:
        return idx.astype(np.uint8)

    sw = nc._sw
    p = pal.astype(np.float32) * sw
    a, b = p[segs[:, 0]], p[segs[:, 1]]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-6)

    ys, xs = np.nonzero(mask)
    px = x[ys, xs] * sw
    thr = pattern_threshold(pattern, h, w)[ys, xs]
    s = float(np.clip(strength, 0, 1))
    out = idx.copy()
    for st in range(0, len(px), chunk):
        q = px[st: st + chunk]
        t = ((q[:, None, :] - a[None]) * ab[None]).sum(2) / ab2[None]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None] + t[:, :, None] * ab[None]
        d = ((q[:, None, :] - proj) ** 2).sum(2)
        if factor is not None:
            d = d * factor[None]
            weak = is_cross[None] & ((t < lo_mix) | (t > 1.0 - lo_mix))
            d = np.where(weak, np.inf, d)
        # Only mix pairs that contain the pixel's nearest colour; otherwise a
        # pixel of a one-shade ramp would be dithered with an unrelated ramp.
        near = idx[ys[st: st + chunk], xs[st: st + chunk]].astype(np.int64)
        has = (segs[None, :, 0] == near[:, None]) | (segs[None, :, 1] == near[:, None])
        d = np.where(has, d, np.inf)
        best = d.argmin(1)
        ok = np.isfinite(d[np.arange(len(q)), best])
        tb = t[np.arange(len(q)), best]
        tb = np.round(tb) + (tb - np.round(tb)) * s
        light = tb > thr[st: st + chunk]
        chosen = np.where(light, segs[best, 1], segs[best, 0])
        chosen = np.where(ok, chosen, near)
        out[ys[st: st + chunk], xs[st: st + chunk]] = chosen
    return out.astype(np.uint8)


def indices_to_image(idx: np.ndarray, palette: np.ndarray) -> Image.Image:
    h, w = idx.shape
    out = Image.frombytes("P", (w, h), np.ascontiguousarray(idx, dtype=np.uint8).tobytes())
    out.putpalette(np.asarray(palette, np.uint8)[:256].ravel().tolist())
    return out
