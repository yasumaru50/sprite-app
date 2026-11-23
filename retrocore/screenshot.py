"""Recover the native pixel grid and palette of a (scaled) screenshot.

Reference CG usually arrives as an emulator screenshot that was enlarged by a
non-integer factor (e.g. 640x400 shown at 1024x640, ×1.6) with a smoothing
filter, so it contains 100k+ colours instead of 16.  Statistics about dither
patterns, outlines or isolated pixels are meaningless on such an image, so the
analyzer first

1. estimates the scale per axis from the periodicity of the column / row
   edge-energy profile (a DFT peak; ambiguous aliases are resolved by a
   reconstruction test), then fits scale and phase so that the predicted
   native pixel boundaries carry maximum edge energy;
2. samples the screenshot pixel nearest to every native pixel centre (nearest
   instead of bilinear sampling avoids blurring the already-blurred image a
   second time — 1-pixel checker dither survives);
3. re-quantises to a small palette (k-means with an elbow rule, optional
   rounding to the PC-98 4096-colour space) and returns an index image.

Images that are already native (≤ 256 colours) are only checked for integer
nearest-neighbour enlargement.  NumPy only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .dither import NearestColor
from .palette import _kmeans, snap_to_pc98, sort_by_luminance, unique_colors


@dataclass
class GridInfo:
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0       # screenshot x of the first native pixel centre
    offset_y: float = 0.0
    native_width: int = 0
    native_height: int = 0
    confidence: float = 0.0     # spectral peak / median (0 when not scaled)
    method: str = "native"      # native / integer / spectral

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("scale_x", "scale_y", "offset_x", "offset_y", "confidence"):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class NativeImage:
    indices: np.ndarray         # HxW uint8 palette indices
    palette: np.ndarray         # (K, 3) uint8
    grid: GridInfo
    source_colors: int          # unique colours in the input screenshot

    def rgb(self) -> np.ndarray:
        return self.palette[self.indices]

    def image(self) -> Image.Image:
        h, w = self.indices.shape
        img = Image.frombytes("P", (w, h), np.ascontiguousarray(self.indices).tobytes())
        img.putpalette(self.palette.ravel().tolist())
        return img


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------


def _edge_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """Edge energy between neighbouring columns (axis=1) or rows (axis=0)."""
    return np.abs(np.diff(gray, axis=axis)).sum(axis=1 - axis)


def _peak_frequency(d: np.ndarray) -> tuple[float, float]:
    """Dominant frequency (cycles / pixel, 0.12-0.5) of a profile and its SNR.

    0.12 limits the candidates to scales <= ~8 (1/f) so that slow variations
    of the picture content are not mistaken for the pixel grid.
    """
    d = d - d.mean()
    x = np.arange(len(d), dtype=np.float64)
    fs = np.linspace(0.12, 0.5, 1300)
    mag = np.abs(np.exp(-2j * np.pi * np.outer(fs, x)) @ d)
    f = fs[int(mag.argmax())]
    fine = np.linspace(f - 4e-4, f + 4e-4, 81)
    mag2 = np.abs(np.exp(-2j * np.pi * np.outer(fine, x)) @ d)
    snr = float(mag.max() / max(np.median(mag), 1e-9))
    return float(fine[int(mag2.argmax())]), snr


def _fit_grid(d: np.ndarray, s0: float, span: float = 0.004, ds: float = 2e-4,
              do: float = 0.05) -> tuple[float, float]:
    """Refine scale ``s`` and boundary offset to maximise boundary edge energy.

    Returns ``(s, first_centre)`` in screenshot pixel coordinates.
    """
    n = len(d)
    best = (-1.0, s0, 0.0)
    for s in np.arange(s0 - span, s0 + span + 1e-12, ds):
        if s <= 1.0:
            continue
        offs = np.arange(0.0, s, do)
        k = np.arange(int(n / s) + 1)
        pos = offs[:, None] + k[None, :] * s           # boundary positions (diff index)
        valid = pos < n - 1
        i = np.clip(np.floor(pos).astype(int), 0, n - 2)
        f = pos - np.floor(pos)
        e = np.where(valid, d[i] * (1 - f) + d[i + 1] * f, 0.0).sum(1) / valid.sum(1)
        j = int(e.argmax())
        if e[j] > best[0]:
            best = (float(e[j]), float(s), float(offs[j]))
    _, s, o = best
    # Boundary between diff index i and i+1 lies at pixel i+0.5; centres are s/2 further.
    return s, (o + 0.5 + s / 2) % s


def _sample_axis(n_src: int, s: float, o: float) -> np.ndarray:
    count = max(1, int((n_src - o) / s + 0.5))
    pos = np.round(o + np.arange(count) * s).astype(int)
    return pos[pos < n_src]


def _recon_error(gray: np.ndarray, axis: int, s: float, o: float) -> float:
    """Mean error after nearest down- and up-sampling along one axis."""
    n = gray.shape[axis]
    idx = _sample_axis(n, s, o)
    k = np.clip(np.round((np.arange(n) - o) / s).astype(int), 0, len(idx) - 1)
    src = np.take(gray, idx[k], axis=axis)
    return float(np.abs(gray - src).mean())


def _integer_factor(a: np.ndarray, axis: int) -> int:
    n = a.shape[axis]
    for k in (8, 6, 5, 4, 3, 2):
        if n % k == 0 or n > 8 * k:
            m = (n // k) * k
            b = np.take(a, np.arange(m), axis=axis)
            base = np.take(b, np.arange(0, m, k), axis=axis)
            if np.array_equal(np.repeat(base, k, axis=axis), b):
                return k
    return 1


def detect_grid(img, min_snr: float = 5.0, max_native_colors: int = 256) -> GridInfo:
    a = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    h, w = a.shape[:2]
    flat = a.reshape(-1, 3).astype(np.uint32)
    n_colors = len(np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]))
    if n_colors <= max_native_colors:
        kx, ky = _integer_factor(a, 1), _integer_factor(a, 0)
        return GridInfo(kx, ky, (kx - 1) / 2, (ky - 1) / 2, w // kx, h // ky, 0.0,
                        "integer" if kx * ky > 1 else "native")

    gray = a.astype(np.float32).mean(2)
    out = []
    for axis in (1, 0):
        d = _edge_profile(gray, axis)
        f, snr = _peak_frequency(d)
        if snr < min_snr or len(d) < 16:
            out.append((1.0, 0.0, snr))
            continue
        # f and 1-f are indistinguishable on an integer grid: scale 1/(1-f) in
        # (1, 2] or 1/f >= 2.  Prefer the coarser grid if it reconstructs as well.
        cands = sorted({round(1 / (1 - f), 6), round(1 / f, 6)})
        fits = []
        for s0 in cands:
            s, o = _fit_grid(d, s0)
            fits.append((s, o, _recon_error(gray, axis, s, o)))
        best = fits[0]
        for cand in fits[1:]:
            if cand[2] <= best[2] * 1.15 + 0.5:
                best = cand
        out.append((best[0], best[1], snr))
    (sx, ox, cx), (sy, oy, cy) = out
    nw = len(_sample_axis(w, sx, ox))
    nh = len(_sample_axis(h, sy, oy))
    scaled = sx > 1.0 or sy > 1.0
    return GridInfo(sx, sy, ox, oy, nw, nh, min(cx, cy) if scaled else 0.0,
                    "spectral" if scaled else "native")


def resample_native(img, grid: GridInfo) -> np.ndarray:
    a = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    ys = _sample_axis(a.shape[0], grid.scale_y, grid.offset_y)
    xs = _sample_axis(a.shape[1], grid.scale_x, grid.offset_x)
    return a[ys][:, xs]


# ---------------------------------------------------------------------------
# Palette recovery
# ---------------------------------------------------------------------------


def _histogram(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], 1).astype(np.uint8)
    return cols, counts.astype(np.float64)


def pure_pixel_mask(rgb: np.ndarray, tol: int = 6) -> np.ndarray:
    """Pixels whose whole 3x3 neighbourhood has (almost) the same colour.

    Resampling blends only occur next to colour changes, so these pixels carry
    the true palette colours (plus a few levels of noise).
    """
    a = np.asarray(rgb, dtype=np.int16)
    h, w = a.shape[:2]
    ok = np.zeros((h, w), bool)
    if h < 3 or w < 3:
        return ok
    c = a[1:-1, 1:-1]
    inner = np.ones(c.shape[:2], bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nb = a[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx]
            inner &= np.abs(nb - c).max(2) <= tol
    ok[1:-1, 1:-1] = inner
    return ok


def _segment_distance(px: np.ndarray, pal: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Distance of each pixel to the nearest segment between two palette colours."""
    i, j = np.triu_indices(len(pal), 1)
    if len(i) == 0:
        return np.full(len(px), np.inf, np.float32)
    a, ab = pal[i], pal[j] - pal[i]
    ab2 = np.maximum((ab ** 2).sum(1), 1e-6)
    out = np.empty(len(px), np.float32)
    for s in range(0, len(px), chunk):
        q = px[s: s + chunk]
        t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(2) / ab2[None], 0, 1)
        d = ((q[:, None] - a[None] - t[..., None] * ab[None]) ** 2).sum(2)
        out[s: s + chunk] = np.sqrt(d.min(1))
    return out


def recover_palette(rgb: np.ndarray, max_colors: int = 16, min_share: float = 0.0005,
                    snap_4096: bool = True, merge_dist: float = 20.0,
                    residual_dist: float = 40.0, blend_dist: float = 14.0,
                    min_residual: float = 0.0015) -> np.ndarray:
    """Palette of a resampled screenshot.

    Colours are collected from "pure" pixels (uniform 3x3 neighbourhood),
    grouped on the 4-bit grid, and every group holding at least ``min_share``
    of the pure pixels becomes a palette entry (its mean colour, optionally
    rounded to 4096 colours); at most ``max_colors`` entries by frequency.
    Groups closer than ``merge_dist`` to an already chosen colour are noise
    and skipped.  If budget remains, colours of small details (no pure pixels,
    e.g. thin eye highlights) are recovered from pixels lying further than
    ``residual_dist`` from the palette.  Falls back to k-means when almost
    no pixel is pure.
    """
    a = np.asarray(rgb, dtype=np.uint8)
    pure = pure_pixel_mask(a)
    if pure.mean() < 0.02:
        cols, counts = _histogram(a)
        pal = _kmeans(cols, counts, min(max_colors, len(cols)))
    else:
        px = a[pure].astype(np.float64)
        q = np.round(px / 17.0).astype(np.int64)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, px)
        means = sums / counts[:, None]
        order = np.argsort(-counts, kind="stable")
        chosen: list[np.ndarray] = []
        for i in order:
            if counts[i] < min_share * len(px) or len(chosen) >= max_colors:
                break
            if all(np.sqrt(((means[i] - c) ** 2).sum()) > merge_dist for c in chosen):
                chosen.append(means[i])
        if not chosen:
            chosen.append(means[order[0]])
        # Residual pass for small details.  Pixels that lie on the segment
        # between two palette colours are resampling blends, not new colours.
        ucols, ucnt = _histogram(a)
        cand = ucols.astype(np.float32)
        cw = ucnt.astype(np.float64)
        total = cw.sum()
        pal_f = np.array(chosen, np.float32)
        d = np.full(len(cand), np.inf, np.float32)
        for c in pal_f:
            d = np.minimum(d, np.sqrt(((cand - c) ** 2).sum(1)))
        keep = d > residual_dist
        cand, cw = cand[keep], cw[keep]
        while len(chosen) < max_colors and len(cand):
            blend = _segment_distance(cand, np.array(chosen, np.float32)) <= blend_dist
            cand, cw = cand[~blend], cw[~blend]
            if cw.sum() < min_residual * total:
                break
            q2 = np.round(cand / 17.0).astype(np.int64)
            k2 = q2[:, 0] * 256 + q2[:, 1] * 16 + q2[:, 2]
            u2, inv2 = np.unique(k2, return_inverse=True)
            j = int(np.bincount(inv2, weights=cw).argmax())
            m = inv2 == j
            new = (cand[m] * cw[m, None]).sum(0) / cw[m].sum()
            chosen.append(new)
            far = np.sqrt(((cand - new) ** 2).sum(1)) > residual_dist
            cand, cw = cand[far], cw[far]
        pal = np.round(np.array(chosen)).astype(np.uint8)
    if snap_4096:
        pal = snap_to_pc98(pal)
    return sort_by_luminance(unique_colors(np.asarray(pal, np.uint8)))


def refine_palette(native: np.ndarray, idx: np.ndarray, pal: np.ndarray,
                   snap_4096: bool = True) -> np.ndarray:
    """Replace each colour by the 4-bit mode of the pixels assigned to it.

    Colours that only occur inside dither patterns have no pure pixels and
    their first estimate is pulled towards the neighbouring colour.  With
    nearest sampling a good part of the samples still sits close to a native
    pixel centre, so the most frequent 4-bit value is the unblended colour.
    """
    px = native.reshape(-1, 3).astype(np.float64)
    lab = idx.ravel()
    out = pal.copy()
    for i in range(len(pal)):
        sel = px[lab == i]
        if len(sel) < 4:
            continue
        q = np.round(sel / 17.0).astype(np.int64)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        u, inv, c = np.unique(keys, return_inverse=True, return_counts=True)
        mode = sel[inv == int(c.argmax())].mean(0)
        out[i] = np.round(mode).astype(np.uint8)
    if snap_4096:
        out = snap_to_pc98(out)
    return sort_by_luminance(unique_colors(out))


def normalize_screenshot(img, snap_4096: bool = True, max_colors: int = 64) -> NativeImage:
    """Grid detection + nearest resampling + palette recovery."""
    rgb_img = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img)
    grid = detect_grid(rgb_img)
    native = resample_native(rgb_img, grid)
    cols, _ = _histogram(np.asarray(rgb_img))
    if grid.method != "spectral" and len(_histogram(native)[0]) <= 256:
        pal = sort_by_luminance(_histogram(native)[0])     # already indexed: exact
    else:
        pal = recover_palette(native, max_colors=max_colors, snap_4096=snap_4096)
    idx = NearestColor(pal).query(native.reshape(-1, 3)).reshape(native.shape[:2])
    if grid.method == "spectral" or len(cols) > 256:
        pal = refine_palette(native, idx, pal, snap_4096)
        idx = NearestColor(pal).query(native.reshape(-1, 3)).reshape(native.shape[:2])
    # Drop palette entries that ended up unused and re-index.
    used = np.unique(idx)
    remap = np.zeros(len(pal), np.uint8)
    remap[used] = np.arange(len(used))
    return NativeImage(remap[idx].astype(np.uint8), pal[used], grid, int(len(cols)))
