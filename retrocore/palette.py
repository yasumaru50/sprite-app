"""Palette extraction, presets and palette file I/O.

A palette is always represented as a ``numpy.ndarray`` of shape ``(K, 3)``
and dtype ``uint8`` (RGB).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .color import rgb_to_hsv

# ---------------------------------------------------------------------------
# Preset palettes
# ---------------------------------------------------------------------------


def _from_4bit(rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Build a palette from PC-98 style 4-bit (0-15) channel values."""
    return (np.array(rows, dtype=np.uint16) * 17).astype(np.uint8)


# PC-9801 digital 8 colours (GRB 1 bit each).
PC98_DIGITAL_8 = _from_4bit([
    (0, 0, 0), (0, 0, 15), (15, 0, 0), (15, 0, 15),
    (0, 15, 0), (0, 15, 15), (15, 15, 0), (15, 15, 15),
])

# Default analog 16-colour palette of PC-9801 (approximation of the
# N88-BASIC power-on palette).
PC98_DEFAULT_16 = _from_4bit([
    (0, 0, 0), (0, 0, 15), (15, 0, 0), (15, 0, 15),
    (0, 15, 0), (0, 15, 15), (15, 15, 0), (15, 15, 15),
    (7, 7, 7), (0, 0, 10), (10, 0, 0), (10, 0, 10),
    (0, 10, 0), (0, 10, 10), (10, 10, 0), (10, 10, 10),
])

# Classic IBM/VGA 16 colours.
VGA_16 = np.array([
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
], dtype=np.uint8)


def _rgb332() -> np.ndarray:
    r = np.repeat(np.arange(8), 32)
    g = np.tile(np.repeat(np.arange(8), 4), 8)
    b = np.tile(np.arange(4), 64)
    pal = np.stack([r * 255 / 7, g * 255 / 7, b * 255 / 3], axis=1)
    return np.round(pal).astype(np.uint8)


RGB332_256 = _rgb332()

PRESET_PALETTES: dict[str, np.ndarray] = {
    "PC-98 標準16色": PC98_DEFAULT_16,
    "PC-98 デジタル8色": PC98_DIGITAL_8,
    "VGA 16色": VGA_16,
    "RGB332 256色": RGB332_256,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def snap_to_pc98(palette: np.ndarray) -> np.ndarray:
    """Round every channel to the PC-98 analog 4096-colour space (4 bit/ch)."""
    levels = np.round(palette.astype(np.float32) / 17.0)
    return (levels * 17).clip(0, 255).astype(np.uint8)


def unique_colors(palette: np.ndarray) -> np.ndarray:
    """Remove duplicate colours while preserving the original order."""
    _, idx = np.unique(palette, axis=0, return_index=True)
    return palette[np.sort(idx)]


def sort_by_luminance(palette: np.ndarray) -> np.ndarray:
    lum = palette.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return palette[np.argsort(lum, kind="stable")]


def _pixels(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1, 3)


def color_histogram(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (colors, counts) sorted by count descending."""
    px = _pixels(img).astype(np.uint32)
    keys = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    uniq, counts = uniq[order], counts[order]
    colors = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1)
    return colors.astype(np.uint8), counts


# ---------------------------------------------------------------------------
# Extraction algorithms
# ---------------------------------------------------------------------------


def _kmeans(
    colors: np.ndarray,
    weights: np.ndarray,
    k: int,
    iterations: int = 24,
    seed: int = 0,
) -> np.ndarray:
    """Weighted k-means (k-means++ init) over unique colours."""
    x = colors.astype(np.float32)
    w = weights.astype(np.float64)
    n = len(x)
    if n <= k:
        return colors.copy()

    rng = np.random.default_rng(seed)
    # k-means++ initialisation, weighted by pixel count.
    centers = np.empty((k, 3), dtype=np.float32)
    first = rng.choice(n, p=w / w.sum())
    centers[0] = x[first]
    d2 = ((x - centers[0]) ** 2).sum(axis=1, dtype=np.float64)
    for i in range(1, k):
        prob = d2 * w
        total = prob.sum()
        if total <= 0:
            centers[i:] = centers[0]
            break
        idx = rng.choice(n, p=prob / total)
        centers[i] = x[idx]
        d2 = np.minimum(d2, ((x - centers[i]) ** 2).sum(axis=1, dtype=np.float64))

    x_sq = (x ** 2).sum(axis=1)[:, None]
    for _ in range(iterations):
        dist = x_sq - 2.0 * x @ centers.T + (centers ** 2).sum(axis=1)[None, :]
        labels = dist.argmin(axis=1)
        sums = np.zeros((k, 3), dtype=np.float64)
        for c in range(3):
            sums[:, c] = np.bincount(labels, weights=x[:, c] * w, minlength=k)
        tot = np.bincount(labels, weights=w, minlength=k)
        moved = tot > 0
        new_centers = centers.copy()
        new_centers[moved] = (sums[moved] / tot[moved, None]).astype(np.float32)
        # Re-seed empty clusters at the worst represented colours.
        empty = np.flatnonzero(~moved)
        if len(empty):
            err = dist[np.arange(n), labels] * w
            worst = np.argsort(-err)[: len(empty)]
            new_centers[empty[: len(worst)]] = x[worst]
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < 0.5:
            break
    return np.clip(np.round(centers), 0, 255).astype(np.uint8)


def extract_kmeans(img: Image.Image, n: int, max_unique: int = 16000) -> np.ndarray:
    colors, counts = color_histogram(img)
    if len(colors) > max_unique:
        # Bin to 5 bits per channel (weighted mean per bin) so clustering
        # stays fast on photos with hundreds of thousands of colours.
        coarse = (colors.astype(np.uint32) >> 3)
        keys = (coarse[:, 0] << 10) | (coarse[:, 1] << 5) | coarse[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)
        w = np.bincount(inv, weights=counts).astype(np.float64)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, colors.astype(np.float64) * counts[:, None])
        colors = np.round(sums / w[:, None]).astype(np.uint8)
        counts = w
    return _kmeans(colors, counts, n)


def extract_median_cut(img: Image.Image, n: int) -> np.ndarray:
    q = img.convert("RGB").quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    used = sorted(c for _, c in q.getcolors(256) or [])
    pal = np.array(q.getpalette()[: 3 * 256], dtype=np.uint8).reshape(-1, 3)
    return pal[used] if used else pal[:n]


def extract_exact(img: Image.Image, n: int) -> np.ndarray:
    colors, _ = color_histogram(img)
    return colors[:n]


EXTRACT_METHODS = {
    "auto": "自動判定",
    "exact": "実色 (出現頻度順)",
    "kmeans": "k-means クラスタリング",
    "mediancut": "メディアンカット",
}


def extract_palette(
    img: Image.Image,
    n: int,
    method: str = "auto",
    snap_4096: bool = False,
    coverage: float = 0.97,
) -> np.ndarray:
    """Extract up to ``n`` representative colours from ``img``.

    ``auto`` is tuned for screenshots of retro machines: when the image only
    contains a handful of distinct colours (or the ``n`` most frequent colours
    cover nearly every pixel, e.g. a screenshot with slight scaling noise) the
    exact colours are used; otherwise k-means clustering is applied.
    """
    n = max(1, min(256, int(n)))
    if method == "auto":
        colors, counts = color_histogram(img)
        if len(colors) <= n or counts[:n].sum() >= coverage * counts.sum():
            pal = colors[:n]
        else:
            pal = extract_kmeans(img, n)
    elif method == "exact":
        pal = extract_exact(img, n)
    elif method == "mediancut":
        pal = extract_median_cut(img, n)
    else:
        pal = extract_kmeans(img, n)

    if snap_4096:
        pal = snap_to_pc98(pal)
    return sort_by_luminance(unique_colors(pal))


# ---------------------------------------------------------------------------
# Palette files (GIMP .gpl / JASC .pal / Photoshop .act)
# ---------------------------------------------------------------------------


def save_palette(palette: np.ndarray, path: str | Path, name: str = "Retro98") -> None:
    path = Path(path)
    ext = path.suffix.lower()
    rows = [tuple(int(v) for v in c) for c in palette]
    if ext == ".gpl":
        lines = ["GIMP Palette", f"Name: {name}", "Columns: 16", "#"]
        lines += [f"{r:3d} {g:3d} {b:3d}\t#{r:02X}{g:02X}{b:02X}" for r, g, b in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif ext == ".pal":
        lines = ["JASC-PAL", "0100", str(len(rows))]
        lines += [f"{r} {g} {b}" for r, g, b in rows]
        path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    elif ext == ".act":
        data = bytearray(768)
        flat = bytes(v for c in rows[:256] for v in c)
        data[: len(flat)] = flat
        data += len(rows[:256]).to_bytes(2, "big") + b"\xff\xff"
        path.write_bytes(bytes(data))
    else:
        raise ValueError(f"未対応のパレット形式です: {ext}")


def load_palette(path: str | Path) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    colors: list[tuple[int, int, int]] = []
    if ext == ".act":
        data = path.read_bytes()
        count = 256
        if len(data) >= 772:
            count = int.from_bytes(data[768:770], "big") or 256
        for i in range(min(count, len(data[:768]) // 3)):
            colors.append(tuple(data[i * 3: i * 3 + 3]))
    else:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not text:
            raise ValueError("パレットファイルが空です")
        if text[0].strip() == "JASC-PAL":
            body = text[3:]
        elif text[0].strip() == "GIMP Palette":
            body = text[1:]
        else:
            raise ValueError("パレット形式を認識できません (.gpl / .pal / .act)")
        for line in body:
            line = line.strip()
            if not line or line.startswith("#") or ":" in line.split()[0]:
                continue
            parts = line.split()
            try:
                r, g, b = (int(p) for p in parts[:3])
            except ValueError:
                continue
            colors.append((r, g, b))
    if not colors:
        raise ValueError("パレットに色が含まれていません")
    return np.clip(np.array(colors[:256], dtype=np.int32), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ramp palettes (style conversion)
# ---------------------------------------------------------------------------
#
# 16-colour CG of the PC-98 era was painted with a few colour "ramps": for each
# hue (skin, hair, sky, ...) 2-5 shades from dark to light, plus a grey ramp
# that also supplies black outlines and white highlights.  A k-means palette
# instead puts colours wherever the pixel mass is, which gives muddy averages
# and no clean light/dark steps to dither between.


SKIN_HUE_MAX = 45 / 360


@dataclass
class RampPalette:
    """Palette plus its ramp structure.

    ``ramps`` lists, for each ramp, palette indices ordered dark -> light.
    Every palette index belongs to exactly one ramp.
    """

    palette: np.ndarray
    ramps: list[list[int]] = field(default_factory=list)


def _binned_histogram(img: Image.Image, max_unique: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    colors, counts = color_histogram(img)
    counts = counts.astype(np.float64)
    if len(colors) > max_unique:
        coarse = colors.astype(np.uint32) >> 3
        keys = (coarse[:, 0] << 10) | (coarse[:, 1] << 5) | coarse[:, 2]
        uniq, inv = np.unique(keys, return_inverse=True)
        w = np.bincount(inv, weights=counts)
        sums = np.zeros((len(uniq), 3))
        np.add.at(sums, inv, colors.astype(np.float64) * counts[:, None])
        colors = sums / w[:, None]
        counts = w
    return np.asarray(colors, np.float64), counts


def _hue_peaks(h: np.ndarray, w: np.ndarray, bins: int = 36) -> list[tuple[float, float]]:
    """Local maxima of the smoothed, weighted hue histogram: ``[(hue, share)]`` by share."""
    if len(h) == 0 or w.sum() <= 0:
        return []
    hist = np.bincount((h * bins).astype(int) % bins, weights=w, minlength=bins)
    sm = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    # Share of a peak = weight of its bin and both neighbours / total weight.
    share = (hist + np.roll(hist, 1) + np.roll(hist, -1)) / hist.sum()
    peaks = [((b + 0.5) / bins, float(share[b])) for b in range(bins)
             if sm[b] > 0 and sm[b] >= sm[b - 1] and sm[b] > sm[(b + 1) % bins]]
    return sorted(peaks, key=lambda p: -p[1])


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _hue_kmeans(h: np.ndarray, w: np.ndarray, k: int, iterations: int = 30,
                init: list[float] | None = None) -> np.ndarray:
    """Weighted 1-D k-means on the hue circle; returns labels.

    ``init`` gives the starting centres (otherwise histogram peaks).
    """
    if init is not None:
        k = len(init)
    # Initialise from the peaks of a smoothed 36-bin hue histogram.
    bins = 36
    hist = np.bincount((h * bins).astype(int) % bins, weights=w, minlength=bins)
    sm = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    centers: list[float] = []
    for b in np.argsort(-sm):
        c = (b + 0.5) / bins
        if all(min(abs(c - o) % 1.0, 1 - abs(c - o) % 1.0) > 1.5 / bins for o in centers):
            centers.append(c)
        if len(centers) == k:
            break
    while len(centers) < k:
        centers.append((centers[-1] + 0.5 / k) % 1.0 if centers else 0.0)
    if init is not None:
        centers = list(init)
    ang = h * 2 * np.pi
    vx, vy = np.cos(ang), np.sin(ang)
    c = np.array(centers)
    for _ in range(iterations):
        d = np.abs(h[:, None] - c[None, :]) % 1.0
        d = np.minimum(d, 1 - d)
        labels = d.argmin(axis=1)
        sx = np.bincount(labels, weights=vx * w, minlength=k)
        sy = np.bincount(labels, weights=vy * w, minlength=k)
        new = np.where((sx == 0) & (sy == 0), c, (np.arctan2(sy, sx) / (2 * np.pi)) % 1.0)
        if np.abs(new - c).max() < 1e-4:
            c = new
            break
        c = new
    d = np.abs(h[:, None] - c[None, :]) % 1.0
    return np.minimum(d, 1 - d).argmin(axis=1)


def _auto_hue_count(h: np.ndarray, w: np.ndarray, max_ramps: int, min_share: float = 0.06) -> int:
    """Number of distinct hue peaks holding at least ``min_share`` of the weight."""
    bins = 24
    hist = np.bincount((h * bins).astype(int) % bins, weights=w, minlength=bins)
    hist = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    total = hist.sum()
    if total <= 0:
        return 0
    peaks = (hist >= np.roll(hist, 1)) & (hist > np.roll(hist, -1)) & (hist / total >= min_share)
    return int(np.clip(peaks.sum(), 1, max_ramps))


def _allocate_steps(weights: np.ndarray, spreads: np.ndarray, total: int,
                    lo: int = 2, hi: int = 6) -> np.ndarray:
    """Split ``total`` colours over ramps, favouring heavy, high-contrast ramps."""
    n = len(weights)
    score = weights * (0.3 + spreads)
    score = score / score.sum() if score.sum() > 0 else np.full(n, 1.0 / n)
    lo = min(lo, max(1, total // n))
    steps = np.full(n, lo)
    remaining = total - steps.sum()
    while remaining > 0:
        # Give the next colour to the ramp with the largest deficit.
        deficit = score * total - steps
        deficit[steps >= hi] = -np.inf
        if not np.isfinite(deficit).any():
            break
        steps[int(deficit.argmax())] += 1
        remaining -= 1
    return steps


def _ramp_colors(cols: np.ndarray, w: np.ndarray, steps: int, spacing: float = 0.5,
                 min_lum_gap: float = 10.0) -> np.ndarray:
    """``steps`` shades of one ramp.

    The ramp's luminance range is cut into ``steps`` bands whose edges are a
    blend of equal-pixel-weight quantiles and equal luminance spacing
    (``spacing`` = weight of the latter).  Pure quantiles bunch shades where
    the pixel mass is (muddy, near-duplicate steps); even spacing gives the
    clear light/dark steps of hand-made ramps.  Each shade is the weighted mean
    colour of its band.  Shades closer than ``min_lum_gap`` in luminance are
    merged: on a nearly uniform area (a flat face) they would only turn noise
    into speckles.
    """
    lum = cols @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum, kind="stable")
    cols, w, lum = cols[order], w[order], lum[order]
    cw = np.cumsum(w) / w.sum()
    qs = np.arange(1, steps) / steps
    q_edges = lum[np.minimum(np.searchsorted(cw, qs), len(lum) - 1)]
    lo = lum[min(np.searchsorted(cw, 0.02), len(lum) - 1)]
    hi = lum[min(np.searchsorted(cw, 0.98), len(lum) - 1)]
    lin_edges = lo + (hi - lo) * qs
    edges = (1 - spacing) * q_edges + spacing * lin_edges
    band = np.searchsorted(edges, lum, side="right")
    out, ow, ol = [], [], []
    for b in range(steps):
        m = band == b
        if not m.any():
            continue
        col = (cols[m] * w[m, None]).sum(0) / w[m].sum()
        cl = float(col @ np.array([0.299, 0.587, 0.114]))
        if out and cl - ol[-1] < min_lum_gap:
            tw = ow[-1] + w[m].sum()
            out[-1] = (out[-1] * ow[-1] + col * w[m].sum()) / tw
            ow[-1] = tw
            ol[-1] = float(out[-1] @ np.array([0.299, 0.587, 0.114]))
            continue
        out.append(col)
        ow.append(w[m].sum())
        ol.append(cl)
    return np.array(out)


def extract_ramp_palette(
    img: Image.Image,
    n_colors: int = 16,
    n_ramps: int = 0,
    steps: int = 0,
    snap_4096: bool = True,
    gray_chroma: float = 28.0,
    max_ramps: int = 6,
    hue_separation: bool = False,
    reserve_skin: bool = False,
    min_hue_share: float = 0.01,
    hue_sep_deg: float = 35.0,
) -> RampPalette:
    """Build a palette made of light->dark ramps per hue.

    * Pixels with chroma below ``gray_chroma`` form an achromatic ramp (only
      when they make up a noticeable share of the image).
    * The rest is clustered on the hue circle into ``n_ramps`` groups
      (``0`` = automatic from the hue histogram).
    * ``steps`` shades are assigned to every ramp (``0`` = distribute the
      ``n_colors`` budget by pixel share and luminance range, 2-6 per ramp).
    * Each shade is the mean colour of an equal-weight luminance band of that
      hue group, then optionally rounded to the PC-98 4096-colour space.

    Counting ramps by pixel share alone loses small but important hues (a red
    tie on a blue picture, faces on a large green garment).  Two options fix
    this (used by the style pipeline):

    * ``reserve_skin``: skin-coloured pixels (hue 0-45°; saturation 0.1-0.6
      at value ≥ 0.45, plus their warm shadows up to saturation 0.75 at value
      0.2-0.45) holding at least 0.3 % of the image get their own ramp of at
      least 3 shades, so faces are not mapped to a grey or unrelated ramp.
    * ``hue_separation``: additional ramps for hue peaks holding at least
      ``min_hue_share`` of the chromatic weight that lie ``hue_sep_deg`` or
      more from every other hue ramp (the skin ramp excluded), and an "accent" ramp when a hue group mixes
      clearly different saturations (vivid red cords inside dull dark-red
      hair).  Extra ramps are added while the colour budget allows two shades
      each.
    """
    n_colors = max(2, min(256, int(n_colors)))
    cols, w = _binned_histogram(img)
    c = cols.max(1) - cols.min(1)
    gray = c < gray_chroma
    gray_share = w[gray].sum() / w.sum()
    has_gray = gray.any() and (gray_share >= 0.04 or gray.all())
    if not has_gray:
        gray = np.zeros_like(gray)

    hsv = rgb_to_hsv(cols)
    hue = hsv[:, 0].astype(np.float64)
    sat, val = hsv[:, 1], hsv[:, 2]
    chrom = ~gray
    total_w = w.sum()
    hw = w * (c + 1)                         # hue weight: chroma emphasises clear hues

    skin = np.zeros_like(chrom)
    if reserve_skin:
        # Light skin and its warm shadows (incl. brown hair, which shares the
        # warm ramp in hand-made palettes).
        cand = chrom & (hue <= SKIN_HUE_MAX) & (sat >= 0.1) & (
            ((sat <= 0.6) & (val >= 0.45)) | ((sat <= 0.75) & (val >= 0.2) & (val < 0.45)))
        if w[cand].sum() / total_w >= 0.003:
            skin = cand
    rest = chrom & ~skin

    max_chrom = max(1, max_ramps - int(has_gray))
    if not rest.any():
        k = 0
    elif n_ramps > 0:
        k = max(1, min(int(n_ramps) - int(has_gray), max_chrom))
    else:
        k = _auto_hue_count(hue[rest], hw[rest], max_chrom)
    if steps > 0:
        # Fixed number of steps: drop ramps that do not fit in the budget.
        fit = max(1, n_colors // int(steps))
        k = min(k, max(0, fit - int(has_gray) - int(skin.any()))) if rest.any() else 0
        if k == 0 and rest.any() and not has_gray:
            k = 1

    # (colours, weights, kind, priority) ; lower priority is dropped first
    groups_k: list[tuple[np.ndarray, np.ndarray, str, float]] = []
    if has_gray:
        groups_k.append((cols[gray], w[gray], "gray", 3.0))
    if skin.any():
        groups_k.append((cols[skin], w[skin], "skin", 2.5))
    if k > 0:
        peaks = _hue_peaks(hue[rest], hw[rest])
        centers: list[float] = []
        for hp, _ in peaks:
            if len(centers) == k:
                break
            if all(_hue_dist(hp, o) >= 25 / 360 for o in centers):
                centers.append(hp)
        n_major = len(centers)
        if hue_separation:
            # Skin is not an anchor: it is defined by saturation as well, so a
            # vivid red / orange next to the skin hue is a different colour.
            anchors = list(centers)
            for hp, share in peaks:
                if share >= min_hue_share and all(_hue_dist(hp, o) >= hue_sep_deg / 360 for o in anchors):
                    centers.append(hp)
                    anchors.append(hp)
        if centers:
            labels = _hue_kmeans(hue[rest], hw[rest], len(centers), iterations=8, init=centers)
            cc, ww, ss, vv = cols[rest], w[rest], sat[rest], val[rest]
            for i in range(len(centers)):
                m = labels == i
                if not m.any():
                    continue
                # Priority ≥ 2: main hues (kept); < 2: optional, ranked below.
                prio = 2.0 if i < n_major else 1.5
                gc, gw, gs, gv = cc[m], ww[m], ss[m], vv[m]
                if hue_separation:
                    vivid = (gs >= 0.55) & (gv >= 0.35)
                    vs = gw[vivid].sum()
                    if vivid.any() and (~vivid).any() and 0.03 <= vs / gw.sum() <= 0.85 \
                            and vs / total_w >= 0.003 and \
                            np.average(gs[vivid], weights=gw[vivid]) - np.average(gs[~vivid], weights=gw[~vivid]) >= 0.3:
                        groups_k.append((gc[~vivid], gw[~vivid], "hue", prio))
                        groups_k.append((gc[vivid], gw[vivid], "accent", 1.0 + vs / total_w))
                        continue
                groups_k.append((gc, gw, "hue", prio + gw.sum() / total_w))

    # Optional ramps (distant small hues, accents) are ranked by how much
    # colour would be lost without them: share × distance of their colours to
    # the nearest colour of every other ramp.  A vivid red inside dull dark-red
    # hair loses a lot; gold next to skin / brown ramps loses little.
    optional = [i for i, g in enumerate(groups_k) if g[3] < 2.0]
    if optional:
        def sample(cs, ws, n):
            if len(cs) <= n:
                return cs.astype(np.float32), ws
            sel = np.argsort(-ws)[:n]
            return cs[sel].astype(np.float32), ws[sel]
        for i in optional:
            ci, wi = sample(groups_k[i][0], groups_k[i][1], 300)
            others = [sample(g[0], g[1], 400)[0] for j, g in enumerate(groups_k) if j != i]
            if not others:
                continue
            oc = np.concatenate(others)
            d = np.sqrt(((ci[:, None, :] - oc[None]) ** 2).sum(2)).min(1)
            err = float(np.average(d, weights=wi))
            sal = groups_k[i][1].sum() / total_w * err
            groups_k[i] = groups_k[i][:3] + (1.0 + sal / (1.0 + sal),)

    # Respect the budget.  Minimum shades: skin 3 (light / mid / shadow), a
    # large grey area 3 (white, grey, dark), other ramps 2 (1 if steps == 1).
    def min_shades(g) -> int:
        if steps > 0:
            return int(steps)
        if g[2] == "skin" or (g[2] == "gray" and g[1].sum() / total_w >= 0.15):
            return 3
        return 2
    while len(groups_k) > 1 and sum(min(min_shades(g), n_colors) for g in groups_k) > n_colors:
        droppable = [i for i, g in enumerate(groups_k) if g[2] not in ("gray", "skin")] \
            or list(range(len(groups_k)))
        groups_k.pop(min(droppable, key=lambda i: groups_k[i][3]))
    groups = [(g[0], g[1]) for g in groups_k]
    kinds = [g[2] for g in groups_k]
    mins = np.array([min_shades(g) for g in groups_k])

    weights = np.array([g[1].sum() for g in groups])
    weights = weights / weights.sum()
    spreads = []
    for gc, gw in groups:
        lum = gc @ np.array([0.299, 0.587, 0.114])
        spreads.append((np.percentile(lum, 95) - np.percentile(lum, 5)) / 255.0 if len(lum) > 1 else 0.0)
    if steps > 0:
        alloc = np.full(len(groups), int(steps))
        while alloc.sum() > n_colors:
            alloc[int(np.argmax(alloc))] -= 1
    else:
        hi = max(6, -(-n_colors // len(groups)))
        alloc = _allocate_steps(weights, np.array(spreads), n_colors, hi=hi)
        # Enforce the minimum shades by taking from ramps above their minimum.
        for gi in range(len(alloc)):
            while alloc[gi] < mins[gi]:
                spare = np.where(np.arange(len(alloc)) == gi, -1, alloc - mins)
                donor = int(np.argmax(spare))
                if spare[donor] <= 0:
                    break
                alloc[donor] -= 1
                alloc[gi] += 1

    colors: list[np.ndarray] = []
    ramp_of: list[int] = []
    for gi, ((gc, gw), st) in enumerate(zip(groups, alloc)):
        if st <= 0:
            continue
        rc = _ramp_colors(gc, gw, int(st))
        colors.extend(rc)
        ramp_of.extend([gi] * len(rc))
    pal = np.clip(np.round(np.array(colors)), 0, 255).astype(np.uint8)
    if snap_4096:
        pal = snap_to_pc98(pal)

    # Remove duplicates (snapping can merge shades), keep the first ramp.
    seen: dict[tuple, int] = {}
    final: list[np.ndarray] = []
    ramps: dict[int, list[int]] = {}
    for col, gi in zip(pal, ramp_of):
        key = tuple(int(v) for v in col)
        if key in seen:
            continue
        seen[key] = len(final)
        ramps.setdefault(gi, []).append(len(final))
        final.append(col)
    pal = np.array(final, dtype=np.uint8)
    lum = pal.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    ramp_list = [sorted(idx, key=lambda i: lum[i]) for idx in ramps.values()]
    return RampPalette(pal, ramp_list)


def infer_ramps(palette: np.ndarray, gray_chroma: float = 28.0, hue_tol: float = 1 / 12,
                extend_with_gray: bool = False) -> list[list[int]]:
    """Group an arbitrary palette into ramps (by hue, greys separately).

    Used when a preset or transferred palette is combined with the selective
    dither, which needs to know which colours are neighbours on a ramp.

    ``extend_with_gray`` adds the darkest grey below and the lightest grey
    above every chromatic ramp (when they extend it).  Fixed palettes with
    one shade per hue (digital 8 colours) need this so that dark / light
    tones can be dithered towards black and white, as was done by hand.
    """
    pal = np.asarray(palette, dtype=np.float32)
    if len(pal) == 0:
        return []
    c = pal.max(1) - pal.min(1)
    lum = pal @ np.array([0.299, 0.587, 0.114], np.float32)
    hue = rgb_to_hsv(pal)[:, 0]
    gray = np.flatnonzero(c < gray_chroma)
    chrom = [int(i) for i in np.argsort(hue) if c[i] >= gray_chroma]
    groups: list[list[int]] = []
    for i in chrom:
        if groups:
            prev = groups[-1][-1]
            d = abs(hue[i] - hue[prev]) % 1.0
            if min(d, 1 - d) <= hue_tol:
                groups[-1].append(i)
                continue
        groups.append([i])
    # Merge the wrap-around group (reds near hue 0 and 1).
    if len(groups) > 1:
        d = abs(hue[groups[0][0]] - hue[groups[-1][-1]]) % 1.0
        if min(d, 1 - d) <= hue_tol:
            groups[0] = groups.pop() + groups[0]
    if len(gray):
        groups.append([int(i) for i in gray])
    ramps = [sorted(g, key=lambda i: lum[i]) for g in groups]
    if extend_with_gray and len(gray):
        lo, hi = int(gray[np.argmin(lum[gray])]), int(gray[np.argmax(lum[gray])])
        out = []
        for r in ramps:
            if r[0] in gray:
                out.append(r)
                continue
            r = list(r)
            if lum[lo] < lum[r[0]]:
                r.insert(0, lo)
            if lum[hi] > lum[r[-1]]:
                r.append(hi)
            out.append(r)
        ramps = out
    return ramps
