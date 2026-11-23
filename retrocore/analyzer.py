"""Style profile analysis of reference CG.

``analyze_image`` normalises a (possibly scaled) screenshot to its native
pixel grid and palette (see ``screenshot.py``) and measures:

* colours  – palette (4-bit ``#RGB``), shares, saturation / value distribution
* ramps    – hue groups and number of light/dark steps
* dither   – share of dithered pixels, pattern classes (checker / tile / other),
             which colour pairs are mixed and where (luminance / hue bands)
* outline  – dark pixels on strong edges, outline colour, thickness,
             hand-made anti-aliasing (intermediate shades next to outlines)
* fill     – size distribution of same-colour connected regions, isolated pixels
* resolution – scale of the screenshot, native size, 400/200 lines

``analyze_images`` combines several images (e.g. one software title) into a
group profile.  The JSON layout is documented in ``docs/profile_schema.md``.
NumPy + Pillow only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .color import LUMA, rgb_to_hsv
from .palette import infer_ramps
from .screenshot import GridInfo, NativeImage, normalize_screenshot

SCHEMA = "retro98-style-profile"
VERSION = 1

DITHER_CLASSES = ["checker", "tile2x2", "stripe", "tile4x4", "ordered_mix", "irregular"]
DITHER_GROUPS = {"checker": ["checker"], "tile": ["tile2x2", "stripe", "tile4x4", "ordered_mix"],
                 "other": ["irregular"]}
LUM_BANDS = [0, 51, 102, 153, 204, 256]
HUE_BANDS = ["red", "yellow", "green", "cyan", "blue", "magenta", "gray"]
SIZE_BINS = [1, 2, 3, 5, 9, 17, 65, 257, 1025, 4097]   # lower bounds
SIZE_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-64", "65-256", "257-1024",
               "1025-4096", "4097+"]
# Colour pairs mixed by dither: relation of the two colours and their distance.
MIX_RELATIONS = ["same_ramp_adjacent", "same_ramp_far", "cross_ramp"]
MIX_LUM_EDGES = [0, 16, 32, 48, 64, 96, 128, 256]       # |Δ luminance| bins (0-255)
MIX_HUE_EDGES = [0, 15, 30, 60, 90, 181]                # |Δ hue| bins (degrees)
MIX_GRAY_CHROMA = 28                                    # = infer_ramps' grey threshold


def hex4(rgb) -> str:
    """``#RGB`` with PC-98 4-bit channels."""
    r, g, b = (int(round(int(v) / 17.0)) for v in rgb)
    return f"#{r:X}{g:X}{b:X}"


def _r(x, n=4):
    return round(float(x), n)


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------


def _shift(a: np.ndarray, dy: int, dx: int, fill=-1) -> np.ndarray:
    """out[y, x] = a[y+dy, x+dx]; outside -> ``fill``."""
    h, w = a.shape
    out = np.full_like(a, fill)
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    yd = slice(max(0, dy), min(h, h + dy))
    xd = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = a[yd, xd]
    return out


def _spread(mask: np.ndarray, dy: range, dx: range) -> np.ndarray:
    """OR of ``mask`` shifted so that an anchored block marks all its pixels."""
    out = np.zeros_like(mask)
    for y in dy:
        for x in dx:
            out |= _shift(mask, -y, -x, False)
    return out


def run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Length of the run of True values each pixel belongs to (0 for False)."""
    m = np.moveaxis(np.asarray(mask, bool), axis, -1)
    h, w = m.shape
    pad = np.zeros((h, 1), bool)
    p = np.concatenate([pad, m, pad], axis=1).astype(np.int8)
    d = np.diff(p, axis=1)
    ys, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    lengths = ends - starts
    # Paint each run with its length via a cumulative trick.
    marks = np.zeros((h, w + 1), np.int32)
    np.add.at(marks, (ys, starts), lengths)
    np.add.at(marks, (ys, ends), -lengths)
    out = np.cumsum(marks, axis=1)[:, :w]
    return np.moveaxis(out, -1, axis)


def label_components(idx: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """4-connected labels of equal values; invalid pixels get -1.

    NumPy only (no SciPy): horizontal runs of equal, valid pixels are
    numbered first, then runs touching vertically are merged by min-label
    propagation with pointer jumping on the (much smaller) run graph.
    """
    h, w = idx.shape
    n = h * w
    flat = idx.ravel()
    ok = np.ones(n, bool) if valid is None else valid.ravel()
    right = (flat[:-1] == flat[1:]) & ok[:-1] & ok[1:]
    right[np.arange(1, h) * w - 1] = False          # no wrap-around between rows
    start = np.ones(n, bool)
    start[1:] = ~right
    run = np.cumsum(start) - 1                      # run id of every pixel
    down = np.flatnonzero((flat[:-w] == flat[w:]) & ok[:-w] & ok[w:])
    edges = np.unique(np.stack([run[down], run[down + w]], 1), axis=0) if len(down) else \
        np.zeros((0, 2), np.int64)
    u, v = edges[:, 0], edges[:, 1]
    lab = np.arange(int(run[-1]) + 1, dtype=np.int64)
    while len(u):
        old = lab.copy()
        m = np.minimum(lab[u], lab[v])
        np.minimum.at(lab, u, m)
        np.minimum.at(lab, v, m)
        for _ in range(8):
            lab = lab[lab]
        if np.array_equal(lab, old):
            break
    out = lab[run].reshape(h, w)
    out[~ok.reshape(h, w)] = -1
    return out


# ---------------------------------------------------------------------------
# Dither classification
# ---------------------------------------------------------------------------


def classify_dither(idx: np.ndarray, lum: np.ndarray | None = None,
                    max_pair_lum: float = 110.0, radius: int = 2, tol: float = 0.15) -> np.ndarray:
    """Per-pixel dither class: 0 = none, ``k`` = ``DITHER_CLASSES[k-1]``.

    Works on the fractions, over a (2r+1)² window, of pixels equal to their
    neighbour at a given shift.  Using window statistics instead of exact
    pattern matching tolerates the occasional wrong pixel left by screenshot
    resampling.  With ``F(·)`` the window mean and ``tol`` the tolerance:

    * checker : F(=right) ≈ 0, F(=down) ≈ 0, F(=diagonal) ≈ 1, period 2
    * stripe  : period 2 in both axes, equal along one axis only
    * tile2x2 : period 2 in both axes otherwise (25 / 75 % dot grids)
    * tile4x4 : period 4 in both axes (sparse dots, Bayer-like 1/16 levels)
    * dotted regions: two-colour 3x3 windows with dot-like pixels (differs
      from ≥ 3 of 4 neighbours) occurring densely, colours not too far apart
      in luminance (excludes outlines / text).  They are split into
      ordered_mix — clear even-period structure (equal at shift 2 much more
      often than at shift 3), i.e. ordered patterns whose density changes
      within a few pixels (checker / tile gradients) — and irregular (random
      dots, error-diffusion-like).

    Flat areas (window nearly uniform) never count as dither.
    """
    from .color import box_mean

    a = np.asarray(idx).astype(np.int32)
    out = np.zeros(a.shape, np.int8)

    def frac(dy, dx, r=radius):
        nb = _shift(a, dy, dx)
        eq = np.where(nb >= 0, (nb == a), True).astype(np.float32)
        return box_mean(eq, r)

    hi, lo = 1.0 - tol, tol
    f1x, f1y, fd = frac(0, 1), frac(1, 0), frac(1, 1)
    f2x, f2y = frac(0, 2), frac(2, 0)
    nonflat = np.minimum(f1x, f1y) < 0.9
    # Sparse period-4 dots also match at shift 2 most of the time (7/8), so
    # prefer period 4 when it fits clearly better, measured on a wider window.
    r4 = 2 * radius
    f4m = np.minimum(frac(0, 4, r4), frac(4, 0, r4))
    f2m = np.minimum(frac(0, 2, r4), frac(2, 0, r4))
    nonflat4 = np.minimum(frac(0, 1, r4), frac(1, 0, r4)) < 0.97
    tile4 = (f4m >= hi) & (f2m < f4m - 0.05) & nonflat4
    p2 = (f2x >= hi) & (f2y >= hi) & nonflat & ~tile4
    checker = p2 & (f1x <= lo) & (f1y <= lo) & (fd >= hi)
    stripe = p2 & ~checker & (((f1x >= hi) & (f1y <= lo)) | ((f1y >= hi) & (f1x <= lo)))
    tile2 = p2 & ~checker & ~stripe
    for k, m in ((1, checker), (2, tile2), (3, stripe), (4, tile4)):
        out[(out == 0) & m] = k

    # Irregular dots
    nb = np.stack([_shift(a, y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    valid = (nb >= 0).all(0)
    mn = np.where(nb >= 0, nb, 1 << 20).min(0)
    mx = nb.max(0)
    two3 = valid & ((nb == mn) | (nb == mx)).all(0) & (mn != mx)
    diff4 = sum((a != _shift(a, y, x)) & (_shift(a, y, x) >= 0)
                for y, x in ((0, 1), (0, -1), (1, 0), (-1, 0)))
    dots = two3 & (diff4 >= 3)
    if lum is not None:
        n = len(lum)
        dots &= np.abs(np.take(lum, mx.clip(0, n - 1)) - np.take(lum, mn.clip(0, n - 1))) < max_pair_lum
    dense = box_mean(dots.astype(np.float32), radius) >= 0.12
    dotted = dense & nonflat & _spread(dots, range(-1, 2), range(-1, 2))
    f2 = (f2x + f2y) / 2
    f3 = (frac(0, 3) + frac(3, 0)) / 2
    ordered = (f2 >= 0.55) & (f2 - f3 >= 0.25)
    out[(out == 0) & dotted & ordered] = 5
    out[(out == 0) & dotted] = 6
    return out


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _colors_section(idx, pal, counts) -> dict:
    total = counts.sum()
    order = np.argsort(-counts, kind="stable")
    hsv = rgb_to_hsv(pal)
    w = counts / total
    sat_hist = np.bincount(np.minimum((hsv[:, 1] * 8).astype(int), 7), weights=w, minlength=8)
    val_hist = np.bincount(np.minimum((hsv[:, 2] * 8).astype(int), 7), weights=w, minlength=8)
    sat_mean = float((hsv[:, 1] * w).sum())
    val_mean = float((hsv[:, 2] * w).sum())
    return {
        "n_colors": int((counts > 0).sum()),
        "top16_coverage": _r(np.sort(counts)[::-1][:16].sum() / total),
        "is_pc98_4096": bool((pal % 17 == 0).all()),
        "palette": [{"hex": hex4(pal[i]), "rgb": [int(v) for v in pal[i]],
                     "share": _r(counts[i] / total)} for i in order if counts[i] > 0],
        "saturation": {"hist": [_r(v) for v in sat_hist], "mean": _r(sat_mean),
                       "std": _r(np.sqrt((w * (hsv[:, 1] - sat_mean) ** 2).sum()))},
        "value": {"hist": [_r(v) for v in val_hist], "mean": _r(val_mean),
                  "std": _r(np.sqrt((w * (hsv[:, 2] - val_mean) ** 2).sum()))},
    }


def _ramps_section(pal, counts, ramps) -> dict:
    total = counts.sum()
    hsv = rgb_to_hsv(pal)
    items = []
    for r in sorted(ramps, key=lambda r: -counts[r].sum()):
        c = pal[r].astype(int)
        achro = bool((c.max(1) - c.min(1)).max() < 28)
        ang = hsv[r, 0] * 2 * np.pi
        wt = counts[r] + 1e-9
        hue = (np.degrees(np.arctan2((np.sin(ang) * wt).sum(), (np.cos(ang) * wt).sum())) % 360)
        items.append({
            "hue_deg": None if achro else _r(hue, 1),
            "achromatic": achro,
            "steps": len(r),
            "colors": [hex4(pal[i]) for i in r],
            "share": _r(counts[r].sum() / total),
        })
    steps = np.array([it["steps"] for it in items]) if items else np.zeros(1)
    shares = np.array([it["share"] for it in items]) if items else np.ones(1)
    multi = [it for it in items if it["steps"] >= 2]
    return {
        "n_ramps": len(items),
        "n_ramps_multi": len(multi),
        "steps_mean": _r(steps.mean(), 2),
        "steps_weighted": _r((steps * shares).sum() / max(shares.sum(), 1e-9), 2),
        "steps_max": int(steps.max()),
        "items": items,
    }


def _band_rates(values: np.ndarray, dither: np.ndarray, n_bands: int) -> tuple[list, list]:
    tot = np.bincount(values.ravel(), minlength=n_bands).astype(float)
    dit = np.bincount(values.ravel(), weights=dither.ravel().astype(float), minlength=n_bands)
    rate = np.where(tot > 0, dit / np.maximum(tot, 1), 0.0)
    share = dit / max(dit.sum(), 1)
    return [_r(v) for v in rate], [_r(v) for v in share]


def _diff_summary(values: np.ndarray, edges: list, n: int = 2) -> dict:
    """Histogram (share per bin) and mean / median / 90th percentile."""
    if len(values) == 0:
        return {"bin_edges": edges, "hist": [0.0] * (len(edges) - 1),
                "mean": 0.0, "median": 0.0, "p90": 0.0}
    b = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    hist = np.bincount(b, minlength=len(edges) - 1) / len(values)
    return {"bin_edges": edges, "hist": [_r(v) for v in hist], "mean": _r(values.mean(), n),
            "median": _r(np.median(values), n), "p90": _r(np.percentile(values, 90), n)}


def _mix_section(lo: np.ndarray, hi: np.ndarray, pal, lum_pal, same, same_adj,
                 top: int = 12) -> dict:
    """Statistics of the colour pairs mixed by dither (one pair per dithered pixel).

    ``relation`` splits the pairs into adjacent shades of one ramp, other
    shades of one ramp, and colours of different ramps; ``lum_diff`` and
    ``hue_diff`` summarise how far apart the two colours are (hue only for
    pairs of two chromatic colours), overall and for cross-ramp pairs.
    """
    k = len(pal)
    rel = np.where(same_adj, 0, np.where(same, 1, 2))          # k x k relation code
    c = pal.astype(np.int32)
    chroma = c.max(1) - c.min(1)
    hue = rgb_to_hsv(pal)[:, 0] * 360.0
    dl = np.abs(lum_pal[:, None] - lum_pal[None, :])
    dh = np.abs(hue[:, None] - hue[None, :])
    dh = np.minimum(dh, 360.0 - dh)
    chrom2 = (chroma[:, None] >= MIX_GRAY_CHROMA) & (chroma[None, :] >= MIX_GRAY_CHROMA)
    n = len(lo)
    r = rel[lo, hi] if n else np.zeros(0, int)
    rel_share = np.bincount(r, minlength=3) / max(n, 1)
    keys = lo.astype(np.int64) * k + hi
    uk, uc = np.unique(keys, return_counts=True) if n else (np.array([], int), np.array([], int))
    pairs = []
    for i in np.argsort(-uc, kind="stable")[:top]:
        a, b = int(uk[i] // k), int(uk[i] % k)
        pairs.append({"colors": [hex4(pal[a]), hex4(pal[b])], "share": _r(uc[i] / n),
                      "relation": MIX_RELATIONS[int(rel[a, b])],
                      "lum_diff": _r(dl[a, b], 1),
                      "hue_diff": _r(dh[a, b], 1) if chrom2[a, b] else None})

    def diffs(sel):
        lsel, hsel = lo[sel], hi[sel]
        ch = chrom2[lsel, hsel]
        return {"lum_diff": _diff_summary(dl[lsel, hsel], MIX_LUM_EDGES),
                "hue_diff": dict(_diff_summary(dh[lsel, hsel][ch], MIX_HUE_EDGES, 1),
                                 chromatic_share=_r(ch.mean()) if len(ch) else 0.0)}

    every = diffs(np.ones(n, bool))
    return {
        "pair_count": int(n),
        "relation": {name: _r(v) for name, v in zip(MIX_RELATIONS, rel_share)},
        "pairs": pairs,
        "lum_diff": every["lum_diff"],
        "hue_diff": every["hue_diff"],
        "cross_ramp": diffs(r == 2),
    }


def _dither_section(idx, pal, lum_pal, ramps, cls) -> dict:
    total = idx.size
    dmask = cls > 0
    n_d = int(dmask.sum())
    classes = {name: _r((cls == i + 1).sum() / total) for i, name in enumerate(DITHER_CLASSES)}
    groups = {g: _r(sum(classes[c] for c in members)) for g, members in DITHER_GROUPS.items()}
    # Pair = own colour + most relevant different 4-neighbour.
    a = idx.astype(np.int32)
    other = np.full(a.shape, -1, np.int32)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = _shift(a, dy, dx)
        take = (other < 0) & (nb >= 0) & (nb != a)
        other[take] = nb[take]
    sel = dmask & (other >= 0)
    lo = np.minimum(a[sel], other[sel])
    hi = np.maximum(a[sel], other[sel])
    pos = {}
    for ri, r in enumerate(ramps):
        for k, i in enumerate(r):
            pos[i] = (ri, k)
    k = len(pal)
    same_adj = np.zeros((k, k), bool)
    same = np.zeros((k, k), bool)
    for i in range(k):
        for j in range(k):
            if i in pos and j in pos and pos[i][0] == pos[j][0]:
                same[i, j] = True
                same_adj[i, j] = abs(pos[i][1] - pos[j][1]) == 1
    keys = lo * k + hi
    uk, uc = np.unique(keys, return_counts=True) if len(keys) else (np.array([], int), np.array([], int))
    order = np.argsort(-uc)[:8]
    top_pairs = [{"colors": [hex4(pal[u // k]), hex4(pal[u % k])], "share": _r(c / max(len(keys), 1))}
                 for u, c in zip(uk[order], uc[order])]
    # Region tendencies, measured on the local mean colour (3x3) of every pixel.
    rgb = pal[a].astype(np.float32)
    from .color import box_mean
    local = box_mean(rgb, 1).astype(np.float32)
    lum_local = local @ LUMA
    lb = np.clip(np.searchsorted(LUM_BANDS, lum_local, side="right") - 1, 0, len(LUM_BANDS) - 2)
    hsv = rgb_to_hsv(local)
    chroma = local.max(2) - local.min(2)
    hb = np.where(chroma < 24, 6, ((hsv[..., 0] * 6 + 0.5).astype(int) % 6))
    l_rate, l_share = _band_rates(lb, dmask, len(LUM_BANDS) - 1)
    h_rate, h_share = _band_rates(hb, dmask, len(HUE_BANDS))
    return {
        "ratio": _r(n_d / total),
        "classes": classes,
        "groups": groups,
        "pattern_mix": {g: _r(v / max(n_d / total, 1e-9)) for g, v in groups.items()},
        "pair_same_ramp_adjacent": _r(same_adj[lo, hi].mean()) if len(lo) else 0.0,
        "pair_same_ramp": _r(same[lo, hi].mean()) if len(lo) else 0.0,
        "pair_lum_diff_mean": _r(np.abs(lum_pal[hi] - lum_pal[lo]).mean(), 2) if len(lo) else 0.0,
        "top_pairs": top_pairs,
        "by_luminance": {"band_edges": LUM_BANDS, "rate": l_rate, "share": l_share},
        "by_hue": {"bands": HUE_BANDS, "rate": h_rate, "share": h_share},
        "mix": _mix_section(lo, hi, pal, lum_pal, same, same_adj),
    }


def _outline_section(idx, pal, lum_pal, dmask, edge_lum: float = 64.0,
                     dark_lum: float = 60.0) -> dict:
    a = idx.astype(np.int32)
    lum = lum_pal[a]
    darks, edge_pix, pairs_dark = [], np.zeros(a.shape, bool), 0
    n_pairs = 0
    for dy, dx in ((0, 1), (1, 0)):
        nb = _shift(a, dy, dx)
        ok = nb >= 0
        ln = np.where(ok, lum_pal[nb.clip(0)], 0)
        strong = ok & (np.abs(lum - ln) >= edge_lum) & ~dmask
        n_pairs += int(strong.sum())
        dark_val = np.where(lum <= ln, lum, ln)
        dark_idx = np.where(lum <= ln, a, nb)
        dsel = strong & (dark_val <= dark_lum)
        pairs_dark += int(dsel.sum())
        darks.append(dark_idx[dsel])
        # mark the dark side pixel
        side_self = dsel & (lum <= ln)
        edge_pix |= side_self
        side_nb = dsel & (lum > ln)
        edge_pix |= _shift(side_nb, -dy, -dx, False)
    darks = np.concatenate(darks) if darks else np.array([], int)
    if len(darks):
        ci = int(np.bincount(darks).argmax())
    else:
        ci = int(np.argmin(lum_pal))
    out_mask = (a == ci)
    on_edge = out_mask & edge_pix
    # Thickness: shorter of the horizontal / vertical run of the outline colour.
    if on_edge.any():
        th = np.minimum(run_lengths(out_mask, 1), run_lengths(out_mask, 0))[on_edge]
        thick_med = float(np.median(th))
        hist = {"1": _r((th == 1).mean()), "2": _r((th == 2).mean()), "3+": _r((th >= 3).mean())}
    else:
        thick_med, hist = 0.0, {"1": 0.0, "2": 0.0, "3+": 0.0}
    # Anti-aliasing: outline -> b -> c with c much lighter; is b an intermediate?
    aa = direct = 0
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        b = _shift(a, dy, dx)
        c = _shift(a, 2 * dy, 2 * dx)
        base = out_mask & (b >= 0) & (c >= 0) & (b != ci)
        la = lum_pal[ci]
        lb_ = lum_pal[b.clip(0)]
        lc = lum_pal[c.clip(0)]
        delta = lc - la
        cand = base & (delta >= 96)
        inter = cand & (lb_ > la + 0.2 * delta) & (lb_ < lc - 0.2 * delta)
        aa += int(inter.sum())
        direct += int((cand & ~inter & (lb_ >= lc - 0.2 * delta)).sum())
    aa_ratio = aa / max(aa + direct, 1)
    return {
        "dark_edge_ratio": _r(pairs_dark / max(n_pairs, 1)),
        "color": hex4(pal[ci]),
        "color_luminance": _r(lum_pal[ci], 1),
        "outline_pixel_ratio": _r(on_edge.sum() / a.size),
        "thickness": {"median": thick_med, "hist": hist},
        "antialias": {"ratio": _r(aa_ratio), "detected": bool(aa_ratio >= 0.25)},
    }


def _fill_section(idx, dmask) -> dict:
    valid = ~dmask
    lab = label_components(idx, valid)
    ok = lab >= 0
    if not ok.any():
        return {"components": 0, "size_hist": {"bins": SIZE_LABELS, "pixel_share": [0.0] * len(SIZE_LABELS)},
                "median_region_pixels": 0, "flat_ratio": 0.0, "isolated_ratio": 0.0}
    _, inv, sizes = np.unique(lab[ok], return_inverse=True, return_counts=True)
    px_size = sizes[inv]
    bins = np.searchsorted(SIZE_BINS, px_size, side="right") - 1
    share = np.bincount(bins, minlength=len(SIZE_BINS)) / idx.size
    # 8-neighbour isolation (outside dither)
    a = idx.astype(np.int32)
    same = sum((_shift(a, y, x) == a) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0))
    isolated = (same == 0) & valid
    return {
        "components": int(len(sizes)),
        "size_hist": {"bins": SIZE_LABELS, "pixel_share": [_r(v) for v in share]},
        "median_region_pixels": int(np.median(px_size)),
        "flat_ratio": _r((px_size >= 65).sum() / idx.size),
        "isolated_ratio": _r(isolated.sum() / idx.size),
    }


def _resolution_section(grid: GridInfo, idx: np.ndarray, src_size, src_colors) -> tuple[dict, bool]:
    h = idx.shape[0]
    pair_eq = 0.0
    if h >= 4:
        top = idx[0: h - h % 2: 2]
        bot = idx[1: h - h % 2: 2]
        pair_eq = float((top == bot).all(1).mean())
    aspect = grid.scale_y / max(grid.scale_x, 1e-9)
    doubled_rows = pair_eq >= 0.9
    tall = abs(aspect - 2.0) < 0.2          # rows already doubled in the screenshot
    lines = 200 if (doubled_rows or tall) else 400
    return {
        "source_width": int(src_size[0]),
        "source_height": int(src_size[1]),
        "source_colors": int(src_colors),
        "grid": grid.to_dict(),
        "native_width": int(idx.shape[1]),
        "native_height": int(h),
        "pixel_aspect_in_source": _r(aspect),
        "line_pair_equal_ratio": _r(pair_eq),
        "line_doubled": bool(doubled_rows or tall),
        "lines": lines,
    }, doubled_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_native(native: NativeImage, name: str = "", src_size=None) -> dict:
    """Profile of an already normalised (indexed) image."""
    idx, pal = native.indices, native.palette
    res, doubled = _resolution_section(native.grid, idx, src_size or idx.shape[::-1],
                                       native.source_colors)
    if doubled:
        idx = idx[0: idx.shape[0] - idx.shape[0] % 2: 2]
    counts = np.bincount(idx.ravel(), minlength=len(pal)).astype(np.float64)
    lum_pal = pal.astype(np.float32) @ LUMA
    ramps = infer_ramps(pal)
    cls = classify_dither(idx, lum_pal)
    return {
        "name": name,
        "resolution": res,
        "colors": _colors_section(idx, pal, counts),
        "ramps": _ramps_section(pal, counts, ramps),
        "dither": _dither_section(idx, pal, lum_pal, ramps, cls),
        "outline": _outline_section(idx, pal, lum_pal, cls > 0),
        "fill": _fill_section(idx, cls > 0),
    }


def check_reference(img: Image.Image, max_colors: int = 16) -> str | None:
    """Why ``img`` is not an exact PC-98 analog-palette image, or ``None`` if it is.

    An exact reference (e.g. saved from an emulator as BMP / PNG) uses at most
    ``max_colors`` colours, all on the 16-level grid (multiples of 17).  Such
    images are analysed as they are; anything else needs screenshot recovery.
    """
    flat = np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    keys = np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2])
    cols = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1)
    off = int((cols % 17 != 0).any(1).sum())
    problems = []
    if len(cols) > max_colors:
        problems.append(f"{len(cols)} 色 (> {max_colors})")
    if off:
        problems.append(f"16 段階から外れた色 {off} 色")
    return "、".join(problems) or None


def analyze_image(img: Image.Image, name: str = "", max_colors: int = 64,
                  snap_4096: bool = True, strict: bool = False) -> tuple[dict, NativeImage]:
    """Analyse one image; returns ``(stats, native)``.

    Exact indexed images (see ``check_reference``) are used pixel for pixel.
    Other images go through screenshot recovery (``screenshot.py``) unless
    ``strict`` is set, in which case ``ValueError`` is raised.
    """
    if strict:
        reason = check_reference(img)
        if reason:
            raise ValueError(f"{name or 'image'}: お手本として使えません ({reason})")
    native = normalize_screenshot(img, snap_4096=snap_4096, max_colors=max_colors)
    return analyze_native(native, name, img.size), native


def _merge(items: list, weights: np.ndarray):
    first = items[0]
    if isinstance(first, bool):
        return bool(np.average([float(i) for i in items], weights=weights) >= 0.5)
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return _r(np.average([float(i) for i in items], weights=weights))
    if isinstance(first, dict):
        keys = [k for k in first if all(isinstance(i, dict) and k in i for i in items)]
        return {k: _merge([i[k] for i in items], weights) for k in keys
                if not (isinstance(first[k], list) and first[k] and isinstance(first[k][0], dict))}
    if isinstance(first, list) and first and all(isinstance(v, (int, float)) for v in first) \
            and all(isinstance(i, list) and len(i) == len(first) for i in items):
        arr = np.array(items, dtype=float)
        return [_r(v) for v in np.average(arr, axis=0, weights=weights)]
    return first


def _merged_palette(images: list[dict], limit: int = 32) -> list[dict]:
    acc: dict[str, float] = {}
    rgb: dict[str, list] = {}
    for im in images:
        n = im["resolution"]["native_width"] * im["resolution"]["native_height"]
        for e in im["colors"]["palette"]:
            acc[e["hex"]] = acc.get(e["hex"], 0.0) + e["share"] * n
            rgb[e["hex"]] = e["rgb"]
    total = sum(acc.values()) or 1.0
    top = sorted(acc.items(), key=lambda kv: -kv[1])[:limit]
    return [{"hex": h, "rgb": rgb[h], "share": _r(v / total)} for h, v in top]


def build_profile(images: list[dict], name: str = "", excluded: dict | None = None) -> dict:
    """Group profile: per-image stats plus a pixel-weighted summary."""
    if not images:
        raise ValueError("画像がありません")
    weights = np.array([im["resolution"]["native_width"] * im["resolution"]["native_height"]
                        for im in images], float)
    summary = _merge([{k: v for k, v in im.items() if k != "name"} for im in images], weights)
    summary["colors"]["palette"] = _merged_palette(images)
    summary["ramps"]["n_ramps"] = _r(np.average([im["ramps"]["n_ramps"] for im in images], weights=weights), 2)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "name": name,
        "image_count": len(images),
        "excluded": dict(excluded or {}),
        "summary": summary,
        "images": images,
    }


def analyze_images(imgs: list[tuple[str, Image.Image]], name: str = "", **kw) -> dict:
    return build_profile([analyze_image(img, n, **kw)[0] for n, img in imgs], name)


def save_profile(profile: dict, path) -> None:
    Path(path).write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")


def load_profile(path) -> dict:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("schema") != SCHEMA:
        raise ValueError("スタイルプロファイルの形式ではありません")
    return p


def _mix_text(m: dict) -> str:
    rel = m["relation"]
    return (f"混色: 同ランプ隣接 {rel['same_ramp_adjacent']:.0%} / 同ランプ非隣接 "
            f"{rel['same_ramp_far']:.0%} / ランプ間 {rel['cross_ramp']:.0%}, "
            f"明度差 中央値 {m['lum_diff']['median']:.0f}, "
            f"ランプ間の色相差 中央値 {m['cross_ramp']['hue_diff']['median']:.0f}°")


def profile_summary_text(profile: dict) -> str:
    """Short human readable summary (used by the GUI and CLI)."""
    s = profile["summary"]
    pal = " ".join(e["hex"] for e in s["colors"]["palette"][:16])
    d, o, f = s["dither"], s["outline"], s["fill"]
    return "\n".join([
        f"{profile.get('name') or 'profile'}: {profile['image_count']} 枚",
        f"色数 {s['colors']['n_colors']:.1f} / ランプ {s['ramps']['n_ramps']} "
        f"(平均 {s['ramps']['steps_mean']} 段)",
        f"パレット {pal}",
        f"ディザ率 {d['ratio']:.1%} (市松 {d['pattern_mix']['checker']:.0%} / "
        f"タイル {d['pattern_mix']['tile']:.0%} / その他 {d['pattern_mix']['other']:.0%}), "
        f"同ランプ隣接ペア {d['pair_same_ramp_adjacent']:.0%}",
        *([_mix_text(d["mix"])] if "mix" in d else []),
        f"輪郭: エッジ上の暗色 {o['dark_edge_ratio']:.0%}, 太さ中央値 {o['thickness']['median']:.1f}, "
        f"中間色AA {o['antialias']['ratio']:.0%}",
        f"塗り: 64px以上の領域 {f['flat_ratio']:.0%}, 孤立ピクセル {f['isolated_ratio']:.2%}",
        f"解像度: {s['resolution']['native_width']:.0f}×{s['resolution']['native_height']:.0f}"
        f" ({s['resolution']['lines']:.0f} ライン)",
    ])
