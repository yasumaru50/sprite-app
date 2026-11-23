"""Synthetic 16-colour "CG" with known regions, used by the analyzer tests."""

import numpy as np

from retrocore.palette import _from_4bit

PAL = _from_4bit([
    (0, 0, 0),        # 0 outline black
    (15, 15, 15),     # 1 white
    (15, 13, 12),     # 2 skin light
    (14, 9, 8),       # 3 skin mid
    (10, 5, 5),       # 4 skin dark
    (4, 6, 12),       # 5 blue dark
    (8, 11, 15),      # 6 blue light
    (2, 9, 4),        # 7 green
])

W, H = 200, 120


def make_cg():
    """Return (indices HxW, palette, region masks)."""
    idx = np.full((H, W), 1, np.uint8)                 # white background
    yy, xx = np.mgrid[0:H, 0:W]
    regions = {}
    # Flat skin block with a 1px black outline.
    idx[10:60, 10:80] = 0
    idx[11:59, 11:79] = 2
    regions["flat"] = np.zeros((H, W), bool)
    regions["flat"][14:56, 14:76] = True
    # Checker between blue dark / light.
    m = (yy >= 10) & (yy < 60) & (xx >= 100) & (xx < 190)
    idx[m] = np.where((yy + xx) % 2 == 0, 5, 6)[m]
    regions["checker"] = m
    # 2x2 tile (25 %): skin mid dots on skin light.
    m = (yy >= 70) & (yy < 110) & (xx >= 10) & (xx < 90)
    idx[m] = np.where((yy % 2 == 0) & (xx % 2 == 0), 3, 2)[m]
    regions["tile2x2"] = m
    # 4x4 sparse dots: green on white.
    m = (yy >= 70) & (yy < 110) & (xx >= 110) & (xx < 190)
    idx[m] = np.where((yy % 4 == 0) & (xx % 4 == 0), 7, 1)[m]
    regions["tile4x4"] = m
    return idx, PAL, regions


def rgb(idx=None):
    idx = make_cg()[0] if idx is None else idx
    return PAL[idx]
