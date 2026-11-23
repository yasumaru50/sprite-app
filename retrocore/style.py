"""Style presets and the mapping from a style profile to conversion settings.

``profile_to_settings`` turns the statistics measured by ``analyzer`` into
``ConvertSettings`` overrides for the style-conversion pipeline (mode
``"style"``); every rule is a simple, documented heuristic and returns a note
explaining it, so the GUI can show why a value was chosen.  ``optimize.py``
can refine the result by searching for settings whose output statistics are
closest to the profile.

Preset names describe era and technique only.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from .pipeline import ConvertSettings

STYLE_FIELDS = [f.name for f in fields(ConvertSettings) if f.name.startswith("style_")]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Values are educated guesses for the look of each period; the reasoning is
# given per preset.  Measured reference (exact 16-colour emulator captures,
# three sample groups): 46-50 % dithered pixels, tile / ordered patterns
# dominating (one group 43 % checker), dark 1-pixel outlines on 53-79 % of
# strong edges, 25-33 % of the picture in flat regions of 65+ pixels and
# 1.9-2.7 % isolated pixels outside dither.

STYLE_PRESETS: dict[str, dict] = {
    # Early 8-colour digital RGB (fixed palette, mostly 200-line modes):
    # intermediate colours could only be faked by mixing the 8 primaries, so
    # tile patterns (25/50/75 %) are everywhere, towards black and white too.
    "1983 デジタル8色・タイルディザ": {
        "mode": "style",
        "preset": "PC-98 640×200 (ライン倍化)",
        "palette_source": "preset:PC-98 デジタル8色",
        "style_ramp_extend": True,
        "style_smooth": "bilateral", "style_smooth_strength": 0.7,
        "style_pattern": "tile2", "style_dither_strength": 1.0,
        "style_grad_low": 0.3, "style_grad_high": 14.0,
        "style_despeckle": 1,
        "style_outline": True, "style_outline_threshold": 45.0, "style_outline_thickness": 1,
        # Mixing two primaries (red + yellow -> orange) was the usual way to get
        # an intermediate hue; luminance differences between primaries are large.
        # Cross-ramp dither is opt-in: raise style_cross_ramp (0 = off) to use it.
        "style_cross_ramp": 0.0, "style_cross_ramp_max_lum": 64.0, "style_cross_ramp_max_hue": 90.0,
    },
    # First analog-16 CG: freely chosen 4096-colour palette, artists painted
    # flat cel areas with short ramps and used dither sparingly (checker only
    # on large soft gradients); bold outlines.
    "1987 アナログ16色・ベタ塗り": {
        "mode": "style",
        "preset": "PC-98 640×400",
        "colors": 16, "snap_4096": True,
        "style_ramps": 4, "style_steps": 4,
        # Kuwahara radius 2 (strength 0.45): flat cels without the blotchy
        # artefacts radius 3 produces on detailed sources.
        "style_smooth": "kuwahara", "style_smooth_strength": 0.45,
        "style_pattern": "checker", "style_dither_strength": 0.8,
        "style_grad_low": 1.2, "style_grad_high": 4.0,
        "style_despeckle": 2,
        "style_outline": True, "style_outline_threshold": 35.0, "style_outline_thickness": 1,
        # Sparse dither: if enabled, only very close colours are mixed.
        # Cross-ramp dither is opt-in: raise style_cross_ramp (0 = off) to use it.
        "style_cross_ramp": 0.0, "style_cross_ramp_max_lum": 24.0, "style_cross_ramp_max_hue": 45.0,
    },
    # Mature analog-16 style: more ramps, checker gradients on skin / sky,
    # still mostly flat paint (less dither than the measured samples, which
    # are later and dither about half of the picture).
    "1990 アナログ16色・市松グラデーション": {
        "mode": "style",
        "preset": "PC-98 640×400",
        "colors": 16, "snap_4096": True,
        "style_ramps": 5, "style_steps": 0,
        "style_smooth": "bilateral", "style_smooth_strength": 0.6,
        "style_pattern": "checker", "style_dither_strength": 1.0,
        "style_grad_low": 0.7, "style_grad_high": 6.0,
        "style_despeckle": 1,
        "style_outline": True, "style_outline_threshold": 40.0, "style_outline_thickness": 1,
        # Checker of a skin / sky shade with a neighbouring hue of similar brightness.
        # Cross-ramp dither is opt-in: raise style_cross_ramp (0 = off) to use it.
        "style_cross_ramp": 0.0, "style_cross_ramp_max_lum": 32.0, "style_cross_ramp_max_hue": 90.0,
    },
    # Early-90s analog-16 CG with heavy tile dithering (≈ the measured sample
    # groups: ~50 % dither, tile patterns dominating): 2x2 tiles give 25 % steps
    # between ramp shades, used on most shading, not only on gradients.
    "1993 アナログ16色・タイルディザ": {
        "mode": "style",
        "preset": "PC-98 640×400",
        "colors": 16, "snap_4096": True,
        "style_ramps": 0, "style_steps": 0,
        "style_smooth": "bilateral", "style_smooth_strength": 0.45,
        "style_pattern": "tile2", "style_dither_strength": 1.0,
        "style_grad_low": 0.4, "style_grad_high": 10.0,
        "style_despeckle": 1,
        "style_outline": True, "style_outline_threshold": 40.0, "style_outline_thickness": 1,
        # ≈ measured groups with mostly tile patterns (35-57 % cross-ramp pairs,
        # luminance difference ≤ ~45, hue difference up to 100-140°).
        # Cross-ramp dither is opt-in: raise style_cross_ramp (0 = off) to use it.
        "style_cross_ramp": 0.0, "style_cross_ramp_max_lum": 45.0, "style_cross_ramp_max_hue": 140.0,
    },
    # Mid-90s: dither applied to almost every shaded area with finer ordered
    # patterns (4x4 levels), less smoothing so more texture survives.
    "1995 アナログ16色・規則ディザ多用": {
        "mode": "style",
        "preset": "PC-98 640×400",
        "colors": 16, "snap_4096": True,
        "style_ramps": 0, "style_steps": 0,
        "style_smooth": "bilateral", "style_smooth_strength": 0.35,
        "style_pattern": "bayer4", "style_dither_strength": 1.0,
        "style_grad_low": 0.25, "style_grad_high": 14.0,
        "style_despeckle": 0,
        "style_outline": True, "style_outline_threshold": 50.0, "style_outline_thickness": 1,
        # ≈ measured group with most cross-ramp pairs (87 %), often complementary
        # hues of equal brightness mixed into a muted tone.
        # Cross-ramp dither is opt-in: raise style_cross_ramp (0 = off) to use it.
        "style_cross_ramp": 0.0, "style_cross_ramp_max_lum": 40.0, "style_cross_ramp_max_hue": 180.0,
    },
}


# ---------------------------------------------------------------------------
# Profile -> settings
# ---------------------------------------------------------------------------


def _interp(x: float, xs, ys) -> float:
    return float(np.interp(x, xs, ys))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        return tuple(int(c, 16) * 17 for c in h)  # type: ignore[return-value]
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def profile_palette(profile: dict, n: int = 16) -> np.ndarray:
    """The ``n`` most used colours of the profile (for "use reference colours")."""
    pal = profile["summary"]["colors"]["palette"][:n]
    return np.array([e["rgb"] for e in pal], dtype=np.uint8)


def _hist_quantile(hist: dict, q: float) -> float | None:
    """Quantile from a ``{bin_edges, hist}`` summary (linear within a bin)."""
    edges, h = hist.get("bin_edges"), hist.get("hist")
    if not edges or not h or sum(h) <= 0:
        return None
    cum = np.concatenate([[0.0], np.cumsum(h) / sum(h)])
    return float(np.interp(q, cum, edges[: len(cum)]))


def cross_ramp_settings(profile: dict) -> tuple[dict, str]:
    """Cross-ramp dither settings from ``dither.mix`` of a profile.

    * ``style_cross_ramp`` (how freely pairs of different ramps are used):
      share of cross-ramp pairs in the reference / 0.7, clamped to 0..1; off
      below 10 % (the reference practically only mixes within ramps).
    * ``style_cross_ramp_max_lum``: 75th percentile of the luminance
      difference of the reference's cross-ramp pairs (16..64).
    * ``style_cross_ramp_max_hue``: 90th percentile of their hue difference
      (30..180°).
    """
    mix = profile["summary"]["dither"].get("mix")
    if not mix:
        return {"style_cross_ramp": 0.0}, "ランプ間の混色: プロファイルに混色の統計がない → なし"
    share = float(mix["relation"]["cross_ramp"])
    cr = mix["cross_ramp"]
    lum = _hist_quantile(cr["lum_diff"], 0.75)
    lum = cr["lum_diff"].get("median", 32.0) if lum is None else lum
    hue = float(cr["hue_diff"].get("p90", 180.0)) or 180.0
    ov = {
        "style_cross_ramp": 0.0 if share < 0.1 else round(float(np.clip(share / 0.7, 0, 1)), 2),
        "style_cross_ramp_max_lum": round(float(np.clip(lum, 16, 64)), 1),
        "style_cross_ramp_max_hue": round(float(np.clip(hue, 30, 180)), 0),
    }
    note = (f"ランプ間の混色: お手本のランプ間の組 {share:.0%} → 度合い {ov['style_cross_ramp']} "
            f"(明度差 ≤ {ov['style_cross_ramp_max_lum']}, 色相差 ≤ {ov['style_cross_ramp_max_hue']:.0f}°)")
    return ov, note


def profile_to_settings(profile: dict, cross_ramp: bool = False) -> tuple[dict, list[str]]:
    """``ConvertSettings`` overrides (style mode) derived from a profile.

    Returns ``(overrides, notes)``; each note explains one decision.
    ``cross_ramp`` (opt-in) also derives the cross-ramp dither settings from
    the mixed-pair statistics (``cross_ramp_settings``); otherwise
    ``style_cross_ramp`` is 0 (only adjacent shades of one ramp are mixed).
    """
    s = profile["summary"]
    c, r, d, o, f, res = (s["colors"], s["ramps"], s["dither"], s["outline"], s["fill"],
                          s["resolution"])
    ov: dict = {"mode": "style"}
    notes: list[str] = []

    # Colours: the target is a 16-colour mode, so clamp to 8..16 (profiles
    # built from recovered screenshots may report a few extra colours).
    n = int(round(c["n_colors"]))
    ov["colors"] = int(np.clip(n, 8, 16))
    ov["snap_4096"] = bool(c.get("is_pc98_4096", True))
    notes.append(f"色数: お手本 {c['n_colors']:.1f} 色 → {ov['colors']} 色 (16 色モード想定で上限 16)")

    nr = r.get("n_ramps_multi", r["n_ramps"])
    ov["style_ramps"] = int(np.clip(round(nr), 2, 6))
    ov["style_steps"] = 0
    notes.append(f"ランプ: 2 段以上のランプ {nr:.1f} 本 → {ov['style_ramps']} 本 (段数は自動配分)")

    # Dither amount -> width of the gradation band of the mask.
    ratio = float(d["ratio"])
    if ratio < 0.02:
        ov["style_pattern"] = "none"
        notes.append(f"ディザ: お手本のディザ率 {ratio:.1%} → ディザなし")
    else:
        ov["style_grad_low"] = round(_interp(ratio, [0.05, 0.40], [1.2, 0.3]), 2)
        ov["style_grad_high"] = round(_interp(ratio, [0.05, 0.40], [4.0, 12.0]), 1)
        ov["style_mask_curvature"] = round(_interp(ratio, [0.05, 0.40], [2.0, 8.0]), 1)
        ov["style_dither_strength"] = 1.0
        cls = d["classes"]
        fine = cls.get("tile4x4", 0) + cls.get("ordered_mix", 0)
        tile = cls.get("tile2x2", 0) + cls.get("stripe", 0)
        checker = cls.get("checker", 0)
        if fine > max(tile, checker):
            ov["style_pattern"] = "bayer4"
        elif tile > checker:
            ov["style_pattern"] = "tile2"
        else:
            ov["style_pattern"] = "checker"
        notes.append(
            f"ディザ: 率 {ratio:.1%} → 勾配マスク {ov['style_grad_low']}〜{ov['style_grad_high']}"
            f" (曲率許容 {ov['style_mask_curvature']}); "
            f"市松 {checker:.1%} / 2×2系 {tile:.1%} / 4×4・規則混在 {fine:.1%} → {ov['style_pattern']}")

    # Saturation: era CG is strongly saturated; nudge the source towards it.
    sat = float(c["saturation"]["mean"])
    ov["saturation"] = int(round(_interp(sat, [0.2, 0.5], [0, 40])))
    notes.append(f"彩度: お手本の平均彩度 {sat:.2f} → 彩度補正 {ov['saturation']:+d}")

    # Flat paint -> smoothing strength.
    flat = float(f["flat_ratio"])
    ov["style_smooth"] = "bilateral"
    ov["style_smooth_strength"] = round(_interp(flat, [0.2, 0.6], [0.3, 0.8]), 2)
    notes.append(f"塗り: ベタ領域 {flat:.0%} → 平滑化の強さ {ov['style_smooth_strength']}")

    # Isolated pixels.  Exact references keep ~2-3 % on purpose (highlights,
    # texture dots), so only clean up when the reference has fewer.
    iso = float(f["isolated_ratio"])
    ov["style_despeckle"] = 2 if iso <= 0.01 else 1 if iso <= 0.025 else 0
    notes.append(f"孤立ピクセル {iso:.1%} → 除去 {ov['style_despeckle']} 回")

    # Outlines.
    dark = float(o["dark_edge_ratio"])
    ov["style_outline"] = dark >= 0.45
    if ov["style_outline"]:
        ov["style_outline_threshold"] = round(_interp(dark, [0.45, 0.8], [60.0, 30.0]), 1)
        ov["style_outline_thickness"] = int(np.clip(round(o["thickness"]["median"]), 1, 3))
        ov["style_outline_color"] = hex_to_rgb(o["color"])
        notes.append(f"輪郭: エッジ上の暗色 {dark:.0%} → 輪郭あり (色 {o['color']}, 太さ "
                     f"{ov['style_outline_thickness']}, 閾値 {ov['style_outline_threshold']})")
    else:
        notes.append(f"輪郭: エッジ上の暗色 {dark:.0%} → 輪郭線の付加なし")

    if cross_ramp and ov.get("style_pattern") != "none":
        cov, note = cross_ramp_settings(profile)
        ov.update(cov)
        notes.append(note)
    else:
        ov["style_cross_ramp"] = 0.0
        mix = d.get("mix")
        if mix:
            notes.append(f"ランプ間の混色: お手本のランプ間の組 {mix['relation']['cross_ramp']:.0%} "
                         "(反映はオプトイン) → なし")

    lines = int(round(res.get("lines", 400)))
    ov["preset"] = "PC-98 640×200 (ライン倍化)" if lines == 200 else "PC-98 640×400"
    notes.append(f"解像度: {lines} ライン → {ov['preset']}")
    return ov, notes


def apply_overrides(base: ConvertSettings, overrides: dict) -> ConvertSettings:
    vals = {f.name: getattr(base, f.name) for f in fields(ConvertSettings)}
    for k, v in overrides.items():
        if k in vals:
            vals[k] = tuple(v) if isinstance(v, list) else v
    return ConvertSettings(**vals)
