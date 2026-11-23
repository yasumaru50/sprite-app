"""Search style settings whose output statistics match a reference profile.

The output of each candidate conversion is analysed with the same analyzer
as the reference CG; ``profile_distance`` compares a few scale-free
statistics (dither share and pattern mix, flat area, isolated pixels,
outlines).  A plain coordinate search with shrinking steps and a hard
evaluation budget keeps the cost predictable (one evaluation = one
conversion + one analysis, about 1 s for 640×400).

The search minimises ``distance + regularization · Σ (Δᵢ / rangeᵢ)²`` where
Δᵢ is how far parameter i moved from its starting value (the profile
mapping or preset) and rangeᵢ its search range; a pattern change counts as
half the range.  Without this term the search happily drives parameters to
extremes (smoothing 1.0, saturation +45) for a tiny statistical gain, which
looks worse than the starting point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .analyzer import analyze_native
from .pipeline import ConvertResult, ConvertSettings, convert
from .screenshot import GridInfo, NativeImage
from .style import apply_overrides

# (dotted path, weight).  Weights reflect how visible a difference is.
# Isolated pixels and hand-made anti-aliasing are kept low: references keep a
# few % of both on purpose, and the conversion has no direct control of AA.
FEATURES: list[tuple[str, float]] = [
    ("dither.ratio", 3.0),
    ("dither.pattern_mix.checker", 1.0),
    ("dither.pattern_mix.tile", 1.0),
    ("dither.pattern_mix.other", 0.5),
    ("dither.pair_same_ramp_adjacent", 0.5),
    ("fill.flat_ratio", 2.0),
    ("fill.isolated_ratio", 2.0),
    ("outline.dark_edge_ratio", 1.0),
    ("outline.outline_pixel_ratio", 6.0),
    ("outline.antialias.ratio", 0.0),
    ("colors.saturation.mean", 1.0),
]

# name -> (kind, lo, hi, initial step) ; kind "f" float, "i" int, "c" categorical
SEARCH_SPACE: dict[str, tuple] = {
    "style_grad_low": ("f", 0.1, 3.0, 0.4),
    "style_grad_high": ("f", 2.0, 20.0, 3.0),
    "style_mask_curvature": ("f", 1.0, 12.0, 2.0),
    "saturation": ("i", -50, 100, 25),
    "style_smooth_strength": ("f", 0.0, 1.0, 0.2),
    "style_despeckle": ("i", 0, 3, 1),
    "style_outline_threshold": ("f", 15.0, 90.0, 12.0),
    "style_pattern": ("c", ["checker", "tile2", "bayer4"]),
}


CATEGORICAL_PENALTY = 0.25     # = (0.5 range)²

DEFAULT_REGULARIZATION = 0.1


def parameter_penalty(cand: dict, start: dict) -> float:
    """Σ (Δ / range)² over the searched parameters (see module docstring)."""
    total = 0.0
    for p, v in cand.items():
        spec = SEARCH_SPACE.get(p)
        if spec is None or p not in start:
            continue
        if spec[0] == "c":
            total += CATEGORICAL_PENALTY if v != start[p] else 0.0
        else:
            span = float(spec[2] - spec[1]) or 1.0
            total += ((float(v) - float(start[p])) / span) ** 2
    return total


def _get(d: dict, path: str):
    for k in path.split("."):
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, (int, float)) else None


def profile_distance(stats: dict, target: dict) -> float:
    """Weighted RMS difference of the comparable statistics."""
    num = den = 0.0
    for path, w in FEATURES:
        a, b = _get(stats, path), _get(target, path)
        if a is None or b is None:
            continue
        num += w * (float(a) - float(b)) ** 2
        den += w
    return float(np.sqrt(num / den)) if den else 0.0


def analyze_result(result: ConvertResult) -> dict:
    idx = np.asarray(result.image, dtype=np.uint8)
    h, w = idx.shape
    native = NativeImage(idx, np.asarray(result.palette, np.uint8),
                         GridInfo(1.0, 1.0, 0.0, 0.0, w, h, 0.0, "native"), len(result.palette))
    return analyze_native(native, "output")


@dataclass
class FitResult:
    overrides: dict
    settings: ConvertSettings
    distance: float              # statistical distance of the chosen settings
    start_distance: float
    evaluations: int
    penalty: float = 0.0         # regularisation term of the chosen settings
    history: list[tuple[dict, float]] = field(default_factory=list)
    elapsed: float = 0.0


# Parameters measured in "per pixel" units scale with resolution; in the
# half-resolution phase they are multiplied by this factor.
COARSE_PARAM_SCALE = {"style_grad_low": 2.0, "style_grad_high": 2.0,
                      "style_mask_curvature": 2.0, "style_outline_threshold": 2.0}


def _coarse_settings(src: Image.Image, start: ConvertSettings, factor: float = 0.5):
    """Settings producing the same picture at ``factor`` of the target size."""
    from .pipeline import target_size
    w, h, aspect = target_size(src.width, src.height, start)
    return apply_overrides(start, {"preset": "カスタム", "fit": "stretch",
                                   "custom_width": max(16, int(round(w * factor))),
                                   "custom_height": max(16, int(round(h * factor))),
                                   "pixel_aspect": aspect})


def _coordinate_search(evaluate, names, cur, best, steps, budget, should_stop):
    """Greedy coordinate search; ``evaluate`` returns the objective.  Mutates ``steps``."""
    used = 0
    while used < budget and not should_stop():
        improved = False
        for p in names:
            if used >= budget or should_stop():
                break
            spec = SEARCH_SPACE[p]
            if spec[0] == "c":
                cands = [v for v in spec[1] if v != cur[p]]
            else:
                lo, hi = spec[1], spec[2]
                st = steps[p]
                cands = [cur[p] + st, cur[p] - st]
                cands = [int(round(v)) if spec[0] == "i" else round(float(v), 3)
                         for v in cands if lo <= v <= hi]
            for v in cands:
                if used >= budget or should_stop():
                    break
                trial = dict(cur, **{p: v})
                obj, fresh = evaluate(trial)
                used += fresh
                if obj < best - 1e-4:
                    best, cur, improved = obj, trial, True
                    break
        if not improved:
            shrinkable = [p for p in steps if SEARCH_SPACE[p][0] == "f"]
            if not shrinkable or all(steps[p] < SEARCH_SPACE[p][3] / 8 for p in shrinkable):
                break
            for p in shrinkable:
                steps[p] /= 2
    return best, cur


def fit_to_profile(src: Image.Image, profile: dict, start: ConvertSettings,
                   palette: np.ndarray | None = None, max_evals: int = 24,
                   params: list[str] | None = None, progress=None,
                   regularization: float = DEFAULT_REGULARIZATION,
                   coarse: bool = False, coarse_share: float = 0.75) -> FitResult:
    """Coordinate search from ``start`` (a style-mode ``ConvertSettings``).

    ``max_evals`` bounds the number of evaluations.  With ``coarse`` the
    first ``coarse_share`` of them run at half resolution (≈ 1/4 of the cost;
    per-pixel thresholds scaled by ``COARSE_PARAM_SCALE``), the rest refine
    the result at full resolution with halved steps — about 2× faster for
    the same budget.  Reported distances are always full-resolution ones.

    ``progress(i, max_evals, best_objective)`` is called after every
    evaluation; returning ``False`` stops the search early.
    """
    t0 = time.perf_counter()
    target = profile["summary"]
    names = [p for p in (params or list(SEARCH_SPACE))
             if p != "style_outline_threshold" or start.style_outline]
    names = [p for p in names if p != "style_pattern" or start.style_pattern != "none"]
    origin = {p: getattr(start, p) for p in names}
    steps = {p: SEARCH_SPACE[p][3] for p in names if SEARCH_SPACE[p][0] != "c"}
    history: list[tuple[dict, float]] = []
    state = {"stop": False, "n": 0, "best": float("inf")}

    def keyof(cand: dict) -> tuple:
        return tuple(sorted((k, round(v, 4) if isinstance(v, float) else v) for k, v in cand.items()))

    def make_eval(base: ConvertSettings, scale: dict, record: bool):
        cache: dict[tuple, tuple[float, float]] = {}
        stage1: dict = {}            # smoothing / palette results, shared across candidates

        def evaluate(cand: dict) -> tuple[float, int]:
            key = keyof(cand)
            if key in cache:
                return cache[key][0], 0
            applied = {k: (v * scale.get(k, 1.0) if isinstance(v, (int, float))
                           and not isinstance(v, bool) and k in scale else v) for k, v in cand.items()}
            stats = analyze_result(convert(src, apply_overrides(base, applied), palette, cache=stage1))
            dist = profile_distance(stats, target)
            obj = dist + regularization * parameter_penalty(cand, origin)
            cache[key] = (obj, dist)
            state["n"] += 1
            if record:
                history.append((dict(cand), obj))
                state["best"] = min(state["best"], obj)
            if progress is not None and progress(state["n"], max_evals,
                                                 state["best"] if record else obj) is False:
                state["stop"] = True
            return obj, 1

        evaluate.cache = cache  # type: ignore[attr-defined]
        return evaluate

    stop = lambda: state["stop"]  # noqa: E731
    full = make_eval(start, {}, True)
    cur = dict(origin)
    best, _ = full(cur)
    start_dist = best                 # penalty is 0 at the start
    budget = max_evals - 1
    if coarse and budget >= 4:
        n_coarse = int(budget * coarse_share)
        rough = make_eval(_coarse_settings(src, start), COARSE_PARAM_SCALE, False)
        cbest, _ = rough(cur)
        _, ccur = _coordinate_search(rough, names, cur, cbest, steps, n_coarse, stop)
        budget -= state["n"] - 1
        if ccur != cur and not state["stop"]:
            obj, _ = full(ccur)
            budget -= 1
            if obj < best:
                best, cur = obj, ccur
        for p in steps:
            steps[p] = max(steps[p], SEARCH_SPACE[p][3] / 4) / 2
    if budget > 0 and not state["stop"]:
        best, cur = _coordinate_search(full, names, cur, best, steps, budget, stop)
    pen = parameter_penalty(cur, origin)
    return FitResult(overrides=cur, settings=apply_overrides(start, cur),
                     distance=full.cache[keyof(cur)][1], start_distance=start_dist,
                     evaluations=state["n"], penalty=pen, history=history,
                     elapsed=time.perf_counter() - t0)
