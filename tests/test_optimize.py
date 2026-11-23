import numpy as np
from PIL import Image

from retrocore import ConvertSettings, convert
from retrocore.analyzer import analyze_image, build_profile
from retrocore.optimize import analyze_result, fit_to_profile, profile_distance
from retrocore.style import apply_overrides, profile_to_settings
from synth_cg import rgb
from test_style import photo


def target():
    st, _ = analyze_image(Image.fromarray(rgb()), "synth")
    return build_profile([st], "synth")


def test_distance_zero_for_identical_stats():
    prof = target()
    s = prof["summary"]
    assert profile_distance(s, s) == 0
    other = {**s, "dither": {**s["dither"], "ratio": s["dither"]["ratio"] + 0.2}}
    assert profile_distance(other, s) > 0


def test_analyze_result_matches_output():
    r = convert(photo(), ConvertSettings(mode="style", preset="カスタム",
                                         custom_width=160, custom_height=100))
    st = analyze_result(r)
    assert st["resolution"]["native_width"] == 160
    assert st["colors"]["n_colors"] == r.used_colors


def test_fit_respects_budget_and_never_worsens():
    prof = target()
    ov, _ = profile_to_settings(prof)
    ov.update(preset="カスタム", custom_width=160, custom_height=100)
    start = apply_overrides(ConvertSettings(), ov)
    calls = []
    fr = fit_to_profile(photo(), prof, start, max_evals=6,
                        progress=lambda i, n, b: calls.append((i, b)))
    assert fr.evaluations <= 6 and len(calls) == fr.evaluations
    assert fr.distance <= fr.start_distance + 1e-9
    assert all(b2 <= b1 + 1e-9 for (_, b1), (_, b2) in zip(calls, calls[1:]))
    # Early stop
    fr2 = fit_to_profile(photo(), prof, start, max_evals=10, progress=lambda i, n, b: False)
    assert fr2.evaluations == 1


def test_parameter_penalty():
    from retrocore.optimize import SEARCH_SPACE, parameter_penalty
    start = {"style_smooth_strength": 0.4, "saturation": 0, "style_pattern": "tile2"}
    assert parameter_penalty(start, start) == 0
    lo, hi = SEARCH_SPACE["style_smooth_strength"][1:3]
    moved = dict(start, style_smooth_strength=0.4 + (hi - lo))
    assert abs(parameter_penalty(moved, start) - 1.0) < 1e-9
    assert parameter_penalty(dict(start, style_pattern="checker"), start) == 0.25


def test_regularization_keeps_settings_closer():
    prof = target()
    ov, _ = profile_to_settings(prof)
    ov.update(preset="カスタム", custom_width=160, custom_height=100)
    start = apply_overrides(ConvertSettings(), ov)
    free = fit_to_profile(photo(), prof, start, max_evals=10, regularization=0.0)
    tied = fit_to_profile(photo(), prof, start, max_evals=10, regularization=5.0)
    from retrocore.optimize import parameter_penalty
    origin = {k: getattr(start, k) for k in free.overrides}
    assert parameter_penalty(tied.overrides, origin) <= parameter_penalty(free.overrides, origin) + 1e-9
    assert tied.penalty >= 0 and free.penalty >= 0
