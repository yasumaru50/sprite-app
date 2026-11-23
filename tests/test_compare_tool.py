import importlib.util
from pathlib import Path

from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "compare_tool", Path(__file__).resolve().parent.parent / "tools" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def test_parse_config():
    label, over = compare.parse_config("fs:dither=floyd,dither_strength=0.5,snap_4096=False")
    assert label == "fs"
    assert over == {"dither": "floyd", "dither_strength": 0.5, "snap_4096": False}


def test_compare_writes_sheets_for_synthetic_inputs(tmp_path):
    written = compare.main(["--synthetic", "--out", str(tmp_path),
                            "-c", "none:dither=none", "-c", "b4:dither=bayer4"])
    assert written == 0
    sheets = sorted(tmp_path.glob("*.png"))
    names = {p.name for p in sheets}
    assert {"synthetic_cel.png", "synthetic_ramps.png", "_overview.png"} <= names
    w, h = Image.open(tmp_path / "synthetic_cel.png").size
    assert w > 3 * 640 and h > 2 * 400   # original + 2 configs, two rows


def test_compare_cross_ramp_columns(tmp_path):
    from retrocore.analyzer import analyze_image, build_profile, save_profile
    from synth_cg import rgb

    st, _ = analyze_image(Image.fromarray(rgb()), "synth")
    prof = build_profile([st], "synth")
    prof["summary"]["dither"]["mix"]["relation"]["cross_ramp"] = 0.6
    path = tmp_path / "p.json"
    save_profile(prof, path)
    configs = []
    orig = compare.build_sheet
    compare.build_sheet = lambda name, src, results, **kw: (configs.extend(r[0] for r in results),
                                                            orig(name, src, results, **kw))[1]
    try:
        assert compare.main(["--synthetic", "--max-inputs", "1", "--no-overview", "--out",
                             str(tmp_path / "o"), "--profile", str(path), "--cross-ramp"]) == 0
    finally:
        compare.build_sheet = orig
    assert any("ランプ間混色" in c for c in configs) and "プロファイル: synth" in configs
