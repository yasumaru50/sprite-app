import json

import numpy as np
from PIL import Image

from retrocore.analyzer import (
    DITHER_CLASSES,
    analyze_image,
    build_profile,
    classify_dither,
    label_components,
    load_profile,
    profile_summary_text,
    run_lengths,
    save_profile,
)
from retrocore.color import LUMA
from synth_cg import PAL, make_cg, rgb


def test_classify_dither_regions():
    idx, pal, reg = make_cg()
    cls = classify_dither(idx, pal.astype(np.float32) @ LUMA)
    inner = lambda m: m & np.roll(m, 3, 0) & np.roll(m, -3, 0) & np.roll(m, 3, 1) & np.roll(m, -3, 1)
    for name in ("checker", "tile2x2", "tile4x4"):
        k = DITHER_CLASSES.index(name) + 1
        assert (cls[inner(reg[name])] == k).mean() > 0.95, name
    assert (cls[reg["flat"]] == 0).all()


def test_label_components_and_runs():
    a = np.array([[1, 1, 0, 2],
                  [0, 1, 0, 2],
                  [0, 0, 0, 1]], np.uint8)
    lab = label_components(a)
    assert lab[0, 0] == lab[1, 1] and lab[0, 0] != lab[2, 3]
    assert len(np.unique(lab)) == 4
    lab = label_components(a, valid=a != 0)
    assert (lab[a == 0] == -1).all()
    m = np.array([[1, 1, 0, 1, 1, 1]], bool)
    assert run_lengths(m, 1).tolist() == [[2, 2, 0, 3, 3, 3]]
    assert run_lengths(m.T, 0).ravel().tolist() == [2, 2, 0, 3, 3, 3]


def test_analyze_synthetic_profile():
    idx, pal, reg = make_cg()
    st, native = analyze_image(Image.fromarray(rgb(idx)), "synth")
    c = st["colors"]
    assert c["n_colors"] == 7 and c["is_pc98_4096"]   # colour 4 unused
    assert {e["hex"] for e in c["palette"]} >= {"#000", "#FFF", "#FDC", "#46C", "#8BF"}
    assert abs(sum(e["share"] for e in c["palette"]) - 1) < 1e-3
    d = st["dither"]
    total_regions = sum(reg[k].mean() for k in ("checker", "tile2x2", "tile4x4"))
    assert abs(d["ratio"] - total_regions) < 0.05
    assert d["classes"]["checker"] > 0.15
    assert d["pair_same_ramp_adjacent"] > 0.6           # blue dark/light, skin mid/light
    o = st["outline"]
    assert o["color"] == "#000" and o["thickness"]["median"] == 1
    assert o["dark_edge_ratio"] > 0.5
    f = st["fill"]
    assert f["isolated_ratio"] < 0.005 and f["flat_ratio"] > 0.3
    r = st["resolution"]
    assert r["lines"] == 400 and not r["line_doubled"]
    assert st["ramps"]["n_ramps"] >= 3


def test_line_doubling_detected():
    idx, _, _ = make_cg()
    st, _ = analyze_image(Image.fromarray(rgb(np.repeat(idx[::2], 2, axis=0))))
    assert st["resolution"]["line_doubled"] and st["resolution"]["lines"] == 200


def test_outline_thickness_and_antialias():
    idx = np.full((60, 60), 1, np.uint8)
    idx[10:50, 10:13] = 0                      # 3px thick vertical line
    idx[10:50, 13] = 3                         # intermediate shade next to it
    idx[10:50, 14:40] = 2
    st, _ = analyze_image(Image.fromarray(PAL[idx]))
    assert st["outline"]["thickness"]["median"] >= 2
    assert st["outline"]["antialias"]["ratio"] > 0.2


def test_profile_roundtrip_and_merge(tmp_path):
    idx, _, _ = make_cg()
    a, _ = analyze_image(Image.fromarray(rgb(idx)), "a")
    b, _ = analyze_image(Image.fromarray(rgb(np.fliplr(idx))), "b")
    prof = build_profile([a, b], "synthetic group")
    assert prof["image_count"] == 2 and len(prof["images"]) == 2
    assert abs(prof["summary"]["dither"]["ratio"] - a["dither"]["ratio"]) < 0.02
    assert prof["summary"]["colors"]["palette"][0]["hex"].startswith("#")
    path = tmp_path / "p.json"
    save_profile(prof, path)
    assert load_profile(path) == json.loads(path.read_text(encoding="utf-8"))
    assert "ディザ率" in profile_summary_text(prof)


def test_check_reference_and_strict_mode():
    from retrocore.analyzer import check_reference
    exact = Image.fromarray(rgb())
    assert check_reference(exact) is None
    off = rgb().copy()
    off[0, 0] = (1, 2, 3)                       # one colour off the 16-level grid
    assert "16 段階" in check_reference(Image.fromarray(off))
    many = np.random.default_rng(0).integers(0, 16, (20, 20, 3)).astype(np.uint8) * 17
    assert "色" in check_reference(Image.fromarray(many))
    import pytest
    with pytest.raises(ValueError):
        analyze_image(Image.fromarray(off), "bad", strict=True)
    st, _ = analyze_image(exact, "ok", strict=True)
    prof = build_profile([st], "g", excluded={"bad.png": "reason"})
    assert prof["excluded"] == {"bad.png": "reason"}


def test_analyze_cli_excludes_invalid(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "analyze_tool", Path(__file__).resolve().parent.parent / "tools" / "analyze.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ref = tmp_path / "refs"
    ref.mkdir()
    Image.fromarray(rgb()).save(ref / "good.png")
    bad = rgb().copy()
    bad[:5, :5] = (100, 101, 102)
    Image.fromarray(bad).save(ref / "bad.png")
    out = tmp_path / "out"
    assert tool.main([str(ref), "--name", "g", "--out", str(out), "--no-debug"]) == 0
    prof = load_profile(out / "g.json")
    assert prof["image_count"] == 1 and "bad.png" in prof["excluded"]


def test_label_components_matches_flood_fill():
    rng = np.random.default_rng(3)
    b = rng.integers(0, 3, (25, 33)).astype(np.uint8)
    lab = label_components(b)
    ref = -np.ones(b.shape, int)
    k = 0
    for y in range(b.shape[0]):
        for x in range(b.shape[1]):
            if ref[y, x] >= 0:
                continue
            stack, ref[y, x] = [(y, x)], k
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < b.shape[0] and 0 <= nx < b.shape[1] and ref[ny, nx] < 0 \
                            and b[ny, nx] == b[cy, cx]:
                        ref[ny, nx] = k
                        stack.append((ny, nx))
            k += 1
    pairs = set(zip(ref.ravel(), lab.ravel()))
    assert len(pairs) == k == len(np.unique(lab))


def test_mix_pair_relations():
    from retrocore.palette import _from_4bit
    pal = _from_4bit([(4, 0, 0), (9, 2, 2), (15, 6, 6),      # red ramp dark -> light
                      (2, 4, 12), (15, 15, 15)])              # blue, white
    h, w = 60, 180
    yy, xx = np.mgrid[0:h, 0:w]
    chk = (yy + xx) % 2 == 0
    idx = np.full((h, w), 4, np.uint8)
    for x0, (a, b) in zip((0, 60, 120), ((0, 1), (0, 2), (1, 3))):
        m = (xx >= x0 + 2) & (xx < x0 + 58) & (yy >= 2) & (yy < 58)
        idx[m] = np.where(chk, a, b)[m]
    st, _ = analyze_image(Image.fromarray(pal[idx]), "mix")
    m = st["dither"]["mix"]
    rel = m["relation"]
    assert abs(sum(rel.values()) - 1) < 1e-3
    for k in ("same_ramp_adjacent", "same_ramp_far", "cross_ramp"):
        assert 0.25 < rel[k] < 0.42, rel
    assert m["pair_count"] > 0
    by = {tuple(p["colors"]): p for p in m["pairs"]}
    cross = by[("#922", "#24C")]
    assert cross["relation"] == "cross_ramp" and cross["hue_diff"] > 90
    assert by[("#400", "#F66")]["relation"] == "same_ramp_far"
    assert by[("#400", "#922")]["relation"] == "same_ramp_adjacent"
    assert abs(sum(p["share"] for p in m["pairs"]) - 1) < 1e-2
    for d in (m["lum_diff"], m["hue_diff"], m["cross_ramp"]["lum_diff"]):
        assert abs(sum(d["hist"]) - 1) < 1e-2 and len(d["hist"]) == len(d["bin_edges"]) - 1
    # Cross-ramp pairs here are a single pair: their luminance difference.
    assert abs(m["cross_ramp"]["lum_diff"]["median"] - cross["lum_diff"]) < 0.5
    prof = build_profile([st], "g")
    assert "ランプ間" in profile_summary_text(prof)
    assert "pairs" not in prof["summary"]["dither"]["mix"]       # lists of objects dropped
    assert abs(prof["summary"]["dither"]["mix"]["relation"]["cross_ramp"] - rel["cross_ramp"]) < 1e-3


def test_mix_synthetic_cg():
    idx, _, _ = make_cg()
    st, _ = analyze_image(Image.fromarray(rgb(idx)), "synth")
    m = st["dither"]["mix"]
    # Green dots on the white background: the only pair of different ramps.
    assert m["relation"]["cross_ramp"] > 0.05
    assert any(p["relation"] == "cross_ramp" and set(p["colors"]) == {"#294", "#FFF"}
               for p in m["pairs"])
    assert m["cross_ramp"]["hue_diff"]["chromatic_share"] == 0.0   # white is grey
