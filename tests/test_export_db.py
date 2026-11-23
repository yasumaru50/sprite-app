import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from retrocore.palette import load_palette
from synth_cg import rgb

_spec = importlib.util.spec_from_file_location(
    "export_db", Path(__file__).resolve().parent.parent / "tools" / "export_db.py")
export_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_db)


def test_export_db_records_csv_and_gpl(tmp_path):
    ref = tmp_path / "grp"
    ref.mkdir()
    Image.fromarray(rgb()).save(ref / "good.png")
    bad = rgb().copy()
    bad[:5, :5] = (100, 101, 102)
    Image.fromarray(bad).save(ref / "bad.png")
    out = tmp_path / "db"
    assert export_db.main([str(ref), "--out", str(out)]) == 0

    rec = json.loads((out / "records" / "grp" / "good.json").read_text(encoding="utf-8"))
    assert rec["id"] == "grp/good" and rec["check"] == {"ok": True, "reason": None}
    assert rec["analyzed"] and rec["n_colors"] == 7 and rec["lines"] == 400
    assert abs(sum(e["share"] for e in rec["palette"]) - 1) < 1e-3
    assert set(rec["dither"]["pattern_mix"]) == {"checker", "tile", "other"}
    assert abs(sum(rec["mix"]["relation"].values()) - 1) < 1e-3 and rec["mix"]["pairs"]
    for k in ("dark_edge_ratio", "outline_pixel_ratio"):
        assert 0 <= rec["outline"][k] <= 1
    assert 0 <= rec["fill"]["flat_ratio"] <= 1
    # No image data in the export.
    assert not list(out.rglob("*.png"))
    assert "base64" not in json.dumps(rec)

    pal = load_palette(out / rec["gpl_path"])
    assert [tuple(c) for c in pal] == [tuple(e["rgb"]) for e in rec["palette"]]

    bad_rec = json.loads((out / "records" / "grp" / "bad.json").read_text(encoding="utf-8"))
    assert not bad_rec["check"]["ok"] and not bad_rec["analyzed"] and "gpl_path" not in bad_rec

    with open(out / "index.csv", encoding="utf-8-sig") as fh:
        rows = {r["file"]: r for r in csv.DictReader(fh)}
    assert set(rows) == {"good.png", "bad.png"}
    g = rows["good.png"]
    assert g["check_ok"] == "1" and g["n_colors"] == "7" and g["palette"].startswith("#")
    assert float(g["mix_cross_ramp"]) == rec["mix"]["relation"]["cross_ramp"]
    assert rows["bad.png"]["check_ok"] == "0" and rows["bad.png"]["dither_ratio"] == ""
