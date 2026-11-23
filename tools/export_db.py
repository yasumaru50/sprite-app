"""Export per-image data of the reference CG for the database site.

    python tools/export_db.py                       # every samples/reference/<group>/
    python tools/export_db.py path/to/dir --group mygroup
    python tools/export_db.py --out out/db

For every image one record is written (images themselves are not exported):

    out/db/records/<group>/<image stem>.json   one record per image
    out/db/palettes/<group>/<image stem>.gpl   palette (GIMP), by share
    out/db/index.csv                           one row per image (UTF-8 with BOM)

A record holds the result of ``check_reference`` (16 colours / 16 levels),
the number of colours, the palette (``#RGB`` and share), the dither ratio and
pattern breakdown, the colour pairs mixed by dither (``dither.mix``), the
outline ratios, the flat-paint ratio and the number of lines.  Images that
fail ``check_reference`` get a record with ``check.ok == false`` and no
statistics, unless ``--allow-screenshots`` is given.  The record layout is
described in ``docs/db_export.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrocore.analyzer import analyze_native, check_reference  # noqa: E402
from retrocore.screenshot import normalize_screenshot  # noqa: E402

SCHEMA = "retro98-db-record"
VERSION = 1
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

CSV_COLUMNS = [
    "id", "group", "file", "check_ok", "check_reason", "width", "height", "lines",
    "n_colors", "palette",
    "dither_ratio", "pattern_checker", "pattern_tile", "pattern_other",
    "class_checker", "class_tile2x2", "class_stripe", "class_tile4x4", "class_ordered_mix",
    "class_irregular",
    "mix_same_ramp_adjacent", "mix_same_ramp_far", "mix_cross_ramp",
    "mix_lum_diff_median", "mix_cross_lum_diff_median", "mix_cross_hue_diff_median",
    "mix_top_pair", "mix_top_pair_share",
    "outline_dark_edge_ratio", "outline_pixel_ratio", "outline_color",
    "flat_ratio", "isolated_ratio", "record", "gpl",
]


def collect(paths: list[str], group: str | None) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    if not paths:
        base = ROOT / "samples" / "reference"
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if files:
                groups[d.name] = files
        return groups
    for p in map(Path, paths):
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) \
            if p.is_dir() else [p]
        name = group or (p.name if p.is_dir() else p.parent.name)
        groups.setdefault(name, []).extend(files)
    return groups


def write_gpl(palette: list[dict], path: Path, name: str) -> None:
    """GIMP palette; the colour name is ``#RGB share%`` (readable by ``load_palette``)."""
    lines = ["GIMP Palette", f"Name: {name}", "Columns: 16", "#"]
    for e in palette:
        r, g, b = e["rgb"]
        lines.append(f"{r:3d} {g:3d} {b:3d}\t{e['hex']} {e['share'] * 100:.2f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_record(group: str, path: Path, allow_screenshots: bool = False) -> tuple[dict, dict | None]:
    """``(record, stats)``; ``stats`` is the full analyzer output (``None`` if skipped)."""
    img = Image.open(path)
    img.load()
    reason = check_reference(img)
    rec: dict = {
        "schema": SCHEMA,
        "version": VERSION,
        "id": f"{group}/{path.stem}",
        "group": group,
        "file": path.name,
        "check": {"ok": reason is None, "reason": reason},
        "width": img.width,
        "height": img.height,
    }
    if reason and not allow_screenshots:
        rec["analyzed"] = False
        return rec, None
    native = normalize_screenshot(img)
    st = analyze_native(native, path.name, img.size)
    c, d, o, f, res = st["colors"], st["dither"], st["outline"], st["fill"], st["resolution"]
    rec.update({
        "analyzed": True,
        "recovered": reason is not None,
        "lines": res["lines"],
        "native_width": res["native_width"],
        "native_height": res["native_height"],
        "n_colors": c["n_colors"],
        "palette": [{"hex": e["hex"], "rgb": e["rgb"], "share": e["share"]} for e in c["palette"]],
        "ramps": [{"colors": it["colors"], "hue_deg": it["hue_deg"], "share": it["share"]}
                  for it in st["ramps"]["items"]],
        "dither": {
            "ratio": d["ratio"],
            "pattern_mix": d["pattern_mix"],
            "groups": d["groups"],
            "classes": d["classes"],
        },
        "mix": d["mix"],
        "outline": {
            "dark_edge_ratio": o["dark_edge_ratio"],
            "outline_pixel_ratio": o["outline_pixel_ratio"],
            "color": o["color"],
            "thickness_median": o["thickness"]["median"],
        },
        "fill": {
            "flat_ratio": f["flat_ratio"],
            "isolated_ratio": f["isolated_ratio"],
            "median_region_pixels": f["median_region_pixels"],
        },
    })
    return rec, st


def csv_row(rec: dict) -> dict:
    row = {k: "" for k in CSV_COLUMNS}
    row.update({
        "id": rec["id"], "group": rec["group"], "file": rec["file"],
        "check_ok": int(rec["check"]["ok"]), "check_reason": rec["check"]["reason"] or "",
        "width": rec["width"], "height": rec["height"],
        "record": rec.get("record_path", ""), "gpl": rec.get("gpl_path", ""),
    })
    if not rec.get("analyzed"):
        return row
    d, m, o, f = rec["dither"], rec["mix"], rec["outline"], rec["fill"]
    top = m["pairs"][0] if m["pairs"] else None
    row.update({
        "lines": rec["lines"], "n_colors": rec["n_colors"],
        "palette": " ".join(e["hex"] for e in rec["palette"]),
        "dither_ratio": d["ratio"],
        "pattern_checker": d["pattern_mix"]["checker"],
        "pattern_tile": d["pattern_mix"]["tile"],
        "pattern_other": d["pattern_mix"]["other"],
        **{f"class_{k}": v for k, v in d["classes"].items()},
        "mix_same_ramp_adjacent": m["relation"]["same_ramp_adjacent"],
        "mix_same_ramp_far": m["relation"]["same_ramp_far"],
        "mix_cross_ramp": m["relation"]["cross_ramp"],
        "mix_lum_diff_median": m["lum_diff"]["median"],
        "mix_cross_lum_diff_median": m["cross_ramp"]["lum_diff"]["median"],
        "mix_cross_hue_diff_median": m["cross_ramp"]["hue_diff"]["median"],
        "mix_top_pair": "+".join(top["colors"]) if top else "",
        "mix_top_pair_share": top["share"] if top else "",
        "outline_dark_edge_ratio": o["dark_edge_ratio"],
        "outline_pixel_ratio": o["outline_pixel_ratio"],
        "outline_color": o["color"],
        "flat_ratio": f["flat_ratio"], "isolated_ratio": f["isolated_ratio"],
    })
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="画像ファイルまたはフォルダ (既定: samples/reference/*/)")
    ap.add_argument("--group", help="グループ名 (paths 指定時。既定はフォルダ名)")
    ap.add_argument("--allow-screenshots", action="store_true",
                    help="16色・16段階でない画像も、拡大された画像として格子を推定して統計を出す")
    ap.add_argument("--out", default=str(ROOT / "out" / "db"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    rows = []
    for group, files in collect(args.paths, args.group).items():
        for f in files:
            rec, _ = make_record(group, f, args.allow_screenshots)
            rec_path = Path("records") / group / f"{f.stem}.json"
            if rec.get("analyzed"):
                gpl_path = Path("palettes") / group / f"{f.stem}.gpl"
                (out / gpl_path).parent.mkdir(parents=True, exist_ok=True)
                write_gpl(rec["palette"], out / gpl_path, rec["id"])
                rec["gpl_path"] = gpl_path.as_posix()
            rec["record_path"] = rec_path.as_posix()
            (out / rec_path).parent.mkdir(parents=True, exist_ok=True)
            (out / rec_path).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            rows.append(csv_row(rec))
            state = "OK" if rec["check"]["ok"] else f"条件外 ({rec['check']['reason']})"
            print(f"  {rec['id']}: {state}" + ("" if rec.get("analyzed") else "  統計なし"))
    if not rows:
        print("画像がありません")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out} ({len(rows)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
