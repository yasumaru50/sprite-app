"""作らなかった理由の記録（出力の JSON の `skipped`）。

動きを付けなかったとき（`MotionResult.skipped`）と、変換の工程が警告を出したときに、その理由を形のある辞書にする。
新しい判定はしない。今ある警告（表の問題 `timeline.Problem`、うなずくの「首が見つからない」、パレットの警告）を写すだけ。

1 件の形（設計は docs/sprite-app-plan.md の「作らなかった理由の記録（skipped）」）:

    {"stage": "motion", "code": "split_not_found", "message": "2・4 行目（upper_y = −1、範囲 upper）: …",
     "frame": 1, "frames": [1, 3], "row": 2, "rows": [2, 4], "track": "upper_y", "region": "upper",
     "value": -1, "values": [-1], "rows_from_bottom": [12, 16],
     "suggestions": [{"breathe_split": 11}], "result": "held"}

- `frame`・`frames` はシートの JSON の `index` と同じ 0 から、`row`・`rows` は表と警告と同じ 1 から（`frame` = `row` − 1）
- `suggestions` は「○○ で試す」と同じ候補（実際に作って確かめた数だけ）。プリセットは CLI の引数の名前
  （`{"nod_split": 14}`）、表は範囲の名前と分け目の高さ（`{"region": "upper", "at": 14}`）か帯・四角の端（`{"region": "head",
  "edges": {...}}`）。無ければ空
- `result` はそのフレーム（変換の工程なら、その色など）をどうしたか: `held`（動かさずに元の絵のまま）、`merged`（パレットの
  同じ色にまとめた）、`nearest`（いちばん近いパレットの色にした）
"""

from __future__ import annotations

import json

from .sheet import SHEET_VERSION

STAGES = ("motion", "grid", "cleanup", "outline", "palette", "frames", "same_as_first")

# code の一覧（docs/sprite-app-plan.md にも同じ表）。動きの code は表の問題の決まりの名前（`rule`）と同じ
CODES = {
    # 動き（stage = "motion"。result はどれも "held"）
    "split_not_found": "分けられる所（縮める・伸ばす継ぎ目、横にずらす分け目）が見つからない",
    "new_color": "元の絵に無い色が出る",
    "size": "フレームの大きさが変わる",
    "foot": "足元の位置が変わる",
    "root_shift": "体全体のずれが、絵を縦にずらしただけの形にならない",
    "parts": "部品か輪郭のつながりが変わる",
    "edge_color": "シルエットの縁に輪郭以外の色が出る",
    "euler": "輪郭の輪が切れるか、絵の中に隙間ができる",
    "small_parts": "目・頬などの小さな部品の形が変わる（小さな部品を守るが入のとき）",
    "hidden": "帯・四角を動かした後ろから見えてくる所の色が分からない",
    "overlap": "帯・四角の外に出た所が、外の違う色のドットを隠す",
    "no_image": "絵が無い（不透明なドットが無い）",
    "no_neck_found": "うなずくで、首の高さを自動で決められない",
    # パレット（stage = "palette"）
    "palette_colors_merged": "違う色が、パレットの同じ色になった（result は merged）",
    "palette_color_far": "パレットにはっきり近い色が無く、いちばん近い色にした（result は nearest）",
    # コマの切り分け（stage = "frames"。`frames.py`。コマを作らなかったときの `<出力名>_<名前>_skipped.json`）
    "frame_count": "見つかったコマの数が、指定した数と合わない",
    "no_frames": "コマが見つからない（不透明な画素が無い）",
    "empty_frame": "グリッドに合わせたら、不透明なマスが無くなったコマがある",
    # 1 コマ目と同じにする範囲（stage = "same_as_first"。`same.py`。写さなかった範囲。result は "held"。foot も使う）
    "low_match": "範囲が 1 コマ目と合う位置が見つからない（合い方が低い）",
    "seam": "範囲の境目で段差・輪郭の途切れ・透明のすき間ができる",
    "hole": "範囲を写すと透明の穴ができる",
    "small_part": "範囲に入ってきた部品（腕・髪など）が目・口などの小さな部品に触れる",
    "intrusion": "範囲に入ってきた部品（腕・髪など）が大きすぎる",
    "edge": "写す中身がキャンバスの外に出る",
    "empty": "1 コマ目の範囲の中に不透明なドットが無い",
    # グリッド（stage = "grid"。今はコマの切り分けのときだけ）
    "grid_not_found": "全部のコマを合わせても、ドットの格子が見つからない",
}

# プリセットの「○○ で試す」の候補の名前（CLI の --breathe-split などと同じ。_ でつなぐ）
SUGGEST_KEYS = {"breathe": "breathe_split", "sway": "sway_split", "nod": "nod_split", "stretch": "stretch_split"}


def _suggestions(near: list, region: str | None, key: str | None) -> list[dict]:
    out = []
    for v in near or []:
        if isinstance(v, dict):
            out.append({"region": region, "edges": dict(v)})
        elif key is not None:
            out.append({key: int(v)})
        else:
            out.append({"region": region, "at": int(v)})
    return out


def motion_entry(code: str, message: str, rows: list[int], track: str | None = None, region: str | None = None,
                 values: list | None = None, rows_from_bottom: list | None = None, box: dict | None = None,
                 suggestions: list | None = None, pixels: list | None = None) -> dict:
    """動きの 1 件。`rows` は表と同じ 1 から。"""
    rows = sorted(int(r) for r in rows)
    vals = [None if v is None else int(v) for v in (values or [])]
    d = {"stage": "motion", "code": code, "message": message,
         "frame": rows[0] - 1 if rows else None, "frames": [r - 1 for r in rows],
         "row": rows[0] if rows else None, "rows": rows,
         "track": track, "region": region, "value": vals[0] if vals else None, "values": vals}
    if rows_from_bottom is not None:
        d["rows_from_bottom"] = [int(v) for v in rows_from_bottom]
    if box is not None:
        d["box"] = {k: int(v) for k, v in box.items()}
    d["suggestions"] = list(suggestions or [])
    if pixels:
        d["pixels"] = [[int(x), int(y)] for x, y in pixels]
    d["result"] = "held"
    return d


def problem_entries(problems: list, preset: str | None = None, boxes: dict | None = None) -> list[dict]:
    """表の問題（`timeline.Problem`）を記録にする。`preset` はプリセットの名前（候補を CLI の引数の名前にする）。

    `boxes` は範囲の名前 → 実際に試した帯・四角の端（数にしたもの。分かるときだけ）。
    """
    key = SUGGEST_KEYS.get(preset) if preset is not None else None
    out = []
    for q in problems:
        out.append(motion_entry(q.rule, q.text(), q.rows, q.track, q.region, q.values or [q.value],
                                q.rows_from_bottom, (boxes or {}).get(q.region),
                                _suggestions(q.near, q.region, key if preset is not None else None), q.pixels))
    return out


def palette_entries(match) -> list[dict]:
    """パレットに合わせたときの警告（`palette.PaletteMatch` の collisions・far）を記録にする。並びは今の警告と同じ。"""
    from ..palette import _hexs
    out = []
    for c in match.collisions:
        out.append({"stage": "palette", "code": "palette_colors_merged", "message": c.message(),
                    "colors": [_hexs(s) for s, _ in c.sources], "dots": [int(n) for _, n in c.sources],
                    "target": _hexs(c.target), "suggestions": [], "result": "merged"})
    for f in match.far:
        out.append({"stage": "palette", "code": "palette_color_far", "message": f.message(),
                    "colors": [_hexs(f.source)], "dots": [int(f.dots)], "target": _hexs(f.target),
                    "distance": round(float(f.distance), 3), "suggestions": [], "result": "nearest"})
    return out


def check_entries(entries) -> list[dict]:
    """画面から受け取った記録（変換の工程の分）を確かめて、そのまま使える並びにする。"""
    if entries is None:
        return []
    if not isinstance(entries, list) or not all(isinstance(e, dict) and e.get("stage") in STAGES for e in entries):
        raise ValueError("作らなかった理由の記録は、stage の付いた辞書の並びで渡してください")
    return json.loads(json.dumps(entries))


def skipped_info(result, skipped: list | None = None) -> dict:
    """動きを作らなかったときに書く JSON（`<出力名>_<動き>_skipped.json`）の辞書。シートは無いので `image` は null。

    `frame_count`・`frames` は作ろうとしたフレーム（表の行）の数と表示時間。`skipped` の `frame` はこの `index`。
    """
    planned = [int(d) for d in (result.planned_ms or result.durations)]
    return {"version": SHEET_VERSION, "motion": result.name, "image": None, "made": False,
            "frame_count": len(planned), "loop": True, "total_duration_ms": int(sum(planned)),
            "frames": [{"index": i, "duration_ms": d} for i, d in enumerate(planned)],
            "skipped": list(skipped or []) + list(result.skips)}


def skipped_json(result, skipped: list | None = None) -> str:
    """`skipped_info()` の JSON の文字列（CLI の保存とブラウザ版の保存で同じもの）。"""
    return json.dumps(skipped_info(result, skipped), ensure_ascii=False, indent=2) + "\n"

