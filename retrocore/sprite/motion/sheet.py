"""動きの書き出し: 横に並べたスプライトシート（透明 PNG）と、フレームの情報を書いた JSON。

Unity と Godot のどちらでも読みやすいように、素直な形にする:
- シートはフレームを左から順に、すき間なく横 1 列に並べる（拡大しない。1 ドット＝1 画素）
- JSON にはフレームの大きさ・枚数・各フレームの位置と表示時間（ミリ秒）、足元の位置（`pivot`。左上が原点、下向きが +y の画素）

    {"version": 1, "motion": "breathe", "image": "hero_breathe_sheet.png",
     "frame_width": 49, "frame_height": 45, "frame_count": 2, "columns": 2, "rows": 1, "loop": true,
     "pivot": {"x": 24.5, "y": 45}, "total_duration_ms": 1000,
     "frames": [{"index": 0, "x": 0, "y": 0, "width": 49, "height": 45, "duration_ms": 500}, ...]}

横揺れ（`motion` が "sway"）では、揺れてはみ出す分として元のスプライトの左右に足した透明の列の数を `padding` に書く
（`{"left": 1, "right": 1, "top": 0, "bottom": 0}`。足していなければ 0）。`frame_width` は足したあとの幅。呼吸の JSON には書かない。

浮遊（`motion` が "float"。足元も動く例外の動き）では、上下に動いてはみ出す分として上と下に足した透明の行の数を `padding`
（`{"left": 0, "right": 0, "top": 1, "bottom": 1}`）に、フレームごとの上下のずれ（ドット。下向きが +、上が −）を
`frames[].offset_y` に書く。`pivot.y` は 1 枚目の足元（元の絵の下端）なので、フレームの高さから下に足した行の数を引いた値。
ゲームでは pivot を地面に置き、影や当たり判定は pivot のまま動かさないか、`offset_y` だけ動かして合わせる。

伸び（`motion` が "stretch"）と、表で縦の変形の +（伸ばす）を使ったときは、上に伸びてはみ出す分として上に足した透明の行の数を
`padding.top` に書く（`offset_y` は書かない。体全体は動かないので、`pivot` はフレームの下端のまま）。

動きの表（`--motion-spec`）から作ったときも同じ決まりで書く: `padding` は横の位置・体全体のずれ・伸ばすのどれかを使ったときだけ
（左右は同じ数）、`offset_y` は体全体のずれを使ったときだけ、`pivot` は (幅 / 2, 高さ − 下に足した行)。範囲の高さが
`"auto"` か割合のときは、実際に使った高さと線をいちばん外の `regions_used` に書く（今の `--motion breathe` などには書かない）。

いちばん外の `skipped`（版 2 から）は作らなかった理由の記録の一覧（`report.py`。何もなければ `[]`）。シートを作れたときは
変換の工程（パレットなど）の分だけが入る。動きを付けなかったときはシートを書かず、同じ形の記録を `<出力名>_<動き>_skipped.json`
（`report.skipped_json()`）に書く。

Unity ではシートを Sprite Mode = Multiple にして Grid By Cell Size（frame_width × frame_height）で切り、
Pivot は (pivot.x / frame_width, 1 − pivot.y / frame_height)。Godot では SpriteFrames に Hframes = frame_count で読み、
各フレームの表示時間は duration_ms から決める。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..image import save_rgba
from .base import MotionResult

SHEET_VERSION = 2   # 2: いちばん外に作らなかった理由の記録 `skipped` を足した（ほかの項目は 1 と同じ）


def make_sheet(frames: list[np.ndarray]) -> np.ndarray:
    """フレームを左から順に横 1 列に並べた RGBA 配列（すき間なし、拡大なし）。"""
    if not frames:
        raise ValueError("フレームがありません")
    shape = frames[0].shape
    if any(f.shape != shape for f in frames):
        raise ValueError("フレームの大きさがそろっていません")
    return np.concatenate([np.asarray(f, np.uint8) for f in frames], axis=1)


def sheet_info(result: MotionResult, image_name: str, skipped: list | None = None) -> dict:
    """シートの情報（JSON にする辞書）。`skipped` は変換の工程の記録（`report.py`。動きの記録 `result.skips` の前に入れる）。"""
    w, h = result.size
    n = len(result.frames)
    info = {
        "version": SHEET_VERSION,
        "motion": result.name,
        "image": image_name,
        "frame_width": w,
        "frame_height": h,
        "frame_count": n,
        "columns": n,
        "rows": 1,
        "loop": True,
        "pivot": {"x": w / 2, "y": h},
        "total_duration_ms": int(sum(result.durations)),
        "frames": [{"index": i, "x": i * w, "y": 0, "width": w, "height": h, "duration_ms": int(d)}
                   for i, d in enumerate(result.durations)],
    }
    # padding は横の位置（横揺れ）・体全体のずれ（浮遊）を使ったときだけ書く（左右は同じ数）
    p = int(result.info.get("padding", 0))
    if "padding" in result.info:
        # 横揺れ: 揺れてはみ出さないように、元のスプライトの左右に足した透明の列の数（全フレーム同じ。pivot はそのまま真ん中）
        info["padding"] = {"left": p, "right": p, "top": 0, "bottom": 0}
    if "padding_y" in result.info:
        # 浮遊: 上下に足した透明の行の数。pivot は 1 枚目の足元（元の絵の下端の中央）
        top, bottom = (int(v) for v in result.info["padding_y"])
        info["padding"] = {"left": p, "right": p, "top": top, "bottom": bottom}
        info["pivot"] = {"x": w / 2, "y": h - bottom}
        offsets = result.info.get("offset_y")   # 表から作ったときは offset_y（体全体のずれを使ったときだけ）
        if offsets is None and result.name in ("float", "bounce"):
            offsets = result.info.get("offsets")
        for f, dy in zip(info["frames"], offsets or []):
            f["offset_y"] = int(dy)   # このフレームで絵が元の絵から縦にずれた量（下向きが +）
    if "regions_used" in result.info:
        # 表（--motion-spec）の "auto"・割合の範囲で、実際に使った高さと線（同じ結果を作り直せるように）
        info["regions_used"] = result.info["regions_used"]
    # 作らなかった理由の記録（版 2 から。何もなければ空の一覧）。シートを作れたときは、変換の工程の分だけが入る
    info["skipped"] = list(skipped or []) + list(result.skips)
    return info


def sheet_json(result: MotionResult, image_name: str, skipped: list | None = None) -> str:
    """JSON ファイルの中身（文字列。保存するときは UTF-8）。CLI の保存とブラウザ版の保存で同じものを使う。"""
    return json.dumps(sheet_info(result, image_name, skipped), ensure_ascii=False, indent=2) + "\n"


def save_sheet(result: MotionResult, png_path: str | Path, skipped: list | None = None) -> tuple[Path, Path]:
    """シートの PNG と、同じ名前の .json を保存する。保存したパス (PNG, JSON) を返す。"""
    png_path = Path(png_path)
    save_rgba(make_sheet(result.frames), png_path)
    json_path = png_path.with_suffix(".json")
    json_path.write_text(sheet_json(result, png_path.name, skipped), encoding="utf-8")
    return png_path, json_path
