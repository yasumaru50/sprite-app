"""前の版の出力と比べるための「足した分を取り除く」手伝い。

作らなかった理由の記録（シートの JSON の版 2 の `skipped`、動きを作らなかったときの `<出力名>_<動き>_skipped.json`、
画面の結果の "skipped_records"・"skipped_json"、変換の結果の "skipped"）は足しただけで、ほかの項目は前と同じ。
前の出力のハッシュと比べるテストは、ここで足した分を取り除いてから比べる（前の版の出力に使っても何も変わらない）。
"""

from __future__ import annotations

import json

SKIPPED_SUFFIX = "_skipped.json"
REPORT_LINE = "作らなかった理由"


def legacy_sheet_text(text: str) -> str:
    """シートの JSON の文字列から `skipped` を除き、版を 1 に戻す（シートの JSON でなければそのまま）。"""
    try:
        d = json.loads(text)
    except ValueError:
        return text
    if not isinstance(d, dict) or "frame_width" not in d or "skipped" not in d:
        return text
    d.pop("skipped")
    d["version"] = 1
    return json.dumps(d, ensure_ascii=False, indent=2) + "\n"


def legacy_bytes(data: bytes) -> bytes:
    """ファイルのバイト列（改行は LF にそろえたもの）。シートの JSON なら `legacy_sheet_text()` を通す。"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    new = legacy_sheet_text(text)
    return data if new == text else new.encode("utf-8")


def legacy_files(paths):
    """フォルダの中のファイルから、足した `_skipped.json` を除く。"""
    return [p for p in paths if not p.name.endswith(SKIPPED_SUFFIX)]


def legacy_stdout(text: str) -> str:
    """CLI の表示から、`_skipped.json` を書いた行を除く。"""
    return "\n".join(line for line in text.split("\n") if not (REPORT_LINE in line and SKIPPED_SUFFIX in line))


def legacy_web(d: dict) -> dict:
    """画面の結果（motion_for_web・convert_for_web の辞書）から足した分を除く。"""
    d = dict(d)
    d.pop("skipped_records", None)
    d.pop("skipped_json", None)
    if isinstance(d.get("skipped"), list):   # 変換の結果の記録（動きの "skipped" は真偽なので残す）
        d.pop("skipped")
    if isinstance(d.get("sheet_json"), str):
        d["sheet_json"] = legacy_sheet_text(d["sheet_json"])
    return d
