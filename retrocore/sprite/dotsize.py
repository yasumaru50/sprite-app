"""ドットの大きさの指定と、格子が見つからなかったときの「次に試せる大きさ」の候補。

- 画面の「ドットの大きさ」の欄と CLI の `--dot-size` は同じ `SpriteSettings.dot_size`（横, 縦）になる
  （欄は 1 つの数で、正方形のドット。`parse_dot_size()` が数・[横, 縦]・"6.5"・"6x7" を (横, 縦) にそろえる）
- 自動の検出で格子が見つからなかったとき（`GridNotFoundError`）は、推定でいちばん点が高かった大きさ（`estimate`）を
  0.5 刻みに丸めたものを軸に、0.5 刻みで近い大きさを最大 `MAX_CANDIDATES` 個並べる（`candidate_sizes()`）。
  画面は候補ごとに変換して見本の絵とキャラの高さを出し、CLI は「次に試せる大きさ」として名前だけ出す（選び方は同じ関数）

自動で見つかったときは候補を作らない。候補の数は、AI の絵で推定が半ドットほどずれることがある（ai20 の回転: 推定が
弱く、6.5 で 3 段の高さがそろった）ので、±1 画素の範囲を 0.5 刻みで見れば足りるようにした。
"""

from __future__ import annotations

import math

CANDIDATE_STEP = 0.5
MAX_CANDIDATES = 5
MIN_DOT_SIZE = 2.0      # CLI の --dot-size と同じ下限
MAX_DOT_SIZE = 256.0


def parse_dot_size(value) -> tuple[float, float] | None:
    """ドットの大きさを (横, 縦) にする。None・空の文字はそのまま None（自動）。合わなければ ValueError。

    受け取れる形: 数（6.5）、[横, 縦]、"6.5"、"6x7"（CLI の --dot-size と同じ書き方）。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"ドットの大きさは数で指定してください: {value}")
    if isinstance(value, str):
        parts = value.strip().lower().replace("×", "x").split("x")
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            raise ValueError(f"ドットの大きさは 6.5 や 6x7 の形で指定してください: {value}") from None
    elif isinstance(value, (int, float)):
        vals = [float(value)]
    elif isinstance(value, (list, tuple)):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            raise ValueError(f"ドットの大きさは [横, 縦] の数で指定してください: {value}")
        vals = [float(v) for v in value]
    else:
        raise ValueError(f"ドットの大きさは数で指定してください: {value}")
    if len(vals) == 1:
        vals = vals * 2
    if len(vals) != 2 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"ドットの大きさは 6.5 や [6, 7] の形で指定してください: {value}")
    if min(vals) < MIN_DOT_SIZE or max(vals) > MAX_DOT_SIZE:
        raise ValueError(f"ドットの大きさは {size_text(MIN_DOT_SIZE)}〜{size_text(MAX_DOT_SIZE)} 画素で指定してください: {value}")
    return vals[0], vals[1]


def candidate_sizes(estimate: float | None, count: int = MAX_CANDIDATES, step: float = CANDIDATE_STEP) -> list[float]:
    """次に試せるドットの大きさ（小さい順）。`estimate` を step 刻みに丸めた大きさを真ん中にして、前後に step ずつ。

    下限（`MIN_DOT_SIZE`）より小さいものは除く（そのぶん数が減る）。`estimate` が無ければ空。
    """
    if estimate is None or not math.isfinite(estimate) or estimate <= 0:
        return []
    center = math.floor(estimate / step + 0.5) * step
    half = (count - 1) // 2
    out = [center + k * step for k in range(-half, count - half)]
    return [v for v in out if MIN_DOT_SIZE <= v <= MAX_DOT_SIZE]


def size_text(v: float) -> str:
    """6.0 → "6"、6.5 → "6.5"（CLI の引数と画面の欄に入れる書き方）。"""
    return f"{v:g}"


def candidates_of(error) -> list[float]:
    """`GridNotFoundError` の推定から作った候補（推定が無ければ空）。"""
    return candidate_sizes(getattr(error, "estimate", None))


def cli_hint(sizes: list[float]) -> str:
    """CLI のメッセージに添える文（候補が無ければ空）。例: 次に試せる大きさ: --dot-size 6 / 6.5 / 7"""
    if not sizes:
        return ""
    return "次に試せる大きさ: --dot-size " + " / ".join(size_text(v) for v in sizes)
