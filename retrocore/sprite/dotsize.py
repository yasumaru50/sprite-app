"""ドットの大きさの指定と、格子が見つからなかったときの「次に試せる大きさ」の候補。

- 画面の「ドットの大きさ」の欄と CLI の `--dot-size` は同じ `SpriteSettings.dot_size`（横, 縦）になる
  （欄は 1 つの数で、正方形のドット。`parse_dot_size()` が数・[横, 縦]・"6.5"・"6x7" を (横, 縦) にそろえる）
- 自動の検出で格子が見つからなかったとき（`GridNotFoundError`）は、推定でいちばん点が高かった大きさ（`estimate`）から候補を作る
  （`candidate_sizes()`）。画面は候補ごとに変換して見本の絵とキャラの高さを出し、CLI は「次に試せる大きさ」として名前だけ出す（選び方は同じ関数）

候補の作り方: 推定の半分が 4 画素（自動の検出が探す下限）以上なら、半分を 0.5 刻みに丸めた所を真ん中に 0.5 刻みで 5 つと、推定そのもの（0.5 刻みに丸める）。
半分が 4 画素より小さければ、推定を真ん中に 0.5 刻みで 5 つ。

**理由**: ai21（12 コマの回転を 3 段に並べた AI の絵）は、見つからなかったときの推定が 11.7〜12.1 画素で、
利用者が 3 段の高さをそろえて選んだ 6.5 画素のほぼ 2 倍だった（AI の絵の「2 ドット」の模様の山。ai08 の推定が 12.7 で 1/2 にしたのと同じ）。
推定だけを真ん中にすると 11〜13 になり、正しい大きさが入らない。半分と推定のどちらがよいかを線の揃い方・マスの中の色のばらつきで
選ぶ案は、ai21 ではどちらも区別できなかった（揃い方 6.5 で 0.12・12 で 0.13。ばらつきは 4 画素がいちばん小さい）ので、両方を並べて人が選ぶ。
自動で見つかったときは候補を作らない。0.5 刻みにしたのは、AI の絵の推定が半ドットほどずれることがあるため。
"""

from __future__ import annotations

import math

CANDIDATE_STEP = 0.5
MAX_CANDIDATES = 5
MIN_DOT_SIZE = 2.0      # CLI の --dot-size と同じ下限
MAX_DOT_SIZE = 256.0
HALF_MIN = 4.0          # 推定の半分も候補にする下限（grid.DEFAULT_MIN_DOT。自動の検出はこれより小さい大きさを探さない）


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
    """次に試せるドットの大きさ（小さい順。重なりなし）。作り方はこのファイルの説明のとおり。`estimate` が無ければ空。

    下限（`MIN_DOT_SIZE`）より小さいもの・上限より大きいものは除く（そのぶん数が減る）。
    """
    if estimate is None or not math.isfinite(estimate) or estimate <= 0:
        return []

    def snap(v: float) -> float:
        return math.floor(v / step + 0.5) * step

    half = estimate / 2
    center = snap(half) if half >= HALF_MIN else snap(estimate)
    k0 = (count - 1) // 2
    out = [center + k * step for k in range(-k0, count - k0)]
    if half >= HALF_MIN:
        out.append(snap(estimate))
    return sorted({v for v in out if MIN_DOT_SIZE <= v <= MAX_DOT_SIZE})


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
