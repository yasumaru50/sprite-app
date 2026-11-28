"""比率に合わせて切り抜く（Still App。`--aspect W:H`・`--anchor X,Y`）。

スマホの写真（4:3・縦長）や AI の絵（3:2）を、16:9（ゲームの背景）・16:10（640×400）などの比率に、
**拡大・縮小なしで、切り抜くだけで** 合わせる（余白を足したいときは `--canvas`）。切る位置は `anchor`
（横・縦それぞれ 0〜100 の%。50,50 は中央、50,0 は上寄せ、0,50 は左寄せ）。

切り抜く時点はマス目の決め方で違う:

- 横のマス数で分けるとき（`input_box()`）: 入力の段階で切り抜いてから工程に通す。仕上がりの縦のマス数を
  `H = 横のマス数 × 高さ ÷ 幅` の四捨五入（.5 は上）で先に決め、入力を ちょうど 横のマス数:H の比の、いちばん大きい四角に切る
  （幅 320・16:9 なら 320×180）。横のマス数で割り切れないときの比のずれは、この四捨五入の分だけ
- マス目を探すとき（`cell_box()`）: マス目を見つけたあと、マス単位で切る（格子をずらさない）。
  残すマスの数は、長すぎる方の辺を `もう一方 × 比` の四捨五入（.5 は上）にしたもの

切る量の左右（上下）の分け方は `切る量 × anchor ÷ 100` の四捨五入（.5 は上）を左（上）に。計算は整数だけ。
"""

from __future__ import annotations

import re

MAX_TERM = 100000   # 比の数の上限（1200:630 のような大きさの書き方も入るように）
DEFAULT_ANCHOR = (50, 50)
# ブラウザ版のよく使う比率の選択（「元のまま」と「数値で入れる」はひな形の側）
CHOICES = ((16, 9), (16, 10), (4, 3), (3, 2), (1, 1))
WIDTH_CHOICES = {(16, 9): (320, 480, 640)}   # この比率のときに出す、横のマス数の候補
_NONE = ("", "none", "off", "元のまま")


def _round_div(num: int, den: int) -> int:
    """num ÷ den の四捨五入（.5 は上。どちらも 0 以上）。"""
    return (2 * num + den) // (2 * den)


def parse_aspect(value) -> tuple[int, int] | None:
    """'16:9'・[16, 9] を (16, 9) にする。'元のまま'・'none'・None は None（切り抜かない）。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NONE:
            return None
        parts = re.split(r"\s*[:：]\s*", text)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    try:
        if len(parts) != 2 or any(isinstance(p, bool) for p in parts):
            raise ValueError
        nums = []
        for p in parts:
            if isinstance(p, float) and not p.is_integer():
                raise ValueError
            s = str(int(p)) if isinstance(p, (int, float)) else str(p).strip()
            if not s.isdigit() or not s.isascii():
                raise ValueError
            nums.append(int(s))
    except (ValueError, TypeError):
        raise ValueError(f"比率（--aspect）は 幅:高さ を正の整数どうしで指定してください（例 16:9・21:9・1200:630。"
                         f"いまは「{value}」）") from None
    a, b = nums
    if not (1 <= a <= MAX_TERM and 1 <= b <= MAX_TERM):
        raise ValueError(f"比率（--aspect）の数は 1〜{MAX_TERM} の整数で指定してください（いまは {a}:{b}）")
    return a, b


def parse_anchor(value) -> tuple[int, int]:
    """'50,20'・[50, 20] を (50, 20) にする（横・縦の%。0〜100 の整数）。None は 50,50。"""
    if value is None:
        return DEFAULT_ANCHOR
    parts = re.split(r"\s*[,，、]\s*", value.strip()) if isinstance(value, str) else (
        list(value) if isinstance(value, (list, tuple)) else [value])
    try:
        if len(parts) != 2 or any(isinstance(p, bool) for p in parts):
            raise ValueError
        nums = []
        for p in parts:
            if isinstance(p, float) and not p.is_integer():
                raise ValueError
            s = str(int(p)) if isinstance(p, (int, float)) else str(p).strip()
            if not s.isdigit() or not s.isascii():
                raise ValueError
            nums.append(int(s))
    except (ValueError, TypeError):
        raise ValueError(f"切る位置（--anchor）は 横,縦 を 0〜100 の整数（%）で指定してください（例 50,50 は中央・50,0 は上寄せ。"
                         f"いまは「{value}」）") from None
    x, y = nums
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError(f"切る位置（--anchor）は 0〜100 の%で指定してください（いまは {x},{y}）")
    return x, y


def _split(cut: int, anchor: int) -> int:
    """切る量 cut のうち、左（上）で切る量。"""
    return _round_div(cut * anchor, 100)


def target_height(width: int, aspect: tuple[int, int]) -> int:
    """横 width マスのときの仕上がりの縦のマス数（四捨五入。1 以上）。"""
    a, b = aspect
    return max(1, _round_div(width * b, a))


def input_box(w: int, h: int, width: int, aspect: tuple[int, int],
              anchor: tuple[int, int] = DEFAULT_ANCHOR) -> tuple[tuple[int, int, int, int], int]:
    """横のマス数で分けるときに、入力の w × h から切り抜く四角 (x0, y0, x1, y1) と、仕上がりの縦のマス数。

    四角は ちょうど width : 縦のマス数 の比の、いちばん大きいもの（切り抜いたあとに横 width マスに分けると、
    縦がちょうどその数になる）。
    """
    H = target_height(width, aspect)
    if w * H >= h * width:   # 入力が横に長い（か同じ）: 高さはそのまま、幅を切る
        ch, cw = h, min(w, _round_div(h * width, H))
    else:
        cw, ch = w, min(h, _round_div(w * H, width))
    cw, ch = max(1, cw), max(1, ch)
    x0, y0 = _split(w - cw, anchor[0]), _split(h - ch, anchor[1])
    return (x0, y0, x0 + cw, y0 + ch), H


def cell_box(nx: int, ny: int, aspect: tuple[int, int],
             anchor: tuple[int, int] = DEFAULT_ANCHOR) -> tuple[int, int, int, int]:
    """マス目を探したときに、nx × ny マスから残すマスの四角 (左, 上, 右, 下)（マスの番号。右・下は含まない）。"""
    a, b = aspect
    if nx * b > ny * a:   # 横に長い: 横のマスを減らす
        kx, ky = min(nx, max(1, _round_div(ny * a, b))), ny
    else:
        kx, ky = nx, min(ny, max(1, _round_div(nx * b, a)))
    x0, y0 = _split(nx - kx, anchor[0]), _split(ny - ky, anchor[1])
    return x0, y0, x0 + kx, y0 + ky


def describe(aspect: tuple[int, int], anchor: tuple[int, int], stage: str, area: tuple[int, int, int, int],
             box: tuple[int, int, int, int], picture: tuple[int, int],
             cells: tuple[int, int, int, int] | None = None, grid_cells: tuple[int, int] | None = None) -> dict:
    """JSON の `aspect`（比率・切る位置・切り抜いた時点・残した範囲・切り落とした量）。

    area は切り抜く前の範囲（入力の画素。入力の段階なら画像全体、マス単位ならマス目の範囲）、box は残した範囲。
    `cut` は area から切り落とした画素の数、マス単位のときは `cut_cells` に切り落としたマスの数も書く。
    """
    ax0, ay0, ax1, ay1 = area
    x0, y0, x1, y1 = box
    pw, ph = picture
    a, b = aspect
    d = {"ratio": [a, b], "anchor": list(anchor), "stage": stage,
         "box": [int(x0), int(y0), int(x1), int(y1)],
         "cut": {"left": int(x0 - ax0), "top": int(y0 - ay0), "right": int(ax1 - x1), "bottom": int(ay1 - y1)},
         "exact": pw * b == ph * a}
    if cells is not None:
        c0, r0, c1, r1 = cells
        nx, ny = grid_cells
        d["cut_cells"] = {"left": int(c0), "top": int(r0), "right": int(nx - c1), "bottom": int(ny - r1)}
    return d
