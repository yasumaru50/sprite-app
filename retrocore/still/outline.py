"""輪郭の線を引く（Still App。CLI の `--outline`、どのモードでも既定は切）。

減色した後の絵で、明るさの差が大きい色の境目に、暗い方の側へ 1 ドットの線を引く:

- 上下左右のとなりのドットとの明るさ（OKLab の L。0〜1）の差が `MIN_DIFF` 以上で、そのドットの方が暗いとき、そのドットを線にする
- 線の色はパレットのいちばん暗い色（0 番。パレットは暗い順）。新しい色は作らない
- もともといちばん暗い色のドットは数えない（変わらないため）。どのドットも、線を引く前の絵で判断する（1 回だけ）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sprite.simplify import rgb_to_oklab

MIN_DIFF = 0.25     # 線を引く明るさの差（OKLab の L。0.2 では床のごみの縁にも線が付きすぎた）
# 見える化の色（表示だけ）
OVERLAY_DOT = (0, 200, 255)


@dataclass
class Outline:
    """輪郭の線を引いた結果。"""

    index: np.ndarray     # 高さ × 幅（uint8）。線を引いた後
    drawn: np.ndarray     # 高さ × 幅（bool）。線を引いたドット

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.drawn))

    @property
    def settings(self) -> dict:
        return {"min_diff": MIN_DIFF, "space": "OKLab L", "neighbors": 4, "side": "darker", "color": "darkest"}


def _lightness(palette: np.ndarray) -> np.ndarray:
    """パレットの各色の明るさを、1/10000 単位の整数にしたもの（CLI とブラウザ版で比べ方をそろえるため）。"""
    lab = rgb_to_oklab(np.asarray(palette, np.uint8).reshape(-1, 3))
    return np.rint(lab[:, 0] * 10000).astype(np.int64)


def draw_outline(index: np.ndarray, palette: np.ndarray) -> Outline:
    """明るさの差が大きい境目の、暗い方の側のドットを 0 番の色にする（上の決まり）。"""
    idx = np.asarray(index, np.uint8)
    light = _lightness(palette)[idx]
    limit = int(round(MIN_DIFF * 10000))
    h, w = idx.shape
    mark = np.zeros((h, w), bool)
    # 上下左右のとなりが limit 以上明るいドット
    mark[1:, :] |= light[:-1, :] - light[1:, :] >= limit
    mark[:-1, :] |= light[1:, :] - light[:-1, :] >= limit
    mark[:, 1:] |= light[:, :-1] - light[:, 1:] >= limit
    mark[:, :-1] |= light[:, 1:] - light[:, :-1] >= limit
    mark &= idx != 0
    out = np.where(mark, np.uint8(0), idx).astype(np.uint8)
    return Outline(out, mark)


def overlay_rgba(drawn: np.ndarray) -> np.ndarray:
    """線を引いたドットを色で塗った RGBA（ほかは透明。表示だけ）。"""
    m = np.asarray(drawn, bool)
    out = np.zeros(m.shape + (4,), np.uint8)
    out[m] = OVERLAY_DOT + (255,)
    return out
