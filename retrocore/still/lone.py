"""孤立したドットをそろえる（Still App。CLI の `--tidy-dots`、既定は切）。

減色の後に残った、まわりから浮いた 1 ドット（AI のにじみ・ノイズ）を、まわりの色にそろえる。迷う所は変えない:

- 置き換えるのは、真ん中のドットの色がまわりの 8 ドットに 1 つも無く、まわりの 8 ドットのうち `MIN_SAME`（7）つ以上が
  同じ色 A のときだけ。そのドットを A にする（パレットの色なので、新しい色は作らない）
- いちばん暗い色（`keep` の番号。輪郭・目の点に多い）のドットと、絵の端（まわりが 8 ドットそろわない所）のドットは変えない
- 1 回だけ通す（くり返すと、細い線の先などが少しずつ削れていくため）。どのドットも、変える前の絵で判断する
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SAME = 7   # まわりの 8 ドットのうち、同じ色が要る数

# 見える化の色（置き換えたドットは濃く、まわり 1 ドットは薄く。表示だけ）
OVERLAY_DOT = (255, 0, 200)
OVERLAY_RING_ALPHA = 90


@dataclass
class Lone:
    """孤立したドットをそろえた結果。"""

    index: np.ndarray     # 高さ × 幅（uint8）。置き換えた後
    changed: np.ndarray   # 高さ × 幅（bool）。置き換えたドット

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.changed))

    @property
    def settings(self) -> dict:
        return {"min_same": MIN_SAME, "neighbors": 8, "passes": 1, "keep_darkest": True, "keep_edges": True}


def _neighbors(index: np.ndarray) -> np.ndarray:
    """内側のドット（端を除く）の、まわり 8 ドットの番号（8 × (高さ-2) × (幅-2)）。"""
    h, w = index.shape
    return np.stack([index[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])


def tidy_lone_dots(index: np.ndarray, keep: list[int] | tuple[int, ...] = (0,)) -> Lone:
    """孤立したドットを、まわりの多い色にそろえる（上の決まり。1 回だけ）。"""
    idx = np.asarray(index, np.uint8)
    h, w = idx.shape
    out = idx.copy()
    changed = np.zeros((h, w), bool)
    if h < 3 or w < 3:
        return Lone(out, changed)
    center = idx[1:-1, 1:-1]
    nb = _neighbors(idx)
    lone = ~(nb == center[None]).any(axis=0)
    for i in keep:
        lone &= center != int(i)
    # まわりに 7 つ以上ある色: 8 つのうち 7 つ以上が同じなら、その色は 1 番目か 2 番目のドットの色のどちらか
    best = np.zeros(center.shape, np.uint8)
    ok = np.zeros(center.shape, bool)
    for cand in (nb[0], nb[1]):
        same = (nb == cand[None]).sum(axis=0)
        hit = (same >= MIN_SAME) & ~ok
        best = np.where(hit, cand, best)
        ok |= hit
    take = lone & ok
    out[1:-1, 1:-1] = np.where(take, best, center)
    changed[1:-1, 1:-1] = take
    return Lone(out, changed)


def overlay_rgba(changed: np.ndarray) -> np.ndarray:
    """置き換えたドットを濃い色で、そのまわり 1 ドットを薄い色で塗った RGBA（ほかは透明。表示だけ）。"""
    m = np.asarray(changed, bool)
    h, w = m.shape
    out = np.zeros((h, w, 4), np.uint8)
    if not m.any():
        return out
    pad = np.pad(m, 1)
    ring = np.zeros((h, w), bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            ring |= pad[dy:dy + h, dx:dx + w]
    show = ring | m
    out[..., :3] = np.where(show[..., None], np.array(OVERLAY_DOT, np.uint8), 0)
    out[..., 3] = np.where(m, 255, np.where(ring, OVERLAY_RING_ALPHA, 0))
    return out
