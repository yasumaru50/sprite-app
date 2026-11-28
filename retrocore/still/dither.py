"""ディザを作る（Still App の `--dither 2x2`・`4x4`。どの初めの設定でも既定は off）。

写真の空のようになめらかに色が変わる所では、減色すると色が大きな段に分かれる。そこで、隣り合う 2 色の間を
決まった網目（Bayer 型）でつなぎ、90 年代の PC の取り込み画像のようなグラデーションにする。
減色（と似た色をまとめる）の後、孤立したドットをそろえる前に、ドットに置く色を選び直す（`convert.convert_still()`）。

手順（`make_dither()`）:

1. **なめらかな所を見つける**: 減色する前のドットの色（OKLab）で、上下・左右に隣り合うドットの色の差を測る。
   まわり 5×5（`WINDOW`）のどこにも差が `EDGE_MAX`（0.03）以上の所が無いドットを「なめらか」とする。
   輪郭・はっきりした境目・細かい描き込み（木の葉など）は差が大きいので外れる。なめらかな所が 4 近傍でつながった
   かたまりが `MIN_AREA`（64）ドットより小さいときも外す（小さな平らな所に網目を入れない）
2. **2 色を選ぶ**: なめらかなドットごとに、まわり 5×5 の色の平均を測る。1 色目は平均にいちばん近いパレットの色、
   2 色目は、1 色目に近く（OKLab の距離が `PAIR_MAX`（0.15）未満）、1 色目との間に平均をはさむ色のうち、
   平均が 2 色を結ぶ線にいちばん近い色。入れるのは、平均が線のそば（線からの離れ方が 2 色の距離の `OFF_LINE_MAX`（0.5）倍以下）
   にあるときだけ。離れた色どうし・いちばん暗い色（輪郭に多い）は混ぜない。
   （いちばん近い 2 色をそのまま使うと、明るさがほぼ同じで色みだけ違う 2 色が選ばれ、空の段がつながらなかった）
3. **濃さ**: 平均が 2 色の間のどこにあるか（暗い方 0〜明るい方 1）を、網目の段（2×2 は 5 段、4×4 は 17 段）の
   いちばん近い段にする
4. **網目**: Bayer の表（2×2 は `pattern.BAYER2`、4×4 は `BAYER4`）で、段の数より小さい所を明るい方の色にする。
   表の位置は絵の左上を基準にした、絵全体で共通の座標（模様を整える `pattern.py` と同じ考え方）。
   色を置き換えるのは、ディザなしでその 2 色のどちらかだったドットだけ（3 色目のドットは変えない）

新しい色は作らない（パレットはディザなしのときと同じ。どのドットにどちらの色を置くかだけを変える）。
迷ったら入れない（どの条件にも当てはまらない所はディザなしのまま）。
計算は、色を OKLab の 1/4096 単位の整数にしてから整数で行う（CLI とブラウザ版で同じ結果にするため）。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..sprite.simplify import rgb_to_oklab
from .pattern import BAYER2, label_regions

CHOICES = ("off", "2x2", "4x4")
LABELS = {"off": "しない", "2x2": "2×2（5 段）", "4x4": "4×4（17 段）"}
WINDOW = 5            # なめらかさと、色の平均を測るまわりの大きさ（5×5）
EDGE_MAX = 0.03       # 隣り合うドットの色の差（OKLab）がこれ以上の所は、なめらかではない
MIN_AREA = 64         # なめらかな所のかたまりが、これより小さいときは入れない
PAIR_MAX = 0.15       # 混ぜる 2 色の距離（OKLab）の上限
OFF_LINE_MAX = 0.5    # 平均の、2 色を結ぶ線からの離れ方の上限（2 色の距離の何倍か）
SCALE = 4096          # OKLab を整数にするときの倍率
# 4×4 の Bayer の表（[y % 4][x % 4]）。段 n（0〜16）では、表の値が n より小さい所が明るい方の色
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.int8)
TILES = {"2x2": BAYER2, "4x4": BAYER4}

# 見える化の色（網目で色を変えたドットは濃く、網目の範囲のほかのドットは薄く。表示だけ）
OVERLAY_DOT = (0, 200, 120)
OVERLAY_AREA_ALPHA = 80


@dataclass
class Dither:
    """ディザを作った結果。"""

    index: np.ndarray     # 高さ × 幅（uint8）。網目を入れた後（パレットは入れる前と同じ）
    area: np.ndarray      # 高さ × 幅（bool）。網目の範囲（2 色の間の段が 0 と最大のどちらでもない所）
    changed: np.ndarray   # 高さ × 幅（bool）。網目で色を変えたドット
    kind: str             # "2x2" か "4x4"
    pairs: int            # 網目でつないだ 2 色の組の数

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.changed))

    @property
    def settings(self) -> dict:
        n = len(TILES[self.kind]) ** 2
        return {"tile": "bayer" + self.kind, "levels": n + 1, "window": WINDOW, "edge_max": EDGE_MAX,
                "min_area": MIN_AREA, "pair_max": PAIR_MAX, "off_line_max": OFF_LINE_MAX, "space": "OKLab"}

    def describe(self) -> dict:
        return {"kind": self.kind, "changed": self.count, "area": int(np.count_nonzero(self.area)), "pairs": self.pairs,
                "settings": self.settings}


def _lab_int(rgb: np.ndarray) -> np.ndarray:
    """RGB（… × 3、uint8）を、OKLab の 1/4096 単位の整数（int64）にする。同じ色は 1 回だけ計算する。"""
    a = np.asarray(rgb, np.uint8).reshape(-1, 3)
    key = (a[:, 0].astype(np.int64) << 16) | (a[:, 1].astype(np.int64) << 8) | a[:, 2].astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.float64)
    lab = np.rint(np.round(rgb_to_oklab(cols), 9) * SCALE).astype(np.int64)
    return lab[inv.reshape(-1)].reshape(np.asarray(rgb).shape)


def _box(a: np.ndarray, n: int) -> np.ndarray:
    """n×n（n は奇数）の和（int64）。外は 0 とみなす。"""
    r = n // 2
    h, w = a.shape
    p = np.zeros((h + n, w + n), np.int64)
    p[r + 1:r + 1 + h, r + 1:r + 1 + w] = a
    p = p.cumsum(0).cumsum(1)
    return p[n:, n:] - p[:-n, n:] - p[n:, :-n] + p[:-n, :-n]


def smooth_mask(lab: np.ndarray) -> np.ndarray:
    """なめらかなドット（手順 1）。lab は `_lab_int()` の値。"""
    h, w = lab.shape[:2]
    lim = int(round(EDGE_MAX * SCALE)) ** 2
    edge = np.zeros((h, w), np.int64)
    if w > 1:
        e = (((lab[:, 1:] - lab[:, :-1]) ** 2).sum(-1) >= lim).astype(np.int64)
        edge[:, 1:] += e
        edge[:, :-1] += e
    if h > 1:
        e = (((lab[1:] - lab[:-1]) ** 2).sum(-1) >= lim).astype(np.int64)
        edge[1:] += e
        edge[:-1] += e
    smooth = _box(edge, WINDOW) == 0
    if smooth.any():
        lab_id = label_regions(smooth.astype(np.int8), smooth)
        ids, sizes = np.unique(lab_id[smooth], return_counts=True)
        smooth &= np.isin(lab_id, ids[sizes >= MIN_AREA])
    return smooth


def _pairs(mean: np.ndarray, pl: np.ndarray):
    """平均の色（n × 3、整数）ごとに、混ぜる 2 色を選ぶ（手順 2・3）。

    戻り値: (1 色目, 2 色目, 明るい方の割合 × 距離², 2 色の距離², 入れてよいか)。
    1 色目は平均にいちばん近い色（同じ近さなら番号の小さい方）。2 色目は、近く（距離 < PAIR_MAX）、
    1 色目との間に平均をはさみ、平均が 2 色を結ぶ線にいちばん近い色（同じなら番号の小さい方）。
    """
    n, k = len(mean), len(pl)
    near = np.zeros(n, np.intp)
    dn = np.full(n, np.iinfo(np.int64).max, np.int64)
    for j in range(k):
        d = ((mean - pl[j]) ** 2).sum(-1)
        g = d < dn
        near = np.where(g, j, near)
        dn = np.where(g, d, dn)
    rel = mean - pl[near]
    rel2 = (rel ** 2).sum(-1)
    pair_lim = int(round(PAIR_MAX * SCALE)) ** 2
    other = np.full(n, -1, np.intp)
    b_num = np.zeros(n, np.int64)    # 線の上の位置（1 色目から）× 距離²
    b_dd = np.ones(n, np.int64)      # 2 色の距離²
    b_off = np.zeros(n, np.int64)    # 線からの離れ方² × 距離²
    for j in range(k):
        v = pl[j] - pl[near]
        dd = (v ** 2).sum(-1)
        num = (rel * v).sum(-1)
        off = rel2 * dd - num ** 2
        cand = (dd > 0) & (dd < pair_lim) & (num > 0) & (num <= dd)
        better = cand & ((other < 0) | (off * b_dd < b_off * dd))
        other = np.where(better, j, other)
        b_num = np.where(better, num, b_num)
        b_dd = np.where(better, dd, b_dd)
        b_off = np.where(better, off, b_off)
    f = Fraction(OFF_LINE_MAX).limit_denominator(100)
    valid = (other >= 0) & (b_off * f.denominator ** 2 <= f.numerator ** 2 * b_dd * b_dd)
    other = np.where(other >= 0, other, near)
    # パレットは暗い順なので、番号の小さい方が暗い。明るい方の割合は、1 色目が暗い方なら線の上の位置そのもの
    num_light = np.where(near <= other, b_num, b_dd - b_num)
    return near, other, num_light, b_dd, valid


def make_dither(rgb: np.ndarray, index: np.ndarray, palette: np.ndarray, kind: str = "2x2",
                keep: list[int] | tuple[int, ...] = (0,)) -> Dither:
    """減色する前のドットの色 rgb（高さ × 幅 × 3）と、ディザなしの index・palette から、網目を入れた index を作る。"""
    if kind not in TILES:
        raise ValueError(f"ディザ（--dither）は {', '.join(CHOICES)} のどれかです（{kind}）")
    idx = np.asarray(index, np.uint8)
    h, w = idx.shape
    pal = np.asarray(palette, np.uint8).reshape(-1, 3)
    k = len(pal)
    tile = TILES[kind]
    n_lv = tile.size   # 段の数 − 1（2×2 は 4、4×4 は 16）
    out = idx.copy()
    empty = np.zeros((h, w), bool)
    if k < 2 or h == 0 or w == 0:
        return Dither(out, empty, empty.copy(), kind, 0)
    lab = _lab_int(rgb)
    smooth = smooth_mask(lab)
    if not smooth.any():
        return Dither(out, empty, empty.copy(), kind, 0)
    # まわり 5×5 の色の平均（1/4096 単位の整数に丸める）
    cnt = _box(np.ones((h, w), np.int64), WINDOW)
    mean = np.stack([(2 * _box(lab[..., c], WINDOW) + cnt) // (2 * cnt) for c in range(3)], axis=-1)
    pl = _lab_int(pal[None])[0]
    # 2 色を選ぶのは平均の色だけで決まるので、なめらかなドットの平均の色ごとに 1 回だけ計算する（速くするため）
    sy, sx = np.nonzero(smooth)
    ms = mean[sy, sx]
    key = ((ms[:, 0] + 2 * SCALE) << 32) | ((ms[:, 1] + 2 * SCALE) << 16) | (ms[:, 2] + 2 * SCALE)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.reshape(-1).astype(np.intp)
    um = ms[first]
    near, other, num_light, dd, valid = _pairs(um, pl)
    a_u = np.minimum(near, other)
    b_u = np.maximum(near, other)
    lv_u = np.clip((2 * n_lv * num_light + dd) // (2 * dd), 0, n_lv)
    for i in keep:
        valid &= (a_u != int(i)) & (b_u != int(i))
    # 絵の大きさに戻す（なめらかでないドットは入れない）
    a = np.zeros((h, w), np.intp)
    b = np.zeros((h, w), np.intp)
    lv = np.zeros((h, w), np.int64)
    ok = np.zeros((h, w), bool)
    a[sy, sx] = a_u[inv]
    b[sy, sx] = b_u[inv]
    lv[sy, sx] = lv_u[inv]
    ok[sy, sx] = valid[inv]
    ok &= (idx == a) | (idx == b)   # ディザなしで 2 色のどちらかだったドットだけ（3 色目は変えない）
    yy = np.arange(h, dtype=np.intp) % tile.shape[0]
    xx = np.arange(w, dtype=np.intp) % tile.shape[1]
    light = tile[yy[:, None], xx[None, :]] < lv
    new = np.where(light, b, a).astype(np.uint8)
    out = np.where(ok, new, idx).astype(np.uint8)
    area = ok & (lv > 0) & (lv < n_lv)
    changed = out != idx
    if area.any():
        used = np.unique(a[area].astype(np.int64) * k + b[area].astype(np.int64))
        pairs = int(len(used))
    else:
        pairs = 0
    return Dither(out, area, changed, kind, pairs)


def overlay_rgba(changed: np.ndarray, area: np.ndarray) -> np.ndarray:
    """網目で色を変えたドットを濃い色で、網目の範囲のほかのドットを薄い色で塗った RGBA（ほかは透明。表示だけ）。"""
    m = np.asarray(changed, bool)
    ar = np.asarray(area, bool) | m
    h, w = m.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.where(ar[..., None], np.array(OVERLAY_DOT, np.uint8), 0)
    out[..., 3] = np.where(m, 255, np.where(ar, OVERLAY_AREA_ALPHA, 0))
    return out
