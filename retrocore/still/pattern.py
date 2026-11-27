"""ざらついた範囲を、2 色の決まった模様に整える（Still App の `--tidy-pattern`。既定は切）。

AI が「ディザ模様」として描いた所は、形は網目でもドットの格子に乗っておらず色もにじんでいるので、減色すると
ばらばらの点（ノイズ）になる。そこを、パレットの中の 2 色だけを使った 2×2 の決まった模様（Bayer 型。50% は市松）に置き換える。
減色の後・キャンバスの余白を足す前に行う（`convert.convert_still()`）。

手順（`tidy_pattern()`）:

1. **輪郭を守る**: パレットのいちばん暗い色（0 番）と、取っておいたいちばん暗い色のドットのうち、8 近傍でつながって
   `LINE_MIN` ドット以上になる所は線とみなし、候補から外して色も変えない
2. **ざらついた所を見つける**: ドットごとに、まわり 5×5（`WINDOW`）で
   - 色の入れ替わりの多さ（上下・左右にとなり合う組のうち、色が違う組の割合）が `CHANGE_MIN` 以上
   - 多い 2 色が `TOP2_MIN` 以上を占め、2 番目の色も `SECOND_MIN` ドット以上ある
   所を候補にする。平塗りの面は入れ替わりが少ないので候補にならない
3. **範囲に分ける**: 候補に囲まれた 1〜2 ドットの穴を候補に入れてから（`_fill_holes()`）、となり合う（4 近傍）候補のうち「多い 2 色の組」が同じものを 1 つの範囲にする。
   `MIN_REGION` ドット未満の範囲と、範囲の中で 2 色のどちらでもない色（3 色目）が `THIRD_MAX` 以上ある範囲
   （描き込みとみなす）は整えない。3 色目は範囲とそのまわり（5×5 の窓が届く所）で数える（描き込みのドットは候補から
   外れて、範囲がそれをよけて広がるため）。また、同じ色が 4 近傍で `THIRD_CLUMP`〜`MIN_REGION` − 1 ドットかたまった所だけを
   数える（それより大きいのは、となりの別の面）。
   1〜2 ドットの点はノイズとみなす（ai17 の壁は、暗い点が 1〜2 ドットずつ散らばって 3 色目が 10.1〜11.9% あり、
   ただの割合では、いちばん整えたい壁が外れたため。かたまりだけを数えると 1% 未満）
4. **濃さ**: 3 色目のドットを 2 色のうち近い方（OKLab の距離）とみなしてから、範囲の中の場所ごとに、まわり 7×7
   （`DENSITY_WINDOW`。範囲の中のドットだけで数える）の「明るい方の色の割合」を測り、5 段（0・25・50・75・100%）の
   いちばん近い段にする
5. **模様**: 2×2 の Bayer の表（`BAYER2`）で、段の数より小さい所を明るい方の色にする。表の位置は絵の左上を基準にした
   絵全体で共通の座標で決めるので、範囲の境目でも模様がずれない

新しい色は作らない（範囲の 2 色だけを使う）。整えた後に使われなくなった色はパレットから外す（暗い順のまま）。
迷ったら何もしない（どの条件にも当てはまらない所はそのまま）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sprite.simplify import _distance, rgb_to_oklab

WINDOW = 5            # ざらつきを測るまわりの大きさ（5×5）
CHANGE_MIN = 0.30     # 色の入れ替わりの割合（となり合う組のうち色が違う組）
TOP2_MIN = 0.80       # 多い 2 色が占める割合
SECOND_MIN = 3        # 2 番目の色に要るドット数（5×5 の中で。平塗りに点が少しある所を外す）
MIN_REGION = 64       # これより小さい範囲は整えない
THIRD_MAX = 0.10      # 範囲の中の 3 色目の割合がこれ以上なら、描き込みとみなして整えない
THIRD_CLUMP = 3       # 3 色目は、同じ色が 4 近傍でこのドット数以上かたまった所だけを数える（1〜2 ドットの点はノイズ。
#                       MIN_REGION 以上のかたまりは、となりの別の面なので数えない）
DENSITY_WINDOW = 7    # 濃さを測るまわりの大きさ（7×7。5×5 では段がちらついた）
LINE_MIN = 8          # 暗い色が 8 近傍でこのドット数以上つながっていたら輪郭とみなす
LEVELS = 5            # 濃さの段（0・25・50・75・100%）
# 2×2 の Bayer の表（[y % 2][x % 2]）。段 n（0〜4）では、表の値が n より小さい所が明るい方の色。段 2 は市松
BAYER2 = np.array([[0, 2], [3, 1]], np.int8)


@dataclass
class TidyRegion:
    """整えた（または整えなかった）範囲。"""

    colors: tuple[int, int]          # 2 色（暗い方, 明るい方）のパレットの番号（整える前の番号）
    size: int                        # ドット数
    box: tuple[int, int, int, int]   # 左, 上, 右, 下（右・下は含まない）
    third: float                     # 3 色目の割合（かたまった所だけ。範囲とそのまわりで数える）
    changed: int = 0                 # 変えたドット数
    skipped: str | None = None       # 整えなかった理由（"third_color"）
    colors_rgb: tuple = ()           # 2 色の RGB


@dataclass
class Tidied:
    index: np.ndarray                # 整えた後（パレットの番号。整える前と同じパレット）
    regions: list[TidyRegion]        # 整えた範囲（`MIN_REGION` 以上のもの。整えなかった範囲も含む）
    label: np.ndarray                # 高さ × 幅。整えた範囲の番号（regions の添字）、範囲の外は -1
    protected: np.ndarray            # 輪郭とみなして守ったドット
    small: int = 0                   # 小さくて整えなかった範囲の数
    changed: int = 0
    settings: dict = field(default_factory=dict)

    @property
    def tidied(self) -> list[TidyRegion]:
        return [r for r in self.regions if r.skipped is None]


def tidy_settings() -> dict:
    """JSON に書く、中で決めた設定。"""
    return {"window": WINDOW, "change_min": CHANGE_MIN, "top2_min": TOP2_MIN, "second_min": SECOND_MIN,
            "min_region": MIN_REGION, "third_max": THIRD_MAX, "third_clump": THIRD_CLUMP, "density_window": DENSITY_WINDOW,
            "line_min": LINE_MIN, "levels": [i * 100 // (LEVELS - 1) for i in range(LEVELS)], "tile": "bayer2x2"}


def _box_sum(a: np.ndarray, n: int) -> np.ndarray:
    """n×n（n は奇数）の和。外は 0 とみなす。"""
    r = n // 2
    p = np.zeros((a.shape[0] + n, a.shape[1] + n), np.int32)
    p[r + 1:r + 1 + a.shape[0], r + 1:r + 1 + a.shape[1]] = a
    p = p.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    return p[n:, n:] - p[:-n, n:] - p[n:, :-n] + p[:-n, :-n]


def label_regions(key: np.ndarray, valid: np.ndarray, diagonal: bool = False) -> np.ndarray:
    """同じ key の、となり合う valid なドットのつながりの番号（0 から。valid でない所は -1）。

    横の連なりに番号を付けてから、上下（diagonal なら斜めも）で触れる連なりを、小さい番号に寄せてまとめる
    （NumPy だけ。retrocore.analyzer.label_components と同じ考え方）。
    """
    h, w = key.shape
    if not valid.any():
        return np.full((h, w), -1, np.intp)
    flat = key.ravel()
    ok = valid.ravel()
    right = (flat[:-1] == flat[1:]) & ok[:-1] & ok[1:]
    right[np.arange(1, h, dtype=np.intp) * w - 1] = False
    start = np.ones(h * w, bool)
    start[1:] = ~right
    run = (np.cumsum(start, dtype=np.intp) - 1).reshape(h, w)
    pairs = []
    # 上の行のドットと、下の行の真下（diagonal なら右下・左下も）のドット
    cols = {0: (slice(None), slice(None))}
    if diagonal:
        cols[1] = (slice(None, -1), slice(1, None))
        cols[-1] = (slice(1, None), slice(None, -1))
    for a, b in cols.values():
        m = (key[:-1, a] == key[1:, b]) & valid[:-1, a] & valid[1:, b]
        if m.any():
            pairs.append(np.stack([run[:-1, a][m], run[1:, b][m]], 1))
    n = int(run[-1, -1]) + 1
    lab = np.arange(n, dtype=np.intp)
    if pairs:
        e = np.unique(np.concatenate(pairs).astype(np.intp), axis=0)
        u, v = e[:, 0], e[:, 1]
        while True:
            old = lab.copy()
            m = np.minimum(lab[u], lab[v])
            np.minimum.at(lab, u, m)
            np.minimum.at(lab, v, m)
            for _ in range(8):
                lab = lab[lab]
            if np.array_equal(lab, old):
                break
    out = lab[run]
    # 番号を 0 から詰める（valid な所だけ）
    roots, dense = np.unique(out[valid], return_inverse=True)
    res = np.full((h, w), -1, np.intp)
    res[valid] = dense.astype(np.intp).ravel()
    return res


def line_mask(index: np.ndarray, dark: list[int]) -> np.ndarray:
    """輪郭とみなすドット（暗い色が 8 近傍で `LINE_MIN` ドット以上つながっている所）。"""
    m = np.isin(index, np.asarray(sorted(set(dark)), index.dtype))
    if not m.any():
        return m
    lab = label_regions(m.astype(np.int8), m, diagonal=True)
    ids, sizes = np.unique(lab[m], return_counts=True)
    big = ids[sizes >= LINE_MIN]
    return m & np.isin(lab, big)


def tidy_pattern(index: np.ndarray, palette: np.ndarray, dark: list[int] | None = None) -> Tidied:
    """ざらついた範囲を 2 色の模様に整える。index は絵の部分（キャンバスの余白を足す前）。"""
    idx = np.asarray(index)
    h, w = idx.shape
    k = len(palette)
    protected = line_mask(idx, [0] if dark is None else dark)
    n = WINDOW
    area = n * n
    # まわりの色ごとのドット数（多い 2 色を選ぶ。同じ数なら番号の小さい方）
    best1 = np.full((h, w), -1, np.intp)
    best2 = np.full((h, w), -1, np.intp)
    cnt1 = np.full((h, w), -1, np.int32)
    cnt2 = np.full((h, w), -1, np.int32)
    for c in np.unique(idx).tolist():
        cnt = _box_sum((idx == c).astype(np.int32), n)
        g1 = cnt > cnt1
        g2 = ~g1 & (cnt > cnt2)
        best2 = np.where(g1, best1, np.where(g2, c, best2))
        cnt2 = np.where(g1, cnt1, np.where(g2, cnt, cnt2))
        best1 = np.where(g1, c, best1)
        cnt1 = np.where(g1, cnt, cnt1)
    # 色の入れ替わり（窓の中の上下・左右の組のうち、色が違う組の割合）。端は窓の中にある組だけで数える
    dh = np.zeros((h, w), np.int32)
    dh[:, :-1] = idx[:, :-1] != idx[:, 1:]
    dv = np.zeros((h, w), np.int32)
    dv[:-1] = idx[:-1] != idx[1:]
    ph = np.zeros((h, w), np.int32)
    ph[:, :-1] = 1
    pv = np.zeros((h, w), np.int32)
    pv[:-1] = 1
    # 窓の右端・下端からはみ出す組を数えないよう、組は左・上のドットに置き、窓を 1 つ狭めて数える
    change = (_box_sum(dh, n) - _edge_pairs(dh, n, 1)) + (_box_sum(dv, n) - _edge_pairs(dv, n, 0))
    pairs = (_box_sum(ph, n) - _edge_pairs(ph, n, 1)) + (_box_sum(pv, n) - _edge_pairs(pv, n, 0))
    inside = _box_sum(np.ones((h, w), np.int32), n)
    rate = change / np.maximum(pairs, 1)
    cand = ((rate >= CHANGE_MIN) & (cnt1 + cnt2 >= TOP2_MIN * inside) & (cnt2 >= SECOND_MIN) & (best2 >= 0)
            & ~protected)
    lo = np.minimum(best1, best2)
    hi = np.maximum(best1, best2)
    key = np.where(cand, lo * k + hi, -1)
    cand, key = _fill_holes(cand, key, protected)
    lab = label_regions(key, cand)
    # 3 色目として数えるかたまり（同じ色が 4 近傍で THIRD_CLUMP〜MIN_REGION − 1 ドット。大きいのはとなりの面）
    same = label_regions(idx, np.ones((h, w), bool))
    sid, ssize = np.unique(same, return_counts=True)
    clump = np.isin(same, sid[(ssize >= THIRD_CLUMP) & (ssize < MIN_REGION)])
    out = idx.copy()
    label = np.full((h, w), -1, np.intp)
    regions: list[TidyRegion] = []
    small = 0
    if cand.any():
        ids, sizes = np.unique(lab[cand], return_counts=True)
        lab3 = rgb_to_oklab(np.asarray(palette, np.uint8).reshape(-1, 3))
        for rid, size in zip(ids.tolist(), sizes.tolist()):
            if size < MIN_REGION:
                small += 1
                continue
            m = lab == rid
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            sub = m[y0:y1, x0:x1]
            vals = idx[y0:y1, x0:x1]
            a, b = int(key[ys[0], xs[0]] // k), int(key[ys[0], xs[0]] % k)
            third = _third_share(idx, m, protected | ~clump, (a, b), (x0, y0, x1, y1))
            reg = TidyRegion((a, b), int(size), (x0, y0, x1, y1), third,
                             colors_rgb=(tuple(int(v) for v in palette[a]), tuple(int(v) for v in palette[b])))
            regions.append(reg)
            if third >= THIRD_MAX:
                reg.skipped = "third_color"
                continue
            # 3 色目は近い方の色とみなす
            dist = _distance(lab3, lab3[[a, b]])
            near_b = (dist[:, 1] < dist[:, 0])[vals]
            light = (np.where((vals == a) | (vals == b), vals == b, near_b) & sub).astype(np.int32)
            dn = DENSITY_WINDOW
            frac = _box_sum(light, dn) / np.maximum(_box_sum(sub.astype(np.int32), dn), 1)
            new = np.where(light_dots(level_of(frac), y0, x0), b, a).astype(idx.dtype)
            region_out = out[y0:y1, x0:x1]
            diff = sub & (new != vals)
            reg.changed = int(np.count_nonzero(diff))
            region_out[sub] = new[sub]
            label[y0:y1, x0:x1][sub] = len(regions) - 1
    changed = int(np.count_nonzero(out != idx))
    return Tidied(out, regions, label, protected, small, changed, tidy_settings())


def level_of(frac) -> np.ndarray:
    """明るい方の色の割合（0〜1）を、5 段（0〜4。0・25・50・75・100%）のいちばん近い段にする。"""
    return np.rint(np.asarray(frac, np.float64) * (LEVELS - 1)).astype(np.int8)


def light_dots(level: np.ndarray, y0: int = 0, x0: int = 0) -> np.ndarray:
    """段ごとの模様で、明るい方の色にするドット。(y0, x0) は level[0, 0] の絵の中の位置（模様は絵全体で共通の座標）。"""
    lv = np.asarray(level)
    h, w = lv.shape
    yy = (np.arange(h, dtype=np.intp) + y0) % 2
    xx = (np.arange(w, dtype=np.intp) + x0) % 2
    return BAYER2[yy[:, None], xx[None, :]] < lv


def _fill_holes(cand: np.ndarray, key: np.ndarray, protected: np.ndarray, passes: int = 2):
    """候補に囲まれた小さな穴（上下左右の 4 つのうち 3 つ以上が同じ組の候補）を、その組の候補に入れる。

    ざらつきの測り方の都合で、範囲の中に 1〜2 ドットの穴が残り、元の点がそのまま模様の中に浮くのを防ぐ。輪郭は入れない。
    """
    for _ in range(passes):
        pad = np.pad(key, 1, constant_values=-1)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        best = np.full(key.shape, -1, key.dtype)
        for i in range(4):
            same = (nb == nb[i]).sum(axis=0)
            best = np.where((best < 0) & (nb[i] >= 0) & (same >= 3), nb[i], best)
        add = ~cand & ~protected & (best >= 0)
        if not add.any():
            break
        cand = cand | add
        key = np.where(add, best, key)
    return cand, key


def _third_share(idx: np.ndarray, region: np.ndarray, ignore: np.ndarray, pair: tuple[int, int],
                 box: tuple[int, int, int, int]) -> float:
    """範囲とそのまわり（ざらつきを測る窓の届く所）の、かたまった 3 色目の割合。

    描き込みのドットは多い 2 色の割合を下げるので候補から外れ、範囲はそれをよけて広がる。範囲の中だけで数えると
    3 色目がほとんど見えないので、まわりも含めて数える。ignore（輪郭・かたまりでない点・となりの大きな面）は数えない。
    """
    r = WINDOW // 2
    h, w = idx.shape
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(x0 - r, 0), max(y0 - r, 0), min(x1 + r, w), min(y1 + r, h)
    vals = idx[y0:y1, x0:x1]
    near = _box_sum(region[y0:y1, x0:x1].astype(np.int32), WINDOW) > 0
    other = near & (vals != pair[0]) & (vals != pair[1]) & ~ignore[y0:y1, x0:x1]
    base = near & (region[y0:y1, x0:x1] | other)
    return round(float(np.count_nonzero(other) / max(int(np.count_nonzero(base)), 1)), 4)


def _edge_pairs(d: np.ndarray, n: int, axis: int) -> np.ndarray:
    """n×n の窓の右端（axis=1）か下端（axis=0）の列・行に置いた組の和（その組の相手は窓の外にある）。"""
    r = n // 2
    h, w = d.shape
    if axis == 1:
        col = np.zeros((h, w), np.int32)
        col[:, :w - r] = d[:, r:]          # 窓の右端の列 = 中心から r 右
        return _vsum(col, n)
    row = np.zeros((h, w), np.int32)
    row[:h - r] = d[r:]
    return _hsum(row, n)


def _vsum(a: np.ndarray, n: int) -> np.ndarray:
    """縦 n の和（横は 1）。"""
    r = n // 2
    p = np.zeros((a.shape[0] + n, a.shape[1]), np.int32)
    p[r + 1:r + 1 + a.shape[0]] = a
    p = p.cumsum(0, dtype=np.int32)
    return p[n:] - p[:-n]


def _hsum(a: np.ndarray, n: int) -> np.ndarray:
    """横 n の和（縦は 1）。"""
    r = n // 2
    p = np.zeros((a.shape[0], a.shape[1] + n), np.int32)
    p[:, r + 1:r + 1 + a.shape[1]] = a
    p = p.cumsum(1, dtype=np.int32)
    return p[:, n:] - p[:, :-n]


# ---- 見える化（表示だけ。出力は変えない） ----

OVERLAY_COLORS = np.array([[255, 0, 160], [0, 200, 255], [255, 220, 0], [0, 255, 120], [255, 120, 0], [160, 80, 255]],
                          np.uint8)


def overlay_rgba(label: np.ndarray, protected: np.ndarray | None = None, alpha: int = 110) -> np.ndarray:
    """整えた範囲を薄い色で、範囲の縁を濃い色で塗った RGBA（範囲の外は透明）。隣どうしは色を変える。"""
    h, w = label.shape
    out = np.zeros((h, w, 4), np.uint8)
    m = label >= 0
    if not m.any():
        return out
    col = OVERLAY_COLORS[np.where(m, label, 0) % len(OVERLAY_COLORS)]
    out[..., :3] = np.where(m[..., None], col, 0)
    pad = np.pad(label, 1, constant_values=-1)
    edge = m & ((pad[:-2, 1:-1] != label) | (pad[2:, 1:-1] != label) | (pad[1:-1, :-2] != label)
                | (pad[1:-1, 2:] != label))
    out[..., 3] = np.where(edge, 255, np.where(m, alpha, 0))
    return out
