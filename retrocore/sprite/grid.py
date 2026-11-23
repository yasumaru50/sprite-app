"""グリッド検出（ドット絵風の AI 画像を、1 ドット＝1 画素の本物のドット絵に戻す）。

AI が描いた「ドット絵風の画像」は、ドット 1 個が約 20px 四方で描かれているが、場所によって 19〜22px にばらつき、
等間隔の格子にはならない。ぼかしの名残で 1 マスの中に似た色が何種類も混ざり、輪郭には半ドットのずれもある。
この工程は、ドットの境目（グリッド線）の位置を見つけて、各マスを 1 色にする。背景除去のあと、元の大きさのまま行う。

手順:

1. **境目の強さ（プロファイル）**: 横方向に隣り合う画素の色の差（R・G・B の差の最大値）を列の境目ごとに縦に足し合わせる
   （縦方向も同様）。不透明どうしの差に加えて、透明と不透明の境目も一定の強さ（`ALPHA_EDGE`）で数える
2. **ドットの大きさの推定**: 既定ではドットは正方形とみなし、縦横のプロファイルの自己相関を（範囲の長さで重み付けして）
   平均したものから 1 つの大きさを求め、縦横の両方に使う（`square=True`）。縦横を別々に推定すると、AI 画像では
   縦横の大きさが 1 割ほど食い違うことがあり（ai03 で横 20.2・縦 18.4px）、1 ドット＝1 画素で表示すると
   元の絵より縦長になったため。正方形でも、線の間隔は ±`snap_tolerance` の範囲で境目に合わせる。
   `square=False` で縦横を別々に推定する（横長・縦長のドットで描かれた画像向け）。
   どちらも、プロファイル（不透明な部分の範囲。少しならす）の自己相関を取り、`min_dot`〜`max_dot` 画素の間の山のうち
   最も目立つもの（山の高さ − それより短いずれでの最小値）を周期とする。その約数の位置にも同じくらいの山があれば、
   短い方を選ぶ（2 ドット分を 1 ドットと誤らないため）。山の位置は放物線で補間して小数で求める
3. **グリッド線の位置（場所によるずれへの対応）**: 等間隔の格子を当てはめると、ドットの大きさが 19〜22px にばらつく画像では
   端に行くほど境目がずれる。そこで、線の間隔を推定値の ±`snap_tolerance`（既定 20%）に制限したうえで、
   **線の位置での境目の強さの合計が最大になる線の並びを、動的計画法でまとめて求める**。
   計画書の案（線を 1 本ずつ近くの色の変わり目に合わせる）は、境目が弱い所（同じ色が何マスも続く所）で
   1 本ずれると次の線もずれていく。動的計画法なら全体で最もよく合う並びが選ばれ、境目の無い所では推定値に近い間隔で
   線が補われる（間隔が推定値から離れるほど小さな罰点を引く）。計算は 1 方向あたり（画素数 × 候補の間隔の数）で軽い。
   線は列・行ごとに 1 本で、画像のゆがみ（行によって列の境目がずれる）には対応しない
4. **グリッドが見つからない判定**: 自己相関の山の目立ち方（`min_prominence`）と、境目の強さのうち線の近く（±1 割）に
   集まっている割合（でたらめな位置に置いた場合を 0、すべてが線の上なら 1 とした値。`min_alignment`）の両方を見る。
   基準に届かなければ `GridNotFoundError` で止める（縦横別のときは、片方だけ届かない場合は届いた方の大きさを使う）。
   ドットの大きさを手動で指定したときは判定しない
5. **マスの色**: マスの縁はぼけていて、半ドットずれた輪郭もかかるので、縁から `inner_margin`（既定 2 割）より内側の画素を重視する
   （内側の重み 1、縁の重み `edge_weight`）。色の候補は内側の画素で、各候補について「差（R・G・B の差の最大値）が
   `color_tolerance` 以内の画素の重みの合計」を数え、最も多い候補（最頻色）を選ぶ。マスの色は、その候補に近い画素
   （差が `color_tolerance` 以内）の重み付き平均にする（ノイズを均す。似た色をまとめる処理は次の工程で行うのでここではしない）
6. **透明**: マスの中の透明な画素（α < 128）が `transparent_ratio`（既定 0.5）以上なら透明。結果は透明か不透明の 2 段階
7. **切り詰め**: 不透明なマスを囲む最小の長方形に切り詰める（キャラクターの大きさになる）

`retrocore` の方針どおり NumPy と Pillow だけを使う（SciPy は使わない）。1254×1254 で 0.5 秒ほど。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pipeline import Step

ALPHA_OPAQUE = 128          # これ以上の α を不透明とみなす（downscale.py と同じ）
ALPHA_EDGE = 128            # 透明と不透明の境目を、色の差がこの値の境目として数える

DEFAULT_MIN_DOT = 4
DEFAULT_MAX_DOT = 64
DEFAULT_SNAP_TOLERANCE = 0.2   # 線の間隔は推定値の ±20% まで（19〜22px のばらつきに余裕を持たせる）
DEFAULT_SNAP_PENALTY = 4.0     # 間隔が推定値から離れたときの罰点（((間隔 − 推定値) / 推定値)² 倍）
DEFAULT_MIN_PROMINENCE = 0.2   # 自己相関の山の目立ち方の下限
DEFAULT_MIN_ALIGNMENT = 0.4     # 境目の強さが線の近くに集まっている度合いの下限
DEFAULT_INNER_MARGIN = 0.2     # マスの縁から内側とみなすまでの幅（マスの大きさに対する割合）
DEFAULT_EDGE_WEIGHT = 0.25     # 縁の画素の重み（内側は 1）
DEFAULT_COLOR_TOLERANCE = 24   # 最頻色を数えるとき、同じ色とみなす差（R・G・B の差の最大値）
DEFAULT_TRANSPARENT_RATIO = 0.5


class GridNotFoundError(ValueError):
    """ドットの格子が見つからないとき（普通のイラストなど）。メッセージは利用者向けの日本語。"""


@dataclass
class Grid:
    """グリッド線の位置（入力画像の座標）。`xs[i]`〜`xs[i+1]` が i 列目のマス。"""

    xs: np.ndarray
    ys: np.ndarray
    period: tuple[float, float]               # 推定（または指定）したドットの大きさ（横, 縦）
    prominence: tuple[float, float] = (0.0, 0.0)
    alignment: tuple[float, float] = (0.0, 0.0)
    manual: bool = False
    square: bool = False                      # 縦横で同じ大きさを使ったか

    @property
    def cell_size(self) -> tuple[float, float]:
        """線の間隔の中央値（横, 縦）。"""
        return float(np.median(np.diff(self.xs))), float(np.median(np.diff(self.ys)))

    @property
    def gap_range(self) -> tuple[int, int, int, int]:
        """線の間隔の最小・最大（横の最小, 横の最大, 縦の最小, 縦の最大）。端の欠けたマスも含む。"""
        dx, dy = np.diff(self.xs), np.diff(self.ys)
        return int(dx.min()), int(dx.max()), int(dy.min()), int(dy.max())


@dataclass
class GridResult:
    """グリッド検出の結果。"""

    image: np.ndarray                  # 1 ドット＝1 画素の RGBA（切り詰め済み）
    grid: Grid
    crop: tuple[int, int, int, int]    # 切り詰めた範囲（マス単位: 列の始め, 行の始め, 列の終わり, 行の終わり。終わりは含まない）
    colors: int                        # 不透明な画素の色数
    duplicates: tuple[int, int] = (0, 0)  # 同じ色の並びが続く行・列の組の数（`duplicate_lines()`）

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    @property
    def lines(self) -> tuple[np.ndarray, np.ndarray]:
        """切り詰めた範囲のグリッド線（入力画像の座標）。"""
        c0, r0, c1, r1 = self.crop
        return self.grid.xs[c0:c1 + 1], self.grid.ys[r0:r1 + 1]


# ---------------------------------------------------------------- 境目の強さ

def edge_profile(rgba: np.ndarray, axis: int) -> np.ndarray:
    """境目の強さ。axis=1 なら列の境目（長さ 幅+1）、axis=0 なら行の境目（長さ 高さ+1）。

    `p[i]` は画素 i-1 と i の間の強さ（両端の 0 と n は 0）。不透明どうしは R・G・B の差の最大値、
    透明と不透明の境目は `ALPHA_EDGE`、透明どうしは 0。
    """
    a = np.asarray(rgba)
    if axis == 0:
        a = a.transpose(1, 0, 2)
    rgb = a[..., :3].astype(np.int16)
    op = a[..., 3] >= ALPHA_OPAQUE
    d = np.abs(np.diff(rgb, axis=1)).max(-1).astype(np.float64)
    both = op[:, 1:] & op[:, :-1]
    one = op[:, 1:] ^ op[:, :-1]
    d = np.where(both, d, 0.0) + np.where(one, float(ALPHA_EDGE), 0.0)
    out = np.zeros(a.shape[1] + 1)
    out[1:-1] = d.sum(0)
    return out


def _smooth(p: np.ndarray) -> np.ndarray:
    """[1, 2, 1] / 4 でならす（ぼけた境目の中心を山にする）。"""
    q = np.pad(p, 1, mode="edge")
    return (q[:-2] + 2 * q[1:-1] + q[2:]) / 4


def _content_range(op: np.ndarray, axis: int) -> tuple[int, int]:
    """不透明な画素のある範囲 [始め, 終わり)。axis=1 なら列、0 なら行。"""
    any_ = op.any(axis=1 - axis)
    idx = np.nonzero(any_)[0]
    if len(idx) == 0:
        return 0, op.shape[axis]
    return int(idx[0]), int(idx[-1]) + 1


# ---------------------------------------------------------------- ドットの大きさ

def autocorrelation(profile: np.ndarray, start: int, stop: int, max_lag: int) -> np.ndarray | None:
    """プロファイルの [start, stop] の範囲の自己相関（ずれ 0 で 1 にそろえる。長さ max_lag+2）。求められなければ None。

    ならしてから使う（間隔が 1〜2 画素ずつばらつく境目でも、同じずれの山にまとまるように）。
    """
    seg = _smooth(_smooth(np.asarray(profile, np.float64)))[start:stop + 1]
    n = len(seg)
    if n < 3 or not seg.any():
        return None
    seg = seg - seg.mean()
    m = 1 << int(np.ceil(np.log2(2 * n + max_lag + 2)))
    f = np.fft.rfft(seg, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:max_lag + 2]
    if ac[0] <= 0:
        return None
    return ac / ac[0]


def _lag_range(n: int, min_dot: float, max_dot: float) -> tuple[int, int]:
    return max(2, int(np.floor(min_dot))), int(min(max_dot, n // 3))


def prominence_at(ac: np.ndarray, lag: float) -> float:
    """自己相関の、ずれ lag（小数は四捨五入）での山の目立ち方（その値 − それより短いずれでの最小値）。"""
    i = int(round(lag))
    if ac is None or i < 2 or i >= len(ac):
        return 0.0
    return float(ac[i] - ac[1:i].min())


def pick_period(ac: np.ndarray | None, lo: int, hi: int) -> tuple[float, float]:
    """自己相関の [lo, hi] の山から周期 (周期, 山の目立ち方) を選ぶ。山が無ければ (0, 0)。

    最も目立つ山を選び、その約数の位置に同じくらい（6 割以上）の山があれば短い方にする
    （2 ドット分を 1 ドットと誤らないため）。山の位置は放物線で補間して小数で求める。
    """
    if ac is None or hi < lo + 1 or len(ac) < hi + 2:
        return 0.0, 0.0
    peaks = []
    for lag in range(lo, hi + 1):
        if ac[lag] >= ac[lag - 1] and ac[lag] > ac[lag + 1]:
            peaks.append((lag, float(ac[lag] - ac[1:lag].min())))
    if not peaks:
        return 0.0, 0.0
    best_lag, best = max(peaks, key=lambda t: (t[1], -t[0]))
    for lag, prom in peaks:
        if lag < best_lag and prom >= 0.6 * best:
            ratio = best_lag / lag
            if abs(ratio - round(ratio)) < 0.15:
                best_lag, best = lag, prom
                break
    y0, y1, y2 = ac[best_lag - 1], ac[best_lag], ac[best_lag + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    return float(best_lag + np.clip(off, -0.5, 0.5)), float(best)


def estimate_period(profile: np.ndarray, start: int, stop: int, min_dot: float = DEFAULT_MIN_DOT,
                    max_dot: float = DEFAULT_MAX_DOT) -> tuple[float, float]:
    """プロファイルの [start, stop] の範囲の周期 (周期, 山の目立ち方) を推定する。山が無ければ (0, 0)。"""
    lo, hi = _lag_range(stop - start + 1, min_dot, max_dot)
    if hi < lo + 1:
        return 0.0, 0.0
    return pick_period(autocorrelation(profile, start, stop, hi), lo, hi)


def estimate_square_period(profiles, ranges, min_dot: float = DEFAULT_MIN_DOT,
                           max_dot: float = DEFAULT_MAX_DOT) -> tuple[float, float, tuple[float, ...]]:
    """縦横のプロファイルを合わせて、正方形のドットの大きさを 1 つ推定する。

    各方向の自己相関を、その方向の範囲の長さで重み付けして平均し、その山から周期を選ぶ。
    戻り値は (周期, 合わせた山の目立ち方, 各方向のその周期での目立ち方)。
    """
    n = min(e - s + 1 for s, e in ranges)
    lo, hi = _lag_range(max(e - s + 1 for s, e in ranges), min_dot, max_dot)
    acs, ws = [], []
    for prof, (s, e) in zip(profiles, ranges):
        ac = autocorrelation(prof, s, e, hi)
        acs.append(ac)
        if ac is not None:
            ws.append((ac, float(e - s + 1)))
    if not ws or hi < lo + 1 or n < 3:
        return 0.0, 0.0, tuple(0.0 for _ in profiles)
    joint = sum(ac * w for ac, w in ws) / sum(w for _, w in ws)
    p, prom = pick_period(joint, lo, hi)
    return p, prom, tuple(prominence_at(ac, p) if ac is not None else 0.0 for ac in acs)


# ---------------------------------------------------------------- 線の位置

def _normalize(profile: np.ndarray, start: int, stop: int) -> np.ndarray:
    """動的計画法の点数: ならしたプロファイルを、範囲内の上位 5% の値で 1 にそろえ、中央値を 0 にする。

    中央値を引くのは、ノイズで少しだけ正の値を持つ平らな所に線を詰め込んで点数を稼がないようにするため。
    """
    s = _smooth(profile)
    seg = s[start:stop + 1]
    scale = np.percentile(seg, 95) if len(seg) else 0.0
    if scale <= 0:
        scale = max(float(seg.max()) if len(seg) else 0.0, 1.0)
    return (s - np.median(seg)) / scale


def fit_lines(score: np.ndarray, period: float, start: int, stop: int,
              tolerance: float = DEFAULT_SNAP_TOLERANCE, penalty: float = DEFAULT_SNAP_PENALTY) -> np.ndarray:
    """境目の点数 `score`（長さ n+1）に合わせて、[start, stop] を覆うグリッド線を動的計画法で求める。

    最初の線は start 以下、最後の線は stop 以上。隣り合う線の間隔は period × (1 ± tolerance) の範囲に限り、
    間隔が period から離れるほど penalty × ((間隔 − period) / period)² を引く。画像の端（0 と n）は
    間隔の制限なしに線にできる（端で欠けたマス）。点数の合計が最大の並びを返す（整数の昇順の配列）。
    """
    n = len(score) - 1
    gmin = max(1, int(np.floor(period * (1 - tolerance))))
    gmax = max(gmin, int(np.ceil(period * (1 + tolerance))))
    lo = max(0, start - gmax)
    hi = min(n, stop + gmax)
    gaps = np.arange(gmin, gmax + 1)
    gpen = penalty * ((gaps - period) / period) ** 2
    neg = -np.inf
    best = np.full(n + 1, neg)
    prev = np.full(n + 1, -1, np.int64)
    for i in range(lo, hi + 1):
        cand = neg
        arg = -1
        if i <= start:
            cand = 0.0  # ここから始める
        j = i - gaps
        ok = j >= lo
        if ok.any():
            v = best[j[ok]] - gpen[ok]
            k = int(np.argmax(v))
            if v[k] > cand:
                cand, arg = float(v[k]), int(j[ok][k])
        if i == n and lo <= n - 1:  # 画像の右端（下端）: 間隔の制限なし
            jj = np.arange(max(lo, n - gmax), n)
            v = best[jj]
            k = int(np.argmax(v))
            if v[k] > cand:
                cand, arg = float(v[k]), int(jj[k])
        if i > 0 and lo == 0 and i <= gmax and best[0] > neg and best[0] > cand:
            cand, arg = float(best[0]), 0  # 画像の左端（上端）から: 間隔の制限なし
        if cand > neg:
            best[i] = cand + (score[i] if 0 < i < n else 0.0)
            prev[i] = arg
    ends = np.arange(max(stop, lo), hi + 1)
    ends = ends[best[ends] > neg]
    if len(ends) == 0:
        raise GridNotFoundError("グリッド線を当てはめられませんでした")
    i = int(ends[np.argmax(best[ends])])
    out = [i]
    while prev[i] >= 0:
        i = int(prev[i])
        out.append(i)
    return np.array(out[::-1], np.int64)


def uniform_lines(period: float, origin: float, start: int, stop: int, n: int) -> np.ndarray:
    """等間隔のグリッド線（origin + k × period）のうち、[start, stop] を覆うもの（0〜n に収める）。"""
    k0 = int(np.floor((start - origin) / period))
    k1 = int(np.ceil((stop - origin) / period))
    xs = np.rint(origin + period * np.arange(k0, k1 + 1)).astype(np.int64)
    xs = np.unique(np.clip(xs, 0, n))
    if xs[0] > start:
        xs = np.concatenate([[0], xs])
    if xs[-1] < stop:
        xs = np.concatenate([xs, [n]])
    return xs


def best_origin(score: np.ndarray, period: float, start: int, stop: int) -> float:
    """等間隔の格子の始まりの位置（0 ≤ origin < period）を、線の位置の点数の合計が最大になるように選ぶ。"""
    best, best_o = -np.inf, 0.0
    for o in np.arange(0, period, 0.25):
        xs = uniform_lines(period, o, start, stop, len(score) - 1)
        xs = xs[(xs > 0) & (xs < len(score) - 1)]
        v = float(score[xs].sum())
        if v > best:
            best, best_o = v, float(o)
    return best_o


def alignment(profile: np.ndarray, lines: np.ndarray, start: int, stop: int, period: float) -> float:
    """境目の強さが線の近く（±period の 1 割、最低 1 画素）に集まっている度合い。

    ノイズで全体が底上げされる分は、範囲内の中央値を引いて除く。

    でたらめな位置に線を置いたときに期待される割合を 0、すべてが線の近くにあれば 1 とする（負にもなる）。
    """
    seg = np.asarray(profile[start:stop + 1], np.float64)
    seg = np.maximum(seg - np.median(seg), 0.0)  # ノイズによる一様な底上げを除く
    total = seg.sum()
    if total <= 0 or period <= 0:
        return 0.0
    r = max(1, int(round(period * 0.1)))
    near = np.zeros(len(profile), bool)
    for d in range(-r, r + 1):
        idx = lines + d
        near[idx[(idx >= 0) & (idx < len(profile))]] = True
    frac = seg[near[start:stop + 1]].sum() / total
    rand = min(1.0, (2 * r + 1) / period)
    if rand >= 1.0:
        return 0.0
    return float((frac - rand) / (1 - rand))


def _pair(v, name: str) -> tuple[float, float] | None:
    if v is None:
        return None
    if np.isscalar(v):
        v = (v, v)
    a, b = (float(x) for x in v)
    return a, b


def detect_grid(rgba: np.ndarray, dot_size=None, origin=None, snap: bool = True, *,
                min_dot: float = DEFAULT_MIN_DOT, max_dot: float = DEFAULT_MAX_DOT,
                snap_tolerance: float = DEFAULT_SNAP_TOLERANCE, snap_penalty: float = DEFAULT_SNAP_PENALTY,
                min_prominence: float = DEFAULT_MIN_PROMINENCE,
                min_alignment: float = DEFAULT_MIN_ALIGNMENT, square: bool = True) -> Grid:
    """グリッド線を求める。

    - `dot_size`: ドットの大きさ（数値、または (横, 縦)）。省略時は推定し、見つからなければ `GridNotFoundError`
    - `origin`: グリッドの始まる位置 (x, y)。指定すると等間隔の格子にする（線を境目に合わせない）
    - `snap`: True（既定）なら線を 1 本ずつ近くの色の境目に合わせる（動的計画法）。False なら等間隔の格子
    - `square`: True（既定）ならドットは正方形とみなし、縦横の情報を合わせて 1 つの大きさを推定して両方向に使う。
      False なら縦横を別々に推定する（横長・縦長のドットの画像向け）。`dot_size` を指定したときは使わない
    """
    a = np.asarray(rgba)
    h, w = a.shape[:2]
    op = a[..., 3] >= ALPHA_OPAQUE
    if not op.any():
        raise GridNotFoundError("不透明な画素がありません（背景除去で全部消えた可能性があります）")
    size = _pair(dot_size, "dot_size")
    org = _pair(origin, "origin")
    if size is not None and (size[0] < 1 or size[1] < 1):
        raise ValueError(f"ドットの大きさは 1 以上で指定してください（{dot_size}）")

    profs, ranges = [], []
    for axis in (1, 0):
        profs.append(edge_profile(a, axis))
        ranges.append(_content_range(op, axis))
    manual = size is not None
    joint = square and not manual
    if manual:
        periods, proms = [size[0], size[1]], [0.0, 0.0]
    elif joint:
        p, prom, per_axis = estimate_square_period(profs, ranges, min_dot, max_dot)
        if p <= 0 or prom < min_prominence:
            raise GridNotFoundError(_not_found_message(list(per_axis), None))
        periods, proms = [p, p], list(per_axis)
    else:
        periods, proms = [], []
        for prof, (s, e) in zip(profs, ranges):
            p, pr = estimate_period(prof, s, e, min_dot, max_dot)
            periods.append(p)
            proms.append(pr)
    if not manual and not joint:
        ok = [p > 0 and pr >= min_prominence for p, pr in zip(periods, proms)]
        if not any(ok):
            raise GridNotFoundError(_not_found_message(proms, None))
        if not all(ok):  # 片方だけ見つかった: ドットは正方形とみなす
            good = periods[ok.index(True)]
            periods = [good if not k else p for k, p in zip(ok, periods)]

    scores = [_normalize(prof, s, e) for prof, (s, e) in zip(profs, ranges)]

    def place(k: int, p: float) -> np.ndarray:
        s, e = ranges[k]
        n = len(profs[k]) - 1
        if org is not None:
            return uniform_lines(p, org[k], s, e, n)
        if not snap:
            return uniform_lines(p, best_origin(scores[k], p, s, e), s, e, n)
        return fit_lines(scores[k], p, s, e, snap_tolerance, snap_penalty)

    lines = [place(k, periods[k]) for k in range(2)]
    if snap and org is None and not manual:
        # 線の並びから大きさを求め直してもう一度（自己相関の山の位置は 1 画素ほどずれることがある）。
        # 正方形のときは縦横の線をまとめて 1 つの大きさにする
        spans = []
        for k in range(2):
            s, e = ranges[k]
            inner = lines[k][(lines[k] >= s) & (lines[k] <= e)]
            spans.append((float(inner[-1] - inner[0]), len(inner) - 1) if len(inner) >= 3 else None)
        if joint:
            ok = [sp for sp in spans if sp is not None]
            if ok:
                p2 = sum(sp[0] for sp in ok) / sum(sp[1] for sp in ok)
                new = [p2, p2]
            else:
                new = periods
        else:
            new = [sp[0] / sp[1] if sp is not None else periods[k] for k, sp in enumerate(spans)]
        for k in range(2):
            if abs(new[k] - periods[k]) > 0.1:
                periods[k] = new[k]
                lines[k] = place(k, new[k])
    aligns = [alignment(profs[k], lines[k], *ranges[k], periods[k]) for k in range(2)]

    if not manual and not all(al >= min_alignment for al in aligns):
        raise GridNotFoundError(_not_found_message(proms, aligns))
    return Grid(xs=lines[0], ys=lines[1], period=(periods[0], periods[1]),
                prominence=(proms[0], proms[1]), alignment=(aligns[0], aligns[1]), manual=manual,
                square=joint or (manual and size[0] == size[1]))


def _not_found_message(proms, aligns) -> str:
    detail = f"周期の目立ち方 横 {proms[0]:.2f}・縦 {proms[1]:.2f}（基準 {DEFAULT_MIN_PROMINENCE}）"
    if aligns is not None:
        detail += f"、境目の揃い方 横 {aligns[0]:.2f}・縦 {aligns[1]:.2f}（基準 {DEFAULT_MIN_ALIGNMENT}）"
    return ("ドットの格子が見つかりませんでした（" + detail + "）。"
            "ドット絵風の画像ではない可能性があります。普通のイラストなら入力の種類を「普通のイラスト」にして"
            "縮小してください。ドットの大きさが分かっている場合は、手動で指定すると変換できます。")


# ---------------------------------------------------------------- マスの色

def _cell_color(block: np.ndarray, margin_y: int, margin_x: int, edge_weight: float,
                tol: int, stride: int) -> np.ndarray:
    """不透明な画素のあるマス 1 つの色（R, G, B）。"""
    ch, cw = block.shape[:2]
    yy, xx = np.mgrid[0:ch, 0:cw]
    inner = ((yy >= margin_y) & (yy < ch - margin_y) & (xx >= margin_x) & (xx < cw - margin_x))
    op = block[..., 3] >= ALPHA_OPAQUE
    sub = (yy % stride == stride // 2) | (ch <= stride)
    sub &= (xx % stride == stride // 2) | (cw <= stride)
    voters = op & sub
    if not voters.any():
        voters = op
    cand = voters & inner
    if not cand.any():
        cand = voters
    rgb = block[..., :3].astype(np.int16)
    vc = rgb[voters]
    vw = np.where(inner[voters], 1.0, edge_weight)
    cc = rgb[cand]
    close = np.abs(cc[:, None, :] - vc[None, :, :]).max(-1) <= tol    # 候補 × 投票する画素
    score = close @ vw
    best = score.max()
    top = np.nonzero(score >= best - 1e-9)[0]
    if len(top) > 1:  # 同点: 重み付き平均に近い候補 → 色の値が小さい方
        mean = (vc * vw[:, None]).sum(0) / vw.sum()
        d = np.abs(cc[top] - mean).sum(-1)
        order = np.lexsort((cc[top][:, 2], cc[top][:, 1], cc[top][:, 0], d))
        k = int(top[order[0]])
    else:
        k = int(top[0])
    m = close[k]
    wsum = vw[m].sum()
    return np.rint((vc[m] * vw[m][:, None]).sum(0) / wsum).astype(np.uint8)


def cells_to_pixels(rgba: np.ndarray, xs: np.ndarray, ys: np.ndarray, *,
                    inner_margin: float = DEFAULT_INNER_MARGIN, edge_weight: float = DEFAULT_EDGE_WEIGHT,
                    color_tolerance: int = DEFAULT_COLOR_TOLERANCE,
                    transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO) -> np.ndarray:
    """グリッド線で区切った各マスを 1 色にした RGBA（高さ len(ys)-1 × 幅 len(xs)-1）。"""
    a = np.asarray(rgba)
    op_all = a[..., 3] >= ALPHA_OPAQUE
    nx, ny = len(xs) - 1, len(ys) - 1
    out = np.zeros((ny, nx, 4), np.uint8)
    # マスごとの不透明な画素の数（積分画像で一度に数える）
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
    ii[1:, 1:] = op_all.cumsum(0).cumsum(1)
    X0, Y0 = np.meshgrid(xs[:-1], ys[:-1])
    X1, Y1 = np.meshgrid(xs[1:], ys[1:])
    cnt = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
    area = (X1 - X0) * (Y1 - Y0)
    opaque = (area > 0) & ((area - cnt) < transparent_ratio * area)
    for r, c in zip(*np.nonzero(opaque)):
        y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
        block = a[y0:y1, x0:x1]
        ch, cw = block.shape[:2]
        my = int(round(ch * inner_margin)) if ch > 2 else 0
        mx = int(round(cw * inner_margin)) if cw > 2 else 0
        stride = max(1, int(round(min(ch, cw) / 10)))
        out[r, c, :3] = _cell_color(block, my, mx, edge_weight, int(color_tolerance), stride)
        out[r, c, 3] = 255
    return out


def duplicate_lines(rgba: np.ndarray, tol: int = DEFAULT_COLOR_TOLERANCE,
                    min_ratio: float = 1.0) -> tuple[int, int]:
    """隣り合う行どうし・列どうしで「同じ色の並び」になっている組の数 (行, 列)。

    透明・不透明が一致し、不透明な画素の色の差（R・G・B の差の最大値）が `tol` 以内の画素が、
    どちらかが不透明な画素の `min_ratio` 以上（既定 1.0 = すべて）なら同じとみなす。全部透明の組は数えない。
    グリッド線が余分に入ると、1 ドットが 2 行（2 列）に分かれて同じ色の行（列）が続くので、その目安にする。
    """
    a = np.asarray(rgba)
    out = []
    for x in (a, a.transpose(1, 0, 2)):
        rgb = x[..., :3].astype(np.int16)
        op = x[..., 3] >= ALPHA_OPAQUE
        any_ = op[1:] | op[:-1]
        close = np.abs(rgb[1:] - rgb[:-1]).max(-1) <= tol
        same = (op[1:] == op[:-1]) & (close | ~any_)
        cnt = any_.sum(1)
        ratio = np.where(cnt > 0, (same & any_).sum(1) / np.maximum(cnt, 1), 0.0)
        out.append(int(((cnt > 0) & (ratio >= min_ratio - 1e-9)).sum()))
    return out[0], out[1]


def _bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.nonzero(alpha.any(1))[0]
    cols = np.nonzero(alpha.any(0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def count_colors(rgba: np.ndarray) -> int:
    """不透明な画素の色数。"""
    a = np.asarray(rgba)
    rgb = a[..., :3][a[..., 3] >= ALPHA_OPAQUE]
    if len(rgb) == 0:
        return 0
    return int(len(np.unique(rgb.astype(np.int64) @ np.array([65536, 256, 1]))))


def grid_to_pixels(rgba: np.ndarray, dot_size=None, origin=None, snap: bool = True, *,
                   transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO,
                   inner_margin: float = DEFAULT_INNER_MARGIN, edge_weight: float = DEFAULT_EDGE_WEIGHT,
                   color_tolerance: int = DEFAULT_COLOR_TOLERANCE, **detect_kw) -> GridResult:
    """グリッドを検出し、1 ドット＝1 画素の画像にして、不透明な部分に切り詰めて返す。

    グリッドが見つからない、または不透明なマスが 1 つも無いときは `GridNotFoundError`。
    """
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"H×W×4 の RGBA 配列が必要です（{a.shape}）")
    grid = detect_grid(a, dot_size, origin, snap, **detect_kw)
    cells = cells_to_pixels(a, grid.xs, grid.ys, inner_margin=inner_margin, edge_weight=edge_weight,
                            color_tolerance=color_tolerance, transparent_ratio=transparent_ratio)
    alpha = cells[..., 3] > 0
    if not alpha.any():
        raise GridNotFoundError("不透明なマスがありません（透明な部分が多すぎます）")
    c0, r0, c1, r1 = _bbox(alpha)
    img = cells[r0:r1, c0:c1].copy()
    return GridResult(image=img, grid=grid, crop=(c0, r0, c1, r1), colors=count_colors(img),
                      duplicates=duplicate_lines(img, color_tolerance))


@dataclass
class GridStep(Step):
    """グリッド検出の工程（ドット絵風の入力のとき、縮小の代わりに使う）。

    出力は 1 ドット＝1 画素で、不透明な部分に切り詰めた画像。グリッドが見つからなければ `GridNotFoundError`。
    直前の結果は `last`（`GridResult`）に残り、`describe()` にドットの大きさ・出力の大きさ・色数が出る。
    """

    dot_size: float | tuple[float, float] | None = None
    origin: tuple[float, float] | None = None
    snap: bool = True
    transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE
    square: bool = True
    name: str = "grid"
    last: GridResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not 0 < self.transparent_ratio <= 1:
            raise ValueError(f"透明の割合は 0 より大きく 1 以下で指定してください（{self.transparent_ratio}）")
        if not 0 < self.snap_tolerance < 1:
            raise ValueError(f"線の間隔の許容は 0〜1 の間で指定してください（{self.snap_tolerance}）")

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        self.last = grid_to_pixels(rgba, self.dot_size, self.origin, self.snap,
                                   transparent_ratio=self.transparent_ratio,
                                   snap_tolerance=self.snap_tolerance, square=self.square)
        return self.last.image

    def describe(self) -> str:
        base = "グリッド検出"
        r = self.last
        if r is None:
            return base
        cw, ch = r.grid.cell_size
        kind = "指定" if r.grid.manual else ("推定・正方形" if r.grid.square else "推定・縦横別")
        px, py = r.grid.period
        size = f"{px:.1f}px" if r.grid.square else f"{px:.1f}×{py:.1f}px"
        gx0, gx1, gy0, gy1 = _inner_gap_range(r)
        dr, dc = r.duplicates
        return (f"{base}（ドット {size}［{kind}］、間隔 横 {gx0}〜{gx1}・縦 {gy0}〜{gy1}px、"
                f"{r.size[0]}×{r.size[1]} ドット、{r.colors} 色、同じ色が続く行 {dr}・列 {dc}）")


def _inner_gap_range(r: GridResult) -> tuple[int, int, int, int]:
    """切り詰めた範囲の線の間隔の最小・最大（横の最小, 横の最大, 縦の最小, 縦の最大）。"""
    xs, ys = r.lines
    dx, dy = np.diff(xs), np.diff(ys)
    return int(dx.min()), int(dx.max()), int(dy.min()), int(dy.max())
