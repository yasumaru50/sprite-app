"""縁取りの工程: キャラクターの外側の輪郭（シルエットの縁）に、暗い色の 1 ドットの線を付ける。

クリーンアップの後に行う。対象はシルエットの縁（透明と不透明の境目）だけで、部品と部品の間の内側の線は扱わない
（すでにある内側の線は、シルエットの輪郭とつながっていれば、切れ目を埋めるときに使う）。

動き方（`mode`）:

| 動き方 | 内容 |
| --- | --- |
| `auto`（既定） | シルエットの縁にすでに暗い輪郭があるかを判定し、あれば切れ目だけを埋める。無ければ `add` と同じ |
| `add` | シルエットの外側（透明の部分）に 1 ドットの輪郭を付ける。絵の中（不透明なドット）は変えない |
| `thin` | 太すぎる外側の輪郭を減らす。輪郭があれば `auto` と同じく切れ目を埋めたうえで、2 ドットの太さの外側の輪郭を 1 ドットにする。無ければ、暗くない縁の外側にだけ付ける |
| `off` | 何もしない |

**「すでに輪郭がある」の判定**（`detect_outline()`）: シルエットの縁のドット（上下左右のどこかが透明か画像の外）ごとに、
透明の側と反対の向き（内側）へ 3 ドットまでと、まわり 8 ドットの不透明なドットのうち、自分と違う色のどれかより
OKLab の明るさで `DARK_MARGIN`（0.1）以上暗ければ「暗い縁」、違う色はあるが暗くなければ「暗くない縁」、
違う色が無く内側 3 ドットが同じ色なら「暗くない縁」（広い面の端。黒い服の端など）、内側がすぐ透明になる細い所は
「判定しない」とする。判定できた縁のうち暗い縁の割合（`coverage`）が `HAS_OUTLINE_SHARE`（0.6）以上なら「輪郭がある」。
暗い縁で多く使われている色（暗い縁の 1 割以上）を「輪郭の色」とする。

**切れ目の埋め方**（輪郭があるとき）: 輪郭の色のドットのうち線の端（8 近傍の輪郭のドットが 1 個、または上下左右に隣り合う 2 個。
輪郭の角は端ではない）から、
2〜3 ドット先（チェビシェフ距離）にある別の輪郭のドットまでを、間の 1〜2 ドットを塗ってつなぐ。
次のときだけつなぐ: 端の線がシルエットの縁の輪郭とつながっている（内側だけの線は対象外）、2 つのドットが輪郭の線をたどって
近く（`LOCAL_STEPS` 歩以内）でつながっていない（同じ線の曲がり角を太らせない）、間のドットがどれも不透明で輪郭の色でない
（シルエットは変えない）、道のドットを塗ると、その色の面（上下左右のつながり）が 3 ドット以上の 2 つ以上に分かれない
（塗りの面を横切らない。マントの内側の影の境目などを輪郭とつながないため。ただし 1 ドットの切れ目で、線がまっすぐ
そのまま続くときは分かれても埋める。体と腕の境目の線の先など）。1 ドットの切れ目を先に、2 ドットの切れ目を後に埋める。間の道が何通りかあるときは、
線が伸びてきた向きにまっすぐ続く道、次に輪郭の色に近いドットを通る道にする。塗る色は線の端の色。

**外側に付ける線の形**（`shape`）: `thin`（既定）は上下左右に不透明なドットがある透明のドット。角では斜めにつながる 1 ドットの線になり、
角が 2 ドットの太さにならない。`thick` は斜めに不透明なドットがあるドットも含める（上下左右でつながる線。角が埋まる）。
シルエットが画像の端に接しているときは、その辺に 1 ドットの余白（透明）を足す（`pad=False` で足さず、はみ出す分は付けない）。

**輪郭の色**（`color`）:
- `darken`（既定）: 輪郭のすぐ内側の色（上下左右の不透明なドットで多い色。同数なら暗い色）を暗くした色。明るさは
  `min(0.18 + 0.25 × L, 0.7 × L)`（明るい色でも暗めの線、暗い色ならさらに暗く）、色みは 0.8 倍に残す
  （赤の隣は暗い赤、肌の隣は暗い茶色）。色の数を少なく保つため、(1) 近い色（距離 `MERGE_DISTANCE` 未満）はまとめ、
  `max_colors`（既定 4）色までにする（使われているドットの数で重み付けした平均）、(2) すでに絵にある色で十分近い
  （`REUSE_DISTANCE` 以内）ものがあれば、その色を使う（新しい色を増やさない）
- `single`: 1 色。色を指定しなければ、絵の中でいちばん暗い色（それが明るすぎる、L が 0.4 より大きいときは、それを暗くした色）

**パレットを指定したとき**（`palette`。パレット統合の `--palette`）: 外側に付ける輪郭の色も、パレットの中から選ぶ
（パレットにない色を作らない）。`darken` は、暗くした色（まとめた後の色）に最も近いパレットの色を、内側の色
（まとめた中でいちばん暗いもの）より `DARK_MARGIN` 以上暗いパレットの色の中から選ぶ。そういう色が無ければ内側より少しでも
暗い色から、それも無ければパレットでいちばん暗い色にする。`single` の自動の色も同じように選び、指定の色はパレットで最も近い色にする。
切れ目を埋めるときは、もともと絵の中にある輪郭の色を使うので変わらない。パレットを指定しなければ今までと同じ。

**太すぎる輪郭を細くする**（`thin` で輪郭があるとき。`_thin_outline()`）: 輪郭の色（判定で決めた色）を D、
それ以外の不透明なドットを「中身」とする。D のドットのうち、中身に 8 近傍で接するものは残す（中身を囲む 1 ドットの線）。
次の全部に当てはまる D のドットだけを透明にする: (1) シルエットの縁（上下左右のどこかが透明か画像の外）、(2) 中身に 8 近傍で接していない、
(3) 8 近傍に「残す D」がある（2 ドットの太さの外側の段）、(4) 8 近傍の D がすべて「残す D」か、同じように消すドット
（3 ドット以上の暗い面や、中身の無い暗い部品（暗い髪の束・暗い靴など）には手を付けない）。
消したドットは中身に接していないので、明るい色が透明に直接触れることはなく、輪郭は途切れない。1 回だけ（1 ドットだけ）細くする。
AI の絵をドット 2 個分の細かいグリッドで読んだとき（コマの絵）、元の絵の太い輪郭は 2 ドットの太さになる。手直しでこの外側の段を
消すことが多かったので、自動でできるようにした。

**`thin` で輪郭が無いとき**: `add` と同じ位置・色の線のうち、すぐ内側（上下左右、`thick` は斜めも）の縁のドットが
すべて「暗い縁」（判定と同じ基準）である所には付けない。明るい色が透明に直接触れている縁にだけ付ける（迷う所は `add` と同じく付ける）。

色の近さは単純化と同じ OKLab（色み a・b は 1.5 倍。`simplify._distance()`）で測る。乱数は使わず、同点は位置（上から、左から）や
色の番号で決めるので、同じ入力からは必ず同じ結果になる。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .cleanup import _N4, _N8, _RING, _to_labels
from .pipeline import Step
from .simplify import ALPHA_OPAQUE, _distance, rgb_to_oklab

MODES = {"auto": "自動（輪郭があれば切れ目だけ埋め、無ければ外側に付ける）", "add": "外側に 1 ドットの輪郭を付ける",
         "thin": "細く（auto に加えて、2 ドットの太さの外側の輪郭を 1 ドットにする）", "off": "しない"}
DEFAULT_MODE = "auto"
SHAPES = {"thin": "斜めでつながる 1 ドットの線（角が太らない）", "thick": "上下左右でつながる線（角も埋める）"}
DEFAULT_SHAPE = "thin"
COLOR_MODES = {"darken": "内側の色を暗くした色（場所ごとに変わる）", "single": "1 色"}
DEFAULT_COLOR_MODE = "darken"

# 「すでに輪郭がある」の判定
DARK_MARGIN = 0.1          # 縁のドットが内側よりこれ以上暗い（OKLab の L）なら「暗い縁」
INWARD_STEPS = 3           # 内側へ見るドットの数
HAS_OUTLINE_SHARE = 0.6    # 判定できた縁のうち暗い縁がこの割合以上なら「輪郭がある」
OUTLINE_COLOR_SHARE = 0.1  # 暗い縁のうちこの割合以上を占める色を「輪郭の色」とする
# 切れ目を埋める
MAX_GAP = 2                # 埋める切れ目の長さ（ドット）
LOCAL_STEPS = 10           # 2 つのドットが輪郭をたどってこの歩数以内でつながっていれば、同じ線とみなして埋めない
STRAIGHT_COS = 0.99        # 線の向きとの cos がこれ以上なら「まっすぐ続く」
SPLIT_MIN = 3              # 埋めると、道のドットの色の面（上下左右のつながり）が、どちらもこのドット数以上の
                           # 2 つ以上に分かれるなら、塗りの面を横切る道とみなして埋めない
# 輪郭の色（darken）
DARK_L_BASE, DARK_L_SLOPE, DARK_L_RATIO = 0.18, 0.25, 0.7
CHROMA_KEEP = 0.8
MERGE_DISTANCE = 0.08      # 新しい輪郭の色どうしがこれより近ければまとめる
DEFAULT_MAX_COLORS = 4     # 新しい輪郭の色の数の上限（darken）
REUSE_DISTANCE = 0.05      # すでにある色がこれ以内なら、その色を使う
SINGLE_MAX_L = 0.4         # single の自動の色: いちばん暗い色がこれより明るければ、暗くして使う

# 変わったドットの種類（OutlineResult.kinds の値）
KIND_NONE, KIND_ADDED, KIND_GAP, KIND_THIN = range(4)
KIND_NAMES = {KIND_ADDED: "外側に付けた輪郭", KIND_GAP: "切れ目を埋めた", KIND_THIN: "太い輪郭の外側を消した"}


@dataclass
class OutlineDetection:
    """「すでに輪郭がある」の判定の結果。"""

    has_outline: bool
    coverage: float            # 判定できた縁のうち、暗い縁の割合
    dark_edges: int            # 暗い縁のドット数
    judged_edges: int          # 判定できた縁のドット数
    edges: int                 # シルエットの縁のドット数
    colors: list               # 輪郭の色（RGB のタプル、多い順）


@dataclass
class OutlineResult:
    """縁取りの結果。"""

    image: np.ndarray          # 縁取りした RGBA（余白を足したときは大きくなる）
    changed: np.ndarray        # 変わったドット（結果の大きさの真偽）
    kinds: np.ndarray          # 変わった理由（KIND_*）
    mode: str                  # 指定された動き方
    action: str                # 実際にしたこと: "add" / "fill" / "thin"（切れ目を埋めて細くした）/ "none"
    detection: OutlineDetection | None
    padding: tuple             # 足した余白（上, 下, 左, 右）。0 か 1
    new_colors: list           # 新しく増えた色（RGB のタプル）
    outline_colors: list       # 使った輪郭の色（RGB のタプル）
    before: np.ndarray = field(repr=False, default=None)  # 余白を足した後・縁取りの前の RGBA（比較用）

    @property
    def count(self) -> int:
        return int(self.changed.sum())

    @property
    def padded(self) -> bool:
        return any(self.padding)


# --- 色 ------------------------------------------------------------------------------------

def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """OKLab（…×3）を sRGB（0〜255 の uint8。範囲外は切り詰める）にする。"""
    m2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    m1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    lms = np.asarray(lab, np.float64) @ m2i.T
    lin = np.clip((lms ** 3) @ m1i.T, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def darken_lab(lab: np.ndarray) -> np.ndarray:
    """輪郭用に暗くした色（OKLab）。明るさは min(0.18 + 0.25 L, 0.7 L)、色みは 0.8 倍。"""
    lab = np.asarray(lab, np.float64)
    out = lab.copy()
    L = lab[..., 0]
    out[..., 0] = np.minimum(DARK_L_BASE + DARK_L_SLOPE * L, DARK_L_RATIO * L)
    out[..., 1:] = lab[..., 1:] * CHROMA_KEEP
    return out


def darken_rgb(rgb) -> tuple:
    """RGB を輪郭用に暗くした RGB（`darken_lab()`）。"""
    lab = rgb_to_oklab(np.asarray(rgb, np.float64)[None])
    return tuple(int(v) for v in oklab_to_rgb(darken_lab(lab))[0])


# --- 下ごしらえ ----------------------------------------------------------------------------

def _shift(m: np.ndarray, dy: int, dx: int, fill=False) -> np.ndarray:
    """m を (dy, dx) ずらした配列（out[y, x] = m[y - dy, x - dx]。はみ出した所は fill）。"""
    h, w = m.shape
    out = np.full_like(m, fill)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out


def _at(lab: np.ndarray, y: int, x: int) -> int:
    h, w = lab.shape
    return int(lab[y, x]) if 0 <= y < h and 0 <= x < w else -1


def silhouette_edges(opaque: np.ndarray) -> np.ndarray:
    """シルエットの縁: 上下左右のどこかが透明か画像の外の、不透明なドット。"""
    p = np.pad(opaque, 1, constant_values=False)
    h, w = opaque.shape
    near_clear = np.zeros_like(opaque)
    for dy, dx in _N4:
        near_clear |= ~p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return opaque & near_clear


def outer_ring(opaque: np.ndarray, shape: str = DEFAULT_SHAPE) -> np.ndarray:
    """外側に付ける輪郭の位置: 透明のドットのうち、上下左右（thick は斜めも）に不透明なドットがあるもの。"""
    offs = _N4 if shape == "thin" else _N8
    near = np.zeros_like(opaque)
    for dy, dx in offs:
        near |= _shift(opaque, dy, dx)
    return near & ~opaque


# --- 判定 ----------------------------------------------------------------------------------

def _edge_votes(lab: np.ndarray, L: np.ndarray):
    """縁のドットごとの判定: 1 = 暗い縁、0 = 暗くない縁、-1 = 判定しない。(縁のマスク, 判定の配列)。

    比べる相手（内側）: 透明の側と反対の向きへ `INWARD_STEPS` ドットまで（透明に出たらそこまで）と、まわり 8 ドットの不透明なドット。
    そのうち自分と違う色のどれかより `DARK_MARGIN` 以上暗ければ 1。違う色が無いときは、内側へ `INWARD_STEPS` ドット続けて
    同じ色なら 0（広い面の端）、途中で透明に出たら -1（細い所）。
    """
    opaque = lab >= 0
    edges = silhouette_edges(opaque)
    votes = np.full(lab.shape, -1, np.int64)
    for y, x in zip(*np.nonzero(edges)):
        c = int(lab[y, x])
        ref = [_at(lab, y + dy, x + dx) for dy, dx in _N8]
        full = False
        for dy, dx in _N4:
            if _at(lab, y + dy, x + dx) >= 0:
                continue  # この向きは透明ではない
            k = 0
            for k in range(1, INWARD_STEPS + 1):
                v = _at(lab, y - dy * k, x - dx * k)
                if v < 0:
                    break
                ref.append(v)
            else:
                full = True
        diff = [v for v in ref if v >= 0 and v != c]
        if diff:
            votes[y, x] = 1 if L[c] <= max(L[v] for v in diff) - DARK_MARGIN else 0
        elif full:
            votes[y, x] = 0
    return edges, votes


def _detect(lab: np.ndarray, cols: np.ndarray, L: np.ndarray) -> OutlineDetection:
    edges, votes = _edge_votes(lab, L)
    dark = votes == 1
    judged = int((votes >= 0).sum())
    nd = int(dark.sum())
    cov = nd / judged if judged else 0.0
    colors = []
    if nd:
        cnt = np.bincount(lab[dark].astype(np.intp), minlength=len(cols))  # 32 ビット（Pyodide）では int64 を渡せない
        for c in np.argsort(-cnt, kind="stable"):
            if cnt[c] >= OUTLINE_COLOR_SHARE * nd and cnt[c] > 0:
                colors.append(tuple(int(v) for v in cols[c]))
    return OutlineDetection(cov >= HAS_OUTLINE_SHARE and bool(colors), cov, nd, judged, int(edges.sum()), colors)


def detect_outline(rgba: np.ndarray) -> OutlineDetection:
    """シルエットの縁にすでに暗い輪郭があるかを判定する（基準はモジュールの説明）。"""
    a = _check(rgba)
    lab, cols = _to_labels(a)
    if len(cols) == 0:
        return OutlineDetection(False, 0.0, 0, 0, 0, [])
    L = rgb_to_oklab(cols.astype(np.float64))[:, 0]
    return _detect(lab, cols, L)


# --- 切れ目を埋める ------------------------------------------------------------------------

def _components8(mask: np.ndarray) -> np.ndarray:
    """真のドットの 8 近傍のつながりの番号（偽は -1）。"""
    h, w = mask.shape
    comp = np.full((h, w), -1, np.int64)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if comp[y0, x0] >= 0:
            continue
        comp[y0, x0] = n
        q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in _N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and comp[yy, xx] < 0:
                    comp[yy, xx] = n
                    q.append((yy, xx))
        n += 1
    return comp


def _ring_mask(line: np.ndarray, y: int, x: int) -> int:
    h, w = line.shape
    m = 0
    for i, (dy, dx) in enumerate(_RING):
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and line[yy, xx]:
            m |= 1 << i
    return m


def _end_direction(line: np.ndarray, y: int, x: int):
    """線の端なら、線が伸びてきた向き（端のドット − 隣の輪郭のドットの平均）。端でなければ None。

    端: 8 近傍の輪郭のドットが 0 個、1 個、または 2 個で互いに上下左右に隣り合う（輪郭の角は端ではない）。
    """
    m = _ring_mask(line, y, x)
    on = [i for i in range(8) if m >> i & 1]
    if len(on) > 2 or (len(on) == 2 and (on[1] - on[0]) % 8 not in (1, 7)):
        return None
    if not on:
        return (0.0, 0.0)
    return (-sum(_RING[i][0] for i in on) / len(on), -sum(_RING[i][1] for i in on) / len(on))


def _near_on_line(line: np.ndarray, a, b, limit: int) -> bool:
    """輪郭の線をたどって a から b まで `limit` 歩以内で行けるか。"""
    h, w = line.shape
    seen = {a: 0}
    q = deque([a])
    while q:
        p = q.popleft()
        d = seen[p]
        if p == b:
            return True
        if d >= limit:
            continue
        for dy, dx in _N8:
            n = (p[0] + dy, p[1] + dx)
            if 0 <= n[0] < h and 0 <= n[1] < w and line[n] and n not in seen:
                seen[n] = d + 1
                q.append(n)
    return False


def _paths(a, b, n: int):
    """a から b への、間に n ドットを通る最短の 8 近傍の道（間のドットの列）をすべて。"""
    if n == 0:
        return [[]] if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 else []
    out = []
    for dy, dx in _N8:
        p = (a[0] + dy, a[1] + dx)
        if max(abs(p[0] - b[0]), abs(p[1] - b[1])) == n:
            out.extend([p] + rest for rest in _paths(p, b, n - 1))
    return out


def _splits_fill(lab: np.ndarray, path) -> bool:
    """道のドットを塗ると、そのドットの色の面（同じ色の上下左右のつながり）が、どれも `SPLIT_MIN` ドット以上の
    2 つ以上に分かれるか（塗りの面を横切る道か）。

    本当の切れ目は、面の端（シルエットの縁や、線に挟まれた小さなすき間）にあり、塗っても面は分かれないか、
    分かれても片方はごく小さい。マントの内側の影のように、面の真ん中を横切る道は、面を 2 つに分けてしまう。
    """
    h, w = lab.shape
    blocked = set(path)
    for color in {int(lab[p]) for p in path}:
        starts = [(py + dy, px + dx) for py, px in path for dy, dx in _N4
                  if 0 <= py + dy < h and 0 <= px + dx < w and (py + dy, px + dx) not in blocked
                  and lab[py + dy, px + dx] == color]
        seen: set = set()
        big = 0
        for st in starts:
            if st in seen:
                continue
            seen.add(st)
            q = deque([st])
            n = 0
            while q:
                y, x = q.popleft()
                n += 1
                for dy, dx in _N4:
                    nb = (y + dy, x + dx)
                    if (0 <= nb[0] < h and 0 <= nb[1] < w and nb not in seen and nb not in blocked
                            and lab[nb] == color):
                        seen.add(nb)
                        q.append(nb)
            if n >= SPLIT_MIN:
                big += 1
        if big >= 2:
            return True
    return False


def _fill_gaps(lab: np.ndarray, line_colors: set, dist_to: np.ndarray, kinds: np.ndarray) -> None:
    """輪郭の切れ目（1〜MAX_GAP ドット）を、線の端の色で埋める（`lab` を書き換える）。短い切れ目から順に埋める。

    間の道が何通りかあるときは、線が伸びてきた向きにまっすぐ続く道、次に輪郭の色に近いドットを通る道を選ぶ。
    """
    h, w = lab.shape
    line = np.isin(lab, list(line_colors))
    edges = silhouette_edges(lab >= 0)
    comp = _components8(line)
    on_edge = set(np.unique(comp[line & edges]).tolist())
    for gap in range(1, MAX_GAP + 1):
        r = gap + 1
        for y, x in zip(*np.nonzero(line)):
            if comp[y, x] not in on_edge:
                continue
            d = _end_direction(line, y, x)
            if d is None:
                continue
            a = (int(y), int(x))
            best = None
            for ty in range(y - r, y + r + 1):
                for tx in range(x - r, x + r + 1):
                    if max(abs(ty - y), abs(tx - x)) != r or not (0 <= ty < h and 0 <= tx < w) or not line[ty, tx]:
                        continue
                    b = (ty, tx)
                    if _near_on_line(line, a, b, LOCAL_STEPS):
                        continue
                    for path in _paths(a, b, gap):
                        if not all(0 <= py < h and 0 <= px < w and lab[py, px] >= 0 and not line[py, px]
                                   for py, px in path):
                            continue
                        sy, sx = path[0][0] - y, path[0][1] - x   # 線の向きとの cos が大きいほどまっすぐ
                        straight = -round((sy * d[0] + sx * d[1]) / np.hypot(sy, sx), 6)
                        # 塗りの面を 2 つに分ける道は、1 ドットの切れ目で線がそのまままっすぐ続くときだけ埋める
                        # （体と腕の境目の線の先など）。曲がる道・2 ドットの道は、面の中の影の境目などかもしれないので埋めない
                        if not (gap == 1 and -straight >= STRAIGHT_COS) and _splits_fill(lab, path):
                            continue
                        cost = (straight, round(sum(float(dist_to[lab[p]]) for p in path), 6), path)
                        if best is None or cost < best:
                            best = cost
            if best is None:
                continue
            c = int(lab[y, x])
            for p in best[2]:
                lab[p] = c
                line[p] = True
                comp[p] = comp[y, x]
                kinds[p] = KIND_GAP


# --- 太い輪郭を細くする --------------------------------------------------------------------

def _dilate8(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    for dy, dx in _N8:
        out |= _shift(m, dy, dx)
    return out


def thin_mask(lab: np.ndarray, line_colors) -> np.ndarray:
    """2 ドットの太さの外側の輪郭のうち、消すドット（基準はモジュールの説明の「太すぎる輪郭を細くする」）。"""
    line = np.isin(lab, list(line_colors)) & (lab >= 0)
    inside = (lab >= 0) & ~line
    near_inside = _dilate8(inside)
    keep = line & near_inside
    cand = line & silhouette_edges(lab >= 0) & ~near_inside & _dilate8(keep)
    # 8 近傍の輪郭のドットが、すべて「残す」か「消す候補」のときだけ（3 ドット以上の暗い面・中身の無い暗い部品は残す）
    other = line & ~keep & ~cand
    return cand & ~_dilate8(other)


def _thin_outline(lab: np.ndarray, line_colors, kinds: np.ndarray) -> None:
    """2 ドットの太さの外側の輪郭を透明（-1）にする（`lab` を書き換える）。"""
    m = thin_mask(lab, line_colors)
    lab[m] = -1
    kinds[m] = KIND_THIN


# --- 外側に付ける --------------------------------------------------------------------------

def _inside_color(lab: np.ndarray, L: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """外側に付ける各ドットの内側の色の番号: 上下左右の不透明なドットで多い色（同数なら暗い色、さらに同数なら番号の小さい色）。
    上下左右に無ければ（thick の角）斜めのドットから同じように選ぶ。"""
    out = np.full(lab.shape, -1, np.int64)
    for y, x in zip(*np.nonzero(ring)):
        for offs in (_N4, _N8[4:]):
            nb = [_at(lab, y + dy, x + dx) for dy, dx in offs]
            nb = [v for v in nb if v >= 0]
            if nb:
                out[y, x] = min(set(nb), key=lambda v: (-nb.count(v), L[v], v))
                break
    return out


def _reduce_colors(lab3: np.ndarray, weights: np.ndarray, max_colors: int):
    """近い色（MERGE_DISTANCE 未満）をまとめ、max_colors 色までにする。(各色のグループ番号, グループの OKLab)。"""
    groups = [[i] for i in range(len(lab3))]

    def center(g):
        w = weights[g]
        return (lab3[g] * w[:, None]).sum(0) / w.sum()

    while len(groups) > 1:
        cs = np.array([center(g) for g in groups])
        d = _distance(cs, cs)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[i, j] >= MERGE_DISTANCE and len(groups) <= max_colors:
            break
        i, j = min(i, j), max(i, j)
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
    grp = np.zeros(len(lab3), np.int64)
    for k, g in enumerate(groups):
        grp[g] = k
    return grp, np.array([center(g) for g in groups])


def _palette_dark(target: np.ndarray, base_L: float, pal: np.ndarray, plab: np.ndarray) -> np.ndarray:
    """パレットの中から輪郭の色を選ぶ: 内側の明るさ base_L より DARK_MARGIN 以上暗い色の中で、target（OKLab）に最も近い色。
    無ければ base_L より暗い色の中から、それも無ければパレットでいちばん暗い色。同点はパレットの早い色。"""
    for ok in (plab[:, 0] <= base_L - DARK_MARGIN, plab[:, 0] < base_L):
        if ok.any():
            cand = np.flatnonzero(ok)
            d = _distance(np.asarray(target, np.float64)[None], plab[cand])[0]
            return pal[cand[int(np.argmin(d))]]
    return pal[int(np.argmin(plab[:, 0]))]


def _add_outline(lab, cols, lab3, ring, color_mode, color, max_colors, palette=None):
    """外側の輪郭の各ドットの RGB を決める。(輪郭の色の配列 n×3（ring の True の順）, 使った色のリスト)。"""
    n = int(ring.sum())
    if n == 0:
        return np.zeros((0, 3), np.uint8), []
    if color_mode == "single":
        if color is None:
            darkest = int(np.argmin(lab3[:, 0])) if len(cols) else None
            if darkest is None:
                rgb = (0, 0, 0)
            elif lab3[darkest, 0] <= SINGLE_MAX_L:
                rgb = tuple(int(v) for v in cols[darkest])
            elif palette is not None:
                rgb = tuple(int(v) for v in _palette_dark(darken_lab(lab3[darkest]), lab3[darkest, 0], *palette))
            else:
                rgb = tuple(int(v) for v in oklab_to_rgb(darken_lab(lab3[darkest][None]))[0])
        else:
            rgb = tuple(int(v) for v in color)
        if palette is not None and rgb not in {tuple(int(v) for v in c) for c in palette[0]}:
            # 指定の色、または絵のいちばん暗い色がパレットにない: パレットで最も近い色にする
            pal, plab = palette
            j = int(np.argmin(_distance(rgb_to_oklab(np.array(rgb, np.float64)[None]), plab)[0]))
            rgb = tuple(int(v) for v in pal[j])
        return np.tile(np.array(rgb, np.uint8), (n, 1)), [rgb]
    # darken: 内側の色ごとに暗くした色を作り、近い色をまとめる
    inside = _inside_color(lab, lab3[:, 0], ring)[ring]
    base, cnt = np.unique(inside, return_counts=True)
    dark = darken_lab(lab3[base])
    grp, centers = _reduce_colors(dark, cnt.astype(np.float64), max_colors)
    rgbs = oklab_to_rgb(centers)
    if palette is not None:
        # パレットの中から選ぶ（まとめた中でいちばん暗い内側の色より暗い色）
        pal, plab = palette
        for k in range(len(rgbs)):
            rgbs[k] = _palette_dark(centers[k], float(lab3[base[grp == k], 0].min()), pal, plab)
    elif len(cols):
        # すでにある色で十分近いものがあれば、その色を使う
        d = _distance(rgb_to_oklab(rgbs.astype(np.float64)), lab3)
        for k in range(len(rgbs)):
            j = int(np.argmin(d[k]))
            if d[k, j] <= REUSE_DISTANCE:
                rgbs[k] = cols[j]
    lut = {int(b): rgbs[grp[i]] for i, b in enumerate(base)}
    out = np.array([lut[int(v)] for v in inside], np.uint8)
    used = sorted({tuple(int(v) for v in c) for c in out})
    return out, used


# --- 本体 ----------------------------------------------------------------------------------

def _check(rgba) -> np.ndarray:
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8:
        raise ValueError("RGBA の uint8 配列（高さ × 幅 × 4）が必要です")
    return a


def _padding(opaque: np.ndarray) -> tuple:
    """シルエットが接している辺（上, 下, 左,右）に 1 ドットの余白を足す。"""
    if not opaque.any():
        return (0, 0, 0, 0)
    return (int(opaque[0].any()), int(opaque[-1].any()), int(opaque[:, 0].any()), int(opaque[:, -1].any()))


def _colors_of(a: np.ndarray) -> set:
    return {tuple(int(v) for v in c) for c in a[a[..., 3] >= ALPHA_OPAQUE][:, :3]}


def outline_sprite(rgba: np.ndarray, mode: str = DEFAULT_MODE, shape: str = DEFAULT_SHAPE,
                   color_mode: str = DEFAULT_COLOR_MODE, color=None, pad: bool = True,
                   max_colors: int = DEFAULT_MAX_COLORS, palette=None) -> OutlineResult:
    """シルエットの縁を縁取りする（`mode` は auto / add / off。詳しくはモジュールの説明）。

    `color` を指定すると `color_mode` に関係なくその 1 色で描く。`palette`（n×3）を指定すると、外側に付ける輪郭の色を
    パレットの中から選ぶ（指定の `color` もパレットで最も近い色にする）。入力の配列は書き換えない。
    """
    if mode not in MODES:
        raise ValueError(f"縁取りの動き方は {' / '.join(MODES)} のどれかで指定してください（{mode}）")
    if shape not in SHAPES:
        raise ValueError(f"縁取りの線の形は {' / '.join(SHAPES)} のどれかで指定してください（{shape}）")
    if color_mode not in COLOR_MODES:
        raise ValueError(f"縁取りの色は {' / '.join(COLOR_MODES)} のどれかで指定してください（{color_mode}）")
    if max_colors < 1:
        raise ValueError("縁取りの色の数は 1 以上で指定してください")
    if color is not None:
        color_mode = "single"
    a = _check(rgba)
    h, w = a.shape[:2]

    def unchanged(action, det):
        return OutlineResult(a.copy(), np.zeros((h, w), bool), np.zeros((h, w), np.int64), mode, action, det,
                             (0, 0, 0, 0), [], det.colors if det else [], a.copy())

    if mode == "off":
        return unchanged("none", None)
    lab, cols = _to_labels(a)
    if len(cols) == 0:
        return unchanged("none", None)
    lab3 = rgb_to_oklab(cols.astype(np.float64))
    det = _detect(lab, cols, lab3[:, 0]) if mode in ("auto", "thin") else None

    if det is not None and det.has_outline:
        # すでに輪郭がある: 切れ目だけを、輪郭の色で埋める（シルエットも大きさも変えない）。
        # thin は、そのあと 2 ドットの太さの外側の輪郭を 1 ドットにする（シルエットが 1 ドット小さくなる所がある）
        key = {tuple(int(v) for v in c): i for i, c in enumerate(cols)}
        line_colors = {key[c] for c in det.colors}
        main = key[det.colors[0]]
        dist_to = _distance(lab3, lab3[[main]])[:, 0]
        new = lab.copy()
        kinds = np.zeros((h, w), np.int64)
        _fill_gaps(new, line_colors, dist_to, kinds)
        if mode == "thin":
            _thin_outline(new, line_colors, kinds)
        out = a.copy()
        ch = new != lab
        keep = ch & (new >= 0)
        out[..., :3][keep] = cols[new[keep]]
        out[ch & (new < 0)] = 0
        return OutlineResult(out, ch, np.where(ch, kinds, KIND_NONE), mode, "thin" if mode == "thin" else "fill",
                             det, (0, 0, 0, 0), [], det.colors, a.copy())

    # 輪郭が無い（または add）: 外側に付ける
    opaque = lab >= 0
    padding = _padding(opaque) if pad else (0, 0, 0, 0)
    t, b, l, r = padding
    a2 = np.pad(a, ((t, b), (l, r), (0, 0)))
    lab2 = np.pad(lab, ((t, b), (l, r)), constant_values=-1)
    ring = outer_ring(lab2 >= 0, shape)
    if mode == "thin":
        # 暗い縁（すでに輪郭のような縁）の外側には付けない。明るい色が透明に触れている縁にだけ付ける
        _, votes = _edge_votes(lab2, lab3[:, 0])
        light = (lab2 >= 0) & (votes != 1)
        near_light = np.zeros_like(ring)
        for dy, dx in (_N4 if shape == "thin" else _N8):
            near_light |= _shift(light, dy, dx)
        ring &= near_light
    pal = None
    if palette is not None:
        p = np.asarray(palette, np.uint8).reshape(-1, 3)
        if len(p) == 0:
            raise ValueError("パレットに色がありません")
        pal = (p, rgb_to_oklab(p.astype(np.float64)))
    rgb, used = _add_outline(lab2, cols, lab3, ring, color_mode, color, max_colors, pal)
    out = a2.copy()
    out[..., :3][ring] = rgb
    out[..., 3][ring] = 255
    kinds = np.where(ring, KIND_ADDED, KIND_NONE)
    before = _colors_of(a)
    new_colors = [c for c in used if c not in before]
    return OutlineResult(out, ring, kinds, mode, "add", det, padding, new_colors, used, a2)


@dataclass
class OutlineStep(Step):
    """縁取りの工程。直前の結果は `last`（`OutlineResult`）に残る。"""

    mode: str = DEFAULT_MODE
    shape: str = DEFAULT_SHAPE
    color_mode: str = DEFAULT_COLOR_MODE
    color: tuple | None = None
    pad: bool = True
    max_colors: int = DEFAULT_MAX_COLORS
    palette: np.ndarray | None = field(default=None, repr=False)   # 指定すると外側の輪郭の色をパレットから選ぶ
    name: str = "outline"
    last: OutlineResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"縁取りの動き方は {' / '.join(MODES)} のどれかで指定してください（{self.mode}）")
        if self.shape not in SHAPES:
            raise ValueError(f"縁取りの線の形は {' / '.join(SHAPES)} のどれかで指定してください（{self.shape}）")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"縁取りの色は {' / '.join(COLOR_MODES)} のどれかで指定してください（{self.color_mode}）")

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        self.last = outline_sprite(rgba, self.mode, self.shape, self.color_mode, self.color, self.pad,
                                   self.max_colors, self.palette)
        return self.last.image

    def describe(self) -> str:
        r = self.last
        if r is None:
            return f"縁取り（{self.mode}）"
        parts = [self.mode]
        d = r.detection
        if d is not None:
            parts.append(f"輪郭{'あり' if d.has_outline else 'なし'}［暗い縁 {d.coverage:.0%}］")
        if r.action == "fill":
            parts.append(f"切れ目を {r.count} ドット埋めた")
        elif r.action == "thin":
            parts.append(f"切れ目を {int((r.kinds == KIND_GAP).sum())} ドット埋めた")
            parts.append(f"2 ドットの太さの外側の輪郭を {int((r.kinds == KIND_THIN).sum())} ドット消した")
        elif r.action == "add":
            parts.append(f"外側に {r.count} ドット付けた（{self.shape}）")
            pad = "・".join(n for n, v in zip(("上", "下", "左", "右"), r.padding) if v)
            parts.append(f"余白 {pad} に 1 ドット" if pad else "余白なし")
            parts.append(f"増えた色 {len(r.new_colors)}" + ("［パレットの色］" if self.palette is not None else ""))
        return "縁取り（" + "、".join(parts) + "）"
