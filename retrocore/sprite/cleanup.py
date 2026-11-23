"""クリーンアップの工程: 単純化の後に残る余計なドットを片付ける（新しい色は作らない）。

ドット絵風の AI 画像を グリッド検出 → 単純化 まで通すと、全体は平塗りになるが、ぼかしや半ドットのずれの名残で
次のような余計なドットが残る（ai03 で数〜十数ドットずつ）。

- 面の中に、まわりと違う色の 1 ドットだけの点
- 輪郭と面の境目に細く残る、2 色の中間の色のドット（輪郭と赤の中間の暗い赤など）
- 輪郭の階段の角に余分に付いた 1 ドット（L 字の角。ダブルコーナー）
- 輪郭が部分的に 2 ドットの太さになった所

どれも **そのドットを、まわりにある、すでに使われている色に置き換える** だけで直す。新しい色は作らないので、
パレットは増えない（減ることはある）。透明は、外側に浮いた不透明の 1 ドットと、内側に空いた 1 ドットの透明の穴だけを片付け、
それ以外のシルエット（透明と不透明の境目）は変えない。輪郭を新しく付ける処理（縁取り）はしない。

色の近さは単純化と同じ OKLab（色み a・b は 1.5 倍。`simplify._distance()`）で測る。

小さな部品を守る基準（`cleanup_sprite()` の各手順）:
- **同じ色が 2 ドット以上つながって（斜めも含む）いれば部品とみなし、孤立ドットとしては扱わない**（2 ドットの目など）
- 孤立した 1 ドットでも、まわりのどの色とも離れていて（距離 `near` より遠い）、まわりの 2 色の中間の色でもなければ、
  意味のある点（頬・ハイライト・1 ドットの目）とみなして残す。消すのは `strong` だけ。
  ただし、まわりのどの色よりはっきり暗い 1 ドット（OKLab の明るさで 0.15 以上。1 ドットの目）は `strong` でも残す
- 「まわりの色に近い（溶け込む）」は、すぐまわりの色で判定する: 上下左右に接する色か、斜めに 2 つ以上接する色だけと比べる
  （暗い目の中のハイライトの斜めに肌が 1 ドット接していても、肌に溶け込むとはみなさない）
- 上下左右（3 つ以上が不透明）のどの色よりはっきり明るい 1 ドット（OKLab の明るさで 0.15 以上。目のハイライト）は、
  どの強さでも残す
- 中間色の塊（手順 b）は、次をすべて満たすものだけを直す
  - 塊のまわりにある 2 色を結ぶ線分の途中にある色（ぼかしの名残）。輪郭（まわりより暗い）・目（まわりより暗い）・
    頬（まわりより色みが強い）は線分の途中に来ないので対象外
  - 画像全体でほとんど使われていない色（不透明な部分の 3% 以下）。広く使われている色（手の影の色など）は、
    細い所で 2 色の中間に見えても、意味のある色として残す
  - 同じ色のつながりが小さい（`normal` は 6 ドット以下で、幅 1 ドットの細い塊だけ）
  - 塊のドットの半分より多くが、横か縦に見て両側を「その色を中間にする 2 色」に挟まれていて、どちらの向きでも
    「はっきり違う 2 つの面を分ける線」になっていない（`_bracketed()`）。白と薄い灰色の間に並んだ剣の影のような線は残す
- 輪郭の手直し（手順 c・d）は、輪郭の色（シルエットの縁で最も多い、暗い色）だけを対象にし、透明に接するドットは変えない。
  さらに、シルエットの輪郭（透明に接する輪郭のドット）と、そこから輪郭の色を 8 近傍でたどってつながる線だけを対象にする。
  塗りの面の中に浮いた輪郭の色の塊（まわりを別の色で囲まれた目・口）は部品として変えない（丸い目の角を削らない）。
  直したあとも輪郭が斜めも含めてつながっていること（そのドットのまわりの輪郭のドットが 1 つにつながっていること）を
  確かめてから置き換える
- 置き換え先の選び方: 候補の色のうち、まわりに多い色。ただし、シルエットの縁のドットは輪郭の色が候補にあれば輪郭の色に
  （縁の輪郭が途切れないように）、内側のドットで上下左右に輪郭があるときは輪郭以外の色に（輪郭が太らないように）する

強さ（`LEVELS`）:

| 強さ | 内容 |
| --- | --- |
| `off` | 何もしない |
| `light` | 透明の 1 ドット（浮いた点・穴）、まわりに溶け込む孤立ドット（まわりの色に近いか、まわりの 2 色の中間）|
| `normal`（既定） | `light` ＋ (b) 中間色の細い塊（6 ドットまで）＋ (d) 輪郭の角の余分な 1 ドット（L 字の角） |
| `strong` | `normal` ＋ 目立つ孤立ドットも消す ＋ 中間色の塊は 12 ドットまで（太さも問わず、使われている割合 6% まで）＋ (c) 2 ドットの太さの輪郭を 1 ドットに |

乱数は使わず、同点は色の番号（RGB の値の小さい順）で決めるので、同じ入力からは必ず同じ結果になる。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .pipeline import Step
from .simplify import ALPHA_OPAQUE, _distance, rgb_to_oklab

# 8 近傍（上下左右が先頭の 4 つ）
_N4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
# 8 近傍を時計回りに並べたもの（輪郭のつながりの判定用）
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))

# 中間色の判定（simplify.transition_mask() と同じ考え方。2 色の差・線分上の位置・線分からのずれの割合）
BETWEEN_MIN_DIFF = 0.08
BETWEEN_T_RANGE = (0.15, 0.85)
BETWEEN_PERP = 0.3
BETWEEN_PERP_ABS = 0.05      # 線分からのずれの上限（距離）。2 色が遠いとき（輪郭と明るい面など）に、
                             # 色みの強い頬の色まで「中間」とみなさないように
# 輪郭の色とみなす条件: シルエットの縁（透明に接する不透明なドット）のうち、この割合以上を占める色
OUTLINE_MIN_SHARE = 0.3
# まわりのどの色よりこれ以上暗い（OKLab の L）孤立ドットは、目とみなして strong でも残す
DARK_DOT_MARGIN = 0.15
# まわり（上下左右）のどの色よりこれ以上明るい（OKLab の L）孤立ドットは、ハイライトとみなしてどの強さでも残す
BRIGHT_DOT_MARGIN = 0.15
# 「溶け込む」の判定に使うまわりの色: 8 近傍での重み（上下左右 2、斜め 1）の合計がこれ以上の色だけ
# （斜めに 1 ドット接しているだけの色には溶け込むとみなさない）
NEAR_MIN_WEIGHT = 2
# 繰り返しの上限（1 回直すと、新しく孤立ドットができることがあるため）
MAX_PASSES = 4


@dataclass(frozen=True)
class CleanupLevel:
    """クリーンアップの強さ。"""

    alpha: bool = True              # 外側に浮いた不透明の 1 ドット・内側の 1 ドットの透明の穴を片付ける
    isolated: bool = True           # (a) 孤立した 1 ドットを、まわりの色に置き換える
    near: float = 0.12              # (a) まわりの色との距離がこれ以下なら「溶け込む」ドットとして直す
    distinct_isolated: bool = False  # (a) まわりと離れた色の 1 ドット（頬・ハイライトなど）も直す
    run: int = 0                    # (b) 中間色の塊をこのドット数まで直す（0 = しない）
    thin_runs: bool = True          # (b) 幅 1 ドットの塊だけを直す
    minor_share: float = 0.03       # (b) 画像全体でこの割合以下しか使われていない色だけを直す
    corners: bool = False           # (d) 輪郭の角の余分な 1 ドット（L 字の角）を直す
    thin: bool = False              # (c) 2 ドットの太さの輪郭を 1 ドットにする


LEVELS: dict[str, CleanupLevel | None] = {
    "off": None,
    "light": CleanupLevel(),
    "normal": CleanupLevel(run=6, corners=True),
    "strong": CleanupLevel(near=0.18, distinct_isolated=True, run=12, thin_runs=False, minor_share=0.06,
                           corners=True, thin=True),
}
LEVEL_HELP = {"off": "しない", "light": "弱い（溶け込む孤立ドットと透明の 1 ドットだけ）",
              "normal": "ふつう（＋中間色の小さな塊・輪郭の角）", "strong": "強い（＋目立つ 1 ドット・2 ドットの太さの輪郭）"}
DEFAULT_LEVEL = "normal"

# 変更の種類（CleanupResult.kinds の値）
KIND_NONE, KIND_ALPHA, KIND_ISOLATED, KIND_BETWEEN, KIND_CORNER, KIND_THIN = range(6)
KIND_NAMES = {KIND_ALPHA: "透明の点・穴", KIND_ISOLATED: "孤立ドット", KIND_BETWEEN: "中間色",
              KIND_CORNER: "輪郭の角", KIND_THIN: "輪郭の太さ"}


@dataclass
class CleanupResult:
    """クリーンアップの結果。"""

    image: np.ndarray          # 直した RGBA
    changed: np.ndarray        # 変わったドット（高さ × 幅の真偽）
    kinds: np.ndarray          # 変わった理由（高さ × 幅、KIND_*。最後に変えた理由）
    outline: tuple | None      # 輪郭の色とみなした色（RGB）。見つからなければ None
    level: str

    @property
    def count(self) -> int:
        return int(self.changed.sum())

    def counts(self) -> dict[str, int]:
        """変わったドットの数（理由ごと）。"""
        return {name: int(((self.kinds == k) & self.changed).sum()) for k, name in KIND_NAMES.items()}


# --- 下ごしらえ ----------------------------------------------------------------------------

def _to_labels(a: np.ndarray):
    """RGBA → (色番号の配列（透明は -1）, 色の表 k×3)。色の番号は RGB の値の小さい順。"""
    opaque = a[..., 3] >= ALPHA_OPAQUE
    key = a[..., :3].astype(np.int64) @ np.array([65536, 256, 1])
    uniq, inv = np.unique(key[opaque], return_inverse=True)
    lab = np.full(opaque.shape, -1, np.int64)
    lab[opaque] = inv.reshape(-1)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    return lab, cols


def _between_table(lab3: np.ndarray) -> np.ndarray:
    """between[c, a, b]: 色 c が色 a と色 b を結ぶ線分の途中にある（2 色の中間の色）か。"""
    k = len(lab3)
    w = np.array([1.0, 1.5, 1.5])  # 色み a・b は 1.5 倍（simplify と同じ）
    p = lab3 * w
    out = np.zeros((k, k, k), bool)
    t0, t1 = BETWEEN_T_RANGE
    for a in range(k):
        v = p - p[a]                     # a → b
        n2 = (v ** 2).sum(1)             # (b,)
        u = p - p[a]                     # a → c
        t = (u @ v.T) / np.where(n2 > 0, n2, 1.0)[None, :]   # (c, b)
        perp2 = (u ** 2).sum(1)[:, None] - t ** 2 * n2[None, :]
        ok = (n2[None, :] >= BETWEEN_MIN_DIFF ** 2) & (t >= t0) & (t <= t1) & (perp2 <= np.minimum(BETWEEN_PERP ** 2 * n2[None, :], BETWEEN_PERP_ABS ** 2))
        out[:, a, :] = ok
    idx = np.arange(k)
    out[idx, idx, :] = False
    out[idx, :, idx] = False
    return out


def _padded(lab: np.ndarray) -> np.ndarray:
    """外側を透明（-1）で 1 ドット囲んだ配列（画像の端はシルエットの外とみなす）。"""
    return np.pad(lab, 1, constant_values=-1)


def _neighbors(lab: np.ndarray, offsets) -> np.ndarray:
    """(len(offsets), 高さ, 幅) の近傍の色番号。"""
    p = _padded(lab)
    h, w = lab.shape
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in offsets])


def _outline_color(lab: np.ndarray, lab3: np.ndarray) -> int | None:
    """輪郭の色の番号: シルエットの縁（4 近傍に透明がある不透明なドット）で最も多く、不透明な部分の中央より暗い色。"""
    n4 = _neighbors(lab, _N4)
    edge = (lab >= 0) & (n4 < 0).any(0)
    if not edge.any():
        return None
    cnt = np.bincount(lab[edge].astype(np.intp), minlength=len(lab3))  # 32 ビット（Pyodide）では int64 を渡せない
    c = int(np.argmax(cnt))
    if cnt[c] < OUTLINE_MIN_SHARE * edge.sum():
        return None
    if lab3[c, 0] >= np.median(lab3[lab[lab >= 0], 0]):
        return None
    return c


def _components(lab: np.ndarray):
    """同じ色の 8 近傍のつながり。(つながりの番号の配列（透明は -1）, 各つながりのドット数)。

    1 ドットずつたどるので、NumPy の配列ではなく Python のリストで計算する（要素 1 つの読み書きが速い）。
    """
    h, w = lab.shape
    L = lab.tolist()
    comp = [[-1] * w for _ in range(h)]
    sizes = []
    for y0, x0 in zip(*np.nonzero(lab >= 0)):
        y0, x0 = int(y0), int(x0)
        if comp[y0][x0] >= 0:
            continue
        c = L[y0][x0]
        n = len(sizes)
        comp[y0][x0] = n
        q = deque([(y0, x0)])
        size = 0
        while q:
            y, x = q.popleft()
            size += 1
            for dy, dx in _N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and comp[yy][xx] < 0 and L[yy][xx] == c:
                    comp[yy][xx] = n
                    q.append((yy, xx))
        sizes.append(size)
    return np.array(comp, np.int64).reshape(h, w), np.array(sizes, np.int64)


# 8 近傍のパターン（_RING の順のビット）ごとの、8 近傍どうしのつながりの数
def _ring_components_table() -> np.ndarray:
    tbl = np.zeros(256, np.int64)
    for m in range(256):
        on = [i for i in range(8) if m >> i & 1]
        seen = set()
        n = 0
        for s in on:
            if s in seen:
                continue
            n += 1
            st = [s]
            seen.add(s)
            while st:
                i = st.pop()
                yi, xi = _RING[i]
                for j in on:
                    if j not in seen:
                        yj, xj = _RING[j]
                        if max(abs(yi - yj), abs(xi - xj)) == 1:
                            seen.add(j)
                            st.append(j)
        tbl[m] = n
    return tbl


_RING_COMPONENTS = _ring_components_table()


@dataclass
class _Ctx:
    """手順で共有する表。"""

    dist: np.ndarray       # 色どうしの距離（k×k）
    between: np.ndarray    # between[c, a, b]: 色 c が a と b の中間の色か
    outline: int | None    # 輪郭の色の番号
    level: CleanupLevel
    lab3: np.ndarray       # 各色の OKLab
    # 1 ドットずつの判定で使う、同じ表の Python のリスト版（要素 1 つの読み出しが NumPy より速い）
    distl: list = field(default=None, repr=False)        # dist.tolist()
    light: list = field(default=None, repr=False)        # 各色の OKLab の明るさ L
    _between_rows: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.distl = self.dist.tolist()
        self.light = self.lab3[:, 0].tolist()

    def between_row(self, c: int) -> list:
        """between[c] のリスト版（色ごとに必要になったときに作る）。"""
        row = self._between_rows.get(c)
        if row is None:
            row = self._between_rows[c] = self.between[c].tolist()
        return row


# --- 各手順 --------------------------------------------------------------------------------

def _alpha_specks(lab: np.ndarray, kinds: np.ndarray) -> None:
    """外側に浮いた不透明の 1 ドット（8 近傍がすべて透明）を透明に、内側の 1 ドットの穴（4 近傍がすべて不透明）を埋める。"""
    n8 = _neighbors(lab, _N8)
    speck = (lab >= 0) & (n8 < 0).all(0)
    new = lab.copy()
    new[speck] = -1
    kinds[speck] = KIND_ALPHA
    n4 = n8[:4]
    hole = (lab < 0) & (n4 >= 0).all(0)
    for y, x in zip(*np.nonzero(hole)):
        new[y, x] = _majority(n8[:, y, x], weights4=True)
        kinds[y, x] = KIND_ALPHA
    lab[:] = new


_NB_WEIGHTS = np.array([2, 2, 2, 2, 1, 1, 1, 1])  # 8 近傍の重み（上下左右 2、斜め 1）
_NB_WEIGHTS_L = _NB_WEIGHTS.tolist()


def _majority(nb: np.ndarray, weights4: bool = True) -> int:
    """近傍の色番号（透明 -1 を除く）の多数決。上下左右は 2、斜めは 1 と数える。同点は番号の小さい色。"""
    wts = np.array([2, 2, 2, 2, 1, 1, 1, 1]) if weights4 else np.ones(len(nb), int)
    best, best_w = -1, -1
    for c in np.unique(nb[nb >= 0]):
        wsum = int(wts[nb == c].sum())
        if wsum > best_w:
            best, best_w = int(c), wsum
    return best


def _pick(c: int, nb: np.ndarray, ctx: "_Ctx", allowed=None) -> int:
    """近傍の色から置き換え先を選ぶ。`allowed`（候補の色の集合）があればその中から、無ければ近傍のすべての色から、
    近傍に多い色（上下左右は 2、斜めは 1 と数える）を選ぶ。同点は元の色に近い色、さらに同点は番号の小さい色。
    候補の色が近傍に無いときは、候補の中で元の色に最も近い色。

    輪郭の色は特別に扱う: シルエットの縁（上下左右に透明がある）のドットは、輪郭の色が候補にあれば輪郭の色にする
    （縁の輪郭が途切れないように）。内側のドットで上下左右に輪郭があるときは、ほかに候補があれば輪郭の色にしない
    （輪郭が 2 ドットの太さにならないように）。
    """
    nb = [int(v) for v in nb]
    dist, outline = ctx.distl[c], ctx.outline
    weight: dict[int, int] = {}
    for v, wt in zip(nb, _NB_WEIGHTS_L):
        if v >= 0:
            weight[v] = weight.get(v, 0) + wt
    cands = [(-wt, dist[o], o) for o, wt in sorted(weight.items())
             if o != c and (allowed is None or o in allowed)]
    if not cands:
        return min(allowed, key=lambda o: (dist[o], o))
    if outline is not None and any(o == outline for *_, o in cands):
        if any(v < 0 for v in nb[:4]):
            return outline
        if outline in nb[:4] and len(cands) > 1:
            cands = [t for t in cands if t[2] != outline]
    return min(cands)[2]


def _between_pairs(c: int, present, ctx: "_Ctx") -> set[int]:
    """`present` の 2 色で、色 c がその中間になる組に入る色の集合（無ければ空）。"""
    pres = sorted(o for o in present if o >= 0 and o != c)
    if len(pres) < 2:
        return set()
    row = ctx.between_row(c)
    out = set()
    for a in pres:
        ra = row[a]
        for b in pres:
            if ra[b]:
                out.add(a)
                out.add(b)
    return out


def _isolated(lab, kinds, ctx: "_Ctx") -> None:
    """(a) 孤立した 1 ドット（8 近傍に同じ色が無い）を、まわりの色に置き換える（`lab` を書き換える）。

    ラスタ順に 1 ドットずつ、直した結果を見ながら判定する（隣り合う 2 つの孤立ドットが色を入れ替え合わないように）。
    """
    n8 = _neighbors(lab, _N8)
    cand = (lab >= 0) & ~(n8 == lab[None]).any(0) & (n8 >= 0).any(0)
    p = _padded(lab).tolist()  # 直した結果を見ながら判定するための作業用（外側 1 ドットは透明）
    lv = ctx.level
    light = ctx.light
    for y, x in zip(*np.nonzero(cand)):
        y, x = int(y), int(x)
        c = p[y + 1][x + 1]
        nb = [p[y + 1 + dy][x + 1 + dx] for dy, dx in _N8]
        if c in nb or max(nb) < 0:
            continue
        side = [v for v in nb[:4] if v >= 0]
        if len(side) >= 3 and light[c] >= max(light[v] for v in side) + BRIGHT_DOT_MARGIN:
            continue  # 上下左右のどの色よりはっきり明るい 1 ドット（ハイライト）は、どの強さでも残す
        weight: dict[int, int] = {}
        for v, wt in zip(nb, _NB_WEIGHTS_L):
            if v >= 0:
                weight[v] = weight.get(v, 0) + wt
        present = set(weight)
        pair = _between_pairs(c, present, ctx)
        dc = ctx.distl[c]
        near = {o for o in present if dc[o] <= lv.near and weight[o] >= NEAR_MIN_WEIGHT}
        allowed = pair | near
        if not allowed:
            if not lv.distinct_isolated:
                continue  # まわりと離れた色の点（頬・ハイライト・1 ドットの目）は残す
            if light[c] <= min(light[o] for o in present) - DARK_DOT_MARGIN:
                continue  # まわりのどの色よりはっきり暗い 1 ドット（1 ドットの目）は strong でも残す
            allowed = None
        p[y + 1][x + 1] = _pick(c, nb, ctx, allowed)
        kinds[y, x] = KIND_ISOLATED
    lab[:] = np.array(p, np.int64)[1:-1, 1:-1]


def _between_runs(lab, kinds, ctx: "_Ctx") -> None:
    """(b) まわりの 2 色の中間の色の小さな塊を、まわりの色に置き換える（`lab` を書き換える）。

    対象は同じ色のつながり（斜めも含む）が `lv.run` ドット以下で、つながりのまわりにある色の 2 色の中間の色のもの。
    さらに、その色が画像全体でほとんど使われていない（不透明な部分の `lv.minor_share` 以下）こと。
    広く使われている色（手の影の色など）は、細い所で 2 色の中間に見えても、意味のある色として残す。
    `lv.thin_runs` のときは、幅 1 ドットの塊（2×2 のかたまりを含まない）だけを対象にする。
    各ドットは、その 2 色（中間になる組に入る色）のうち、そのドットのまわりに多い色にする。
    """
    lv = ctx.level
    comp, sizes = _components(lab)
    n8 = _neighbors(lab, _N8)
    used = np.bincount(lab[lab >= 0].astype(np.intp), minlength=len(ctx.dist))  # 32 ビットでは int64 を渡せない
    minor = used <= lv.minor_share * used.sum()
    small = np.flatnonzero(sizes <= lv.run)
    if len(small) == 0:
        return
    by_comp: dict[int, list] = {}
    for y, x in zip(*np.nonzero(np.isin(comp, small))):
        by_comp.setdefault(int(comp[y, x]), []).append((int(y), int(x)))
    out = lab.copy()
    for k in sorted(by_comp):
        pix = by_comp[k]
        c = int(lab[pix[0]])
        if not minor[c]:
            continue
        present = set()
        for y, x in pix:
            present |= {int(v) for v in n8[:, y, x] if v >= 0 and v != c}
        pair = _between_pairs(c, present, ctx)
        if not pair:
            continue
        if lv.thin_runs and _has_block(pix):
            continue
        if not _bracketed(lab, pix, c, ctx):
            continue  # 2 色の間に挟まれていない（線や面として並んだ影など）は残す
        for y, x in pix:
            out[y, x] = _pick(c, n8[:, y, x], ctx, pair)
            kinds[y, x] = KIND_BETWEEN
    lab[:] = out


# 塊を横切る向き（横・縦。斜めは数えない: 斜めの線の影は、斜めに見ると輪郭と白に挟まれて見えるため）
_AXES = ((0, 1), (1, 0))
# 塊のドットのうち、この割合より多くがぼかしの名残に見えれば、塊をぼかしの名残とみなす
BRACKET_SHARE = 0.5


def _bracketed(lab: np.ndarray, pix, c: int, ctx: "_Ctx") -> bool:
    """塊がぼかしの名残に見えるか（塊のドットの半分より多くが、次の両方を満たすか）。

    - 横か縦に塊を出た両側の色が、色 c をその中間にする 2 色（2 色に挟まれている）
    - 横にも縦にも、両側が「c を中間にしない、はっきり違う 2 色（距離 `BETWEEN_MIN_DIFF` 以上）」になっていない
      （2 つの面を分ける線になっていない。ほとんど同じ 2 色の赤に挟まれているのは、1 つの面の中とみなす）

    ぼかしの名残は 2 色の境目にあり、横か縦に見ると両側を 2 色に挟まれている。剣の影のように、白と薄い灰色の
    2 つの面の間に線として並んだ暗めの色は、色だけ見れば輪郭と白の中間で、線の端が輪郭に接していても、
    横から見ると「白 | 影 | 薄い灰色」と 2 つの面を分けているので、意味のある線として残す。
    """
    h, w = lab.shape
    s = set(pix)

    def exit_color(y, x, dy, dx):
        while (y, x) in s:
            y, x = y + dy, x + dx
        return int(lab[y, x]) if 0 <= y < h and 0 <= x < w else -1

    n = 0
    for y, x in pix:
        bracket = separator = False
        for dy, dx in _AXES:
            a, b = exit_color(y, x, dy, dx), exit_color(y, x, -dy, -dx)
            if a < 0 or b < 0 or a == b:
                continue
            if ctx.between[c, a, b]:
                bracket = True
            elif ctx.dist[a, b] >= BETWEEN_MIN_DIFF:
                separator = True
        n += bracket and not separator
    return n > BRACKET_SHARE * len(pix)


def _has_block(pix) -> bool:
    """ドットの集まりが 2×2 のかたまりを含むか（幅 2 ドット以上の所があるか）。"""
    s = set(pix)
    return any((y + 1, x) in s and (y, x + 1) in s and (y + 1, x + 1) in s for y, x in pix)


def _ring_mask(line: np.ndarray, y: int, x: int) -> int:
    h, w = line.shape
    m = 0
    for i, (dy, dx) in enumerate(_RING):
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and line[yy, xx]:
            m |= 1 << i
    return m


def _doubled(lab: np.ndarray, outline: int, y: int, x: int, dy: int, dx: int) -> bool:
    """(y, x) から見て、(y+dy, x+dx) が塗り、(y, x) と (y-dy, x-dx) が輪郭、(y-2dy, x-2dx) が輪郭以外（透明か塗り）か。
    つまり (y, x) が 2 ドットの太さの輪郭の内側のドットか。"""
    h, w = lab.shape

    def at(yy, xx):
        return lab[yy, xx] if 0 <= yy < h and 0 <= xx < w else -1

    f = at(y + dy, x + dx)
    return (f >= 0 and f != outline and at(y, x) == outline and at(y - dy, x - dx) == outline
            and at(y - 2 * dy, x - 2 * dx) != outline)


def _edge_connected(lab: np.ndarray, outline: int) -> np.ndarray:
    """輪郭の色のドットのうち、シルエットの縁（上下左右に透明か画像の外がある）のドットから、輪郭の色を 8 近傍で
    たどって行けるもの（シルエットの輪郭と、そこからつながる線）。

    塗りの面の中に浮いた輪郭の色の塊（まわりを別の色で囲まれた目・口など）は含まない。
    """
    line = lab == outline
    edge = line & (_neighbors(lab, _N4) < 0).any(0)
    h, w = lab.shape
    out = edge.copy()
    q = deque(zip(*np.nonzero(edge)))
    while q:
        y, x = q.popleft()
        for dy, dx in _N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and line[yy, xx] and not out[yy, xx]:
                out[yy, xx] = True
                q.append((yy, xx))
    return out


def _outline_fix(lab, kinds, outline: int, lv: CleanupLevel) -> np.ndarray:
    """(d) 輪郭の角の余分な 1 ドットと、(c) 2 ドットの太さの輪郭を、内側の塗りの色に置き換える。

    対象は輪郭の色のドットで、上下左右に透明が無く（シルエットは変えない）、上下左右のどこかが塗り（輪郭以外の不透明）のもの。
    置き換えても、まわり 8 ドットの輪郭のドットが 1 つにつながったまま（2 つ以上）のときだけ置き換える。
    ラスタ順に 1 ドットずつ、直した結果を見ながら判定する（同時に直すと輪郭が切れることがあるため）。
    """
    before = lab
    lab = lab.copy()
    h, w = lab.shape
    line = lab == outline
    on_edge = _edge_connected(lab, outline)

    def at(y, x):
        return lab[y, x] if 0 <= y < h and 0 <= x < w else -1

    for y, x in zip(*np.nonzero(line)):
        if not line[y, x] or not on_edge[y, x]:
            continue  # 塗りの中に浮いた輪郭の色の塊（目・口）は部品として変えない
        n4 = [at(y + dy, x + dx) for dy, dx in _N4]  # 上、左、右、下
        if min(n4) < 0:
            continue
        m = _ring_mask(line, y, x)
        if bin(m).count("1") < 2 or _RING_COMPONENTS[m] != 1:
            continue
        fills = [v for v in n4 if v != outline]
        if not fills:
            continue
        kind = None
        # (d) L 字の角: 上下のどちらか 1 つと左右のどちらか 1 つだけが輪郭で、残り 2 つが同じ塗りの色。
        #     直す前の状態で判定する（直した結果で判定すると、2 ドットの太さの線の端から次々に角になって削れていくため）
        b4 = [before[y + dy, x + dx] for dy, dx in _N4]
        is_line = [v == outline for v in b4]
        up, left, right, down = is_line
        if lv.corners and sum(is_line) == 2 and (up != down) and (left != right) and len(set(fills)) == 1:
            kind = KIND_CORNER
        # (c) 2 ドットの太さ: 塗り → このドット → 輪郭 → 輪郭以外（透明か塗り）と並ぶ方向がある
        #     となりのドット（線の向き）にも同じ並びがあるときだけ（2 本の線が出会う所の角を削らないように）
        elif lv.thin:
            for (dy, dx), v in zip(_N4, n4):
                if v != outline and _doubled(before, outline, y, x, dy, dx) and (
                        _doubled(before, outline, y + dx, x + dy, dy, dx)
                        or _doubled(before, outline, y - dx, x - dy, dy, dx)):
                    kind = KIND_THIN
                    break
        if kind is None:
            continue
        # 置き換え先: 上下左右の塗りの色の多数決（同点は番号の小さい色）
        lab[y, x] = _majority(np.array([v if v != outline else -1 for v in n4]), weights4=False)
        line[y, x] = False
        kinds[y, x] = kind
    return lab


# --- 本体 ----------------------------------------------------------------------------------

def cleanup_sprite(rgba: np.ndarray, level: str = DEFAULT_LEVEL) -> CleanupResult:
    """余計なドットを、まわりにある色に置き換える（新しい色は作らない）。`level` は `LEVELS` のキー。"""
    if level not in LEVELS:
        raise ValueError(f"クリーンアップの強さは {' / '.join(LEVELS)} のどれかで指定してください（{level}）")
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8:
        raise ValueError("RGBA の uint8 配列（高さ × 幅 × 4）が必要です")
    h, w = a.shape[:2]
    lv = LEVELS[level]
    lab0, cols = _to_labels(a)
    kinds = np.zeros((h, w), np.int64)
    if lv is None or len(cols) == 0:
        return CleanupResult(a.copy(), np.zeros((h, w), bool), kinds, None, level)
    lab3 = rgb_to_oklab(cols.astype(np.float64))
    dist = _distance(lab3, lab3)
    between = _between_table(lab3)
    outline = _outline_color(lab0, lab3)

    ctx = _Ctx(dist, between, outline, lv, lab3)
    lab = lab0.copy()
    for _ in range(MAX_PASSES):
        prev = lab.copy()
        if lv.alpha:
            _alpha_specks(lab, kinds)
        if lv.isolated:
            _isolated(lab, kinds, ctx)
        if lv.run > 0:
            _between_runs(lab, kinds, ctx)
        if np.array_equal(prev, lab):
            break
    if outline is not None and (lv.corners or lv.thin):
        lab = _outline_fix(lab, kinds, outline, lv)
        if lv.isolated:  # 輪郭を直した後に孤立したドットを片付ける（1 回だけ）
            _isolated(lab, kinds, ctx)

    out = a.copy()
    changed = lab != lab0
    op = lab >= 0
    out[..., :3][op] = cols[lab[op]]
    out[..., 3] = np.where(op, np.where(lab0 >= 0, a[..., 3], 255), 0)
    # 透明だったドットを埋めたときの色は上で入れた。透明にしたドットの RGB は 0 にそろえる
    out[..., :3][~op & (lab0 >= 0)] = 0
    kinds[~changed] = KIND_NONE
    return CleanupResult(out, changed, kinds, tuple(int(v) for v in cols[outline]) if outline is not None else None,
                         level)


@dataclass
class CleanupStep(Step):
    """クリーンアップの工程（余計なドットを片付ける）。直前の結果は `last`（`CleanupResult`）に残る。"""

    level: str = DEFAULT_LEVEL
    name: str = "cleanup"
    last: CleanupResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"クリーンアップの強さは {' / '.join(LEVELS)} のどれかで指定してください（{self.level}）")

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        self.last = cleanup_sprite(rgba, self.level)
        return self.last.image

    def describe(self) -> str:
        r = self.last
        head = f"クリーンアップ（{self.level}"
        if r is None:
            return head + "）"
        detail = "、".join(f"{k} {v}" for k, v in r.counts().items() if v)
        return f"{head}、{r.count} ドットを変更" + (f"：{detail}" if detail else "") + "）"
