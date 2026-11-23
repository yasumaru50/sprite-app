"""単純化の工程: 似た色をまとめて色数を減らす（グリッド検出・縮小の後に行う）。

ドット絵風の AI 画像をグリッド検出すると、ぼかしの名残で 1 枚に数百色が残る（ai03 で約 600 色）。
本来は 10〜20 色ほどの平塗りなので、似た色を 1 色にまとめる。ディザは使わず、どの画素も残した色のどれか 1 色にする。
透明な画素（α < 128）は色数に数えず、そのまま残す。

色の近さは **OKLab**（Björn Ottosson, 2020）のユークリッド距離で測る（色み a・b は 1.5 倍）。
- 人の目の明るさ・色みの感じ方にほぼ揃った色空間で、距離がそのまま「見た目の差」に近い
  （RGB では暗い色どうしの差が小さく出て、輪郭の暗い色と暗い塗りがまとまりやすい）
- CIELAB より色相のゆがみ（青→紫など）が小さく、変換は行列 2 回と立方根だけで NumPy で速い
- 距離の目安: 0.02 でほぼ見分けがつかない、0.05 で並べると分かる、0.1 以上ではっきり違う色
- 色みを 1.5 倍にする理由: 小さなスプライトでは、色みの違い（肌と頬の赤み）は別の部品を表し、
  ぼかしの名残は主に明るさの違いになる。ai03 で 1.0 / 1.5 / 2.0 を試し、1.0 では 12 色で頬の赤みが肌に吸収され、
  2.0 では 16 色でほとんど同じ赤が 2 色に分かれた

手順（`simplify_colors()`）:
1. **中間色の判定**（`transition_mask()`）: 不透明な画素ごとに、向かい合う隣（左右・上下・斜め 2 組）の 2 色を結ぶ
   線分の途中にある色（2 色の差が 0.08 以上、線分上の位置 15〜85%、線分からのずれが差の 3 割以下）を「中間色」とみなす。
   ぼかしの名残（塗りと輪郭・塗りと模様の境目のにじみ）がこれに当たる。輪郭（両隣より暗い）、目（周りと違う色）、
   頬の赤み（両隣が同じ肌色で差が小さい）は中間色にならない。中間色の画素は重みを 0.1 にする（数に 0.1 個と数える）
2. **ごく近い色をまとめる**（`merge_close_colors()` と同じ処理。色数の上限に関係なく行う）: 代表色どうしの距離が
   `merge_distance`（既定 0.03）未満の組を、いちばん近い組から順に 1 つにする（割引なし）。見分けのつかない色
   （ai05 の輪郭の紫 4 色、距離 0.003〜0.021）が、上限に余裕があるからといって何色も残らないようにする。
   これで `colors` 色以下になれば終わり（ai05 は 129 色 → 9 色）
3. **上限までまとめる**（凝集型のクラスタリング。今までの方法）: まだ `colors` 色より多ければ、最初（1 色 1 グループ）から
   「代表色どうしの距離 × 小ささの割引」が最も小さい 2 グループを 1 つにすることを繰り返す。
   - 代表色は、グループの中で「近くの色（距離 0.03 程度）の重みの合計」が最も大きい **実在の色**
     （そのグループの平塗りの中心の色）。平均しないので、何度まとめても代表色は濁らない
   - 小ささの割引は `sqrt(w / (w + 2))`（w は軽いほうのグループの重み）。中間色だけのグループ（重みがほぼ 0）は
     すぐに近いほうへまとまる。中間色でない画素が数個ある小さな色（目・頬・輪郭）は割引がほとんど効かず、
     距離が離れていれば最後まで残る。面積で重み付けする方法（k-means やメディアンカットなど）と違い、
     面積の小さい色が大きな面の色に吸収されない
   - 減らしていく途中の段階（`colors` 〜 2×`colors` グループ）のうち、2 と同じようにごく近い色をまとめた後に
     `colors` 以下になる、いちばんグループの多い段階を選び、ごく近い色をまとめる（まとめて空いた枠を別の色に使う）。
     ごく近い組が残らない画像（ai03・ai04）では、今までとまったく同じ結果になる
   - 2 でまとめたグループから続けて減らさないのは、まとめると代表色が少し変わり、上限ぎりぎりで残るかどうかの
     小さな色が入れ替わるため（ai03 の 12 色では、頬のピンク 3 ドットと、かさの明るい橙の差は 1% ほどしかなく、
     2 から続けると頬が消えた）
4. `merge_distance=0` なら 2 は何もせず、3 は `colors` まで減らした段階になる（以前の既定の動きと同じ）。
   既定 0.03 の理由: ai03・ai04・ai05 で 0〜0.045 を試した。見分けのつかない色の組は 0.021 以下（ai05 の紫・ミント・クリーム）、
   残したい色の差でいちばん近いのは 0.035（ai03 の 16 色の暗い赤 #b44b54 と赤 #c25359。かさの 2 段の赤は 0.049、
   頬と肌は 0.054、目の暗い色と輪郭は 0.048 以上）。0.025〜0.035 では 3 枚とも同じ結果で、その真ん中の 0.03 にした
   （0.02 では ai05 の紫が 2 色残り、0.04 では ai03 の 16 色で暗い赤がまとまる）。
   距離は色みを 1.5 倍にして測るので、0.03 は「明るさだけの差なら 0.03、色みだけの差なら（重みなしの OKLab で）0.02」に当たる
4b. **大きな面の色を守る**（`keep_large`、既定 0.01）: 3 で、どちらのグループも重みが全体の `keep_large` 以上
   （大きな面の色）で、代表色どうしの距離が `LARGE_DISTANCE`（0.06）以上ある組は、ほかにまとめられる組が無くなるまでまとめない
   （費用に `LARGE_PENALTY` を足して後回しにする。順番はその中で今まで通り）。
   理由: 小ささの割引は重みの絶対値（`SMALL_WEIGHT` = 2 ドット）で効くので、何コマも並べた絵（ai07〜ai09。1 万ドット近い）では、
   輪郭と赤の境目の暗い紫・上着のすその赤紫・輪郭と肌の混ざったくすんだ茶のような、50〜100 ドットの「境目の色」にも割引が効かず、
   色の差が大きいので最後まで残った。一方、淡い色どうし（肌 #fcf1ec と髪 #dcf0fc）は OKLab の距離が 0.064 と小さく
   （CIEDE2000 では 14 で、髪と髪の影の 13 より大きい）、12 色の枠からあふれて、肌・上着の薄い緑・白が髪の水色にまとめられ、
   キャラ全体が青っぽくなった（20 色にしても肌は髪にまとめられた）。
   色合いの重みを上げる・CIEDE2000 で測る案も試したが、ai03 の赤や ai04 の輪郭がほとんど同じ 2 色に分かれたり頬が消えたりした。
   この段は面積の大きい色だけに効くので、ai03・ai05 は 12 色・16 色とも画素単位で変わらず、ai04 の 12 色で肌の明るい所 6 ドットが
   中間の茶 22 ドットに入れ替わっただけだった（ai04 の 16 色と同じ組み合わせ）。1% と 0.06 の理由（2%・3%・5% と 0.03〜0.08 を試した）:
   2% では ai08 の髪の影（尻尾の青。全体の 1.4%）が守られず上着の緑にまとめられた。3% 以上では横向き（ai07・ai09）の上着の緑
   （全体の 2% ほど）も守られなかった。1% で距離 0.04 以下も守ると、ai03 の 12 色で 26 ドット変わった（近い色まで守るため）。
   0.08 では肌と髪（0.064）を守らない。0.06 は肌と髪・肌と頬（0.09）・上着と髪（0.093）を守り、肌と白（0.038）や
   同じ面の明るさ違い（0.03〜0.05）は守らない。0 にすると今までと同じ
5. **残す色**: 各グループの代表色と、それにごく近い（距離 0.03 以内の）同じグループの色の重み付き平均。
   元の画像にある色か、そのすぐ近くの色になる
6. **割り当て**: 不透明な画素をすべて、残した色の中で最も近い色にする（ディザなし）

乱数は使わず、同点は色の並び（RGB の値の小さい順）で決めるので、同じ入力からは必ず同じ結果になる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pipeline import Step

ALPHA_OPAQUE = 128
DEFAULT_COLORS = 16
# ごく近い色をまとめる距離の既定（OKLab。色み a・b は 1.5 倍で測る）。0 にするとまとめない（色数の上限だけで決める）
DEFAULT_MERGE_DISTANCE = 0.03
# 中間色の判定
TRANSITION_MIN_DIFF = 0.08   # 向かい合う 2 色の差がこれ以上のときだけ判定する
TRANSITION_T_RANGE = (0.15, 0.85)
TRANSITION_PERP = 0.3        # 線分からのずれ（2 色の差に対する割合）
TRANSITION_WEIGHT = 0.1      # 中間色の画素の重み
# 小ささの割引 sqrt(w / (w + SMALL_WEIGHT))
SMALL_WEIGHT = 2.0
# 残す色を決めるときの「ごく近い色」の範囲
CORE_RADIUS = 0.03
# 代表色を選ぶときの密度の幅（ガウス関数の標準偏差）
DENSITY_SIGMA = 0.03
# 色み（OKLab の a・b）の重み
CHROMA_WEIGHT = 1.5
# 大きな面の色を守る: 全体の重みに対するこの割合以上のグループどうしは、距離が LARGE_DISTANCE 以上なら最後までまとめない
DEFAULT_KEEP_LARGE = 0.01
LARGE_DISTANCE = 0.06
LARGE_PENALTY = 10.0          # 後回しにする組の費用に足す値（距離は 1 を超えないので、ほかのどの組より後になる）
# 色の種類がこれより多いときは、先に OKLab の細かい区間でまとめる（クラスタリングの計算量を抑える）
MAX_CLUSTER_INPUT = 512


# --- OKLab --------------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB（0〜255、…×3）を OKLab（L は 0〜1）にする。"""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.cbrt(lin @ m1.T)
    return lms @ m2.T


# --- 中間色の判定 --------------------------------------------------------------------------

_PAIRS = (((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, -1)))


def _shift(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """a[y + dy, x + dx] を (y, x) に置いた配列（外は fill）。"""
    h, w = a.shape[:2]
    out = np.full_like(a, fill)
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def transition_mask(rgba: np.ndarray) -> np.ndarray:
    """ぼかしの名残の中間色の画素（向かい合う隣の 2 色の線分の途中にある色）の真偽配列。"""
    a = np.asarray(rgba)
    opaque = a[..., 3] >= ALPHA_OPAQUE
    lab = rgb_to_oklab(a[..., :3])
    out = np.zeros(opaque.shape, bool)
    t0, t1 = TRANSITION_T_RANGE
    for (dy0, dx0), (dy1, dx1) in _PAIRS:
        pa, pb = _shift(lab, dy0, dx0, 0.0), _shift(lab, dy1, dx1, 0.0)
        ok = opaque & _shift(opaque, dy0, dx0, False) & _shift(opaque, dy1, dx1, False)
        v = pb - pa
        n2 = np.einsum("...k,...k->...", v, v)
        good = ok & (n2 >= TRANSITION_MIN_DIFF ** 2)
        t = np.einsum("...k,...k->...", lab - pa, v) / np.where(good, n2, 1.0)
        perp = lab - pa - t[..., None] * v
        p2 = np.einsum("...k,...k->...", perp, perp)
        out |= good & (t >= t0) & (t <= t1) & (p2 <= (TRANSITION_PERP ** 2) * n2)
    return out


# --- クラスタリング -------------------------------------------------------------------------

@dataclass
class SimplifyResult:
    """単純化の結果。"""

    image: np.ndarray            # 色をまとめた RGBA
    palette: np.ndarray          # 残した色（k×3, uint8。使われた画素の多い順）
    counts: np.ndarray           # 各色の画素数
    colors_before: int           # 元の色数（不透明な画素）
    transitions: int             # 中間色と判定した画素の数
    colors_after_close: int = 0  # ごく近い色をまとめた直後のグループの数（色が 512 より多い画像は、細かい区間でまとめた後の数え方）


def _unique_colors(rgb: np.ndarray, weights: np.ndarray):
    key = rgb.astype(np.int64) @ np.array([65536, 256, 1])
    uniq, inv = np.unique(key, return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    inv = inv.reshape(-1)
    return cols, inv, np.bincount(inv, weights=weights, minlength=len(uniq))


def _units(cols: np.ndarray, lab: np.ndarray, w: np.ndarray, limit: int):
    """クラスタリングの単位。色の種類が limit 以下なら 1 色 1 単位。超えるときは OKLab の細かい区間でまとめる。

    戻り値: (各色の単位番号, 単位の OKLab, 単位の RGB, 単位の重み)
    """
    n = len(lab)
    if n <= limit:
        return np.arange(n), lab, cols.astype(np.float64), w
    step = 0.004
    while True:
        q = np.floor(lab / step).astype(np.int64)
        _, grp = np.unique(q, axis=0, return_inverse=True)
        grp = grp.reshape(-1)
        m = int(grp.max()) + 1
        if m <= limit:
            break
        step *= 1.25
    ww = w + 1e-9
    gw = np.bincount(grp, weights=ww, minlength=m)
    glab = np.stack([np.bincount(grp, weights=lab[:, c] * ww, minlength=m) for c in range(3)], 1) / gw[:, None]
    grgb = np.stack([np.bincount(grp, weights=cols[:, c] * ww, minlength=m) for c in range(3)], 1) / gw[:, None]
    return grp, glab, grgb, np.bincount(grp, weights=w, minlength=m)


def _distance(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """色の距離の表（OKLab のユークリッド距離。色み a・b は CHROMA_WEIGHT 倍）。"""
    d = lab1[:, None, :] - lab2[None, :, :]
    d[..., 1:] *= CHROMA_WEIGHT
    return np.sqrt((d ** 2).sum(-1))


def _discount(w: np.ndarray) -> np.ndarray:
    """小ささの割引（重み 0 で 0、重みが大きいほど 1 に近づく）。"""
    return np.sqrt(w / (w + SMALL_WEIGHT))


def _density(dist: np.ndarray, w: np.ndarray) -> np.ndarray:
    """kern[i, j]: 色 j が色 i の密度（近くの色の重みの合計）に足す量。"""
    return np.exp(-0.5 * (dist / DENSITY_SIGMA) ** 2) * w[None, :]


def _merge_close(dist: np.ndarray, kern: np.ndarray, w: np.ndarray, threshold: float):
    """ごく近い色をまとめる（色数に関係なく、代表色どうしの距離が threshold 未満の組がなくなるまで）。

    いちばん近い組から順に 1 つにする。代表色はグループの中で密度が最大の実在の色
    （平均しないので、まとめても濁らず、鎖のように少しずつずれた色をつないでいくこともない）。
    重い（数の多い）ほうのグループに軽いほうを入れる。同点は番号の小さい組。
    戻り値: (各色の根の番号, 各根の代表の色番号。根でない所は不定)。threshold が 0 なら何もしない
    """
    m = len(dist)
    label = np.arange(m)
    rep = np.arange(m)
    if threshold <= 0 or m < 2:
        return label, rep
    gw = w.astype(np.float64).copy()
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    d[d >= threshold] = np.inf          # しきい値以上の組は最初から考えない
    members = [[i] for i in range(m)]
    dead = np.zeros(m, bool)
    rowarg = np.argmin(d, 1)            # 各行で最も近い相手（とその距離）を覚えておき、1 回ごとの計算を行 1 本分にする
    rowmin = d[np.arange(m), rowarg]
    while True:
        i = int(np.argmin(rowmin))      # 同点は番号の小さい組
        if not rowmin[i] < threshold:
            break
        j = int(rowarg[i])
        keep, drop = (i, j) if (gw[i], -i) >= (gw[j], -j) else (j, i)
        mem = members[keep] + members[drop]
        members[keep], members[drop] = mem, []
        gw[keep] += gw[drop]
        label[mem] = keep
        idx = np.array(mem)
        rep[keep] = idx[int(np.argmax(kern[np.ix_(idx, idx)].sum(1)))]
        dead[drop] = True
        d[drop, :] = np.inf
        d[:, drop] = np.inf
        rowmin[drop] = np.inf
        nd = dist[rep[keep]].copy()
        nd[(nd >= threshold) | dead] = np.inf
        nd[keep] = np.inf
        d[keep, :] = nd
        d[:, keep] = nd
        rowarg[keep] = int(np.argmin(nd))
        rowmin[keep] = nd[rowarg[keep]]
        # 最も近い相手が keep・drop だった行は計算し直す。ほかの行は、keep との新しい距離と比べるだけでよい
        stale = np.flatnonzero(~dead & ((rowarg == keep) | (rowarg == drop)))
        stale = stale[stale != keep]
        if len(stale):
            rowarg[stale] = np.argmin(d[stale], 1)
            rowmin[stale] = d[stale, rowarg[stale]]
        better = ~dead & (nd < rowmin)
        better[keep] = False
        rowarg[better] = keep
        rowmin[better] = nd[better]
    return label, rep


def _rep_of(members: np.ndarray, kern: np.ndarray) -> int:
    """グループの代表 = グループの中で密度（近くの色の重みの合計）が最大の色。平均しないので実在の色のまま。"""
    return int(members[int(np.argmax(kern[np.ix_(members, members)].sum(1)))])


def _cost(dist: np.ndarray, wmin: np.ndarray, large_weight: float) -> np.ndarray:
    """まとめる費用: 代表色どうしの距離 × 小ささの割引。大きな面の色どうしで差のある組は LARGE_PENALTY を足して後回し。"""
    c = dist * _discount(wmin)
    if large_weight > 0:
        c = np.where((wmin >= large_weight) & (dist >= LARGE_DISTANCE), c + LARGE_PENALTY, c)
    return c


def _agglomerate(dist: np.ndarray, kern: np.ndarray, w: np.ndarray, stop: int, keep_from: int,
                 large_weight: float = 0.0):
    """今までの方法で、1 色 1 グループから stop グループまでまとめる。

    「代表色どうしの距離 × 小ささの割引」が最小の組をまとめることを繰り返す（同点は番号の小さい組）。
    各行の最小（とその相手）を覚えておき、1 回ごとの計算を行と列 1 本分にしている（結果は表全体で最小を探すのと同じ）。
    戻り値: グループの数が keep_from 以下 stop 以上の各段階の {グループの数: (各色の根の番号, 各根の代表の色番号)}
    """
    m = len(dist)
    label = np.arange(m)
    rep = np.arange(m)
    snaps = {}
    if m <= keep_from:
        snaps[m] = (label.copy(), rep.copy())
    if m <= stop:
        return snaps
    gw = w.astype(np.float64).copy()
    members = [[i] for i in range(m)]
    dead = np.zeros(m, bool)
    cost = _cost(dist, np.minimum(gw[:, None], gw[None, :]), large_weight)
    np.fill_diagonal(cost, np.inf)
    rowarg = np.argmin(cost, 1)
    rowmin = cost[np.arange(m), rowarg]
    count = m
    while count > stop:
        i = int(np.argmin(rowmin))
        j = int(rowarg[i])
        keep, drop = (i, j) if (gw[i], -i) >= (gw[j], -j) else (j, i)
        mem = members[keep] + members[drop]
        members[keep], members[drop] = mem, []
        gw[keep] += gw[drop]
        dead[drop] = True
        label[mem] = keep
        rep[keep] = _rep_of(np.array(mem), kern)
        cost[drop, :] = np.inf
        cost[:, drop] = np.inf
        rowmin[drop] = np.inf
        nc = _cost(dist[rep[keep]], np.minimum(gw[keep], gw), large_weight)
        nc[dead] = np.inf
        nc[keep] = np.inf
        cost[keep, :] = nc
        cost[:, keep] = nc
        rowarg[keep] = int(np.argmin(nc))
        rowmin[keep] = nc[rowarg[keep]]
        # 最小の相手が keep・drop だった行は計算し直す。ほかの行は keep との新しい値と比べる（同点は番号の小さいほう）
        stale = np.flatnonzero(~dead & ((rowarg == keep) | (rowarg == drop)))
        stale = stale[stale != keep]
        if len(stale):
            rowarg[stale] = np.argmin(cost[stale], 1)
            rowmin[stale] = cost[stale, rowarg[stale]]
        better = ~dead & ((nc < rowmin) | ((nc == rowmin) & (keep < rowarg)))
        better[keep] = False
        rowarg[better] = keep
        rowmin[better] = nc[better]
        count -= 1
        if count <= keep_from:
            snaps[count] = (label.copy(), rep.copy())
    return snaps


def _merge_close_groups(label: np.ndarray, rep: np.ndarray, dist: np.ndarray, kern: np.ndarray, w: np.ndarray,
                        threshold: float):
    """グループ分け（label・rep）の代表色どうしで、ごく近い組をまとめる。代表色は色ごとの密度で選び直す。

    選び直した代表色どうしがまた近くなったときは、近い組がなくなるまで繰り返す。戻り値は (label, rep, グループの数)。
    """
    label, rep = label.copy(), rep.copy()
    while True:
        roots = np.flatnonzero(label == np.arange(len(label)))
        if threshold <= 0 or len(roots) < 2:
            return label, rep, len(roots)
        reps = rep[roots]
        gw = np.bincount(label, weights=w, minlength=len(label))[roots]
        sd = dist[np.ix_(reps, reps)]
        lab2, _ = _merge_close(sd, _density(sd, gw), gw, threshold)
        if np.array_equal(lab2, np.arange(len(roots))):
            return label, rep, len(roots)
        new_root = roots[lab2][np.searchsorted(roots, label)]   # 各色の新しい根
        label = new_root
        for r in np.unique(new_root):
            rep[r] = _rep_of(np.flatnonzero(label == r), kern)


def _cluster(lab: np.ndarray, w: np.ndarray, colors: int, merge_distance: float, keep_large: float = 0.0):
    """似た色をまとめる。(各単位のグループ番号 0..k-1, 各グループの代表の単位番号, ごく近い色をまとめた直後の数) を返す。

    1. ごく近い色をまとめる（`_merge_close()`。距離が merge_distance 未満の組。色数の上限に関係なく）。
       それで colors 以下になれば終わり
    2. まだ多ければ、今までの方法（`_agglomerate()`）で減らしていく。その途中の段階（colors 〜 2×colors グループ）のうち、
       ごく近い色をまとめた後に colors 以下になる、いちばんグループの多い段階を選び、ごく近い色をまとめる。
       今までの方法で colors まで減らした段階は必ず条件を満たす。ごく近い組が無ければ結果は今までとまったく同じ
       （ごく近い色をまとめて先に減らすと、各グループの代表色が少し変わり、小さな色（頬など）の残り方が変わることがあるため、
       上限までの減らし方は今まで通りにして、まとめて空いた枠の分だけ減らす前の段階を使う）
    merge_distance が 0 なら、今までの方法で colors まで減らした結果そのもの。
    keep_large（0 より大きいとき）: 重みが全体の keep_large 以上のグループどうしで差のある組は最後まで残す（`_cost()`）。
    """
    m = len(lab)
    w = w.astype(np.float64)
    dist = _distance(lab, lab)
    kern = _density(dist, w)
    n_close = m
    label = rep = None
    if merge_distance > 0:
        l1, r1 = _merge_close(dist, kern, w, merge_distance)
        label, rep, n_close = _merge_close_groups(l1, r1, dist, kern, w, merge_distance)
        if n_close > colors:
            label = None
    if label is None:
        large_weight = keep_large * float(w.sum()) if keep_large > 0 else 0.0
        snaps = _agglomerate(dist, kern, w, colors, 2 * colors if merge_distance > 0 else colors, large_weight)
        for c in sorted(snaps, reverse=True):
            l2, r2, k = _merge_close_groups(*snaps[c], dist, kern, w, merge_distance)
            if k <= colors:
                label, rep = l2, r2
                break
    roots = np.flatnonzero(label == np.arange(m))
    remap = np.full(m, -1)
    remap[roots] = np.arange(len(roots))
    return remap[label], rep[roots], n_close


@dataclass
class CloseMergeResult:
    """`merge_close_colors()` の結果。"""

    labels: np.ndarray           # 入力の各色が入ったグループの番号（0..k-1）
    keep: np.ndarray             # 各グループで残した色の、入力での番号（グループの中で最も密な実在の色）
    colors: np.ndarray           # 残した色（k×3, uint8。入力の色そのまま）

    def mapped(self, colors: np.ndarray) -> np.ndarray:
        """入力の各色を、まとめた先の色にした配列（入力と同じ並び）。"""
        return self.colors[self.labels]


def merge_close_colors(colors: np.ndarray, weights: np.ndarray | None = None,
                       threshold: float = DEFAULT_MERGE_DISTANCE) -> CloseMergeResult:
    """ごく近い色どうし（OKLab の距離。色み a・b は 1.5 倍）が threshold 未満なら 1 色にまとめる（色数の上限は考えない）。

    単純化の最初の段階と同じ処理を、色の一覧（と各色の重み＝ドット数など）に対して行う。パレット統合などほかの工程から使う。
    - いちばん近い組から順にまとめ、代表どうしの距離が threshold 未満の組がなくなったら止める
    - 残す色は、グループの中で「近くの色の重みの合計」が最大の **入力にある色**（平均しないので濁らない）
    - threshold が 0 なら何もしない（1 色 1 グループ）。乱数なし。同点は入力の並びの早いほう
    """
    cols = np.asarray(colors)
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError("色は n×3 の配列で指定してください")
    if threshold < 0:
        raise ValueError(f"まとめる距離は 0 以上で指定してください（{threshold}）")
    n = len(cols)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    if w.shape != (n,):
        raise ValueError("重みは色と同じ数だけ指定してください")
    if n == 0:
        return CloseMergeResult(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), np.uint8))
    lab = rgb_to_oklab(cols.astype(np.float64))
    dist = _distance(lab, lab)
    label, rep = _merge_close(dist, _density(dist, w), w, threshold)
    roots = np.flatnonzero(label == np.arange(n))
    remap = np.full(n, -1)
    remap[roots] = np.arange(len(roots))
    keep = rep[roots]
    return CloseMergeResult(remap[label], keep, np.clip(np.round(cols[keep]), 0, 255).astype(np.uint8))


def simplify_colors(rgba: np.ndarray, colors: int = DEFAULT_COLORS,
                    merge_distance: float = DEFAULT_MERGE_DISTANCE,
                    *, detect_transitions: bool = True, keep_large: float = DEFAULT_KEEP_LARGE) -> SimplifyResult:
    """似た色をまとめて、不透明な部分を `colors` 色以下にする（透明な画素はそのまま）。

    `merge_distance`（OKLab の距離。色み a・b は 1.5 倍）より近い色どうしは、色数の上限に関係なく 1 色にまとめる
    （先にまとめ、それでも `colors` より多ければ上限までまとめる）。0 にすると色数の上限だけで決める（以前の既定の動き）。
    大きくすると（0.06 など）色数ではなく近さで決められる（`colors` は上限になる）。
    `keep_large`: 面積（重み）が全体のこの割合以上の色どうしは、距離 `LARGE_DISTANCE` 以上の差があれば最後までまとめない
    （肌・上着などの大きな面が髪の色に吸われないように）。0 にすると以前の動き。
    """
    if colors < 1:
        raise ValueError(f"色数は 1 以上で指定してください（{colors}）")
    if merge_distance < 0:
        raise ValueError(f"まとめる距離は 0 以上で指定してください（{merge_distance}）")
    if not 0 <= keep_large <= 1:
        raise ValueError(f"大きな面の割合は 0〜1 で指定してください（{keep_large}）")
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8:
        raise ValueError("RGBA の uint8 配列（高さ × 幅 × 4）が必要です")
    out = a.copy()
    opaque = a[..., 3] >= ALPHA_OPAQUE
    if not opaque.any():
        return SimplifyResult(out, np.zeros((0, 3), np.uint8), np.zeros(0, np.int64), 0, 0)
    trans = transition_mask(a) if detect_transitions else np.zeros(opaque.shape, bool)
    rgb = a[..., :3][opaque]
    pw = np.where(trans[opaque], TRANSITION_WEIGHT, 1.0)
    cols, inv, w = _unique_colors(rgb, pw)
    lab = rgb_to_oklab(cols.astype(np.float64))
    _, ulab, urgb, uw = _units(cols, lab, w, MAX_CLUSTER_INPUT)
    grp, reps, n_close = _cluster(ulab, uw, colors, merge_distance, keep_large)
    # 残す色: 代表の色と、それにごく近い（CORE_RADIUS 以内の）同じグループの色の重み付き平均
    pal = []
    for g, r in enumerate(reps):
        sel = (grp == g) & (_distance(ulab[r:r + 1], ulab)[0] <= CORE_RADIUS)
        ww = uw[sel] + 1e-9
        pal.append((urgb[sel] * ww[:, None]).sum(0) / ww.sum())
    pal = np.clip(np.round(np.array(pal)), 0, 255).astype(np.uint8)
    k = len(pal)
    # 割り当て: 不透明な画素を、残した色の中で最も近い色にする
    idx = np.argmin(_distance(lab, rgb_to_oklab(pal.astype(np.float64))), axis=1)
    pix = idx[inv]
    used = np.bincount(pix, minlength=k)
    # 使われた色だけを、画素の多い順（同点は RGB の小さい順）に並べる
    keyp = pal.astype(np.int64) @ np.array([65536, 256, 1])
    order = [g for g in np.lexsort((keyp, -used)) if used[g] > 0]
    remap = np.full(k, -1)
    remap[order] = np.arange(len(order))
    pal, used = pal[order], used[order]
    newrgb = pal[remap[pix]]
    o = out[..., :3]
    o[opaque] = newrgb
    return SimplifyResult(out, pal, used, len(cols), int(trans.sum()), n_close)


@dataclass
class SimplifyStep(Step):
    """単純化の工程（似た色をまとめる）。直前の結果は `last`（`SimplifyResult`）に残る。"""

    colors: int = DEFAULT_COLORS
    merge_distance: float = DEFAULT_MERGE_DISTANCE
    keep_large: float = DEFAULT_KEEP_LARGE
    name: str = "simplify"
    last: SimplifyResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.colors < 1:
            raise ValueError(f"色数は 1 以上で指定してください（{self.colors}）")
        if self.merge_distance < 0:
            raise ValueError(f"まとめる距離は 0 以上で指定してください（{self.merge_distance}）")
        if not 0 <= self.keep_large <= 1:
            raise ValueError(f"大きな面の割合は 0〜1 で指定してください（{self.keep_large}）")

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        self.last = simplify_colors(rgba, self.colors, self.merge_distance, keep_large=self.keep_large)
        return self.last.image

    def describe(self) -> str:
        r = self.last
        if r is None:
            return "単純化"
        dist = f"、距離 {self.merge_distance:g} 未満はまとめる" if self.merge_distance > 0 else ""
        return f"単純化（{r.colors_before} 色 → {len(r.palette)} 色［上限 {self.colors}{dist}］、中間色 {r.transitions} 画素）"
