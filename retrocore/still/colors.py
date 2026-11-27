"""1 枚絵の減色（Still App）。ドットに戻した絵を、決まった色数のパレットにそろえる。ディザは使わない。

手順（`reduce_colors()`）:

1. **4 ビットの丸めは、選んだ色にだけ行う**（手順 4）。ドットの色は丸めずに色を選び、置き換える。
   計画の案（先に全部のドットを丸める）では、平らな面の色がちょうど丸めの境目にあると（ai15 の壁は G が 127〜129 で、
   16 段階の 7.5 段のあたり）、ドットごとに上下の段に分かれて 2 色になり、まだらになったため
2. **いちばん暗い色を取っておく**（`keep_darkest=True` のとき）: 絵の中でいちばん暗い色をパレットの 1 色にする。
   残りの色は、その色に近いドットを除いた絵から選ぶ（色数 − 1 色）。中間の色に吸われて暗い輪郭の線が消えないようにするため。
   「いちばん暗い色」は、色を 4 ビット（各チャンネル 16 段階）の組に分け、全体の `DARK_MIN_SHARE`（1%）以上のドットがある
   組のうちいちばん暗い組（OKLab の明るさ L が最小）の、いちばん多い元の色。ほんの数ドットの黒（ノイズ）に 1 色を
   使わないようにするため（ai15 では、いちばん暗い組は 132 ドット＝0.84% で、代表の (8,8,8) は丸めると #000000 になるが、
   見た目では線の #111111 に近く、ほとんどが #111111 に置き換わって #000000 は 17 ドットしか使われなかった。
   0.2% ではこの組が選ばれ、1% では線の #111111 の組が選ばれる。1 枚絵の輪郭の線は、ふつう全体の 1% より多い）。
   その組のドットと、組の色より暗いドットを除いて選ぶ（除いたドットは、手順 5 でいちばん近い色＝ふつうは取っておいた色になる）
3. **色を選ぶ**: Sprite App の単純化（`retrocore.sprite.simplify.simplify_colors()`）で似た色をまとめる。面積の小さい色
   （目・小物）が大きな面の色に吸われにくく、乱数を使わない。「大きな面の色を守る」割合だけ `KEEP_LARGE`（0.5%）にする
4. **選んだ色を丸める**（`four_bit=True` のとき）: 4096 色にする（`snap_4bit()`。各チャンネルの上下の段のうち、
   明るさの感じ方で近い方。ただの四捨五入では暗い色が見た目と食い違うため）。
   丸めで色が重なったら、少ない色数のままにする（`merged` に数を残し、警告にする）
5. **置き換え**: 全部のドットを、パレットの中でいちばん近い色（OKLab の距離。単純化と同じ測り方）にする。
   新しい色は作らない

パレットの並びは暗い順（OKLab の L の小さい順。同じなら RGB の値の小さい順）。0 番がいちばん暗い色になる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sprite.simplify import _distance, _srgb_to_linear, rgb_to_oklab, simplify_colors

MIN_COLORS = 2
MAX_COLORS = 256
# 単純化の「大きな面の色を守る」割合（Sprite App の既定は 1%）。1 枚絵はドットが多く、1% では
# ai16 の赤いポスト（100 ドット＝0.7%）が屋根の茶色にまとめられたため、0.5% にする
KEEP_LARGE = 0.005
DARK_MIN_SHARE = 0.01    # いちばん暗い色とみなす組に要るドットの割合（全体の 1%）


@dataclass
class Reduced:
    """減色の結果。"""

    index: np.ndarray        # 高さ × 幅（uint8）。パレットの番号
    palette: np.ndarray      # k × 3（uint8）。暗い順。使った色だけ
    counts: np.ndarray       # 各色のドット数
    darkest: np.ndarray | None   # 取っておいたいちばん暗い色（丸めた後）。取っておかないときは None
    chosen: int              # 色を選んだ直後の色数（4 ビットに丸める前。取っておいた色を含む）
    merged: int              # 選んだ色を 4 ビットに丸めて重なり、減った色の数


def round_4bit(rgb: np.ndarray) -> np.ndarray:
    """各チャンネルを 0〜15 に丸めて ×17 に戻す（4096 色のどれかにする）。"""
    a = np.asarray(rgb)
    return (np.rint(a.astype(np.float64) / 17.0) * 17).astype(np.uint8)


def to_4bit(rgb: np.ndarray) -> np.ndarray:
    """0〜255 の値を 0〜15 にする（丸める）。"""
    return np.rint(np.asarray(rgb, np.float64) / 17.0).astype(np.int64)


def _perceived(v: np.ndarray) -> np.ndarray:
    """0〜255 の値の、明るさの感じ方（線形の値の立方根。OKLab の明るさと同じ考え方）。"""
    return np.cbrt(_srgb_to_linear(np.asarray(v, np.float64) / 255.0))


def snap_4bit(cols: np.ndarray) -> np.ndarray:
    """選んだ色を 4096 色にする。各チャンネルの上下の段のうち、明るさの感じ方（`_perceived()`）で近い方を選ぶ。

    ただの四捨五入では暗い色で見た目と食い違う（(8,8,8) は #000000 になるが、見た目では #111111 に近く、
    ドットを置き換える OKLab の「いちばん近い色」では #111111 の方へ行ってしまう）。チャンネルごとに決めるので、
    灰色は灰色のまま（3 つの値が同じなら同じ段になる）。同じ近さなら下の段。
    """
    c = np.asarray(cols, np.float64).reshape(-1, 3)
    lo = np.floor(c / 17.0) * 17
    hi = np.minimum(lo + 17, 255)
    pc = _perceived(c)
    pick_hi = np.abs(_perceived(hi) - pc) < np.abs(pc - _perceived(lo))
    return np.where(pick_hi, hi, lo).astype(np.uint8)


def _keys(cols: np.ndarray) -> np.ndarray:
    c = np.asarray(cols).astype(np.int64)
    return (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]


def dark_order(cols: np.ndarray) -> np.ndarray:
    """色を暗い順に並べる番号（OKLab の L の小さい順、同じなら RGB の値の小さい順）。"""
    c = np.asarray(cols)
    if len(c) == 0:
        return np.zeros(0, np.int64)
    lum = np.round(rgb_to_oklab(c.astype(np.float64))[:, 0], 9)
    return np.lexsort((_keys(c), lum))


def _bucket(rgb: np.ndarray) -> np.ndarray:
    q = to_4bit(rgb)
    return (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]


def darkest_color(rgb: np.ndarray, min_share: float = DARK_MIN_SHARE) -> tuple[np.ndarray, int]:
    """絵の中でいちばん暗い色と、その 4 ビットの組の番号（`reduce_colors()` の手順 2）。"""
    a = np.asarray(rgb).reshape(-1, 3)
    b = _bucket(a)
    ub, cnt = np.unique(b, return_counts=True)
    ok = cnt >= max(1.0, min_share * len(a))
    if not ok.any():
        ok = cnt == cnt.max()
    ub = ub[ok]
    rep = np.stack([(ub >> 8) & 15, (ub >> 4) & 15, ub & 15], axis=1) * 17
    dark_b = int(ub[dark_order(rep)[0]])
    cols, cc = np.unique(a[b == dark_b], axis=0, return_counts=True)
    order = np.lexsort((_keys(cols), -cc))
    return cols[order[0]].astype(np.uint8), dark_b


def nearest_index(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """各画素の、パレットの中でいちばん近い色の番号（OKLab の距離。同じ距離なら番号の小さい方）。"""
    a = np.asarray(rgb).reshape(-1, 3)
    key = _keys(a)
    uniq, inv = np.unique(key, return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.float64)
    d = _distance(rgb_to_oklab(cols), rgb_to_oklab(np.asarray(palette, np.float64)))
    near = np.argmin(d, axis=1)
    return near[inv.reshape(-1)].reshape(np.asarray(rgb).shape[:2])


def _unique_keep_order(cols: np.ndarray) -> np.ndarray:
    _, first = np.unique(_keys(cols), return_index=True)
    return cols[np.sort(first)]


def reduce_colors(rgb: np.ndarray, colors: int = 16, *, four_bit: bool = True, keep_darkest: bool = True) -> Reduced:
    """ドットの絵（高さ × 幅 × 3）を colors 色以下のパレットにそろえる。"""
    colors = int(colors)
    if not MIN_COLORS <= colors <= MAX_COLORS:
        raise ValueError(f"色数は {MIN_COLORS}〜{MAX_COLORS} で指定してください（{colors}）")
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("RGB の uint8 配列（高さ × 幅 × 3）が必要です")
    img = a.copy()
    h, w = img.shape[:2]
    rgba = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    dark = None
    rest = colors
    if keep_darkest:
        raw, dark_b = darkest_color(img)
        # 取っておく色の組と、それより暗いドット（数が少なくて組に選ばれなかったもの）は、色を選ぶ材料から除く
        dark_l = rgb_to_oklab(np.array([[(dark_b >> 8) & 15, (dark_b >> 4) & 15, dark_b & 15]], np.float64) * 17)[0, 0]
        lum = rgb_to_oklab(img.reshape(-1, 3).astype(np.float64))[:, 0].reshape(h, w)
        rgba[(_bucket(img) == dark_b) | (lum <= dark_l), 3] = 0
        dark = snap_4bit(raw)[0] if four_bit else raw
        rest = colors - 1
    pal = simplify_colors(rgba, rest, keep_large=KEEP_LARGE).palette if (rgba[..., 3] > 0).any() else np.zeros((0, 3), np.uint8)
    if dark is not None:
        pal = np.concatenate([dark[None, :], pal], axis=0)
    chosen = len(pal)
    if four_bit:
        pal = snap_4bit(pal)
    pal = _unique_keep_order(pal)
    merged = chosen - len(pal)
    pal = pal[dark_order(pal)]
    idx = nearest_index(img, pal)
    # 使った色だけを残す（暗い順のまま番号を詰める）
    used, counts = np.unique(idx.reshape(-1), return_counts=True)
    remap = np.full(len(pal), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Reduced(remap[idx].astype(np.uint8), pal[used].astype(np.uint8), counts.astype(np.int64),
                   dark, chosen, merged)
