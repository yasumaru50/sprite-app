"""ドット絵向けの縮小（ブロック内の最頻色・中央値）。

出力の 1 画素は入力の長方形ブロック 1 つに対応する。平均して新しい色を作る Pillow の box などと違い、
**ブロック内に実在する色だけ** を採用するので、縮小後も色数が増えず、輪郭や塗りの色がにじまない。

- `mode`: ブロック内で最も多い色。AI 画像は同じ色がほぼ繰り返さないため、各チャンネルを `bin_size` 幅の
  区間でまとめてから数える。採用するのは、勝った区間の中でその区間の平均色に最も近い実在の画素の色
- `median`: ブロック内の各チャンネルの中央値に最も近い、実在する画素の色
- 線優先（`line_priority`）: ブロック内に暗い画素（輝度 ≤ `line_luma`）が `line_ratio` 以上あれば、
  暗い画素だけを対象に上の方法で色を選ぶ。細い輪郭線が縮小で途切れないようにするため。
  透過より優先する（透過の背景に接する輪郭も残す）
- 透過: ブロック内の透過画素（α < 128）が `transparent_ratio` 以上なら透過（RGBA すべて 0）。
  それ以外は不透明画素だけから色を選び、α は 255 にする

ブロックの境界は `round(i × 入力幅 / 出力幅)`。倍率が整数でなくてもブロックの大きさの差は 1 画素以内で、
境界と理想位置のずれは 0.5 画素以内（片側に大きいブロックが寄らない）。
同点は決まった規則（ブロック平均に近い区間 → 区間番号の小さい方、距離が同じ画素は色の値が小さい方）で
解くので、同じ入力からは必ず同じ結果になり、ブロック内の画素の並び（左右反転など）にも左右されない。すべて NumPy でベクトル化している。
"""

from __future__ import annotations

import numpy as np

from ..color import LUMA

PIXEL_METHODS = ("mode", "median")

# 既定値（CLI と ResizeStep でも使う）
DEFAULT_BIN_SIZE = 32
DEFAULT_LINE_RATIO = 0.25
DEFAULT_LINE_LUMA = 64.0
DEFAULT_TRANSPARENT_RATIO = 0.5
ALPHA_OPAQUE = 128  # これ以上の α を不透明とみなす


def block_edges(src: int, dst: int) -> np.ndarray:
    """入力 `src` 画素を `dst` 個のブロックに分ける境界（長さ dst+1、先頭 0・末尾 src）。

    境界は i × src / dst を四捨五入した位置。拡大（dst > src）では空のブロックができるが、
    実際の処理（`_block_ranges`）では各ブロックに少なくとも 1 画素を割り当てる（最近傍と同じになる）。
    """
    if src < 1 or dst < 1:
        raise ValueError(f"サイズは 1 以上が必要です（{src} → {dst}）")
    i = np.arange(dst + 1, dtype=np.int64)
    return (2 * i * src + dst) // (2 * dst)  # round(i*src/dst)（0.5 は切り上げ）を整数演算で


def _block_ranges(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    e = block_edges(src, dst)
    start = np.minimum(e[:-1], src - 1)
    stop = np.maximum(e[1:], start + 1)
    return start, stop


def _gather(rgba: np.ndarray, size: tuple[int, int]):
    """ブロックごとの画素を (ブロック数, 最大画素数, 4) に並べ、有効な位置のマスクを返す。"""
    h, w = rgba.shape[:2]
    out_w, out_h = size
    xs0, xs1 = _block_ranges(w, out_w)
    ys0, ys1 = _block_ranges(h, out_h)
    bw = int((xs1 - xs0).max())
    bh = int((ys1 - ys0).max())
    kx = np.arange(bw)
    ky = np.arange(bh)
    xi = xs0[:, None] + kx[None, :]                 # (out_w, bw)
    yi = ys0[:, None] + ky[None, :]                 # (out_h, bh)
    xv = xi < xs1[:, None]
    yv = yi < ys1[:, None]
    xi = np.minimum(xi, w - 1)
    yi = np.minimum(yi, h - 1)
    # (out_h, bh, out_w, bw, 4) → (out_h, out_w, bh, bw, 4)
    px = rgba[yi[:, :, None, None], xi[None, None, :, :]].transpose(0, 2, 1, 3, 4)
    valid = (yv[:, None, :, None] & xv[None, :, None, :])  # (out_h, out_w, bh, bw)
    n = bh * bw
    return px.reshape(out_h * out_w, n, 4), valid.reshape(out_h * out_w, n)


def _packed(rgb: np.ndarray) -> np.ndarray:
    """RGB を 24 ビットの整数にする（同点の解決に使う）。"""
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _pick_mode(rgb: np.ndarray, use: np.ndarray, bin_size: int) -> np.ndarray:
    """各ブロックの最頻区間から、実在する画素の位置（ブロック内の添字）を選ぶ。"""
    nb, n = use.shape
    pick = np.zeros(nb, np.int64)
    if not use.any():
        return pick
    levels = 256 // bin_size + 1
    q = rgb // bin_size
    key = (q[..., 0] * levels + q[..., 1]) * levels + q[..., 2]          # (nb, n)
    nkeys = levels ** 3
    bi, pi = np.nonzero(use)                                             # ラスター順（ブロック → 画素）
    c = rgb[bi, pi].astype(np.float64)
    gk = bi.astype(np.int64) * nkeys + key[bi, pi]
    uniq, inv, cnt = np.unique(gk, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    ub = uniq // nkeys
    # 区間の平均色、ブロックの平均色（同点の解決に使う）
    bin_mean = np.stack([np.bincount(inv, c[:, k], len(uniq)) for k in range(3)], -1) / cnt[:, None]
    bcnt = np.bincount(bi, minlength=nb).astype(np.float64)
    blk_mean = np.stack([np.bincount(bi, c[:, k], nb) for k in range(3)], -1) / np.maximum(bcnt, 1)[:, None]
    tie = ((bin_mean - blk_mean[ub]) ** 2).sum(-1)
    order = np.lexsort((uniq, tie, -cnt, ub))                            # ブロック → 多い → 平均に近い → 番号
    first = order[np.r_[True, ub[order][1:] != ub[order][:-1]]]
    win = np.full(nb, -1, np.int64)
    win[ub[first]] = first
    # 勝った区間の画素のうち、区間の平均色に最も近いもの（同じなら色の値が小さいほう）
    member = inv == win[bi]
    mb, mp, mi = bi[member], pi[member], inv[member]
    d = ((c[member] - bin_mean[mi]) ** 2).sum(-1)
    order = np.lexsort((mp, _packed(rgb[mb, mp]), d, mb))
    first = order[np.r_[True, mb[order][1:] != mb[order][:-1]]]
    pick[mb[first]] = mp[first]
    return pick


def _pick_median(rgb: np.ndarray, use: np.ndarray) -> np.ndarray:
    """各ブロックで、チャンネルごとの中央値に最も近い実在の画素の位置を選ぶ。"""
    nb, n = use.shape
    cnt = use.sum(1)
    big = np.int32(1 << 20)
    v = np.where(use[..., None], rgb.astype(np.int32), big)
    s = np.sort(v, axis=1)                                               # 使わない画素は末尾へ
    lo = np.maximum(cnt - 1, 0) // 2
    hi = np.maximum(cnt, 1) // 2
    hi = np.where(cnt % 2 == 0, hi, lo)
    r = np.arange(nb)
    med = (s[r, lo].astype(np.float64) + s[r, hi]) / 2.0                 # (nb, 3)
    # 中央値は 0.5 刻みなので、距離の 4 倍は整数。距離が同じなら色の値が小さいほう
    d4 = np.rint(((rgb.astype(np.float64) - med[:, None, :]) ** 2).sum(-1) * 4).astype(np.int64)
    score = (d4 << 24) | _packed(rgb)
    score[~use] = np.iinfo(np.int64).max
    return np.argmin(score, axis=1)


def pixel_downscale(rgba: np.ndarray, size: tuple[int, int], method: str = "mode", *,
                    bin_size: int = DEFAULT_BIN_SIZE,
                    line_priority: bool = False,
                    line_ratio: float = DEFAULT_LINE_RATIO,
                    line_luma: float = DEFAULT_LINE_LUMA,
                    transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO) -> np.ndarray:
    """RGBA 配列（H×W×4, uint8）を `size`=(幅, 高さ) に縮小する。結果の色はすべて入力に実在する色。

    - `method`: "mode"（最頻色）または "median"（中央値に最も近い色）
    - `bin_size`: mode で色をまとめる区間の幅（1〜256。1 なら完全に同じ色だけを数える。大きいほど粗い）
    - `line_priority`: 暗い画素が `line_ratio`（0〜1）以上のブロックは暗い画素から色を選ぶ
    - `line_luma`: 暗い画素とみなす輝度（0〜255）の上限
    - `transparent_ratio`: 透過画素がこの割合（0 より大きく 1 以下）以上のブロックを透過にする
    """
    if method not in PIXEL_METHODS:
        raise ValueError(f"縮小方法が不明です: {method}（{', '.join(PIXEL_METHODS)}）")
    if not 1 <= int(bin_size) <= 256:
        raise ValueError(f"bin_size は 1〜256 で指定してください（{bin_size}）")
    if not 0 < line_ratio <= 1:
        raise ValueError(f"line_ratio は 0 より大きく 1 以下で指定してください（{line_ratio}）")
    if not 0 < transparent_ratio <= 1:
        raise ValueError(f"transparent_ratio は 0 より大きく 1 以下で指定してください（{transparent_ratio}）")
    a = np.asarray(rgba)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"H×W×4 の uint8 配列が必要です（{a.shape}, {a.dtype}）")
    out_w, out_h = int(size[0]), int(size[1])

    px, valid = _gather(a, (out_w, out_h))
    rgb = px[..., :3].astype(np.int64)
    nvalid = valid.sum(1)
    opaque = valid & (px[..., 3] >= ALPHA_OPAQUE)
    transparent = (nvalid - opaque.sum(1)) >= transparent_ratio * nvalid - 1e-9
    use = opaque
    lined = np.zeros(len(valid), bool)
    if line_priority:
        dark = opaque & ((px[..., :3].astype(np.float32) @ LUMA) <= line_luma)
        lined = dark.sum(1) >= line_ratio * nvalid - 1e-9
        lined &= dark.any(1)
        use = np.where(lined[:, None], dark, opaque)
    show = lined | ~transparent                                          # 線は透過より優先
    use &= show[:, None]

    if method == "mode":
        pick = _pick_mode(rgb, use, int(bin_size))
    else:
        pick = _pick_median(rgb, use)
    out = np.zeros((len(valid), 4), np.uint8)
    chosen = px[np.arange(len(valid)), pick]
    out[show, :3] = chosen[show, :3]
    out[show, 3] = 255
    return out.reshape(out_h, out_w, 4)
