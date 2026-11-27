"""1 枚絵をドットに戻す（Still App）。

AI が描いた「ドット絵風の 1 枚絵」（背景も絵の一部）を、1 ドット＝1 画素の絵にする。

- 透明な所は黒で埋める（`flatten_black()`。半透明は黒の上に重ねた色）。背景も絵の一部として扱うので、Sprite App の
  背景除去・切り詰めはしない
- マスの区切り方は 2 通り:
  - グリッド検出（既定）: Sprite App と同じ `retrocore.sprite.grid.detect_grid()` で線を求める（正方形のドット）。
    画像の端で半ドットより細いマスは、絵の外のはみ出しとして落とす
  - 横のマス数（`--width N`）: 画像を横 N マスに等分する。縦のマス数は縦横の比から（四捨五入）
- マスの色は平均ではなく **いちばん多い色**（`mode_colors()`）。平均すると輪郭と塗りの境目のマスが濁った中間の色になるため。
  AI 画像は同じ色でも 1〜2 ずつ揺れるので、まず各チャンネルを 16 段階（4 ビット）にまとめた組で数えていちばん多い組を選び、
  その組の中でいちばん多い元の色をマスの色にする。縁はぼけているので、縁から 2 割より内側の画素だけで数える
  （3 画素より小さいマスは全部の画素）。同点は色の値の小さい方。乱数は使わない
- グリッドが見つからないときや大きさの上限を超えるときに添える `--width` の候補は `width_candidates()`・`fit_width()`

32 ビット（Pyodide）でも動くように、`np.bincount` などは使わず `np.unique` と `np.lexsort` で数える。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sprite import grid as grid_mod

INNER_MARGIN = 0.2        # マスの縁から内側とみなすまでの幅（マスの大きさに対する割合。grid.py と同じ）
EDGE_KEEP = 0.5           # 画像の端のマスは、ドットの大きさのこの割合以上の幅（高さ）があるときだけ残す
CANDIDATE_DOTS = (6, 8, 10, 12)   # グリッドが見つからないときに添える、1 ドットの大きさ（px）の例
MIN_CANDIDATE_WIDTH = 16


class StillGridError(ValueError):
    """グリッドが見つからないとき。`candidates` に --width の候補（[{"width", "note"}...]）を持つ。"""

    def __init__(self, message: str, candidates: list[dict]):
        super().__init__(message)
        self.candidates = candidates


@dataclass
class Dots:
    """ドットに戻した結果。"""

    rgb: np.ndarray                    # 高さ × 幅 × 3（uint8）。1 ドット＝1 画素
    method: str                        # "grid"（グリッド検出）か "width"（--width）
    xs: np.ndarray                     # 使ったマスの境目（入力画像の座標。xs[i]〜xs[i+1] が i 列目）
    ys: np.ndarray
    dot_size: float | None = None      # グリッド検出で求めた 1 ドットの大きさ（px）
    trimmed: tuple[int, int, int, int] = (0, 0, 0, 0)   # 端で落としたマスの数（左, 上, 右, 下）

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])

    def describe(self) -> dict:
        d = {"method": self.method, "cells": list(self.size)}
        if self.method == "grid":
            d["dot_size"] = round(float(self.dot_size), 3)
            gx, gy = np.diff(self.xs), np.diff(self.ys)
            d["gap_range"] = {"x": [int(gx.min()), int(gx.max())], "y": [int(gy.min()), int(gy.max())]}
            d["trimmed_edges"] = dict(zip(("left", "top", "right", "bottom"), (int(t) for t in self.trimmed)))
        else:
            d["width"] = self.size[0]
            d["cell_size"] = [round((float(self.xs[-1]) - float(self.xs[0])) / self.size[0], 3),
                              round((float(self.ys[-1]) - float(self.ys[0])) / self.size[1], 3)]
        d["area"] = [int(self.xs[0]), int(self.ys[0]), int(self.xs[-1]), int(self.ys[-1])]
        return d


def flatten_black(rgba: np.ndarray) -> np.ndarray:
    """RGBA を、黒の上に重ねた RGB（uint8）にする。不透明な画素はそのまま、透明な画素は黒。"""
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] not in (3, 4) or a.dtype != np.uint8:
        raise ValueError("RGB か RGBA の uint8 配列（高さ × 幅 × 3 か 4）が必要です")
    if a.shape[2] == 3:
        return a.copy()
    alpha = a[..., 3:4].astype(np.uint32)
    return ((a[..., :3].astype(np.uint32) * alpha + 127) // 255).astype(np.uint8)


def _cell_index(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """画素ごとのマスの番号（範囲外は -1）と、マスの中の内側か。"""
    pos = np.arange(n)
    idx = np.searchsorted(edges, pos, side="right") - 1
    ncell = len(edges) - 1
    valid = (idx >= 0) & (idx < ncell) & (pos < edges[-1])
    idx = np.where(valid, idx, -1)
    size = np.diff(edges)
    margin = np.where(size > 2, np.rint(size * INNER_MARGIN), 0).astype(np.int64)
    safe = np.clip(idx, 0, ncell - 1)
    off = pos - edges[safe]
    inner = valid & (off >= margin[safe]) & (off < size[safe] - margin[safe])
    return idx, inner


def _pick_most(cell: np.ndarray, value: np.ndarray, ncell: int, scale: int) -> np.ndarray:
    """マスごとに、いちばん多い value（同点は小さい方）。cell・value は int64、0 ≤ value < scale。"""
    key = cell * scale + value
    uniq, cnt = np.unique(key, return_counts=True)
    ucell, uval = uniq // scale, uniq % scale
    order = np.lexsort((uval, -cnt, ucell))
    first = np.ones(len(order), bool)
    first[1:] = ucell[order][1:] != ucell[order][:-1]
    best = np.full(ncell, -1, np.int64)
    best[ucell[order][first]] = uval[order][first]
    return best


def mode_colors(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """境目 xs・ys で区切った各マスを、いちばん多い色にした絵（高さ len(ys)-1 × 幅 len(xs)-1 × 3）。"""
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    xs = np.asarray(xs, np.int64)
    ys = np.asarray(ys, np.int64)
    nx, ny = len(xs) - 1, len(ys) - 1
    cx, ix = _cell_index(xs, w)
    cy, iy = _cell_index(ys, h)
    inside = (cy[:, None] >= 0) & (cx[None, :] >= 0)
    inner = iy[:, None] & ix[None, :]
    cell = (cy[:, None].astype(np.int64) * nx + cx[None, :].astype(np.int64))
    # 内側の画素が無いマス（無いはずだが念のため）は、マスの全部の画素で数える
    has_inner = np.zeros(nx * ny, bool)
    has_inner[np.unique(cell[inner])] = True
    use = inside & (inner | ~has_inner[np.where(inside, cell, 0)])
    cells = cell[use]
    pix = a[use].astype(np.int64)
    color = (pix[:, 0] << 16) | (pix[:, 1] << 8) | pix[:, 2]
    q = np.rint(pix / 17.0).astype(np.int64)
    bucket = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    best_bucket = _pick_most(cells, bucket, nx * ny, 4096)
    sel = bucket == best_bucket[cells]
    best = _pick_most(cells[sel], color[sel], nx * ny, 1 << 24)
    out = np.stack([(best >> 16) & 255, (best >> 8) & 255, best & 255], axis=-1).astype(np.uint8)
    return out.reshape(ny, nx, 3)


def _trim_edges(lines: np.ndarray, period: float) -> tuple[np.ndarray, int, int]:
    """端の細いマス（period × EDGE_KEEP 未満）を落とす。(線, 落とした数（始め側）, 落とした数（終わり側）)。"""
    lo = hi = 0
    if len(lines) > 2 and lines[1] - lines[0] < period * EDGE_KEEP:
        lines, lo = lines[1:], 1
    if len(lines) > 2 and lines[-1] - lines[-2] < period * EDGE_KEEP:
        lines, hi = lines[:-1], 1
    return lines, lo, hi


def grid_dots(rgb: np.ndarray, max_size: tuple[int, int] | None = None) -> Dots:
    """Sprite App と同じグリッド検出でドットに戻す。見つからなければ `StillGridError`（--width の候補つき）。"""
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    rgba = np.concatenate([a, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    try:
        g = grid_mod.detect_grid(rgba)
    except grid_mod.GridNotFoundError as e:
        raise StillGridError(_grid_message(str(e)), width_candidates(rgba, max_size)) from e
    period = float(g.period[0])
    xs, l0, l1 = _trim_edges(np.asarray(g.xs, np.int64), period)
    ys, t0, t1 = _trim_edges(np.asarray(g.ys, np.int64), period)
    return Dots(mode_colors(a, xs, ys), "grid", xs, ys, period, (l0, t0, l1, t1))


def _grid_message(detail: str) -> str:
    detail = detail.split("）。", 1)[0] + "）" if "）。" in detail else detail
    return ("ドットのマス目が見つかりませんでした。--width で横のマス数を指定すると、マス目を探さずに変換できます。"
            f"\n（くわしい理由: {detail}）")


def width_dots(rgb: np.ndarray, width: int) -> Dots:
    """画像を横 width マスに等分してドットに戻す（縦のマス数は縦横の比から四捨五入）。"""
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    width = int(width)
    if width < 1:
        raise ValueError(f"横のマス数は 1 以上で指定してください（{width}）")
    if width > w:
        raise ValueError(f"横のマス数（{width}）が、画像の幅（{w}px）より多くなっています")
    height = height_for_width(width, w, h)
    xs = np.rint(np.arange(width + 1) * (w / width)).astype(np.int64)
    ys = np.rint(np.arange(height + 1) * (h / height)).astype(np.int64)
    return Dots(mode_colors(a, xs, ys), "width", xs, ys)


def height_for_width(width: int, w: int, h: int) -> int:
    """横 width マスのときの縦のマス数（縦横の比から四捨五入。1〜h）。"""
    return int(min(h, max(1, int(np.floor(width * h / w + 0.5)))))


def fit_width(w: int, h: int, max_w: int, max_h: int) -> int:
    """縦横の比を保って max_w × max_h に収まる、いちばん大きい横のマス数（0 なら収まらない）。"""
    n = min(int(max_w), int(w))
    while n > 0 and height_for_width(n, w, h) > max_h:
        n -= 1
    return n


def size_candidates(w: int, h: int, max_w: int, max_h: int) -> list[dict]:
    """上限に収まる --width の候補（いちばん大きいもの・その 3/4・半分）。"""
    n = fit_width(w, h, max_w, max_h)
    out = []
    for k, note in ((1.0, "上限に収まるいちばん大きい数"), (0.75, "その 3/4"), (0.5, "その半分")):
        v = int(np.floor(n * k + 0.5))
        if v >= 1 and all(c["width"] != v for c in out):
            out.append({"width": v, "height": height_for_width(v, w, h), "note": note})
    return out


def width_candidates(rgba: np.ndarray, max_size: tuple[int, int] | None = None) -> list[dict]:
    """グリッドが見つからないときの --width の候補。

    縦・横の境目から別々に推定した 1 ドットの大きさ（目立ち方が足りなくて使えなかったもの。片方がもう片方のほぼ整数倍なら
    小さい方だけ）を先に、
    次に 1 ドットを `CANDIDATE_DOTS` px とした場合を並べる。上限（max_size）に収まらないもの・小さすぎるものは除く。
    """
    a = np.asarray(rgba)
    h, w = a.shape[:2]
    op = a[..., 3] >= grid_mod.ALPHA_OPAQUE
    dots = []
    for axis, name in ((1, "横"), (0, "縦")):
        prof = grid_mod.edge_profile(a, axis)
        s, e = grid_mod._content_range(op, axis)
        p, _ = grid_mod.estimate_period(prof, s, e)
        if p > 0:
            dots.append((p, f"{name}の境目から推定した 1 ドット 約 {p:.1f}px"))
    # もう片方の推定のほぼ整数倍（2 倍以上）の推定は、何ドット分かを 1 ドットと誤ったものとみなして除く
    if len(dots) == 2:
        small, large = sorted(dots, key=lambda t: t[0])
        r = large[0] / small[0]
        if r >= 1.8 and abs(r - round(r)) <= 0.25:
            dots = [small]
    dots += [(float(d), f"1 ドットを {d}px とした場合") for d in CANDIDATE_DOTS]
    out = []
    for p, note in dots:
        n = int(np.floor(w / p + 0.5))
        if n < MIN_CANDIDATE_WIDTH or n > w or any(c["width"] == n for c in out):
            continue
        hh = height_for_width(n, w, h)
        if max_size is not None and (n > max_size[0] or hh > max_size[1]):
            continue
        out.append({"width": n, "height": hh, "note": note})
    return out
