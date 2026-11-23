"""背景除去（外周からの塗りつぶし）。

AI が描いたドット絵風の画像のように、背景がほぼ単色（少し色むら・ノイズがある）の画像から背景を透明にする。
元の大きさのまま、グリッド検出・縮小の前に行う（縮小後に消すと、背景に接する白い服などまで消えるため）。

手順:

1. 背景色を決める。指定が無ければ画像の外周の画素から推定する（`estimate_background()`）。
   外周の色を各チャンネル 16 刻みの区間でまとめ、最も多い区間の画素の平均色にする。
   キャラクターが外周に少しはみ出していても、外周の大部分を占める背景の色が選ばれる
2. 背景色との差（R・G・B の差の絶対値の最大値）が `tolerance` 以下の画素を「背景に近い色」とする
3. 背景に近い色のうち、**外周から上下左右につながっている部分だけ** を透明にする（塗りつぶし）。
   キャラクターの中にある背景と似た色（白い体・白い服）は、輪郭で外周から切り離されていれば残る。
   斜めにはつながらない（4 近傍）。1 画素幅の輪郭の斜めの継ぎ目から中へ漏れないようにするため
4. `remove_enclosed=True` のときは、外周につながっていない背景に近い色の部分（腕と体のすき間など、
   キャラクターに囲まれた背景）のうち、面積が `enclosed_min_area` 画素以上のものも透明にする。
   背景と同じ色の白い服・白い体も同じ扱いになるので既定ではオフ（小さな部分は目のハイライトなどとみなして残す）

結果は透明（RGBA すべて 0）か不透明（α=255）の 2 段階で、半透明は作らない（ドット絵のため）。
入力にすでに透過（α < 255 の画素）がある画像は、背景が抜かれているとみなしてそのまま返す。

塗りつぶしは、行・列ごとの「背景に近い色の連続区間（ラン）」の単位で、外周から届いた区間を広げることを
変化が無くなるまで繰り返す（NumPy でベクトル化。SciPy は使わない）。1254×1254 で 1 秒未満。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import Step

# 既定の許容量（背景色との R・G・B の差の最大値）。ai01〜ai03（preview/ai-check ブランチ）で 8〜96 を試して決めた。
# - 背景の色むら・ノイズは外周で背景色との差が最大 4〜6。これは 8 以上ならすべて消える
# - 輪郭と背景の境目の 1 画素の明るいにじみ（差 約 25）と、背景のうすいすじ（差 20〜40）は 32 でほぼ消える
# - キノコ（ai03）のかさの外側に接する、クリーム色のハイライトのドット（差 約 42）は 40 以上で消えてしまう
# - 白い服（ai01・ai02）は輪郭で外周から切り離されているため、96 でも輪郭の 3 画素より内側は消えない
# よって、にじみをできるだけ消しつつ、背景に接する明るい色まで 10 ほど余裕を残す 32 にした。
DEFAULT_TOLERANCE = 32
DEFAULT_ENCLOSED_MIN_AREA = 0.0005  # 1 未満なら画像の面積に対する割合、1 以上なら画素数
_ESTIMATE_BIN = 16


def _parse_color(color) -> np.ndarray:
    c = np.asarray(color, dtype=np.int64).reshape(-1)
    if c.shape != (3,) or (c < 0).any() or (c > 255).any():
        raise ValueError(f"背景色は 0〜255 の (R, G, B) で指定してください（{color}）")
    return c


def parse_color(text: str) -> tuple[int, int, int]:
    """`#RRGGBB`・`RRGGBB`・`R,G,B` の形式の文字列を (R, G, B) にする。"""
    s = text.strip()
    try:
        if "," in s:
            parts = [int(p) for p in s.split(",")]
        else:
            h = s[1:] if s.startswith("#") else s
            if len(h) != 6:
                raise ValueError
            parts = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
        return tuple(int(v) for v in _parse_color(parts))  # type: ignore[return-value]
    except ValueError:
        raise ValueError(f"背景色は #RRGGBB または R,G,B の形式で指定してください: {text}") from None


def _border(a: np.ndarray) -> np.ndarray:
    """外周の画素（重複なし）を (N, C) で返す。"""
    h, w = a.shape[:2]
    if h <= 2 or w <= 2:
        return a.reshape(-1, a.shape[-1])
    return np.concatenate([a[0], a[-1], a[1:-1, 0], a[1:-1, -1]])


def estimate_background(rgba: np.ndarray) -> tuple[int, int, int]:
    """外周の画素から背景色 (R, G, B) を推定する。

    外周の色を各チャンネル 16 刻みの区間でまとめ、最も多い区間（同数なら区間番号の小さい方）の画素の平均色。
    """
    b = _border(np.asarray(rgba)[..., :3]).astype(np.int64)
    q = b // _ESTIMATE_BIN
    key = (q[:, 0] * 16 + q[:, 1]) * 16 + q[:, 2]
    cnt = np.bincount(key.astype(np.intp), minlength=16 ** 3)  # 32 ビット（Pyodide）では int64 を渡せない
    best = int(np.argmax(cnt))
    mean = b[key == best].mean(0)
    return tuple(int(v) for v in np.rint(mean))  # type: ignore[return-value]


def _run_ids(cand: np.ndarray) -> np.ndarray:
    """行ごとの True の連続区間に 0 からの番号を付ける（False の画素の値は使わない）。"""
    h, w = cand.shape
    c = np.zeros((h, w + 1), bool)  # 行末に False を挟んで、行をまたぐ区間を作らない
    c[:, :w] = cand
    flat = c.ravel()
    start = flat.copy()
    start[1:] &= ~flat[:-1]
    ids = np.cumsum(start) - 1
    return ids.reshape(h, w + 1)[:, :w]


def _propagate(cand: np.ndarray, init: np.ndarray, reduce) -> np.ndarray:
    """True の画素の値 `init` を、横の区間・縦の区間ごとに `reduce`（np.minimum など）でまとめることを、
    変化が無くなるまで繰り返す。上下左右につながった部分の中で値がそろう。戻り値は True の画素の値（ラスター順）。
    """
    rid = _run_ids(cand)[cand]                 # ラスター順なので横の区間の番号は昇順
    cid = _run_ids(cand.T).T[cand]
    order = np.argsort(cid, kind="stable")     # 縦の区間の番号で並べ替える
    cs = cid[order]
    r_start = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    c_start = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    r_idx = np.repeat(np.arange(len(r_start)), np.diff(np.r_[r_start, len(rid)]))
    c_idx = np.repeat(np.arange(len(c_start)), np.diff(np.r_[c_start, len(cs)]))
    cur = init[cand]
    while True:
        nxt = reduce.reduceat(cur, r_start)[r_idx]
        tmp = nxt[order]
        tmp = reduce.reduceat(tmp, c_start)[c_idx]
        nxt = np.empty_like(tmp)
        nxt[order] = tmp
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt


def _label(cand: np.ndarray) -> np.ndarray:
    """True の画素を上下左右のつながりでまとめ、つながった部分ごとに同じ番号を付ける（False は -1）。"""
    h, w = cand.shape
    out = np.full((h, w), -1, np.int64)
    if cand.any():
        out[cand] = _propagate(cand, np.arange(h * w, dtype=np.int64).reshape(h, w), np.minimum)
    return out


def _flood_from_border(cand: np.ndarray) -> np.ndarray:
    """外周から上下左右につながっている True の画素（塗りつぶしの結果）。"""
    h, w = cand.shape
    seed = np.zeros((h, w), bool)
    seed[0], seed[-1], seed[:, 0], seed[:, -1] = True, True, True, True
    out = np.zeros((h, w), bool)
    if cand.any():
        out[cand] = _propagate(cand, seed & cand, np.logical_or)
    return out


def background_mask(rgba: np.ndarray, color=None, tolerance: float = DEFAULT_TOLERANCE, *,
                    remove_enclosed: bool = False,
                    enclosed_min_area: float = DEFAULT_ENCLOSED_MIN_AREA) -> np.ndarray:
    """背景として透明にする画素の真偽マスク（H×W）。

    - `color`: 背景色 (R, G, B)。None なら外周から推定する
    - `tolerance`: 背景色との R・G・B の差の最大値がこの値以下の画素を背景に近い色とする（0〜255）
    - `remove_enclosed`: 外周につながっていない背景に近い色の部分も、面積が `enclosed_min_area` 以上なら消す
    - `enclosed_min_area`: 1 未満なら画像の面積に対する割合、1 以上なら画素数
    """
    a = np.asarray(rgba)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"H×W×3 または H×W×4 の uint8 配列が必要です（{a.shape}, {a.dtype}）")
    if not 0 <= tolerance <= 255:
        raise ValueError(f"許容量は 0〜255 で指定してください（{tolerance}）")
    if enclosed_min_area < 0:
        raise ValueError(f"enclosed_min_area は 0 以上で指定してください（{enclosed_min_area}）")
    bg = _parse_color(color) if color is not None else np.array(estimate_background(a), np.int64)
    rgb = a[..., :3].astype(np.int16)
    diff = np.abs(rgb - bg.astype(np.int16)).max(-1)
    cand = diff <= tolerance
    mask = _flood_from_border(cand)
    if remove_enclosed:
        rest = cand & ~mask
        if rest.any():
            lab = _label(rest)
            ids, inv, cnt = np.unique(lab[rest], return_inverse=True, return_counts=True)
            h, w = cand.shape
            min_px = enclosed_min_area * h * w if enclosed_min_area < 1 else enclosed_min_area
            keep = cnt >= min_px
            mask[rest] = keep[inv.reshape(-1)]
    return mask


def remove_background(rgba: np.ndarray, color=None, tolerance: float = DEFAULT_TOLERANCE, *,
                      remove_enclosed: bool = False,
                      enclosed_min_area: float = DEFAULT_ENCLOSED_MIN_AREA) -> np.ndarray:
    """背景を透明にした RGBA 配列を返す（引数は `background_mask()` と同じ）。

    背景は RGBA すべて 0、それ以外は α=255。入力にすでに透過（α < 255）がある画像はそのまま（複製を）返す。
    """
    a = np.asarray(rgba)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"H×W×4 の uint8 配列が必要です（{a.shape}, {a.dtype}）")
    if has_transparency(a):
        return a.copy()
    mask = background_mask(a, color, tolerance, remove_enclosed=remove_enclosed,
                           enclosed_min_area=enclosed_min_area)
    out = a.copy()
    out[mask] = 0
    out[~mask, 3] = 255
    return out


def has_transparency(rgba: np.ndarray) -> bool:
    """α < 255 の画素があるか（透過 PNG なら True）。"""
    a = np.asarray(rgba)
    return a.ndim == 3 and a.shape[2] == 4 and bool((a[..., 3] < 255).any())


@dataclass
class BackgroundStep(Step):
    """背景除去の工程（`remove_background()`）。入力がすでに透過付きならそのまま通す。"""

    color: tuple[int, int, int] | None = None
    tolerance: float = DEFAULT_TOLERANCE
    remove_enclosed: bool = False
    enclosed_min_area: float = DEFAULT_ENCLOSED_MIN_AREA
    name: str = "background"

    def __post_init__(self):
        if self.color is not None:
            self.color = tuple(int(v) for v in _parse_color(self.color))  # type: ignore[assignment]
        if not 0 <= self.tolerance <= 255:
            raise ValueError(f"許容量は 0〜255 で指定してください（{self.tolerance}）")
        if self.enclosed_min_area < 0:
            raise ValueError(f"enclosed_min_area は 0 以上で指定してください（{self.enclosed_min_area}）")
        self._last = ""

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        if has_transparency(rgba):
            self._last = "入力が透過付きのためそのまま"
            return np.asarray(rgba).copy()
        color = self.color if self.color is not None else estimate_background(rgba)
        out = remove_background(rgba, color, self.tolerance, remove_enclosed=self.remove_enclosed,
                                enclosed_min_area=self.enclosed_min_area)
        ratio = float((out[..., 3] == 0).mean())
        how = "指定" if self.color is not None else "推定"
        self._last = (f"背景色 #{color[0]:02x}{color[1]:02x}{color[2]:02x}（{how}）、"
                      f"許容量 {self.tolerance:g}、透明 {ratio:.0%}")
        return out

    def describe(self) -> str:
        base = "背景除去" + ("（囲まれた背景も消す）" if self.remove_enclosed else "")
        return f"{base}: {self._last}" if self._last else base
