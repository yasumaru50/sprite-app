"""写真の下書き（Still App。`--mode photo`）。自分で撮った写真（部屋・建物）を、手直しの出発点になる「色を塗り分けた下絵」にする。

完成したドット絵は目指さない（Pixelorama などで塗り直す前の下絵）。工程（`photo_dots()`）:

1. 写真の向き（EXIF の Orientation）を見て正しく回す（`decode_photo()`。読み込みのとき）
2. 大きな写真は、まず仕上がりの `PRESCALE`（4）倍の大きさに縮める（`box_shrink()`。マスの画素の平均）。
   ブラウザ版（Pyodide）で、この後の処理を軽くするため
3. 色の塊をまとめる（`kuwahara()`）: 桑原フィルタ。各画素のまわりの 4 つの正方形（左上・右上・左下・右下。真ん中の画素を角に含む）
   のうち、明るさのばらつきがいちばん小さいものの平均の色にする。境目をまたぐ正方形はばらつきが大きいので選ばれず、
   境目は残ったまま、似た色の近所だけがならされる。強さ（弱・中・強、`SMOOTH`）は正方形の大きさと、くり返す回数
4. 仕上がりの大きさ（横 `width` マス。縦は縦横の比から）に縮める。マスの色はいちばん多い色（`dots.mode_colors()`。1 枚絵と同じ）

この後の減色・似た色をまとめる・孤立したドット・輪郭・余白は `convert.py`（今風モードと同じ減色）。

CLI とブラウザ版でバイト単位で同じ結果にするため、計算は整数だけで行う（exp などの浮動小数の関数は使わない）。
32 ビット（Pyodide）でも動くように `np.bincount`・`np.add.reduceat` などは使わず、`np.cumsum` で和を取る。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from . import dots as dots_mod

DEFAULT_WIDTH = 320     # 既定の横のマス数
PRESCALE = 4            # 色の塊をまとめる前に縮める大きさ（仕上がりの何倍か）
# まとめの強さ。radius は正方形の 1 辺 − 1（radius 2 なら 3×3）、passes はくり返す回数（縮めた後の画素で数える）
SMOOTH = {
    "weak": {"radius": 2, "passes": 1},
    "medium": {"radius": 4, "passes": 2},
    "strong": {"radius": 5, "passes": 3},
}
DEFAULT_SMOOTH = "medium"
SMOOTH_LABELS = {"weak": "弱", "medium": "中", "strong": "強"}
# 明るさ（ばらつきを測る値）の重み（ITU-R BT.601 を 256 倍した整数。合計 256）
GRAY_WEIGHTS = (77, 150, 29)

# HEIC・HEIF（iPhone などの写真の形式）は Pillow で読めないので、先頭のバイトで見分けて知らせる
_HEIF_BRANDS = (b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"avif")


def is_heif(data: bytes) -> bool:
    """HEIC・HEIF・AVIF のファイルか（先頭の ftyp の箱で見分ける）。"""
    head = bytes(data[:12])
    return len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS


def decode_photo(data: bytes) -> tuple[np.ndarray, int]:
    """写真のバイト列を、向きを直した RGB（uint8）にする。(配列, EXIF の Orientation の値（無ければ 1）)。

    透明のある画像は黒の上に重ねる（1 枚絵と同じ）。
    """
    with Image.open(io.BytesIO(bytes(data))) as img:
        img.load()
        try:
            orientation = int(img.getexif().get(0x0112, 1) or 1)
        except Exception:   # 壊れた EXIF は向きの情報が無いものとして扱う
            orientation = 1
        if orientation not in range(1, 9):
            orientation = 1
        img = ImageOps.exif_transpose(img) if orientation != 1 else img
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            return dots_mod.flatten_black(np.asarray(img.convert("RGBA"), np.uint8)), orientation
        return np.ascontiguousarray(np.asarray(img.convert("RGB"), np.uint8)), orientation


def _edges(n_from: int, n_to: int) -> np.ndarray:
    """n_from 画素を n_to マスに分ける境目（n_to + 1 個。整数の四捨五入で決める）。"""
    k = np.arange(n_to + 1, dtype=np.int64)
    return (k * (2 * n_from) + n_to) // (2 * n_to)


def _sum_cells(a: np.ndarray, edges: np.ndarray, axis: int) -> np.ndarray:
    """axis に沿って、境目 edges で区切った各マスの和（int64）。"""
    c = np.cumsum(a, axis=axis, dtype=np.int64)
    zero = np.zeros(tuple(c.shape[i] if i != axis else 1 for i in range(c.ndim)), np.int64)
    c = np.concatenate([zero, c], axis=axis)
    e = np.asarray(edges).astype(np.intp)   # np.take の番号は intp（Pyodide は 32 ビット）
    return np.take(c, e[1:], axis=axis) - np.take(c, e[:-1], axis=axis)


def box_shrink(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """RGB を width × height に縮める（各マスの画素の平均。整数で四捨五入）。拡大はしない。"""
    a = np.asarray(rgb, np.uint8)
    h, w = a.shape[:2]
    if width > w or height > h:
        raise ValueError(f"縮める先（{width}×{height}）が元（{w}×{h}）より大きくなっています")
    xs, ys = _edges(w, width), _edges(h, height)
    area = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.int64)
    out = np.empty((height, width, 3), np.uint8)
    for c in range(3):   # 1 色ずつ（大きな写真でも、一度に使うメモリを抑える）
        s = _sum_cells(_sum_cells(a[..., c], xs, 1), ys, 0)
        out[..., c] = ((2 * s + area) // (2 * area)).astype(np.uint8)
    return out


def _window_sums(p: np.ndarray, k: int) -> np.ndarray:
    """2 次元の配列 p の、k × k の正方形の和（int64）。out[i, j] = p[i:i+k, j:j+k].sum()。"""
    c = np.cumsum(p, axis=1, dtype=np.int64)
    c = np.concatenate([np.zeros((c.shape[0], 1), np.int64), c], axis=1)
    r = c[:, k:] - c[:, :-k]
    c = np.cumsum(r, axis=0, dtype=np.int64)
    c = np.concatenate([np.zeros((1, c.shape[1]), np.int64), c], axis=0)
    return c[k:] - c[:-k]


def kuwahara(rgb: np.ndarray, radius: int) -> np.ndarray:
    """桑原フィルタを 1 回かける（上の説明。境目を残して、似た色の近所をならす）。整数だけで計算する。"""
    a = np.asarray(rgb, np.uint8)
    h, w = a.shape[:2]
    r = int(radius)
    if r < 1:
        return a.copy()
    k = r + 1
    n = k * k
    pad = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    wr, wg, wb = GRAY_WEIGHTS
    gray = (pad[..., 0].astype(np.int64) * wr + pad[..., 1].astype(np.int64) * wg
            + pad[..., 2].astype(np.int64) * wb) >> 8
    s1 = _window_sums(gray, k)
    s2 = _window_sums(gray * gray, k)
    # 4 つの正方形（左上・右上・左下・右下）の、ばらつき × n²（n·Σg² − (Σg)²。整数のまま比べる）
    offs = ((0, 0), (0, r), (r, 0), (r, r))
    var = np.stack([n * s2[dy:dy + h, dx:dx + w] - s1[dy:dy + h, dx:dx + w] ** 2 for dy, dx in offs])
    pick = np.argmin(var, axis=0)   # 同じばらつきなら先の正方形
    out = np.empty_like(a)
    for c in range(3):
        sc = _window_sums(pad[..., c], k)
        cand = np.stack([sc[dy:dy + h, dx:dx + w] for dy, dx in offs])
        best = np.take_along_axis(cand, pick[None].astype(np.intp), axis=0)[0]
        out[..., c] = ((2 * best + n) // (2 * n)).astype(np.uint8)
    return out


def smooth(rgb: np.ndarray, level: str) -> np.ndarray:
    """強さ level（weak・medium・strong）で色の塊をまとめる。"""
    p = SMOOTH[level]
    out = np.asarray(rgb, np.uint8)
    for _ in range(p["passes"]):
        out = kuwahara(out, p["radius"])
    return out


@dataclass
class PhotoWork:
    """写真の下書きの途中の記録（JSON に書く）。"""

    work_size: tuple[int, int]    # 色の塊をまとめた大きさ（縮めた後）
    prescaled: bool               # 先に縮めたか
    level: str

    def describe(self, orientation: int) -> dict:
        p = SMOOTH[self.level]
        return {"orientation": orientation, "prescale": PRESCALE if self.prescaled else None,
                "work_size": list(self.work_size),
                "smooth": {"level": self.level, "method": "kuwahara", "radius": p["radius"], "passes": p["passes"]}}


def photo_dots(rgb: np.ndarray, width: int, level: str) -> tuple[dots_mod.Dots, PhotoWork]:
    """写真を、横 width マスのドットにする（先に縮める → 色の塊をまとめる → 仕上がりの大きさに縮める）。

    Dots の xs・ys は元の写真の座標（JSON の cell_size・area が元の写真の画素で出るように）。
    """
    a = np.asarray(rgb, np.uint8)
    h, w = a.shape[:2]
    height = dots_mod.height_for_width(width, w, h)
    if w >= PRESCALE * width and h >= PRESCALE * height:
        work = box_shrink(a, PRESCALE * width, PRESCALE * height)
        prescaled = True
    else:
        work = a
        prescaled = False
    work = smooth(work, level)
    wh, ww = work.shape[:2]
    small = dots_mod.mode_colors(work, _edges(ww, width), _edges(wh, height))
    xs, ys = _edges(w, width), _edges(h, height)
    return dots_mod.Dots(small, "photo", xs, ys), PhotoWork((int(ww), int(wh)), prescaled, level)
