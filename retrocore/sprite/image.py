"""RGBA 配列の読み書き。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_rgba(img: Image.Image | np.ndarray) -> np.ndarray:
    """Pillow 画像または配列を RGBA の uint8 配列（H×W×4）にする。

    透過付きの画像は透過をそのまま保ち、透過の無い画像は不透明（α=255）にする。
    """
    if isinstance(img, Image.Image):
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        return np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError(f"uint8 の配列が必要です（{a.dtype}）")
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"H×W×3 または H×W×4 の配列が必要です（{a.shape}）")
    if a.shape[2] == 3:
        a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    return np.ascontiguousarray(a).copy()


def load_rgba(path: str | Path) -> np.ndarray:
    """画像ファイルを RGBA 配列として読み込む。"""
    with Image.open(path) as img:
        img.load()
        return to_rgba(img)


def save_rgba(rgba: np.ndarray, path: str | Path) -> None:
    """RGBA 配列を PNG で保存する（親フォルダが無ければ作る）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgba(rgba), "RGBA").save(path, format="PNG")


def scale_nearest(rgba: np.ndarray, factor: int) -> np.ndarray:
    """ニアレストネイバーで整数倍に拡大する（1 画素を factor×factor の正方形にする。確認用のプレビュー向け）。"""
    if int(factor) != factor or factor < 1:
        raise ValueError(f"倍率は 1 以上の整数で指定してください（{factor}）")
    a = to_rgba(rgba)
    factor = int(factor)
    if factor == 1:
        return a
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)
