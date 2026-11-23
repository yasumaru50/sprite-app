"""縮小の工程。

Pillow の再標本化（`RESIZE_METHODS`: nearest / box / bilinear / lanczos）と、
ドット絵向けの縮小（`PIXEL_METHODS`: mode / median。`downscale.py`）から選ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .downscale import (DEFAULT_BIN_SIZE, DEFAULT_LINE_LUMA, DEFAULT_LINE_RATIO,
                        DEFAULT_TRANSPARENT_RATIO, PIXEL_METHODS, pixel_downscale)
from .pipeline import Step

# 名前 → Pillow の再標本化フィルタ
RESIZE_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
}

# 選べる縮小方法すべて（Pillow の再標本化 → ドット絵向け）
ALL_RESIZE_METHODS = (*RESIZE_METHODS, *PIXEL_METHODS)


def resolve_size(src_w: int, src_h: int, width: int | None = None, height: int | None = None,
                 scale: float | None = None) -> tuple[int, int]:
    """出力サイズを決める。

    - `scale` を指定: 元のサイズ × scale（四捨五入、最小 1）
    - `width` と `height` の両方: そのサイズ（縦横比は保たない）
    - 片方だけ: 縦横比を保ってもう片方を決める
    """
    if scale is not None:
        if width is not None or height is not None:
            raise ValueError("倍率と幅・高さは同時に指定できません")
        if not scale > 0:
            raise ValueError(f"倍率は正の数で指定してください（{scale}）")
        return max(1, round(src_w * scale)), max(1, round(src_h * scale))
    if width is None and height is None:
        raise ValueError("幅・高さ・倍率のいずれかを指定してください")
    for label, v in (("幅", width), ("高さ", height)):
        if v is not None and v < 1:
            raise ValueError(f"{label}は 1 以上で指定してください（{v}）")
    if width is None:
        width = max(1, round(src_w * height / src_h))
    elif height is None:
        height = max(1, round(src_h * width / src_w))
    return int(width), int(height)


@dataclass
class ResizeStep(Step):
    """指定サイズ（幅・高さ、または倍率）に縮小する。

    Pillow の方法では透過をそのまま保つ。mode / median（ドット絵向け）では `bin_size` 以降の引数を使い、
    透過は `transparent_ratio` で 0 か 255 に決める（`downscale.pixel_downscale()` を参照）。
    """

    width: int | None = None
    height: int | None = None
    scale: float | None = None
    method: str = "nearest"
    bin_size: int = DEFAULT_BIN_SIZE
    line_priority: bool = False
    line_ratio: float = DEFAULT_LINE_RATIO
    line_luma: float = DEFAULT_LINE_LUMA
    transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO
    name: str = "resize"

    def __post_init__(self):
        if self.method not in ALL_RESIZE_METHODS:
            raise ValueError(f"縮小方法が不明です: {self.method}（{', '.join(ALL_RESIZE_METHODS)}）")
        if self.scale is None and self.width is None and self.height is None:
            raise ValueError("幅・高さ・倍率のいずれかを指定してください")

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        h, w = rgba.shape[:2]
        size = resolve_size(w, h, self.width, self.height, self.scale)
        if size == (w, h):
            return rgba.copy()
        if self.method in PIXEL_METHODS:
            return pixel_downscale(rgba, size, self.method, bin_size=self.bin_size,
                                   line_priority=self.line_priority, line_ratio=self.line_ratio,
                                   line_luma=self.line_luma, transparent_ratio=self.transparent_ratio)
        # Pillow は RGBA の再標本化で α を考慮する（透明部分の色がにじまない）
        img = Image.fromarray(rgba, "RGBA").resize(size, RESIZE_METHODS[self.method])
        return np.asarray(img, dtype=np.uint8).copy()

    def describe(self) -> str:
        if self.scale is not None:
            target = f"×{self.scale:g}"
        else:
            target = f"{self.width or '自動'}×{self.height or '自動'}"
        method = self.method
        if self.line_priority and method in PIXEL_METHODS:
            method += "＋線優先"
        return f"縮小 {target}（{method}）"
