"""Sprite App 用の処理（NumPy と Pillow のみ）。

画像は RGBA の uint8 配列（高さ × 幅 × 4）で受け渡す。工程（`Step`）を `SpritePipeline` に並べて順に実行する。
既存の Retro98 の変換（`retrocore.pipeline`）とは独立しており、既存の関数の挙動は変えない。

フェーズ 1 で足していく予定の工程（docs/sprite-app-plan.md）:
背景除去 → グリッド検出（ドット絵風の入力）または縮小 → 単純化 → クリーンアップ → 縁取り → パレット統合。
現在は背景除去（`BackgroundStep`。`background.py`）、グリッド検出（`GridStep`。`grid.py`。ドット絵風の入力）と
縮小（`ResizeStep`。普通のイラスト。ドット絵向けの最頻色・中央値は `downscale.py`）、
単純化（`SimplifyStep`。`simplify.py`。似た色をまとめる）、クリーンアップ（`CleanupStep`。`cleanup.py`。
余計なドットをまわりの色に置き換える）、縁取り（`OutlineStep`。`outline.py`。シルエットの縁に暗い輪郭を付ける・
輪郭の切れ目を埋める）、パレット統合の第一歩（`PaletteStep`。`palette.py`。パレットファイルの読み書きと、
指定したパレットに合わせる）。

画像と設定を受け取って結果を返す入口は `convert_sprite()`（`convert.py`。CLI とブラウザ UI で共通）。
"""

from .cleanup import CleanupLevel, CleanupResult, CleanupStep, cleanup_sprite
from .background import (BackgroundStep, background_mask, estimate_background, parse_color,
                         remove_background)
from .grid import (Grid, GridNotFoundError, GridResult, GridStep, count_colors, detect_grid,
                   duplicate_lines, grid_to_pixels)
from .image import load_rgba, save_rgba, scale_nearest, to_rgba
from .pipeline import SpritePipeline, Step, StepResult
from .downscale import PIXEL_METHODS, block_edges, pixel_downscale
from .resize import ALL_RESIZE_METHODS, RESIZE_METHODS, ResizeStep, resolve_size
from .outline import OutlineDetection, OutlineResult, OutlineStep, detect_outline, outline_sprite
from .palette import (Collision, FarColor, PaletteError, PaletteMatch, PaletteStep, load_palette, match_palette,
                      save_palette, sort_palette, used_colors)
from .convert import SpriteResult, SpriteSettings, convert_sprite
from .simplify import (CloseMergeResult, SimplifyResult, SimplifyStep, merge_close_colors, rgb_to_oklab,
                       simplify_colors, transition_mask)

__all__ = [
    "ALL_RESIZE_METHODS",
    "BackgroundStep",
    "CleanupLevel",
    "CleanupResult",
    "CleanupStep",
    "Grid",
    "GridNotFoundError",
    "GridResult",
    "GridStep",
    "OutlineDetection",
    "OutlineResult",
    "OutlineStep",
    "Collision",
    "FarColor",
    "PaletteError",
    "PaletteMatch",
    "PaletteStep",
    "load_palette",
    "match_palette",
    "save_palette",
    "sort_palette",
    "used_colors",
    "PIXEL_METHODS",
    "RESIZE_METHODS",
    "ResizeStep",
    "SimplifyResult",
    "SimplifyStep",
    "SpritePipeline",
    "SpriteResult",
    "SpriteSettings",
    "convert_sprite",
    "Step",
    "StepResult",
    "background_mask",
    "block_edges",
    "cleanup_sprite",
    "count_colors",
    "detect_outline",
    "detect_grid",
    "duplicate_lines",
    "estimate_background",
    "grid_to_pixels",
    "load_rgba",
    "outline_sprite",
    "parse_color",
    "pixel_downscale",
    "remove_background",
    "resolve_size",
    "rgb_to_oklab",
    "save_rgba",
    "scale_nearest",
    "simplify_colors",
    "merge_close_colors",
    "CloseMergeResult",
    "to_rgba",
    "transition_mask",
]
