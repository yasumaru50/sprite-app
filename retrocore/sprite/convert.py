"""画像と設定を受け取ってスプライトを返す、Sprite App の変換の入口（CLI とブラウザ UI で共通）。

    from retrocore.sprite.convert import SpriteSettings, convert_sprite
    result = convert_sprite(rgba, SpriteSettings(colors=8))
    result.image      # 変換後の RGBA 配列
    result.warnings   # パレットに合わせたときの警告

CLI（`apps/sprite/cli.py`）は引数を `SpriteSettings` に直してからこの関数を呼び、ブラウザ UI
（`apps/sprite/web/`）は Pyodide の中で `convert_for_web()` を呼ぶ。動き（呼吸）は、変換の結果の PNG を材料に
`motion_for_web()` で作る（中身は CLI の `--motion` と同じ `make_motion()`。変換はやり直さない）。工程の組み立てはここだけで行うので、
CLI と画面で結果がずれない。各設定の既定値は CLI の既定と同じ。

工程の順番（docs/sprite-app-plan.md）: 背景除去 → グリッド検出または縮小 → 単純化 → パレットに合わせる
→ クリーンアップ → 縁取り → パレットにない色が無いかの確認。
"""

from __future__ import annotations

import base64
import io
import json
import platform
import sys
import traceback
from dataclasses import dataclass, field, fields
from typing import Callable

import numpy as np
from PIL import Image

from . import outline as outline_mod
from .background import DEFAULT_TOLERANCE, BackgroundStep
from .cleanup import DEFAULT_LEVEL as DEFAULT_CLEANUP, LEVEL_HELP as CLEANUP_LEVELS, CleanupStep
from .downscale import (DEFAULT_BIN_SIZE, DEFAULT_LINE_LUMA, DEFAULT_LINE_RATIO, DEFAULT_TRANSPARENT_RATIO,
                        PIXEL_METHODS)
from .grid import DEFAULT_SNAP_TOLERANCE, GridNotFoundError, GridStep
from .image import to_rgba
from .outline import OutlineStep
from .palette import PaletteError, PaletteStep
from .pipeline import SpritePipeline, StepResult
from .resize import ResizeStep
from .simplify import DEFAULT_COLORS, DEFAULT_KEEP_LARGE, DEFAULT_MERGE_DISTANCE, SimplifyStep

# 入力の種類: ドット絵風（グリッド検出）/ 普通のイラスト（縮小）。自動判定はしない（docs/sprite-app-plan.md の未決事項）
INPUT_TYPES = {"pixel": "ドット絵風（グリッド検出）", "illust": "普通のイラスト（縮小）"}
DEFAULT_INPUT_TYPE = "pixel"
# Sprite App の縮小の既定（ドット絵向けの mode ＋線優先）。retrocore の ResizeStep の既定（nearest）とは別
DEFAULT_RESIZE_METHOD = "mode"
DOT_SHAPES = {"square": "正方形（縦横で同じ大きさ）", "separate": "縦横を別々に推定"}
DEFAULT_DOT_SHAPE = "square"
MAX_COLORS = 256


@dataclass
class SpriteSettings:
    """変換の設定。既定値は CLI（`python -m apps.sprite`）の既定と同じ。

    `line_priority` だけは None（自動: 縮小方法が mode / median のときだけ有効）を既定にする。
    `outline_color` は "darken" / "single" か、(R, G, B) の 1 色。`palette` は (N, 3) の色の配列（None で使わない）。
    """

    input_type: str = DEFAULT_INPUT_TYPE
    # サイズ（input_type="illust" のとき）
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    # 背景除去
    background: bool = True
    bg_color: tuple[int, int, int] | None = None
    bg_tolerance: float = DEFAULT_TOLERANCE
    bg_remove_enclosed: bool = False
    # グリッド検出（input_type="pixel" のとき）
    dot_size: tuple[float, float] | None = None
    dot_shape: str = DEFAULT_DOT_SHAPE
    grid_origin: tuple[float, float] | None = None
    grid_snap: bool = True
    grid_tolerance: float = DEFAULT_SNAP_TOLERANCE
    # 縮小（input_type="illust" のとき）
    resize_method: str = DEFAULT_RESIZE_METHOD
    mode_bin: int = DEFAULT_BIN_SIZE
    line_priority: bool | None = None
    line_ratio: float = DEFAULT_LINE_RATIO
    line_luma: float = DEFAULT_LINE_LUMA
    transparent_ratio: float = DEFAULT_TRANSPARENT_RATIO
    # 単純化
    simplify: bool = True
    colors: int = DEFAULT_COLORS
    merge_distance: float = DEFAULT_MERGE_DISTANCE
    keep_large: float = DEFAULT_KEEP_LARGE   # 大きな面の色を守る割合（0 で以前の動き）
    # クリーンアップ・縁取り
    cleanup: str = DEFAULT_CLEANUP
    outline: str = outline_mod.DEFAULT_MODE
    outline_color: str | tuple[int, int, int] = outline_mod.DEFAULT_COLOR_MODE
    outline_shape: str = outline_mod.DEFAULT_SHAPE
    outline_colors: int = outline_mod.DEFAULT_MAX_COLORS
    outline_pad: bool = True
    # パレット
    palette: np.ndarray | None = field(default=None, repr=False)

    def validate(self) -> None:
        """値の範囲を確かめる（合わなければ ValueError。ブラウザ UI など、CLI の引数チェックを通らない呼び出し向け）。"""
        if self.input_type not in INPUT_TYPES:
            raise ValueError(f"入力の種類は {' / '.join(INPUT_TYPES)} のどれかです: {self.input_type}")
        if not isinstance(self.colors, (int, np.integer)) or not 1 <= self.colors <= MAX_COLORS:
            raise ValueError(f"色数は 1〜{MAX_COLORS} の整数で指定してください: {self.colors}")
        if self.cleanup not in CLEANUP_LEVELS:
            raise ValueError(f"クリーンアップの強さは {' / '.join(CLEANUP_LEVELS)} のどれかです: {self.cleanup}")
        if self.outline not in outline_mod.MODES:
            raise ValueError(f"縁取りは {' / '.join(outline_mod.MODES)} のどれかです: {self.outline}")
        if self.input_type == "illust" and self.width is None and self.height is None and self.scale is None:
            raise ValueError("普通のイラストでは幅・高さ・倍率のどれかを指定してください")

    @classmethod
    def from_dict(cls, values: dict) -> "SpriteSettings":
        """辞書（ブラウザ UI から渡す設定など）から作る。知らない名前があれば ValueError。"""
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        return cls(**values)


@dataclass
class SpriteResult:
    """変換の結果。"""

    image: np.ndarray
    steps: list[StepResult]
    warnings: list[str]
    skipped: list[dict] = field(default_factory=list)   # 作らなかった理由の記録（`motion/report.py`。今はパレットの警告だけ）


def build_pipeline(settings: SpriteSettings) -> SpritePipeline:
    """設定から工程を組み立てる。工程を足すときはここに順番どおり追加する。

    `palette` を指定したときは、単純化の直後（単純化しないときはグリッド検出・縮小の直後）にパレットに合わせ、
    縁取りの色もパレットから選び、最後にパレットにない色が無いかを確かめる。
    """
    s = settings
    pipe = SpritePipeline()
    if s.background:
        pipe.add(BackgroundStep(color=s.bg_color, tolerance=s.bg_tolerance, remove_enclosed=s.bg_remove_enclosed))
    if s.input_type == "pixel":
        pipe.add(GridStep(dot_size=s.dot_size, origin=s.grid_origin, snap=s.grid_snap,
                          snap_tolerance=s.grid_tolerance, square=s.dot_shape == "square",
                          transparent_ratio=s.transparent_ratio))
    else:
        line_priority = s.line_priority
        if line_priority is None:  # 指定が無ければ mode / median のときだけ有効
            line_priority = s.resize_method in PIXEL_METHODS
        pipe.add(ResizeStep(width=s.width, height=s.height, scale=s.scale, method=s.resize_method,
                            bin_size=s.mode_bin, line_priority=line_priority, line_ratio=s.line_ratio,
                            line_luma=s.line_luma, transparent_ratio=s.transparent_ratio))
    if s.simplify:
        pipe.add(SimplifyStep(colors=s.colors, merge_distance=s.merge_distance, keep_large=s.keep_large))
    if s.palette is not None:
        pipe.add(PaletteStep(palette=s.palette, merge_distance=s.merge_distance))
    if s.cleanup != "off":
        pipe.add(CleanupStep(level=s.cleanup))
    if s.outline != "off":
        oc = s.outline_color
        pipe.add(OutlineStep(mode=s.outline, shape=s.outline_shape,
                             color_mode=oc if isinstance(oc, str) else "single",
                             color=None if isinstance(oc, str) else tuple(oc),
                             pad=s.outline_pad, max_colors=s.outline_colors, palette=s.palette))
    if s.palette is not None:
        pipe.add(PaletteStep(palette=s.palette, merge_distance=s.merge_distance, final=True))
    return pipe



def convert_sprite(rgba: np.ndarray, settings: SpriteSettings | None = None,
                   on_step: Callable[[StepResult], None] | None = None) -> SpriteResult:
    """RGBA 配列（または H×W×3）を設定どおりに変換する。CLI もブラウザ UI もこの関数を使う。

    グリッドが見つからないときは `GridNotFoundError` をそのまま送る（呼び出し側で利用者向けの言葉にする）。
    """
    settings = settings or SpriteSettings()
    pipe = build_pipeline(settings)
    steps: list[StepResult] = []

    def _on_step(res: StepResult) -> None:
        steps.append(res)
        if on_step is not None:
            on_step(res)

    out = pipe.run(to_rgba(rgba), on_step=_on_step)
    from .motion.report import palette_entries
    warnings: list[str] = []
    skipped: list[dict] = []
    for step in pipe.steps:
        if isinstance(step, PaletteStep) and not step.final and step.last is not None:
            warnings.extend(step.last.warnings)
            skipped.extend(palette_entries(step.last))
    return SpriteResult(out, steps, warnings, skipped)


# ---- ブラウザ UI（Pyodide）向け: 画像ファイルのバイト列でやりとりする ----

def decode_image(data: bytes) -> np.ndarray:
    """画像ファイルのバイト列を RGBA 配列にする（`load_rgba()` と同じ読み方）。"""
    with Image.open(io.BytesIO(bytes(data))) as img:
        img.load()
        return to_rgba(img)


def encode_png(rgba: np.ndarray) -> bytes:
    """RGBA 配列を PNG のバイト列にする（`save_rgba()` と同じ書き方）。"""
    buf = io.BytesIO()
    Image.fromarray(to_rgba(rgba), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def friendly_error(exc: BaseException) -> str:
    """変換できなかった理由を、画面に出す分かりやすい言葉にする。"""
    if isinstance(exc, GridNotFoundError):
        return ("ドットのマス目が見つからなかったので、変換できませんでした。"
                "この画面は「ドット絵風の画像」（ドット 1 個が何画素もある、AI が描いたドット絵など）向けです。"
                f"\n（くわしい理由: {exc}）")
    if isinstance(exc, PaletteError):
        return f"パレットを使えませんでした: {exc}"
    if isinstance(exc, (OSError, Image.DecompressionBombError)) or type(exc).__name__ == "UnidentifiedImageError":
        return "画像として読み込めませんでした。PNG・JPEG・WebP などの画像ファイルを選んでください。"
    if isinstance(exc, ValueError):
        return f"設定または画像に問題があり、変換できませんでした: {exc}"
    return f"思いがけないエラーで変換できませんでした（{type(exc).__name__}: {exc}）"


def error_detail(exc: BaseException) -> str:
    """画面の「詳細」に出す情報: 動いている環境（Python・NumPy・Pillow の版、32 / 64 ビット）と、Python のトレースバック。

    利用者がそのまま貼り付けて知らせられるようにする。
    """
    import PIL
    env = (f"Python {platform.python_version()} ({sys.platform}) / NumPy {np.__version__} / Pillow {PIL.__version__} / "
           f"intp {np.dtype(np.intp).itemsize * 8} ビット")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{env}\n{tb}".rstrip()


def convert_image_bytes(data: bytes, settings: dict | SpriteSettings | None = None) -> dict:
    """画像ファイルのバイト列を変換し、結果を辞書で返す（ブラウザ UI から呼ぶ。例外は投げない）。

    成功: {"ok": True, "png": PNG のバイト列, "width", "height", "colors": 透明を除く色数, "steps": [説明...],
    "warnings": [...]}。失敗: {"ok": False, "error": 利用者向けの説明, "detail": 詳しい情報（Python のトレースバックなど）}。
    """
    try:
        if not isinstance(settings, SpriteSettings):
            settings = SpriteSettings.from_dict(dict(settings or {}))
        settings.validate()
        rgba = decode_image(data)
        res = convert_sprite(rgba, settings)
    except Exception as e:  # 画面にそのまま出すので、どんなエラーでも言葉にして返す
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    img = res.image
    opaque = img[img[..., 3] > 0][:, :3]
    n_colors = int(len(np.unique(opaque.reshape(-1, 3), axis=0))) if len(opaque) else 0
    return {"ok": True, "png": encode_png(img), "width": int(img.shape[1]), "height": int(img.shape[0]),
            "colors": n_colors, "steps": [r.description for r in res.steps], "warnings": list(res.warnings),
            "skipped": list(res.skipped)}


def convert_for_web(data: bytes, settings_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。設定は JSON 文字列で受け取り、結果も JSON 文字列で返す。

    中身は `convert_image_bytes()` と同じで、PNG だけ base64 の文字列（"png_base64"）にする
    （JavaScript との受け渡しを文字列だけにして、型の変換で結果が変わらないようにするため）。
    """
    try:
        values = json.loads(settings_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"colors\": 12} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": friendly_error(ValueError(f"設定を読めません: {e}")),
                           "detail": error_detail(e)}, ensure_ascii=False)
    res = convert_image_bytes(data, values)
    if res.get("ok"):
        res["png_base64"] = base64.b64encode(res.pop("png")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: 動き（フェーズ 2） ----
# 変換の結果（PNG のバイト列）を材料にして、CLI の --motion と同じ `make_motion()` で動きを作る。
# 変換はやり直さないので、動きの設定を変えたときは、これだけを呼び直せばよい。

# 画面の表の標準の列（設計の「動きの表」の例と同じ名前。プリセットの表もこの名前を使う）と、その列が使う範囲
TABLE_REGION = "upper"
TABLE_COLUMNS = {
    "upper_x": {"type": "move", "region": TABLE_REGION, "axis": "x"},
    "upper_y": {"type": "deform", "region": TABLE_REGION, "axis": "y"},
    "root_y": {"type": "root", "axis": "y"},
}


def motion_spec() -> dict:
    """画面の「動き」の欄の選択肢と既定値（CLI の --motion-frames などの既定と同じ）。

    いちばん外の frames・duration・depth・split は呼吸のもの（前の版と同じ）。横揺れは "sway"、浮遊は "float"、弾むは "bounce"、
    うなずくは "nod" の中（浮遊・弾むは分ける高さが無い。弾むのフレーム数は高さで、うなずくのフレーム数は深さで決まり、
    表示時間の既定は高さ・深さごとの並び "durations"）。
    """
    from .motion import MOTIONS
    from .motion import bouncing as bounce_mod
    from .motion import breathe as breathe_mod
    from .motion import floating as float_mod
    from .motion import nod as nod_mod
    from .motion import spec as spec_mod
    from .motion import stretch as stretch_mod
    from .motion import sway as sway_mod
    return {
        "frames": {"default": breathe_mod.DEFAULT_FRAMES, "min": 2, "max": breathe_mod.MAX_FRAMES},
        "duration": {"default": breathe_mod.DEFAULT_DURATION, "min": 1},
        "depth": {"default": breathe_mod.DEFAULT_DEPTH, "min": 1, "max": breathe_mod.MAX_DEPTH},
        "split": {"default": breathe_mod.AUTO, "range": breathe_mod.SPLIT_RANGE},
        "choices": [[k, v] for k, v in MOTIONS.items()],
        "sway": {
            "frames": {"default": sway_mod.DEFAULT_FRAMES, "min": sway_mod.MIN_FRAMES, "max": sway_mod.MAX_FRAMES},
            "duration": {"default": sway_mod.DEFAULT_DURATION, "min": 1},
            "amount": {"default": sway_mod.DEFAULT_AMOUNT, "min": 1, "max": sway_mod.MAX_AMOUNT},
            "split": {"default": sway_mod.AUTO, "range": sway_mod.SPLIT_RANGE},
        },
        "float": {
            "frames": {"default": float_mod.DEFAULT_FRAMES, "min": float_mod.MIN_FRAMES, "max": float_mod.MAX_FRAMES},
            "duration": {"default": float_mod.DEFAULT_DURATION, "min": 1},
            "amount": {"default": float_mod.DEFAULT_AMOUNT, "min": 1, "max": float_mod.MAX_AMOUNT},
        },
        "bounce": {
            "height": {"default": bounce_mod.DEFAULT_HEIGHT, "min": 1, "max": bounce_mod.MAX_HEIGHT},
            "offsets": {str(h): bounce_mod.bounce_offsets(h) for h in bounce_mod.HEIGHTS},
            "durations": {str(h): bounce_mod.default_durations(h) for h in bounce_mod.HEIGHTS},
            "air": bounce_mod.AIR_DURATION, "landing": bounce_mod.LANDING_DURATION,
        },
        "nod": {
            "depth": {"default": nod_mod.DEFAULT_DEPTH, "min": 1, "max": nod_mod.MAX_DEPTH},
            "offsets": {str(d): [-k for k in nod_mod.nod_offsets(d)] for d in nod_mod.DEPTHS},
            "durations": {str(d): nod_mod.default_durations(d) for d in nod_mod.DEPTHS},
            "up": nod_mod.UP_DURATION, "down": nod_mod.DOWN_DURATION,
            "split": {"default": nod_mod.AUTO, "range": nod_mod.SPLIT_RANGE},
        },
        "stretch": {
            "amount": {"default": stretch_mod.DEFAULT_AMOUNT, "min": 1, "max": stretch_mod.MAX_AMOUNT},
            "offsets": {str(a): stretch_mod.stretch_offsets(a) for a in stretch_mod.AMOUNTS},
            "duration": {"default": stretch_mod.DEFAULT_DURATION, "min": 1},
            "split": {"default": stretch_mod.AUTO, "range": stretch_mod.SPLIT_RANGE},
        },
        # 小さな部品を守る（チェックボックス）を出す動き。既定は切
        "keep_small_parts": {"motions": ["breathe", "nod", "stretch"], "default": False},
        # 画面の表（動きの表を編集する欄）。値の範囲は spec.py と同じ。列は標準の 3 つ（読み込んだ表にほかの列があれば、それも出す）
        "table": {
            "min_rows": spec_mod.MIN_ROWS, "max_rows": spec_mod.MAX_ROWS, "max_value": spec_mod.MAX_VALUE,
            "max_root": spec_mod.MAX_ROOT_VALUE, "default_range": spec_mod.DEFAULT_RANGE,
            "columns": [[name, track] for name, track in TABLE_COLUMNS.items()],
            "region": TABLE_REGION,
        },
    }


def _motion_options(values: dict) -> dict:
    """画面から受け取った動きの設定を、`make_motion()` の引数にする（CLI の引数の確かめと同じ範囲）。"""
    from .motion import MOTIONS
    from .motion import bouncing as bounce_mod
    from .motion import breathe as breathe_mod
    from .motion import floating as float_mod
    from .motion import sway as sway_mod
    known = {"motion", "frames", "durations", "depth", "amount", "split", "height", "image_name", "keep_small_parts"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"知らない動きの設定です: {', '.join(unknown)}")
    name = values.get("motion", breathe_mod.NAME)
    if name not in MOTIONS:
        raise ValueError(f"動きは {' / '.join(MOTIONS)} のどれかです: {name}")
    # 小さな部品を守る（CLI の --keep-small-parts。縮める・伸ばす動きだけ。false は付けないのと同じ）
    values = dict(values)
    keep = values.pop("keep_small_parts", False)
    if not isinstance(keep, bool):
        raise ValueError(f"keep_small_parts は true か false で指定してください: {keep}")
    if keep and name not in ("breathe", "nod", "stretch"):
        raise ValueError(f"keep_small_parts は {name} の設定ではありません（縮める・伸ばす動き（breathe / nod / stretch）の設定です）")
    out = _motion_options_of(name, values)
    if keep:
        out["keep_small_parts"] = True
    return out


def _motion_options_of(name: str, values: dict) -> dict:
    from .motion import bouncing as bounce_mod
    from .motion import breathe as breathe_mod
    from .motion import floating as float_mod
    from .motion import sway as sway_mod
    if name == bounce_mod.NAME:
        return _bounce_options(values)
    if name == "nod":
        return _nod_options(values)
    if name == "stretch":
        return _stretch_options(values)
    if "height" in values:
        raise ValueError(f"height は {name} の設定ではありません（弾む高さです）")
    mod = {sway_mod.NAME: sway_mod, float_mod.NAME: float_mod}.get(name, breathe_mod)
    # CLI の --breathe-depth を --motion sway に付けたときと同じく、ほかの動きの設定はエラー
    owners = {"depth": "呼吸の下がる量", "amount": "横揺れ・浮遊の揺れ幅", "split": "呼吸・横揺れの分ける高さ"}
    mine = {breathe_mod: ("depth", "split"), sway_mod: ("amount", "split"), float_mod: ("amount",)}[mod]
    for other in ("depth", "amount", "split"):
        if other in values and other not in mine:
            raise ValueError(f"{other} は {name} の設定ではありません（{owners[other]}です）")

    def _int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    frames = values.get("frames", mod.DEFAULT_FRAMES)
    low = getattr(mod, "MIN_FRAMES", 2)
    if not _int(frames) or not low <= frames <= breathe_mod.MAX_FRAMES:
        raise ValueError(f"フレーム数は {low}〜{breathe_mod.MAX_FRAMES} の整数で指定してください: {frames}")
    if mod is not breathe_mod:
        amount = values.get("amount", mod.DEFAULT_AMOUNT)
        if not _int(amount) or not 1 <= amount <= mod.MAX_AMOUNT:
            raise ValueError(f"揺れ幅は 1〜{mod.MAX_AMOUNT} ドットの整数で指定してください: {amount}")
        extra = {"amount": amount}
    else:
        depth = values.get("depth", breathe_mod.DEFAULT_DEPTH)
        if not _int(depth) or not 1 <= depth <= breathe_mod.MAX_DEPTH:
            raise ValueError(f"下がる量は 1〜{breathe_mod.MAX_DEPTH} ドットの整数で指定してください: {depth}")
        extra = {"depth": depth}
    durations = values.get("durations", mod.DEFAULT_DURATION)
    if _int(durations):
        durations = [durations]
    if (not isinstance(durations, list) or not durations or not all(_int(d) for d in durations)
            or min(durations) < 1):
        raise ValueError(f"表示時間は 1 以上のミリ秒の整数（500 や 600,400）で指定してください: {durations}")
    if len(durations) not in (1, frames):
        raise ValueError(f"表示時間の数（{len(durations)}）はフレーム数（{frames}）と同じか、1 つ（全フレーム共通）にしてください")
    if "split" in mine:
        split = values.get("split", breathe_mod.AUTO)
        if split != breathe_mod.AUTO and (not _int(split) or split < 1):
            raise ValueError(f"分ける高さは {breathe_mod.AUTO} か、下から何ドット目かの整数（1 = いちばん下の行）で指定してください: {split}")
        extra["split"] = split
    # CLI（write_motion）と同じ渡し方: 表示時間が 1 つなら全フレーム共通の数にする
    return {"name": name, "frames": frames, "durations": durations[0] if len(durations) == 1 else durations, **extra}


def _bounce_options(values: dict) -> dict:
    """弾むの設定（CLI の --bounce-height・--motion-duration と同じ範囲。フレーム数は高さで決まる）。"""
    from .motion import bouncing as bounce_mod

    def _int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    for other, owner in (("depth", "呼吸の下がる量"), ("amount", "横揺れ・浮遊の揺れ幅"), ("split", "呼吸・横揺れの分ける高さ")):
        if other in values:
            raise ValueError(f"{other} は bounce の設定ではありません（{owner}です）")
    height = values.get("height", bounce_mod.DEFAULT_HEIGHT)
    if not _int(height) or not 1 <= height <= bounce_mod.MAX_HEIGHT:
        raise ValueError(f"弾む高さは 1〜{bounce_mod.MAX_HEIGHT} ドットの整数で指定してください: {height}")
    frames = bounce_mod.bounce_frames(height)
    if "frames" in values and values["frames"] != frames:
        raise ValueError(f"弾むのフレーム数は高さで決まります（高さ {height} → {frames} フレーム）: {values['frames']}")
    durations = values.get("durations")
    if durations is None:   # CLI で --motion-duration を省いたときと同じ（着地だけ長め）
        return {"name": bounce_mod.NAME, "height": height}
    if _int(durations):
        durations = [durations]
    if (not isinstance(durations, list) or not durations or not all(_int(d) for d in durations)
            or min(durations) < 1):
        raise ValueError(f"表示時間は 1 以上のミリ秒の整数（150 や 300,150,150,150）で指定してください: {durations}")
    if len(durations) not in (1, frames):
        raise ValueError(f"表示時間の数（{len(durations)}）はフレーム数（高さ {height} → {frames}）と同じか、"
                         "1 つ（全フレーム共通）にしてください")
    return {"name": bounce_mod.NAME, "height": height, "durations": durations[0] if len(durations) == 1 else durations}


def _nod_options(values: dict) -> dict:
    """うなずくの設定（CLI の --nod-depth・--nod-split・--motion-duration と同じ範囲。フレーム数は深さで決まる）。"""
    from .motion import nod as nod_mod

    def _int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    for other, owner in (("amount", "横揺れ・浮遊の揺れ幅"), ("height", "弾む高さ")):
        if other in values:
            raise ValueError(f"{other} は nod の設定ではありません（{owner}です）")
    depth = values.get("depth", nod_mod.DEFAULT_DEPTH)
    if not _int(depth) or depth not in nod_mod.DEPTHS:
        raise ValueError(f"うなずく深さは 1〜{nod_mod.MAX_DEPTH} ドットの整数で指定してください: {depth}")
    frames = nod_mod.nod_frames(depth)
    if "frames" in values and values["frames"] != frames:
        raise ValueError(f"うなずくのフレーム数は深さで決まります（深さ {depth} → {frames} フレーム）: {values['frames']}")
    split = values.get("split", nod_mod.AUTO)
    if split != nod_mod.AUTO and not (_int(split) and split >= 1) and not isinstance(split, str):
        raise ValueError(f"分ける高さは {nod_mod.AUTO} か、下から何ドット目かの整数（1 = いちばん下の行）で指定してください: {split}")
    out = {"name": nod_mod.NAME, "depth": depth, "split": split}
    durations = values.get("durations")
    if durations is None:   # CLI で --motion-duration を省いたときと同じ（上を向いているコマだけ長め）
        return out
    if _int(durations):
        durations = [durations]
    if (not isinstance(durations, list) or not durations or not all(_int(d) for d in durations)
            or min(durations) < 1):
        raise ValueError(f"表示時間は 1 以上のミリ秒の整数（200 や 600,200）で指定してください: {durations}")
    if len(durations) not in (1, frames):
        raise ValueError(f"表示時間の数（{len(durations)}）はフレーム数（深さ {depth} → {frames}）と同じか、"
                         "1 つ（全フレーム共通）にしてください")
    out["durations"] = durations[0] if len(durations) == 1 else durations
    return out


def _stretch_options(values: dict) -> dict:
    """伸びの設定（CLI の --stretch-amount・--stretch-split・--motion-duration と同じ範囲。フレーム数は量で決まる）。"""
    from .motion import stretch as stretch_mod

    def _int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    for other, owner in (("depth", "呼吸・うなずくの下がる量"), ("height", "弾む高さ")):
        if other in values:
            raise ValueError(f"{other} は stretch の設定ではありません（{owner}です）")
    amount = values.get("amount", stretch_mod.DEFAULT_AMOUNT)
    if not _int(amount) or amount not in stretch_mod.AMOUNTS:
        raise ValueError(f"伸びる量は 1〜{stretch_mod.MAX_AMOUNT} ドットの整数で指定してください: {amount}")
    frames = stretch_mod.stretch_frames(amount)
    if "frames" in values and values["frames"] != frames:
        raise ValueError(f"伸びのフレーム数は量で決まります（量 {amount} → {frames} フレーム）: {values['frames']}")
    split = values.get("split", stretch_mod.AUTO)
    if split != stretch_mod.AUTO and not (_int(split) and split >= 1) and not isinstance(split, str):
        raise ValueError(f"分ける高さは {stretch_mod.AUTO} か、下から何ドット目かの整数（1 = いちばん下の行）で指定してください: {split}")
    out = {"name": stretch_mod.NAME, "amount": amount, "split": split}
    durations = values.get("durations")
    if durations is None:   # CLI で --motion-duration を省いたときと同じ（500 ミリ秒）
        return out
    if _int(durations):
        durations = [durations]
    if (not isinstance(durations, list) or not durations or not all(_int(d) for d in durations)
            or min(durations) < 1):
        raise ValueError(f"表示時間は 1 以上のミリ秒の整数（500 や 500,300）で指定してください: {durations}")
    if len(durations) not in (1, frames):
        raise ValueError(f"表示時間の数（{len(durations)}）はフレーム数（量 {amount} → {frames}）と同じか、"
                         "1 つ（全フレーム共通）にしてください")
    out["durations"] = durations[0] if len(durations) == 1 else durations
    return out


def motion_image_bytes(sprite_png: bytes, options: dict | None = None) -> dict:
    """変換の結果（PNG のバイト列）から動きを作り、スプライトシートの PNG と JSON を辞書で返す（例外は投げない）。

    `options` は {"motion": "breathe", "frames": 2, "durations": 500 か [600, 400], "depth": 1, "split": "auto" か 12,
    "image_name": JSON の "image" に書くシートのファイル名}。横揺れは "motion": "sway" で、"depth" の代わりに "amount"（揺れ幅）。
    浮遊は "motion": "float" で "amount"（揺れ幅）だけ（分ける高さ "split" は無い）。弾むは "motion": "bounce" で "height"（1〜3）と
    "durations"（省くと着地だけ長め。フレーム数は高さで決まるので "frames" は要らない）。うなずくは "motion": "nod" で
    "depth"（1〜2）と "split"（"auto" は首を自動で探す）と "durations"（省くと上を向いているコマだけ長め。フレーム数は深さで決まる）。中身は CLI の `--motion` と同じ `make_motion()` と
    `make_sheet()` / `sheet_json()` なので、同じ設定なら CLI が書くファイルとバイト単位で同じになる。

    作れた: {"ok": True, "skipped": False, "sheet_png": PNG, "sheet_json": JSON の文字列, "frame_width", "frame_height",
    "durations": [...], "warnings": [], "split_limits": [下, 上] か None, "height"}。
    付けられなかった（形の都合。CLI では警告）: {"ok": True, "skipped": True, "warnings": [理由...], ...}（シートなし）。
    作らなかった理由の記録は、どちらでも "skipped_records"（`motion/report.py` の一覧）に入れ、付けられなかったときは
    CLI の `<出力名>_<動き>_skipped.json` と同じ中身を "skipped_json" に入れる。`options` の "convert_skipped" に変換の結果の
    "skipped" を渡すと、その記録も前に入る（CLI が変換の結果の記録を入れるのと同じ）。
    設定が合わない（CLI ではエラー）: {"ok": False, "error": 説明, "detail": 詳しい情報}。
    """
    from .motion import make_motion
    from .motion.base import split_limits
    from .motion.report import check_entries
    values = dict(options or {})
    try:   # 変換の工程の記録（画面は変換の結果の "skipped" をそのまま渡す。CLI は変換の結果から渡す）
        conv = check_entries(values.pop("convert_skipped", None))
        swaps = _swap_values(values)
        canvas = _canvas_values(values)
    except ValueError as e:
        return {"ok": False, "error": f"動きの設定に問題があり、作れませんでした: {e}", "detail": error_detail(e)}
    if "spec" in values:
        return _table_motion(sprite_png, values, conv, swaps, canvas)
    try:
        image_name = values.pop("image_name", None) or f"sprite_{values.get('motion', 'breathe')}_sheet.png"
        if not isinstance(image_name, str):
            raise ValueError(f"シートのファイル名は文字列で指定してください: {image_name}")
        opts = _motion_options(values)
        sprite = decode_image(sprite_png)
        name = opts.pop("name")
        res = make_motion(sprite, name, **opts)
    except ValueError as e:  # 設定が合わない（CLI の「エラー（ファイル名）: …」と同じ文言）
        return {"ok": False, "error": f"動きの設定に問題があり、作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:  # 画面にそのまま出すので、どんなエラーでも言葉にして返す
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    lim = split_limits(sprite)
    w, h = res.size
    out = {"ok": True, "skipped": bool(res.skipped), "warnings": list(res.warnings), "motion": res.name,
           "height": int(sprite.shape[0]), "split_limits": None if lim is None else [int(lim[0]), int(lim[1])],
           "frame_width": w, "frame_height": h, "durations": [int(d) for d in res.durations],
           "seam_from_bottom": res.info.get("seam_from_bottom")}
    if res.name == "sway":   # 呼吸の返す値は前の版と同じにする
        out["split_from_bottom"] = res.info.get("split_from_bottom")
        out["padding"] = int(res.info.get("padding", 0))
    elif res.name in ("float", "bounce"):   # 浮遊・弾む: 上下に足した透明の行の数と、フレームごとの上下のずれ（下向きが +）
        out["padding_y"] = [int(v) for v in res.info.get("padding_y", [0, 0])]
        out["offset_y"] = [int(v) for v in res.info.get("offsets", [])]
    elif res.name == "nod":   # うなずく: 自動で見つけた首と、実際に分けた高さ（表の at）。表で編集するときの表
        neck = res.info.get("neck")
        out["neck"] = None if neck is None else {"from_bottom": neck.get("from_bottom"), "used": neck.get("used"),
                                                 "band": neck.get("band")}
        out["split_used"] = None if res.spec is None else res.spec["regions"]["upper"]["at"]
    elif res.name == "stretch":   # 伸び: 上に足した透明の行の数（上, 下）
        out["padding_y"] = [int(v) for v in res.info.get("padding_y", [0, 0])]
    _add_sheet(out, res, image_name, conv, swaps, sprite, canvas)
    return out


def _canvas_values(values: dict) -> tuple[int, str]:
    """設定から "canvas"（{"multiple": 16, "align": "bottom"}。キャンバスを倍数にそろえる。`canvas.py`）を取り出して読む。"""
    from . import canvas as C
    return C.from_dict(values.pop("canvas", None))


def canvas_still_for_web(sprite_png: bytes, options_json: str = "{}") -> str:
    """画面の「③ 変換の結果（PNG）」を保存するときに、キャンバスを倍数にそろえる（CLI の 1 枚の絵の --canvas-multiple と同じ）。
    設定は {"multiple": 16, "align": "bottom", "image_name": "<出力名>.png"}。結果は {"ok", "png_base64", "json"（そろえないときは
    None）, "width", "height", "canvas"}（JSON 文字列）。"""
    from . import canvas as C
    try:
        values = json.loads(options_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"multiple\": 16} のような形で渡してください")
        name = str(values.pop("image_name", None) or "sprite.png")
        multiple, align = C.from_dict(values)
        img, text = C.pad_still(decode_image(sprite_png), name, multiple, align)
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"キャンバスの設定に問題があります: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": friendly_error(e), "detail": error_detail(e)}, ensure_ascii=False)
    out = {"ok": True, "png_base64": base64.b64encode(encode_png(img)).decode("ascii"), "json": text,
           "width": int(img.shape[1]), "height": int(img.shape[0]),
           "canvas": None if text is None else json.loads(text)["canvas"]}
    return json.dumps(out, ensure_ascii=False)


def _swap_values(values: dict):
    """動き・シートの設定から "swaps"（差し替えの一覧。`swap.py` のファイルと同じ形）を取り出して読む。無ければ None。"""
    from .swap import swaps_from_dict
    raw = values.pop("swaps", None)
    if raw is None:
        return None
    s = swaps_from_dict(raw)
    return None if s.empty() else s


def _apply_swaps_view(out: dict, res, swaps, ref, palette=None):
    """差し替えを入れた結果を返し、画面に出す記録を `out` に足す（"swaps"・"swaps_warnings"・"swaps_error"・
    "sheet_durations"・"swap_frames"）。当てられない（大きさ・パレットが違う）ときは元の結果のまま返し、"swaps_error" に理由。"""
    from .swap import SwapError, apply_swaps
    out["swaps"] = None
    out["swaps_error"] = None
    out["swaps_warnings"] = []
    if swaps is None:
        return res
    try:
        got, rep = apply_swaps(res, ref, swaps, palette)
    except SwapError as e:
        out["swaps_error"] = {"code": e.code, "message": str(e)}
        return res
    if rep is None:   # まばたきが無い（口パクだけ）: ふつうのシートは今までと同じ
        return res
    out["swaps"] = rep.info()
    out["swaps_warnings"] = list(rep.warnings)
    out["sheet_durations"] = [int(d) for d in got.durations]
    out["swap_frames"] = got.info.get("swap_frames")
    return got


def _talk_view(out: dict, anim, swaps, ref, palette=None, mirror_anim=None):
    """しゃべりのシート（口パク）の結果を作り、画面に出す記録を `out` に足す（"talk_swaps"・"talk_warnings"・"talk_error"）。
    口パクが無ければ何も足さずに [] を返す。作れれば [しゃべりの結果]（`mirror_anim` があれば反転も）。"""
    from .swap import SwapError, apply_talk, mirror_swapped
    if swaps is None or swaps.find("talk") is None:
        return []
    out["talk_swaps"] = None
    out["talk_error"] = None
    out["talk_warnings"] = []
    try:
        got, rep = apply_talk(anim, ref, swaps, palette)
    except SwapError as e:
        out["talk_error"] = {"code": e.code, "message": str(e)}
        return []
    out["talk_swaps"] = rep.info()
    out["talk_warnings"] = list(rep.warnings)
    if got is None:
        return []
    import dataclasses
    talks = [got]
    if mirror_anim is not None:
        talks.append(mirror_swapped(got, dataclasses.replace(mirror_anim, name=got.name + "_mirror")))
    return talks


def _add_sheet(out: dict, res, image_name: str, conv: list, swaps=None, ref=None, canvas=(0, "bottom")) -> None:
    """作れたらシートの PNG と JSON、作れなかったら作らなかった理由の JSON（CLI の `_skipped.json` と同じ）を足す。
    `swaps` があれば、シートに差し替え（まばたきなど）を入れる（`ref` は差し替えを描いた絵。変換したスプライト）。
    口パクがあれば、しゃべりのシートも "talk"（{"name", "sheet_png", "sheet_json", ...}）に足す（CLI の
    `<出力名>_<動き>_talk_sheet.png` と同じ中身）。`canvas` は (倍数, 並べ方)。シートに透明な余白を足す（`canvas.py`）。"""
    from . import canvas as C
    from .motion import make_sheet, sheet_json
    from .motion.report import skipped_json
    out["skipped_records"] = list(conv) + list(res.skips)
    if swaps is not None and not res.skipped:
        from .swap import swaps_json, talk_image_name
        from .touchup import palette_of
        talks = _talk_view(out, res, swaps, ref)
        if "talk_swaps" in out:
            out["talk"] = None
        for t in talks:
            t = C.pad_result(t, *canvas)
            w, h = t.size
            name = talk_image_name(image_name)
            out["talk"] = {"name": name, "sheet_png": encode_png(make_sheet(t.frames)),
                           "sheet_json": sheet_json(t, name, conv), "frame_width": int(w), "frame_height": int(h),
                           "frame_count": len(t.frames), "durations": [int(d) for d in t.durations]}
        res = _apply_swaps_view(out, res, swaps, ref)
        out["swaps_json"] = swaps_json(swaps, palette_of([ref]))   # 保存する <出力名>_<動き>_swaps.json の中身
    if not res.skipped:
        res = C.pad_result(res, *canvas)
        out["canvas"] = res.info.get("canvas")
        out["sheet_png"] = encode_png(make_sheet(res.frames))
        out["sheet_json"] = sheet_json(res, image_name, conv)
    else:
        out["skipped_json"] = skipped_json(res, conv)


def motion_for_web(sprite_png: bytes, options_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。設定は JSON 文字列で受け取り、結果も JSON 文字列で返す（PNG は base64）。"""
    try:
        values = json.loads(options_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("動きの設定は {\"frames\": 2} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"動きの設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    res = motion_image_bytes(sprite_png, values)
    if "sheet_png" in res:
        res["sheet_png_base64"] = base64.b64encode(res.pop("sheet_png")).decode("ascii")
    if res.get("talk"):
        res["talk"]["sheet_png_base64"] = base64.b64encode(res["talk"].pop("sheet_png")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: 動きの表（画面の表） ----
# 画面の表は、CLI の --motion-spec と同じ `make_motion_from_spec()` でシートを作る。保存する表の中身は
# CLI の --write-motion-spec と同じ（`with_height_style()` → `spec_json()`）。

_SPEC_ROW_RE = None


def spec_error_items(problems: list[str]) -> list[dict]:
    """表の書き間違い（`SpecError` の言葉）を、画面で赤くするマスの情報にする（row は 1 から。分からなければ None）。"""
    import re
    global _SPEC_ROW_RE
    if _SPEC_ROW_RE is None:
        _SPEC_ROW_RE = re.compile(r"^(\d+) 行目: (?:([A-Za-z0-9_-]+) の値|(ms)（|知らない列です: ([A-Za-z0-9_-]+))?")
    out = []
    for text in problems:
        m = _SPEC_ROW_RE.match(text)
        row = int(m.group(1)) if m else None
        track = (m.group(2) or m.group(3) or m.group(4)) if m else None
        out.append({"row": row, "track": track, "message": text})
    return out


def _spec_error_result(e) -> dict:
    return {"ok": False, "error": "動きの表に書き間違いがあります（表は変えていません）:\n"
                                  + "\n".join(f"- {p}" for p in e.problems),
            "spec_errors": spec_error_items(e.problems), "detail": ""}


def _table_motion(sprite_png: bytes, values: dict, conv: list | None = None, swaps=None, canvas=(0, "bottom")) -> dict:
    """表（{"spec": 表の辞書か JSON の文字列, "image_name": シートの名前}）から動きを作る（`motion_image_bytes()` の表の経路）。

    作れた・作れなかったの返し方はプリセットと同じで、ほかに "problems"（作れない行。row は 1 から。画面でマスを赤くする）、
    "spec"（読んだ表。作れたときは範囲に "used" つき）、"spec_files"（保存する表の中身。"number"・"percent" の 2 通り。
    CLI の --write-motion-spec --spec-height number|percent と同じ文字列）を返す。表の書き間違いは {"ok": False, "spec_errors": [...]}。
    """
    from .motion import SpecError, make_motion_from_spec, parse_spec, spec_json, with_height_style
    from .motion.base import split_limits
    try:
        unknown = sorted(set(values) - {"spec", "image_name"})
        if unknown:
            raise ValueError(f"知らない動きの設定です: {', '.join(unknown)}")
        spec = parse_spec(values["spec"])
        image_name = values.get("image_name") or f"sprite_{spec.name}_sheet.png"
        if not isinstance(image_name, str):
            raise ValueError(f"シートのファイル名は文字列で指定してください: {image_name}")
        sprite = decode_image(sprite_png)
        res = make_motion_from_spec(sprite, spec)
    except SpecError as e:
        return _spec_error_result(e)
    except ValueError as e:  # 画像に合わない高さ（CLI の「エラー（ファイル名）: …」と同じ文言）
        return {"ok": False, "error": f"動きの表がこの画像に合わず、作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    lim = split_limits(sprite)
    w, h = res.size
    out = {"ok": True, "skipped": bool(res.skipped), "warnings": list(res.warnings), "motion": res.name, "table": True,
           "height": int(sprite.shape[0]), "split_limits": None if lim is None else [int(lim[0]), int(lim[1])],
           "frame_width": w, "frame_height": h, "durations": [int(d) for d in res.durations],
           "problems": list(res.info.get("problems", [])), "regions": res.info.get("regions", {}),
           "padding": int(res.info.get("padding", 0)), "padding_y": [int(v) for v in res.info.get("padding_y", [0, 0])],
           "offset_y": [int(v) for v in res.info.get("offset_y", [])], "poses": res.info.get("poses"),
           "spec": res.spec,
           "spec_files": {style: spec_json(with_height_style(res.spec, sprite, style)) for style in ("number", "percent")}}
    _add_sheet(out, res, image_name, conv or [], swaps, sprite, canvas)
    return out


def table_for_web(request_json: str = "{}", sprite_png: bytes | None = None) -> str:
    """画面の表の準備（Pyodide から呼ぶ。結果は JSON 文字列）。画像は要らない（うなずくの自動の首だけは `sprite_png` から決める）。

    - {"preset": {プリセットの設定。motion_for_web と同じ形}}: そのプリセットが埋める表（CLI の --write-motion-spec の
      "used" を除いたもの。`preset_table()`）→ {"ok": True, "spec": 表}。うなずくで分ける高さが自動のときは、
      変換の結果（`sprite_png`）から首の高さを決めて数で書く（CLI の --motion nod --write-motion-spec と同じ数）
    - {"text": ".motion.json の中身"}: 表を読んで書き間違いを確かめる → {"ok": True, "spec": 読んだ表（`spec_to_dict()`）}。
      書き間違いがあれば全部まとめて {"ok": False, "error": ..., "spec_errors": [{"row", "track", "message"}, ...]}
    """
    from .motion import SpecError, parse_spec, preset_table
    from .motion.spec import spec_to_dict
    try:
        req = json.loads(request_json or "{}")
        if not isinstance(req, dict) or len(set(req) & {"preset", "text"}) != 1:
            raise ValueError("{\"preset\": {...}} か {\"text\": \"...\"} の形で渡してください")
        if "preset" in req:
            values = dict(req["preset"])
            values.pop("image_name", None)
            opts = _motion_options(values)
            name = opts.pop("name")
            if name == "nod" and sprite_png is not None:
                opts["rgba"] = decode_image(sprite_png)
            return json.dumps({"ok": True, "spec": preset_table(name, **opts)}, ensure_ascii=False)
        text = req["text"]
        if not isinstance(text, str):
            raise ValueError("表のファイルの中身は文字列で渡してください")
        return json.dumps({"ok": True, "spec": spec_to_dict(parse_spec(text.lstrip("\ufeff")))}, ensure_ascii=False)
    except SpecError as e:
        return json.dumps(_spec_error_result(e), ensure_ascii=False)
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"表を用意できませんでした: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": friendly_error(e), "detail": error_detail(e)}, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: GIF アニメの保存 ----
# 画面は、保存できるシートの PNG と JSON（motion_for_web の結果）から GIF を作る。中身は CLI の --gif と同じ
# `motion/gif.py` の `make_gif()`（シートのフレームと表示時間をそのまま渡す）なので、同じ設定なら同じ GIF になる。

def gif_image_bytes(sheet_png: bytes, sheet_json_text: str, options: dict | None = None) -> dict:
    """シートの PNG と JSON から GIF を作る（例外は投げない）。`options` は {"scale": 1〜8, "background": None か "#rrggbb"}。

    作れた: {"ok": True, "gif": GIF のバイト列, "name": 保存する名前（<出力名>_<動き>.gif）, "durations", "frame_durations",
    "frames_of", "merged", "colors", "warnings", "scale", "background"}。作れない: {"ok": False, "error", "detail"}。
    """
    from .motion.gif import DEFAULT_SCALE, frames_from_sheet, gif_path, make_gif
    values = dict(options or {})
    try:
        unknown = sorted(set(values) - {"scale", "background"})
        if unknown:
            raise ValueError(f"知らない GIF の設定です: {', '.join(unknown)}")
        info = json.loads(sheet_json_text)
        if not isinstance(info, dict):
            raise ValueError("シートの情報は JSON の {...} で渡してください")
        frames, durations = frames_from_sheet(decode_image(sheet_png), info)
        gif = make_gif(frames, durations, values.get("scale", DEFAULT_SCALE), values.get("background"))
    except ValueError as e:
        return {"ok": False, "error": f"GIF を作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    out = {"ok": True, "gif": gif.data, "name": gif_path(str(info.get("image") or "sprite_sheet.png")).name}
    out.update(gif.summary())
    return out


def gif_for_web(sheet_png: bytes, sheet_json_text: str, options_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（GIF は base64 の "gif_base64"）。"""
    try:
        values = json.loads(options_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("GIF の設定は {\"scale\": 4} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"GIF の設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    res = gif_image_bytes(sheet_png, sheet_json_text, values)
    if "gif" in res:
        res["gif_base64"] = base64.b64encode(res.pop("gif")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: コマが並んだ絵（CLI の --frames） ----
# 画面は 2 段で呼ぶ: (1) `frames_convert_for_web()` で絵を変換してコマをそろえ（重い。結果は Pyodide の中に覚えておく）、
# (2) `frames_sheets_for_web()` で順番・表示時間・体の上下からシートを作る（軽い。変換はやり直さない）。
# 中身は CLI の `write_frames()` と同じ関数（`convert_frames()` → `assemble_frames()` → `make_sheet()` / `frames_sheet_json()`、
# ちらつきは `flicker_report()` → `flicker_image()` / `flicker_json()`）なので、同じ設定なら CLI が書くファイルとバイト単位で同じになる。

_FRAMES_CACHE: dict = {}   # 最後に変換したコマ（{"token": 番号, "result": FramesResult}）。画面は 1 枚ずつ変換するので 1 つだけ覚える
_FRAMES_PREV: dict = {}    # その 1 つ前の変換（直しを当てられないときの「やめる」で戻す。`frames_restore()`）


def _same_values(values: dict) -> dict:
    """画面の範囲の設定（"same_ranges": [{"type": "rect", "x", "y", "w", "h"}, ...], "same_search", "same_from"）を
    FrameSettings の引数にする。"""
    from .same import DEFAULT_SEARCH, range_from_dict
    rs = values.get("same_ranges") or []
    if not isinstance(rs, list):
        raise ValueError("1 コマ目と同じにする範囲は [{\"x\": 0, \"y\": 0, \"w\": 10, \"h\": 10}] のような並びで渡してください")
    search = values.get("same_search", DEFAULT_SEARCH)
    same_from = values.get("same_from", 1)
    for k, v in (("same_search", search), ("same_from", same_from)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{k} は整数で指定してください: {v}")
    return {"same_ranges": tuple(range_from_dict(r) for r in rs), "same_search": search, "same_from": same_from}


def _frame_settings(values: dict):
    from .frames import FrameSettings
    from .steady import DEFAULT_MODE as STEADY_MODE, DEFAULT_SIZE as STEADY_SIZE
    unknown = sorted(set(values) - {"frame_count", "align", "align_y", "steady", "steady_size",
                                    "same_ranges", "same_search", "same_from", "shift_y"})
    if unknown:
        raise ValueError(f"知らないコマの設定です: {', '.join(unknown)}")
    fc = values.get("frame_count")
    if fc is not None and (isinstance(fc, bool) or not isinstance(fc, int)):
        raise ValueError(f"コマの数は整数で指定してください: {fc}")
    sy = values.get("shift_y")
    if sy is not None and not isinstance(sy, list):
        raise ValueError(f"コマごとの上げ下げは [0, -3, -6, 0] のような整数の並びで指定してください: {sy}")
    fs = FrameSettings(frame_count=fc, align=values.get("align", "match"), align_y=values.get("align_y", "foot"),
                       steady=values.get("steady", STEADY_MODE), steady_size=values.get("steady_size", STEADY_SIZE),
                       shift_y=None if sy is None else tuple(sy), **_same_values(values))
    fs.validate()
    return fs


def _frames_view(res) -> dict:
    """変換・範囲のやり直しの後に画面へ返す、コマの結果（そろえたコマの絵・ちらつき・範囲の記録）。"""
    from . import frames as F
    from .join import file_starts
    strip = F.source_strip(res)
    rep = F.flicker_report(res.frames)
    out = {"png": encode_png(strip), "width": int(strip.shape[1]), "height": int(strip.shape[0]),
           "warnings": list(res.warnings), "skipped": list(res.skipped),
           "flicker_png": encode_png(F.flicker_image(res.frames, rep, file_starts=file_starts(res))),
           "flicker_json": F.flicker_json(rep),
           "flicker_rows": rep.rows, "flicker_counts": [f.count for f in rep.frames],
           "steady": res.steady.info() if res.steady is not None else None,
           "steady_counts": res.steady.counts() if res.steady is not None else None,
           "same": res.same.info() if res.same is not None else None, "same_png": None,
           "same_warnings": list(res.same.warnings) if res.same is not None else []}
    out.update(_touchup_view(res))
    if res.same is not None:
        from .same import same_image
        out["same_png"] = encode_png(same_image(res.aligned, F.same_view_frames(res), res.same))
    return out


def _touchup_view(res) -> dict:
    """「ドットを直す」欄に返すもの: 直す前のコマを並べた絵（直しの下の色）、パレット、直しの記録・警告・下の色が変わったドット。"""
    from . import frames as F
    from .touchup import palette_of
    base = res.before_touchup or res.frames
    rep = res.touchup
    def changed(f):
        e = {"source_frame": int(f.frame)}
        if f.mirrored:
            e["mirrored"] = True   # 裏返したコマ（順番の r）だけの直し。座標は裏返した絵の上
        e["dots"] = [[int(x), int(y)] for x, y in f.changed_under]
        return e
    a, b, fx = res.flip_layout()
    return {"touchup_base_png": encode_png(np.concatenate(base, 1)), "touchup_palette": palette_of(base),
            "touchup": rep.info() if rep is not None else None,
            "touchup_warnings": list(rep.warnings) if rep is not None else [],
            "touchup_changed": [changed(f) for f in rep.frames if f.changed_under] if rep is not None else [],
            # 裏返したコマ（r）の置き方（左に足す列・右に足す列・裏返したコマの横の位置）。再生の見本で使う
            "flip_layout": [int(a), int(b), int(fx)]}


def _touchup_apply(finished, edits) -> tuple:
    """最後の結果（直しの前）に今の直しを当てる。(結果, 当てられなかった理由か None)。当てられなければ直しの前のまま。"""
    from .touchup import TouchupError, touchup_result
    if edits is None or edits.empty():
        return finished, None
    try:
        return touchup_result(finished, edits), None
    except TouchupError as e:
        return finished, {"code": e.code, "message": str(e)}


def _edits_origins(res) -> bool:
    """直しの一覧に、ファイルが 1 つのときもコマを置いた位置（origin）を書くか。縦のそろえ方 keep・コマごとの上げ下げを
    使ったときだけ（上げ下げを変えて変換し直しても、直しがキャラの同じ所に付くように。使わないときのファイルは前と同じ）。"""
    # 動きをまとめたとき（anims）は、ほかの動きの大きさでキャンバスが変わるので、いつも書く
    return res.align_y == "keep" or getattr(res, "shift_y", None) is not None or getattr(res, "anims", None) is not None


def frames_convert_bytes(data, request: dict) -> dict:
    """コマが並んだ絵を変換してそろえる（例外は投げない）。`request` は
    {"settings": 変換の設定（convert_for_web と同じ）, "frames": {"frame_count": N か None, "align", "align_y",
    "steady": "off" / "on"（CLI の --steady）, "steady_size": N（--steady-size）}, "name": 名前}。

    作れた: {"ok": True, "made": True, "png": そろえたコマを元の順に並べた絵（CLI の <出力名>.png）, "token",
    "frame_width", "frame_height", "frame_count", "boxes": [[x0, y0, x1, y1], ...]（元の絵の画素。終わりは含まない）,
    "dot_size", "colors", "notes", "warnings", "steps", "flicker_png", "flicker_json", "flicker_counts",
    "steady": ちらつきを減らした記録（シートの JSON の steady と同じ。off なら None）, "steady_counts": 各コマで戻したドット数,
    "same": 1 コマ目と同じにした記録（範囲が無ければ None）, "same_png": 範囲の図, "same_warnings",
    "aligned_png": そろえた直後のコマを並べた絵（範囲を描くための 1 コマ目の元）}。
    コマを作らなかった（CLI では「コマを作りませんでした」と <出力名>_<名前>_skipped.json）: {"ok": True, "made": False,
    "error": 説明, "skipped_json": 理由の JSON, "boxes": 見つかったコマ（分かれば）}。設定の間違いなど: {"ok": False, "error", "detail"}。

    別々のファイルの絵をつなげるとき（CLI の --join）は、`data` を [(ファイルの名前, バイト列), ...]（並べた順）にする。
    `request` の "match_height": True でキャラの高さを合わせる（CLI の --join-match-height。既定は合わせない）。結果には、どのときも
    "files": ファイルごとの記録（`join.JoinFile.info()` と "boxes"）, "sources": コマごとの [ファイルの名前, 何コマ目か],
    "joins": 境目の記録, "file_starts": 2 つ目からのファイルの 1 コマ目の番号 が入る（1 つのときの絵・ファイルは前と同じ）。
    """
    from . import frames as F
    from .grid import GridNotFoundError
    try:
        unknown = sorted(set(request) - {"settings", "frames", "name", "match_height"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        settings = SpriteSettings.from_dict(dict(request.get("settings") or {}))
        settings.validate()
        fs = _frame_settings(dict(request.get("frames") or {}))
        name = str(request.get("name") or F.DEFAULT_NAME)
        match_height = request.get("match_height", False)
        if not isinstance(match_height, bool):
            raise ValueError(f"match_height は true か false で指定してください: {match_height}")
        files = [("image", data)] if isinstance(data, (bytes, bytearray, memoryview)) else list(data)
        images = []
        for fname, fdata in files:
            try:
                images.append((str(fname), decode_image(bytes(fdata))))
            except Exception as e:
                raise ValueError(f"{fname}: {friendly_error(e)}") from None
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    try:
        from .join import convert_frame_files
        res = convert_frame_files(images, settings, fs, match_height=match_height)
    except (F.FramesError, GridNotFoundError) as e:
        boxes = e.extra.get("boxes") if isinstance(e, F.FramesError) else None
        return {"ok": True, "made": False, "error": str(e), "code": getattr(e, "code", "grid_not_found"),
                "skipped_json": F.frames_skipped_json(e, name), "boxes": boxes}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    token = int(_FRAMES_CACHE.get("token", 0)) + 1
    _FRAMES_PREV.clear()
    _FRAMES_PREV.update(_FRAMES_CACHE)
    _FRAMES_CACHE.clear()
    _FRAMES_CACHE.update({"token": token, "result": res, "base": res, "fs": fs, "finished": res, "edits": None})
    strip = F.source_strip(res)
    opaque = strip[strip[..., 3] > 0][:, :3]
    w, h = res.size
    out = {"ok": True, "made": True, "token": token,
           "frame_width": int(w), "frame_height": int(h), "frame_count": len(res.frames),
           "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in res.boxes],
           "dot_size": round(float(res.dot_size[0]), 3),
           "colors": int(len(np.unique(opaque.reshape(-1, 3), axis=0))) if len(opaque) else 0,
           "notes": list(res.notes), "steps": [r.description for r in res.steps],
           "aligned_png": encode_png(np.concatenate(res.aligned, 1)),
           "files": [dict(f.info(), boxes=[[b.x0, b.y0, b.x1, b.y1] for b in f.boxes]) for f in res.files],
           "sources": [[n, int(k)] for n, k in res.sources], "joins": list(res.joins or []),
           "offsets": [[int(x), int(y)] for x, y in res.offsets],
           # 裏返したコマ（順番の r）の中で、各コマの絵を置いた位置（裏返したコマの直しの origin）
           "mirror_offsets": [[int(x), int(y)] for x, y in res.mirror_offsets()],
           "shift_y": None if res.shift_y is None else [int(v) for v in res.shift_y],
           "edits_origin": _edits_origins(res),   # 保存する直しの一覧に origin を書くか（ファイルが 1 つのとき）
           "file_starts": [f.first for f in res.files[1:]]}
    out.update(_frames_view(res))
    return out


def frames_same(request: dict) -> dict:
    """最後に変換したコマの「1 コマ目と同じにする範囲」だけをやり直す（変換はやり直さない。軽い）。`request` は
    {"token": 変換の番号, "same_ranges": [{"type": "rect" / "band", "x", "y", "w", "h"}, ...], "same_search": N, "same_from": K}。
    ちらつきを減らす設定は変換のときのまま（範囲を写した後に動かす）。結果は CLI の `--same-as-first` と同じコマ・同じ記録。

    作れた: {"ok": True, "token", "png", "flicker_png", …（`frames_convert_bytes()` と同じ項目）, "same": 範囲の記録,
    "same_png": 範囲の図, "same_warnings"}。設定の間違い: {"ok": False, "error", "detail"}。
    """
    import dataclasses

    from . import frames as F
    try:
        unknown = sorted(set(request) - {"token", "same_ranges", "same_search", "same_from"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        if not _FRAMES_CACHE or request.get("token") != _FRAMES_CACHE.get("token"):
            raise ValueError("コマの変換の結果がありません（もう一度変換してください）")
        values = {k: v for k, v in request.items() if k != "token"}
        fs = dataclasses.replace(_FRAMES_CACHE["fs"], **_same_values(values))
        if fs.same_from > len(_FRAMES_CACHE["base"].frames):
            raise ValueError(f"元にするコマは 1〜{len(_FRAMES_CACHE['base'].frames)} にしてください: {fs.same_from}")
        res = F.refinish_frames(_FRAMES_CACHE["base"], fs)
    except ValueError as e:
        return {"ok": False, "error": f"範囲の設定に問題があり、写せませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    _FRAMES_CACHE["finished"] = res
    finished = res
    edits = _FRAMES_CACHE.get("edits")
    res, err = _touchup_apply(res, edits)
    _FRAMES_CACHE["result"] = res
    out = {"ok": True, "token": _FRAMES_CACHE["token"], "touchup_error": err, "edits": None, "edits_json": None}
    if edits is not None and err is None:
        from .touchup import palette_of, touchup_dict, touchup_json
        pal = palette_of(finished.frames)
        og = _edits_origins(finished)
        out["edits"] = touchup_dict(edits, pal, finished.sources, finished.offsets, finished.mirror_offsets(), og)
        out["edits_json"] = touchup_json(edits, pal, finished.sources, finished.offsets, finished.mirror_offsets(), og)
    out.update(_frames_view(res))
    return out


def frames_restore(request: dict) -> dict:
    """1 つ前の変換の結果（範囲・直しも含めて覚えていたもの）に戻す。`request` は {"token": 戻したい変換の番号}。
    画面で変換し直した後、直しを当てられない確認で「やめる」を選んだときに使う。戻せるのは 1 回だけ。
    戻せた: {"ok": True, "token"}。戻せない: {"ok": False, "error", "detail"}。
    """
    try:
        unknown = sorted(set(request) - {"token"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        if not _FRAMES_PREV or request.get("token") != _FRAMES_PREV.get("token"):
            raise ValueError("前の変換の結果がもうありません（もう一度変換してください）")
    except ValueError as e:
        return {"ok": False, "error": f"前の結果に戻せませんでした: {e}", "detail": error_detail(e)}
    prev = dict(_FRAMES_PREV)
    _FRAMES_PREV.clear()
    _FRAMES_CACHE.clear()
    _FRAMES_CACHE.update(prev)
    return {"ok": True, "token": _FRAMES_CACHE["token"]}


def frames_restore_for_web(request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"token\": 1} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"前の結果に戻せませんでした: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    return json.dumps(frames_restore(values), ensure_ascii=False)


def frames_edits(request: dict) -> dict:
    """最後のコマ（変換・範囲の後）に手の直しを当て直す（変換も範囲もやり直さない。軽い）。`request` は
    {"token": 変換の番号, "edits": 直しの一覧（`touchup` のファイルと同じ形）か None（直しを全部消す）}。
    結果は CLI の `--edits` と同じコマ・同じ記録。

    当てた: {"ok": True, "applied": True, "token", "png", …（`frames_convert_bytes()` と同じ項目）,
    "edits": 今のパレットの番号に直した一覧, "edits_json": 保存する <出力名>_<名前>_edits.json の中身}。
    当てられない（大きさ・コマの数・パレットが違う）: {"ok": True, "applied": False, "code", "error"}（覚えている直しは変えない）。
    設定の間違い: {"ok": False, "error", "detail"}。
    """
    from .touchup import (TouchupError, palette_of, remap_touchup, touchup_dict, touchup_from_dict, touchup_json,
                          touchup_result)
    try:
        unknown = sorted(set(request) - {"token", "edits"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        if not _FRAMES_CACHE or request.get("token") != _FRAMES_CACHE.get("token"):
            raise ValueError("コマの変換の結果がありません（もう一度変換してください）")
        raw = request.get("edits")
        edits = None if raw is None else touchup_from_dict(raw)
    except ValueError as e:
        return {"ok": False, "error": f"直しの一覧に問題があり、当てられませんでした: {e}", "detail": error_detail(e)}
    finished = _FRAMES_CACHE.get("finished") or _FRAMES_CACHE["result"]
    try:
        # つなげたときは、直しのファイルの記録で今のコマの番号に付け直す（並び順を入れ替えても同じ絵のコマに付く）
        edits = remap_touchup(edits, finished.sources, finished.offsets, finished.size,
                              finished.mirror_offsets()) if edits is not None else None
        res = touchup_result(finished, edits)
    except TouchupError as e:
        return {"ok": True, "applied": False, "code": e.code, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    _FRAMES_CACHE["edits"] = None if edits is None or edits.empty() else edits
    _FRAMES_CACHE["result"] = res
    out = {"ok": True, "applied": True, "token": _FRAMES_CACHE["token"], "edits": None, "edits_json": None}
    if _FRAMES_CACHE["edits"] is not None:
        pal = palette_of(finished.frames)
        og = _edits_origins(finished)
        out["edits"] = touchup_dict(edits, pal, finished.sources, finished.offsets, finished.mirror_offsets(), og)
        out["edits_json"] = touchup_json(edits, pal, finished.sources, finished.offsets, finished.mirror_offsets(), og)
    out.update(_frames_view(res))
    return out


def frames_edits_for_web(request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（PNG は base64 の "png_base64" など）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"token\": 1, \"edits\": {...}} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"直しの一覧を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    return json.dumps(_web_b64(frames_edits(values)), ensure_ascii=False)


def _web_b64(res: dict) -> dict:
    for key in ("png", "flicker_png", "same_png", "aligned_png", "touchup_base_png"):
        if res.get(key) is not None:
            res[key + "_base64"] = base64.b64encode(res.pop(key)).decode("ascii")
        else:
            res.pop(key, None)
    return res


def frames_same_for_web(request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（PNG は base64 の "png_base64" など）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"token\": 1, \"same_ranges\": [...]} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"範囲の設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    return json.dumps(_web_b64(frames_same(values)), ensure_ascii=False)


def frames_convert_for_web(data: bytes, request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（PNG は base64 の "png_base64"・"flicker_png_base64"）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"settings\": {...}, \"frames\": {...}} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": friendly_error(ValueError(f"設定を読めません: {e}")),
                           "detail": error_detail(e)}, ensure_ascii=False)
    return json.dumps(_web_b64(frames_convert_bytes(data, values)), ensure_ascii=False)


def frames_convert_files_for_web(datas: list, names: list, request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形（ファイルを何枚も選んだとき。`datas` はバイト列、`names` はファイルの名前の並び）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"settings\": {...}, \"frames\": {...}} のような形で渡してください")
        if len(datas) != len(names) or not datas:
            raise ValueError("ファイルの名前とバイト列の数が合いません")
    except ValueError as e:
        return json.dumps({"ok": False, "error": friendly_error(ValueError(f"設定を読めません: {e}")),
                           "detail": error_detail(e)}, ensure_ascii=False)
    return json.dumps(_web_b64(frames_convert_bytes(list(zip(names, datas)), values)), ensure_ascii=False)


def frames_sheets(request: dict, result=None) -> dict:
    """そろえたコマ（`result`。省くと最後に変換したもの）からシートを作る（例外は投げない）。`request` は
    {"token": 変換の番号, "base": 出力名（<出力名>.png の名前の部分）, "name": 名前,
    "order": [1, 2, 3, 2] か [1, 2, 3, "2r"]（r は左右に裏返したコマ）か "1,2,3,2r" か None,
    "durations": [150] か フレームごとの並び, "root_y": [0, -1, 0, -1] か None, "mirror": 左右反転のシートも作るか}。

    作れた: {"ok": True, "sheets": [{"name": シートの PNG の名前, "png": PNG, "json": JSON の文字列, "frame_width",
    "frame_height", "frame_count", "durations", "mirror", "piskel_name", "piskel": .piskel の中身（CLI の --piskel）,
    "piskel_fps", "piskel_notes"}, ...]}（反転は 2 つ目）。作れない: {"ok": False, "error", "detail"}。
    """
    from . import frames as F
    from .motion import make_sheet
    try:
        unknown = sorted(set(request) - {"token", "base", "name", "order", "durations", "root_y", "mirror", "swaps",
                                         "canvas"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        if result is None:
            if not _FRAMES_CACHE or request.get("token") != _FRAMES_CACHE.get("token"):
                raise ValueError("コマの変換の結果がありません（もう一度変換してください）")
            result = _FRAMES_CACHE["result"]
        base = str(request.get("base") or "sprite")
        name = str(request.get("name") or F.DEFAULT_NAME)
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"名前は半角の英数字と _ - で書いてください（walk・run など）: {name}")
        order = request.get("order")
        durations = request.get("durations")
        root_y = request.get("root_y")
        swaps = _swap_values(dict(request))
        canvas = _canvas_values(dict(request))
        anims = [F.assemble_frames(result, order, durations, root_y, name)]
        if request.get("mirror"):
            anims.append(F.assemble_frames(result, order, durations, root_y, name, mirror=True))
    except ValueError as e:
        return {"ok": False, "error": f"シートの設定に問題があり、作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    from .piskel import fps_for_durations, piskel_text
    out: dict = {"ok": True}
    fw = F.flip_warning(order)
    if fw:
        out["flip_warning"] = fw   # 順番に r があるとき: 左右で違う物は反対側に移る、の注意
    talks = []
    if swaps is not None:
        from .swap import mirror_swapped, swaps_json
        from .touchup import palette_of
        pal = palette_of(result.frames)
        talks = _talk_view(out, anims[0], swaps, result.frames[0], pal, anims[1] if len(anims) > 1 else None)
        got = _apply_swaps_view(out, anims[0], swaps, result.frames[0], pal)
        if got is not anims[0]:
            anims = [got] + [mirror_swapped(got, a) for a in anims[1:]]
        # つなげたとき: 描いた 1 コマ目の記録（読み込んだ一覧・画面で描いたときの記録があればそれ、無ければいまの 1 コマ目）
        first = (tuple(swaps.base) if swaps.base is not None else tuple(result.sources[0])) \
            if getattr(result, "joined", False) else None
        out["swaps_json"] = swaps_json(swaps, pal, first)   # 保存する <出力名>_<名前>_swaps.json の中身
        from .swap import base_warning
        msg = base_warning(swaps, result.sources if first is not None else None)
        if msg:
            out["swaps_base_warning"] = msg

    def sheet_entry(anim):
        png_name = F.sheet_file_name(base, anim)
        w, h = anim.size
        fps, fps_notes = fps_for_durations(anim.durations)
        return {"name": png_name, "png": encode_png(make_sheet(anim.frames)),
                       "piskel_name": png_name[:-4] + ".piskel", "piskel": piskel_text(anim.frames, png_name[:-4], fps),
                       "piskel_fps": fps, "piskel_notes": fps_notes,
                       "json": F.frames_sheet_json(anim, png_name, result.skipped, result.dot_size),
                       "frame_width": int(w), "frame_height": int(h), "frame_count": len(anim.frames),
                       "durations": [int(d) for d in anim.durations], "mirror": anim.name.endswith("_mirror"),
                       "offset_y": [int(v) for v in anim.info.get("offset_y", [])],
                "padding_y": [int(v) for v in anim.info.get("padding_y", (0, 0))]}
    from . import canvas as C
    anims = [C.pad_result(a, *canvas) for a in anims]   # キャンバスを倍数にそろえる（直し・差し替えの後。canvas.py）
    talks = [C.pad_result(a, *canvas) for a in talks]
    out["canvas"] = anims[0].info.get("canvas")
    out["sheets"] = [sheet_entry(a) for a in anims]
    if "talk_swaps" in out:   # 口パクがあるときだけ: しゃべりのシート（CLI と同じ名前・中身。反転は 2 つ目）
        out["talk_sheets"] = [sheet_entry(a) for a in talks]
    return out


def frames_sheets_for_web(request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（各シートの PNG は base64 の "png_base64"）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"order\": [1, 2, 3, 2]} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"シートの設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    res = frames_sheets(values)
    for s in res.get("sheets", []) + res.get("talk_sheets", []):
        s["png_base64"] = base64.b64encode(s.pop("png")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: 手直ししたシートの読み戻し（CLI の --readback） ----
# 中身は CLI と同じ `edit.readback()` なので、同じファイル・同じ設定なら CLI が書くファイルとバイト単位で同じになる。

def readback_for_web(edited: bytes, edited_name: str, info_text: str, original: bytes | None = None,
                     options_json: str = "{}") -> str:
    """直したシート（PNG か .piskel）と元の JSON（と、あれば直す前のシート）から作り直す。結果は JSON 文字列。

    {"ok": True, "made": 作れたか, "errors", "warnings", "notes", "report": 確かめた結果, "names": {種類: ファイル名},
    "files": [{"name", "base64"}...]（書く順）, "frame_width", "frame_height", "durations"}。設定の間違い: {"ok": False, "error"}。
    """
    from .edit import ReadbackOptions, readback
    try:
        values = json.loads(options_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("読み戻しの設定は {\"gif_scale\": 4} のような形で渡してください")
        opt = ReadbackOptions.from_dict(values)
        opt.validate()
    except (ValueError, TypeError) as e:
        return json.dumps({"ok": False, "error": f"読み戻しの設定に問題があります: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    try:
        res = readback(edited, edited_name, info_text, original, opt)
    except Exception as e:   # 思いがけないエラー（readback は自分では投げない）
        return json.dumps({"ok": False, "error": friendly_error(e), "detail": error_detail(e)}, ensure_ascii=False)
    out = {"ok": True, "made": res.made, "errors": res.errors, "warnings": res.warnings, "notes": res.notes,
           "report": res.report, "names": res.names,
           "files": [{"name": k, "base64": base64.b64encode(v).decode("ascii")} for k, v in res.files.items()]}
    if res.made:
        fr = res.info["frames"]
        anim = res.report.get("animation")
        if anim:   # まとめたシート: 直した動きのフレームだけを再生する（シートの中の位置 "rects" も渡す）
            fr = [fr[i] for i in res.info["animations"][anim]["frames"]]
            out["animation"] = anim
            out["rects"] = [[int(f["x"]), int(f["y"])] for f in fr]
        out.update({"frame_width": int(res.info["frame_width"]), "frame_height": int(res.info["frame_height"]),
                    "durations": [int(f["duration_ms"]) for f in fr]})
    return json.dumps(out, ensure_ascii=False)


# ---- ブラウザ UI（Pyodide）向け: 動きに名前を付けてまとめる（CLI の --anims） ----
# 変換は `anims.convert_anims()`（結果はコマの変換と同じ所に覚えるので、「ドットを直す」はそのまま使える）、
# シートは `anims.anims_outputs()`、GIF は `anims.anim_gif()`。CLI と同じ関数なので、同じ設定ならバイト単位で同じになる。

def _anims_config(values, settings=None):
    from .anims import config_from_dict
    d = dict(values or {})
    if settings is not None:
        d["settings"] = settings
    return config_from_dict(d)


def anims_convert_bytes(files: list, request: dict) -> dict:
    """動きごとのファイルを変換してそろえる（例外は投げない）。`files` は [(ファイルの名前, バイト列), ...]、
    `request` は {"settings": 変換の設定, "anims": 動きをまとめる設定（anims.json と同じ形。settings は無くてよい）}。
    結果の形は `frames_convert_bytes()` と同じ（"anims": 動きごとの {"name", "first", "count", "files"} を足す）。"""
    from . import frames as F
    from .anims import convert_anims
    from .grid import GridNotFoundError
    try:
        unknown = sorted(set(request) - {"settings", "anims"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        settings = SpriteSettings.from_dict(dict(request.get("settings") or {}))
        settings.validate()
        cfg = _anims_config(request.get("anims"), dict(request.get("settings") or {}))
        images = []
        for fname, fdata in files:
            try:
                images.append((str(fname), decode_image(bytes(fdata))))
            except Exception as e:
                raise ValueError(f"{fname}: {friendly_error(e)}") from None
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    try:
        res = convert_anims(images, settings, cfg)
    except (F.FramesError, GridNotFoundError) as e:
        return {"ok": True, "made": False, "error": str(e), "code": getattr(e, "code", "grid_not_found"),
                "skipped_json": F.frames_skipped_json(e, cfg.name), "boxes": None}
    except ValueError as e:
        return {"ok": False, "error": f"動きをまとめられませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    token = int(_FRAMES_CACHE.get("token", 0)) + 1
    _FRAMES_PREV.clear()
    _FRAMES_PREV.update(_FRAMES_CACHE)
    _FRAMES_CACHE.clear()
    _FRAMES_CACHE.update({"token": token, "result": res, "base": res, "fs": F.FrameSettings(align=cfg.align, grid=cfg.grid),
                          "finished": res, "edits": None, "anims_cfg": cfg})
    strip = F.source_strip(res)
    opaque = strip[strip[..., 3] > 0][:, :3]
    w, h = res.size
    out = {"ok": True, "made": True, "token": token,
           "frame_width": int(w), "frame_height": int(h), "frame_count": len(res.frames),
           "boxes": [[b.x0, b.y0, b.x1, b.y1] for b in res.boxes],
           "dot_size": round(float(res.dot_size[0]), 3),
           "colors": int(len(np.unique(opaque.reshape(-1, 3), axis=0))) if len(opaque) else 0,
           "notes": list(res.notes), "steps": [r.description for r in res.steps],
           "aligned_png": encode_png(np.concatenate(res.aligned, 1)),
           "files": [dict(f.info(), boxes=[[b.x0, b.y0, b.x1, b.y1] for b in f.boxes]) for f in res.files],
           "sources": [[n, int(k)] for n, k in res.sources], "joins": list(res.joins or []),
           "offsets": [[int(x), int(y)] for x, y in res.offsets],
           "mirror_offsets": [[int(x), int(y)] for x, y in res.mirror_offsets()],
           "shift_y": None, "edits_origin": True, "file_starts": [f.first for f in res.files[1:]],
           "anims": [dict(r) for r in res.anims]}
    out.update(_frames_view(res))
    return out


def anims_convert_files_for_web(datas: list, names: list, request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（PNG は base64）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"settings\": {...}, \"anims\": {...}} のような形で渡してください")
        if len(datas) != len(names) or not datas:
            raise ValueError("ファイルの名前とバイト列の数が合いません")
    except ValueError as e:
        return json.dumps({"ok": False, "error": friendly_error(ValueError(f"設定を読めません: {e}")),
                           "detail": error_detail(e)}, ensure_ascii=False)
    return json.dumps(_web_b64(anims_convert_bytes(list(zip(names, datas)), values)), ensure_ascii=False)


def anims_sheets(request: dict, result=None) -> dict:
    """まとめたシート（PNG・JSON・反転・動きごとの .piskel）を作る（例外は投げない）。`request` は
    {"token", "base": 出力名, "anims": 動きをまとめる設定（変換したときと同じ動き・ファイル・縦のそろえ方・上げ下げ）,
    "settings": 変換の設定（保存する設定のファイルに入れる）}。
    作れた: {"ok": True, "sheets": [{"name", "png", "json", "frame_width", "frame_height", "frame_count", "columns", "rows"}...]
    （反転は 2 つ目）, "piskels": [{"anim", "name", "text"}], "config_json": 保存する設定のファイル, "config_name"}。"""
    from . import frames as F
    from .anims import anims_outputs, config_json
    try:
        unknown = sorted(set(request) - {"token", "base", "anims", "settings"})
        if unknown:
            raise ValueError(f"知らない設定です: {', '.join(unknown)}")
        if result is None:
            if not _FRAMES_CACHE or request.get("token") != _FRAMES_CACHE.get("token") or "anims_cfg" not in _FRAMES_CACHE:
                raise ValueError("動きをまとめた変換の結果がありません（もう一度変換してください）")
            result = _FRAMES_CACHE["result"]
            done = _FRAMES_CACHE["anims_cfg"]
        else:
            done = None
        base = str(request.get("base") or "sprite")
        cfg = _anims_config(request.get("anims"), request.get("settings"))
        if done is not None:
            same = [(a.name, a.files, a.align_y, a.shift_y) for a in cfg.anims] == \
                [(a.name, a.files, a.align_y, a.shift_y) for a in done.anims] and \
                (cfg.align, cfg.grid, cfg.match_height) == (done.align, done.grid, done.match_height)
            if not same:
                raise ValueError("動き・ファイル・そろえ方が変換したときと違います（「② 変換する」をもう一度押してください）")
        made = anims_outputs(result, cfg, base)
    except ValueError as e:
        return {"ok": False, "error": f"シートの設定に問題があり、作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    files = made["files"]
    out: dict = {"ok": True, "sheets": [], "piskels": []}
    for sh in made["sheets"]:
        w, h = sh.size
        out["sheets"].append({"name": sh.image_name, "png": files[sh.image_name],
                              "json": files[sh.image_name[:-4] + ".json"].decode("utf-8"),
                              "frame_width": int(w), "frame_height": int(h), "frame_count": len(sh.frames),
                              "columns": sh.columns, "rows": len(sh.anims), "mirror": sh.name.endswith("_mirror")})
    for a in cfg.anims:
        from .anims import anim_file_stem
        name = anim_file_stem(made["sheets"][0].info, a.name) + ".piskel"
        out["piskels"].append({"anim": a.name, "name": name, "text": files[name].decode("ascii")})
    warn = [w for a in cfg.anims for w in [F.flip_warning(a.order)] if w]
    if warn:
        out["flip_warning"] = warn[0]
    out["config_json"] = config_json(cfg)
    out["config_name"] = f"{base}_{cfg.name}.anims.json"
    return out


def anims_sheets_for_web(request_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。結果は JSON 文字列（各シートの PNG は base64 の "png_base64"）。"""
    try:
        values = json.loads(request_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"anims\": {...}} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"シートの設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    res = anims_sheets(values)
    for s in res.get("sheets", []):
        s["png_base64"] = base64.b64encode(s.pop("png")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)


def anims_gif_bytes(sheet_png: bytes, sheet_json_text: str, anim: str, options: dict | None = None) -> dict:
    """まとめたシートの 1 つの動きの GIF（例外は投げない。CLI の --anims --gif と同じ）。"""
    from .anims import anim_gif
    from .motion.gif import DEFAULT_SCALE
    values = dict(options or {})
    try:
        unknown = sorted(set(values) - {"scale", "background"})
        if unknown:
            raise ValueError(f"知らない GIF の設定です: {', '.join(unknown)}")
        info = json.loads(sheet_json_text)
        if not isinstance(info, dict) or not isinstance(info.get("animations"), dict):
            raise ValueError("まとめたシートの情報（animations のある JSON）を渡してください")
        if anim not in info["animations"]:
            raise ValueError(f"シートの情報に動き「{anim}」がありません")
        name, g = anim_gif(decode_image(sheet_png), info, anim, values.get("scale", DEFAULT_SCALE), values.get("background"))
    except ValueError as e:
        return {"ok": False, "error": f"GIF を作れませんでした: {e}", "detail": error_detail(e)}
    except Exception as e:
        return {"ok": False, "error": friendly_error(e), "detail": error_detail(e)}
    out = {"ok": True, "gif": g.data, "name": name, "loop": bool(info["animations"][anim].get("loop", True))}
    out.update(g.summary())
    return out


def anims_gif_for_web(sheet_png: bytes, sheet_json_text: str, anim: str, options_json: str = "{}") -> str:
    try:
        values = json.loads(options_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("GIF の設定は {\"scale\": 4} のような形で渡してください")
    except ValueError as e:
        return json.dumps({"ok": False, "error": f"GIF の設定を読めません: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    res = anims_gif_bytes(sheet_png, sheet_json_text, anim, values)
    if "gif" in res:
        res["gif_base64"] = base64.b64encode(res.pop("gif")).decode("ascii")
    return json.dumps(res, ensure_ascii=False)
