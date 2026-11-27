"""Still App の入口（CLI とブラウザ版で共通）。1 枚絵を、決まった色数・格子の本物のドット絵にそろえる。

工程: 読み込み（透明は黒で埋める）→ ドットに戻す（グリッド検出か --width）→ 大きさの上限を確かめる
→ 減色（4 ビットの丸め・いちばん暗い色を取っておく）→ キャンバス（--canvas のときだけ余白を足す）→ PNG（インデックスカラー）と JSON。

`--mode`（`MODES`）は初めの設定の組み合わせ（プリセット）で、強制ではない。個別の設定で上書きできる。
同じ入力・同じ設定からは、CLI でもブラウザ版でも同じバイト列の PNG と JSON ができる（`convert_still_bytes()`）。
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import platform
import re
import struct
import sys
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..sprite.image import to_rgba
from . import colors as colors_mod
from . import dots as dots_mod

APP = "still"
FORMAT_VERSION = 1
# プリセット。値を変えると出力が変わる（足すのはよい）
MODES = {
    "pc98": {"label": "PC-98 風（16 色・4096 色・640×400 まで）",
             "colors": 16, "four_bit": True, "max_size": (640, 400), "keep_darkest": True},
}
DEFAULT_MODE = "pc98"
MAX_SIDE = 4096   # 上限・キャンバスに指定できる 1 辺の最大


class StillError(ValueError):
    """出力を作らずに止めるとき（利用者向けの日本語）。`candidates` に --width の候補を持つことがある。"""

    def __init__(self, message: str, candidates: list[dict] | None = None, kind: str = "error"):
        super().__init__(message)
        self.candidates = list(candidates or [])
        self.kind = kind


def parse_size(text, name: str = "大きさ") -> tuple[int, int]:
    """'640x400' や [640, 400] を (幅, 高さ) にする。"""
    if isinstance(text, (list, tuple)):
        parts = list(text)
    else:
        parts = str(text).lower().replace("×", "x").replace("*", "x").split("x")
    try:
        if len(parts) != 2:
            raise ValueError
        w, h = (int(str(p).strip()) for p in parts)
    except ValueError:
        raise ValueError(f"{name}は 幅x高さ（例 640x400）で指定してください（{text}）") from None
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        raise ValueError(f"{name}の幅・高さは 1〜{MAX_SIDE} で指定してください（{w}x{h}）")
    return w, h


@dataclass
class StillSettings:
    """変換の設定。既定は `--mode pc98` と同じ。"""

    mode: str = DEFAULT_MODE
    colors: int = MODES[DEFAULT_MODE]["colors"]
    four_bit: bool = MODES[DEFAULT_MODE]["four_bit"]
    max_size: tuple[int, int] = MODES[DEFAULT_MODE]["max_size"]
    width: int | None = None                 # 横のマス数（None ならグリッド検出）
    canvas: tuple[int, int] | None = None    # 余白を足して、この大きさにする
    keep_darkest: bool = MODES[DEFAULT_MODE]["keep_darkest"]

    @classmethod
    def from_mode(cls, mode: str = DEFAULT_MODE, **overrides) -> "StillSettings":
        """プリセットの値に、None でない overrides を重ねた設定。"""
        if mode not in MODES:
            raise ValueError(f"--mode は {', '.join(MODES)} のどれかです（{mode}）")
        base = {k: v for k, v in MODES[mode].items() if k != "label"}
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(mode=mode, **base)

    @classmethod
    def from_dict(cls, values: dict) -> "StillSettings":
        """画面（JSON）の値から。無い値・null はプリセットの値。"""
        known = {f.name for f in dataclasses.fields(cls)} - {"mode"}
        extra = set(values) - known - {"mode"}
        if extra:
            raise ValueError(f"知らない設定です: {', '.join(sorted(extra))}")
        v = {k: values.get(k) for k in known}
        for k in ("max_size", "canvas"):
            if v[k] is not None:
                v[k] = parse_size(v[k], "大きさの上限" if k == "max_size" else "キャンバス")
        for k in ("colors", "width"):
            if v[k] is not None:
                if isinstance(v[k], bool) or not isinstance(v[k], (int, float)) or int(v[k]) != v[k]:
                    raise ValueError(f"{k} は整数で指定してください（{v[k]}）")
                v[k] = int(v[k])
        for k in ("four_bit", "keep_darkest"):
            if v[k] is not None and not isinstance(v[k], bool):
                raise ValueError(f"{k} は true か false で指定してください（{v[k]}）")
        s = cls.from_mode(values.get("mode") or DEFAULT_MODE, **v)
        s.validate()
        return s

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"--mode は {', '.join(MODES)} のどれかです（{self.mode}）")
        if not colors_mod.MIN_COLORS <= self.colors <= colors_mod.MAX_COLORS:
            raise ValueError(f"色数は {colors_mod.MIN_COLORS}〜{colors_mod.MAX_COLORS} で指定してください（{self.colors}）")
        self.max_size = parse_size(self.max_size, "大きさの上限")
        if self.width is not None and not 1 <= self.width <= MAX_SIDE:
            raise ValueError(f"横のマス数は 1〜{MAX_SIDE} で指定してください（{self.width}）")
        if self.canvas is not None:
            self.canvas = parse_size(self.canvas, "キャンバス")

    def to_json(self) -> dict:
        return {"mode": self.mode, "colors": self.colors, "four_bit": self.four_bit,
                "max_size": list(self.max_size), "width": self.width,
                "canvas": None if self.canvas is None else list(self.canvas), "keep_darkest": self.keep_darkest}


@dataclass
class StillResult:
    index: np.ndarray                # 高さ × 幅（uint8）。パレットの番号
    palette: np.ndarray              # k × 3（uint8）。暗い順
    counts: np.ndarray
    dots: dots_mod.Dots
    reduced: colors_mod.Reduced
    settings: StillSettings
    input_size: tuple[int, int]
    canvas_offset: tuple[int, int] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.index.shape[1]), int(self.index.shape[0])

    def rgb(self) -> np.ndarray:
        return self.palette[self.index]


def _hex(c) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in c))


def convert_still(rgba: np.ndarray, settings: StillSettings | None = None) -> StillResult:
    """RGBA（か RGB）の配列を変換する。止めるときは `StillError`。"""
    s = settings or StillSettings()
    s.validate()
    rgb = dots_mod.flatten_black(rgba)
    h, w = rgb.shape[:2]
    max_w, max_h = s.max_size
    if s.width is None:
        try:
            dots = dots_mod.grid_dots(rgb, s.max_size)
        except dots_mod.StillGridError as e:
            raise StillError(str(e), e.candidates, "grid") from e
    else:
        if s.width > w:
            raise StillError(f"横のマス数（{s.width}）が、画像の幅（{w}px）より多くなっています。",
                             dots_mod.size_candidates(w, h, max_w, max_h), "width")
        dots = dots_mod.width_dots(rgb, s.width)
    dw, dh = dots.size
    if dw > max_w or dh > max_h:
        how = "グリッド検出" if dots.method == "grid" else f"--width {s.width}"
        aw, ah = int(dots.xs[-1] - dots.xs[0]), int(dots.ys[-1] - dots.ys[0])
        raise StillError(f"{how}で {dw}×{dh} ドットになり、大きさの上限 {max_w}×{max_h} を超えたので、出力しませんでした。"
                         "--width で横のマス数を小さくするか、--max-size で上限を変えてください。",
                         dots_mod.size_candidates(aw, ah, max_w, max_h), "size")
    if s.canvas is not None:
        cw, ch = s.canvas
        if cw < dw or ch < dh:
            raise StillError(f"キャンバス {cw}×{ch} が、絵（{dw}×{dh} ドット）より小さくなっています。", [], "canvas")
        if cw > max_w or ch > max_h:
            raise StillError(f"キャンバス {cw}×{ch} が、大きさの上限 {max_w}×{max_h} を超えています。", [], "canvas")
    red = colors_mod.reduce_colors(dots.rgb, s.colors, four_bit=s.four_bit, keep_darkest=s.keep_darkest)
    index, palette, counts = red.index, red.palette, red.counts
    warnings = []
    if red.merged:
        warnings.append(f"選んだ色を 4096 色に丸めたら {red.merged} 色が重なり、{red.chosen - red.merged} 色になりました"
                        f"（上限 {s.colors} 色）")
    offset = None
    if s.canvas is not None:
        cw, ch = s.canvas
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        offset = (ox, oy)
        if (cw, ch) != (dw, dh):
            fill = 0   # パレットは暗い順なので 0 番がいちばん暗い色（新しい色は作らない）
            big = np.full((ch, cw), fill, np.uint8)
            big[oy:oy + dh, ox:ox + dw] = index
            index = big
            counts = counts.copy()
            counts[fill] += cw * ch - dw * dh
    return StillResult(index, palette, counts, dots, red, s, (w, h), offset, warnings)


# ---- ファイルの形 ----

def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bit_depth(colors: int) -> int:
    """パレットの色数に足りる、いちばん小さいビット深度（1・2・4・8）。"""
    return 1 if colors <= 2 else 2 if colors <= 4 else 4 if colors <= 16 else 8


def encode_indexed_png(index: np.ndarray, palette: np.ndarray) -> bytes:
    """インデックスカラーの PNG（パレットは palette の色だけ。ビット深度は色数に合わせて 1・2・4・8）。

    Pillow で書くと、Pillow の版で圧縮のしかたが変わり（手元の 13.0 と Pyodide の 10.2 で IDAT が違った）、
    CLI とブラウザ版でバイトがそろわないので、標準ライブラリの zlib で自分で書く（行のフィルタはなし、圧縮は 9）。
    """
    idx = np.ascontiguousarray(index, dtype=np.uint8)
    pal = np.asarray(palette, np.uint8).reshape(-1, 3)
    h, w = idx.shape
    if not 1 <= len(pal) <= 256:
        raise ValueError(f"パレットの色数は 1〜256 です（{len(pal)}）")
    if idx.size and int(idx.max()) >= len(pal):
        raise ValueError("パレットに無い番号があります")
    depth = png_bit_depth(len(pal))
    per = 8 // depth
    pad = (-w) % per
    rows = np.concatenate([idx, np.zeros((h, pad), np.uint8)], axis=1) if pad else idx
    rows = rows.reshape(h, -1, per).astype(np.uint16)
    shifts = (8 - depth * (np.arange(per) + 1)).astype(np.uint16)
    packed = (rows << shifts).sum(axis=-1).astype(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), packed], axis=1).tobytes()   # 各行の先頭はフィルタ 0
    ihdr = struct.pack(">IIBBBBB", w, h, depth, 3, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"PLTE", pal.tobytes())
            + _png_chunk(b"IDAT", zlib.compress(raw, 9)) + _png_chunk(b"IEND", b""))


def result_info(res: StillResult, source: str = "") -> dict:
    """JSON に書く中身。"""
    s = res.settings
    pal = []
    for i, (c, n) in enumerate(zip(res.palette, res.counts)):
        c4 = colors_mod.to_4bit(c)
        exact = bool(np.all(c4 * 17 == c))
        pal.append({"index": i, "rgb4": [int(v) for v in c4] if exact else None,
                    "rgb": [int(v) for v in c], "hex": _hex(c), "count": int(n)})
    w, h = res.size
    canvas = None
    if res.canvas_offset is not None:
        canvas = {"width": w, "height": h, "offset": list(res.canvas_offset),
                  "fill": {"index": 0, "hex": _hex(res.palette[0])}}
    red = res.reduced
    return {
        "app": APP, "version": FORMAT_VERSION, "source": source,
        "input": {"width": res.input_size[0], "height": res.input_size[1]},
        "width": w, "height": h,
        "picture": {"width": res.dots.size[0], "height": res.dots.size[1]},
        "pixel_aspect": "1:1",
        "dots": res.dots.describe(),
        "canvas": canvas,
        "colors": {"limit": s.colors, "used": len(res.palette), "chosen": red.chosen, "merged_by_4bit": red.merged,
                   "darkest_kept": None if red.darkest is None else _hex(red.darkest)},
        "palette": pal,
        "settings": s.to_json(),
        "warnings": list(res.warnings),
    }


_NUMBER_LIST = re.compile(r"\[\s+(-?[\d.]+(?:,\s+-?[\d.]+)*)\s+\]")


def info_json_text(info: dict) -> str:
    """JSON ファイルの中身（字下げ 2。数だけの並びは 1 行にする）。"""
    text = json.dumps(info, ensure_ascii=False, indent=2)
    text = _NUMBER_LIST.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    return text + "\n"


def decode_image(data: bytes) -> np.ndarray:
    """PNG・JPEG などのバイト列を RGBA の配列にする（Sprite App と同じ読み方）。"""
    with Image.open(io.BytesIO(bytes(data))) as img:
        img.load()
        return to_rgba(img)


def convert_still_bytes(data: bytes, source: str = "", settings: StillSettings | dict | None = None) -> dict:
    """画像ファイルのバイト列から、PNG と JSON のバイト列を作る（CLI とブラウザ版で共通。例外は投げる）。

    戻り値: {"png": bytes, "json": str, "info": dict, "result": StillResult}
    """
    if not isinstance(settings, StillSettings):
        settings = StillSettings.from_dict(dict(settings or {}))
    res = convert_still(decode_image(data), settings)
    info = result_info(res, source)
    return {"png": encode_indexed_png(res.index, res.palette), "json": info_json_text(info), "info": info,
            "result": res}


# ---- ブラウザ版（Pyodide）向け ----

def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, StillError):
        return str(exc)
    if isinstance(exc, (OSError, Image.DecompressionBombError)) or type(exc).__name__ == "UnidentifiedImageError":
        return "画像として読み込めませんでした。PNG・JPEG の画像ファイルを選んでください。"
    if isinstance(exc, ValueError):
        return f"設定または画像に問題があり、変換できませんでした: {exc}"
    return f"思いがけないエラーで変換できませんでした（{type(exc).__name__}: {exc}）"


def error_detail(exc: BaseException) -> str:
    import PIL
    env = (f"Python {platform.python_version()} ({sys.platform}) / NumPy {np.__version__} / Pillow {PIL.__version__} / "
           f"intp {np.dtype(np.intp).itemsize * 8} ビット")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{env}\n{tb}".rstrip()


def still_for_web(data: bytes, name: str = "", settings_json: str = "{}") -> str:
    """画面から呼ぶ形。結果は JSON 文字列（PNG は base64、JSON ファイルの中身は文字列のまま）。例外は投げない。"""
    try:
        values = json.loads(settings_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("設定は {\"colors\": 16} のような形で渡してください")
        out = convert_still_bytes(data, name, values)
    except Exception as e:  # 画面にそのまま出すので、どんなエラーでも言葉にして返す
        res = {"ok": False, "error": friendly_error(e), "kind": getattr(e, "kind", "error"),
               "candidates": list(getattr(e, "candidates", []) or [])}
        if not isinstance(e, StillError):
            res["detail"] = error_detail(e)
        return json.dumps(res, ensure_ascii=False)
    info = out["info"]
    return json.dumps({"ok": True, "png_base64": base64.b64encode(out["png"]).decode("ascii"), "json_text": out["json"],
                       "width": info["width"], "height": info["height"], "palette": info["palette"],
                       "dots": info["dots"], "colors": info["colors"], "canvas": info["canvas"],
                       "warnings": info["warnings"]}, ensure_ascii=False)


def spec() -> dict:
    """画面に埋め込む設定の選択肢と既定値（CLI の既定と同じ）。"""
    return {"modes": {k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}}
                      for k, v in MODES.items()},
            "default_mode": DEFAULT_MODE,
            "colors": {"min": colors_mod.MIN_COLORS, "max": colors_mod.MAX_COLORS},
            "max_side": MAX_SIDE}
