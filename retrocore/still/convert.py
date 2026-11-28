"""Still App の入口（CLI とブラウザ版で共通）。1 枚絵を、決まった色数・格子の本物のドット絵にそろえる。

工程は 1 本（項目 `ITEMS` の順）: 読み込み（透明は黒で埋める。`orient` のとき写真の向き（EXIF）を直す）
→ ドットに戻す（`find_grid` ならグリッド検出、でなければ横 `width` マス）→ 色の塊をまとめる（`smooth` が off でないとき。`photo.py`）
→ 大きさの上限を確かめる → 減色（4 ビットの丸め・いちばん暗い色を取っておく）→ 似た色をまとめる（`merge_similar`）
→ ディザを作る（`dither`。`dither.py`）→ 孤立したドットをそろえる（`tidy_dots`）→ 模様を整える（`tidy_pattern`）→ 輪郭の線（`outline`）→ キャンバス（`canvas`）
→ PNG（インデックスカラー）と JSON。

`--mode`（`MODES`）は初めの設定（項目の値の組み合わせ）で、強制ではない。どの項目も個別に変えられ、変えたときは
JSON に `preset`（名前・変えた項目・項目の値すべて）が入る（変えていなければ入れず、前の版と同じバイト列）。
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
from . import dither as dither_mod
from . import dots as dots_mod
from . import lone as lone_mod
from . import outline as outline_mod
from . import pattern as pattern_mod
from . import photo as photo_mod

APP = "still"
FORMAT_VERSION = 1
# 項目（工程の順）。初めの設定（MODES）は、この項目の値の組み合わせ。JSON の `preset.values` もこの順
ITEMS = ("orient", "find_grid", "width", "max_size", "smooth", "colors", "four_bit", "keep_darkest", "merge_similar",
         "dither", "tidy_dots", "tidy_pattern", "outline", "canvas")
_OFF = {"dither": "off", "tidy_pattern": False, "outline": False, "canvas": None}
# 初めの設定（プリセット）。値を変えると出力が変わる（足すのはよい）。どの値も、項目（CLI のオプション・画面の
# 「必要なときに開く」）で 1 つずつ変えられる
MODES = {
    "pc98": {"label": "PC-98 風（16 色・4096 色・640×400 まで）",
             "colors": 16, "four_bit": True, "max_size": (640, 400), "keep_darkest": True, "merge_similar": False,
             "orient": False, "find_grid": True, "width": None, "smooth": "off", "tidy_dots": False, **_OFF},
    # 今風（最近のインディーゲームの背景のようなドット絵の手直しの出発点）。色が多いと AI のにじみの色が残りやすいので、
    # 見た目でほぼ同じ色はまとめる（色数は上限で、少なくなってよい）
    "modern": {"label": "今風（32 色・丸めない・1920×1080 まで・似た色をまとめる）",
               "colors": 32, "four_bit": False, "max_size": (1920, 1080), "keep_darkest": True, "merge_similar": True,
               "orient": False, "find_grid": True, "width": None, "smooth": "off", "tidy_dots": False, **_OFF},
    # 写真の下書き（自分で撮った写真を、手直しの出発点になる色を塗り分けた下絵にする）。写真の向きを直し、マス目は探さずに
    # 横 320 マスに分け、色の塊をまとめる。減色は今風と同じ。写真は浮いた点が多いので、孤立したドットもそろえる。
    # 似た色はまとめない（写真では近い色が段として要る。まとめると photo03 の 16 色が 11 色に減り、空の段が粗くなった）
    "photo": {"label": "写真の下書き（横 320・32 色・色の塊をまとめる・孤立したドットをそろえる）",
              "colors": 32, "four_bit": False, "max_size": (1920, 1080), "keep_darkest": True, "merge_similar": False,
              "tidy_dots": True, "width": photo_mod.DEFAULT_WIDTH, "smooth": photo_mod.DEFAULT_SMOOTH,
              "orient": True, "find_grid": False, **_OFF},
}
SMOOTH_LEVELS = tuple(photo_mod.SMOOTH)
SMOOTH_CHOICES = ("off",) + SMOOTH_LEVELS   # 色の塊をまとめる: しない・弱・中・強
# ブラウザ版の色数の欄で、候補として出す数（CLI の --colors と同じく 2〜256 のどれでも入れられる）
MODERN_COLOR_CHOICES = (16, 24, 32, 48, 64, 96, 128)
DITHER_CHOICES = dither_mod.CHOICES   # ディザを作る: しない・2×2・4×4
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


def _json_value(v):
    return list(v) if isinstance(v, tuple) else v


@dataclass
class StillSettings:
    """変換の設定。既定は `--mode pc98` と同じ。`mode` は初めの設定の名前で、ほかの値はすべて項目（`ITEMS`）。"""

    mode: str = DEFAULT_MODE
    colors: int = MODES[DEFAULT_MODE]["colors"]
    four_bit: bool = MODES[DEFAULT_MODE]["four_bit"]
    max_size: tuple[int, int] = MODES[DEFAULT_MODE]["max_size"]
    width: int | None = None                 # 横のマス数（マス目を探すときは None）
    canvas: tuple[int, int] | None = None    # 余白を足して、この大きさにする
    keep_darkest: bool = MODES[DEFAULT_MODE]["keep_darkest"]
    tidy_pattern: bool = False               # ざらついた範囲を 2 色の模様に整える（どの初めの設定でも切）
    merge_similar: bool = MODES[DEFAULT_MODE]["merge_similar"]   # 似た色をまとめる（今風だけ入）
    dither: str = "off"                      # ディザを作る（off・2x2・4x4。どの初めの設定でも off）
    tidy_dots: bool = False                  # 孤立したドットをそろえる（写真の下書きだけ入）
    smooth: str | None = None                # 色の塊をまとめる強さ（off・weak・medium・strong。None は mode の既定）
    outline: bool = False                    # 輪郭の線を引く（どの初めの設定でも切）
    orient: bool | None = None               # 写真の向き（EXIF）を直して読む（None は mode の既定。写真の下書きだけ入）
    find_grid: bool | None = None            # マス目を探す（False なら横 width マスに分ける。None は width から決める）

    def __post_init__(self):
        # 直接作ったとき（StillSettings(width=40) など）の、決まっていない値を埋める（前の版と同じ意味になるように）
        photo = self.mode == "photo"
        if self.find_grid is None:
            self.find_grid = self.width is None and not photo
        if self.orient is None:
            self.orient = photo
        if self.smooth is None:
            self.smooth = photo_mod.DEFAULT_SMOOTH if photo else "off"

    @classmethod
    def from_mode(cls, mode: str = DEFAULT_MODE, **overrides) -> "StillSettings":
        """初めの設定の値に、None でない overrides を重ねた設定。

        横のマス数（width）だけを渡したときは、マス目を探さずにその数で分ける（前の版と同じ）。
        マス目を探す（find_grid=True）だけを渡したときは、初めの設定の横のマス数を使わない。
        """
        if mode not in MODES:
            raise ValueError(f"--mode は {', '.join(MODES)} のどれかです（{mode}）")
        base = {k: MODES[mode][k] for k in ITEMS}
        ov = {k: v for k, v in overrides.items() if v is not None}
        if "width" in ov and "find_grid" not in ov:
            ov["find_grid"] = False
        if ov.get("find_grid") is True and "width" not in ov:
            ov["width"] = None
        base.update(ov)
        return cls(mode=mode, **base)

    @classmethod
    def from_dict(cls, values: dict) -> "StillSettings":
        """画面（JSON）の値から。無い値・null は初めの設定の値。"""
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
        for k in ("four_bit", "keep_darkest", "tidy_pattern", "merge_similar", "tidy_dots", "outline", "orient",
                  "find_grid"):
            if v[k] is not None and not isinstance(v[k], bool):
                raise ValueError(f"{k} は true か false で指定してください（{v[k]}）")
        if v["smooth"] is not None and not isinstance(v["smooth"], str):
            raise ValueError(f"smooth は {', '.join(SMOOTH_CHOICES)} のどれかです（{v['smooth']}）")
        if v["dither"] is not None and not isinstance(v["dither"], str):
            raise ValueError(f"dither は {', '.join(DITHER_CHOICES)} のどれかです（{v['dither']}）")
        s = cls.from_mode(values.get("mode") or DEFAULT_MODE, **v)
        s.validate()
        return s

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"--mode は {', '.join(MODES)} のどれかです（{self.mode}）")
        if not colors_mod.MIN_COLORS <= self.colors <= colors_mod.MAX_COLORS:
            raise ValueError(f"色数は {colors_mod.MIN_COLORS}〜{colors_mod.MAX_COLORS} で指定してください（{self.colors}）")
        self.max_size = parse_size(self.max_size, "大きさの上限")
        if self.find_grid:
            if self.width is not None:
                raise ValueError(f"マス目を探す（--grid）ときは、横のマス数（--width {self.width}）は使えません。"
                                 "一緒には使えないので、どちらか 1 つにしてください")
        elif self.width is None:
            if self.mode == "photo":   # 前の版と同じ（写真の下書きで横のマス数が空なら 320）
                self.width = photo_mod.DEFAULT_WIDTH
            else:
                raise ValueError("マス目を探さないときは、横のマス数（--width）を指定してください")
        if self.width is not None and not 1 <= self.width <= MAX_SIDE:
            raise ValueError(f"横のマス数は 1〜{MAX_SIDE} で指定してください（{self.width}）")
        if self.canvas is not None:
            self.canvas = parse_size(self.canvas, "キャンバス")
        if self.smooth not in SMOOTH_CHOICES:
            raise ValueError(f"色の塊のまとめの強さ（--smooth）は {', '.join(SMOOTH_CHOICES)} のどれかです（{self.smooth}）")
        if self.dither not in DITHER_CHOICES:
            raise ValueError(f"ディザ（--dither）は {', '.join(DITHER_CHOICES)} のどれかです（{self.dither}）")

    def values(self) -> dict:
        """項目の値すべて（工程の順。JSON の `preset.values`）。"""
        return {k: _json_value(getattr(self, k)) for k in ITEMS}

    def changed(self) -> list[str]:
        """初めの設定から変えた項目（工程の順）。"""
        base = StillSettings.from_mode(self.mode)
        base.validate()
        mine, ref = self.values(), base.values()
        return [k for k in ITEMS if mine[k] != ref[k]]

    def preset_info(self) -> dict | None:
        """初めの設定から変えたときの記録（変えていなければ None。JSON は前の版と同じバイト列のまま）。"""
        changed = self.changed()
        if not changed:
            return None
        ref = StillSettings.from_mode(self.mode)
        ref.validate()
        mine, base = self.values(), ref.values()
        return {"name": self.mode, "changed": {k: {"preset": base[k], "used": mine[k]} for k in changed},
                "values": mine}

    def to_json(self) -> dict:
        """JSON の `settings`（前の版と同じ形。項目の値すべては `preset.values`）。"""
        out = {"mode": self.mode, "colors": self.colors, "four_bit": self.four_bit,
               "max_size": list(self.max_size), "width": self.width,
               "canvas": None if self.canvas is None else list(self.canvas), "keep_darkest": self.keep_darkest}
        if self.tidy_pattern:   # 切のときは書かない（前の版と同じバイト列にするため）
            out["tidy_pattern"] = True
        if self.mode in ("modern", "photo"):   # 今風・写真の下書きのときだけ書く（98 モードは前と同じバイト列）
            out["merge_similar"] = self.merge_similar
        if self.dither != "off":   # off のときは書かない（前と同じバイト列にするため）
            out["dither"] = self.dither
        if self.tidy_dots or MODES[self.mode].get("tidy_dots"):   # 切のときは書かない（写真の下書きでは切も書く）
            out["tidy_dots"] = self.tidy_dots
        if self.mode == "photo":
            out["smooth"] = self.smooth
        if self.outline or self.mode == "photo":   # 98・今風モードでは、切のときは書かない
            out["outline"] = self.outline
        return out


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
    tidied: pattern_mod.Tidied | None = None   # --tidy-pattern のときだけ
    similar: colors_mod.Similar | None = None  # 似た色をまとめたときだけ
    lone: lone_mod.Lone | None = None          # --tidy-dots のときだけ
    outline: outline_mod.Outline | None = None  # --outline のときだけ
    photo: photo_mod.PhotoWork | None = None   # 写真の下書きのときだけ
    orientation: int = 1                       # 写真の向き（EXIF の Orientation。写真の下書きで読んだときだけ 1 以外）
    dither: dither_mod.Dither | None = None    # --dither が off でないときだけ

    def overlay(self) -> np.ndarray:
        """整えた範囲の見える化（出力の大きさの RGBA。表示だけに使う）。整えていなければ全部透明。"""
        h, w = self.index.shape
        label = np.full((h, w), -1, np.intp)
        if self.tidied is not None:
            ox, oy = self.canvas_offset or (0, 0)
            th, tw = self.tidied.label.shape
            label[oy:oy + th, ox:ox + tw] = self.tidied.label
        return pattern_mod.overlay_rgba(label)

    def dots_overlay(self) -> np.ndarray:
        """孤立したドットを置き換えた所の見える化（出力の大きさの RGBA。表示だけ）。置き換えていなければ全部透明。"""
        h, w = self.index.shape
        m = np.zeros((h, w), bool)
        if self.lone is not None:
            ox, oy = self.canvas_offset or (0, 0)
            th, tw = self.lone.changed.shape
            m[oy:oy + th, ox:ox + tw] = self.lone.changed
        return lone_mod.overlay_rgba(m)

    def dither_overlay(self) -> np.ndarray:
        """網目を入れたドットの見える化（出力の大きさの RGBA。表示だけ）。入れていなければ全部透明。"""
        h, w = self.index.shape
        m = np.zeros((h, w), bool)
        ar = np.zeros((h, w), bool)
        if self.dither is not None:
            ox, oy = self.canvas_offset or (0, 0)
            th, tw = self.dither.changed.shape
            m[oy:oy + th, ox:ox + tw] = self.dither.changed
            ar[oy:oy + th, ox:ox + tw] = self.dither.area
        return dither_mod.overlay_rgba(m, ar)

    def outline_overlay(self) -> np.ndarray:
        """輪郭の線を引いたドットの見える化（出力の大きさの RGBA。表示だけ）。引いていなければ全部透明。"""
        h, w = self.index.shape
        m = np.zeros((h, w), bool)
        if self.outline is not None:
            ox, oy = self.canvas_offset or (0, 0)
            th, tw = self.outline.drawn.shape
            m[oy:oy + th, ox:ox + tw] = self.outline.drawn
        return outline_mod.overlay_rgba(m)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.index.shape[1]), int(self.index.shape[0])

    def rgb(self) -> np.ndarray:
        return self.palette[self.index]


def _hex(c) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(v) for v in c))


def _dark_indices(palette: np.ndarray, darkest) -> list[int]:
    """いちばん暗い色の番号（0 番と、取っておいたいちばん暗い色）。"""
    dark = [0]
    if darkest is not None:
        dark += [int(i) for i in np.flatnonzero(np.all(palette == darkest, axis=1)) if int(i) != 0]
    return dark


def _drop_unused(index: np.ndarray, palette: np.ndarray):
    """使われなくなった色をパレットから外す（暗い順のまま番号を詰める。新しい色は作らない）。"""
    used, counts = np.unique(index, return_counts=True)
    remap = np.zeros(len(palette), np.uint8)
    remap[used] = np.arange(len(used), dtype=np.uint8)
    return remap[index], palette[used], counts.astype(np.int64)


def convert_still(rgba: np.ndarray, settings: StillSettings | None = None) -> StillResult:
    """RGBA（か RGB）の配列を変換する。止めるときは `StillError`。"""
    s = settings or StillSettings()
    s.validate()
    a = np.asarray(rgba)
    photo = None
    if a.ndim == 3 and a.shape[2] == 3 and a.dtype == np.uint8:
        rgb = a   # 写真は大きいので、写しを作らずにそのまま使う（中身は変えない）
    else:
        rgb = dots_mod.flatten_black(a)
    h, w = rgb.shape[:2]
    max_w, max_h = s.max_size
    smoothing = s.smooth != "off"
    what = "写真" if s.mode == "photo" else "画像"
    if not s.find_grid and smoothing:
        # 横のマス数で分けて色の塊をまとめる（写真の下書きの工程）。重い処理の前に大きさを確かめる
        if s.width > w:
            raise StillError(f"横のマス数（{s.width}）が、{what}の幅（{w}px）より多くなっています。",
                             dots_mod.size_candidates(w, h, max_w, max_h), "width")
        if dots_mod.height_for_width(s.width, w, h) > max_h or s.width > max_w:
            dw, dh = s.width, dots_mod.height_for_width(s.width, w, h)
            raise StillError(f"--width {s.width} で {dw}×{dh} ドットになり、大きさの上限 {max_w}×{max_h} を超えたので、"
                             "出力しませんでした。--width で横のマス数を小さくするか、--max-size で上限を変えてください。",
                             dots_mod.size_candidates(w, h, max_w, max_h), "size")
        dots, photo = photo_mod.photo_dots(rgb, s.width, s.smooth)
    elif s.find_grid:
        try:
            dots = dots_mod.grid_dots(rgb, s.max_size)
        except dots_mod.StillGridError as e:
            raise StillError(str(e), e.candidates, "grid") from e
    else:
        if s.width > w:
            raise StillError(f"横のマス数（{s.width}）が、{what}の幅（{w}px）より多くなっています。",
                             dots_mod.size_candidates(w, h, max_w, max_h), "width")
        dots = dots_mod.width_dots(rgb, s.width)
    dw, dh = dots.size
    if dw > max_w or dh > max_h:
        how = "グリッド検出" if dots.method == "grid" else f"--width {s.width}"
        aw, ah = int(dots.xs[-1] - dots.xs[0]), int(dots.ys[-1] - dots.ys[0])
        raise StillError(f"{how}で {dw}×{dh} ドットになり、大きさの上限 {max_w}×{max_h} を超えたので、出力しませんでした。"
                         "--width で横のマス数を小さくするか、--max-size で上限を変えてください。",
                         dots_mod.size_candidates(aw, ah, max_w, max_h), "size")
    if s.find_grid and smoothing:   # マス目を探したうえで、マスの中で色の塊をまとめる
        dots, photo = photo_mod.smooth_grid_dots(rgb, dots, s.smooth)
    if photo is None and s.orient:   # 向きだけを直したとき（色の塊はまとめない）
        photo = photo_mod.PhotoWork(None, False, "off")
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
    similar = None
    if s.merge_similar:
        similar = colors_mod.merge_similar(index, palette, counts, _dark_indices(palette, red.darkest))
        index, palette, counts = similar.index, similar.palette, similar.counts
        if similar.pairs:
            warnings.append(f"見た目でほぼ同じ色 {len(similar.pairs)} 組をまとめ、{len(palette)} 色になりました（上限 {s.colors} 色）")
    dither = None
    if s.dither != "off":
        # 減色する前のドットの色から、近い 2 色の間に網目を入れる（パレットは変えない。いちばん暗い色は混ぜない）
        dither = dither_mod.make_dither(dots.rgb, index, palette, s.dither, _dark_indices(palette, red.darkest))
        index = dither.index
        if dither.count:
            index, palette, counts = _drop_unused(index, palette)
    lone = None
    if s.tidy_dots:
        # ディザの網目の範囲のドットは変えない（網目が孤立したドットとして消されないように）
        lone = lone_mod.tidy_lone_dots(index, _dark_indices(palette, red.darkest),
                                       None if dither is None else dither.area)
        index = lone.index
        if lone.count:
            index, palette, counts = _drop_unused(index, palette)
    tidied = None
    if s.tidy_pattern:
        dark = [0]
        if red.darkest is not None:
            dark += [int(i) for i in np.flatnonzero(np.all(palette == red.darkest, axis=1))]
        tidied = pattern_mod.tidy_pattern(index, palette, dark, None if dither is None else dither.area)
        # 使われなくなった色はパレットから外す（暗い順のまま。新しい色は作らない）
        index, palette, counts = _drop_unused(tidied.index, palette)
    outline = None
    if s.outline:
        outline = outline_mod.draw_outline(index, palette)
        index = outline.index
        if outline.count:
            index, palette, counts = _drop_unused(index, palette)
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
    return StillResult(index, palette, counts, dots, red, s, (w, h), offset, warnings, tidied, similar, lone,
                       outline, photo, dither=dither)


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
    colors = {"limit": s.colors, "used": len(res.palette), "chosen": red.chosen, "merged_by_4bit": red.merged,
              "darkest_kept": None if red.darkest is None else _hex(red.darkest)}
    if res.similar is not None:   # 似た色をまとめたときだけ（98 モードの JSON は前と同じ）
        colors["merged_similar"] = len(res.similar.pairs)
    return {
        "app": APP, "version": FORMAT_VERSION, "source": source,
        "input": {"width": res.input_size[0], "height": res.input_size[1]},
        "width": w, "height": h,
        "picture": {"width": res.dots.size[0], "height": res.dots.size[1]},
        "pixel_aspect": "1:1",
        "dots": res.dots.describe(),
        **({} if res.photo is None else {"photo": res.photo.describe(res.orientation if s.orient else None)}),
        "canvas": canvas,
        "colors": colors,
        "palette": pal,
        **({} if res.similar is None else {"similar": similar_info(res.similar)}),
        **({} if res.dither is None else {"dither": res.dither.describe()}),
        **({} if res.lone is None else {"tidy_dots": {"changed": res.lone.count, "settings": res.lone.settings}}),
        **({} if res.tidied is None else {"pattern": pattern_info(res.tidied)}),
        **({} if res.outline is None else {"outline": {"drawn": res.outline.count, "settings": res.outline.settings}}),
        "settings": s.to_json(),
        **({} if (preset := s.preset_info()) is None else {"preset": preset}),
        "warnings": list(res.warnings),
    }


def similar_info(sim: colors_mod.Similar) -> dict:
    """似た色をまとめた記録（まとめた組の数・しきい値・まとめた順の組）。"""
    return {"merged": len(sim.pairs), "distance": colors_mod.SIMILAR_DISTANCE, "space": "OKLab",
            "list": [{"from": _hex(p["from"]), "to": _hex(p["to"]), "count": p["count"],
                      "distance": round(p["distance"], 4)} for p in sim.pairs]}


def pattern_info(t: pattern_mod.Tidied) -> dict:
    """--tidy-pattern の記録（範囲の数・変えたドット数・中で決めた設定・範囲ごと）。色の番号は整える前のパレットのもの。"""
    return {"regions": len(t.tidied), "changed": t.changed,
            "skipped": {"third_color": sum(r.skipped == "third_color" for r in t.regions), "small": t.small},
            "protected_line_dots": int(np.count_nonzero(t.protected)),
            "settings": t.settings,
            "list": [{"box": list(r.box), "size": r.size, "colors": [_hex(c) for c in r.colors_rgb],
                      "changed": r.changed, "third": r.third, **({"skipped": r.skipped} if r.skipped else {})}
                     for r in t.regions]}


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
    if photo_mod.is_heif(data):
        raise StillError(HEIF_MESSAGE, kind="format")
    if settings.orient:   # 写真の向き（EXIF）を直して読む（切のときの読み方は前のまま）
        rgb, orientation = photo_mod.decode_photo(data)
        res = convert_still(rgb, settings)
        res.orientation = orientation
    else:
        res = convert_still(decode_image(data), settings)
    info = result_info(res, source)
    return {"png": encode_indexed_png(res.index, res.palette), "json": info_json_text(info), "info": info,
            "result": res}


# ---- ブラウザ版（Pyodide）向け ----

HEIF_MESSAGE = "HEIC（iPhone などの写真の形式）は読み込めません。JPEG か PNG にしてください。"


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, StillError):
        return str(exc)
    if isinstance(exc, (OSError, Image.DecompressionBombError)) or type(exc).__name__ == "UnidentifiedImageError":
        return "画像として読み込めませんでした。JPEG か PNG にしてください。"
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
    extra = {}

    def b64png(rgba):
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    if out["result"].tidied is not None:   # 見える化の重ね絵（表示だけ。保存する PNG には入らない）
        extra = {"pattern": info["pattern"], "overlay_base64": b64png(out["result"].overlay())}
    if out["result"].lone is not None:
        extra.update(tidy_dots=info["tidy_dots"], dots_overlay_base64=b64png(out["result"].dots_overlay()))
    if out["result"].similar is not None:
        extra["similar"] = info["similar"]
    if out["result"].dither is not None:
        extra.update(dither=info["dither"], dither_overlay_base64=b64png(out["result"].dither_overlay()))
    if out["result"].outline is not None:
        extra.update(outline=info["outline"], outline_overlay_base64=b64png(out["result"].outline_overlay()))
    if out["result"].photo is not None:
        extra["photo"] = info["photo"]
    return json.dumps({"ok": True, "png_base64": base64.b64encode(out["png"]).decode("ascii"), "json_text": out["json"],
                       "width": info["width"], "height": info["height"], "palette": info["palette"],
                       "dots": info["dots"], "colors": info["colors"], "canvas": info["canvas"],
                       "warnings": info["warnings"], **extra}, ensure_ascii=False)


def spec() -> dict:
    """画面に埋め込む設定の選択肢と既定値（CLI の既定と同じ）。"""
    return {"modes": {k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}}
                      for k, v in MODES.items()},
            "default_mode": DEFAULT_MODE,
            "colors": {"min": colors_mod.MIN_COLORS, "max": colors_mod.MAX_COLORS},
            "modern_colors": list(MODERN_COLOR_CHOICES),
            "smooth_levels": [{"value": k, "label": photo_mod.SMOOTH_LABELS[k]} for k in SMOOTH_LEVELS],
            "smooth_choices": [{"value": k, "label": "しない" if k == "off" else photo_mod.SMOOTH_LABELS[k]}
                               for k in SMOOTH_CHOICES],
            "dither_choices": [{"value": k, "label": dither_mod.LABELS[k]} for k in DITHER_CHOICES],
            "items": list(ITEMS),
            "max_side": MAX_SIDE}
