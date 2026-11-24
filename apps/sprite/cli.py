"""Sprite App の CLI。

    python -m apps.sprite 入力 [入力 ...] [-o 出力]                                  # ドット絵風（グリッド検出）
    python -m apps.sprite 入力 --input-type illust (--width W | --height H | --scale S)  # 普通のイラスト（縮小）
    python -m apps.sprite 入力 --motion breathe          # 呼吸のスプライトシート（<出力名>_breathe_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion breathe --breathe-split 12   # 下から 12 ドット目のあたりで上下を分ける
    python -m apps.sprite 入力 --motion sway             # 横揺れのスプライトシート（<出力名>_sway_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion sway --sway-split 10   # 下から 10 ドット目のあたりから上を揺らす
    python -m apps.sprite 入力 --motion float            # 浮遊（全体が上下する）のスプライトシート（<出力名>_float_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion bounce           # 弾む（全体が上にだけ跳ねる）のスプライトシート（<出力名>_bounce_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion bounce --bounce-height 3   # 3 ドットの高さまで跳ねる（0,-1,-2,-3,-2,-1）
    python -m apps.sprite 入力 --motion nod              # うなずく（首から上が下がる）のスプライトシート（<出力名>_nod_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion nod --nod-split 16 --nod-depth 2   # 下から 16 ドット目のあたりで分け、頭を 2 ドットまで下げる
    python -m apps.sprite 入力 --motion stretch          # 伸び（体の上が 1 ドット伸びる）のスプライトシート（<出力名>_stretch_sheet.png / .json）も書く
    python -m apps.sprite 入力 --motion nod --nod-depth 2 --keep-small-parts   # 目・頬などの小さな部品の長さを変えない
    python -m apps.sprite 入力 --motion breathe --write-motion-spec breathe.motion.json   # プリセットが埋めた動きの表も書く
    python -m apps.sprite 入力 --motion-spec bounce.motion.json   # 動きの表で動きを作る（<出力名>_<表の name>_sheet.png / .json）
    python -m apps.sprite 入力 --motion breathe --gif    # シートと一緒に GIF アニメ（<出力名>_breathe.gif。4 倍・透明）も書く
    python -m apps.sprite 入力 --motion sway --gif --gif-scale 2 --gif-background "#203040"   # 2 倍・背景色あり
    python -m apps.sprite 歩き.png --frames --frame-count 4 --gif   # 横に並んだコマを切り分けてそろえ、シート（<出力名>_frames_<幅>x<高さ>_<N>f.png / .json）と GIF を書く
    python -m apps.sprite 正面.png --frames --frame-order 1,2,3,2 --frame-duration 150 --mirror   # 1・2・3・2 の順、左右反転のシートも書く
    python -m apps.sprite 走り.png --frames --align-y match --frame-root-y 0,-1,0,-1 --flicker   # 頭の高さでそろえ、体全体を上下させる。ちらつきの画像も書く
    python -m apps.sprite 歩き.png --frames --piskel     # シートと一緒に Piskel で開ける .piskel も書く
    python -m apps.sprite dance1.png dance2.png --frames --join -o out/sprite/dance.png   # 別々のファイルの絵を並べた順につなげて 1 つの動きにする
    python -m apps.sprite 入力 --motion breathe --fx surprise --gif   # ！（驚き）を別のシート（<出力名>_breathe_fx_surprise.png / .json）にして重ねる
    python -m apps.sprite 直した.piskel --readback 元.json --gif --mirror --piskel   # 手直ししたシート（PNG か .piskel）を読み戻して作り直す

入力にはファイルかフォルダ（中の画像をすべて処理）を指定する。工程は `build_pipeline()` で組み立てる。
入力の種類（`--input-type`）は利用者が選ぶ。既定はドット絵風（`pixel`）で、出力の大きさはグリッド検出で決まる。
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

from retrocore.sprite import (ALL_RESIZE_METHODS, PIXEL_METHODS, GridNotFoundError, PaletteError, SpritePipeline,
                              load_palette, load_rgba, parse_color, save_palette, save_rgba,
                              scale_nearest, used_colors)
from retrocore.sprite import palette as palette_mod
from retrocore.sprite import outline as outline_mod
from retrocore.sprite import convert as convert_mod
from retrocore.sprite.convert import (DEFAULT_DOT_SHAPE, DEFAULT_INPUT_TYPE, DEFAULT_RESIZE_METHOD, DOT_SHAPES,
                                      INPUT_TYPES, MAX_COLORS, SpriteSettings, convert_sprite)
from retrocore.sprite.cleanup import DEFAULT_LEVEL as DEFAULT_CLEANUP, LEVEL_HELP as CLEANUP_LEVELS
from retrocore.sprite.background import DEFAULT_TOLERANCE
from retrocore.sprite.grid import DEFAULT_SNAP_TOLERANCE
from retrocore.sprite.simplify import DEFAULT_COLORS, DEFAULT_KEEP_LARGE, DEFAULT_MERGE_DISTANCE, LARGE_DISTANCE
from retrocore.sprite.downscale import (DEFAULT_BIN_SIZE, DEFAULT_LINE_LUMA, DEFAULT_LINE_RATIO,
                                        DEFAULT_TRANSPARENT_RATIO)
from retrocore.sprite import motion as motion_mod
from retrocore.sprite.motion import bouncing as bounce_mod
from retrocore.sprite.motion import breathe as breathe_mod
from retrocore.sprite.motion import floating as float_mod
from retrocore.sprite.motion import gif as gif_mod
from retrocore.sprite.motion import nod as nod_mod
from retrocore.sprite.motion import stretch as stretch_mod
from retrocore.sprite.motion import sway as sway_mod
from retrocore.sprite.motion.report import skipped_json
from retrocore.sprite import frames as frames_mod
from retrocore.sprite import join as join_mod
from retrocore.sprite import anims as anims_mod
from retrocore.sprite import same as same_mod
from retrocore.sprite import steady as steady_mod
from retrocore.sprite import edit as edit_mod
from retrocore.sprite import fx as fx_mod

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
DEFAULT_OUT = Path("out") / "sprite"
SIZE_OPTIONS = ("width", "height", "scale")
RESIZE_OPTIONS = ("resize_method", "mode_bin", "line_priority", "line_ratio", "line_luma")
SIMPLIFY_OPTIONS = ("colors", "merge_distance", "keep_large")
GRID_OPTIONS = ("dot_size", "dot_shape", "grid_origin", "grid_snap", "grid_tolerance")
OUTLINE_OPTIONS = ("outline_color", "outline_shape", "outline_colors", "outline_pad")
BREATHE_OPTIONS = ("breathe_depth", "breathe_split")
SWAY_OPTIONS = ("sway_amount", "sway_split")
FLOAT_OPTIONS = ("float_amount",)
BOUNCE_OPTIONS = ("bounce_height",)
NOD_OPTIONS = ("nod_depth", "nod_split")
STRETCH_OPTIONS = ("stretch_amount", "stretch_split")
MOTION_OPTIONS = (("motion_frames", "motion_duration") + BREATHE_OPTIONS + SWAY_OPTIONS + FLOAT_OPTIONS + BOUNCE_OPTIONS
                  + NOD_OPTIONS + STRETCH_OPTIONS + ("keep_small_parts",))
# 小さな部品を守る（--keep-small-parts）を使える動き（縦の変形を使うもの）
KEEP_MOTIONS = ("breathe", "nod", "stretch")
SPEC_HEIGHTS = ("number", "percent")
# 動きごとの専用のオプションと、その名前（ほかの動きのオプションを付けたときのエラーに出す）
OWN_OPTIONS = {breathe_mod.NAME: (BREATHE_OPTIONS, "呼吸"), sway_mod.NAME: (SWAY_OPTIONS, "横揺れ"),
               float_mod.NAME: (FLOAT_OPTIONS, "浮遊"), bounce_mod.NAME: (BOUNCE_OPTIONS, "弾む"),
               nod_mod.NAME: (NOD_OPTIONS, "うなずく"), stretch_mod.NAME: (STRETCH_OPTIONS, "伸び")}
MOTION_MODULES = {breathe_mod.NAME: breathe_mod, sway_mod.NAME: sway_mod, float_mod.NAME: float_mod,
                  bounce_mod.NAME: bounce_mod, nod_mod.NAME: nod_mod, stretch_mod.NAME: stretch_mod}
# 弾むのフレーム数は高さ（--bounce-height）、うなずくは深さ（--nod-depth）、伸びは量（--stretch-amount）で決まる（check_args で数える）
MOTION_DEFAULT_FRAMES = {k: m.DEFAULT_FRAMES for k, m in MOTION_MODULES.items()
                         if k not in (bounce_mod.NAME, nod_mod.NAME, stretch_mod.NAME)}
MAX_PREVIEW_SCALE = 32
# コマを切り分けてそろえる（--frames）のオプション（--frames と一緒にだけ使える）
FRAME_OPTIONS = ("frame_count", "align", "align_y", "frame_grid", "frame_order", "frame_duration", "frame_name", "frame_root_y",
                 "frame_shift_y",
                 "mirror", "flicker", "flicker_rows", "steady", "steady_size", "steady_rows",
                 "same_ranges", "same_search", "same_from", "edits", "join", "join_match_height")
# 手直しの読み戻し（--readback）だけのオプション
READBACK_OPTIONS = ("original", "same_frames", "fix_colors", "anim")
# 読み戻しで使えるほかのオプション（ほかの変換・動きのオプションは使えない）
READBACK_ALLOWED = ("readback", "output", "gif", "gif_scale", "gif_background", "mirror", "piskel", "quiet", "inputs")


def _scale(text: str) -> float:
    """倍率: 0.25 や 1/4 の形式。"""
    try:
        v = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"倍率は 0.25 や 1/4 の形式で指定してください: {text}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"倍率は正の数で指定してください: {text}")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定してください: {text}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"1 以上で指定してください: {text}")
    return v


def _preview_scale(text: str) -> int:
    v = _positive_int(text)
    if v > MAX_PREVIEW_SCALE:
        raise argparse.ArgumentTypeError(f"1〜{MAX_PREVIEW_SCALE} で指定してください: {text}")
    return v


def _ratio(text: str) -> float:
    """割合: 0 より大きく 1 以下。"""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 < v <= 1:
        raise argparse.ArgumentTypeError(f"0 より大きく 1 以下で指定してください: {text}")
    return v


def _bin_size(text: str) -> int:
    v = _positive_int(text)
    if v > 256:
        raise argparse.ArgumentTypeError(f"1〜256 で指定してください: {text}")
    return v


def _luma(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 <= v <= 255:
        raise argparse.ArgumentTypeError(f"0〜255 で指定してください: {text}")
    return v


def _dot_size(text: str) -> tuple[float, float]:
    """ドットの大きさ: 20 または 20x21（横x縦、小数可）。"""
    parts = text.lower().replace("×", "x").split("x")
    try:
        vals = [float(v) for v in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"ドットの大きさは 20 や 20x21 の形式で指定してください: {text}")
    if len(vals) == 1:
        vals = vals * 2
    if len(vals) != 2 or min(vals) < 2:
        raise argparse.ArgumentTypeError(f"ドットの大きさは 2 以上で、20 や 20x21 の形式で指定してください: {text}")
    return vals[0], vals[1]


def _origin(text: str) -> tuple[float, float]:
    """グリッドの始まる位置: X,Y（画素、小数可）。"""
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"グリッドの始まる位置は X,Y の形式で指定してください: {text}")
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError(f"グリッドの始まる位置は 0 以上で指定してください: {text}")
    return x, y


def _snap_tolerance(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 < v < 1:
        raise argparse.ArgumentTypeError(f"0 より大きく 1 未満で指定してください: {text}")
    return v


def _colors(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定してください: {text}")
    if not 1 <= v <= MAX_COLORS:
        raise argparse.ArgumentTypeError(f"1〜{MAX_COLORS} で指定してください: {text}")
    return v


def _merge_distance(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"0〜1 で指定してください（既定 {DEFAULT_MERGE_DISTANCE:g}、0 でまとめない）: {text}")
    return v


def _keep_large(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"0〜1 で指定してください（既定 {DEFAULT_KEEP_LARGE:g}、0 で守らない）: {text}")
    return v


def _gif_scale(text: str) -> int:
    try:
        return gif_mod.check_scale(int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{gif_mod.MIN_SCALE}〜{gif_mod.MAX_SCALE} の整数で指定してください: {text}") from None


def _gif_background(text: str) -> str:
    try:
        gif_mod.check_background(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return text


def _motion_frames(text: str) -> int:
    v = _positive_int(text)
    if not 2 <= v <= breathe_mod.MAX_FRAMES:
        raise argparse.ArgumentTypeError(f"2〜{breathe_mod.MAX_FRAMES} で指定してください: {text}")
    return v


def _durations(text: str) -> list[int]:
    """表示時間（ミリ秒）: 500（全フレーム共通）または 600,400（フレームごと）。"""
    try:
        vals = [int(v) for v in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"表示時間は 500 や 600,400 の形式（ミリ秒の整数）で指定してください: {text}")
    if min(vals) < 1:
        raise argparse.ArgumentTypeError(f"表示時間は 1 ミリ秒以上で指定してください: {text}")
    return vals


def _int_list(text: str) -> list[int]:
    """0,-1,0,-1 のような整数の並び。"""
    try:
        vals = [int(v) for v in text.replace(" ", "").split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"0,-1,0,-1 のように整数をカンマで区切って指定してください: {text}")
    return vals


def _fx_offset(text: str) -> tuple[int, int]:
    """重ねのずらす量: X,Y（ドット。右と下が +）。"""
    vals = _int_list(text)
    if len(vals) != 2:
        raise argparse.ArgumentTypeError(f"X,Y の 2 つの整数で指定してください（例 1,-2）: {text}")
    if max(abs(v) for v in vals) > fx_mod.MAX_OFFSET:
        raise argparse.ArgumentTypeError(f"−{fx_mod.MAX_OFFSET}〜{fx_mod.MAX_OFFSET} で指定してください: {text}")
    return vals[0], vals[1]


def _frame_order(text: str) -> list:
    """コマの順番: 1,2,3,2（元の絵の左から 1, 2, …）。r を付けたコマは左右に裏返して使う（1,2,3,2r）。
    戻り値は数の並び（r が無いとき。今までと同じ）か、"2r" のような文字を混ぜた並び。"""
    try:
        nums, flips = frames_mod.parse_order_flips(text, None)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{e}: {text}") from None
    return [f"{k}r" if f else k for k, f in zip(nums, flips)]


def _order_numbers(order) -> list[int]:
    """--frame-order の並びの番号だけ（r を除く）。"""
    return frames_mod.parse_order_flips(order, None)[0]


def _steady_size(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定してください: {text}") from None
    if not steady_mod.MIN_SIZE <= v <= steady_mod.MAX_SIZE:
        raise argparse.ArgumentTypeError(f"{steady_mod.MIN_SIZE}〜{steady_mod.MAX_SIZE} で指定してください: {text}")
    return v


def _same_rect(text: str):
    try:
        return same_mod.parse_rect(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _same_band(text: str):
    try:
        return same_mod.parse_band(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _same_search(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定してください: {text}") from None
    if not same_mod.MIN_SEARCH <= v <= same_mod.MAX_SEARCH:
        raise argparse.ArgumentTypeError(f"{same_mod.MIN_SEARCH}〜{same_mod.MAX_SEARCH} で指定してください: {text}")
    return v


def _frame_count(text: str) -> int:
    v = _positive_int(text)
    if v > frames_mod.MAX_FRAMES:
        raise argparse.ArgumentTypeError(f"1〜{frames_mod.MAX_FRAMES} で指定してください: {text}")
    return v


def _frame_name(text: str) -> str:
    if not text or any(c in text for c in '/\\:*?"<>|') or text.strip() != text:
        raise argparse.ArgumentTypeError(f"ファイル名に使える名前にしてください（walk など）: {text}")
    return text


def _depth(text: str) -> int:
    v = _positive_int(text)
    if v > breathe_mod.MAX_DEPTH:
        raise argparse.ArgumentTypeError(f"1〜{breathe_mod.MAX_DEPTH} で指定してください: {text}")
    return v


def _amount(text: str) -> int:
    v = _positive_int(text)
    if v > sway_mod.MAX_AMOUNT:
        raise argparse.ArgumentTypeError(f"1〜{sway_mod.MAX_AMOUNT} で指定してください: {text}")
    return v


def _float_amount(text: str) -> int:
    v = _positive_int(text)
    if v > float_mod.MAX_AMOUNT:
        raise argparse.ArgumentTypeError(f"1〜{float_mod.MAX_AMOUNT} で指定してください: {text}")
    return v


def _bounce_height(text: str) -> int:
    v = _positive_int(text)
    if v > bounce_mod.MAX_HEIGHT:
        raise argparse.ArgumentTypeError(f"1〜{bounce_mod.MAX_HEIGHT} で指定してください: {text}")
    return v


def _split(text: str):
    """分ける高さ（呼吸・横揺れ）: auto または下から何ドット目か（1 以上の整数。画像の高さを超えるかは画像ごとに確かめる）。"""
    if text == breathe_mod.AUTO:
        return text
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{breathe_mod.AUTO} か、下から何ドット目かの整数（1 = いちばん下の行）で指定してください: {text}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"1 以上（1 = いちばん下の行）で指定してください: {text}")
    return v


def _nod_depth(text: str) -> int:
    v = _positive_int(text)
    if v > nod_mod.MAX_DEPTH:
        raise argparse.ArgumentTypeError(f"1〜{nod_mod.MAX_DEPTH} で指定してください: {text}")
    return v


def _nod_split(text: str):
    """うなずくで分ける高さ: auto、下から何ドット目か（1 以上の整数）、キャラの高さに対する割合（31.0%。表の at と同じ書き方）。"""
    from retrocore.sprite.motion.spec import parse_percent
    if text.endswith("%"):
        p = parse_percent(text)
        if p is None or not 0 < p <= 100:
            raise argparse.ArgumentTypeError(f"割合は 31.0% のように、0 より大きく 100 以下で指定してください: {text}")
        return text
    return _split(text)


def _stretch_amount(text: str) -> int:
    v = _positive_int(text)
    if v > stretch_mod.MAX_AMOUNT:
        raise argparse.ArgumentTypeError(f"1〜{stretch_mod.MAX_AMOUNT} で指定してください: {text}")
    return v


def _color(text: str) -> tuple[int, int, int]:
    try:
        return parse_color(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _outline_color(text: str):
    """縁取りの色: darken / single / 色（#RRGGBB または R,G,B）。"""
    if text in outline_mod.COLOR_MODES:
        return text
    try:
        return parse_color(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"darken / single または色（#RRGGBB、R,G,B）で指定してください: {text}（{e}）")


def _tolerance(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定してください: {text}")
    if not 0 <= v <= 255:
        raise argparse.ArgumentTypeError(f"0〜255 で指定してください: {text}")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.sprite",
        description="画像をドット絵スプライトに変換する（現在は背景除去、グリッド検出または縮小、単純化、クリーンアップ、縁取り、"
                    "パレットに合わせる・パレットの保存）。")
    p.add_argument("inputs", nargs="*", type=Path,
                   help="入力画像またはフォルダ（--anims では省ける。そのときは設定のファイルのあるフォルダから探す）")
    p.add_argument("-o", "--output", type=Path,
                   help=f"出力先。.png で終わればファイル（入力が 1 枚のとき）、それ以外はフォルダ（既定 {DEFAULT_OUT}）")
    p.add_argument("--input-type", choices=list(INPUT_TYPES), default=DEFAULT_INPUT_TYPE,
                   help="入力の種類。pixel = ドット絵風の画像（ドット 1 個が何画素もある。グリッドを検出して "
                        "1 ドット＝1 画素に戻す。出力の大きさは自動で決まる）、illust = 普通のイラスト（指定の大きさに縮小）。"
                        f"既定 {DEFAULT_INPUT_TYPE}")
    size = p.add_argument_group("サイズ（--input-type illust のとき。幅・高さ、または倍率のどちらか）")
    size.add_argument("--width", type=_positive_int, help="出力の幅。高さを省くと縦横比を保つ")
    size.add_argument("--height", type=_positive_int, help="出力の高さ。幅を省くと縦横比を保つ")
    size.add_argument("--scale", type=_scale, help="縮小倍率（例: 0.25 または 1/4）")
    bg = p.add_argument_group("背景除去（元の大きさのまま、縮小の前に行う）")
    bg.add_argument("--background", action=argparse.BooleanOptionalAction, default=True,
                    help="外周からつながった背景色の部分を透明にする（既定で有効。--no-background で無効）。"
                         "入力がすでに透過付きならそのまま通す")
    bg.add_argument("--bg-color", type=_color, metavar="COLOR",
                    help="背景色（#RRGGBB または R,G,B）。省略時は画像の外周から推定する")
    bg.add_argument("--bg-tolerance", type=_tolerance, default=DEFAULT_TOLERANCE, metavar="T",
                    help=f"背景色とみなす差（R・G・B の差の最大値、0〜255、既定 {DEFAULT_TOLERANCE}）")
    bg.add_argument("--bg-remove-enclosed", action="store_true",
                    help="キャラクターに囲まれた背景（腕と体のすき間など）も消す。"
                         "背景と同じ色の白い服なども消えるので注意")
    grid = p.add_argument_group("グリッド検出（--input-type pixel のとき）")
    grid.add_argument("--dot-size", type=_dot_size, metavar="W[xH]",
                      help="ドットの大きさ（画素。例: 20 または 20x21、小数可）。省略時は自動で推定する。"
                           "指定すると「グリッドが見つからない」の判定をせずに変換する")
    grid.add_argument("--dot-shape", choices=list(DOT_SHAPES), default=None,
                      help=f"ドットの形。square = 正方形とみなし、縦横の情報を合わせて 1 つの大きさを推定する（既定）、"
                           "separate = 縦横の大きさを別々に推定する（横長・縦長のドットの画像向け）")
    grid.add_argument("--grid-origin", type=_origin, metavar="X,Y",
                      help="グリッドの始まる位置（画素）。指定すると線を色の境目に合わせず、等間隔の格子にする")
    grid.add_argument("--grid-snap", action=argparse.BooleanOptionalAction, default=None,
                      help="グリッド線を 1 本ずつ近くの色の境目に合わせる（既定で有効。--no-grid-snap で等間隔の格子）")
    grid.add_argument("--grid-tolerance", type=_snap_tolerance, metavar="R",
                      help=f"線の間隔をドットの大きさの ±R 倍まで許す（既定 {DEFAULT_SNAP_TOLERANCE}）")
    p.add_argument("--resize-method", choices=list(ALL_RESIZE_METHODS), default=None,
                   help=f"縮小方法（--input-type illust のとき。既定 {DEFAULT_RESIZE_METHOD}）。"
                        "mode / median はドット絵向け（ブロック内に実在する色を採用）")
    pix = p.add_argument_group("ドット絵向けの縮小（--input-type illust で --resize-method mode / median のとき）")
    pix.add_argument("--mode-bin", type=_bin_size, default=None,
                     help=f"mode で色をまとめる区間の幅（1〜256、既定 {DEFAULT_BIN_SIZE}）。大きいほど粗くまとめる")
    pix.add_argument("--line-priority", action=argparse.BooleanOptionalAction, default=None,
                     help="暗い輪郭線の画素が一定割合以上のブロックは線の色を採用する"
                          "（mode / median では既定で有効。--no-line-priority で無効）")
    pix.add_argument("--line-ratio", type=_ratio, default=None,
                     help=f"線優先にする暗い画素の割合（既定 {DEFAULT_LINE_RATIO}）")
    pix.add_argument("--line-luma", type=_luma, default=None,
                     help=f"暗い画素とみなす輝度の上限（0〜255、既定 {DEFAULT_LINE_LUMA:g}）")
    simp = p.add_argument_group("単純化（グリッド検出・縮小の後に、似た色をまとめる）")
    simp.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True,
                      help="似た色をまとめて色数を減らす（既定で有効。--no-simplify で無効）。"
                           "ディザは使わず、透明な部分はそのまま残す")
    simp.add_argument("--colors", type=_colors, default=None, metavar="N",
                      help=f"残す色の数の上限（1〜{MAX_COLORS}、既定 {DEFAULT_COLORS}）。透明は数えない")
    simp.add_argument("--merge-distance", type=_merge_distance, default=None, metavar="D",
                      help="この距離（OKLab。色み a・b は 1.5 倍で測る。0.02 でほぼ同じ、0.05 で並べると分かる、"
                           "0.1 ではっきり違う）より近い色は、色数の上限に関係なく必ず 1 色にまとめる"
                           f"（既定 {DEFAULT_MERGE_DISTANCE:g}。見分けのつかない色が上限まで残らないようにする）。"
                           "まとめた後も --colors より多ければ、上限までまとめる。0 でまとめない（色数の上限だけで決める）。"
                           "大きくすると（0.06 など）色数ではなく近さで決められる")
    simp.add_argument("--keep-large", type=_keep_large, default=None, metavar="R",
                      help="面積が全体のこの割合以上の色（肌・服・髪などの大きな面）どうしは、"
                           f"距離 {LARGE_DISTANCE:g} 以上の差があれば、ほかにまとめる組が無くなるまでまとめない"
                           f"（既定 {DEFAULT_KEEP_LARGE:g} = 1%%。淡い色の肌や服が髪の色に吸われて全体が同じ色っぽくなるのを防ぐ）。"
                           "0 で以前の動き")
    clean = p.add_argument_group("クリーンアップ（単純化の後に、余計なドットをまわりの色に置き換える。新しい色は作らない）")
    clean.add_argument("--cleanup", choices=list(CLEANUP_LEVELS), default=None,
                       help="強さ。" + "、".join(f"{k} = {v}" for k, v in CLEANUP_LEVELS.items())
                            + f"。既定 {DEFAULT_CLEANUP}")
    clean.add_argument("--no-cleanup", action="store_true", help="クリーンアップしない（--cleanup off と同じ）")
    ol = p.add_argument_group("縁取り（クリーンアップの後に、キャラクターの外側の輪郭（シルエットの縁）を整える）")
    ol.add_argument("--outline", choices=list(outline_mod.MODES), default=None,
                    help="動き方。auto = シルエットの縁にすでに暗い輪郭があれば切れ目（1〜2 ドット）だけを輪郭の色で埋め、"
                         "無ければ add と同じ、add = 外側（透明の部分）に 1 ドットの輪郭を付ける、"
                         "thin = auto に加えて、2 ドットの太さの外側の輪郭を 1 ドットにする（輪郭が無ければ、"
                         "暗くない縁の外側にだけ付ける。--outline-shape とは別）、off = しない。"
                         f"既定 {outline_mod.DEFAULT_MODE}")
    ol.add_argument("--no-outline", action="store_true", help="縁取りしない（--outline off と同じ）")
    ol.add_argument("--outline-color", type=_outline_color, default=None, metavar="darken|single|COLOR",
                    help="外側に付ける輪郭の色。darken = すぐ内側の色を暗くした色（色みを少し残す。場所ごとに変わる）、"
                         "single = 1 色（絵の中でいちばん暗い色）、#RRGGBB または R,G,B = その 1 色。"
                         f"既定 {outline_mod.DEFAULT_COLOR_MODE}")
    ol.add_argument("--outline-shape", choices=list(outline_mod.SHAPES), default=None,
                    help="外側に付ける線の形。thin = 斜めでつながる 1 ドットの線（角が太らない）、"
                         f"thick = 上下左右でつながる線（角も埋める）。既定 {outline_mod.DEFAULT_SHAPE}")
    ol.add_argument("--outline-colors", type=_positive_int, default=None, metavar="N",
                    help="darken で新しく作る輪郭の色の数の上限（近い色はまとめる。"
                         f"既定 {outline_mod.DEFAULT_MAX_COLORS}）")
    ol.add_argument("--outline-pad", action=argparse.BooleanOptionalAction, default=None,
                    help="シルエットが画像の端に接しているとき、その辺に 1 ドットの余白を足してから輪郭を付ける"
                         "（既定で有効。--no-outline-pad では大きさを変えず、はみ出す分は付けない）")
    pal = p.add_argument_group("パレット（複数のキャラクターの色をそろえる）")
    pal.add_argument("--palette", type=Path, metavar="FILE",
                     help="このパレットの色だけで出力する（" + " / ".join(palette_mod.READ_FORMATS)
                          + "。.png は画像に含まれる色を読む）。単純化の直後にパレットの最も近い色にし（OKLab、ディザなし、"
                          "透明はそのまま）、縁取りの色もパレットから選ぶ。別の色だった部品がパレットの同じ色になったら警告する")
    pal.add_argument("--save-palette", type=Path, action="append", metavar="FILE",
                     help="最終結果で使った色（透明は含めない）をパレットファイルに保存する（拡張子で "
                          + " / ".join(palette_mod.WRITE_FORMATS) + " を決める。何度でも指定できる）。"
                          "入力が複数のときは <名前>_<入力の名前>.<拡張子> に保存する")
    mo = p.add_argument_group("動き（変換の結果を材料にして、アニメーションのスプライトシートも書く。変換の結果は変わらない）")
    mo.add_argument("--motion", choices=list(motion_mod.MOTIONS), default=None,
                    help="付ける動き。" + "、".join(f"{k} = {v}" for k, v in motion_mod.MOTIONS.items())
                         + "。横に並べたシート <出力名>_<動き>_sheet.png（透明、拡大なし）と、フレームの大きさ・枚数・"
                           "表示時間を書いた <出力名>_<動き>_sheet.json を保存する。うまく作れない形なら付けずに警告する")
    mo.add_argument("--motion-frames", type=_motion_frames, default=None, metavar="N",
                    help=f"フレーム数（2〜{breathe_mod.MAX_FRAMES}。既定は呼吸 {breathe_mod.DEFAULT_FRAMES}、"
                         f"横揺れ {sway_mod.DEFAULT_FRAMES}、浮遊 {float_mod.DEFAULT_FRAMES}"
                         f"（横揺れ・浮遊は {sway_mod.MIN_FRAMES} 以上））。"
                         "ゆっくりした波で動く量を決める（呼吸: 2 → 0,1、4 で --breathe-depth 2 → 0,1,2,1。"
                         "横揺れ: 4 → 0,+1,0,-1、8 で --sway-amount 2 → 0,1,2,1,0,-1,-2,-1。"
                         "浮遊: 4 → 0,-1,0,+1（− が上）。弾むは高さで決まるので指定できない）")
    mo.add_argument("--motion-duration", type=_durations, default=None, metavar="MS[,MS...]",
                    help=f"各フレームの表示時間（ミリ秒。既定は呼吸 {breathe_mod.DEFAULT_DURATION}、"
                         f"横揺れ {sway_mod.DEFAULT_DURATION}、浮遊 {float_mod.DEFAULT_DURATION}、"
                         f"弾むは着地だけ {bounce_mod.LANDING_DURATION}・空中 {bounce_mod.AIR_DURATION}）。"
                         "600,400 のようにフレームごとにも指定できる（数はフレーム数と同じにする）。"
                         "1 つだけ書くと全フレーム同じ（弾むでも着地を長くしない）")
    mo.add_argument("--breathe-depth", type=_depth, default=None, metavar="D",
                    help=f"呼吸で下がる量（1〜{breathe_mod.MAX_DEPTH} ドット、既定 {breathe_mod.DEFAULT_DEPTH}）")
    mo.add_argument("--breathe-split", type=_split, default=None, metavar="auto|N",
                    help="呼吸で上下を分ける高さ。auto（既定）は自動。数なら出力の画像の下から何ドット目のあたりか"
                         f"（1 = いちばん下の行。継ぎ目をその上下 {breathe_mod.SPLIT_RANGE} ドット以内に置く）。"
                         "その近くで形を崩さずに分けられなければ呼吸を付けずに警告し、近くで分けられる高さを表示する")
    mo.add_argument("--sway-amount", type=_amount, default=None, metavar="A",
                    help=f"横揺れの揺れ幅（1〜{sway_mod.MAX_AMOUNT} ドット、既定 {sway_mod.DEFAULT_AMOUNT}）。"
                         "2 のときは上ほど大きく揺れる（体が少し傾く）")
    mo.add_argument("--sway-split", type=_split, default=None, metavar="auto|N",
                    help="横揺れで上下を分ける高さ。auto（既定）は自動。数なら出力の画像の下から何ドット目のあたりから上を揺らすか"
                         f"（1 = いちばん下の行。分け目をその上下 {sway_mod.SPLIT_RANGE} ドット以内に置く。"
                         "数え方は --breathe-split と同じ）。その近くで形を崩さずに揺らせなければ横揺れを付けずに警告し、"
                         "近くで分けられる高さを表示する")
    mo.add_argument("--float-amount", type=_float_amount, default=None, metavar="A",
                    help=f"浮遊の揺れ幅（1〜{float_mod.MAX_AMOUNT} ドット、既定 {float_mod.DEFAULT_AMOUNT}）。"
                         "キャラ全体を分けずに上下に動かす（足元も動く）。はみ出す分は上下に透明の行を足し、"
                         "JSON の padding とフレームごとの offset_y（下向きが +）に書く")
    mo.add_argument("--bounce-height", type=_bounce_height, default=None, metavar="H",
                    help=f"弾む高さ（1〜{bounce_mod.MAX_HEIGHT} ドット、既定 {bounce_mod.DEFAULT_HEIGHT}）。"
                         "1 → 0,-1、2 → 0,-1,-2,-1、3 → 0,-1,-2,-3,-2,-1（− が上。1 フレームに 1 ドットずつ）。"
                         "キャラ全体を分けずに上にだけ動かす（足元も動く）。上にだけ透明の行を足し、"
                         "JSON の padding.top とフレームごとの offset_y に書く。pivot はフレームの下端")
    mo.add_argument("--nod-depth", type=_nod_depth, default=None, metavar="D",
                    help=f"うなずく深さ（1〜{nod_mod.MAX_DEPTH} ドット、既定 {nod_mod.DEFAULT_DEPTH}）。"
                         "1 → 0,1、2 → 0,1,2,1（頭が下がる量。フレーム数は深さで決まる）。表示時間の既定は"
                         f"上を向いているコマ {nod_mod.UP_DURATION}・下がったコマ {nod_mod.DOWN_DURATION} ミリ秒")
    mo.add_argument("--nod-split", type=_nod_split, default=None, metavar="auto|N|P%",
                    help="うなずくで頭と体を分ける高さ（首）。auto（既定）は、横幅が頭より細くなり、その下で体が広がる行を"
                         "キャラの高さの下から 20〜85%% の範囲で探す（はっきりしなければ付けずに警告し、試せる高さを表示する）。"
                         "数なら出力の画像の下から何ドット目のあたりか（--breathe-split と同じ数え方）、"
                         "31.0%% のような割合ならキャラの高さに対する下からの割合（--spec-height percent で書いた表と同じ）")
    mo.add_argument("--stretch-amount", type=_stretch_amount, default=None, metavar="A",
                    help=f"伸びる量（1〜{stretch_mod.MAX_AMOUNT} ドット、既定 {stretch_mod.DEFAULT_AMOUNT}）。"
                         "1 → 0,+1、2 → 0,+1,+2,+1（フレーム数は量で決まる）。分け目のあたりの縦に同じ色が続く所のドットを複製して"
                         "上を上げる。上に出る分は透明の行を足し、JSON の padding.top に書く（pivot はフレームの下端のまま）。"
                         f"表示時間の既定は {stretch_mod.DEFAULT_DURATION} ミリ秒（呼吸と同じ）")
    mo.add_argument("--stretch-split", type=_nod_split, default=None, metavar="auto|N|P%",
                    help="伸びで上下を分ける高さ。auto（既定）は呼吸と同じく自動（腰のあたり）。数なら出力の画像の下から何ドット目のあたりか"
                         "（--breathe-split と同じ数え方）、35.0%% のような割合ならキャラの高さに対する下からの割合。"
                         "その近くで伸ばせなければ付けずに警告し、近くで伸ばせる高さを表示する")
    mo.add_argument("--keep-small-parts", action="store_const", const=True, default=None,
                    help="縮める・伸ばすときに、目・頬などの小さな部品（同じ色の塊で縦 3 ドット以下・面積 24 ドット以下）の"
                         "長さを変えない（継ぎ目を通さず、形のまま動かす）。--motion breathe / nod / stretch で使える。"
                         "表では範囲の \"keep_small_parts\": true")
    mo.add_argument("--motion-spec", type=Path, default=None, metavar="FILE",
                    help="動きの表（sprite-motion の JSON。行 = フレーム、列 = 動かし方）で動きを作る。--motion とは一緒に使えない"
                         "（--motion-frames などの設定も表に書く）。シートは <出力名>_<表の name>_sheet.png / .json。"
                         "作れない行が 1 つでもあればシートは書かず、全部の行の問題を警告する（行の番号は 1 から）")
    mo.add_argument("--write-motion-spec", type=Path, default=None, metavar="FILE",
                    help="使った動きの表を書き出す（--motion ならプリセットが埋めた表。--motion-spec なら読んだ表）。"
                         "作れたときは、範囲ごとに実際に使った高さと線を \"used\" に書く。--motion-spec で読み直せる。"
                         "入力が複数のときは <名前>_<入力の名前>.json")
    mo.add_argument("--swaps", type=Path, default=None, metavar="JSON",
                    help="差し替えの一覧（ブラウザ版の「ドットを直す」の「差し替えを描く」で保存した <出力名>_<名前>_swaps.json）で、"
                         "シートの中にまばたきを入れる。--motion・--motion-spec・--frames と一緒に使う。動き 1 周がまばたきの間隔"
                         "（ファイルの interval_ms。既定 3000 ミリ秒）より短いときは、何周か繰り返したシートにして 1 回だけ入れる。"
                         "まばたきがかかるフレームは時間で分ける（全体の時間は同じ）。1 コマ目と合わないフレームには入れずに警告する。"
                         "シート・JSON（swaps）・GIF・左右反転・.piskel に入る。大きさ・パレットが合わないときは作らずに理由を表示する。"
                         "一覧に口パク（talk）があれば、ふつうのシートとは別に、動きの上で口が動く「しゃべりのシート」"
                         "<出力名>_<動き>_talk_sheet.png / .json（--frames は <出力名>_<名前>_talk_<幅>x<高さ>_<N>f.png）も書く"
                         "（まばたきも入る。GIF・左右反転・.piskel も同じ）。口パクだけの一覧なら --motion なしでも使え、"
                         "そのときは口だけ動く <出力名>_talk_sheet.png を書く")
    mo.add_argument("--gif", action="store_true",
                    help="シートと一緒に、そのまま見られる GIF アニメ <出力名>_<動き>.gif も書く（サイトに載せる・人に見せる用。"
                         "--motion でも --motion-spec でも使える）。各フレームはシートのフレームと同じ絵（色を変えない・ずらさない）、"
                         "表示時間はシートの JSON の duration_ms（GIF は 10 ミリ秒単位なので、割り切れないときは近い値にして警告する）。"
                         "ずっと繰り返す。動きを作らなかったときは書かない")
    mo.add_argument("--gif-scale", type=_gif_scale, default=None, metavar="N",
                    help=f"GIF の拡大（{gif_mod.MIN_SCALE}〜{gif_mod.MAX_SCALE} 倍、既定 {gif_mod.DEFAULT_SCALE}）。"
                         "最近傍（1 ドットを N×N の正方形にする）だけ")
    mo.add_argument("--gif-background", type=_gif_background, default=None, metavar="transparent|#RRGGBB",
                    help="GIF の背景。transparent（既定）は透明、#203040 のような色ならその色で塗る")
    mo.add_argument("--spec-height", choices=SPEC_HEIGHTS, default=None,
                    help="--write-motion-spec で書く分ける高さ（at）の書き方。number（既定）は下から何ドット目の数（このキャラ用）、"
                         "percent はキャラの高さに対する割合（\"35.0%%\"。別のキャラにも使う）。auto はそのまま")
    fx = p.add_argument_group(
        "重ね（--fx。！・汗・ハート・Zzz・？・怒りマーク・キラキラなど 14 種の漫画的エフェクトを、キャラのシートとは別の小さなシートにして重ねる。"
        "--motion・--motion-spec・--frames と一緒に使う）")
    fx.add_argument("--fx", choices=fx_mod.effect_names(), default=None,
                    help="重ねるエフェクト（1 つ）。" + "、".join(f"{e['name']} = {e['label']}（{e['anchor_label']}・"
                                                         f"{'くり返し' if e['loop'] else '1 回'}）" for e in fx_mod.fx_spec()["effects"])
                         + "。キャラのシートは今までと同じで、エフェクトのシート <キャラのシートの名前>_fx_<名前>.png / .json を書く"
                         "（--frames はさらに _<幅>x<高さ>_<N>f。--mirror なら左右反転のキャラ用も）。"
                         "JSON の position は、キャラのコマの左上から見たエフェクトのコマの左上（ドット）。"
                         "--gif なら重ねた見た目の GIF <…>_fx_<名前>.gif、--frames --piskel ならレイヤー「キャラ」「エフェクト」に分けた "
                         "<…>_fx_<名前>.piskel も書く")
    fx.add_argument("--fx-offset", type=_fx_offset, default=None, metavar="X,Y",
                    help="自動で決めた置く場所（頭の上・頭の横・顔・顔の上半分・体のまわり）からずらす量（ドット。右と下が +）。− で始まるときは "
                         "--fx-offset=-1,2 のように = でつなぐ。左右反転のシートでは左右も反転する")
    fx.add_argument("--fx-end", choices=list(fx_mod.ENDS), default=None,
                    help="1 回で終わるエフェクト（" + "・".join(e["name"] for e in fx_mod.fx_spec()["effects"] if not e["loop"]) + "）の終わり方。"
                         + "、".join(f"{k} = {v}" for k, v in fx_mod.ENDS.items()) + f"（既定 {fx_mod.DEFAULT_END}）")
    fr = p.add_argument_group(
        "コマを切り分けてそろえる（--frames。何コマも横に並んだ 1 枚の絵（AI が描いた歩き・走りなど）を、"
        "同じグリッド・同じパレット・同じ大きさ・同じ足元にそろえて 1 枚のシートにする）")
    fr.add_argument("--frames", action="store_const", const=True, default=None,
                    help="背景を除いたあと、透明な列のすき間でコマに分け、全部のコマを 1 つのキャラとして変換する"
                         "（ドットの大きさは絵全体で 1 つ。線の位置はコマごとに合わせるか全部のコマで共通（--frame-grid）。"
                         "パレットは全部のコマで共通）。"
                         "<出力名>.png にそろえたコマを元の順に並べた絵、<出力名>_<名前>_<幅>x<高さ>_<N>f.png / .json にシートを書く"
                         "（幅・高さは 1 コマの大きさ、N はコマの数。エディタで読み込むときに入れる数）。"
                         "ドット絵風の入力（--input-type pixel）だけ。--motion とは一緒に使えない")
    fr.add_argument("--join", action="store_const", const=True, default=None,
                    help="入力のファイル（何枚でも。フォルダなら中の画像を名前の順に）を、書いた順につなげて 1 つの動きにする"
                         "（長い動きを AI に何枚かに分けて描かせたとき）。1 枚の中のコマは 1 つでも何コマでもよい。"
                         "グリッドはファイルごとに見つけ、パレットは全部のファイルで 1 つ、大きさと足元は全部のコマでそろえる。"
                         "出力名は 1 つ目のファイルの名前（-o で変えられる）。--join が無いときは、今までどおり 1 枚ずつ別々に変換する")
    fr.add_argument("--join-match-height", action="store_const", const=True, default=None,
                    help="--join で、キャラの高さ（ファイルの中でいちばん高いコマ）をいちばん小さいファイルに合わせる"
                         f"（違いが {round(join_mod.HEIGHT_MAX_RATIO * 100)}% までのとき。ドット絵に戻すときの格子の大きさを"
                         "ファイルごとに少し変える。完成したドット絵の拡大・縮小はしない）。既定は合わせず、高さの違いを警告するだけ"
                         "（しゃがみ・ジャンプのコマだけのファイルが混ざっても、ほかのファイルを小さくしないため）")
    fr.add_argument("--frame-count", type=_frame_count, default=None, metavar="N",
                    help="コマの数。見つかった数が違えば作らずに警告し、理由を <出力名>_<名前>_skipped.json に書く")
    fr.add_argument("--align", choices=list(frames_mod.ALIGNS), default=None,
                    help="横のそろえ方。" + "、".join(f"{k} = {v}" for k, v in frames_mod.ALIGNS.items()))
    fr.add_argument("--align-y", choices=list(frames_mod.ALIGNS_Y), default=None,
                    help="縦のそろえ方。" + "、".join(f"{k} = {v}" for k, v in frames_mod.ALIGNS_Y.items()))
    fr.add_argument("--frame-grid", choices=list(frames_mod.GRID_MODES), default=None,
                    help="グリッドの線の並び。" + "、".join(f"{k} = {v}" for k, v in frames_mod.GRID_MODES.items()))
    fr.add_argument("--frame-order", type=_frame_order, default=None, metavar="1,2,3,2",
                    help="シートに並べるコマの順番（元の絵の左から 1, 2, …。同じコマを何度使ってもよい）。既定は 1, 2, …。"
                         "番号の後に r を付けると、そのコマを左右に裏返して使う（1,2,3,2r。回る動きなど。そろえた基準を軸に裏返すので"
                         "キャラは横にずれない。左右で違う物は反対側に移る）")
    fr.add_argument("--frame-duration", type=_durations, default=None, metavar="MS[,MS...]",
                    help=f"各フレームの表示時間（ミリ秒。既定 {frames_mod.DEFAULT_DURATION}。走りは 100 前後が目安）。"
                         "フレームごとに 150,100,150,100 のようにも書ける（数は --frame-order の数と同じにする）")
    fr.add_argument("--frame-name", type=_frame_name, default=None, metavar="NAME",
                    help=f"シート・GIF のファイル名と JSON の motion に使う名前（既定 {frames_mod.DEFAULT_NAME}。walk・run など）")
    fr.add_argument("--frame-root-y", type=_int_list, default=None, metavar="DY,DY,...",
                    help="フレームごとに体全体を上下にずらす（ドット。− が上。動きの表の root_y と同じ）。数は並べるフレームの数と同じ。"
                         "走りを跳ねて見せるときなど。− で始まるときは --frame-root-y=-1,0,-1,0 のように = でつなぐ。"
                         "はみ出す分は上下に透明の行を足し、JSON の padding と offset_y に書く")
    fr.add_argument("--frame-shift-y", type=_int_list, default=None, metavar="DY,DY,...",
                    help="元のコマごとに、そろえた位置から上げ下げする（ドット。− が上）。数は元のコマの数と同じ"
                         "（--join では全部のファイルの合計）。--align-y keep（元の絵の高さを保つ）でジャンプの高さがうまく出ないときなど。"
                         "− で始まるときは --frame-shift-y=0,-3,-6,0 のように = でつなぐ。キャンバスは全部のコマが入る高さに広がり、"
                         "JSON に地面の線（ground_y）と、フレームごとの足元の高さ（frames[].air）を書く")
    fr.add_argument("--mirror", action="store_const", const=True, default=None,
                    help="左右を反転したシート <出力名>_<名前>_mirror_<幅>x<高さ>_<N>f.png / .json も書く（--gif なら GIF も）。"
                         "--frame-order の r（裏返したコマ）も含めて全部のフレームを反転する（r のコマは元の向きに戻る）")
    fr.add_argument("--flicker", action="store_const", const=True, default=None,
                    help="ちらつきの見える化: 上の方（本来は動かないはずの所）で 1 コマ目と違うドットに色を付けた画像 "
                         "<出力名>_<名前>_flicker.png と、コマごとの数・場所の <出力名>_<名前>_flicker.json を書く"
                         "（自動では直さない。人がエディタで直すときの目印）。数は --flicker が無くても表示する")
    fr.add_argument("--flicker-rows", type=_positive_int, default=None, metavar="N",
                    help="ちらつきを数える範囲（そろえたキャンバスの上から N 行。既定は高さの半分）")
    fr.add_argument("--steady", choices=list(steady_mod.MODES), default=None,
                    help="ちらつきを減らす（そろえた後、シートを書く前）。" + "、".join(f"{k} = {v}" for k, v in steady_mod.MODES.items())
                         + "。on では、上の方（既定は高さの半分）で 1 コマ目と違うドットのうち、小さな点（--steady-size 以下）と"
                         "短い細い線だけを 1 コマ目の色に戻す。腕・脚などの大きな違いとそのすぐ近く、位置が決まらない所、"
                         "目・口などの小さな部品が 1 コマ目と同じ形にならない所、穴・輪郭の切れ目・足元が動く所は戻さない"
                         "（迷ったら何もしない）。戻したドットの数はシートの JSON の steady に書く")
    fr.add_argument("--steady-size", type=_steady_size, default=None, metavar="N",
                    help=f"--steady on で「小さい」とみなす最大のドット数（{steady_mod.MIN_SIZE}〜{steady_mod.MAX_SIZE}、"
                         f"既定 {steady_mod.DEFAULT_SIZE}）。細い線は N の {steady_mod.LINE_FACTOR} 倍の長さまで")
    fr.add_argument("--steady-rows", type=_positive_int, default=None, metavar="N",
                    help="--steady on で戻す範囲（そろえたキャンバスの上から N 行。既定は高さの半分。ちらつきを数える範囲と同じ）")
    fr.add_argument("--same-as-first", dest="same_ranges", action="append", type=_same_rect, default=None,
                    metavar="X,Y,W,H",
                    help="この範囲は 1 コマ目と同じにする（そろえたコマの 1 コマ目の上の四角。左上が 0,0、W・H は幅と高さのドット。"
                         "何回でも指定できる）。2 コマ目以降で範囲の中身がいちばんよく合う位置（上下左右 --same-search ドット以内）を探し、"
                         "そこに 1 コマ目のドットをそのまま写す（目・口は 1 コマ目とまったく同じになるか、そのコマのまま）。"
                         "合い方が低い・境目で段差や輪郭の途切れができる・目や口に腕などが入ってくるコマは写さずに警告し、次に試せる範囲を添える。"
                         "結果はシートの JSON の same_as_first と、図 <出力名>_<名前>_same.png。--steady on と一緒なら、写した後にちらつきを減らす")
    fr.add_argument("--same-band", dest="same_ranges", action="append", type=_same_band, default=None, metavar="Y0,Y1",
                    help="1 コマ目と同じにする帯（上から Y0 行目〜Y1 行目。0 から数え、両端を含む。横いっぱい）。--same-as-first と同じ扱い")
    fr.add_argument("--same-search", type=_same_search, default=None, metavar="N",
                    help=f"1 コマ目と同じにする範囲の位置を探す幅（上下左右のドット。{same_mod.MIN_SEARCH}〜{same_mod.MAX_SEARCH}、"
                         f"既定 {same_mod.DEFAULT_SEARCH}）")
    fr.add_argument("--same-from", type=_positive_int, default=None, metavar="K",
                    help="範囲の元にするコマ（元の絵の左から K 番目。既定 1）")
    fr.add_argument("--edits", type=Path, default=None, metavar="JSON",
                    help="手の直しの一覧（ブラウザ版の「ドットを直す」で保存した <出力名>_<名前>_edits.json）を最後に当てる"
                         "（1 コマ目と同じにする範囲・ちらつきを減らすの後）。直しは元のコマごとなので、1,2,3,2 の 2 と 4 の両方に入る。"
                         "シート・JSON（edits）・GIF・左右反転・.piskel に入る。コマの大きさ・コマの数・パレットが合わず当てられないときは作らずに理由を表示する。"
                         "直したときと下の色が変わったドットがあれば数を警告する")
    fr.add_argument("--piskel", action="store_const", const=True, default=None,
                    help="シートと同じ名前の .piskel（ブラウザのドット絵エディタ Piskel の「Open」で開ける形。コマ・大きさ・速さが入る）も書く。"
                         "Piskel は速さを 1 つしか持てないので、いちばん多い時間から決める（1〜24 FPS）。--frames か --readback と一緒に使う")
    am = p.add_argument_group(
        "動きに名前を付けてまとめる（--anims。「立ち」「歩き」「走り」「ジャンプ」などを 1 つのシート（1 つの動き = 1 行）にまとめ、"
        "JSON の animations に動きの名前とフレームの範囲を書く。ゲームのエンジンで 1 つのキャラに idle・walk などを持たせるとき）")
    am.add_argument("--anims", type=Path, default=None, metavar="JSON",
                    help="動きをまとめる設定のファイル（ブラウザ版で保存した <出力名>_<名前>.anims.json か、自分で書いたもの。"
                         "ファイル名・動きの名前・動きごとの順番・時間・くり返すか・縦のそろえ方・上げ下げと、共通の設定が入る）。"
                         "入力の画像は名前で探す（入力を書かなければ設定のファイルのあるフォルダから）。"
                         "<出力名>_<名前>_<幅>x<高さ>_<N>f.png / .json にまとめたシートを書く（--gif なら動きごとの GIF "
                         "<出力名>_<動き>.gif、--piskel なら動きごとの <出力名>_<動き>.piskel も）。"
                         "まばたき・口パク（--swaps）・重ね（--fx）・範囲・ちらつきを減らす・--flicker・--frame-root-y は警告して無視する")
    rb = p.add_argument_group(
        "手直しの読み戻し（--readback。Piskel などで直したシート（PNG か .piskel）を入力にして、元の JSON から時間・順番・足元を引き継ぎ、"
        "JSON・GIF・左右反転のシート・横 1 列に並べ直したシートを作り直す。直したドットはそのまま使う）")
    rb.add_argument("--readback", type=Path, default=None, metavar="JSON",
                    help="直す前のシートの情報（Sprite App が書いた、シートと同じ名前の .json）。入力は直したシート 1 つ（PNG か .piskel）。"
                         "<元のシートの名前>_edited.png / .json と確かめた結果 <…>_edited_check.png / .json を書く"
                         "（--gif・--mirror・--piskel で GIF・左右反転・.piskel も）。-o は書くフォルダ（既定 out/sprite）。"
                         "大きさ・コマの数が合わない、透明だけのコマがあるときは作らずに理由を表示する。"
                         "新しい色・半透明・同じコマの食い違いは作って警告し、場所を _check.png に描く")
    rb.add_argument("--original", type=Path, default=None, metavar="PNG",
                    help="直す前のシート（PNG）。直したドットに印を付けた <…>_edited_changes.png を書く。JSON に色の一覧（palette）が無い"
                         "古い JSON では、新しい色をこのシートの色と比べて調べる")
    rb.add_argument("--same-frames", choices=list(edit_mod.SAME_FRAMES), default=None,
                    help="同じ元のコマを何度も使う順番（1,2,3,2 など）で、直した結果が違うときの扱い。"
                         + "、".join(f"{k} = {v}" for k, v in edit_mod.SAME_FRAMES.items()) + f"（既定 {edit_mod.DEFAULT_SAME_FRAMES}）")
    rb.add_argument("--anim", type=str, default=None, metavar="NAME",
                    help="（動きをまとめたシートのとき）直した動きの名前。省くと直したファイルの名前（<出力名>_<動き>.piskel など）から決める。"
                         "まとめたシートの読み戻しには、直す前のまとめたシート（--original）も要る")
    rb.add_argument("--fix-colors", action="store_const", const=True, default=None,
                    help="おかしな色を直す: 元のパレットに無い色をいちばん近い元の色に、半透明を透明か不透明（α 128 が境）にする"
                         "（既定は直さずに警告だけ）")
    p.add_argument("--transparent-ratio", type=_ratio, default=DEFAULT_TRANSPARENT_RATIO,
                   help=f"透過画素がこの割合以上のマス（グリッド検出）・ブロック（縮小）を透過にする"
                        f"（既定 {DEFAULT_TRANSPARENT_RATIO}）")
    p.add_argument("--preview-scale", type=_preview_scale, metavar="N",
                   help=f"出力をニアレストネイバーで N 倍（1〜{MAX_PREVIEW_SCALE}）に拡大したプレビューも "
                        "<出力名>_xN.png に保存する")
    p.add_argument("--save-steps", action="store_true",
                   help="工程ごとの中間画像も <出力名>_steps/ に保存する")
    p.add_argument("-q", "--quiet", action="store_true", help="進捗を表示しない")
    return p


def settings_from_args(args: argparse.Namespace) -> SpriteSettings:
    """引数を変換の設定（`SpriteSettings`）に直す。指定の無いものは既定値にする。

    `--palette` を指定したときは、`args.palette_colors` に読み込んだ色を渡す（`main()` が読み込む）。
    """
    oc = args.outline_color
    return SpriteSettings(
        input_type=args.input_type, width=args.width, height=args.height, scale=args.scale,
        background=args.background, bg_color=args.bg_color, bg_tolerance=args.bg_tolerance,
        bg_remove_enclosed=args.bg_remove_enclosed,
        dot_size=args.dot_size, dot_shape=args.dot_shape or DEFAULT_DOT_SHAPE, grid_origin=args.grid_origin,
        grid_snap=args.grid_snap is not False, grid_tolerance=args.grid_tolerance or DEFAULT_SNAP_TOLERANCE,
        resize_method=args.resize_method or DEFAULT_RESIZE_METHOD, mode_bin=_or(args.mode_bin, DEFAULT_BIN_SIZE),
        line_priority=args.line_priority, line_ratio=_or(args.line_ratio, DEFAULT_LINE_RATIO),
        line_luma=_or(args.line_luma, DEFAULT_LINE_LUMA), transparent_ratio=args.transparent_ratio,
        simplify=args.simplify, colors=_or(args.colors, DEFAULT_COLORS),
        merge_distance=_or(args.merge_distance, DEFAULT_MERGE_DISTANCE),
        keep_large=_or(args.keep_large, DEFAULT_KEEP_LARGE),
        cleanup=cleanup_level(args), outline=outline_mode(args),
        outline_color=outline_mod.DEFAULT_COLOR_MODE if oc is None else oc,
        outline_shape=args.outline_shape or outline_mod.DEFAULT_SHAPE,
        outline_colors=_or(args.outline_colors, outline_mod.DEFAULT_MAX_COLORS),
        outline_pad=args.outline_pad is not False,
        palette=getattr(args, "palette_colors", None))


def build_pipeline(args: argparse.Namespace) -> SpritePipeline:
    """引数から工程を組み立てる（中身は `retrocore.sprite.convert.build_pipeline()`。ブラウザ UI と共通）。

    順番（docs/sprite-app-plan.md）: 背景除去 → グリッド検出または縮小 → 単純化 → パレットに合わせる
    → クリーンアップ → 縁取り → パレットにない色が無いかの確認。
    """
    return convert_mod.build_pipeline(settings_from_args(args))


def outline_mode(args: argparse.Namespace) -> str:
    """縁取りの動き方（--no-outline なら off）。"""
    if args.no_outline:
        return "off"
    return args.outline or outline_mod.DEFAULT_MODE


def cleanup_level(args: argparse.Namespace) -> str:
    """クリーンアップの強さ（--no-cleanup なら off）。"""
    if args.no_cleanup:
        return "off"
    return args.cleanup or DEFAULT_CLEANUP


def _or(v, default):
    return default if v is None else v


def _option_names(args: argparse.Namespace, names) -> list[str]:
    """指定されたオプションの名前（--xxx）。"""
    out = []
    for n in names:
        v = getattr(args, n)
        if v is not None:
            out.append("--" + n.replace("_", "-") if v is not False else "--no-" + n.replace("_", "-"))
    return out


def check_frames_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--frames とそのオプションの組み合わせを確かめる。"""
    if not args.frames:
        given = _option_names(args, FRAME_OPTIONS + ("piskel",))
        if given:
            parser.error(f"{' / '.join(given)} はコマを切り分けるオプションです。--frames と一緒に指定してください")
        return
    if args.motion is not None or args.motion_spec is not None or args.write_motion_spec is not None:
        parser.error("--frames は --motion / --motion-spec / --write-motion-spec と一緒に使えません"
                     "（体全体の上下は --frame-root-y で付けられます）")
    if args.input_type != "pixel":
        parser.error("--frames はドット絵風の入力（--input-type pixel、既定）だけで使えます")
    if args.join_match_height and not args.join:
        parser.error("--join-match-height は --join と一緒に指定してください")
    if args.grid_origin is not None:
        parser.error("--frames では --grid-origin は使えません（コマごとにグリッドのずれが違うため、線はコマごとに合わせます）")
    n = len(args.frame_order) if args.frame_order else args.frame_count
    if args.frame_order and args.frame_count is not None and max(_order_numbers(args.frame_order)) > args.frame_count:
        parser.error(f"--frame-order の番号（最大 {max(_order_numbers(args.frame_order))}）が --frame-count {args.frame_count} を超えています")
    if args.frame_shift_y is not None:
        if args.frame_count is not None and len(args.frame_shift_y) != args.frame_count:
            parser.error(f"--frame-shift-y の数（{len(args.frame_shift_y)}）は元のコマの数（--frame-count {args.frame_count}）と同じにしてください")
        if any(abs(v) > frames_mod.MAX_SHIFT_Y for v in args.frame_shift_y):
            parser.error(f"--frame-shift-y は −{frames_mod.MAX_SHIFT_Y}〜{frames_mod.MAX_SHIFT_Y} ドットで指定してください"
                         "（シートが上げ下げの分だけ広がるので、大きすぎる絵にならないように）")
    if args.frame_duration is not None and len(args.frame_duration) != 1 and n is not None \
            and len(args.frame_duration) != n:
        parser.error(f"--frame-duration の数（{len(args.frame_duration)}）は並べるフレームの数（{n}）と同じか、"
                     "1 つ（全フレーム共通）にしてください")
    if args.frame_root_y is not None and n is not None and len(args.frame_root_y) != n:
        parser.error(f"--frame-root-y の数（{len(args.frame_root_y)}）は並べるフレームの数（{n}）と同じにしてください")
    if args.flicker_rows is not None and not args.flicker:
        parser.error("--flicker-rows は --flicker と一緒に指定してください")
    if (args.steady_size is not None or args.steady_rows is not None) and args.steady != "on":
        parser.error("--steady-size・--steady-rows は --steady on と一緒に指定してください")
    if (args.same_search is not None or args.same_from is not None) and not args.same_ranges:
        parser.error("--same-search・--same-from は --same-as-first か --same-band と一緒に指定してください")
    if args.same_ranges and len(args.same_ranges) > same_mod.MAX_RANGES:
        parser.error(f"1 コマ目と同じにする範囲は {same_mod.MAX_RANGES} 個までです")
    if args.same_from is not None and args.frame_count is not None and args.same_from > args.frame_count:
        parser.error(f"--same-from {args.same_from} が --frame-count {args.frame_count} を超えています")


def check_fx_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--fx とそのオプションの組み合わせを確かめる。"""
    if args.fx is None:
        given = _option_names(args, ("fx_offset", "fx_end"))
        if given:
            parser.error(f"{' / '.join(given)} は重ねのオプションです。--fx surprise などと一緒に指定してください")
        return
    if args.motion is None and args.motion_spec is None and not args.frames:
        parser.error("--fx は --motion breathe などか --motion-spec か --frames と一緒に指定してください（キャラのシートに重ねます）")
    if args.fx_end is not None and fx_mod.load_effect(args.fx).loop:
        parser.error(f"--fx-end は 1 回で終わるエフェクトだけで使えます（{args.fx} はくり返します）")


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """入力の種類とオプションの組み合わせを確かめる（合わなければ parser.error で止める）。"""
    given = _option_names(args, READBACK_OPTIONS)
    if given:
        parser.error(f"{' / '.join(given)} は手直しの読み戻しのオプションです。--readback と一緒に指定してください")
    check_frames_args(parser, args)
    check_fx_args(parser, args)
    args.talk_only = False
    if args.swaps is not None and args.motion is None and args.motion_spec is None and not args.frames:
        if load_swaps_cli(args, quiet=True) is None or args._swaps.find("talk") is None:
            parser.error("--swaps は --motion breathe などか --motion-spec か --frames と一緒に指定してください（動きのシートに入れます。"
                         "口パク（talk）のある一覧なら、動きなしで口だけ動くしゃべりのシートを書きます）")
        args.talk_only = True
    if args.motion_spec is not None:
        if args.motion is not None:
            parser.error("--motion-spec と --motion は一緒に使えません（表で作るか、プリセットで作るかのどちらかにしてください）")
        given = _option_names(args, MOTION_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は --motion-spec と一緒に使えません（フレーム数・表示時間・高さなどは表に書きます）")
    elif args.motion is None:
        given = _option_names(args, MOTION_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は動きのオプションです。--motion breathe などと一緒に指定してください")
        if args.write_motion_spec is not None:
            parser.error("--write-motion-spec は --motion breathe などか --motion-spec と一緒に指定してください")
    else:
        for name, (opts, label) in OWN_OPTIONS.items():
            given = _option_names(args, opts) if name != args.motion else []
            if given:
                parser.error(f"{' / '.join(given)} は --motion {args.motion} では使えません（{label}のオプションです）")
        if args.keep_small_parts and args.motion not in KEEP_MOTIONS:
            parser.error(f"--keep-small-parts は --motion {args.motion} では使えません（縮める・伸ばす動き"
                         f"（{' / '.join(KEEP_MOTIONS)}）のオプションです）")
        mod = MOTION_MODULES[args.motion]
        low = getattr(mod, "MIN_FRAMES", 2)
        if mod is bounce_mod:
            height = args.bounce_height or bounce_mod.DEFAULT_HEIGHT
            if args.motion_frames is not None:
                parser.error(f"--motion-frames は --motion bounce では使えません（弾むのフレーム数は高さで決まります。"
                             f"高さ {height} → {bounce_mod.bounce_frames(height)} フレーム。--bounce-height で変えてください）")
            frames = bounce_mod.bounce_frames(height)
        elif mod is nod_mod:
            depth = args.nod_depth or nod_mod.DEFAULT_DEPTH
            if args.motion_frames is not None:
                parser.error(f"--motion-frames は --motion nod では使えません（うなずくのフレーム数は深さで決まります。"
                             f"深さ {depth} → {nod_mod.nod_frames(depth)} フレーム。--nod-depth で変えてください）")
            frames = nod_mod.nod_frames(depth)
        elif mod is stretch_mod:
            amount = args.stretch_amount or stretch_mod.DEFAULT_AMOUNT
            if args.motion_frames is not None:
                parser.error(f"--motion-frames は --motion stretch では使えません（伸びのフレーム数は量で決まります。"
                             f"量 {amount} → {stretch_mod.stretch_frames(amount)} フレーム。--stretch-amount で変えてください）")
            frames = stretch_mod.stretch_frames(amount)
        elif args.motion_frames is not None and args.motion_frames < low:
            parser.error(f"{OWN_OPTIONS[args.motion][1]}の --motion-frames は {low}〜{breathe_mod.MAX_FRAMES} で指定してください"
                         f"（2 では{'揺れ' if mod is sway_mod else '動き'}ません）: {args.motion_frames}")
        if mod not in (bounce_mod, nod_mod, stretch_mod):
            frames = args.motion_frames or MOTION_DEFAULT_FRAMES[args.motion]
        if args.motion_duration is not None and len(args.motion_duration) not in (1, frames):
            parser.error(f"--motion-duration の数（{len(args.motion_duration)}）はフレーム数（{frames}）と同じか、"
                         "1 つ（全フレーム共通）にしてください")
    if args.gif and args.motion is None and args.motion_spec is None and not args.frames and not args.talk_only:
        parser.error("--gif は --motion breathe などか --motion-spec か --frames と一緒に指定してください（動きの GIF を書きます）")
    if not args.gif:
        given = _option_names(args, ("gif_scale", "gif_background"))
        if given:
            parser.error(f"{' / '.join(given)} は GIF のオプションです。--gif と一緒に指定してください")
    if args.spec_height is not None and args.write_motion_spec is None:
        parser.error("--spec-height は --write-motion-spec と一緒に指定してください")
    if args.no_cleanup and args.cleanup not in (None, "off"):
        parser.error(f"--cleanup {args.cleanup} と --no-cleanup は同時に指定できません")
    if args.no_outline and args.outline not in (None, "off"):
        parser.error(f"--outline {args.outline} と --no-outline は同時に指定できません")
    if outline_mode(args) == "off":
        given = _option_names(args, OUTLINE_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は縁取りのオプションです。縁取りをしないとき（--no-outline / --outline off）"
                         "には指定できません")
    if not args.simplify:
        given = _option_names(args, SIMPLIFY_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は単純化のオプションです。--no-simplify と一緒には指定できません")
    if args.input_type == "pixel":
        given = _option_names(args, SIZE_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は使えません: ドット絵風の入力（--input-type pixel、既定）では、"
                         "出力の大きさはグリッド検出で決まります（1 ドット＝1 画素）。"
                         "ドットの大きさを指定するには --dot-size を、決まった大きさに縮小するには "
                         "--input-type illust を使ってください")
        given = _option_names(args, RESIZE_OPTIONS)
        if given:
            parser.error(f"{' / '.join(given)} は縮小のオプションです。--input-type illust と一緒に指定してください")
        return
    given = _option_names(args, GRID_OPTIONS)
    if given:
        parser.error(f"{' / '.join(given)} はグリッド検出のオプションです。"
                     "--input-type pixel（既定）と一緒に指定してください")
    if args.scale is None and args.width is None and args.height is None:
        parser.error("普通のイラスト（--input-type illust）では --width / --height / --scale のいずれかを指定してください")
    if args.scale is not None and (args.width is not None or args.height is not None):
        parser.error("--scale と --width / --height は同時に指定できません")
    if args.line_priority and (args.resize_method or DEFAULT_RESIZE_METHOD) not in PIXEL_METHODS:
        parser.error("--line-priority は --resize-method mode / median と一緒に指定してください")


def collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(f for f in path.iterdir()
                                if f.is_file() and f.suffix.lower() in IMAGE_EXTS))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"入力が見つかりません: {path}")
    return files


def output_paths(inputs: list[Path], output: Path | None) -> list[Path]:
    if output is not None and output.suffix.lower() == ".png":
        if len(inputs) != 1:
            raise ValueError("入力が複数のときは -o にフォルダを指定してください")
        return [output]
    folder = output if output is not None else DEFAULT_OUT
    names = [f.stem for f in inputs]
    # 同じ名前の入力（拡張子違いなど）が上書きし合わないようにする
    return [folder / (f"{f.stem}_{f.suffix.lstrip('.').lower()}.png" if names.count(f.stem) > 1
                      else f"{f.stem}.png") for f in inputs]


def palette_paths(saves: list[Path] | None, src: Path, many: bool) -> list[Path]:
    """--save-palette の保存先（入力が複数のときは <名前>_<入力の名前>.<拡張子>）。"""
    if not saves:
        return []
    return [s.with_name(f"{s.stem}_{src.stem}{s.suffix}") if many else s for s in saves]


def print_warnings(warnings: list[str], src: Path) -> int:
    """パレットに合わせたときの警告を表示し、警告の数を返す（-q でも表示する）。"""
    for m in warnings:
        print(f"警告（{src.name}）: {m}", file=sys.stderr)
    return len(warnings)


def motion_path(output: Path, name: str) -> Path:
    """動きのシートの保存先（出力と同じフォルダの <出力名>_<動き>_sheet.png。情報は同じ名前の .json）。"""
    return output.with_name(f"{output.stem}_{name}_sheet.png")


def spec_out_path(path: Path, dst: Path, many: bool) -> Path:
    """--write-motion-spec の保存先（入力が複数のときは <名前>_<出力の名前>.json）。"""
    return path.with_name(f"{path.stem}_{dst.stem}{path.suffix}") if many else path


def load_motion_spec(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--motion-spec の表を読んで確かめる（書き間違いは変換の前に、引数の間違いと同じく止める）。"""
    args.spec = None
    if args.motion_spec is not None:
        try:
            args.spec = motion_mod.load_spec(args.motion_spec)
        except motion_mod.SpecError as e:
            parser.error(f"--motion-spec {args.motion_spec}: {e}")


def skipped_path(output: Path, name: str) -> Path:
    """動きを作らなかったときの理由の JSON の保存先（<出力名>_<動き>_skipped.json）。"""
    return output.with_name(f"{output.stem}_{name}_skipped.json")


def write_motion(args: argparse.Namespace, sprite, dst: Path, src: Path, many: bool = False,
                 skipped: list | None = None) -> int:
    """--motion / --motion-spec の動きを作ってシートを保存する。付けられなかったら警告を出して 1 を返す（変換の結果はそのまま）。

    `skipped` は変換の工程の作らなかった理由の記録（`convert_sprite()` の結果の `skipped`）。シートの JSON の `skipped` に入る。
    付けられなかったときはシートを書かず、理由の記録を <出力名>_<動き>_skipped.json に書く。

    指定がこの画像に合わない（--breathe-split / --sway-split・表の高さが画像の高さを超える）ときはエラーを表示して 2 を返す。
    --write-motion-spec があれば、使った表も書く（付けられなかったときも、埋めた表は書く）。
    """
    if getattr(args, "motion_spec", None) is not None:
        spec = getattr(args, "spec", None) or motion_mod.load_spec(args.motion_spec)
        try:
            res = motion_mod.make_motion_from_spec(sprite, spec)
        except ValueError as e:
            print(f"エラー（{src.name}）: {e}", file=sys.stderr)
            return 2
        label = f"動きの表 {args.motion_spec.name}（{res.name}）"
    else:
        res = _preset_motion(args, sprite, src)
        if res is None:
            return 2
        label = motion_mod.MOTIONS[args.motion]
    for m in res.warnings:
        print(f"警告（{src.name}）: {m}", file=sys.stderr)
    if getattr(args, "write_motion_spec", None) is not None and res.spec is None:
        print(f"警告（{src.name}）: 埋めた動きの表が無いので、{args.write_motion_spec.name} は書きませんでした"
              "（分ける高さを決められなかったため）", file=sys.stderr)
    if getattr(args, "write_motion_spec", None) is not None and res.spec is not None:
        sp = spec_out_path(args.write_motion_spec, dst, many)
        data = motion_mod.with_height_style(res.spec, sprite, args.spec_height or "number")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(motion_mod.spec_json(data), encoding="utf-8")
        if not args.quiet:
            print(f"  → {sp}（動きの表 {len(data['frames'])} 行）")
    if res.skipped:
        rp = skipped_path(dst, res.name)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(skipped_json(res, skipped), encoding="utf-8")
        if not args.quiet:
            print(f"  → {rp}（作らなかった理由 {len(res.skips) + len(skipped or [])} 件）")
        if getattr(args, "gif", False):
            print(f"警告（{src.name}）: 動きを作らなかったので、GIF も書きませんでした", file=sys.stderr)
        return 1
    base_res = res
    if getattr(args, "swaps", None) is not None:
        res = apply_swaps_cli(args, res, sprite, src)
        if res is None:
            return 1
    png, js = motion_mod.save_sheet(res, motion_path(dst, res.name), skipped)
    if not args.quiet:
        w, h = res.size
        parts = []
        pad = res.info.get("padding", 0)
        if pad:
            parts.append(f"、揺れてはみ出さないように左右に {pad} 列ずつ余白を足した")
        offs = res.info.get("offset_y")
        if offs is None and res.name in (float_mod.NAME, bounce_mod.NAME):
            offs = res.info.get("offsets")
        if "padding_y" in res.info and offs is not None:   # 浮遊・弾む: 上下に足した行と、各フレームの上下のずれ
            top, bottom = res.info["padding_y"]
            parts.append(f"、上下のずれ {'/'.join(f'{d:+d}' for d in offs)}"
                         + (f"、はみ出さないように上に {top} 行・下に {bottom} 行の余白を足した" if top or bottom else ""))
        elif "padding_y" in res.info and res.info["padding_y"][0]:   # 伸び: 上に足した行
            parts.append(f"、伸びてはみ出さないように上に {res.info['padding_y'][0]} 行の余白を足した")
        if res.name == stretch_mod.NAME and not getattr(args, "motion_spec", None):   # 伸び: 継ぎ目
            seams = "・".join(f"{a}〜{b}" for a, b in res.info["seam_from_bottom"])
            parts.append(f"、継ぎ目は下から {seams} ドット目")
        neck = res.info.get("neck") if res.name == nod_mod.NAME else None
        if res.name == nod_mod.NAME:   # うなずく: 分けた高さ（自動なら見つけた首も）と継ぎ目
            at = res.spec["regions"]["upper"]["at"]
            if isinstance(at, str):   # 割合: キャラの高さに対する割合と、それが指す「下から何ドット目」
                found_at = f"{at}（下から {motion_mod.spec.percent_to_height(sprite, at)} ドット目）"
            else:
                found_at = f"下から {at} ドット目"
            seams = "・".join(f"{a}〜{b}" for a, b in res.info["seam_from_bottom"])
            found = (f"首を自動で下から {neck['from_bottom']} ドット目に見つけ、" if neck is not None else "")
            parts.append(f"、{found}{found_at}で分けた（継ぎ目は下から {seams} ドット目）")
        print(f"  → {png}（{label}: {len(res.frames)} フレーム、{w}×{h}、"
              f"{'/'.join(str(d) for d in res.durations)} ミリ秒{''.join(parts)}）")
        print_swaps(res)
        print(f"  → {js}")
    failed = 0
    if getattr(args, "gif", False):
        failed |= write_gif(args, res, png, src)
    if getattr(args, "fx", None) is not None:
        failed |= write_fx(args, res, png, src)
    if getattr(args, "swaps", None) is not None:
        failed |= write_talk(args, base_res, sprite, dst, src, skipped)
    return failed


def write_fx(args: argparse.Namespace, anim, png: Path, src: Path) -> int:
    """--fx: キャラのシート（`anim`。保存した PNG は `png`）に重ねるエフェクトのシート・JSON（--gif なら重ねた GIF、
    --piskel なら 2 つのレイヤーの .piskel も）を書く。作れなければ警告を出して 1 を返す。"""
    try:
        fx = fx_mod.make_fx(args.fx, anim.frames, anim.name.endswith("_mirror"), tuple(args.fx_offset or (0, 0)),
                            args.fx_end)
    except ValueError as e:
        print(f"警告（{src.name}）: エフェクト（{args.fx}）を作れませんでした: {e}", file=sys.stderr)
        return 1
    name = fx_mod.fx_file_name(png.name, fx)
    fp = png.with_name(name)
    save_rgba(fx_mod.fx_sheet(fx), fp)
    fp.with_suffix(".json").write_text(fx_mod.fx_json(fx, name, png.name), encoding="utf-8")
    if not args.quiet:
        w, h = fx.size
        x, y = fx.position
        end = "くり返す" if fx.loop else ("最後の形で止める" if fx.end == "keep" else "最後に消える")
        print(f"  → {fp}（重ね {fx.label}: {len(fx.frames)} コマ、{w}×{h}、{'/'.join(str(d) for d in fx.durations)} ミリ秒、"
              f"{end}。キャラのコマの左上から x {x:+d}・y {y:+d} ドットに置く{'（左右反転）' if fx.mirrored else ''}）")
        print(f"  → {fp.with_suffix('.json')}")
    failed = 0
    stem = fx_mod.fx_stem(png.name, fx.name)
    if getattr(args, "gif", False):
        gp = png.with_name(stem + ".gif")
        try:
            gif = fx_mod.combined_gif(anim.frames, anim.durations, fx, args.gif_scale or gif_mod.DEFAULT_SCALE,
                                      args.gif_background)
        except ValueError as e:
            print(f"警告（{src.name}）: 重ねた GIF を書けませんでした: {e}", file=sys.stderr)
            failed = 1
        else:
            gp.write_bytes(gif.data)
            for m in gif.warnings:
                print(f"警告（{src.name}）: {m}", file=sys.stderr)
            if not args.quiet:
                print(f"  → {gp}（キャラとエフェクトを重ねた GIF: {len(gif.durations)} フレーム、合計 {sum(gif.durations)} ミリ秒。確かめ用）")
    if getattr(args, "piskel", None):
        text, notes = fx_mod.combined_piskel(anim.frames, anim.durations, fx, stem)
        if text is None:
            for m in notes:
                print(f"警告（{src.name}）: {m}", file=sys.stderr)
        else:
            pp = png.with_name(stem + ".piskel")
            pp.write_text(text, encoding="ascii")
            if not args.quiet:
                print(f"  → {pp}（レイヤー「キャラ」「エフェクト」の .piskel）")
                for m in notes:
                    print(f"    メモ: {m}")
    return failed


def write_talk(args: argparse.Namespace, anim, sprite, dst: Path, src: Path, skipped: list | None = None) -> int:
    """--swaps に口パク（talk）があれば、しゃべりのシート <出力名>_<動き>_talk_sheet.png / .json（--gif なら GIF も）を書く。
    `anim` はまばたきを入れる前の動きの結果。口パクが無ければ何もしない。作れなければ警告を出して 1 を返す。"""
    from retrocore.sprite.swap import SwapError, apply_talk, talk_summary
    swaps = load_swaps_cli(args)
    if swaps is None or swaps.find("talk") is None:
        return 0
    try:
        talk, rep = apply_talk(anim, sprite, swaps)
    except SwapError as e:
        print(f"エラー（{src.name}）: 口パク（{args.swaps.name}）を当てられません: {e}。"
              "変換の設定を描いたときと同じにするか、差し替えを描き直してください", file=sys.stderr)
        return 1
    print_warnings(rep.warnings, src)
    if talk is None:
        return 1
    png, js = motion_mod.save_sheet(talk, motion_path(dst, talk.name), skipped)
    if not args.quiet:
        w, h = talk.size
        print(f"  → {png}（しゃべりのシート: {len(talk.frames)} フレーム、{w}×{h}、合計 {sum(talk.durations)} ミリ秒）")
        print("    " + talk_summary(rep))
        print(f"  → {js}")
    if getattr(args, "gif", False):
        return write_gif(args, talk, png, src)
    return 0


def load_swaps_cli(args: argparse.Namespace, quiet: bool = False):
    """--swaps のファイルを読む（1 回だけ）。読めなければエラーを表示して None（`quiet` なら表示しない）。"""
    if getattr(args, "_swaps", None) is None:
        from retrocore.sprite.swap import load_swaps
        try:
            args._swaps = load_swaps(args.swaps.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            if not quiet:
                print(f"エラー（{args.swaps.name}）: 差し替えの一覧を読めません: {e}", file=sys.stderr)
            return None
    return args._swaps


def apply_swaps_cli(args: argparse.Namespace, anim, ref, src: Path, mirror_anim=None, palette=None):
    """--swaps: シートの材料に差し替えを入れる。当てられなければエラーを表示して None（シートは書かない）。
    `mirror_anim` があれば (反転していない方, 反転した方) を返す。"""
    from retrocore.sprite.swap import SwapError, apply_swaps, mirror_swapped
    swaps = load_swaps_cli(args)
    if swaps is None:
        return None
    try:
        out, rep = apply_swaps(anim, ref, swaps, palette)
    except SwapError as e:
        print(f"エラー（{src.name}）: 差し替え（{args.swaps.name}）を当てられません: {e}。"
              "変換の設定を描いたときと同じにするか、差し替えを描き直してください", file=sys.stderr)
        return None
    if rep is None:   # まばたきが無い（口パクだけ）: ふつうのシートは今までと同じ
        return (out, mirror_anim) if mirror_anim is not None else out
    print_warnings(rep.warnings, src)
    if mirror_anim is not None:
        return out, mirror_swapped(out, mirror_anim)
    return out


def frames_talk(args: argparse.Namespace, anims: list, res, src: Path):
    """--frames --swaps: 口パクがあれば、しゃべりのシートの結果（反転も）を返す。無ければ []、作れなければ None（警告を出す）。"""
    import dataclasses

    from retrocore.sprite.swap import SwapError, apply_talk, mirror_swapped
    from retrocore.sprite.touchup import palette_of
    swaps = load_swaps_cli(args)
    if swaps is None or swaps.find("talk") is None:
        return []
    try:
        talk, rep = apply_talk(anims[0], res.frames[0], swaps, palette_of(res.frames))
    except SwapError as e:
        print(f"エラー（{src.name}）: 口パク（{args.swaps.name}）を当てられません: {e}。"
              "変換の設定を描いたときと同じにするか、差し替えを描き直してください", file=sys.stderr)
        return None
    print_warnings(rep.warnings, src)
    if talk is None:
        return None
    out = [talk]
    if len(anims) > 1:
        out.append(mirror_swapped(talk, dataclasses.replace(anims[1], name=talk.name + "_mirror")))
    return out


def print_talk(res) -> None:
    """しゃべりのシートの 1 行（周の数・口の切り替え・まばたき）。"""
    from retrocore.sprite.swap import SwapReport, talk_summary
    sw = res.info["swaps"]
    print("    しゃべり: " + talk_summary(SwapReport(sw["interval_ms"], sw["cycle_ms"], sw["repeats"], sw["items"])))


def print_swaps(res) -> None:
    """差し替えを入れたときの 1 行（周の数・入れた時刻とフレーム）。"""
    if "swaps" in res.info:
        from retrocore.sprite.swap import SwapReport, swap_summary
        sw = res.info["swaps"]
        print("    差し替え: " + swap_summary(SwapReport(sw["interval_ms"], sw["cycle_ms"], sw["repeats"], sw["items"])))


def write_gif(args: argparse.Namespace, res, sheet_png: Path, src: Path) -> int:
    """--gif: シートと同じフレーム・表示時間の GIF を <出力名>_<動き>.gif に書く。作れなければ警告を出して 1 を返す。"""
    gp = gif_mod.gif_path(sheet_png)
    scale = args.gif_scale or gif_mod.DEFAULT_SCALE
    try:
        gif = gif_mod.save_gif(res, gp, scale, args.gif_background)
    except ValueError as e:
        print(f"警告（{src.name}）: GIF を書けませんでした: {e}", file=sys.stderr)
        return 1
    for m in gif.warnings:
        print(f"警告（{src.name}）: {m}", file=sys.stderr)
    if not args.quiet:
        w, h = res.size
        bg = "透明" if gif.background is None else "背景 #%02x%02x%02x" % gif.background
        merged = "".join(f"、{'・'.join(str(i + 1) for i in g)} 枚目は同じ絵なので 1 枚にまとめた" for g in gif.merged)
        print(f"  → {gp}（GIF: {len(gif.durations)} フレーム、{w * scale}×{h * scale}（{scale} 倍）、{bg}、"
              f"{gif.colors} 色、{'/'.join(str(d) for d in gif.durations)} ミリ秒、ずっと繰り返す{merged}）")
    return 0


def _preset_motion(args: argparse.Namespace, sprite, src: Path):
    """--motion のプリセットで動きを作る。指定がこの画像に合わなければエラーを表示して None。"""
    durations = args.motion_duration
    if durations is not None and len(durations) == 1:
        durations = durations[0]
    options = {}
    if args.breathe_depth is not None:
        options["depth"] = args.breathe_depth
    if args.breathe_split is not None:
        options["split"] = args.breathe_split
    if args.sway_amount is not None:
        options["amount"] = args.sway_amount
    if args.sway_split is not None:
        options["split"] = args.sway_split
    if args.float_amount is not None:
        options["amount"] = args.float_amount
    if args.bounce_height is not None:
        options["height"] = args.bounce_height
    if args.nod_depth is not None:
        options["depth"] = args.nod_depth
    if args.nod_split is not None:
        options["split"] = args.nod_split
    if args.stretch_amount is not None:
        options["amount"] = args.stretch_amount
    if args.stretch_split is not None:
        options["split"] = args.stretch_split
    if args.keep_small_parts:
        options["keep_small_parts"] = True
    try:
        return motion_mod.make_motion(sprite, args.motion, frames=args.motion_frames, durations=durations, **options)
    except ValueError as e:
        print(f"エラー（{src.name}）: {e}", file=sys.stderr)
        return None


def run_join(args: argparse.Namespace, settings: SpriteSettings, inputs: list[Path]) -> int:
    """--frames --join: 入力のファイルを書いた順につなげて 1 つの動きにする。"""
    if len(inputs) > join_mod.MAX_FILES:
        print(f"エラー: つなげるファイルは {join_mod.MAX_FILES} 個までです（いま {len(inputs)} 個）", file=sys.stderr)
        return 1
    images = []
    for src in inputs:
        try:
            images.append((src.name, load_rgba(src)))
        except OSError as e:
            print(f"読み込めません: {src}（{e}）", file=sys.stderr)
            return 1
    dst = output_paths(inputs[:1], args.output)[0]
    src = inputs[0]
    steps_dir = dst.with_name(dst.stem + "_steps")

    def on_step(res, _i=[0]):
        _i[0] += 1
        if not args.quiet:
            h, w = res.image.shape[:2]
            print(f"  {_i[0]}. {res.description}: {w}×{h}（{res.seconds:.2f} 秒）")
        if args.save_steps:
            save_rgba(res.image, steps_dir / f"{_i[0]:02d}_{res.name}.png")

    if not args.quiet:
        print("つなげる: " + " ＋ ".join(f"{p}（{a.shape[1]}×{a.shape[0]}）" for p, (_, a) in zip(inputs, images))
              + f"、{INPUT_TYPES[args.input_type]}")
    return write_frames(args, images, settings, dst, src, on_step)


def write_frames(args: argparse.Namespace, images: list, settings: SpriteSettings, dst: Path, src: Path, on_step,
                 many: bool = False) -> int:
    """--frames: コマに切り分けてそろえ、シートなどを書く。作らなかったときは理由を書いて 1 を返す。

    `images` は (ファイルの名前, RGBA) の並び。2 つ以上（--join）なら、並べた順につなげて 1 つの動きにする。
    """
    name = args.frame_name or frames_mod.DEFAULT_NAME
    fs = frames_mod.FrameSettings(frame_count=args.frame_count, align=args.align or frames_mod.DEFAULT_ALIGN,
                                  align_y=args.align_y or frames_mod.DEFAULT_ALIGN_Y,
                                  grid=args.frame_grid or frames_mod.DEFAULT_GRID_MODE,
                                  steady=args.steady or steady_mod.DEFAULT_MODE,
                                  steady_size=args.steady_size or steady_mod.DEFAULT_SIZE, steady_rows=args.steady_rows,
                                  same_ranges=tuple(args.same_ranges or ()),
                                  same_search=same_mod.DEFAULT_SEARCH if args.same_search is None else args.same_search,
                                  same_from=args.same_from or 1,
                                  shift_y=None if args.frame_shift_y is None else tuple(args.frame_shift_y))
    if args.edits is not None:
        from retrocore.sprite.touchup import load_touchup
        try:
            fs.edits = load_touchup(args.edits.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"エラー（{args.edits.name}）: 手の直しの一覧を読めません: {e}", file=sys.stderr)
            return 1
    try:
        res = join_mod.convert_frame_files(images, settings, fs, on_step=on_step,
                                           match_height=bool(args.join_match_height))
    except ValueError as e:
        from retrocore.sprite.touchup import TouchupError
        if isinstance(e, TouchupError):
            print(f"エラー（{src.name}）: 手の直し（{args.edits.name}）を当てられません: {e}。"
                  "変換の設定を直したときと同じにするか、直しを作り直してください", file=sys.stderr)
            return 1
        if not isinstance(e, (frames_mod.FramesError, GridNotFoundError)):
            print(f"エラー（{src.name}）: {e}", file=sys.stderr)
            return 1
        print(f"コマを作りませんでした（{src.name}）: {e}", file=sys.stderr)
        print(f"コマを作りませんでした（{src.name}）: {e}", file=sys.stderr)
        rp = skipped_path(dst, name)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(frames_mod.frames_skipped_json(e, name), encoding="utf-8")
        if not args.quiet:
            print(f"  → {rp}（作らなかった理由）")
        return 1
    if not args.quiet:
        for m in res.notes:
            print(f"  メモ: {m}")
        if res.joined:
            print(f"  つなげた {len(res.files)} ファイル（全部で {len(res.frames)} コマ）:")
            for f in res.files:
                adj = {"adjusted": f"、高さ {f.height} → {f.height_used} ドット（格子 {f.scale:.3f} 倍）",
                       "too_far": f"、高さ {f.height} ドット（違いが大きいので合わせていない）",
                       "off": f"、高さ {f.height} ドット（合わせていない）"}.get(f.status, f"、高さ {f.height} ドット")
                last = f.first + f.frame_count - 1
                print(f"    {f.name}: {f.frame_count} コマ（{f.first}" + (f"〜{last}" if last > f.first else "")
                      + f" コマ目）、ドット {f.dot_size[0]:.2f}px{adj}")
        if res.same is not None:
            ok, total = res.same.copied_count()
            print(f"  1 コマ目と同じにした（範囲 {len(res.same.ranges)} 個、写した {ok}/{total}）:")
            for sf in res.same.frames:
                for rr in sf.ranges:
                    print(f"    {sf.frame} コマ目・範囲 {rr.range_index}: ずれ {rr.shift[0]:+d},{rr.shift[1]:+d}・"
                          f"合い方 {rr.match * 100:.0f}% → " + (f"写した（変えたドット {rr.changed}"
                                                               + (f"、残した部品 {rr.kept}" if rr.kept else "") + "）"
                                                               if rr.copied else f"写さなかった（{rr.code}）"))
        if res.steady is not None:
            print(f"  ちらつきを減らした（上から {res.steady.rows} 行、小さい {res.steady.size} ドット以下）: "
                  + "、".join(f"{f.frame} コマ目 戻した {f.restored_count}・残した {f.kept_count}" for f in res.steady.frames))
        if res.touchup is not None:
            print(f"  手の直しを当てた（直したドット {res.touchup.dot_count}）: "
                  + "、".join(f"{f.label} コマ目 {len(f.dots)}" for f in res.touchup.frames))
    if res.touchup is not None:
        print_warnings(res.touchup.warnings, src)
    strip = frames_mod.source_strip(res)
    save_rgba(strip, dst)
    if res.same is not None:
        from retrocore.sprite.same import same_image
        sp = dst.with_name(f"{dst.stem}_{name}_same.png")
        before = res.aligned
        save_rgba(same_image(before, frames_mod.same_view_frames(res), res.same), sp)
        if not args.quiet:
            print(f"  → {sp}（範囲の図。緑の枠 = 写した位置、赤の枠 = 写さなかった、橙 = 境目の問題、青 = 残した部品）")
    if not args.quiet:
        w, h = res.size
        print(f"  → {dst}（そろえた {len(res.frames)} コマを元の順に並べた絵。1 コマ {w}×{h}）")
    print_warnings(res.warnings, src)
    for pp in palette_paths(args.save_palette, dst, many):
        cols = save_palette(used_colors(strip), pp)
        if not args.quiet:
            print(f"  → {pp}（パレット {len(cols)} 色。全部のコマで共通）")
    if args.preview_scale is not None:
        pv = preview_path(dst, args.preview_scale)
        save_rgba(scale_nearest(strip, args.preview_scale), pv)
        if not args.quiet:
            print(f"  → {pv}（プレビュー {args.preview_scale} 倍）")
    try:
        anims = [frames_mod.assemble_frames(res, args.frame_order, args.frame_duration, args.frame_root_y, name)]
        if args.mirror:
            anims.append(frames_mod.assemble_frames(res, args.frame_order, args.frame_duration, args.frame_root_y,
                                                    name, mirror=True))
    except ValueError as e:
        print(f"エラー（{src.name}）: {e}", file=sys.stderr)
        return 1
    fw = frames_mod.flip_warning(args.frame_order)
    if fw:
        print_warnings([fw], src)
    failed = 0
    talks = []
    if args.swaps is not None:
        from retrocore.sprite.swap import base_warning
        from retrocore.sprite.touchup import palette_of
        sw = load_swaps_cli(args, quiet=True)
        msg = base_warning(sw, res.sources if res.joined else None) if sw is not None else None
        if msg:
            print_warnings([msg], src)
        talks = frames_talk(args, anims, res, src)
        if talks is None:
            failed, talks = 1, []
        got = apply_swaps_cli(args, anims[0], res.frames[0], src, anims[1] if len(anims) > 1 else None,
                              palette_of(res.frames))
        if got is None:
            return 1
        anims = list(got) if isinstance(got, tuple) else [got]
    for anim in anims + talks:
        png, js = frames_mod.save_frames_sheet(anim, dst.with_name(frames_mod.sheet_file_name(dst.stem, anim)),
                                               res.skipped, res.dot_size)
        if not args.quiet:
            w, h = anim.size
            order = frames_mod.order_text(anim.info["source_frames"], anim.info.get("flipped"))
            extra = ""
            if "offset_y" in anim.info:
                top, bottom = anim.info["padding_y"]
                extra = (f"、体全体の上下 {'/'.join(f'{d:+d}' for d in anim.info['offset_y'])}"
                         + (f"（上に {top} 行・下に {bottom} 行の余白を足した）" if top or bottom else ""))
            if "padding_x" in anim.info and any(anim.info["padding_x"]):
                a, b = anim.info["padding_x"]
                extra += f"、裏返したコマのために左に {a} 列・右に {b} 列を足した"
            if anim.info.get("ground"):
                air = frames_mod.frames_sheet_info(anim, png.name)["frames"]
                extra += f"、足元の高さ {'/'.join(str(f['air']) for f in air)} ドット"
            print(f"  → {png}（{'左右反転、' if anim.name.endswith('_mirror') else ''}{len(anim.frames)} フレーム、"
                  f"コマ {order}、{w}×{h}、{'/'.join(str(d) for d in anim.durations)} ミリ秒{extra}）")
            print(f"  → {js}")
            print_talk(anim) if any(anim is t for t in talks) else print_swaps(anim)
            print(f"    （Piskel で直すときは: Import → Import as spritesheet → 幅 {w}・高さ {h}。{len(anim.frames)} コマ）")
        if args.piskel:
            from retrocore.sprite.piskel import fps_for_durations, piskel_text
            fps, fps_notes = fps_for_durations(anim.durations)
            pp = png.with_suffix(".piskel")
            pp.write_text(piskel_text(anim.frames, png.stem, fps), encoding="ascii")
            if not args.quiet:
                print(f"  → {pp}（Piskel の「Open」で開ける形。{fps} FPS）")
                for m in fps_notes:
                    print(f"    メモ: {m}")
        if args.gif:
            failed |= write_gif(args, anim, png, src)
        if args.fx is not None and not any(anim is t for t in talks):
            failed |= write_fx(args, anim, png, src)
    try:
        rep = frames_mod.flicker_report(res.frames, args.flicker_rows)
    except ValueError as e:
        print(f"エラー（{src.name}）: {e}", file=sys.stderr)
        return 1
    if not args.quiet:
        counts = "、".join(f"{f.frame} コマ目 {f.count}" for f in rep.frames[1:])
        print(f"  ちらつき（上から {rep.rows} 行で 1 コマ目と違うドット）: {counts or 'コマが 1 つだけ'}")
    if args.flicker:
        fp = dst.with_name(f"{dst.stem}_{name}_flicker.png")
        save_rgba(frames_mod.flicker_image(res.frames, rep, file_starts=join_mod.file_starts(res)), fp)
        fp.with_suffix(".json").write_text(frames_mod.flicker_json(rep), encoding="utf-8")
        if not args.quiet:
            print(f"  → {fp}（ちらつきの見える化。赤紫の枠が 1 コマ目と違うドット、水色の線から上が数えた範囲"
                  + ("、橙の線と f1・f2… がファイルの境目" if res.joined else "") + "）")
            print(f"  → {fp.with_suffix('.json')}")
    return 1 if failed else 0


def check_readback_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--readback と一緒に使えないオプション・入力の数を確かめる。"""
    defaults = build_parser().parse_args(["x"])
    given = [f"--{k.replace('_', '-')}" for k, v in vars(args).items()
             if k not in READBACK_ALLOWED + READBACK_OPTIONS and k in vars(defaults) and v != getattr(defaults, k)]
    if given:
        parser.error(f"{' / '.join(given)} は --readback と一緒に使えません（読み戻しは変換をやり直さず、直したドットをそのまま使います）")
    if len(args.inputs) != 1 or args.inputs[0].is_dir():
        parser.error("--readback の入力は、直したシート 1 つ（PNG か .piskel）にしてください")
    if args.output is not None and args.output.suffix.lower() == ".png":
        parser.error("--readback の -o は書くフォルダです（ファイルの名前は元の JSON から決めます）")
    if not args.gif:
        given = _option_names(args, ("gif_scale", "gif_background"))
        if given:
            parser.error(f"{' / '.join(given)} は GIF のオプションです。--gif と一緒に指定してください")


def run_readback(args: argparse.Namespace) -> int:
    """--readback: 直したシートを読み戻して、JSON・GIF などを作り直す。作らなかったときは理由を表示して 1 を返す。"""
    src = args.inputs[0]
    try:
        edited = src.read_bytes()
        info_text = args.readback.read_bytes()
        original = args.original.read_bytes() if args.original is not None else None
    except OSError as e:
        print(f"読み込めません: {e}", file=sys.stderr)
        return 1
    opt = edit_mod.ReadbackOptions(gif=bool(args.gif), gif_scale=args.gif_scale or gif_mod.DEFAULT_SCALE,
                                   gif_background=args.gif_background, mirror=bool(args.mirror), piskel=bool(args.piskel),
                                   same_frames=args.same_frames or edit_mod.DEFAULT_SAME_FRAMES,
                                   fix_colors=bool(args.fix_colors), anim=args.anim)
    res = edit_mod.readback(edited, src.name, info_text, original, opt)
    if not args.quiet:
        print(f"{src}（直したシート）＋ {args.readback}（元の情報）"
              + (f" ＋ {args.original}（直す前のシート）" if args.original is not None else ""))
        for m in res.notes:
            print(f"  メモ: {m}")
    if not res.made:
        for m in res.errors:
            print(f"作り直しませんでした（{src.name}）: {m}", file=sys.stderr)
        return 1
    out_dir = args.output if args.output is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in res.files.items():
        (out_dir / name).write_bytes(data)
        if not args.quiet:
            print(f"  → {out_dir / name}")
    for m in res.warnings:
        print(f"警告（{src.name}）: {m}", file=sys.stderr)
    if not args.quiet:
        r = res.report
        fr = res.info["frames"]
        if r.get("animation"):   # まとめたシート: 直した動きのフレームだけ
            fr = [fr[i] for i in res.info["animations"][r["animation"]]["frames"]]
        print(f"  {'動き ' + r['animation'] + ': ' if r.get('animation') else ''}"
              f"{r['frame_count']} コマ（1 コマ {r['frame_width']}×{r['frame_height']}）、"
              f"{'/'.join(str(f['duration_ms']) for f in fr)} ミリ秒（元の JSON から）、"
              f"新しい色 {len(r['new_colors']) if r['palette_source'] else '（調べていない）'}、"
              f"半透明 {r['semi_transparent']['dots']} ドット")
    return 0


# 変換の設定の引数（anims.json に settings があるときは使えない）
CONVERT_OPTIONS = ("input_type", "width", "height", "scale", "background", "bg_color", "bg_tolerance", "bg_remove_enclosed",
                   "dot_size", "dot_shape", "grid_origin", "grid_snap", "grid_tolerance", "resize_method", "mode_bin",
                   "line_priority", "line_ratio", "line_luma", "transparent_ratio", "simplify", "colors", "merge_distance",
                   "keep_large", "cleanup", "no_cleanup", "outline", "no_outline", "outline_color", "outline_shape",
                   "outline_colors", "outline_pad", "palette")
# 動きごとの設定（anims.json に書く）
ANIMS_OWN_OPTIONS = ("frame_count", "align_y", "frame_order", "frame_duration", "frame_name", "frame_shift_y", "join")
# 動きをまとめるときは使えないもの（警告して無視する）と、その理由（anims.UNSUPPORTED の鍵）
ANIMS_IGNORED = {"swaps": "swaps", "fx": "fx", "fx_offset": "fx", "fx_end": "fx", "same_ranges": "same",
                 "same_search": "same", "same_from": "same", "steady": "steady", "steady_size": "steady",
                 "steady_rows": "steady", "flicker": "flicker", "flicker_rows": "flicker", "frame_root_y": "root_y"}


def check_anims_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """--anims の設定を読んで、引数の組み合わせを確かめる。戻り値は (設定, 入力の [(名前, パス)], 無視したものの警告)。"""
    try:
        cfg = anims_mod.load_config(args.anims.read_bytes())
    except OSError as e:
        parser.error(f"--anims {args.anims} を読めません: {e}")
    except ValueError as e:
        parser.error(f"--anims {args.anims}: {e}")
    if args.motion is not None or args.motion_spec is not None or args.write_motion_spec is not None:
        parser.error("--anims は --motion / --motion-spec / --write-motion-spec と一緒に使えません")
    given = _option_names(args, ANIMS_OWN_OPTIONS)
    if given:
        parser.error(f"{' / '.join(given)} は --anims では使えません（順番・時間・縦のそろえ方・上げ下げ・名前は、"
                     f"動きごとに {args.anims.name} に書きます）")
    defaults = build_parser().parse_args([])
    if cfg.settings is not None:
        given = [f"--{k.replace('_', '-')}" for k in CONVERT_OPTIONS if getattr(args, k) != getattr(defaults, k)]
        if given:
            parser.error(f"{' / '.join(given)} は使えません（{args.anims.name} に変換の設定（settings）が書いてあります。"
                         "設定のファイルの settings を直すか、settings を消して引数で指定してください）")
    ignored = []
    for k, why in ANIMS_IGNORED.items():
        v = getattr(args, k)
        if v is not None and v != getattr(defaults, k) and not (k == "steady" and v == "off"):
            msg = anims_mod.UNSUPPORTED[why]
            if msg not in ignored:
                ignored.append(msg)
            setattr(args, k, getattr(defaults, k))
    args.frames = True
    check_args(parser, args)
    # 入力の画像: 書いたものは名前で探す。書かなければ設定のファイルのあるフォルダから
    try:
        found = {p.name: p for p in collect_inputs(args.inputs)} if args.inputs else {}
    except FileNotFoundError as e:
        parser.error(str(e))
    files = []
    for f in cfg.files():
        path = found.get(f) if args.inputs else args.anims.parent / f
        if path is None or not path.is_file():
            parser.error(f"{args.anims.name} の「{f}」が見つかりません"
                         + ("（入力に書いた画像の中に、同じ名前のファイルがありません）" if args.inputs
                            else f"（{args.anims.parent} の中を探しました。入力の画像を引数に書いても探せます）"))
        files.append((f, path))
    extra = [n for n in found if n not in cfg.files()]
    if extra:
        parser.error(f"入力の {'、'.join(extra)} は、{args.anims.name} のどの動きにも入っていません")
    return cfg, files, ignored


def run_anims(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """--anims: 動きに名前を付けて 1 つのシートにまとめる。"""
    cfg, files, ignored = check_anims_args(parser, args)
    if args.mirror:
        cfg.mirror = True
    if args.join_match_height:
        cfg.match_height = True
    if args.align is not None:
        cfg.align = args.align
    if args.frame_grid is not None:
        cfg.grid = args.frame_grid
    for sp in args.save_palette or []:
        if sp.suffix.lower() not in palette_mod.WRITE_FORMATS:
            parser.error(f"--save-palette は {' / '.join(palette_mod.WRITE_FORMATS)} のファイルで指定してください: {sp}")
    args.palette_colors = None
    if args.palette is not None:
        try:
            args.palette_colors = load_palette(args.palette)
        except PaletteError as e:
            parser.error(str(e))
    if cfg.settings is not None:
        try:
            settings = SpriteSettings.from_dict(dict(cfg.settings))
            settings.validate()
        except (ValueError, TypeError) as e:
            parser.error(f"--anims {args.anims}: 変換の設定（settings）に問題があります: {e}")
    else:
        settings = settings_from_args(args)
    try:
        dst = output_paths([files[0][1]], args.output)[0]
    except ValueError as e:
        parser.error(str(e))
    src = files[0][1]
    for m in ignored:
        print(f"警告（{args.anims.name}）: {m}（無視します）", file=sys.stderr)
    images = []
    for name, path in files:
        try:
            images.append((name, load_rgba(path)))
        except OSError as e:
            print(f"読み込めません: {path}（{e}）", file=sys.stderr)
            return 1
    steps_dir = dst.with_name(dst.stem + "_steps")

    def on_step(res, _i=[0]):
        _i[0] += 1
        if not args.quiet:
            h, w = res.image.shape[:2]
            print(f"  {_i[0]}. {res.description}: {w}×{h}（{res.seconds:.2f} 秒）")
        if args.save_steps:
            save_rgba(res.image, steps_dir / f"{_i[0]:02d}_{res.name}.png")

    if not args.quiet:
        print("動きをまとめる: " + "、".join(f"{a.name}（{' ＋ '.join(a.files)}）" for a in cfg.anims)
              + f"、{INPUT_TYPES[settings.input_type]}")
    edits = None
    if args.edits is not None:
        from retrocore.sprite.touchup import load_touchup
        try:
            edits = load_touchup(args.edits.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"エラー（{args.edits.name}）: 手の直しの一覧を読めません: {e}", file=sys.stderr)
            return 1
    try:
        res = anims_mod.convert_anims(images, settings, cfg, edits, on_step=on_step)
    except ValueError as e:
        from retrocore.sprite.touchup import TouchupError
        if isinstance(e, TouchupError):
            print(f"エラー（{src.name}）: 手の直し（{args.edits.name}）を当てられません: {e}。"
                  "変換の設定を直したときと同じにするか、直しを作り直してください", file=sys.stderr)
            return 1
        if not isinstance(e, (frames_mod.FramesError, GridNotFoundError)):
            print(f"エラー（{src.name}）: {e}", file=sys.stderr)
            return 1
        print(f"コマを作りませんでした（{src.name}）: {e}", file=sys.stderr)
        rp = skipped_path(dst, cfg.name)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(frames_mod.frames_skipped_json(e, cfg.name), encoding="utf-8")
        if not args.quiet:
            print(f"  → {rp}（作らなかった理由）")
        return 1
    if not args.quiet:
        for m in res.notes:
            print(f"  メモ: {m}")
        for r in res.anims:
            last = r["first"] + r["count"] - 1
            print(f"    {r['name']}: {' ＋ '.join(r['files'])}、{r['count']} コマ（{r['first']}"
                  + (f"〜{last}" if last > r["first"] else "") + " コマ目）")
        if res.touchup is not None:
            print(f"  手の直しを当てた（直したドット {res.touchup.dot_count}）: "
                  + "、".join(f"{f.label} コマ目 {len(f.dots)}" for f in res.touchup.frames))
    if res.touchup is not None:
        print_warnings(res.touchup.warnings, src)
    strip = frames_mod.source_strip(res)
    save_rgba(strip, dst)
    if not args.quiet:
        w, h = res.size
        print(f"  → {dst}（そろえた {len(res.frames)} コマをファイルの順に並べた絵。1 コマ {w}×{h}）")
    print_warnings(res.warnings, src)
    for pp in palette_paths(args.save_palette, dst, False):
        cols = save_palette(used_colors(strip), pp)
        if not args.quiet:
            print(f"  → {pp}（パレット {len(cols)} 色。全部のコマで共通）")
    if args.preview_scale is not None:
        pv = preview_path(dst, args.preview_scale)
        save_rgba(scale_nearest(strip, args.preview_scale), pv)
        if not args.quiet:
            print(f"  → {pv}（プレビュー {args.preview_scale} 倍）")
    try:
        made = anims_mod.anims_outputs(res, cfg, dst.stem, piskel=bool(args.piskel))
    except ValueError as e:
        print(f"エラー（{src.name}）: {e}", file=sys.stderr)
        return 1
    for a in cfg.anims:
        fw = frames_mod.flip_warning(a.order)
        if fw:
            print_warnings([f"{a.name}: {fw}"], src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for name, data in made["files"].items():
        (dst.parent / name).write_bytes(data)
        if not args.quiet:
            print(f"  → {dst.parent / name}")
    if not args.quiet:
        sh = made["sheets"][0]
        w, h = sh.size
        for a, m in zip(sh.specs, sh.anims):
            info = sh.info["animations"][a.name]
            print(f"    {a.name}: {info['row'] + 1} 行目、フレーム {info['frames'][0]}〜{info['frames'][-1]}、"
                  f"{'/'.join(str(d) for d in m.durations)} ミリ秒、{'くり返す' if a.loop else '1 回で終わる'}"
                  f"、縦 {a.align_y}" + (f"、上げ下げ {','.join(map(str, a.shift_y))}" if a.shift_y else ""))
        print(f"    （1 コマ {w}×{h}、{sh.columns} 列 × {len(sh.anims)} 行、地面の線 ground_y {sh.info['ground_y']}）")
    failed = 0
    if args.gif:
        scale = args.gif_scale or gif_mod.DEFAULT_SCALE
        for sh in made["sheets"]:
            img = sh.image()
            for a in cfg.anims:
                try:
                    name, g = anims_mod.anim_gif(img, sh.info, a.name, scale, args.gif_background)
                except ValueError as e:
                    print(f"警告（{src.name}）: {a.name} の GIF を書けませんでした: {e}", file=sys.stderr)
                    failed = 1
                    continue
                (dst.parent / name).write_bytes(g.data)
                for m in g.warnings:
                    print(f"警告（{src.name}）: {a.name}: {m}", file=sys.stderr)
                if not args.quiet:
                    print(f"  → {dst.parent / name}（GIF: {len(g.durations)} フレーム、{scale} 倍、"
                          f"{'ずっと繰り返す' if a.loop else '1 回だけ再生'}）")
    return failed


def preview_path(output: Path, factor: int) -> Path:
    """プレビュー画像の保存先（出力と同じフォルダの <出力名>_xN.png）。"""
    return output.with_name(f"{output.stem}_x{factor}.png")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.readback is not None:
        check_readback_args(parser, args)
        return run_readback(args)
    if args.anims is not None:
        return run_anims(parser, args)
    if not args.inputs:
        parser.error("入力の画像（またはフォルダ）を指定してください")
    check_args(parser, args)
    load_motion_spec(parser, args)
    try:
        inputs = collect_inputs(args.inputs)
        if not inputs:
            parser.error("処理する画像がありません")
        outputs = output_paths(inputs[:1] if args.join else inputs, args.output)   # --join は 1 つの出力
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    for sp in args.save_palette or []:
        if sp.suffix.lower() not in palette_mod.WRITE_FORMATS:
            parser.error(f"--save-palette は {' / '.join(palette_mod.WRITE_FORMATS)} のファイルで指定してください: {sp}")
    args.palette_colors = None
    if args.palette is not None:
        try:
            args.palette_colors = load_palette(args.palette)
        except PaletteError as e:
            parser.error(str(e))
        if not args.quiet:
            print(f"パレット: {args.palette}（{len(args.palette_colors)} 色）")
    settings = settings_from_args(args)

    if args.join:
        return run_join(args, settings, inputs)
    failed = 0
    for src, dst in zip(inputs, outputs):
        try:
            rgba = load_rgba(src)
        except OSError as e:
            print(f"読み込めません: {src}（{e}）", file=sys.stderr)
            failed += 1
            continue
        steps_dir = dst.with_name(dst.stem + "_steps")

        def on_step(res, _i=[0]):
            _i[0] += 1
            if not args.quiet:
                h, w = res.image.shape[:2]
                print(f"  {_i[0]}. {res.description}: {w}×{h}（{res.seconds:.2f} 秒）")
            if args.save_steps:
                save_rgba(res.image, steps_dir / f"{_i[0]:02d}_{res.name}.png")

        if not args.quiet:
            print(f"{src} ({rgba.shape[1]}×{rgba.shape[0]}、{INPUT_TYPES[args.input_type]})")
        if args.frames:
            if write_frames(args, [(src.name, rgba)], settings, dst, src, on_step, many=len(inputs) > 1):
                failed += 1
            continue
        try:
            result = convert_sprite(rgba, settings, on_step=on_step)
        except GridNotFoundError as e:
            print(f"変換できません: {src}\n  {e}\n"
                  "  （普通のイラストなら --input-type illust --width 64 のように、"
                  "ドットの大きさが分かっていれば --dot-size 20 のように指定してください）", file=sys.stderr)
            failed += 1
            continue
        out = result.image
        save_rgba(out, dst)
        if not args.quiet:
            print(f"  → {dst}")
        print_warnings(result.warnings, src)
        for pp in palette_paths(args.save_palette, dst, len(inputs) > 1):
            cols = save_palette(used_colors(out), pp)
            if not args.quiet:
                print(f"  → {pp}（パレット {len(cols)} 色）")
        if args.preview_scale is not None:
            pv = preview_path(dst, args.preview_scale)
            save_rgba(scale_nearest(out, args.preview_scale), pv)
            if not args.quiet:
                print(f"  → {pv}（プレビュー {args.preview_scale} 倍）")
        if (args.motion is not None or args.motion_spec is not None) and \
                write_motion(args, out, dst, src, many=len(inputs) > 1, skipped=result.skipped) == 2:
            failed += 1
        if getattr(args, "talk_only", False):   # 口パクだけの --swaps（動きなし）: 口だけ動くしゃべりのシート
            from retrocore.sprite.swap import still_motion
            write_talk(args, still_motion(out, args._swaps), out, dst, src, result.skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
