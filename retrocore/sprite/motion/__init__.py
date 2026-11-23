"""動き（フェーズ 2）: 変換が終わったスプライトを材料にして、アニメーションのフレームを作る。

変換の処理（`convert_sprite()`）には手を入れない。決まりごと（新しい色を作らない・大きさと足元をそろえる・輪郭が途切れない・
迷ったら動かさない）は docs/sprite-app-plan.md の「フェーズ 2 の方針」と `base.py`。

    from retrocore.sprite.motion import make_motion, save_sheet
    res = make_motion(sprite, "breathe")
    if not res.skipped:
        save_sheet(res, "out/hero_breathe_sheet.png")   # hero_breathe_sheet.json も書く
        save_gif(res, "out/hero_breathe.gif", scale=4)   # そのまま見られる GIF アニメ（gif.py）

今あるのは呼吸（`breathe.py`）と横揺れ（`sway.py`）と浮遊（`floating.py`。足元も動く例外の動き）と
弾む（`bouncing.py`。浮遊と同じく体全体のずれだけで、上にだけ動く）とうなずく（`nod.py`。呼吸と同じ縦の変形の列で、首から上だけが下がる）と
伸び（`stretch.py`。呼吸と同じ縦の変形の列の +。上が伸びる）。呼吸・うなずく・伸びは `keep_small_parts=True` で目・頬などの
小さな部品の長さを変えないようにできる（既定は切）。
どれも「動きの表（タイムライン）を埋めるプリセット」で、フレームは表の仕組み（`timeline.py`。表の形は `spec.py`）で作る。
表を自分で書いたときは:

    from retrocore.sprite.motion import load_spec, make_motion_from_spec
    res = make_motion_from_spec(sprite, load_spec("bounce.motion.json"))
"""

from __future__ import annotations

import numpy as np

from . import bouncing as bounce_mod
from . import breathe as breathe_mod
from . import floating as float_mod
from . import nod as nod_mod
from . import stretch as stretch_mod
from . import sway as sway_mod
from .base import MotionResult, check_rules, foot_center, shape_signature
from .gif import GifResult, make_gif, motion_gif, save_gif
from .sheet import make_sheet, save_sheet, sheet_info, sheet_json
from .spec import SpecError, load_spec, parse_spec, spec_json, with_height_style
from .timeline import make_motion_from_spec

MOTIONS = {breathe_mod.NAME: breathe_mod.LABEL, sway_mod.NAME: sway_mod.LABEL, float_mod.NAME: float_mod.LABEL,
           bounce_mod.NAME: bounce_mod.LABEL, nod_mod.NAME: nod_mod.LABEL, stretch_mod.NAME: stretch_mod.LABEL}


def make_motion(rgba: np.ndarray, name: str, frames: int | None = None, durations=None, **options) -> MotionResult:
    """名前で選んだ動きのフレームを作る。`frames`・`durations` を省くと、その動きの既定値にする。

    弾む（bounce）のフレーム数は高さ（`height`）で決まる。`durations` を省くと着地だけ長めの既定になる。
    うなずく（nod）のフレーム数は深さ（`depth`）で決まる。`durations` を省くと上だけ長めの既定になる。
    伸び（stretch）のフレーム数は量（`amount`）で決まる。`durations` を省くと 1 フレーム 500 ミリ秒（呼吸と同じ）。
    """
    if name == breathe_mod.NAME:
        return breathe_mod.breathe(rgba, frames=frames or breathe_mod.DEFAULT_FRAMES,
                                   durations=breathe_mod.DEFAULT_DURATION if durations is None else durations,
                                   **options)
    if name == sway_mod.NAME:
        return sway_mod.sway(rgba, frames=frames or sway_mod.DEFAULT_FRAMES,
                             durations=sway_mod.DEFAULT_DURATION if durations is None else durations, **options)
    if name == float_mod.NAME:
        return float_mod.float_motion(rgba, frames=frames or float_mod.DEFAULT_FRAMES,
                                      durations=float_mod.DEFAULT_DURATION if durations is None else durations,
                                      **options)
    if name == bounce_mod.NAME:
        return bounce_mod.bounce(rgba, frames=frames, durations=durations, **options)
    if name == nod_mod.NAME:
        return nod_mod.nod(rgba, frames=frames, durations=durations, **options)
    if name == stretch_mod.NAME:
        return stretch_mod.stretch(rgba, frames=frames, durations=durations, **options)
    raise ValueError(f"動きは {' / '.join(MOTIONS)} のどれかです: {name}")


def preset_table(name: str, frames: int | None = None, durations=None, rgba: np.ndarray | None = None,
                 **options) -> dict:
    """プリセット（呼吸・横揺れ・浮遊・弾む・うなずく・伸び）が埋める動きの表（`make_motion()` と同じ引数）。

    画像は要らない。ただし、うなずくで分ける高さが自動（首を探す）のときだけは、絵（`rgba`）から首の高さを決めて数で書く
    （見つからなければ ValueError）。ほかの動きでは `rgba` を使わない。
    """
    if name == nod_mod.NAME:
        depth = nod_mod.check_depth(options.pop("depth", nod_mod.DEFAULT_DEPTH))
        split = options.pop("split", None)
        keep = options.pop("keep_small_parts", False)
        if options:
            raise ValueError(f"うなずくの設定ではありません: {', '.join(sorted(options))}")
        if frames is not None and frames != nod_mod.nod_frames(depth):
            raise ValueError(f"うなずくのフレーム数は深さで決まります（深さ {depth} → {nod_mod.nod_frames(depth)} フレーム）")
        return nod_mod.preset_nod_table(rgba, depth, durations, split, keep)
    if name == stretch_mod.NAME:
        amount = stretch_mod.check_amount(options.pop("amount", stretch_mod.DEFAULT_AMOUNT))
        split = options.pop("split", None)
        keep = options.pop("keep_small_parts", False)
        if options:
            raise ValueError(f"伸びの設定ではありません: {', '.join(sorted(options))}")
        if frames is not None and frames != stretch_mod.stretch_frames(amount):
            raise ValueError(f"伸びのフレーム数は量で決まります（量 {amount} → {stretch_mod.stretch_frames(amount)} フレーム）")
        if rgba is not None:
            split = stretch_mod.check_stretch_split(split, rgba.shape[0])
        return stretch_mod.stretch_table(amount, durations, split, keep)
    if name == bounce_mod.NAME:
        height = bounce_mod.check_height(options.pop("height", bounce_mod.DEFAULT_HEIGHT))
        if options:
            raise ValueError(f"弾むの設定ではありません: {', '.join(sorted(options))}")
        if frames is not None and frames != bounce_mod.bounce_frames(height):
            raise ValueError(f"弾むのフレーム数は高さで決まります（高さ {height} → {bounce_mod.bounce_frames(height)} フレーム）")
        return bounce_mod.bounce_table(height, durations)
    mods = {breathe_mod.NAME: (breathe_mod, breathe_mod.breathe_table), sway_mod.NAME: (sway_mod, sway_mod.sway_table),
            float_mod.NAME: (float_mod, float_mod.float_table)}
    if name not in mods:
        raise ValueError(f"動きは {' / '.join(MOTIONS)} のどれかです: {name}")
    mod, table = mods[name]
    return table(frames or mod.DEFAULT_FRAMES, mod.DEFAULT_DURATION if durations is None else durations, **options)


__all__ = ["GifResult", "MOTIONS", "MotionResult", "SpecError", "check_rules", "foot_center", "load_spec", "make_motion",
           "make_gif", "make_motion_from_spec", "make_sheet", "motion_gif", "parse_spec", "preset_table", "save_gif", "save_sheet", "shape_signature",
           "sheet_info", "sheet_json", "spec_json", "with_height_style"]
