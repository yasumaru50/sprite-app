"""伸び: 体の上の部分が 1 ドット伸びて戻る動き（足元は動かさない）。息を吸う・背伸び・伸びをする感じ。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」の「伸び」）:
新しい仕組みは作らず、**動きの表を埋めるプリセット**にする（うなずくと同じ）。使うのは呼吸と同じ範囲 `upper`（分け目）と
縦の変形の列 `upper_y` だけで、値が +（伸ばす）になる。呼吸との違いは向きだけ:
- **伸ばし方**: 分け目のあたりの継ぎ目（各列から 1 ドットずつ選んでつないだ横向きの線）のドットを 1 つ**複製**して、そこから上を
  1 ドット上げる（呼吸の逆）。複製するのは元のドットの色なので**新しい色は出ない**。複製してよい所・継ぎ目の選び方・決まりは
  呼吸の抜いてよい所と同じ（縦に同じ色が 3 ドット以上続く所は目立たない、2 ドットの所は少し目立つ、1 ドットの所（横向きの輪郭の線、
  1 ドットの目・口）は使わない）。だから縮めると伸ばすで同じ所を使え、縮めて伸ばすと元に戻る
- **上の余白**: 上に伸びてはみ出す分は、横揺れの左右の余白と同じく、全部のフレームの上に透明の行を足す（JSON の `padding.top`）。
  足元（pivot）はフレームの下のまま
- **量とフレーム**: 量（`amount`）で決まる。量 1 → 0, +1（2 フレーム）、量 2 → 0, +1, +2, +1（4 フレーム）
- **分ける高さ**: 呼吸と同じ（`auto` は腰のあたりを自動で探す。数は下から何ドット目、割合はキャラの高さに対する割合）
- **表示時間**: 呼吸と同じ 1 フレーム 500 ミリ秒（全フレーム共通の数か、フレームごとの並びも渡せる）

伸ばせない形なら付けずに警告し、近くで伸ばせる高さを知らせる（自動のときは腰の目安の高さの近く。画面では「○○ で試す」のボタン）。
"""

from __future__ import annotations

import numpy as np

from .base import AUTO, MotionResult, check_durations, check_split
from .base import is_auto as _is_auto
from .breathe import DEFAULT_DURATION as BREATHE_DURATION
from .breathe import SPLIT_RANGE, check_keep

NAME = "stretch"
LABEL = "伸び（体の上の部分が 1 ドット伸びて戻る）"
AMOUNTS = (1, 2)
DEFAULT_AMOUNT = 1
MAX_AMOUNT = 2
DEFAULT_DURATION = BREATHE_DURATION   # 1 フレームの表示時間（ミリ秒）。呼吸に合わせる

__all__ = ["NAME", "LABEL", "AMOUNTS", "DEFAULT_AMOUNT", "MAX_AMOUNT", "DEFAULT_DURATION", "SPLIT_RANGE",
           "check_amount", "stretch_offsets", "stretch_frames", "stretch_table", "stretch"]


def check_amount(amount) -> int:
    if isinstance(amount, (bool, np.bool_)) or not isinstance(amount, (int, np.integer)) or amount not in AMOUNTS:
        raise ValueError(f"伸びる量は 1〜{MAX_AMOUNT} ドットで指定してください: {amount}")
    return int(amount)


def stretch_offsets(amount: int = DEFAULT_AMOUNT) -> list[int]:
    """各フレームで伸びる量（ドット。+ が伸びる）。量 1 → [0, 1]、量 2 → [0, 1, 2, 1]。"""
    amount = check_amount(amount)
    return [0, 1] if amount == 1 else [0, 1, 2, 1]


def stretch_frames(amount: int = DEFAULT_AMOUNT) -> int:
    """量で決まるフレーム数（2・4）。"""
    return len(stretch_offsets(amount))


def _durations(durations, amount: int) -> list[int]:
    n = stretch_frames(amount)
    if durations is None:
        durations = DEFAULT_DURATION
    try:
        return check_durations(durations, n)
    except ValueError as e:
        if "数" in str(e):
            raise ValueError(f"伸びの表示時間の数（{len(durations)}）は、量 {amount} のフレーム数（{n}）と同じか、"
                             "1 つ（全フレーム共通）にしてください") from None
        raise


def check_stretch_split(split, height: int):
    """分ける高さの指定: None / "auto"（自動）、下から何ドット目の数、キャラの高さに対する割合（"35.0%"）。"""
    from .spec import parse_percent
    if _is_auto(split):
        return None
    if isinstance(split, str):
        p = parse_percent(split)
        if p is None or not 0 < p <= 100:
            raise ValueError(f"分ける高さは {AUTO} か、下から何ドット目かの整数か、\"35.0%\" のような割合"
                             f"（0 より大きく 100 以下）で指定してください: {split}")
        return split
    return check_split(split, height)


def stretch_table(amount: int = DEFAULT_AMOUNT, durations: int | list[int] | None = None,
                  split: int | str | None = None, keep_small_parts: bool = False) -> dict:
    """伸びを動きの表で書いたもの。範囲 `upper`（分け目）と縦の変形の列 `upper_y`（呼吸と同じ列。値は +）。値が 0 の行には書かない。"""
    from .spec import FORMAT, SPEC_VERSION
    amount = check_amount(amount)
    durations = _durations(durations, amount)
    region = {"type": "split", "at": AUTO if _is_auto(split) else (split if isinstance(split, str) else int(split))}
    if check_keep(keep_small_parts):
        region["keep_small_parts"] = True
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "regions": {"upper": region},
            "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
            "frames": [{"ms": d, **({"upper_y": k} if k else {})} for d, k in zip(durations, stretch_offsets(amount))]}


def stretch(rgba: np.ndarray, amount: int = DEFAULT_AMOUNT, durations: int | list[int] | None = None,
            split: int | str | None = None, frames: int | None = None, keep_small_parts: bool = False) -> MotionResult:
    """伸びのフレームを作る。1 枚目は元のスプライト（上に余白を足したもの）。

    `amount` は 1〜2 ドット（フレーム数は量で決まる。`frames` を渡すなら同じ数）。`durations` は省くと 1 フレーム 500 ミリ秒。
    `split` は None / "auto"（自動）、下から何ドット目の数、割合（"35.0%"）。画像の外の数はエラー（`ValueError`）。
    `keep_small_parts` を真にすると、目・頬などの小さな部品の長さを変えない。作れなければ `skipped` にして警告を返す。
    中身は表を埋めるプリセット（`stretch_table()` → `timeline.py`）。`spec` は埋めた表（作れたときは "used" つき）。
    """
    from .base import check_rules
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    amount = check_amount(amount)
    n = stretch_frames(amount)
    if frames is not None and frames != n:
        raise ValueError(f"伸びのフレーム数は量で決まります（量 {amount} → {n} フレーム）。フレーム数は指定できません: {frames}")
    durations = _durations(durations, amount)
    keep = check_keep(keep_small_parts)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    split = check_stretch_split(split, base.shape[0])
    offsets = stretch_offsets(amount)
    spec = parse_spec(stretch_table(amount, durations, split, keep))
    info = {"offsets": offsets, "amount": amount, "split": AUTO if split is None else split}

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"伸びを付けませんでした: {reason}"], True,
                            info, spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, NAME), planned_ms=list(durations))

    built = build(base, spec)
    if built.problems:
        fail = next((q.fail for q in built.problems if q.fail is not None), None)
        if fail is not None:
            return skip(fail.text)
        top = built.stretch_top
        padded = np.pad(base, ((top, 0), (0, 0), (0, 0))) if top else base
        out = [padded if k == 0 else built.raw_poses[("upper_y", k)] for k in offsets]
        why = check_rules(padded, out) or ["目・頬などの小さな部品の形が変わる"]
        return skip("伸ばすと形が崩れるため（" + "、".join(why) + "）")
    track = built.regions["upper"]["tracks"]["upper_y+"]
    seams, costs, h = track["lines"], track["costs"], track["height"]
    info.update({"seams": [s.tolist() for s in seams], "costs": costs,
                 "seam_from_bottom": [[int(h - s.max()), int(h - s.min())] for s in seams],
                 "padding_y": [int(v) for v in built.padding_y]})
    return MotionResult(NAME, built.frames, durations, [], False, info, spec=spec_to_dict(spec, built.used))
