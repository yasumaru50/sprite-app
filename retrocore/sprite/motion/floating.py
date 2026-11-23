"""浮遊: キャラクター全体をそのまま上下に 1 ドット動かす動き（中央 → 上 → 中央 → 下）。幽霊・浮かぶもの・スライムの待機向け。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」の「浮遊」）:
キャラクターは分けずに、絵をまるごと上下にずらすだけ。ドットを 1 つも変えずに位置だけを動かすので、新しい色は出ず、
隙間もできず、輪郭も途切れない（形はどのフレームでも元とまったく同じ）。

**決まりの例外（足元が動いてよい）**: 呼吸・横揺れでは「足元（いちばん下の行の、不透明な範囲の中央）を動かさない」が
決まりだが、浮遊では足元も一緒に上下する。例外にしたのは次の 1 つだけ:
- 足元の位置が、フレームごとに決まった量（`offsets`。JSON の `frames[].offset_y`）だけ**縦に**ずれてよい
今までどおり守るもの（`check_rules(..., shifts=offsets)` で全フレームを確かめる）:
- 新しい色を作らない・全フレームの大きさを同じにする・部品と輪郭のつながりを変えない・縁に輪郭以外の色を出さない
- 足元の**横**の位置は動かさない。縦のずれは決まった量ちょうど（それ以外に動いたらだめ）
- 各フレームは、元の絵を縦に決まった量ずらした**だけ**のもの（1 ドットでも違えば作らない）。はみ出して欠けるドットも無い
守れなければ浮遊を付けずに警告する（迷ったら動かさない。絵が無いときも付けない）。

ずれる量はゆっくりした波（横揺れと同じ `sway_offsets()`）で、上に動くのを先にする（JSON の offset_y は画像と同じく下向きが +）:
4 フレーム → 0, −1, 0, +1（中央 → 上 → 中央 → 下）、8 フレーム・揺れ幅 2 → 0, −1, −2, −1, 0, +1, +2, +1。
1 枚目は元の位置（ずれ 0）なので、ゲームの中で止まっている絵と同じ位置から動き始める。

余白: 変換の結果は不透明な部分に切り詰めてあるので、上下に動くとはみ出す。上に動く量・下に動く量の分だけ、
上と下に透明の行を足す（元の絵の上下に透明の行が余っていれば、その分は足さない）。全フレームが同じ大きさになる。
足した数は `info["padding_y"]`（上, 下）と JSON の `padding`（`{"left": 0, "right": 0, "top": 1, "bottom": 1}`）に書く。
JSON の `pivot` は 1 枚目の足元（元の絵の下端の中央。下に足した行の分だけフレームの下端より上）にする。
ゲームでは pivot を地面に置けば 1 枚目は元のスプライトと同じ位置に立ち、影や当たり判定は pivot のまま動かさないか、
`offset_y` を使って合わせる。
"""

from __future__ import annotations

import numpy as np

from .base import MAX_FRAMES, MotionResult, check_durations, check_frames, check_rules
from .sway import sway_offsets

NAME = "float"
LABEL = "浮遊（キャラ全体をそのまま上下に 1 ドット動かす。足元も動く）"
MOVES_FOOT = True         # 決まりの例外: 足元が（決まった量だけ縦に）動く
DEFAULT_FRAMES = 4        # 中央 → 上 → 中央 → 下
MIN_FRAMES = 3            # 2 フレームでは波が中央の 2 枚になり、動かない
DEFAULT_DURATION = 400    # 1 フレームの表示時間（ミリ秒）。4 フレームで 1.6 秒に 1 往復（横揺れよりゆっくり、ふわふわと）
DEFAULT_AMOUNT = 1        # 揺れ幅（ドット）
MAX_AMOUNT = 2


def float_offsets(frames: int = DEFAULT_FRAMES, amount: int = DEFAULT_AMOUNT) -> list[int]:
    """各フレームの上下のずれ（ドット。画像と同じく下向きが +、上が −）。上に動くのを先にする。

    例: 4 フレーム → [0, -1, 0, 1]、8 フレーム・揺れ幅 2 → [0, -1, -2, -1, 0, 1, 2, 1]。
    """
    return [-k for k in sway_offsets(frames, amount)]


def shift_rows(rgba: np.ndarray, dy: int) -> np.ndarray:
    """絵をまるごと縦に dy（+ が下）ずらす。空いた行は透明。外に出た行は捨てる（浮遊では余白を足してから使う）。"""
    out = np.zeros_like(rgba)
    h = rgba.shape[0]
    if dy >= 0:
        out[dy:] = rgba[:h - dy]
    else:
        out[:h + dy] = rgba[-dy:]
    return out


def float_table(frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
                amount: int = DEFAULT_AMOUNT) -> dict:
    """浮遊を動きの表で書いたもの。範囲は無く、体全体のずれの列 `root_y`（下向きが +）だけ。

    例: 4 フレーム → 0, −1, 0, +1。値が 0 の行には書かない。
    """
    from .spec import FORMAT, SPEC_VERSION
    durations = check_durations(durations, frames)
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "tracks": {"root_y": {"type": "root", "axis": "y"}},
            "frames": [{"ms": d, **({"root_y": k} if k else {})}
                       for d, k in zip(durations, float_offsets(frames, amount))]}


def float_motion(rgba: np.ndarray, frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
                 amount: int = DEFAULT_AMOUNT) -> MotionResult:
    """浮遊のフレームを作る。1 枚目は元のスプライトのまま（上下に余白を足したときは、それを足しただけのもの）。

    `durations` は全フレーム共通の表示時間（ミリ秒）か、フレームごとの並び。`amount` は揺れ幅（1〜2 ドット）。
    決まり（足元の例外を除く）を守れなければ `skipped` にして警告を返す。

    中身は「表を埋めるプリセット」: `float_table()` で表を埋め、表の仕組み（`timeline.py`）で作る。警告の言葉と `info` は前と同じ。
    """
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    check_frames(frames)
    if frames < MIN_FRAMES:
        raise ValueError(f"浮遊のフレーム数は {MIN_FRAMES}〜{MAX_FRAMES} で指定してください（2 では動きません）: {frames}")
    if isinstance(amount, (bool, np.bool_)) or not isinstance(amount, (int, np.integer)) or not 1 <= amount <= MAX_AMOUNT:
        raise ValueError(f"揺れ幅は 1〜{MAX_AMOUNT} ドットで指定してください: {amount}")
    amount = int(amount)
    durations = check_durations(durations, frames)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    offsets = float_offsets(frames, amount)
    spec = parse_spec(float_table(frames, durations, amount))

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"浮遊を付けませんでした: {reason}"], True,
                            {"offsets": offsets, "padding_y": [0, 0], "moves_foot": MOVES_FOOT}, spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, NAME), planned_ms=list(durations))

    built = build(base, spec)
    if built.problems:
        if built.problems[0].rule == "no_image":
            return skip(built.problems[0].fail.legacy)
        # 形を変えずに動かすだけなので起きないはずだが、起きたら付けない（迷ったら動かさない）
        top, bottom = built.padding_y
        padded = np.pad(base, ((top, bottom), (0, 0), (0, 0))) if top or bottom else base
        out = [shift_rows(padded, k) for k in offsets]
        return skip("決まりを守れませんでした（" + "、".join(check_rules(padded, out, shifts=offsets)) + "）")
    return MotionResult(NAME, built.frames, durations, [], False,
                        {"offsets": offsets, "padding_y": [int(v) for v in built.padding_y], "moves_foot": MOVES_FOOT},
                        spec=spec_to_dict(spec, built.used))
