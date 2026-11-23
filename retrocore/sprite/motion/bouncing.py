"""弾む: キャラクター全体を、地面から上にだけ動かして戻す動き（0 → −1 → −2 → −1）。スライム・小動物・はしゃぐキャラ向け。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」の「弾む」）:
浮遊と同じく、キャラクターは分けずに絵をまるごと縦にずらすだけ（動きの表の「体全体のずれ」`root_y` の列だけを使う）。
ドットを 1 つも変えないので、新しい色・隙間・輪郭の途切れはそもそも出ない。浮遊との違いは次の 2 つだけ:
- **上にだけ動く**（ずれは 0 以下）。1 枚目（着地）は元の位置で、下には動かない
- **着地の行だけ表示時間を長く**する（既定。地面で一拍止まるので「浮いている」ではなく「地面から跳ねている」に見える）

高さ（`height`。いちばん上で何ドット浮くか）ごとの並び。1 フレームに 1 ドットずつ動く三角の波（2 ドット飛ぶとちらついて見える）:
- 高さ 1 → 0, −1（2 フレーム）
- 高さ 2 → 0, −1, −2, −1（4 フレーム。既定）
- 高さ 3 → 0, −1, −2, −3, −2, −1（6 フレーム）
フレーム数は高さで決まる（`frames` を別に指定することはできない）。

表示時間の既定は「着地（ずれ 0 の行）だけ 300 ミリ秒、空中は 150 ミリ秒」（`default_durations()`）。
全フレーム共通の数か、フレームごとの並びを渡せば、それを使う（ほかのプリセットと同じ。共通の数なら着地も同じ長さ）。

足元は動くが、決まりの例外は浮遊と同じ（`check_rules(..., shifts=offsets)`。足元が縦に決まった量だけ動く以外は今までどおり）。
余白は上にだけ足す（`padding.top` が高さ、`padding.bottom` は 0）。JSON の `pivot` はフレームの下端（= 1 枚目の足元）のまま。
着地のつぶれは入れない（入れるなら表に範囲と縮める列を足す。`tests/motion_specs/bounce_squash.motion.json`）。
"""

from __future__ import annotations

import numpy as np

from .base import MotionResult, check_durations, check_rules
from .floating import shift_rows

NAME = "bounce"
LABEL = "弾む（キャラ全体が地面から上にだけ跳ねて戻る。足元も動く）"
MOVES_FOOT = True         # 決まりの例外: 足元が（決まった量だけ上に）動く
DEFAULT_HEIGHT = 2        # いちばん上で何ドット浮くか
MAX_HEIGHT = 3
HEIGHTS = (1, 2, 3)
AIR_DURATION = 150        # 空中のフレームの表示時間（ミリ秒）
LANDING_DURATION = 300    # 着地（ずれ 0）のフレームの表示時間（ミリ秒）。空中の 2 倍で、地面で一拍止まる
DEFAULT_DURATION = None   # 既定は着地だけ長め（default_durations()）。CLI の --motion-duration を省いたとき


def check_height(height) -> int:
    if isinstance(height, (bool, np.bool_)) or not isinstance(height, (int, np.integer)) or height not in HEIGHTS:
        raise ValueError(f"弾む高さは 1〜{MAX_HEIGHT} ドットで指定してください: {height}")
    return int(height)


def bounce_offsets(height: int = DEFAULT_HEIGHT) -> list[int]:
    """各フレームの上下のずれ（ドット。下向きが +、上が −）。0 から 1 ドットずつ上がり、1 ドットずつ下りる。

    例: 高さ 1 → [0, -1]、高さ 2 → [0, -1, -2, -1]、高さ 3 → [0, -1, -2, -3, -2, -1]。
    """
    height = check_height(height)
    up = list(range(0, height + 1))
    return [-k for k in up + up[-2:0:-1]]


def bounce_frames(height: int = DEFAULT_HEIGHT) -> int:
    """高さで決まるフレーム数（2・4・6）。"""
    return len(bounce_offsets(height))


def default_durations(height: int = DEFAULT_HEIGHT) -> list[int]:
    """既定の表示時間: 着地（ずれ 0）の行だけ LANDING_DURATION、空中は AIR_DURATION。"""
    return [LANDING_DURATION if k == 0 else AIR_DURATION for k in bounce_offsets(height)]


def _durations(durations, height: int) -> list[int]:
    if durations is None:
        return default_durations(height)
    n = bounce_frames(height)
    try:
        return check_durations(durations, n)
    except ValueError as e:
        if "数" in str(e):
            raise ValueError(f"弾むの表示時間の数（{len(durations)}）は、高さ {height} のフレーム数（{n}）と同じか、"
                             "1 つ（全フレーム共通）にしてください") from None
        raise


def bounce_table(height: int = DEFAULT_HEIGHT, durations: int | list[int] | None = None) -> dict:
    """弾むを動きの表で書いたもの。範囲は無く、体全体のずれの列 `root_y`（下向きが +）だけ。値が 0 の行には書かない。"""
    from .spec import FORMAT, SPEC_VERSION
    height = check_height(height)
    durations = _durations(durations, height)
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "tracks": {"root_y": {"type": "root", "axis": "y"}},
            "frames": [{"ms": d, **({"root_y": k} if k else {})}
                       for d, k in zip(durations, bounce_offsets(height))]}


def bounce(rgba: np.ndarray, height: int = DEFAULT_HEIGHT, durations: int | list[int] | None = None,
           frames: int | None = None) -> MotionResult:
    """弾むのフレームを作る。1 枚目（着地）は元のスプライトに上の余白を足しただけのもの。

    `height` は 1〜3 ドット。`durations` は省くと着地だけ長め、全フレーム共通の数か、フレームごとの並びも渡せる。
    `frames` は高さで決まるので、渡すなら同じ数にする（違えば ValueError）。
    決まり（足元の例外を除く）を守れなければ `skipped` にして警告を返す。中身は表を埋めるプリセット（`timeline.py` で作る）。
    """
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    height = check_height(height)
    n = bounce_frames(height)
    if frames is not None and frames != n:
        raise ValueError(f"弾むのフレーム数は高さで決まります（高さ {height} → {n} フレーム）。フレーム数は指定できません: {frames}")
    durations = _durations(durations, height)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    offsets = bounce_offsets(height)
    spec = parse_spec(bounce_table(height, durations))

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"弾むを付けませんでした: {reason}"], True,
                            {"offsets": offsets, "padding_y": [0, 0], "moves_foot": MOVES_FOOT, "height": height},
                            spec=spec_to_dict(spec),
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
                        {"offsets": offsets, "padding_y": [int(v) for v in built.padding_y], "moves_foot": MOVES_FOOT,
                         "height": height},
                        spec=spec_to_dict(spec, built.used))
