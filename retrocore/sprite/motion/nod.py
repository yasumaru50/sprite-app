"""うなずく: 首から上（頭）が 1 ドット下がって戻る動き。足元も体も動かさない。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」の「うなずく」）:
新しい仕組みは作らず、**動きの表を埋めるプリセット**にする。使うのは呼吸と同じ範囲 `upper`（分け目）と縦の変形の列 `upper_y` だけで、
呼吸との違いは次の 3 つだけ:
- **分ける高さ**: 腰ではなく首。表の `at` には、自動で見つけた首の高さ（下から何ドット目の数）を書く
- **フレーム**: 深さ（`depth`）で決まる。深さ 1 → 0, −1（2 フレーム）、深さ 2 → 0, −1, −2, −1（4 フレーム）
- **表示時間**: 上を向いているコマ（0）を長く 600 ミリ秒、下がったコマを短く 200 ミリ秒（`default_durations()`）。
  全フレーム共通の数かフレームごとの並びを渡せば、それを使う（弾むと同じ）
頭を下げるやり方は呼吸とまったく同じ（首のあたりの継ぎ目のドットを抜いて、その上を下げる。縦に 3 ドット以上同じ色が続く所からだけ
抜く・新しい色を作らない・隙間を作らない・輪郭を途切れさせない・足元を動かさない。確かめも同じ `check_rules()`）。

**首の高さの自動の探し方**（`find_neck()`）: 各行の不透明なドットの数（横幅）を数え、次の 4 つを全部満たす行を「首の候補」にする。
1. キャラの高さの下から 20〜85% の範囲（`NECK_BAND`）で、足元（下から 12%）より上
2. その行より上に、キャラの高さの 25% 以上の頭がある（`HEAD_MIN`）
3. 上下の行より細い（同じでもよい。くびれ）
4. 頭のいちばん広い行の 8 割以下の幅（`NECK_RATIO`）で、下（キャラの高さの 25% 以内）にその行より 2 ドット以上広い行（体）がある
候補のうち「首の幅 ÷ 頭の幅」がいちばん小さいもの（同じ幅の行が続くときは、その真ん中）を首にする。
候補が無い・同じくらいの候補が離れて 2 つある（差が 0.05 未満で 5 行以上離れる）ときは**はっきりしない**とし、
決めずに、うなずくを付けずに警告する。警告には、首を探した範囲で実際に作れた高さを「試せる高さ: 8、12、16」の形で添える
（画面ではその数で試すボタンになる）。

足が無い・頭と体が同じ幅の形（スライムなど）では首は見つからない（迷ったら動かさない）。数で分ける高さを指定すれば作れる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .base import AUTO, MotionResult, check_durations, check_split, color_keys, split_limits
from .base import body_limits as _limits
from .base import is_auto as _is_auto

NAME = "nod"
LABEL = "うなずく（首から上の頭が 1 ドット下がって戻る）"
DEPTHS = (1, 2)
DEFAULT_DEPTH = 1
MAX_DEPTH = 2
UP_DURATION = 600         # 上を向いているコマ（0）の表示時間（ミリ秒）
DOWN_DURATION = 200       # 下がったコマの表示時間（ミリ秒）。すばやく下がって戻る
DEFAULT_DURATION = None   # 既定は上だけ長め（default_durations()）。CLI の --motion-duration を省いたとき
SPLIT_RANGE = 2           # 継ぎ目が分ける高さから離れてよい量（表の range の既定。呼吸と同じ）

# 首の探し方（割合はキャラの高さに対するもの。理由は上の説明と docs/sprite-app-plan.md）
NECK_BAND = (0.20, 0.85)  # 首を探す範囲（下からの割合）
HEAD_MIN = 0.25           # 首より上にある頭の高さの下限
NECK_RATIO = 0.8          # 首の幅 ≤ 頭のいちばん広い幅 × これ
WIDEN = 2                 # 首の下に、首より これ ドット以上広い行（体）がある
BODY_REACH = 0.25         # 体の広がりを探す範囲（首の下、キャラの高さに対する割合）
CLEAR_GAP = 0.05          # 2 番目の候補との「首の幅 ÷ 頭の幅」の差がこれ未満なら、同じくらいの候補とみなす
HINTS = 3                 # 首が見つからないときに添える「試せる高さ」の数
NECK_SLACK = 2            # 首の高さちょうどで作れないとき、上下に何ドットまでずらして試すか


def check_depth(depth) -> int:
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) or depth not in DEPTHS:
        raise ValueError(f"うなずく深さは 1〜{MAX_DEPTH} ドットで指定してください: {depth}")
    return int(depth)


def nod_offsets(depth: int = DEFAULT_DEPTH) -> list[int]:
    """各フレームで頭が下がる量（ドット。+ が下がる）。深さ 1 → [0, 1]、深さ 2 → [0, 1, 2, 1]。"""
    depth = check_depth(depth)
    return [0, 1] if depth == 1 else [0, 1, 2, 1]


def nod_frames(depth: int = DEFAULT_DEPTH) -> int:
    """深さで決まるフレーム数（2・4）。"""
    return len(nod_offsets(depth))


def default_durations(depth: int = DEFAULT_DEPTH) -> list[int]:
    """既定の表示時間: 上を向いているコマ（0）だけ UP_DURATION、下がったコマは DOWN_DURATION。"""
    return [UP_DURATION if k == 0 else DOWN_DURATION for k in nod_offsets(depth)]


def _durations(durations, depth: int) -> list[int]:
    if durations is None:
        return default_durations(depth)
    n = nod_frames(depth)
    try:
        return check_durations(durations, n)
    except ValueError as e:
        if "数" in str(e):
            raise ValueError(f"うなずくの表示時間の数（{len(durations)}）は、深さ {depth} のフレーム数（{n}）と同じか、"
                             "1 つ（全フレーム共通）にしてください") from None
        raise


def check_nod_split(split, height: int):
    """分ける高さの指定: None / "auto"（自動で首を探す）、下から何ドット目の数、キャラの高さに対する割合（"31.0%"）。"""
    from .spec import parse_percent
    if _is_auto(split):
        return None
    if isinstance(split, str):
        p = parse_percent(split)
        if p is None or not 0 < p <= 100:
            raise ValueError(f"分ける高さは {AUTO} か、下から何ドット目かの整数か、\"31.0%\" のような割合"
                             f"（0 より大きく 100 以下）で指定してください: {split}")
        return split
    return check_split(split, height)


# ---- 首を探す ----

@dataclass
class NeckSearch:
    """`find_neck()` の結果。`height` は首の高さ（画像の下から何ドット目。見つからなければ None）、`reason` は見つからない理由。"""

    height: int | None
    reason: str = ""
    band: list | None = None          # 探した範囲（下から何ドット目 [下, 上]）
    width: int | None = None          # 首の行の横幅（不透明なドットの数）
    head_width: int | None = None     # 首より上でいちばん広い行の横幅
    candidates: list = field(default_factory=list)   # [{"from_bottom", "width", "head_width", "ratio"}, ...]（よい順）

    def to_dict(self) -> dict:
        return {"from_bottom": self.height, "band": self.band, "width": self.width, "head_width": self.head_width,
                "candidates": self.candidates, **({"reason": self.reason} if self.reason else {})}


def find_neck(rgba: np.ndarray) -> NeckSearch:
    """首の高さを探す（探し方はこのファイルの最初の説明）。乱数なし。"""
    key = color_keys(rgba)
    h = key.shape[0]
    lim = _limits(key)
    if lim is None:
        return NeckSearch(None, "絵の背が低い、または絵がありません")
    top, bottom, height, foot = lim
    width = (key >= 0).sum(axis=1).astype(np.int64).tolist()
    lo = max(1, math.ceil(NECK_BAND[0] * height - 1e-9))   # キャラの下の端から何ドット目（キャラのいちばん下の行が 1）
    hi = math.floor(NECK_BAND[1] * height + 1e-9)
    reach = max(2, int(round(BODY_REACH * height)))
    band_rows = [y for y in range(top, bottom + 1) if lo <= bottom - y + 1 <= hi and y <= foot
                 and y - top >= HEAD_MIN * height]
    band = None
    if band_rows:
        band = [h - max(band_rows), h - min(band_rows)]
    rows = []
    for y in band_rows:
        w = width[y]
        if w > width[y - 1] or w > width[y + 1]:
            continue
        head = max(width[top:y])
        body = max(width[y + 1:min(bottom, y + reach) + 1])
        if w <= NECK_RATIO * head and body >= w + WIDEN:
            rows.append((y, w, head))
    # 同じ幅で続く行は 1 つの候補にまとめ、その真ん中の行を使う
    groups: list[list] = []
    for y, w, head in rows:
        if groups and groups[-1][-1][0] == y - 1 and groups[-1][-1][1] == w:
            groups[-1].append((y, w, head))
        else:
            groups.append([(y, w, head)])
    cands = []
    for g in groups:
        y, w, head = g[(len(g) - 1) // 2]
        cands.append({"from_bottom": int(h - y), "width": int(w), "head_width": int(head),
                      "ratio": round(w / head, 3)})
    cands.sort(key=lambda c: (c["ratio"], -c["from_bottom"]))
    where = f"下から {band[0]}〜{band[1]} ドット目" if band else "この絵"
    if not cands:
        return NeckSearch(None, f"首（頭より細くなり、その下で体が広がる行）が見つかりませんでした（{where}を探しました）",
                          band, candidates=[])
    best = cands[0]
    rival = next((c for c in cands[1:] if c["ratio"] - best["ratio"] < CLEAR_GAP
                  and abs(c["from_bottom"] - best["from_bottom"]) > 2 * SPLIT_RANGE), None)
    if rival is not None:
        a, b = sorted((best["from_bottom"], rival["from_bottom"]))
        return NeckSearch(None, f"首らしい所が 2 か所（下から {a} と {b} ドット目）あり、どちらか決められませんでした",
                          band, candidates=cands)
    return NeckSearch(best["from_bottom"], "", band, best["width"], best["head_width"], cands)


def workable_heights(rgba: np.ndarray, depth: int, search: NeckSearch, count: int = HINTS,
                     keep: bool = False) -> list[int]:
    """首を探した範囲で、実際に作れる（決まりを守って頭を下げられる）高さを `count` 個まで（小さい順）。

    同じくらいの候補があればそれを先に試し、残りは範囲を下・中・上に分けて、それぞれの真ん中から近い順に試した最初の 1 つ。
    """
    from .breathe import _poses_split
    lim = split_limits(rgba)
    if lim is None or search.band is None:
        return []
    a, b = max(lim[0], search.band[0]), min(lim[1], search.band[1])
    if a > b:
        return []
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    tried: dict = {}

    def ok(s: int) -> bool:
        if s not in tried:
            tried[s] = _poses_split(base, s, depth, keep=keep) is not None
        return tried[s]

    out = [c["from_bottom"] for c in search.candidates[:2] if a <= c["from_bottom"] <= b and ok(c["from_bottom"])]
    n = b - a + 1
    for part in range(count):
        if len(out) >= count:
            break
        p_lo, p_hi = a + part * n // count, a + (part + 1) * n // count - 1
        if p_hi < p_lo:
            continue
        mid = (p_lo + p_hi) // 2
        order = sorted(range(p_lo, p_hi + 1), key=lambda s: (abs(s - mid), s))
        got = next((s for s in order if ok(s)), None)
        if got is not None and got not in out:
            out.append(got)
    return sorted(out[:count])


def hint_text(heights: list[int]) -> str:
    return ("試せる高さ: " + "、".join(str(v) for v in heights)) if heights else "首を探した範囲には、作れる高さがありませんでした"


def auto_height(rgba: np.ndarray, depth: int, search: NeckSearch, keep: bool = False):
    """見つけた首の高さで、実際に作れる分ける高さ（表の at）を選ぶ。(高さか None, 最後に作った `Built`)。

    首の高さ N で決まりを守って頭を下げられなければ、N+1, N−1, N+2, N−2（`NECK_SLACK` まで。頭の側を先に）の順に試し、
    最初に作れた高さを使う（首の線が 1 ドットの線で、ちょうどそこでは抜けない絵があるため）。どれもだめなら
    (None, 首の高さ N で作ったときの `Built`)（警告には N での理由と、近くで分けられる高さを出す）。
    """
    from .spec import parse_spec
    from .timeline import build
    n = search.height
    lim = split_limits(rgba)
    first = None
    order = [n] + [n + s * d for d in range(1, NECK_SLACK + 1) for s in (1, -1)]
    for s in order:
        if lim is None or not lim[0] <= s <= lim[1]:
            continue
        built = build(rgba, parse_spec(nod_table(depth, None, s, keep)))
        if not built.problems:
            return s, built
        if s == n:
            first = built
    if first is None:
        first = build(rgba, parse_spec(nod_table(depth, None, n, keep)))
    return None, first


# ---- 表とプリセット ----

def nod_table(depth: int = DEFAULT_DEPTH, durations: int | list[int] | None = None, split: int | str = 1,
              keep_small_parts: bool = False) -> dict:
    """うなずくを動きの表で書いたもの。範囲 `upper`（分け目。`at` は首の高さ）と縦の変形の列 `upper_y`（呼吸と同じ列）。

    `split` は下から何ドット目の数か割合（"31.0%"）。自動（"auto"）は書けない（自動の首は絵から決めるので、
    先に `find_neck()` で数にする。表の "auto" は呼吸の自動（腰）になるため）。値が 0 の行には書かない。
    """
    from .spec import FORMAT, SPEC_VERSION
    depth = check_depth(depth)
    durations = _durations(durations, depth)
    if _is_auto(split):
        raise ValueError("うなずくの表の分ける高さは、数か割合で書きます（自動の首は絵から決めます）")
    region = {"type": "split", "at": split if isinstance(split, str) else int(split)}
    if keep_small_parts:
        region["keep_small_parts"] = True   # 切のときは書かない（前の版と同じ表）
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "regions": {"upper": region},
            "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
            "frames": [{"ms": d, **({"upper_y": -k} if k else {})} for d, k in zip(durations, nod_offsets(depth))]}


def preset_nod_table(rgba: np.ndarray | None, depth: int = DEFAULT_DEPTH, durations=None, split=None,
                     keep_small_parts: bool = False) -> dict:
    """プリセットが埋める表（`preset_table()` の nod）。自動の首は絵から探す（絵が無い・見つからなければ ValueError）。"""
    depth = check_depth(depth)
    if _is_auto(split):
        if rgba is None:
            raise ValueError("うなずくの自動の分ける高さ（首）は絵から決めるので、絵が要ります")
        search = find_neck(rgba)
        if search.height is None:
            raise ValueError(f"自動では首の高さを決められませんでした（{search.reason}）。分ける高さを数で指定してください。"
                             + hint_text(workable_heights(rgba, depth, search, keep=keep_small_parts)))
        at, _ = auto_height(np.ascontiguousarray(rgba, dtype=np.uint8), depth, search, keep_small_parts)
        split = search.height if at is None else at
    elif rgba is not None:
        split = check_nod_split(split, rgba.shape[0])
    return nod_table(depth, durations, split, keep_small_parts)


def nod(rgba: np.ndarray, depth: int = DEFAULT_DEPTH, durations: int | list[int] | None = None,
        split: int | str | None = None, frames: int | None = None, keep_small_parts: bool = False) -> MotionResult:
    """うなずくのフレームを作る。1 枚目は元のスプライトのまま。

    `depth` は 1〜2 ドット（フレーム数は深さで決まる。`frames` を渡すなら同じ数）。`durations` は省くと上だけ長め、
    全フレーム共通の数か、フレームごとの並びも渡せる。`split` は None / "auto"（首を自動で探す）、下から何ドット目の数、
    キャラの高さに対する割合（"31.0%"）。画像の外の数はエラー（`ValueError`）。
    作れなければ `skipped` にして警告を返す（首がはっきりしないときは、試せる高さを添える）。
    中身は表を埋めるプリセット（`nod_table()` → `timeline.py`）。`spec` は埋めた表（作れたときは "used" つき）。
    首が自動で見つからなかったときは埋める表が無いので `spec` は None。
    """
    from .breathe import check_keep
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    depth = check_depth(depth)
    keep = check_keep(keep_small_parts)
    n = nod_frames(depth)
    if frames is not None and frames != n:
        raise ValueError(f"うなずくのフレーム数は深さで決まります（深さ {depth} → {n} フレーム）。フレーム数は指定できません: {frames}")
    durations = _durations(durations, depth)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    split = check_nod_split(split, base.shape[0])
    offsets = nod_offsets(depth)
    h = base.shape[0]
    info = {"offsets": offsets, "depth": depth, "split": AUTO if split is None else split}
    search = None
    if split is None:
        search = find_neck(base)
        info["neck"] = search.to_dict()
        if search.height is None:
            from .report import motion_entry
            heights = workable_heights(base, depth, search, keep=keep)
            hint = hint_text(heights)
            message = (f"うなずくを付けませんでした: 自動では首の高さを決められませんでした（{search.reason}）。"
                       f"分ける高さを数で指定してください。{hint}")
            # 記録: 頭を下げる予定だった行（1 から）。首を探した範囲（下から何ドット目）があれば、試した高さとして書く
            moved = [i + 1 for i, k in enumerate(offsets) if k]
            entry = motion_entry("no_neck_found", message, moved, "upper_y", "upper", list(dict.fromkeys(-k for k in offsets if k)),
                                 list(search.band) if search.band is not None else None,
                                 suggestions=[{"nod_split": int(v)} for v in heights])
            return MotionResult(NAME, [base.copy()], [sum(durations)], [message], True, info, spec=None,
                                skips=[entry], planned_ms=list(durations))
    at = search.height if search is not None else split
    built = None
    if search is not None:
        at, built = auto_height(base, depth, search, keep)
        info["neck"]["used"] = at
    spec = parse_spec(nod_table(depth, durations, at if at is not None else search.height, keep))

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"うなずくを付けませんでした: {reason}"], True,
                            info, spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, NAME), planned_ms=list(durations))

    if built is None:
        built = build(base, spec)
    if built.problems:
        fail = next((q.fail for q in built.problems if q.fail is not None), None)
        head = ""
        if search is not None:
            neck = search.height
            if fail is not None and fail.rows_from_bottom is not None:
                from .breathe import near_hint
                return skip(f"自動で見つけた首は下から {neck} ドット目でしたが、その上下 {NECK_SLACK} ドット（下から "
                            f"{neck - NECK_SLACK}〜{neck + NECK_SLACK} ドット目）のどこで分けても、決まりを守って頭を下げられませんでした"
                            "（1 ドットの線が横切っている、または分けると部品か輪郭のつながりが変わる"
                            + ("、または小さな部品の形が変わる" if keep else "") + "）。" + near_hint(fail.near))
            head = f"自動で見つけた首は下から {neck} ドット目でしたが、"
        if fail is not None:
            return skip(head + fail.legacy)
        from .base import check_rules
        out = [base if k == 0 else built.raw_poses[("upper_y", -k)] for k in offsets]
        return skip(head + "動かすと形が崩れるため（" + "、".join(check_rules(base, out)) + "）")
    track = built.regions["upper"]["tracks"]["upper_y"]
    seams, costs = track["lines"], track["costs"]
    info.update({"seams": [s.tolist() for s in seams], "costs": costs,
                 "seam_from_bottom": [[int(h - s.max()), int(h - s.min())] for s in seams]})
    return MotionResult(NAME, built.frames, durations, [], False, info, spec=spec_to_dict(spec, built.used))
