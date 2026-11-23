"""動き（フェーズ 2）の共通部分: 結果の形と、決まりごと（docs/sprite-app-plan.md の「フェーズ 2 の方針」）の確かめ。

動きは「変換が終わったスプライト」を材料にして作る。決まりごと:
- 新しい色を作らない（ドットを動かす・増やす・減らすだけ）
- 全フレームの大きさを同じにし、足元（下端の中央）を動かさない
- 輪郭が途切れない
- 迷ったら動かさない（うまく作れない形なら、その動きを付けずに警告を出す）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

_N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

# 呼吸と横揺れで共通の決まり
AUTO = "auto"            # 分ける高さを自動で選ぶ
MAX_FRAMES = 32
MIN_HEIGHT = 8           # これより背の低い絵には付けない（上下を分ける余地が無い）
BAND_FOOT = 0.12         # 足元として動かさない下の部分（絵の高さに対する割合。2 ドット以上）


@dataclass
class MotionResult:
    """動きの結果。

    `frames` は同じ大きさの RGBA 配列の並び（1 枚目は元のスプライトそのもの）、`durations` は各フレームの表示時間（ミリ秒）。
    動きを付けなかったとき（`skipped`）は、`frames` は元のスプライト 1 枚だけで、理由は `warnings` にある。
    `spec` はその動きを書いた表（sprite-motion の JSON にする辞書。`timeline.py`）。プリセット（呼吸など）は自分が埋めた表を、
    表から作ったときは読んだ表を入れる。作れたときは、範囲ごとに実際に使った高さと線（`"used"`）が付く。
    `skips` は作らなかった理由の記録（`report.py`。出力の JSON の `skipped` に入る。作れたときは空）、`planned_ms` は作らなかったときに
    作ろうとしたフレームの表示時間（`durations` は元の絵 1 枚分の合計なので、別に持つ）。
    """

    name: str
    frames: list[np.ndarray]
    durations: list[int]
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    info: dict = field(default_factory=dict)
    spec: dict | None = None
    skips: list = field(default_factory=list)
    planned_ms: list | None = None

    @property
    def size(self) -> tuple[int, int]:
        """フレームの大きさ (幅, 高さ)。"""
        h, w = self.frames[0].shape[:2]
        return int(w), int(h)


def color_keys(rgba: np.ndarray) -> np.ndarray:
    """ドットごとの色の番号（R・G・B・A を 1 つの整数にしたもの。透明（α = 0）は -1）。"""
    a = rgba.astype(np.int64)
    key = (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]
    key[rgba[..., 3] == 0] = -1
    return key


def count_components(mask: np.ndarray) -> int:
    """真のドットの 8 近傍のつながりの数。"""
    h, w = mask.shape
    m = mask.tolist()
    seen = [[False] * w for _ in range(h)]
    n = 0
    for y0 in range(h):
        for x0 in range(w):
            if not m[y0][x0] or seen[y0][x0]:
                continue
            n += 1
            seen[y0][x0] = True
            q = deque([(y0, x0)])
            while q:
                y, x = q.popleft()
                for dy, dx in _N8:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and m[yy][xx] and not seen[yy][xx]:
                        seen[yy][xx] = True
                        q.append((yy, xx))
    return n


def silhouette_edge(opaque: np.ndarray) -> np.ndarray:
    """シルエットの縁: 上下左右のどれかが透明（または画像の外）の不透明なドット。"""
    pad = np.pad(opaque, 1)
    return opaque & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])


def edge_colors(rgba: np.ndarray) -> set:
    """シルエットの縁に出ている色（`color_keys()` の番号）。"""
    key = color_keys(rgba)
    return set(np.unique(key[silhouette_edge(key >= 0)]).tolist())


def shape_signature(rgba: np.ndarray) -> dict:
    """形の目印: 色ごとのつながりの数、不透明な部分のつながりの数、シルエットの縁（輪郭）のつながりの数。

    動かしたあとにこれが変われば、部品が消えた・分かれた・くっついた、輪郭が途切れた、のどれか。
    """
    key = color_keys(rgba)
    opaque = key >= 0
    edge = silhouette_edge(opaque)
    per_color = {int(k): count_components(key == k) for k in np.unique(key[opaque])}
    return {"colors": per_color, "opaque": count_components(opaque), "edge": count_components(edge)}


def foot_center(rgba: np.ndarray) -> tuple[float, int] | None:
    """足元の位置: いちばん下の不透明な行の、不透明な範囲の中央 (x, y)。不透明なドットが無ければ None。"""
    opaque = rgba[..., 3] > 0
    rows = np.flatnonzero(opaque.any(axis=1))
    if len(rows) == 0:
        return None
    y = int(rows[-1])
    xs = np.flatnonzero(opaque[y])
    return (float(xs[0] + xs[-1]) / 2.0, y)


def rule_reference(base: np.ndarray) -> dict:
    """`frame_problems()` で元の絵と比べるための目印（同じ元の絵で何枚も確かめるときに 1 回だけ作る）。"""
    return {"colors": set(np.unique(color_keys(base)).tolist()), "foot": foot_center(base),
            "sig": shape_signature(base), "edge": edge_colors(base)}


def frame_problems(base: np.ndarray, f: np.ndarray, i: int = 0, shift: int | None = None,
                   ref: dict | None = None) -> list[tuple[str, str]]:
    """1 枚のフレームの決まりの確かめ。守れていないものを (決まりの名前, 言葉) の並びで返す（空なら守れている）。

    決まりの名前は `new_color`（新しい色）・`size`（大きさ）・`foot`（足元が動いた）・`root_shift`（体全体のずれが決まった量と違う・
    欠けた）・`parts`（部品か輪郭のつながり）・`edge_color`（縁に輪郭以外の色）。言葉は `check_rules()` と同じ（`i` はフレームの番号）。
    `shift` は `check_rules()` の `shifts` の 1 つ分（None なら足元を動かさない）。
    """
    ref = rule_reference(base) if ref is None else ref
    if f.shape != base.shape:
        return [("size", f"フレーム {i} の大きさが違います（{f.shape[1]}×{f.shape[0]}）")]
    problems = []
    new = set(np.unique(color_keys(f)).tolist()) - ref["colors"]
    if new:
        problems.append(("new_color", f"フレーム {i} に元のスプライトに無い色が {len(new)} 色あります"))
    if shift is None:
        if foot_center(f) != ref["foot"]:
            problems.append(("foot", f"フレーム {i} で足元の位置が変わりました"))
    else:
        dy = int(shift)
        want = None if ref["foot"] is None else (ref["foot"][0], ref["foot"][1] + dy)
        if foot_center(f) != want:
            problems.append(("root_shift", f"フレーム {i} で足元が決まった量（縦に {dy:+d}）と違う動き方をしました"))
        if not _is_vertical_shift(base, f, dy):
            problems.append(("root_shift", f"フレーム {i} が元の絵を縦に {dy:+d} ずらしただけのものではありません"))
    if shape_signature(f) != ref["sig"]:
        problems.append(("parts", f"フレーム {i} で部品か輪郭のつながりが変わりました"))
    if not edge_colors(f) <= ref["edge"]:
        problems.append(("edge_color", f"フレーム {i} でシルエットの縁に輪郭以外の色が出ました"))
    return problems


def check_rules(base: np.ndarray, frames: list[np.ndarray], shifts: list[int] | None = None) -> list[str]:
    """決まりごとを確かめ、守れていないものを言葉で返す（空なら守れている）。

    新しい色が無い・大きさが同じ・足元（下端の中央）が同じ・形の目印（部品と輪郭のつながり）が同じ・
    シルエットの縁に、元の縁に無かった色が出ていない（輪郭に穴があいて中の塗りが外に出ていない）。

    `shifts` は「足元が動く動き」（浮遊）のための例外。フレームごとの縦のずれ（ドット、下向きが +）を渡すと、
    「足元が同じ」の代わりに「足元が縦にちょうどそのずれだけ動き、横には動かない」と
    「フレームが元の絵を縦にそのずれだけずらしただけのもの（欠けたドットも無い）」を確かめる。ほかの確かめは同じ。
    省くと（呼吸・横揺れ）今までどおり足元を動かさない。1 枚ずつの確かめは `frame_problems()`（決まりの名前つき）。
    """
    if shifts is not None and len(shifts) != len(frames):
        return [f"ずれの数（{len(shifts)}）がフレーム数（{len(frames)}）と違います"]
    ref = rule_reference(base)
    problems = []
    for i, f in enumerate(frames):
        problems += [text for _, text in frame_problems(base, f, i, None if shifts is None else shifts[i], ref)]
    return problems


def problem_pixels(base: np.ndarray, f: np.ndarray, rule: str, limit: int = 32) -> list[list[int]]:
    """決まりを守れなかったフレームの、問題の場所 [[x, y], ...]（元の絵と比べて違う所だけ。多ければ `limit` 個まで）。

    `new_color` は元の絵に無い色のドット、`edge_color` は縁に出た輪郭以外の色のドット、`parts` はつながりの数が変わった色の
    ドットのうち元の絵と違う所（無ければ違う所すべて）。大きさが違うときや、場所で示せない決まりは空。
    """
    if f.shape != base.shape:
        return []
    kb, kf = color_keys(base), color_keys(f)
    diff = kb != kf
    if rule == "new_color":
        mask = ~np.isin(kf, np.unique(kb))
    elif rule == "edge_color":
        mask = silhouette_edge(kf >= 0) & ~np.isin(kf, list(edge_colors(base)) or [-2])
    elif rule in ("parts", "euler", "root_shift"):
        if rule == "parts":
            sb, sf = shape_signature(base)["colors"], shape_signature(f)["colors"]
            changed = [k for k in set(sb) | set(sf) if sb.get(k) != sf.get(k)]
            mask = diff & (np.isin(kb, changed) | np.isin(kf, changed)) if changed else diff
            if not mask.any():
                mask = diff
        else:
            mask = diff
    else:
        return []
    ys, xs = np.nonzero(mask)
    return [[int(x), int(y)] for y, x in list(zip(ys.tolist(), xs.tolist()))[:limit]]


# 決まりの名前と、表の警告で使う短い言葉（「横に +1 ずらすと…」「…と」に続く形）
RULE_TEXT = {
    "new_color": "元の絵に無い色が出ます",
    "size": "フレームの大きさが変わります",
    "foot": "足元の位置が変わります",
    "root_shift": "絵を縦にずらしただけの形になりません（足元が決まった量と違う動き方をするか、ドットが欠けます）",
    "parts": "部品か輪郭のつながりが変わります",
    "edge_color": "シルエットの縁に輪郭以外の色が出ます",
    "euler": "輪郭の輪が切れるか、絵の中に隙間ができます",
    "small_parts": "目・頬などの小さな部品の形が変わります",
    # 帯・四角の範囲の境目（region.py）
    "hidden": "動いた後ろから見えてくる所の色が分かりません（範囲の外の同じ色の続きで埋められない）",
    "overlap": "範囲の外に出た所が、外の違う色のドットを隠します",
}


@dataclass
class Fail:
    """動かし方（縮める・横の位置など）が作れなかった理由。表の警告とプリセットの今の警告の両方の言葉を持つ。

    `legacy` はプリセット（`breathe()` など）の今の警告の「…を付けませんでした: 」の後ろ、`text` は表の警告の
    「N 行目（列 = 値、範囲 名前）: 」の後ろ。`depth` は、量の大きさがこれ以上の行が作れないこと（縮める 2 の 2 本目だけが
    見つからなければ 2）。`near` は近くで分けられる高さ（実際に作って確かめたもの）。
    """

    rule: str
    legacy: str
    text: str
    depth: int = 1
    rows_from_bottom: list | None = None
    near: list = field(default_factory=list)
    pixels: list = field(default_factory=list)


def _is_vertical_shift(base: np.ndarray, frame: np.ndarray, dy: int) -> bool:
    """frame が base を縦に dy（+ が下）ずらしただけのものか（外に出て欠けた不透明なドットも無い）。"""
    h = base.shape[0]
    if abs(dy) >= h:
        return False
    keep, lost = (base[:h - dy], base[h - dy:]) if dy >= 0 else (base[-dy:], base[:-dy])
    if (lost[..., 3] > 0).any():
        return False
    moved = np.zeros_like(base)
    if dy >= 0:
        moved[dy:] = keep
    else:
        moved[:h + dy] = keep
    return bool(np.array_equal(moved, frame))


# ---- 呼吸と横揺れで共通の、設定の確かめと「分ける高さ」 ----

def check_frames(frames: int) -> None:
    if not 2 <= frames <= MAX_FRAMES:
        raise ValueError(f"フレーム数は 2〜{MAX_FRAMES} で指定してください: {frames}")


def check_durations(durations, frames: int) -> list[int]:
    """表示時間（全フレーム共通の数か、フレームごとの並び）を、フレームごとの並びにする。合わなければ ValueError。"""
    if isinstance(durations, (int, np.integer)):
        durations = [int(durations)] * frames
    durations = [int(d) for d in durations]
    if len(durations) != frames:
        raise ValueError(f"表示時間の数（{len(durations)}）がフレーム数（{frames}）と違います")
    if min(durations) < 1:
        raise ValueError("表示時間は 1 ミリ秒以上で指定してください")
    return durations


def is_auto(split) -> bool:
    return split is None or (isinstance(split, str) and split == AUTO)


def check_split(split, height: int) -> int | None:
    """分ける高さの指定を確かめる。自動（None / "auto"）なら None、数なら int を返す。画像の外ならエラー（ValueError）。"""
    if is_auto(split):
        return None
    if isinstance(split, (bool, np.bool_)) or not isinstance(split, (int, np.integer)):
        raise ValueError(f"分ける高さは {AUTO} か、下から何ドット目かの整数で指定してください: {split!r}")
    split = int(split)
    if not 1 <= split <= height:
        raise ValueError(f"分ける高さ {split} は画像の範囲の外です。この画像（高さ {height} ドット）では 1〜{height} "
                         f"（1 = いちばん下の行）で指定してください")
    return split


def body_limits(key: np.ndarray) -> tuple[int, int, int, int] | None:
    """(絵の上の端の行, 下の端の行, 絵の高さ, 動かしてよい最も下の行)。背が低すぎる・空なら None。

    「動かしてよい最も下の行」より下（下の端から 12%、2 ドット以上）は足元として、どの動きでも動かさない。
    """
    rows = np.flatnonzero((key >= 0).any(axis=1))
    if len(rows) == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    if height < MIN_HEIGHT:
        return None
    return top, bottom, height, bottom - max(2, int(round(BAND_FOOT * height)))


def split_limits(rgba: np.ndarray) -> tuple[int, int] | None:
    """この絵で指定できる分ける高さの範囲（下から何ドット目、最小, 最大）。足元と、絵より上は除く。付けられない絵なら None。"""
    key = color_keys(rgba)
    lim = body_limits(key)
    if lim is None:
        return None
    h = key.shape[0]
    top, _, _, foot = lim
    lo_row = max(1, top)
    if foot < lo_row:
        return None
    return h - foot, h - lo_row
