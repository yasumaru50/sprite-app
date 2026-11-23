"""何コマも横に並んだ 1 枚の絵（AI が描いた歩き・走りなど）を、コマに切り分けて 1 つのキャラとしてそろえる。

    from retrocore.sprite.frames import FrameSettings, convert_frames, assemble_frames
    res = convert_frames(rgba, SpriteSettings(), FrameSettings(frame_count=4))
    res.frames        # そろえた各コマ（元の並び順。同じ大きさ・同じ足元）
    anim = assemble_frames(res, order=[1, 2, 3, 2], durations=[150])   # シート・GIF の材料（MotionResult）

コマごとに別々に変換すると、色・グリッド・位置がコマごとにずれて、再生したときにちらついたり揺れたりする。
そこで、全部のコマを「1 つのキャラ」として変換する（設計は docs/sprite-app-plan.md の「コマを切り分けてそろえる」）:

1. **背景除去**: 絵全体で 1 回（`BackgroundStep`。1 枚の絵の変換と同じ）
2. **切り分け**（`find_frames()`）: 透明な列のすき間でコマを分ける。体から離れた小さな部品（髪の先など。いちばん大きいコマの
   `PART_RATIO` 未満）は、列がいちばん近いコマに入れる。ごく小さなもの（`NOISE_RATIO` 未満。背景の消し残し）は捨てる。
   数を指定したとき（`frame_count`）に合わなければ作らない（`FramesError`。迷ったら何もしない）
3. **グリッド**（`detect_frame_grids()`）: ドットの大きさは**全部のコマをまとめて 1 回だけ**推定し、全部のコマで同じ大きさを使う。
   ただし線の位置（ずれ・位相）はコマごとに合わせる。AI はコマを別々に描くので、試した 3 枚ではコマごとに位相が
   横・縦とも 0〜5 画素（ドット 1 個ぶん近く）違った。1 つの位相を全部のコマに当てると、あるコマでは線がドットの真ん中を通り、
   2 色が混ざったマスや太った線ができるため。大きさの推定は 1 枚の絵の変換と同じ自己相関の山（`grid.estimate_square_period()`）
   で、各コマの縦横のプロファイルを合わせて求める（コマの間の空白をまたいだ「ずれ」が混ざらない）。山がその 2 倍・3 倍の位置に
   出ることがある（ai08 で 12.7px。本当は約 6.4px）ので、半分・3 分の 1 の大きさでも線を合わせ、境目の揃い方
   （`grid.alignment()`）がはっきり（`DIVISOR_GAIN` 以上）良くなればそちらを使う。
   **線の並びを全部のコマで共通にする**（`shared_frame_grids()`。`FrameSettings.grid`）: コマごとに線を合わせると、
   ぼかしの具合で狭い（5px の）マスが入る所がコマごとに違い、同じ幅に描かれたコマでもドットの数が変わる
   （ai08 の正面歩き: 3 コマとも元の絵では頭・腕・体の形が 1〜3 画素の違いしかないのに、66・68・67 ドットになり、
   そろえると腕が 1 ドット左右に揺れた）。そこで、各コマの境目の強さを 1 コマ目に重なる位置（±1 ドットで探す）にずらして
   足し合わせ、1 つの線の並びを求めて各コマに戻す。既定の `auto` は、どのコマの境目の揃い方もコマごとに合わせたときより
   `SHARED_MAX_DROP` 以上は下がらず、基準（0.4）も下回らないときだけ共通の線を使う（ai08 は使い、横向きで体の動く ai07・ai09 は使わない）
4. **単純化・パレット・クリーンアップ・縁取り**: 全部のコマを、すき間（`WORK_GAP` ドット）を空けて 1 枚に並べて、1 回で通す
   （`build_pipeline()` の工程そのまま）。パレットが全部のコマで共通になり、輪郭の色の判定もそろう
5. **そろえ方**（`align_frames()`）: 全部のコマを同じ大きさのキャンバスに置く。縦は足元（いちばん下の不透明なドット）をそろえる。
   横は `ALIGNS` から選ぶ（既定 `match`: 上の方の形がいちばん重なる位置。脚・腕・後ろになびく髪の長さで位置がぶれない）

ちらつきの見える化（`flicker_report()`）は、本来は動かないはずの上の方のドットが基準のコマとどれだけ違うかを数えるだけ。
小さな違いを自動で 1 コマ目に戻すのは、そろえた後の「ちらつきを減らす」段（`FrameSettings.steady`、`steady.py`。既定は off）。

`retrocore` の方針どおり NumPy と Pillow だけを使う（Pyodide で動く。np.bincount などは使っていない）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import grid as grid_mod
from .background import BackgroundStep
from .grid import ALPHA_OPAQUE, Grid, GridNotFoundError, GridStep
from .image import to_rgba
from .palette import PaletteStep
from .pipeline import SpritePipeline, StepResult
from .resize import ResizeStep

ALIGNS = {
    "match": "上の方（頭のあたり）の形がいちばん重なる位置（既定）",
    "head": "上の方（頭のあたり）の不透明なドットの横の中心",
    "center": "コマの不透明な範囲の横の中心",
    "none": "そろえない（元の絵でコマが等間隔に並んでいるとみなした位置のまま。足元もそろえない）",
}
DEFAULT_ALIGN = "match"
ALIGNS_Y = {
    "foot": "足元（いちばん下の不透明なドット）をそろえる（既定。歩きなど、足がいつも地面に着いている動き）",
    "match": "上の方（頭のあたり）の形がいちばん重なる高さ（走り・跳びなど、足が地面から離れるコマがある動き）",
    "none": "そろえない（元の絵の高さのまま）",
}
DEFAULT_ALIGN_Y = "foot"
DEFAULT_NAME = "frames"
DEFAULT_DURATION = 150         # 歩きの 1 コマの表示時間（ミリ秒）。走りは 100 前後が目安
HEAD_RATIO = 0.3               # 「上の方」: コマの高さ（中央値）の上から 3 割の行
MATCH_RANGE = 0.25             # match で試す横のずれ: コマの幅（最大）の ±25% まで
PART_RATIO = 0.15              # これ未満の画素数のかたまり（列のすき間で分けたもの）は、いちばん大きいコマに対する小さな部品
NOISE_RATIO = 0.002            # これ未満は背景の消し残しとして捨てる
WORK_GAP = 4                   # 単純化などで 1 枚に並べるときのコマのすき間（ドット）。縁取りで両側に 1 ドット付いても離れている
MARGIN_PX = 8                  # グリッドを合わせるときにコマの周りに付ける透明の余白（画素）
DIVISOR_GAIN = 0.1             # 半分・3 分の 1 の大きさを選ぶのに要る、境目の揃い方の良くなり方
DEFAULT_FLICKER_RATIO = 0.5    # ちらつきを数える範囲: キャンバスの上から半分の行
FLICKER_SHIFT_X = 1            # ちらつきを数えるとき、基準のコマに重ねるずれの範囲（体の上下・左右の 1 ドットの揺れは数えない）
FLICKER_SHIFT_Y = 2
MAX_FRAMES = 64
GRID_MODES = {
    "auto": "共通の線で、どのコマも境目の揃い方が大きく下がらないときだけ共通にする（既定）",
    "each": "コマごとに線を合わせる（PR #37 の動き）",
    "shared": "全部のコマで線の並びを共通にする（同じ形に描かれたコマの幅・腕の位置がそろう）",
}
DEFAULT_GRID_MODE = "auto"
SHARED_MAX_DROP = 0.2          # auto で共通の線を使う条件: どのコマ・どの向きも、境目の揃い方の下がり方がこれ未満


class FramesError(ValueError):
    """コマを作らなかったとき。`code` は記録（skipped）の code、メッセージは利用者向けの日本語。"""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.extra = extra


@dataclass
class FrameSettings:
    """コマの切り分け・そろえ方の設定（変換の設定は `SpriteSettings`）。"""

    frame_count: int | None = None   # 期待するコマの数（合わなければ作らない）。None なら確かめない
    align: str = DEFAULT_ALIGN
    align_y: str = DEFAULT_ALIGN_Y
    head_ratio: float = HEAD_RATIO
    grid: str = DEFAULT_GRID_MODE    # グリッドの線の並び（GRID_MODES）
    steady: str = "off"              # ちらつきを減らす（steady.MODES。既定 off）
    steady_size: int = 4             # 「小さい」とみなす最大のドット数（steady.DEFAULT_SIZE）
    steady_rows: int | None = None   # 戻す範囲（そろえたキャンバスの上から何行。None は高さの半分）
    same_ranges: tuple = ()          # 1 コマ目と同じにする範囲（same.SameRange の並び。空なら何もしない＝今までと同じ）
    same_search: int = 3             # 範囲の位置を探す幅（same.DEFAULT_SEARCH）
    same_from: int = 1               # 元にするコマ（1 から）
    edits: object = None             # 手の直し（touchup.Touchup。None か空なら何もしない＝今までと同じ）

    def validate(self) -> None:
        if self.align not in ALIGNS:
            raise ValueError(f"横のそろえ方は {' / '.join(ALIGNS)} のどれかです: {self.align}")
        if self.align_y not in ALIGNS_Y:
            raise ValueError(f"縦のそろえ方は {' / '.join(ALIGNS_Y)} のどれかです: {self.align_y}")
        if self.frame_count is not None and not 1 <= int(self.frame_count) <= MAX_FRAMES:
            raise ValueError(f"コマの数は 1〜{MAX_FRAMES} で指定してください: {self.frame_count}")
        if self.grid not in GRID_MODES:
            raise ValueError(f"グリッドの線の並びは {' / '.join(GRID_MODES)} のどれかです: {self.grid}")
        if not 0 < self.head_ratio <= 1:
            raise ValueError(f"上の方の割合は 0 より大きく 1 以下で指定してください: {self.head_ratio}")
        from .steady import validate as steady_validate
        steady_validate(self.steady, self.steady_size)
        if self.steady_rows is not None and (isinstance(self.steady_rows, bool) or int(self.steady_rows) < 1):
            raise ValueError(f"ちらつきを減らす行は 1 以上で指定してください: {self.steady_rows}")
        from .same import MAX_RANGES, SameRange, validate_search
        validate_search(self.same_search)
        if len(self.same_ranges) > MAX_RANGES:
            raise ValueError(f"1 コマ目と同じにする範囲は {MAX_RANGES} 個までです")
        if any(not isinstance(r, SameRange) for r in self.same_ranges):
            raise ValueError("1 コマ目と同じにする範囲は SameRange で指定してください")
        if isinstance(self.same_from, bool) or not isinstance(self.same_from, (int, np.integer)) or int(self.same_from) < 1:
            raise ValueError(f"元にするコマは 1 以上の整数で指定してください: {self.same_from}")
        if self.edits is not None:
            from .touchup import Touchup
            if not isinstance(self.edits, Touchup):
                raise ValueError("手の直しは touchup.Touchup で指定してください")


@dataclass
class FrameBox:
    """元の絵の中のコマの範囲（画素。終わりは含まない）。`parts` は入れた小さな部品の列の範囲。"""

    x0: int
    x1: int
    y0: int
    y1: int
    pixels: int
    parts: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class FramesResult:
    """コマの変換の結果。`frames` はそろえた各コマ（元の並び順。同じ大きさ）、`raw` はそろえる前（不透明な範囲に切り詰めたもの）。

    `offsets` は各コマを置いた位置（キャンバスの左上からの (x, y)）、`pivot` は足元（そろえた基準の横の位置と下端）。
    `grids` は各コマのグリッド線（元の絵の座標）、`dot_size` は全部のコマで使ったドットの大きさ。
    """

    frames: list[np.ndarray]
    raw: list[np.ndarray]
    boxes: list[FrameBox]
    grids: list[Grid]
    dot_size: tuple[float, float]
    offsets: list[tuple[int, int]]
    pivot: tuple[float, float]
    align: str
    align_y: str
    anchors: list[float]
    steps: list[StepResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    steady: object = None            # ちらつきを減らした記録（steady.SteadyReport。off なら None）
    before_steady: list | None = None  # ちらつきを減らす前のコマ（比較用。off なら None）
    same: object = None              # 1 コマ目と同じにした記録（same.SameReport。範囲が無ければ None）
    aligned: list | None = None      # そろえた直後のコマ（範囲・ちらつきを減らす前。画面で範囲を変えてやり直すのに使う）
    touchup: object = None           # 手の直しの記録（touchup.TouchupReport。直しが無ければ None）
    before_touchup: list | None = None  # 手の直しの前のコマ（直しが無ければ None）

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return int(w), int(h)


# ---------------------------------------------------------------- 1. 切り分け

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """真の続く範囲 [始め, 終わり) の並び。"""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return [(int(d[i]), int(d[i + 1])) for i in range(0, len(d), 2)]


def find_frames(rgba: np.ndarray, frame_count: int | None = None) -> tuple[list[FrameBox], list[str]]:
    """背景を除いた絵（RGBA）からコマの範囲を見つける。戻り値は (左から順のコマ, 説明の並び)。

    透明な列のすき間で分け、小さなかたまりは列がいちばん近いコマに入れる（同じ距離なら左）。ごく小さなものは捨てる。
    コマが見つからない・数が `frame_count` と合わないときは `FramesError`。
    """
    a = to_rgba(rgba)
    op = a[..., 3] >= ALPHA_OPAQUE
    runs = _runs(op.any(0))
    if not runs:
        raise FramesError("no_frames", "不透明な画素がありません（背景除去で全部消えた可能性があります）")
    sizes = [int(op[:, s:e].sum()) for s, e in runs]
    top = max(sizes)
    notes: list[str] = []
    big = [i for i, n in enumerate(sizes) if n >= PART_RATIO * top]
    boxes = {i: FrameBox(runs[i][0], runs[i][1], 0, 0, sizes[i]) for i in big}
    dropped = 0
    for i, n in enumerate(sizes):
        if i in boxes:
            continue
        if n < NOISE_RATIO * top:
            dropped += n
            continue
        s, e = runs[i]

        def dist(j: int) -> int:
            bs, be = runs[j]
            return bs - e if bs >= e else s - be

        j = min(big, key=lambda k: (dist(k), k))
        b = boxes[j]
        b.x0, b.x1 = min(b.x0, s), max(b.x1, e)
        b.pixels += n
        b.parts.append((s, e))
        notes.append(f"小さな部品（列 {s}〜{e - 1}、{n} 画素）を {big.index(j) + 1} コマ目に入れました")
    if dropped:
        notes.append(f"ごく小さなかたまり（合わせて {dropped} 画素）は背景の消し残しとして使いませんでした")
    out = [boxes[i] for i in big]
    for b in out:
        rows = np.flatnonzero(op[:, b.x0:b.x1].any(1))
        b.y0, b.y1 = int(rows[0]), int(rows[-1]) + 1
    if frame_count is not None and len(out) != frame_count:
        found = "、".join(f"{b.x0}〜{b.x1 - 1}" for b in out)
        raise FramesError("frame_count",
                          f"コマの数が合いません: 指定 {frame_count}、見つかったのは {len(out)}（列 {found}）。"
                          "コマが重なっている（すき間が無い）か、離れた部品をコマと数えた可能性があります。"
                          "コマとコマの間に背景だけの列があるかを確かめてください",
                          expected=int(frame_count), found=len(out),
                          boxes=[[b.x0, b.y0, b.x1, b.y1] for b in out])
    return out, notes


def frame_image(rgba: np.ndarray, box: FrameBox, others: list[FrameBox] | None = None,
                margin: int = MARGIN_PX) -> tuple[np.ndarray, int, int]:
    """コマの範囲を透明の余白付きで切り出す（ほかのコマの列は透明にする）。戻り値は (画像, 左の x, 上の y)（元の絵の座標）。"""
    a = to_rgba(rgba)
    x0, y0 = box.x0 - margin, box.y0 - margin
    out = np.zeros((box.y1 - box.y0 + 2 * margin, box.x1 - box.x0 + 2 * margin, 4), np.uint8)
    out[margin:margin + box.y1 - box.y0, margin:margin + box.x1 - box.x0] = a[box.y0:box.y1, box.x0:box.x1]
    # 小さな部品を入れて列の範囲が広がり、ほかのコマの列と重なったときは、そこを透明にする
    for o in others or []:
        if o is box:
            continue
        s, e = max(o.x0, box.x0) - x0, min(o.x1, box.x1) - x0
        if s < e:
            out[:, s:e] = 0
    return out, int(x0), int(y0)


# ---------------------------------------------------------------- 2. グリッド

def _profiles(images: list[np.ndarray], axes=(1, 0)):
    profs, ranges = [], []
    for img in images:
        op = img[..., 3] >= ALPHA_OPAQUE
        for axis in axes:
            profs.append(grid_mod.edge_profile(img, axis))
            ranges.append(grid_mod._content_range(op, axis))
    return profs, ranges


def _fit(images, period: tuple[float, float], snap: bool, tolerance: float) -> list[Grid]:
    return [grid_mod.detect_grid(img, dot_size=period, snap=snap, snap_tolerance=tolerance) for img in images]


def _mean_alignment(grids: list[Grid]) -> float:
    return float(np.mean([v for g in grids for v in g.alignment])) if grids else 0.0


def detect_frame_grids(images: list[np.ndarray], dot_size=None, snap: bool = True, square: bool = True,
                       tolerance: float = grid_mod.DEFAULT_SNAP_TOLERANCE) -> tuple[list[Grid], tuple[float, float], list[str]]:
    """全部のコマで同じドットの大きさを使って、コマごとにグリッド線を合わせる。戻り値は (各コマのグリッド, 大きさ, 説明)。

    `dot_size` を指定すればその大きさ（判定なし）。指定が無ければ全部のコマのプロファイルを合わせて推定し、
    見つからなければ `GridNotFoundError`。
    """
    notes: list[str] = []
    if dot_size is not None:
        p = grid_mod._pair(dot_size, "dot_size")
        return _fit(images, p, snap, tolerance), p, notes
    if square:
        profs, ranges = _profiles(images)
        p, prom, per = grid_mod.estimate_square_period(profs, ranges)
        periods, proms = [p, p], [prom, prom]
    else:
        periods, proms = [], []
        for axis in (1, 0):
            profs, ranges = _profiles(images, (axis,))
            p, prom, _ = grid_mod.estimate_square_period(profs, ranges)
            periods.append(p)
            proms.append(prom)
    if min(periods) <= 0 or min(proms) < grid_mod.DEFAULT_MIN_PROMINENCE:
        raise GridNotFoundError(
            "全部のコマを合わせても、ドットの格子が見つかりませんでした（周期の目立ち方 "
            + "・".join(f"{v:.2f}" for v in proms) + f"（基準 {grid_mod.DEFAULT_MIN_PROMINENCE}））。"
            "ドットの大きさが分かっている場合は --dot-size で指定してください")
    period = (periods[0], periods[1])
    grids = _fit(images, period, snap, tolerance)
    # 山が本当の大きさの 2 倍・3 倍に出ることがある: 小さい大きさで境目がはっきりよくそろうなら、そちらにする
    best = _mean_alignment(grids)
    for k in (2, 3):
        cand = (period[0] / k, period[1] / k)
        if min(cand) < grid_mod.DEFAULT_MIN_DOT:
            break
        g2 = _fit(images, cand, snap, tolerance)
        a2 = _mean_alignment(g2)
        if a2 >= best + DIVISOR_GAIN:
            notes.append(f"推定の山 {period[0]:.1f}px は 2 ドット分以上だったので、{cand[0]:.1f}px にしました"
                         f"（境目の揃い方 {best:.2f} → {a2:.2f}）")
            period, grids, best = cand, g2, a2
    if snap:
        # 線の並びから大きさを求め直してもう一度（1 枚の絵のグリッド検出と同じ。全部のコマの線をまとめる）
        new = []
        for axis in range(2):
            span, cnt = 0.0, 0
            for img, g in zip(images, grids):
                op = img[..., 3] >= ALPHA_OPAQUE
                s, e = grid_mod._content_range(op, 1 - axis)
                lines = g.xs if axis == 0 else g.ys
                inner = lines[(lines >= s) & (lines <= e)]
                if len(inner) >= 3:
                    span += float(inner[-1] - inner[0])
                    cnt += len(inner) - 1
            new.append(span / cnt if cnt else period[axis])
        if square:
            new = [sum(new) / 2, sum(new) / 2]
        if max(abs(new[k] - period[k]) for k in range(2)) > 0.1:
            period = (new[0], new[1])
            grids = _fit(images, period, snap, tolerance)
    low = [i + 1 for i, g in enumerate(grids) if min(g.alignment) < grid_mod.DEFAULT_MIN_ALIGNMENT]
    if low:
        notes.append(f"{'・'.join(map(str, low))} コマ目は境目の揃い方が低めです（基準 {grid_mod.DEFAULT_MIN_ALIGNMENT}）。"
                     "マスの色が混ざっていないか、グリッドを重ねた画像で確かめてください")
    return grids, period, notes


def _best_shift(ref: np.ndarray, s: np.ndarray, r: int) -> int:
    """s[x] が ref[x + d] にいちばん重なる d（-r〜r。同じなら 0 に近い方）。"""
    best = None
    for d in range(-r, r + 1):
        lo, hi = max(0, -d), min(len(s), len(ref) - d)
        if hi <= lo:
            continue
        v = float(np.dot(s[lo:hi], ref[lo + d:hi + d]))
        key = (v, -abs(d), -d)
        if best is None or key > best[0]:
            best = (key, d)
    return 0 if best is None else best[1]


def shared_frame_grids(images: list[np.ndarray], grids: list[Grid], period: tuple[float, float],
                       tolerance: float = grid_mod.DEFAULT_SNAP_TOLERANCE) -> list[Grid]:
    """全部のコマで共通の線の並びを求める（`grids` はコマごとに合わせた線。大きさなどはそこから写す）。

    向きごとに、各コマの境目の点数（`grid._normalize()`）を 1 コマ目に重なる位置（±1 ドット）にずらして平均し、
    `grid.fit_lines()` で 1 つの並びを求め、各コマの座標に戻す。
    """
    n = len(images)
    lines: list[list[np.ndarray]] = [[], []]
    for k, axis in enumerate((1, 0)):
        profs = [grid_mod.edge_profile(img, axis) for img in images]
        rngs = [grid_mod._content_range(img[..., 3] >= ALPHA_OPAQUE, axis) for img in images]
        scores = [grid_mod._normalize(p, *r) for p, r in zip(profs, rngs)]
        r = int(np.ceil(period[k]))
        shifts = [0] + [_best_shift(scores[0], scores[i], r) for i in range(1, n)]
        off = -min(shifts)
        size = max(len(sc) + d for sc, d in zip(scores, shifts)) + off
        tot, cnt = np.zeros(size), np.zeros(size)
        for sc, d in zip(scores, shifts):
            tot[d + off:d + off + len(sc)] += sc
            cnt[d + off:d + off + len(sc)] += 1
        avg = tot / np.maximum(cnt, 1)
        start = min(rg[0] + d + off for rg, d in zip(rngs, shifts))
        stop = max(rg[1] + d + off for rg, d in zip(rngs, shifts))
        common = grid_mod.fit_lines(avg, period[k], start, stop, tolerance)
        for sc, d in zip(scores, shifts):
            lines[k].append(np.unique(np.clip(common - (d + off), 0, len(sc) - 1)).astype(np.int64))
    out = []
    for i, (img, g) in enumerate(zip(images, grids)):
        al = []
        for k, axis in enumerate((1, 0)):
            prof = grid_mod.edge_profile(img, axis)
            rg = grid_mod._content_range(img[..., 3] >= ALPHA_OPAQUE, axis)
            al.append(grid_mod.alignment(prof, lines[k][i], *rg, period[k]))
        out.append(Grid(xs=lines[0][i], ys=lines[1][i], period=period, prominence=g.prominence,
                        alignment=(al[0], al[1]), manual=g.manual, square=g.square))
    return out


def choose_frame_grids(images: list[np.ndarray], grids: list[Grid], period: tuple[float, float], mode: str,
                       tolerance: float = grid_mod.DEFAULT_SNAP_TOLERANCE) -> tuple[list[Grid], str]:
    """`mode`（`GRID_MODES`）に従って、コマごとの線か共通の線を選ぶ。戻り値は (グリッド, 説明)。"""
    if mode == "each" or len(images) < 2:
        return grids, ""
    shared = shared_frame_grids(images, grids, period, tolerance)
    drops = [own - new for g, s in zip(grids, shared) for own, new in zip(g.alignment, s.alignment)]
    worst = max(drops)
    low = min(v for s in shared for v in s.alignment)
    desc = f"境目の揃い方の下がり方 最大 {worst:.2f}・最小 {low:.2f}"
    if mode == "shared":
        return shared, f"グリッドの線の並びを全部のコマで共通にしました（{desc}）"
    if worst < SHARED_MAX_DROP and low >= grid_mod.DEFAULT_MIN_ALIGNMENT:
        return shared, (f"コマの形がよく似ているので、グリッドの線の並びを全部のコマで共通にしました（{desc}。"
                        "コマごとに合わせるときは --frame-grid each）")
    return grids, ""


# ---------------------------------------------------------------- 4. そろえ方

def _bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _round(v: float) -> int:
    return int(np.floor(v + 0.5))


def head_center(frame: np.ndarray, rows: int) -> float:
    """上から `rows` 行の不透明なドットの横の中心（画素の中心を x + 0.5 とする）。"""
    op = frame[:rows, :, 3] > 0
    ys, xs = np.nonzero(op)
    if len(xs) == 0:
        return frame.shape[1] / 2
    return float(xs.mean() + 0.5)


def _match_score(ref: np.ndarray, frame: np.ndarray, x: int, y: int, rows: int) -> int:
    """基準のコマ `ref`（キャンバスに置いたもの）の上から rows 行（不透明な範囲の上端から）と、(x, y) に置いた `frame` を比べる。
    同じ色の不透明なドットの数 − 片方だけ不透明なドットの数。"""
    top = int(np.flatnonzero((ref[..., 3] > 0).any(1))[0])
    a = ref[top:top + rows]
    H, W = ref.shape[:2]
    b = np.zeros_like(a)
    fh, fw = frame.shape[:2]
    y0, y1 = max(top, y), min(top + rows, y + fh, H)
    x0, x1 = max(0, x), min(W, x + fw)
    if y0 < y1 and x0 < x1:
        b[y0 - top:y1 - top, x0:x1] = frame[y0 - y:y1 - y, x0 - x:x1 - x]
    oa, ob = a[..., 3] > 0, b[..., 3] > 0
    same = oa & ob & (a[..., :3] == b[..., :3]).all(-1)
    return int(same.sum()) - int((oa ^ ob).sum())


def align_frames(raw: list[np.ndarray], align: str = DEFAULT_ALIGN, align_y: str = DEFAULT_ALIGN_Y,
                 head_ratio: float = HEAD_RATIO, source_pos: list[tuple[float, float]] | None = None):
    """切り詰めた各コマを同じ大きさのキャンバスに置く。戻り値は (コマ, 置いた位置, pivot, 各コマの横の基準)。

    横は `align`（`ALIGNS`）、縦は `align_y`（`ALIGNS_Y`）。none は `source_pos`（各コマの左上の、元の絵での位置を
    ドット単位にし、等間隔のコマ割りの分を引いたもの）のまま置く。match は 1 コマ目を基準に、上の方（上から head_ratio の行）の
    形がいちばん重なる位置を探す（同じ点数なら、もう一方のそろえ方の位置に近い方）。pivot は (基準の横の位置, キャンバスの下端)。
    """
    if align not in ALIGNS:
        raise ValueError(f"横のそろえ方は {' / '.join(ALIGNS)} のどれかです: {align}")
    if align_y not in ALIGNS_Y:
        raise ValueError(f"縦のそろえ方は {' / '.join(ALIGNS_Y)} のどれかです: {align_y}")
    if "none" in (align, align_y) and source_pos is None:
        raise ValueError("none には元の絵での位置が要ります")
    n = len(raw)
    hs = [f.shape[0] for f in raw]
    ws = [f.shape[1] for f in raw]
    rows = max(1, _round(head_ratio * float(np.median(hs))))
    cs = [w / 2 for w in ws] if align == "center" else [head_center(f, rows) for f in raw]
    # 目安の位置（match の探し始め）。横: 上の方の中心（center は範囲の中心）をそろえる。縦: 足元をそろえる
    if align == "none":
        mx = min(p[0] for p in source_pos)
        gx = [p[0] - mx for p in source_pos]
    else:
        gx = [max(cs) - c for c in cs]
    if align_y == "none":
        my = min(p[1] for p in source_pos)
        gy = [p[1] - my for p in source_pos]
    else:
        gy = [max(hs) - h for h in hs]
    gx = [_round(v) for v in gx]
    gy = [_round(v) for v in gy]
    xs, ys = list(gx), list(gy)
    if n > 1 and "match" in (align, align_y):
        Rx = max(1, _round(MATCH_RANGE * max(ws))) if align == "match" else 0
        Ry = max(1, _round(MATCH_RANGE * max(hs))) if align_y == "match" else 0
        pad_x, pad_y = Rx + max(ws), Ry + max(hs)
        Wc = max(gx) + max(ws) + 2 * pad_x
        Hc = max(gy) + max(hs) + 2 * pad_y
        ref = np.zeros((Hc, Wc, 4), np.uint8)
        ref[gy[0] + pad_y:gy[0] + pad_y + hs[0], gx[0] + pad_x:gx[0] + pad_x + ws[0]] = raw[0]
        for i in range(1, n):
            best = None
            for dy in range(-Ry, Ry + 1):
                for dx in range(-Rx, Rx + 1):
                    x, y = gx[i] + dx, gy[i] + dy
                    sc = _match_score(ref, raw[i], x + pad_x, y + pad_y, rows)
                    key = (sc, -(abs(dx) + abs(dy)), -abs(dy), -abs(dx))
                    if best is None or key > best[0]:
                        best = (key, x, y)
            xs[i], ys[i] = best[1], best[2]
    mx, my = min(xs), min(ys)
    offs = [(x - mx, y - my) for x, y in zip(xs, ys)]
    W = max(o[0] + w for o, w in zip(offs, ws))
    H = max(o[1] + h for o, h in zip(offs, hs))
    anchors = [o[0] + c for o, c in zip(offs, cs)]
    if align == "none":
        pivot_x = W / 2
    elif align == "match":
        pivot_x = round(2 * anchors[0]) / 2          # 基準（1 コマ目）の上の方の中心
    else:
        pivot_x = round(2 * float(np.mean(anchors))) / 2
    frames = []
    for f, (ox, oy) in zip(raw, offs):
        c = np.zeros((H, W, 4), np.uint8)
        c[oy:oy + f.shape[0], ox:ox + f.shape[1]] = f
        frames.append(c)
    return frames, offs, (float(pivot_x), float(H)), anchors


# ---------------------------------------------------------------- 全体

def _work_steps(settings) -> list:
    """`build_pipeline()` の工程のうち、背景除去・グリッド検出・縮小より後のもの（単純化・パレット・クリーンアップ・縁取り）。"""
    from .convert import build_pipeline
    return [s for s in build_pipeline(settings).steps if not isinstance(s, (BackgroundStep, GridStep, ResizeStep))]


def convert_frames(rgba: np.ndarray, settings=None, frame_settings: FrameSettings | None = None,
                   on_step: Callable[[StepResult], None] | None = None) -> FramesResult:
    """何コマも横に並んだ絵を、コマに切り分けて 1 つのキャラとして変換し、そろえる。

    `settings` は 1 枚の絵の変換と同じ `SpriteSettings`（ドット絵風の入力だけ。`grid_origin` は使えない）。
    作らなかったときは `FramesError`（コマの数が合わないなど）か `GridNotFoundError`。
    """
    import time

    from .convert import SpriteSettings
    s = settings or SpriteSettings()
    fs = frame_settings or FrameSettings()
    fs.validate()
    if s.input_type != "pixel":
        raise ValueError("コマの切り分けはドット絵風の入力（グリッド検出）だけで使えます")
    if s.grid_origin is not None:
        raise ValueError("コマの切り分けでは、グリッドの始まる位置（grid_origin）は使えません（コマごとにずれが違うため）")
    steps: list[StepResult] = []

    def emit(name, desc, img, t0):
        r = StepResult(name, desc, img, time.perf_counter() - t0)
        steps.append(r)
        if on_step is not None:
            on_step(r)

    cur = to_rgba(rgba)
    if s.background:
        t0 = time.perf_counter()
        bg = BackgroundStep(color=s.bg_color, tolerance=s.bg_tolerance, remove_enclosed=s.bg_remove_enclosed)
        cur = to_rgba(bg.apply(cur))
        emit(bg.name, bg.describe(), cur, t0)
    t0 = time.perf_counter()
    boxes, notes = find_frames(cur, fs.frame_count)
    emit("frames", f"コマの切り分け（{len(boxes)} コマ: "
         + "、".join(f"{b.x0}〜{b.x1 - 1}" for b in boxes) + "）", cur, t0)

    t0 = time.perf_counter()
    subs = [frame_image(cur, b, boxes) for b in boxes]
    grids_local, period, gnotes = detect_frame_grids([img for img, _, _ in subs], s.dot_size, s.grid_snap,
                                                    s.dot_shape == "square", s.grid_tolerance)
    notes.extend(gnotes)
    if s.grid_snap:   # 等間隔の格子（--no-grid-snap）は、もともと全部のコマで同じ並び
        grids_local, gnote = choose_frame_grids([img for img, _, _ in subs], grids_local, period, fs.grid,
                                                s.grid_tolerance)
        if gnote:
            notes.append(gnote)
    raw, grids, src = [], [], []
    for (img, ox, oy), g in zip(subs, grids_local):
        cells = grid_mod.cells_to_pixels(img, g.xs, g.ys, transparent_ratio=s.transparent_ratio)
        bb = _bbox(cells[..., 3] > 0)
        if bb is None:
            raise FramesError("empty_frame", "グリッドに合わせたら不透明なマスが無くなったコマがあります")
        c0, r0, c1, r1 = bb
        raw.append(cells[r0:r1, c0:c1].copy())
        grids.append(Grid(xs=g.xs + ox, ys=g.ys + oy, period=g.period, prominence=g.prominence,
                          alignment=g.alignment, manual=g.manual, square=g.square))
        src.append((float(g.xs[c0] + ox), float(g.ys[r0] + oy)))
    size = (f"{period[0]:.1f}px" if abs(period[0] - period[1]) < 1e-9 else f"{period[0]:.1f}×{period[1]:.1f}px")
    emit("grid", f"グリッド検出（全部のコマで同じドット {size}、線はコマごとに合わせる: "
         + "、".join(f"{f.shape[1]}×{f.shape[0]}" for f in raw) + " ドット）", np.concatenate(
             [np.pad(f, ((0, max(x.shape[0] for x in raw) - f.shape[0]), (0, 1), (0, 0))) for f in raw], 1), t0)

    # 単純化・パレット・クリーンアップ・縁取りは、全部のコマを 1 枚に並べて 1 回で通す
    work = _work_steps(s)
    if work:
        H = max(f.shape[0] for f in raw) + 2 * (WORK_GAP // 2)
        xs, x = [], WORK_GAP // 2
        for f in raw:
            xs.append(x)
            x += f.shape[1] + WORK_GAP
        strip = np.zeros((H, x - WORK_GAP // 2, 4), np.uint8)
        top = WORK_GAP // 2
        for f, x0 in zip(raw, xs):
            strip[top:top + f.shape[0], x0:x0 + f.shape[1]] = f
        out = SpritePipeline(work).run(strip, on_step=lambda r: (steps.append(r), on_step and on_step(r)))
        if out.shape != strip.shape:
            raise RuntimeError("工程の途中で大きさが変わりました（コマを切り直せません）")
        half = WORK_GAP // 2
        new_raw = []
        for i, (f, x0) in enumerate(zip(raw, xs)):
            cut = out[:, x0 - half:x0 + f.shape[1] + half]
            bb = _bbox(cut[..., 3] > 0)
            c0, r0, c1, r1 = bb
            new_raw.append(cut[r0:r1, c0:c1].copy())
            # 縁取りで広がった分だけ、元の絵での位置もずらす
            sx, sy = src[i]
            src[i] = (sx + (c0 - half) * period[0], sy + (r0 - top) * period[1])
        raw = new_raw
    warnings: list[str] = []
    skipped: list[dict] = []
    from .motion.report import palette_entries
    for step in work:
        if isinstance(step, PaletteStep) and not step.final and step.last is not None:
            warnings.extend(step.last.warnings)
            skipped.extend(palette_entries(step.last))

    # 元の絵での位置（ドット単位）。none のために、コマが等間隔に並んでいるとみなした分を引く
    n = len(boxes)
    pitch = (boxes[-1].center_x - boxes[0].center_x) / (n - 1) if n > 1 else 0.0
    pos = [((sx - i * pitch) / period[0], sy / period[1]) for i, (sx, sy) in enumerate(src)]
    t0 = time.perf_counter()
    frames, offs, pivot, anchors = align_frames(raw, fs.align, fs.align_y, fs.head_ratio, pos)
    aligned = frames
    emit("align", f"そろえる（横 {fs.align}・縦 {fs.align_y}、{frames[0].shape[1]}×{frames[0].shape[0]} ドット）",
         np.concatenate(frames, 1), t0)
    frames, same_rep, steady_rep, before = finish_frames(frames, fs, emit)
    if same_rep is not None:
        warnings.extend(same_rep.warnings)
        skipped.extend(same_entries(same_rep))
    res = FramesResult(frames=frames, raw=raw, boxes=boxes, grids=grids, dot_size=period, offsets=offs,
                       pivot=pivot, align=fs.align, align_y=fs.align_y, anchors=anchors, steps=steps, warnings=warnings,
                       skipped=skipped, notes=notes, steady=steady_rep, before_steady=before, same=same_rep,
                       aligned=aligned)
    return _with_edits(res, fs, emit)


def _with_edits(res: FramesResult, fs: FrameSettings, emit=None) -> FramesResult:
    """最後に手の直し（`fs.edits`）を当てる。当てられなければ touchup.TouchupError（ValueError）。直しが無ければそのまま。"""
    if fs.edits is None or fs.edits.empty():
        return res
    import time

    from .touchup import touchup_result
    t0 = time.perf_counter()
    res = touchup_result(res, fs.edits)
    if emit is not None:
        emit("touchup", f"手の直し（直したドット {res.touchup.dot_count}、"
             + "・".join(f"{f.frame} コマ目 {len(f.dots)}" for f in res.touchup.frames) + "）",
             np.concatenate(res.frames, 1), t0)
    return res


def same_view_frames(res: FramesResult) -> list:
    """範囲の図（`same.same_image()`）に使う「写した後」のコマ（ちらつきを減らす前・手の直しの前）。"""
    return res.before_steady or res.before_touchup or res.frames


def finish_frames(aligned: list[np.ndarray], fs: FrameSettings, emit=None):
    """そろえた後の仕上げ: 1 コマ目と同じにする範囲（`fs.same_ranges`）→ ちらつきを減らす（`fs.steady`）の順。

    戻り値は (コマ, SameReport か None, SteadyReport か None, ちらつきを減らす前のコマか None)。
    `convert_frames()` と、画面で範囲だけを変えてやり直す `refinish_frames()` で共通（同じ設定ならバイト単位で同じ）。
    """
    import time
    frames = aligned
    same_rep = steady_rep = before = None
    if fs.same_ranges and len(frames) > 1:
        from .same import same_frames
        t0 = time.perf_counter()
        frames, same_rep = same_frames(frames, list(fs.same_ranges), fs.same_search, int(fs.same_from))
        if emit is not None:
            ok, total = same_rep.copied_count()
            emit("same", f"1 コマ目と同じにする（範囲 {len(same_rep.ranges)} 個・探す幅 {same_rep.search}、"
                 f"写した {ok}/{total}、変えたドット " + "・".join(str(f.changed) for f in same_rep.frames) + "）",
                 np.concatenate(frames, 1), t0)
    if fs.steady == "on" and len(frames) > 1:
        from .steady import steady_frames
        t0 = time.perf_counter()
        rows = None if fs.steady_rows is None else min(int(fs.steady_rows), frames[0].shape[0])
        before = frames
        frames, steady_rep = steady_frames(frames, fs.steady_size, rows=rows)
        if emit is not None:
            emit("steady", f"ちらつきを減らす（小さい {fs.steady_size} ドット以下・上から {steady_rep.rows} 行、"
                 "1 コマ目に戻したドット " + "・".join(str(f.restored_count) for f in steady_rep.frames) + "）",
                 np.concatenate(frames, 1), t0)
    return frames, same_rep, steady_rep, before


def same_entries(rep) -> list[dict]:
    """写さなかった範囲の記録（シートの JSON の `skipped` に足す。stage は "same_as_first"、frame は 0 から）。"""
    from .same import CODE_TEXT
    out = []
    sugg = {s["range"]: s for s in rep.suggestions}
    for sf in rep.frames:
        for rr in sf.ranges:
            if rr.copied:
                continue
            s = sugg.get(rr.range_index, {})
            out.append({"stage": "same_as_first", "code": rr.code, "message": CODE_TEXT[rr.code],
                        "frame": int(sf.frame) - 1, "region": int(rr.range_index),
                        "shift": {"x": int(rr.shift[0]), "y": int(rr.shift[1])}, "match": round(float(rr.match), 3),
                        "suggestions": [{"cli": n["cli"]} for n in s.get("near", [])], "result": "held"})
    return out


def refinish_frames(res: FramesResult, fs: FrameSettings) -> FramesResult:
    """変換をやり直さずに、範囲・ちらつきを減らすだけをやり直した結果（画面で範囲を描き直したとき）。
    `convert_frames(…, fs)` と同じコマ・同じ記録になる（工程の説明 `steps` は変換のときのまま）。"""
    import dataclasses
    fs.validate()
    base = res.aligned if res.aligned is not None else res.frames
    frames, same_rep, steady_rep, before = finish_frames(base, fs)
    old_same = res.same
    warnings = [w for w in res.warnings if old_same is None or w not in old_same.warnings]
    skipped = [e for e in res.skipped if e.get("stage") != "same_as_first"]
    if same_rep is not None:
        warnings.extend(same_rep.warnings)
        skipped.extend(same_entries(same_rep))
    res = dataclasses.replace(res, frames=frames, same=same_rep, steady=steady_rep, before_steady=before,
                              warnings=warnings, skipped=skipped, touchup=None, before_touchup=None)
    return _with_edits(res, fs)


def source_strip(res: FramesResult) -> np.ndarray:
    """そろえたコマを元の並び順に横に並べた画像（変換の結果として保存する。すき間なし）。"""
    return np.concatenate(res.frames, 1)


# ---------------------------------------------------------------- 並べ方・時間・体全体の上下

def parse_order(text: str | list, count: int) -> list[int]:
    """"1,2,3,2" のような並び（1 から）を確かめて数の並びにする。"""
    if isinstance(text, str):
        try:
            vals = [int(v) for v in text.replace(" ", "").split(",") if v != ""]
        except ValueError:
            raise ValueError(f"コマの順番は 1,2,3,2 のように数をカンマで区切って書いてください: {text}") from None
    else:
        vals = [int(v) for v in text]
    if not vals:
        raise ValueError("コマの順番が空です")
    bad = [v for v in vals if not 1 <= v <= count]
    if bad:
        raise ValueError(f"コマの順番の数は 1〜{count} にしてください（コマは {count} 個）: {', '.join(map(str, bad))}")
    if len(vals) > MAX_FRAMES:
        raise ValueError(f"コマの順番は {MAX_FRAMES} 個までです")
    return vals


def assemble_frames(res: FramesResult, order: list[int] | None = None, durations=None, root_y=None,
                    name: str = DEFAULT_NAME, mirror: bool = False):
    """シート・GIF の材料（`MotionResult`）を作る。`order` は元のコマの番号（1 から）の並び（既定は 1, 2, …）。

    `durations` は 1 つ（全部同じ）か、並びと同じ数。`root_y` は並びの各フレームでの体全体の上下のずれ（ドット。− が上。
    動きの表の root_y と同じ向き）。はみ出す分は上下に透明の行を足し、JSON の padding と offset_y に書く。
    `mirror` なら左右を反転する（pivot の x も反転）。
    """
    from .motion.base import MotionResult
    order = list(order) if order else list(range(1, len(res.frames) + 1))
    order = parse_order(order, len(res.frames))
    ds = [DEFAULT_DURATION] if durations is None else [int(d) for d in durations]
    if len(ds) == 1:
        ds = ds * len(order)
    if len(ds) != len(order):
        raise ValueError(f"表示時間の数（{len(ds)}）はフレームの数（{len(order)}）と同じか、1 つにしてください")
    if any(d < 1 for d in ds):
        raise ValueError("表示時間は 1 ミリ秒以上にしてください")
    frames = [res.frames[k - 1] for k in order]
    w, h = res.size
    px = res.pivot[0]
    info: dict = {"source_frames": order, "align": res.align, "align_y": res.align_y}
    if root_y is not None:
        ry = [int(v) for v in root_y]
        if len(ry) != len(order):
            raise ValueError(f"体全体の上下のずれの数（{len(ry)}）はフレームの数（{len(order)}）と同じにしてください")
        top, bottom = max(0, -min(ry)), max(0, max(ry))
        shifted = []
        for f, dy in zip(frames, ry):
            c = np.zeros((h + top + bottom, w, 4), np.uint8)
            c[top + dy:top + dy + h] = f
            shifted.append(c)
        frames = shifted
        info.update({"padding": 0, "padding_y": (top, bottom), "offset_y": ry})
    if mirror:
        frames = [f[:, ::-1].copy() for f in frames]
        px = w - px
    info["pivot_x"] = px
    if getattr(res, "same", None) is not None:
        info["same_as_first"] = res.same.info()
    if getattr(res, "steady", None) is not None:
        info["steady"] = res.steady.info()
    if getattr(res, "touchup", None) is not None:
        info["edits"] = res.touchup.info()
    return MotionResult(name=name + ("_mirror" if mirror else ""), frames=frames, durations=ds, info=info)


def sheet_file_name(stem: str, result) -> str:
    """コマのシートの PNG の名前: `<出力名>_<名前>_<幅>x<高さ>_<N>f.png`（反転は `<名前>_mirror`）。

    幅・高さは 1 フレームの大きさ（ドット。体の上下で足した行も含む）、N はシートのフレームの数（並べた順番の数）。
    ドット絵のエディタ（Piskel の Import as spritesheet など）で読み込むときに入れる数が、名前を見れば分かるようにする。
    情報の JSON は同じ名前の .json、GIF は同じ名前の .gif（`motion.gif.gif_path()`）。
    """
    w, h = result.size
    return f"{stem}_{result.name}_{w}x{h}_{len(result.frames)}f.png"


def frames_sheet_info(result, image_name: str, skipped: list | None = None, dot_size=None) -> dict:
    """コマのシートの JSON の辞書。形は動きのシート（`motion.sheet.sheet_info()`）と同じで、足したのは
    各フレームの `source_frame`（元の絵の何番目のコマか。1 から）と、いちばん外の `source`（コマの数・そろえ方・ドットの大きさ・反転）、
    `palette`（シートで使った色。手直しの読み戻しで新しい色を見つけるのに使う）。
    """
    from .motion.sheet import sheet_info
    info = sheet_info(result, image_name, skipped)
    info["pivot"]["x"] = float(result.info["pivot_x"])
    for f, k in zip(info["frames"], result.info["source_frames"]):
        f["source_frame"] = int(k)
    src = {"frame_count": int(max(result.info["source_frames"])), "align": result.info["align"],
           "align_y": result.info["align_y"], "mirrored": result.name.endswith("_mirror")}
    if dot_size is not None:
        src["dot_size"] = round(float(dot_size[0]), 3)
    skipped_list = info.pop("skipped")
    info["source"] = src
    if "same_as_first" in result.info:
        # 1 コマ目と同じにした記録（範囲を指定したときだけ。指定しない JSON は前と同じ）
        info["same_as_first"] = result.info["same_as_first"]
    if "steady" in result.info:
        # ちらつきを減らした記録（--steady on のときだけ。off の JSON は前と同じ）
        info["steady"] = result.info["steady"]
    if "edits" in result.info:
        # 手の直し（直しがあるときだけ。無い JSON は前と同じ）。元のコマごとの差し替えの一覧と、直したドットの数
        info["edits"] = result.info["edits"]
    # 使った色（"#rrggbb"。手直ししたシートを読み戻すとき、元のパレットに無い色を見つけるのに使う。edit.py）
    from .edit import palette_hex
    info["palette"] = palette_hex(result.frames)
    info["skipped"] = skipped_list
    return info


def frames_sheet_json(result, image_name: str, skipped: list | None = None, dot_size=None) -> str:
    import json
    return json.dumps(frames_sheet_info(result, image_name, skipped, dot_size), ensure_ascii=False, indent=2) + "\n"


def save_frames_sheet(result, png_path, skipped: list | None = None, dot_size=None):
    """シートの PNG と、同じ名前の .json を保存する。保存したパス (PNG, JSON) を返す。"""
    from pathlib import Path

    from .image import save_rgba
    from .motion.sheet import make_sheet
    png_path = Path(png_path)
    save_rgba(make_sheet(result.frames), png_path)
    json_path = png_path.with_suffix(".json")
    json_path.write_text(frames_sheet_json(result, png_path.name, skipped, dot_size), encoding="utf-8")
    return png_path, json_path


def frames_skipped_info(error: Exception, name: str = DEFAULT_NAME, skipped: list | None = None) -> dict:
    """コマを作らなかったときに書く JSON（`<出力名>_<名前>_skipped.json`）の辞書。形は動きの `_skipped.json` と同じ。"""
    from .motion.sheet import SHEET_VERSION
    if isinstance(error, FramesError):
        entry = {"stage": "frames", "code": error.code, "message": str(error)}
        entry.update(error.extra)
    elif isinstance(error, GridNotFoundError):
        entry = {"stage": "grid", "code": "grid_not_found", "message": str(error)}
    else:
        entry = {"stage": "frames", "code": "no_frames", "message": str(error)}
    entry["suggestions"] = []
    entry["result"] = "held"
    return {"version": SHEET_VERSION, "motion": name, "image": None, "made": False, "frame_count": 0, "loop": True,
            "total_duration_ms": 0, "frames": [], "skipped": list(skipped or []) + [entry]}


def frames_skipped_json(error: Exception, name: str = DEFAULT_NAME, skipped: list | None = None) -> str:
    import json
    return json.dumps(frames_skipped_info(error, name, skipped), ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------- ちらつきの見える化

@dataclass
class FlickerFrame:
    frame: int                 # 元のコマの番号（1 から）
    shift: tuple[int, int]     # 基準のコマにいちばんよく重なったずれ (x, y)
    diff: np.ndarray           # 違うドットの真偽（コマの座標。数える範囲の外は偽）
    opaque: int = 0            # 数える範囲の、このコマの不透明なドットの数（違うドットの割合の目安）

    @property
    def count(self) -> int:
        return int(self.diff.sum())


@dataclass
class FlickerReport:
    reference: int
    rows: int
    frames: list[FlickerFrame]

    def info(self, limit: int = 64) -> dict:
        out = {"reference_frame": self.reference, "rows": self.rows, "frames": []}
        for f in self.frames:
            ys, xs = np.nonzero(f.diff)
            out["frames"].append({"source_frame": f.frame, "shift": {"x": f.shift[0], "y": f.shift[1]},
                                  "diff_dots": f.count, "opaque_dots": f.opaque,
                                  "pixels": [[int(x), int(y)] for y, x in zip(ys[:limit], xs[:limit])]})
        return out


def _shifted(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """絵を (dx, dy) ずらす（はみ出した所は捨て、空いた所は透明）。"""
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _diff_mask(a: np.ndarray, b: np.ndarray, rows: int) -> np.ndarray:
    oa, ob = a[..., 3] > 0, b[..., 3] > 0
    d = (oa != ob) | (oa & ob & (a[..., :3] != b[..., :3]).any(-1))
    d[rows:] = False
    return d


def flicker_report(frames: list[np.ndarray], rows: int | None = None, reference: int = 1) -> FlickerReport:
    """そろえたコマ（同じ大きさ）の上から `rows` 行で、基準のコマと違うドットを数える（自動では直さない）。

    各コマは、基準のコマに ±1 ドット横・±2 ドット縦ずらして重ね、違うドットがいちばん少ないずれで数える
    （歩き・走りで体全体が 1 ドット上下するのは意図した動きなので、ちらつきとは数えない）。
    違うドットは、片方だけ不透明か、どちらも不透明で色が違う所。`rows` の既定はキャンバスの高さの半分。
    """
    h = frames[0].shape[0]
    rows = max(1, _round(DEFAULT_FLICKER_RATIO * h)) if rows is None else int(rows)
    if not 1 <= rows <= h:
        raise ValueError(f"ちらつきを数える行は 1〜{h} にしてください: {rows}")
    ref = frames[reference - 1]
    out = []
    for i, f in enumerate(frames):
        best = None
        for dy in range(-FLICKER_SHIFT_Y, FLICKER_SHIFT_Y + 1):
            for dx in range(-FLICKER_SHIFT_X, FLICKER_SHIFT_X + 1):
                # 基準をコマに合わせてずらす（違うドットの場所はコマの座標で出す）
                d = _diff_mask(_shifted(ref, dx, dy), f, rows)
                key = (int(d.sum()), abs(dx) + abs(dy), abs(dy), abs(dx))
                if best is None or key < best[0]:
                    best = (key, (dx, dy), d)
        out.append(FlickerFrame(i + 1, best[1], best[2], int((f[:rows, :, 3] > 0).sum())))
    return FlickerReport(reference, rows, out)


def flicker_json(report: FlickerReport) -> str:
    """ちらつきの記録（`<出力名>_<名前>_flicker.json`）。`pixels` はコマの座標 [x, y]（64 個まで）。"""
    import json
    return json.dumps(report.info(), ensure_ascii=False, indent=2) + "\n"


def flicker_image(frames: list[np.ndarray], report: FlickerReport, scale: int = 8) -> np.ndarray:
    """ちらつきの見える化の画像（RGBA）。各コマを scale 倍にし、数える範囲の外を薄く、違うドットを赤紫の枠で囲む。

    コマは元の並び順に左から並べ、コマの間は 1 ドット分の灰色。数える範囲の下の線は水色。
    """
    from PIL import Image, ImageDraw
    h, w = frames[0].shape[:2]
    gap = scale
    W = len(frames) * w * scale + (len(frames) + 1) * gap
    Hh = h * scale + 2 * gap + 12
    canvas = Image.new("RGBA", (W, Hh), (200, 200, 200, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (f, ff) in enumerate(zip(frames, report.frames)):
        x0 = gap + i * (w * scale + gap)
        y0 = gap + 12
        bg = np.empty((h, w, 4), np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        bg[...] = 236
        bg[((yy + xx) % 2) == 1] = (222, 222, 222, 255)
        bg[..., 3] = 255
        a = f[..., 3:4].astype(np.float64) / 255
        img = (f[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.float64)
        img[report.rows:] = img[report.rows:] * 0.35 + 255 * 0.65   # 数える範囲の外は薄く
        im = Image.fromarray(img.astype(np.uint8), "RGB").resize((w * scale, h * scale), Image.NEAREST)
        canvas.paste(im, (x0, y0))
        ys, xs = np.nonzero(ff.diff)
        for y, x in zip(ys, xs):
            draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                           outline=(255, 0, 200, 255), width=max(1, scale // 4))
        ly = y0 + report.rows * scale
        draw.line([x0, ly, x0 + w * scale - 1, ly], fill=(0, 170, 255, 255), width=max(1, scale // 4))
        label = f"#{ff.frame}  diff {ff.count}" + ("  (ref)" if ff.frame == report.reference else
                                                   f"  shift {ff.shift[0]:+d},{ff.shift[1]:+d}")
        _draw_label(draw, x0, 1, label)
    return np.asarray(canvas)


# 画像に書く文字（3×5 ドットの字）。Pillow のフォントは版と FreeType の有無で字形が変わる（Pyodide とパソコンで画素がずれる）ので、
# 使う文字だけを自分で持つ。1 は塗る所
_GLYPHS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", "#": "101111101111101", "+": "000010111010000",
    "-": "000000111000000", ",": "000000000010100", "(": "010100100100010", ")": "010001001001010",
    "d": "001001111101111", "i": "010000010010010", "f": "011010111010010", "r": "000101110100100",
    "e": "000111111100111", "s": "000111110011111", "h": "100100111101101", "t": "010111010010011",
    " ": "000000000000000",
    # 手直しの読み戻しの画像（edit.py）で使う字
    "n": "000110101101101", "w": "000101101111101", "m": "000111111101101", "b": "100100111101111",
    "o": "000111101101111", "a": "000011101101011",
    # ちらつきを減らした図（steady.py）で使う字
    "k": "100101110101101", "p": "000111101111100", "x": "000101010101101",
    # 範囲を 1 コマ目と同じにする図（same.py）で使う字
    "%": "101001010100101",
}


def _draw_label(draw, x: int, y: int, text: str, scale: int = 2, fill=(0, 0, 0, 255)) -> None:
    """`_GLYPHS` の字で text を書く（字の間は 1 ドット。知らない字は空白）。"""
    for ch in text:
        g = _GLYPHS.get(ch, _GLYPHS[" "])
        for k, v in enumerate(g):
            if v == "1":
                cx, cy = x + (k % 3) * scale, y + (k // 3) * scale
                draw.rectangle([cx, cy, cx + scale - 1, cy + scale - 1], fill=fill)
        x += 4 * scale
