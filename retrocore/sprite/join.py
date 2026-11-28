"""別々のファイルの絵をつなげて 1 つの動きにする（「コマが並んだ絵」で、ファイルを何枚も選んだとき）。

    from retrocore.sprite.join import convert_frame_files
    res = convert_frame_files([("dance1.png", rgba1), ("dance2.png", rgba2)], SpriteSettings(), FrameSettings())
    res.frames    # 全部のファイルのコマ（並べた順。1 つ目のファイルの 1 コマ目から 1, 2, …）
    res.sources   # コマごとの (ファイルの名前, そのファイルの何コマ目か)
    res.files     # ファイルごとの記録（JoinFile。ドットの大きさ・高さ合わせ）

長い動き（ダンスなど）を AI に何枚かに分けて描かせると、ファイルごとにドットの大きさ・線の位置・色・キャラの大きさが少しずつ違う。
そこで、次のようにしてから、全部のコマを 1 つのキャラとしてそろえる（流れは `frames.convert_frames()` と同じ）:

1. **ファイルごと**: 背景除去 → コマの切り分け（1 枚の中のコマは 1 つでも何コマでもよい）→ グリッド検出（ドットの大きさも
   ファイルごとに見つける。AI はファイルごとに大きさを変えて描くため）→ 1 ドット＝1 画素のコマ
2. **高さを合わせる**（`match_height=True` のときだけ。既定は合わせずに違いを警告する）: ファイルの中でいちばん高いコマの高さ（ドット）を、ファイルの「キャラの高さ」とする。
   いちばん小さいファイルより `HEIGHT_TOLERANCE` ドットより高いファイルは、ドット絵に戻すときの格子を少し大きくして
   （`HEIGHT_TRIES` 回まで試す）、いちばん小さいファイルと同じ高さにする。完成したドット絵の拡大・縮小はしない（新しい色・ぼけが出ないように）。
   違いが `HEIGHT_MAX_RATIO`（1 割）より大きいときは合わせずに警告する（AI に描き直させる前提。大きく変えるとドットがつぶれる）
3. **まとめて**: 全部のファイルのコマを 1 枚に並べて単純化・パレット・クリーンアップ・縁取り（パレットは全部のファイルで 1 つ）
   → そろえる（大きさ・足元か頭の高さ）→ 1 コマ目と同じにする範囲 → ちらつきを減らす → 手の直し

つなぎ目のコマは混ぜない・中割りを作らない。ファイルの境目で体の位置・足元が大きくずれるときは警告だけする（`join_report()`）。
ファイルが 1 つのときは `frames.convert_frames()` とまったく同じ結果になる（記録の `files`・`sources` が付くだけ。出力のバイトは同じ）。
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grid import GridNotFoundError
from .pipeline import StepResult

MAX_FILES = 16
HEIGHT_TOLERANCE = 1      # これ以下のドットの違いは合わせない（脚の上げ下げ・縁の 1 ドットでも出る違い）
HEIGHT_MAX_RATIO = 0.10   # 自動で合わせる高さの違いの上限（いちばん小さいファイルに対する割合）
HEIGHT_TRIES = 6          # 格子の大きさを変えて試す回数
JOIN_MIN_SHIFT = 3        # 境目のずれを警告する最小のドット数（ファイルの中のコマどうしのずれより大きいときだけ）


@dataclass
class JoinFile:
    """つなげたファイルの 1 つの記録。"""

    name: str                       # ファイルの名前（同じ名前が 2 つあれば 2 つ目から "名前#2"）
    frame_count: int                # このファイルのコマの数
    first: int                      # このファイルの 1 コマ目が、つなげた全体の何コマ目か（1 から）
    detected: tuple[float, float]   # 見つけたドットの大きさ（画素）
    dot_size: tuple[float, float]   # 使ったドットの大きさ（高さを合わせたときは変えた後）
    height: int                     # キャラの高さ（いちばん高いコマ。ドット。合わせる前）
    height_used: int                # 使ったキャラの高さ（合わせた後）
    status: str = "base"            # base（いちばん小さい）/ same（合わせなくてよい）/ adjusted（合わせた）/ too_far（違いが大きくて合わせない）/ off
    boxes: list = field(default_factory=list)   # 見つけたコマ（そのファイルの画素。frames.FrameBox）

    @property
    def scale(self) -> float:
        return self.dot_size[1] / self.detected[1] if self.detected[1] else 1.0

    def info(self) -> dict:
        """シートの JSON の `source.files` の 1 つ。"""
        return {"name": self.name, "frame_count": int(self.frame_count), "first_frame": int(self.first),
                "dot_size": round(float(self.dot_size[0]), 3), "detected_dot_size": round(float(self.detected[0]), 3),
                "grid_scale": round(float(self.scale), 4), "height": int(self.height),
                "height_used": int(self.height_used), "height_match": self.status}


def file_keys(names: list[str]) -> list[str]:
    """ファイルの名前（パスは除く）。同じ名前が 2 つ以上あれば、2 つ目から "名前#2" のようにする（直しの記録で区別するため）。"""
    out, seen = [], {}
    for n in names:
        base = str(n).replace("\\", "/").rsplit("/", 1)[-1] or "image"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return out


def _height(raw: list[np.ndarray]) -> int:
    return int(max(f.shape[0] for f in raw))


def _match_height(subs, s, mode: str, period, height: int, target: int):
    """ドット絵に戻すときの格子を変えて、キャラの高さ（いまは height）を target に近づける。戻り値は (_grid_cells の結果, 高さ)。

    線を境目に吸い付けると本当のドットの境目に戻ってしまい、高さがほとんど変わらない（ai07 で試すと 1.35 倍にしてやっと 98 → 92）。
    そこで、合わせるファイルだけは等間隔の格子（`--no-grid-snap` と同じ）にして、大きさを 高さ ÷ 目標 倍から少しずつ直す。
    """
    from .frames import _grid_cells
    flat = dataclasses.replace(s, grid_snap=False)
    best = None
    k = height / target
    for _ in range(HEIGHT_TRIES):
        p = (period[0] * k, period[1] * k)
        try:
            got = _grid_cells(subs, flat, mode, dot_size=p)
        except Exception:   # 格子が大きすぎて不透明なマスが無くなったなど: それまでのいちばん近いものを使う
            break
        h = _height(got[0])
        key = (abs(h - target), h < target, abs(k - 1))
        if best is None or key < best[0]:
            best = (key, got, h)
        if h == target:
            break
        k *= h / target
    return (best[1], best[2]) if best is not None else (None, None)


def convert_frame_files(images: list, settings=None, frame_settings=None,
                        on_step: Callable[[StepResult], None] | None = None, match_height: bool = False):
    """別々のファイルの絵（`images` は (名前, RGBA) の並び。並べた順につなげる）を 1 つの動きとして変換し、そろえる。

    `match_height` が True のときだけキャラの高さを合わせる（既定は合わせない。しゃがみ・ジャンプのコマだけのファイルが混ざると、
    ほかのファイルが黙って小さくされてしまうため。迷ったら何もしない）。合わせないときも違いは警告する。作らなかったときは `frames.FramesError` か
    `GridNotFoundError`（どのファイルかを文の先頭に書く）。戻り値は `frames.FramesResult`（`files`・`sources` 付き）。
    """
    from . import frames as F
    from .convert import SpriteSettings
    s = settings or SpriteSettings()
    fs = frame_settings or F.FrameSettings()
    if not images:
        raise ValueError("つなげるファイルがありません")
    if len(images) > MAX_FILES:
        raise ValueError(f"つなげるファイルは {MAX_FILES} 個までです（いま {len(images)} 個）")
    names = file_keys([n for n, _ in images])
    if len(images) == 1:
        return _single(names[0], images[0][1], s, fs, on_step)
    p = prepare_files(images, s, fs, on_step, match_height)
    return F._finish_convert(p.raw, p.pos, p.boxes, p.grids, p.files[0].dot_size, fs, p.steps, p.emit, p.notes,
                             p.warnings, p.skipped, files=p.files, sources=p.sources)


@dataclass
class Prepared:
    """ファイルごとの変換（背景除去〜グリッド・高さ合わせ）と、全部のコマでの単純化・パレットなどを済ませたコマ（そろえる前）。"""

    raw: list            # 切り詰めたコマ（並べた順）
    pos: list            # 各コマの元の絵での位置（ドット単位。frames._source_pos()）
    boxes: list
    grids: list
    files: list          # JoinFile の並び
    sources: list        # コマごとの (ファイルの名前, 何コマ目か)
    steps: list
    emit: Callable
    notes: list
    warnings: list
    skipped: list


def prepare_files(images: list, s, fs, on_step=None, match_height: bool = False) -> Prepared:
    """`convert_frame_files()` のうち、そろえる前まで（ファイルが 1 つでもこの流れで通す。動きをまとめる `anims.py` も使う）。"""
    from . import frames as F
    names = file_keys([n for n, _ in images])
    F._check_settings(s, fs)
    steps: list[StepResult] = []

    def emit(name, desc, img, t0):
        r = StepResult(name, desc, img, time.perf_counter() - t0)
        steps.append(r)
        if on_step is not None:
            on_step(r)

    per = []   # ファイルごと: (名前, 背景を除いた絵, コマの範囲, 切り出したコマ, _grid_cells の結果)
    notes: list[str] = []
    for name, (_, rgba) in zip(names, images):
        t0 = time.perf_counter()
        cur, bg = F._remove_background(rgba, s)
        if bg is not None:
            emit(bg.name, f"{name}: {bg.describe()}", cur, t0)
        t0 = time.perf_counter()
        try:
            boxes, fnotes = F.find_frames(cur)
        except F.FramesError as e:
            raise F.FramesError(e.code, f"{name}: {e}", file=name, **e.extra) from None
        emit("frames", f"{name}: コマの切り分け（{len(boxes)} コマ: "
             + "、".join(f"{b.x0}〜{b.x1 - 1}" for b in boxes) + "）", cur, t0)
        notes.extend(f"{name}: {m}" for m in fnotes)
        subs = [F.frame_image(cur, b, boxes) for b in boxes]
        try:
            got = F._grid_cells(subs, s, fs.grid)
        except GridNotFoundError as e:
            raise GridNotFoundError(f"{name}: {e}", e.estimate) from None
        except F.FramesError as e:
            raise F.FramesError(e.code, f"{name}: {e}", file=name, **e.extra) from None
        per.append([name, cur, boxes, subs, got])
    if fs.grid == "auto" and s.grid_snap:
        # 線の並びを共通にするのは、2 コマ以上のファイルの全部で共通にできるときだけ（ファイルごとに違うと、
        # 同じ動きを 1 枚に描いたときと分けて描いたときで結果が変わるため。どれかが合わなければ全部コマごと）
        many = [p for p in per if len(p[2]) > 1]
        if any(p[4][5] for p in many) and not all(p[4][5] for p in many):
            for p in many:
                if p[4][5]:
                    p[4] = F._grid_cells(p[3], s, "each")
            notes.append("グリッドの線の並びは、共通にできないファイルがあったので、全部のファイルでコマごとに合わせました")
    for p in per:
        notes.extend(f"{p[0]}: {m}" for m in p[4][4])
    total = sum(len(p[2]) for p in per)
    if total > F.MAX_FRAMES:
        raise F.FramesError("frame_count", f"つなげたコマが多すぎます（{total} コマ。{F.MAX_FRAMES} コマまで）",
                            expected=F.MAX_FRAMES, found=total, files=[[p[0], len(p[2])] for p in per])
    if fs.frame_count is not None and total != fs.frame_count:
        found = "、".join(f"{p[0]} {len(p[2])}" for p in per)
        raise F.FramesError("frame_count",
                            f"コマの数が合いません: 指定 {fs.frame_count}、見つかったのは全部で {total}（{found}）。"
                            "コマが重なっている（すき間が無い）か、離れた部品をコマと数えた可能性があります",
                            expected=int(fs.frame_count), found=total, files=[[p[0], len(p[2])] for p in per])

    # キャラの高さを、いちばん小さいファイルに合わせる（ドット絵に戻すときの格子の大きさを変える）
    heights = [_height(p[4][0]) for p in per]
    target = min(heights)
    base = heights.index(target)
    files: list[JoinFile] = []
    warnings: list[str] = []
    skipped: list[dict] = []
    first = 1
    for i, p in enumerate(per):
        name, _, boxes, subs, got = p
        period = got[3]
        h = heights[i]
        status = "base" if i == base else "same"
        used = h
        if i != base and h - target > HEIGHT_TOLERANCE:
            ratio = h / target - 1
            if ratio > HEIGHT_MAX_RATIO:
                status = "too_far"
            elif not match_height:
                status = "off"
            else:
                mode = fs.grid if fs.grid != "auto" else ("shared" if got[5] else "each")
                new, hh = _match_height(subs, s, mode, period, h, target)
                if new is not None and abs(hh - target) < h - target:
                    p[4] = got = new
                    used = hh
                    status = "adjusted"
                    notes.append(f"{name}: キャラの高さを {names[base]} に合わせるため、ドット絵に戻す格子を "
                                 f"{got[3][1] / period[1]:.3f} 倍（{_size(period)} → {_size(got[3])}）にしました"
                                 f"（高さ {h} → {hh} ドット。{names[base]} は {target} ドット）")
                else:
                    status = "too_far"
            if status in ("too_far", "off"):
                head = f"{name} のキャラの高さ {h} ドットは、いちばん小さい {names[base]}（{target} ドット）より {ratio * 100:.0f}% 高い"
                if status == "off":
                    msg = (head + "です（高さは合わせていません）。同じポーズで大きさだけが違うなら、「キャラの高さを合わせる」に"
                           "印を付けると（CLI は --join-match-height）合わせられます。しゃがみ・ジャンプなどポーズの違いなら、そのままで大丈夫です")
                else:
                    msg = (head + f"ので、合わせませんでした（自動で合わせられるのは {HEIGHT_MAX_RATIO * 100:.0f}% まで）。"
                           "AI に同じ大きさで描き直させるか、大きさの近いファイルどうしでつなげてください"
                           "（しゃがみ・ジャンプなどポーズの違いなら、そのままで大丈夫です）")
                warnings.append(msg)
                skipped.append({"stage": "frames", "code": "join_height", "message": msg, "file": name,
                                "height": int(h), "target": int(target), "ratio": round(float(ratio), 3),
                                "suggestions": [], "result": "held"})
        files.append(JoinFile(name=name, frame_count=len(boxes), first=first, detected=tuple(period),
                              dot_size=tuple(got[3]), height=h, height_used=used, status=status, boxes=boxes))
        first += len(boxes)
        t0 = time.perf_counter()
        raw = got[0]
        emit("grid", f"{name}: グリッド検出（ドット {_size(got[3])}"
             + (f"。高さを合わせるため {got[3][1] / period[1]:.3f} 倍" if status == "adjusted" else "")
             + "、線はコマごとに合わせる: " + "、".join(f"{f.shape[1]}×{f.shape[0]}" for f in raw) + " ドット）",
             F._grid_strip(raw), t0)

    raw_all, src_all, periods = [], [], []
    for p in per:
        raw, _, src, period = p[4][:4]
        raw_all += raw
        src_all += src
        periods += [period] * len(raw)
    raw_all, src_all, wwarn, wskip = F._work_frames(raw_all, src_all, periods, s, steps, on_step)
    warnings = wwarn + warnings
    skipped = wskip + skipped
    pos, k = [], 0
    for p in per:
        n = len(p[2])
        pos += F._source_pos(p[2], src_all[k:k + n], p[4][3])
        k += n
    sources = [(f.name, j + 1) for f in files for j in range(f.frame_count)]
    boxes_all = [b for p in per for b in p[2]]
    grids_all = [g for p in per for g in p[4][1]]
    return Prepared(raw=raw_all, pos=pos, boxes=boxes_all, grids=grids_all, files=files, sources=sources, steps=steps,
                    emit=emit, notes=notes, warnings=warnings, skipped=skipped)


def _size(period) -> str:
    from .frames import _size_text
    return _size_text(period)


def _single(name: str, rgba, s, fs, on_step):
    """ファイルが 1 つ: `frames.convert_frames()` と同じ（記録の files・sources を足すだけ。直しは足した後に当てる）。"""
    from . import frames as F
    res = F.convert_frames(rgba, s, dataclasses.replace(fs, edits=None), on_step=on_step)
    h = _height(res.raw)
    files = [JoinFile(name=name, frame_count=len(res.frames), first=1, detected=tuple(res.dot_size),
                      dot_size=tuple(res.dot_size), height=h, height_used=h, boxes=res.boxes)]
    res = dataclasses.replace(res, files=files, sources=[(name, j + 1) for j in range(len(res.frames))])

    def emit(sname, desc, img, t0):
        r = StepResult(sname, desc, img, time.perf_counter() - t0)
        res.steps.append(r)
        if on_step is not None:
            on_step(r)
    return F._with_edits(res, fs, emit)


# ---------------------------------------------------------------- つなぎ目

def _body(f: np.ndarray, rows: int):
    """(上の方の横の中心, 上端, 下端)。上の方は不透明な範囲の上端から rows 行。"""
    op = f[..., 3] > 0
    ys, xs = np.nonzero(op)
    if len(ys) == 0:
        return None
    top, bottom = int(ys.min()), int(ys.max()) + 1
    head = xs[ys < top + rows]
    return float(head.mean() + 0.5), top, bottom


def _shift(a, b) -> tuple[int, int, int]:
    """a から b への (横, 足元, 上端) のずれ（ドット。四捨五入）。"""
    return (int(np.floor(b[0] - a[0] + 0.5)), b[2] - a[2], b[1] - a[1])


def join_report(aligned: list[np.ndarray], files: list[JoinFile], head_ratio: float):
    """ファイルの境目（あるファイルの最後のコマ → 次のファイルの最初のコマ。くり返しの最後 → 最初も）で、体の位置
    （上の方の横の中心・足元・上端）のずれを測る。ファイルの中の隣どうしのコマのずれの最大より大きく、`JOIN_MIN_SHIFT` 以上なら警告する。
    自動では混ぜない・中割りを作らない。戻り値は (記録, 警告, skipped の記録)。
    """
    from .frames import _round
    hs = [int(np.flatnonzero((f[..., 3] > 0).any(1)).size) for f in aligned]
    rows = max(1, _round(head_ratio * float(np.median(hs))))
    body = [_body(f, rows) for f in aligned]
    inner = [0, 0, 0]
    for f in files:
        for k in range(f.first, f.first + f.frame_count - 1):
            a, b = body[k - 1], body[k]
            if a is not None and b is not None:
                inner = [max(v, abs(d)) for v, d in zip(inner, _shift(a, b))]
    limit = [max(JOIN_MIN_SHIFT, v + 1) for v in inner]
    out, warnings, skipped = [], [], []
    pairs = [(files[i], files[i + 1], False) for i in range(len(files) - 1)] + [(files[-1], files[0], True)]
    for fa, fb, loop in pairs:
        ka, kb = fa.first + fa.frame_count - 1, fb.first
        a, b = body[ka - 1], body[kb - 1]
        if a is None or b is None:
            continue
        dx, dfoot, dtop = _shift(a, b)
        large = [abs(v) >= lim for v, lim in zip((dx, dfoot, dtop), limit)]
        rec = {"from_file": fa.name, "to_file": fb.name, "from_frame": int(ka), "to_frame": int(kb), "loop": loop,
               "shift": {"x": dx, "foot": dfoot, "top": dtop}, "large": bool(any(large))}
        out.append(rec)
        if any(large):
            what = "、".join(t for t, on in zip((f"横 {dx:+d}", f"足元 {dfoot:+d}", f"頭の上 {dtop:+d}"), large) if on)
            msg = (f"{'くり返すときの' if loop else ''}{fa.name} の最後（{ka} コマ目）→ {fb.name} の最初（{kb} コマ目）で、"
                   f"体の位置が大きくずれます（{what} ドット。ファイルの中の隣のコマどうしは最大 横 {inner[0]}・足元 {inner[1]}・"
                   f"頭の上 {inner[2]}）。つなぎ目は自動では混ぜません。並び順・そろえ方を確かめるか、境目のコマを描き直してください")
            warnings.append(msg)
            skipped.append({"stage": "frames", "code": "join_shift", "message": msg, "frame": int(kb) - 1,
                            "frames": [int(ka) - 1, int(kb) - 1], "files": [fa.name, fb.name], "loop": loop,
                            "shift": {"x": dx, "foot": dfoot, "top": dtop}, "suggestions": [], "result": "held"})
    return out, warnings, skipped


def file_starts(res) -> list[int]:
    """2 つ目からのファイルの 1 コマ目の番号（1 から）。ちらつきの見える化で境目に印を付けるのに使う。つなげていなければ []。"""
    if not getattr(res, "joined", False):
        return []
    return [f.first for f in res.files[1:]]
