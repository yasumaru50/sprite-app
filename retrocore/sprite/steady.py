"""コマを束ねたシートの「ちらつき」を自動で減らす（`--frames --steady on`）。

    from retrocore.sprite.steady import steady_frames
    frames, report = steady_frames(res.frames, size=4)

AI が描いた歩き・走りのコマは、動きと関係のない小さなドットの違い（輪郭が 1 ドット太い、髪の中の点など）がコマごとに残り、
再生するとちらつく。そろえたコマ（同じ大きさ・同じ足元。体全体の上下 `root_y` を付ける前）の 2 コマ目以降を
基準のコマ（1 コマ目）と比べ、**上の方（頭のあたり）の小さな違いだけ**を基準のコマの色に戻す。
基準のコマのドットを写すだけなので、新しい色は作らない。

手順（コマごと。どれも迷ったら何もしない）:

1. **位置を合わせる**: 上の方（既定はキャンバスの上から半分。ちらつきを数える範囲と同じ）で、基準のコマを横 ±1・縦 ±2 ドット
   ずらして重ね、違うドットがいちばん少ないずれを探す（頭が 1〜2 ドット上下するのは意図した動き。ずれを戻した位置で比べる）
2. **違うドットをまとまりに分ける**: 片方だけ不透明か色が違うドットを、(基準の色, このコマの色) の組が同じものどうしの
   8 方向のつながりに分ける。AI の描き直しでは、輪郭のずれと髪の中の点などが全部触れ合って 1 つの大きなかたまりになるため
   （ai07・ai09 では上の方の違いのほとんどが 1 つにつながった）、色の組で分ける
3. **小さなまとまり**: ドット数が `size` 以下の「小さな点」か、幅か高さが 1 ドットで長さが `size` の `LINE_FACTOR` 倍以下の「細い線」。
   それより大きいまとまり（腕・脚・髪の束の形の違い）は動きとして残す
4. **大きなまとまりの近く**（`NEAR_LARGE` ドット以内）の小さな違いは残す（腕・脚の縁の可能性）。組（触れ合っている小さなまとまり）の
   ドット数が `size` の `LINE_FACTOR` 倍を超えるときも残す
5. **その場所で位置が決まるか**（顔の目・口を、ずれたまま写して二重・欠けにしないため）: 組のまわり `WINDOW` ドットで、
   基準のコマをずらしたどのずれがいちばんよく重なるかを調べ、上の方のずれがその中にあり、どれで写しても同じ色になるときだけ戻す
   （頭だけが体と違う量で動いた所や、のっぺりした所でずれが決まらない所は戻さない）
6. **近くの違いが全部戻せるときだけ戻す**: 戻す組のまわり `NEIGHBOR` ドット以内に、戻さない違いが 1 つでもあれば戻さない
   （戻した所のまわりは基準のコマと同じになり、目の半分だけが戻ってちぐはぐになることがない。ai07 の 4 コマ目の目で起きたので足した）
7. **戻した結果を確かめる**（組ごと。だめなら戻さない）:
   - 透明の穴を作らない（透明になったドットが外の透明とつながっている）。外の透明を囲んで穴にしない
   - 暗い輪郭を途切れさせない（組のまわりで、輪郭の色のつながりの数が増えない）
   - 足元（いちばん下の不透明な行）を動かさない
   - 目・口などの「小さな部品」（同じ色のつながりで `PART_MAX` ドット以下、透明に触れない）に触れる組は、
     戻した後にその部品のまわりが基準のコマとすっかり同じになるときだけ戻す
   確かめで外れた組があれば、それを除いて 6. からやり直す

上の方の外（脚・足）は戻さない（歩き・走りの動きそのもので、小さな違いでも意図した動きのことが多いため）。
同じ元のコマは同じ結果になる（「行って戻る」1,2,3,2 の 2 コマ目は 1 回だけ計算する）。乱数は使わない。
`retrocore` の方針どおり NumPy と Pillow だけを使う（np.bincount などは使っていない）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

MODES = {
    "off": "ちらつきを減らさない（既定。今までと同じ）",
    "on": "2 コマ目以降の上の方の小さな違い（点・細い線）を 1 コマ目の色に戻す",
}
DEFAULT_MODE = "off"
DEFAULT_SIZE = 4            # 「小さな点」とみなす最大のドット数
MIN_SIZE, MAX_SIZE = 1, 32
LINE_FACTOR = 3             # 「細い線」と組のドット数の上限は size の何倍か
NEAR_LARGE = 2              # 大きなまとまりからこのドット数以内（8 方向の距離）にある小さな違いは戻さない
WINDOW = 3                  # 組のまわりの何ドットの窓で位置を確かめるか
NEIGHBOR = 2                # 戻す組のまわりこのドット数以内の違いが、全部戻せるときだけ戻す
MAX_ROUNDS = 20             # 戻した結果の確かめで外れた組を除いてやり直す回数の上限
PART_MAX = 12               # 「小さな部品」（目・口・頬など）とみなす同じ色のつながりの最大のドット数
SHIFT_X, SHIFT_Y = 1, 2     # 基準のコマに重ねるずれの範囲（ちらつきの数え方と同じ）
DEFAULT_ROWS_RATIO = 0.5    # 戻す範囲: キャンバスの上から半分（ちらつきを数える範囲と同じ）

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SHIFTS = [(dx, dy) for dy in range(-SHIFT_Y, SHIFT_Y + 1) for dx in range(-SHIFT_X, SHIFT_X + 1)]

# 戻さなかった理由（JSON の kept_reasons のキー）
REASON_LARGE = "large"            # 大きなまとまり・大きすぎる組（動き）
REASON_NEAR = "near_large"        # 大きなまとまりの近く（腕・脚の縁の可能性）
REASON_LOWER = "lower"            # 戻す範囲（上の方）の外
REASON_UNALIGNED = "unaligned"    # その場所で位置が決まらない
REASON_HOLE = "hole"              # 透明の穴ができる
REASON_OUTLINE = "outline"        # 暗い輪郭が途切れる
REASON_FOOT = "foot"              # 足元が動く
REASON_MIXED = "near_kept"        # まわりに戻さない違いがある（戻すと形がちぐはぐになる）
REASON_FACE = "small_part"        # 目・口などの小さな部品が基準のコマと同じ形にならない
REASON_EDGE = "edge"              # 基準のコマをずらしたとき、キャンバスの外から来る所


def validate(mode: str, size) -> None:
    if mode not in MODES:
        raise ValueError(f"ちらつきを減らす設定は {' / '.join(MODES)} のどれかです: {mode}")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or not MIN_SIZE <= int(size) <= MAX_SIZE:
        raise ValueError(f"小さいとみなすドット数は {MIN_SIZE}〜{MAX_SIZE} の整数で指定してください: {size}")


def default_rows(height: int) -> int:
    return max(1, int(np.floor(DEFAULT_ROWS_RATIO * height + 0.5)))


@dataclass
class SteadyFrame:
    """1 コマの結果。`restored` は戻したドット、`kept` は戻さずに残した違うドット（どちらもコマの座標の真偽）。"""

    frame: int                          # 元のコマの番号（1 から）
    shift: tuple                        # 上の方で基準のコマにいちばんよく重なったずれ (dx, dy)
    restored: np.ndarray
    kept: np.ndarray
    reasons: dict = field(default_factory=dict)   # 戻さなかった違うドットの、理由ごとの数

    @property
    def restored_count(self) -> int:
        return int(self.restored.sum())

    @property
    def kept_count(self) -> int:
        return int(self.kept.sum())


@dataclass
class SteadyReport:
    mode: str
    size: int
    reference: int
    rows: int                           # 戻す範囲（上から何行）
    frames: list[SteadyFrame]

    def info(self) -> dict:
        """シートの JSON の `steady` に書く辞書。ドットの数はコマ全体（kept_dots は上の方の外の違いも含む）。"""
        out = {"mode": self.mode, "size": int(self.size), "reference_frame": int(self.reference),
               "rows": int(self.rows), "frames": []}
        for f in self.frames:
            out["frames"].append({
                "source_frame": int(f.frame), "shift": {"x": int(f.shift[0]), "y": int(f.shift[1])},
                "restored_dots": f.restored_count, "kept_dots": f.kept_count,
                "kept_reasons": {k: int(v) for k, v in sorted(f.reasons.items())}})
        return out

    def counts(self) -> list[int]:
        """各コマ（元の並び順。基準のコマは 0）で戻したドットの数。"""
        by = {f.frame: f.restored_count for f in self.frames}
        n = max([self.reference] + [f.frame for f in self.frames])
        return [by.get(i, 0) for i in range(1, n + 1)]


def _shifted(a: np.ndarray, dx: int, dy: int):
    """a を (dx, dy) ずらしたものと、中身がキャンバスの中から来た所の真偽。"""
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    valid = np.zeros((h, w), bool)
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    valid[y0:y1, x0:x1] = True
    return out, valid


def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    oa, ob = a[..., 3] > 0, b[..., 3] > 0
    return (oa != ob) | (oa & ob & (a[..., :3] != b[..., :3]).any(-1))


def _keys(img: np.ndarray) -> np.ndarray:
    """各ドットの色の番号（RGB を 1 つの数に。透明は -1）。"""
    k = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2].astype(np.int64)
    return np.where(img[..., 3] > 0, k, -1)


def best_shift(ref: np.ndarray, frame: np.ndarray, rows: int) -> tuple[int, int]:
    """上から rows 行で、基準をずらして重ねたとき違うドットがいちばん少ないずれ（同じなら小さいずれ）。"""
    best = None
    for dx, dy in _SHIFTS:
        s, _ = _shifted(ref, dx, dy)
        key = (int(_diff(s[:rows], frame[:rows]).sum()), abs(dx) + abs(dy), abs(dy), abs(dx))
        if best is None or key < best[0]:
            best = (key, (dx, dy))
    return best[1]


def _components(mask: np.ndarray, same: np.ndarray | None = None) -> list[list[tuple[int, int]]]:
    """真のドットの 8 方向のつながり（上から、左から見つけた順）。`same` を渡すと、その値が同じドットどうしだけつなぐ。"""
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    m = mask.tolist()
    sv = same.tolist() if same is not None else None
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        y0, x0 = int(y0), int(x0)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q = deque([(y0, x0)])
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy, dx in _N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and m[yy][xx] and not seen[yy, xx] \
                        and (sv is None or sv[yy][xx] == sv[y0][x0]):
                    seen[yy, xx] = True
                    q.append((yy, xx))
        out.append(pts)
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """8 方向に r ドットふくらませる。"""
    h, w = mask.shape
    out = mask.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        nxt = p[1:-1, 1:-1].copy()
        for dy, dx in _N8:
            nxt |= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        out = nxt
    return out


def _exterior(opaque: np.ndarray) -> np.ndarray:
    """キャンバスの外とつながった透明（4 方向）。"""
    h, w = opaque.shape
    t = (~opaque).tolist()
    ext = np.zeros((h, w), bool)
    q = deque()
    for y in range(h):
        for x in range(w):
            if (y in (0, h - 1) or x in (0, w - 1)) and t[y][x]:
                ext[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in _N4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and t[yy][xx] and not ext[yy, xx]:
                ext[yy, xx] = True
                q.append((yy, xx))
    return ext


def _line_mask(img: np.ndarray, colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return np.zeros(img.shape[:2], bool)
    return np.isin(_keys(img), colors)


def outline_keys(ref: np.ndarray) -> np.ndarray:
    """基準のコマの輪郭の色の番号（縁取りの判定 `detect_outline()` と同じ。輪郭が無ければ空）。"""
    from .outline import detect_outline
    det = detect_outline(ref)
    if not det.has_outline:
        return np.zeros(0, np.int64)
    return np.array(sorted((int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in det.colors), np.int64)


def small_parts(img: np.ndarray) -> np.ndarray:
    """「小さな部品」（同じ色の 8 方向のつながりで PART_MAX ドット以下、上下左右に透明が無い）の番号（それ以外は -1）。
    戻す範囲の外にはみ出す部品も数える（範囲の境目が目を横切るとき、上の半分だけが戻らないように）。"""
    h, w = img.shape[:2]
    keys = _keys(img)
    op = keys >= 0
    p = np.pad(op, 1)
    inner = op & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    out = np.full((h, w), -1, np.int64)
    k = 0
    for pts in _components(op, keys):
        if len(pts) > PART_MAX:
            continue
        ys = np.array([q[0] for q in pts], np.int64)
        xs = np.array([q[1] for q in pts], np.int64)
        if not inner[ys, xs].all():
            continue
        out[ys, xs] = k
        k += 1
    return out


def _bottom(opaque: np.ndarray) -> int:
    rows = np.flatnonzero(opaque.any(1))
    return int(rows[-1]) if len(rows) else -1


def _static_check(ys, xs, gm, rows, size, near, valid, small, shifted, shift, frame):
    """コマを直す前に決まる、組を戻さない理由（戻してよければ None）。"""
    h, w = gm.shape
    n = len(ys)
    if (ys >= rows).any():
        return REASON_LOWER
    if n > LINE_FACTOR * size:
        return REASON_LARGE
    if near[ys, xs].any():
        return REASON_NEAR
    if not valid[ys, xs].all():
        return REASON_EDGE
    # その場所で位置が決まるか: 組のまわり WINDOW ドット（小さな違いのドットは除く）で違うドットがいちばん少ないずれの中に
    # 上の方のずれがあり、いちばん少ないずれのどれで写しても組の色が同じ
    y0, y1 = max(0, int(ys.min()) - WINDOW), min(h, int(ys.max()) + WINDOW + 1)
    x0, x1 = max(0, int(xs.min()) - WINDOW), min(w, int(xs.max()) + WINDOW + 1)
    ring = ~(small[y0:y1, x0:x1] | gm[y0:y1, x0:x1])
    scores = []
    for s in _SHIFTS:
        sh, v = shifted[s]
        if not v[ys, xs].all():
            continue
        scores.append((int((_diff(sh[y0:y1, x0:x1], frame[y0:y1, x0:x1]) & ring).sum()), s))
    low = min(c for c, _ in scores)
    bests = [s for c, s in scores if c == low]
    if shift not in bests or len({shifted[s][0][ys, xs].tobytes() for s in bests}) != 1:
        return REASON_UNALIGNED
    return None


def _apply(frame, src, cands, line_colors, parts):
    """候補の組を順に戻し、戻した結果を確かめる。戻り値は (直したコマ, 戻したドット, {外れた組: 理由})。"""
    h, w = frame.shape[:2]
    src_part_id, frame_part_id, any_part = parts
    out = frame.copy()
    restored = np.zeros((h, w), bool)
    failed = {}
    opaque = out[..., 3] > 0
    ext = _exterior(opaque)
    foot = _bottom(opaque)
    for gi, (ys, xs, gm) in cands:
        new = out.copy()
        new[ys, xs] = src[ys, xs]
        nop = new[..., 3] > 0
        if _bottom(nop) != foot:
            failed[gi] = REASON_FOOT
            continue
        # 目・口などの小さな部品に触れる組は、戻した後にその部品のまわりが基準のコマとすっかり同じになるときだけ戻す
        touch = _dilate(gm, 1)
        if (any_part & touch).any():
            area = np.zeros((h, w), bool)
            for lab_ in (src_part_id, frame_part_id):
                for v in np.unique(lab_[touch]):
                    if v >= 0:
                        area |= lab_ == v
            area = _dilate(area, 1)
            if _diff(new[area][None], src[area][None]).any():
                failed[gi] = REASON_FACE
                continue
        gone = opaque[ys, xs] & ~nop[ys, xs]
        added = ~opaque[ys, xs] & nop[ys, xs]
        ext2 = ext
        if gone.any() or added.any():
            ext2 = _exterior(nop)
            if (gone & ~ext2[ys, xs]).any() or (ext & ~ext2 & ~nop).any():
                failed[gi] = REASON_HOLE
                continue
        if len(line_colors):
            y0, y1 = max(0, int(ys.min()) - 2), min(h, int(ys.max()) + 3)
            x0, x1 = max(0, int(xs.min()) - 2), min(w, int(xs.max()) + 3)
            before = _line_mask(out[y0:y1, x0:x1], line_colors)
            after = _line_mask(new[y0:y1, x0:x1], line_colors)
            if (before != after).any() and len(_components(after)) > len(_components(before)):
                failed[gi] = REASON_OUTLINE
                continue
        out, opaque, ext = new, nop, ext2
        restored[ys, xs] = True
    return out, restored, failed


def steady_frame(ref: np.ndarray, frame: np.ndarray, size: int = DEFAULT_SIZE, rows: int | None = None,
                 line_colors: np.ndarray | None = None, number: int = 2):
    """1 コマ（`frame`）の上の方の小さな違いを基準のコマ（`ref`）の色に戻す。戻り値は (直したコマ, SteadyFrame)。"""
    h, w = frame.shape[:2]
    rows = default_rows(h) if rows is None else min(max(int(rows), 1), h)
    if line_colors is None:
        line_colors = outline_keys(ref)
    shift = best_shift(ref, frame, rows)
    shifted = {s: _shifted(ref, *s) for s in _SHIFTS}
    src, valid = shifted[shift]
    diff = _diff(src, frame)

    # 1 つの色の組のまとまりごとに、小さいか大きいか
    pair = (_keys(src) + 1) * (1 << 25) + (_keys(frame) + 1)
    large = np.zeros((h, w), bool)
    for pts in _components(diff, pair):
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        n = len(pts)
        thin = min(max(xs) - min(xs), max(ys) - min(ys)) == 0
        if not (n <= size or (thin and n <= LINE_FACTOR * size)):
            for y, x in pts:
                large[y, x] = True
    near = _dilate(large, NEAR_LARGE)
    small = diff & ~large
    # 組: 触れ合っている小さなまとまり（全部戻すか全部残すか）
    groups = sorted(_components(small), key=min)
    # 目・口などの小さな部品（同じ色のつながりで PART_MAX ドット以下、透明に触れない）
    src_part_id = small_parts(src)
    frame_part_id = small_parts(frame)
    parts_src, parts_frame = src_part_id >= 0, frame_part_id >= 0

    reasons_of: dict[int, str] = {}      # 戻さなかった組 → 理由
    masks = []
    for gi, pts in enumerate(groups):
        ys = np.array([p[0] for p in pts], np.int64)
        xs = np.array([p[1] for p in pts], np.int64)
        gm = np.zeros((h, w), bool)
        gm[ys, xs] = True
        masks.append((ys, xs, gm))
        r = _static_check(ys, xs, gm, rows, size, near, valid, small, shifted, shift, frame)
        if r:
            reasons_of[gi] = r
    # 近くの違いが全部戻せるときだけ戻す（戻した所のまわりが基準のコマと同じになる。目の半分だけ戻るのを防ぐ）。
    # 戻した結果の確かめで外れた組があれば、それを除いてやり直す
    for _ in range(MAX_ROUNDS):
        while True:
            cand = [gi for gi in range(len(groups)) if gi not in reasons_of]
            allowed = np.zeros((h, w), bool)
            for gi in cand:
                allowed |= masks[gi][2]
            drop = [gi for gi in cand if (diff & _dilate(masks[gi][2], NEIGHBOR) & ~allowed).any()]
            if not drop:
                break
            for gi in drop:
                reasons_of[gi] = REASON_MIXED
        out, restored, failed = _apply(frame, src, [(gi, masks[gi]) for gi in cand], line_colors,
                                       (src_part_id, frame_part_id, parts_src | parts_frame))
        if not failed:
            break
        reasons_of.update(failed)
    else:   # やり直しが終わらないときは何もしない（迷ったら何もしない）
        out, restored = frame.copy(), np.zeros((h, w), bool)
        for gi in range(len(groups)):
            reasons_of.setdefault(gi, REASON_MIXED)
    reasons: dict = {}
    if large.any():
        reasons[REASON_LARGE] = int(large.sum())
    for gi, r in reasons_of.items():
        reasons[r] = reasons.get(r, 0) + len(groups[gi])
    kept = diff & ~restored
    return out, SteadyFrame(frame=number, shift=shift, restored=restored, kept=kept, reasons=reasons)


def steady_frames(frames: list[np.ndarray], size: int = DEFAULT_SIZE, reference: int = 1,
                  rows: int | None = None) -> tuple[list[np.ndarray], SteadyReport]:
    """そろえたコマ（同じ大きさ）の、基準のコマ以外の小さな違いを基準のコマの色に戻す。戻り値は (コマ, SteadyReport)。"""
    validate("on", size)
    if not frames:
        raise ValueError("コマがありません")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("コマの大きさがそろっていません")
    h = frames[0].shape[0]
    rows = default_rows(h) if rows is None else int(rows)
    if not 1 <= rows <= h:
        raise ValueError(f"ちらつきを減らす行は 1〜{h} にしてください: {rows}")
    ref = frames[reference - 1]
    colors = outline_keys(ref)
    out, rep = [], []
    for i, f in enumerate(frames):
        if i == reference - 1:
            out.append(f.copy())
            continue
        g, sf = steady_frame(ref, f, int(size), rows, colors, i + 1)
        out.append(g)
        rep.append(sf)
    # 念のため: 新しい色を作っていない（基準のコマかそのコマにある色だけ）
    ref_keys = set(np.unique(_keys(ref)).tolist())
    for f, g in zip(frames, out):
        if not set(np.unique(_keys(g)).tolist()) <= ref_keys | set(np.unique(_keys(f)).tolist()):
            raise RuntimeError("ちらつきを減らす段で新しい色ができました（不具合）")
    return out, SteadyReport("on", int(size), reference, rows, rep)


def steady_image(frames: list[np.ndarray], report: SteadyReport, scale: int = 8) -> np.ndarray:
    """違いの図（RGBA）。直した後の各コマを scale 倍にし、赤＝基準のコマに戻したドット、青＝動きとして残したドット。
    水色の線から上が戻す範囲。"""
    from PIL import Image, ImageDraw

    from .frames import _draw_label
    h, w = frames[0].shape[:2]
    gap = scale
    n = len(frames)
    W = n * w * scale + (n + 1) * gap
    H = h * scale + 2 * gap + 12
    canvas = Image.new("RGBA", (W, H), (200, 200, 200, 255))
    draw = ImageDraw.Draw(canvas)
    by = {f.frame: f for f in report.frames}
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where((((yy + xx) % 2) == 1)[..., None], 222, 236).astype(np.float64)
    for i, f in enumerate(frames):
        x0, y0 = gap + i * (w * scale + gap), gap + 12
        a = f[..., 3:4].astype(np.float64) / 255
        img = f[..., :3] * a + bg * (1 - a)
        im = Image.fromarray(img.astype(np.uint8), "RGB").resize((w * scale, h * scale), Image.NEAREST)
        canvas.paste(im, (x0, y0))
        ly = y0 + report.rows * scale
        draw.line([x0, ly, x0 + w * scale - 1, ly], fill=(0, 170, 255, 255), width=max(1, scale // 4))
        sf = by.get(i + 1)
        if sf is None:
            _draw_label(draw, x0, 1, f"#{i + 1}  (ref)")
            continue
        for mask, col in ((sf.kept, (40, 90, 255, 255)), (sf.restored, (255, 30, 30, 255))):
            ys, xs = np.nonzero(mask)
            for y, x in zip(ys, xs):
                draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                               outline=col, width=max(1, scale // 4))
        _draw_label(draw, x0, 1, f"#{i + 1}  fix {sf.restored_count}  keep {sf.kept_count}")
    return np.asarray(canvas)
