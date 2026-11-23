"""横揺れ: 足元は動かさず、体の上の部分が左右に 1 ドットずつ揺れる控えめな動き（中央 → 右 → 中央 → 左）。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」の「横揺れ」）:
左端から右端まで、各列で「この行から上が揺れる」行を 1 つずつ選んでつないだ「分け目」を 1 本決め、分け目から上のドットを
左右に 1 ドットずらす。分け目より下は 1 ドットも変わらないので、足元は動かない。ドットを横に動かすだけなので新しい色は出ない。

分け目は呼吸の継ぎ目と同じように、列ごとに高さを変えてよい（まっすぐな 1 行だけで分けると、丸い体の左右の斜めの輪郭が
ずれて途切れるため）。高さが変わる所では、同じ行の中で「ずれる部分」と「ずれない部分」が隣り合う。そこでは 1 ドットが重なるか
1 ドット空くので、横に同じ色が 3 ドット以上続く所（塗りの中・体の外の透明）でだけ高さを変えてよいことにする
（重なっても空いても、同じ色の部分が 1 ドット短くなるか長くなるだけで、形は変わらない）。

分け目は、次の「目立ち方」の合計が最も小さくなるものを動的計画法で選ぶ（乱数なし）:
- 分け目の上と下で縦に続く色の境目（輪郭・部品の縁）があると、ずれて段になる。その数だけ目立つ（`JOG`）。
  境目が短い（`SHORT_EDGE` 行以下。目の横の縁など）と、部品の形がゆがむので大きく目立つ（`PART`）
- 高さを変えるのは 1 行につき少しだけ目立つ（`STEP`。不要にくねくねしないように）
- 目・頬・ボタンのような小さな部品（同じ色の `PART_MAX` ドット以下の塊）の中を通ると、部品の上だけがずれてゆがむので、
  大きく目立つ（`PART`）。部品のまわりを回って、部品ごと揺れるか、部品ごと止まるようにする
- 腰のあたり（上から 6 割の高さ）から離れるほど少しずつ目立つ（`BIAS`）

作ったあとに、右にも左にもずらした形で決まりごと（`check_rules()`）と、色ごとの穴の数（輪のような線が切れると穴が減る。
1 ドットの線や輪郭の輪が途切れていないことの確かめ）、絵の中の閉じた透明の数（隙間ができていない）を確かめ、守れなければ
次の候補の分け目を試す。どれも守れなければ、横揺れを付けずに警告する（迷ったら動かさない）。

揺れ幅を 2 ドットにするときは分け目を 2 本使う。上の分け目は下の分け目より完全に上に置き、上の分け目から上はさらに
1 ドットずらす（上ほど大きく動き、体が少し傾く）。隣り合う 2 行のずれは、どこでも 1 ドットまで。

揺れてはみ出す分の余白: 揺れる部分が画像の左右の端から揺れ幅より近いときは、左右に同じ数の透明の列を足す（全フレームで同じ大きさ。
足元はフレームの真ん中のまま）。足した数は結果の `info["padding"]` と JSON の `padding` に書く。足す必要が無ければ大きさは元のまま。

分ける高さの指定（`split`）: 呼吸と同じく、省くか "auto" なら自動、数なら「画像の下から何ドット目のあたりから上を揺らすか」
（1 = いちばん下の行）。分け目はその上下 `SPLIT_RANGE` ドット以内に限る。範囲で決まりを守れなければ、ほかの高さに移さず、
揺れを付けずに警告し、近くで揺らせる高さを知らせる。
"""

from __future__ import annotations

import math

import numpy as np

from .base import (AUTO, MAX_FRAMES, Fail, MotionResult, body_limits, check_durations, check_frames, check_rules,
                   RULE_TEXT, check_split, color_keys, frame_problems, problem_pixels,
                   split_limits)

NAME = "sway"
LABEL = "横揺れ（足元は動かさず、体の上の部分が左右に 1 ドット揺れる）"
DEFAULT_FRAMES = 4        # 中央 → 右 → 中央 → 左
MIN_FRAMES = 3            # 2 フレームでは波が中央の 2 枚になり、揺れない
DEFAULT_DURATION = 300    # 1 フレームの表示時間（ミリ秒）。4 フレームで 1.2 秒に 1 往復
DEFAULT_AMOUNT = 1        # 揺れ幅（ドット）
MAX_AMOUNT = 2

BAND_TOP = 0.35           # 分け目を探す範囲の上の端（絵の高さに対する割合。頭のてっぺんだけが揺れるのを避ける）
TARGET = 0.6              # 分けたい高さ（絵の上から 6 割。腰のあたり）
BIAS = 0.05               # TARGET から 1 行離れるごとの目立ち方（1 列ごと）
JOG = 1.0                 # 縦に続く色の境目が、ずれて段になる目立ち方
STEP = 0.2                # 分け目の高さを 1 行変える目立ち方
PART = 3.0                # 小さな部品（目など）の中を通る目立ち方（1 列ごと）
PART_MAX = 24             # 小さな部品とみなす大きさ（ドット）
SHORT_EDGE = 4            # これ以下の長さの縦の色の境目は、小さな部品の縁とみなす（ずれると PART だけ目立つ）
RUN_MIN = 3               # 分け目の高さを変えてよい、横に同じ色が続く長さ
SPLIT_RANGE = 2           # 高さを指定したとき、分け目が指定した高さから離れてよい量（ドット）
WINDOW = 4                # 候補を増やすために分け目を探す小さな帯の高さ（行）
UPPER_TARGET = 0.5        # 揺れ幅 2 の上の分け目: 絵の上の端から下の分け目までの真ん中のあたり
MAX_TRIES = 40            # 決まりを確かめる候補の数の上限（作れないときに時間がかかりすぎないように）
_INF = 1e9


def sway_offsets(frames: int = DEFAULT_FRAMES, amount: int = DEFAULT_AMOUNT) -> list[int]:
    """各フレームで揺れる量（ドット。+ が右）。ゆっくりした波（sin）を整数に丸める（ちょうど半分のときは動かさない）。

    例: 4 フレーム → [0, 1, 0, -1]、8 フレーム・揺れ幅 2 → [0, 1, 2, 1, 0, -1, -2, -1]、
    8 フレーム・揺れ幅 1 → [0, 1, 1, 1, 0, -1, -1, -1]。
    """
    out = []
    for i in range(frames):
        v = amount * math.sin(2 * math.pi * i / frames)
        m = int(math.ceil(abs(v) - 0.5 - 1e-9))
        out.append(m if v > 0 else -m)
    return out


def shift_above(rgba: np.ndarray, cut: np.ndarray, s: int) -> np.ndarray:
    """各列で `cut` の行から上のドットを、横に s（+1 = 右、-1 = 左）ずらす。分け目より下は変えない。

    同じ行で「ずれる部分」と「ずれない部分」が隣り合う所では、ずれる部分が隣に重なる（隣の 1 ドットが消える）か、
    1 ドット空いた所にずれる前のドットが残る。分け目は、そこが横に同じ色が続く所になるように選ぶ。
    """
    h, w = rgba.shape[:2]
    moving = np.arange(h)[:, None] <= np.asarray(cut)[None, :]           # (H, W)
    src = np.arange(w)[None, :] - s                                      # 1 つ手前（ずれてくる元）の列
    inside = (src >= 0) & (src < w)
    srcc = np.clip(src, 0, w - 1)
    from_moving = np.take_along_axis(moving, np.broadcast_to(srcc, (h, w)).astype(np.intp), axis=1) & inside
    cols = np.where(from_moving, np.broadcast_to(srcc, (h, w)), np.arange(w)[None, :])
    return np.take_along_axis(rgba, cols.astype(np.intp)[..., None], axis=1).copy()


def _hruns(key: np.ndarray) -> np.ndarray:
    """各ドットが入っている、横に同じ色が続く部分の長さ。"""
    w = key.shape[1]
    left = np.ones(key.shape, np.int32)
    right = np.ones(key.shape, np.int32)
    for x in range(1, w):
        left[:, x] = np.where(key[:, x] == key[:, x - 1], left[:, x - 1] + 1, 1)
    for x in range(w - 2, -1, -1):
        right[:, x] = np.where(key[:, x] == key[:, x + 1], right[:, x + 1] + 1, 1)
    return left + right - 1


def _row_runs(row: list) -> list[tuple[int, int, int]]:
    """1 行の、同じ色が続く部分 (色, 始まり, 終わり) の並び（透明は含めない）。"""
    out = []
    start = 0
    for x in range(1, len(row) + 1):
        if x == len(row) or row[x] != row[start]:
            if row[start] >= 0:
                out.append((row[start], start, x - 1))
            start = x
    return out


def _linked(a1: int, a2: int, c1: int, c2: int) -> bool:
    """横の範囲 [a1, a2] と、すぐ下の行の [c1, c2] が隣り合う（斜めを含む）か。"""
    return a1 <= c2 + 1 and c1 <= a2 + 1


def crossing_bad(key: np.ndarray) -> np.ndarray:
    """分け目を r 行目（r 行目から上が揺れる）・x 列目に置くと、そのあたりで線や輪郭が切れるか（(H, W) の真偽）。

    分け目のそばはまっすぐと見なし、上の行だけを左右に 1 ドットずらしたときに、次のどれかが起きる列に印を付ける:
    - 上の行と下の行の同じ色の部分（と、絵のある所）が、斜めだけで接していて、ずらすと離れる（1 ドットの線・輪郭が途切れる）
    - 離れていた同じ色の部分が、ずらすと接する（部品がくっつく）
    - 輪郭の色でないドットが、ずらしたあとにシルエットの縁に出る（輪郭に穴があく）
    印の無い列だけを通る分け目でも、最後に全体を確かめる（ここは候補を絞るための目安）。
    """
    h, w = key.shape
    bad = np.zeros((h, w), bool)
    opaque = key >= 0
    from .base import silhouette_edge
    edge_keys = set(np.unique(key[silhouette_edge(opaque)]).tolist())
    on_edge = np.isin(key, list(edge_keys)) if edge_keys else np.zeros_like(opaque)
    rows = key.tolist()
    orow = np.where(opaque, 0, -1).tolist()
    for r in range(h - 1):
        for src in (rows, orow):
            up, down = _row_runs(src[r]), _row_runs(src[r + 1])
            for k, a1, a2 in up:
                for k2, c1, c2 in down:
                    if k != k2 or c1 > a2 + 3 or c2 < a1 - 3:
                        continue
                    before = _linked(a1, a2, c1, c2)
                    for s in (1, -1):
                        if before != _linked(a1 + s, a2 + s, c1, c2):
                            lo, hi = max(a1, c1) - 1, min(a2, c2) + 1   # 接している（接する）あたりの列
                            if lo > hi:
                                lo, hi = hi, lo
                            bad[r, max(0, lo):min(w, hi + 1)] = True
        for s in (1, -1):
            # ずらしたあとの上の行（s だけずれる）
            up = np.full(w, False)
            src_x = np.arange(w) - s
            ok = (src_x >= 0) & (src_x < w)
            up[ok] = opaque[r, src_x[ok]]
            newly_top = opaque[r + 1] & opaque[r] & ~up                  # 下の行のドットの上が透明になる
            bad[r] |= newly_top & ~on_edge[r + 1]
            down_at = np.full(w, False)
            down_at[ok] = opaque[r + 1, np.arange(w)[ok]]
            moved_from = np.flatnonzero(ok & up)                         # ずれたあとの位置
            exposed = moved_from[~opaque[r + 1, moved_from] & opaque[r + 1, moved_from - s]]
            orig = exposed - s                                           # 上の行のドットの元の列
            bad[r, orig[~on_edge[r, orig]]] = True
    return bad


def small_parts(key: np.ndarray) -> np.ndarray:
    """小さな部品（同じ色が 8 近傍でつながった `PART_MAX` ドット以下の塊。目・頬・ボタンなど）の上下の境目 ((H, W) の真偽)。

    r 行目と r+1 行目の x 列目が同じ小さな部品なら真。分け目がここを通ると、部品の上だけがずれて形がゆがむ。
    """
    h, w = key.shape
    label = np.full((h, w), -1, np.int64)
    sizes = []
    k = key.tolist()
    lab = label.tolist()
    for y0 in range(h):
        for x0 in range(w):
            if k[y0][x0] < 0 or lab[y0][x0] >= 0:
                continue
            n = len(sizes)
            lab[y0][x0] = n
            stack = [(y0, x0)]
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and lab[yy][xx] < 0 and k[yy][xx] == k[y0][x0]:
                            lab[yy][xx] = n
                            stack.append((yy, xx))
            sizes.append(size)
    label = np.array(lab, np.int64)
    small = np.zeros((h, w), bool)
    if sizes:
        is_small = np.array(sizes) <= PART_MAX
        same = (label[:-1] == label[1:]) & (label[:-1] >= 0)
        small[:-1] = same & is_small[np.clip(label[:-1], 0, None)]
    return small


def _prepare(key: np.ndarray):
    """(各行・各列の境目でずれて段になるか, 高さを変えてはいけない所の数の累積, 分け目を置けない所)。列の境目 x は x と x+1 の間。"""
    h, w = key.shape
    vedge = key[:, 1:] != key[:, :-1]                                   # (H, W-1) 横に色が変わる
    # 縦に続く色の境目の長さ。短い境目（SHORT_EDGE 行以下）は目などの小さな部品の縁なので、ずれると形がゆがんで目立つ
    up = np.zeros((h, w - 1), np.int32)
    down = np.zeros((h, w - 1), np.int32)
    for y in range(h):
        up[y] = np.where(vedge[y], (up[y - 1] if y else 0) + 1, 0)
    for y in range(h - 1, -1, -1):
        down[y] = np.where(vedge[y], (down[y + 1] if y < h - 1 else 0) + 1, 0)
    length = up + down - 1
    jog = np.zeros((h, w - 1))
    cross = vedge[:-1] & vedge[1:]                                     # 分け目が r 行目なら、r と r+1 の両方で変わる
    jog[:-1] = np.where(cross, np.where(length[:-1] <= SHORT_EDGE, PART, JOG), 0.0)
    run = _hruns(key)
    ok = ~vedge & (run[:, :-1] >= RUN_MIN)                              # 同じ色が RUN_MIN 以上続く所の中
    bad = np.zeros((h + 1, w - 1), np.int64)
    bad[1:] = np.cumsum(~ok, axis=0)
    return jog, bad, crossing_bad(key), small_parts(key)


def _cut_in_band(jog, bad, cross, small, lo: int, hi: int, target: float) -> tuple[np.ndarray, float] | None:
    """lo〜hi 行の中で、目立ち方の合計が最も小さい分け目（動的計画法）。作れなければ None。"""
    w = jog.shape[1] + 1
    band = np.arange(lo, hi + 1)
    nb = len(band)
    node_all = (BIAS * np.abs(band - target)[:, None] + np.where(cross[lo:hi + 1], _INF, 0.0)
                + PART * small[lo:hi + 1])                                     # (B, W)
    node = node_all[:, 0]
    a, b = np.meshgrid(band, band, indexing="ij")                        # [前の列の行, 今の列の行]
    lo_r, hi_r = np.minimum(a, b), np.maximum(a, b)
    step = STEP * (hi_r - lo_r)
    cost = node.copy()
    back = np.zeros((nb, w), np.intp)
    for x in range(1, w):
        g = x - 1
        # 高さを変える行（lo_r+1〜hi_r）がすべて「同じ色が続く所の中」なら変えてよい
        nbad = bad[hi_r + 1, g] - bad[lo_r + 1, g]
        trans = np.where(nbad == 0, step, _INF)
        trans = trans + np.where(a == b, jog[a, g], 0.0)
        tot = cost[:, None] + trans
        arg = np.argmin(tot, axis=0)
        back[:, x] = arg
        cost = tot[arg, np.arange(nb)] + node_all[:, x]
    j = int(np.argmin(cost))
    total = float(cost[j])
    if total >= _INF:
        return None
    cut = np.zeros(w, np.intp)
    for x in range(w - 1, -1, -1):
        cut[x] = j
        j = int(back[j, x])
    return band[cut], round(total, 2)


def cut_candidates(key: np.ndarray, lo: int, hi: int, target: float) -> list[tuple[np.ndarray, float]]:
    """lo〜hi 行に収まる分け目の候補（目立ち方の小さい順、同じものは 1 つ）。範囲全体と、その中の小さな帯ごとに選ぶ。"""
    lo, hi = max(0, lo), min(hi, key.shape[0] - 2)
    if hi < lo:
        return []
    prep = _prepare(key)
    found = {}
    bands = [(lo, hi)] + [(a, min(hi, a + d)) for d in range(WINDOW) for a in range(lo, hi + 1)]
    for blo, bhi in bands:
        got = _cut_in_band(*prep, blo, bhi, target)
        if got is not None:
            found.setdefault(tuple(got[0].tolist()), got)
    return sorted(found.values(), key=lambda c: (c[1], c[0].tolist()))


def euler_numbers(rgba: np.ndarray) -> dict:
    """色ごと（と絵のある所全体）の「つながりの数 − 穴の数」（8 近傍。2×2 の並びを数えるだけで求まる）。

    つながりの数は `check_rules()` で同じことを確かめるので、これも同じなら穴の数も同じ。輪のような線（輪郭の輪など）が
    1 か所でも切れると穴が 1 つ減るので、つながりの数では分からない途切れも見つかる。
    """
    key = color_keys(rgba)
    out = {}
    masks = [(int(k), key == k) for k in np.unique(key[key >= 0])] + [("opaque", key >= 0)]
    for name, m in masks:
        p = np.pad(m, 1).astype(np.int8)
        a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
        n = a + b + c + d
        q1 = int((n == 1).sum())
        q3 = int((n == 3).sum())
        qd = int(((n == 2) & (a == d)).sum())
        out[name] = (q1 - q3 - 2 * qd) // 4
    return out


def _apply(base: np.ndarray, cuts: list[np.ndarray], k: int) -> np.ndarray:
    """揺れる量 k のフレーム。下の分け目から順に、|k| 本の分け目で 1 ドットずつずらす。"""
    if k == 0:
        return base.copy()
    s = 1 if k > 0 else -1
    out = base
    for i, cut in enumerate(cuts[:abs(k)]):
        if i:   # 上の分け目のあたりは、下の分け目でまるごと i ドットずれているので、分け目も同じだけずらす
            cut = np.pad(cut, (i, i), mode="edge")[i - s * i: i - s * i + len(cut)]
        out = shift_above(out, cut, s)
    return out


def _margin(base: np.ndarray, cuts: list[np.ndarray]) -> int:
    """揺れる部分が左右の端からはみ出さないために、左右に足す透明の列の数。"""
    w = base.shape[1]
    op = base[..., 3] > 0
    p = 0
    for i in range(len(cuts)):
        rows = np.flatnonzero(op[:int(np.max(cuts[i])) + 1].any(axis=1))
        for r in rows:
            xs = np.flatnonzero(op[r])
            p = max(p, (i + 1) - int(xs[0]), int(xs[-1]) + (i + 1) - (w - 1))
    return p


def _frames(base: np.ndarray, cuts: list[np.ndarray], offsets: list[int]):
    """分け目 `cuts` で揺らしたフレームの並び（左右の余白込み）と余白の列の数。決まりを守れなければ None。"""
    p = _margin(base, cuts)
    padded = np.pad(base, ((0, 0), (p, p), (0, 0))) if p else base
    cuts = [np.pad(c, (p, p), mode="edge") for c in cuts]
    amount = len(cuts)
    # フレームに使う量に加えて、右と左のいちばん大きく揺れた形も確かめる
    shapes = {k: _apply(padded, cuts, k) for k in sorted(set(offsets) | {amount, -amount})}
    euler = euler_numbers(padded)
    if any(euler_numbers(f) != euler for f in shapes.values()):
        return None
    if check_rules(padded, list(shapes.values())):
        return None
    return [shapes[k] for k in offsets], p


def _try(base: np.ndarray, key: np.ndarray, lower, amount: int, offsets: list[int], top: int):
    """下の分け目の候補を順に試し、決まりを守れた最初のもの (cuts, costs, frames, padding) を返す。"""
    tries = 0
    for c1, cost1 in lower:
        if amount == 1:
            tries += 1
            made = _frames(base, [c1], offsets)
            if made is not None:
                return [c1], [cost1], *made
        else:
            hi = int(c1.min()) - 1
            for c2, cost2 in cut_candidates(key, top, hi, top + UPPER_TARGET * (hi - top))[:8]:
                tries += 1
                made = _frames(base, [c1, c2], offsets)
                if made is not None:
                    return [c1, c2], [cost1, cost2], *made
                if tries >= MAX_TRIES:
                    return None
        if tries >= MAX_TRIES:
            return None
    return None


def _make(base: np.ndarray, split: int | None, amount: int, offsets: list[int], rng: int = SPLIT_RANGE):
    key = color_keys(base)
    lim = body_limits(key)
    if lim is None:
        return None
    top, _, height, foot = lim
    h = key.shape[0]
    if split is None:
        lo, hi, target = top + int(round(BAND_TOP * height)), foot, top + TARGET * height
    else:
        row = h - split
        lo, hi, target = max(1, row - rng), min(foot, row + rng), row
    lo = max(lo, top + amount - 1)   # 揺れ幅 2 のときは、上にもう 1 本の分け目を置ける余地を残す
    if hi < lo:
        return None
    return _try(base, key, cut_candidates(key, lo, hi, target), amount, offsets, top)


def nearby_splits(rgba: np.ndarray, split: int, amount: int = DEFAULT_AMOUNT, count: int = 1,
                  rng: int = SPLIT_RANGE) -> list[int]:
    """`split` の上と下で、決まりを守って揺らせる最も近い高さ（下から何ドット目）を、それぞれ `count` 個まで返す（小さい順）。"""
    lim = split_limits(rgba)
    if lim is None:
        return []
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    offsets = [0, amount, -amount]
    out = []
    for step in (-1, 1):
        got = 0
        s = split + step
        while lim[0] <= s <= lim[1] and got < count:
            if _make(base, s, amount, offsets, rng) is not None:
                out.append(s)
                got += 1
            s += step
    return sorted(out)


# 作れなかったときの理由（プリセットの今の警告と、表の警告）
AUTO_FAIL = ("体の上下を分けて揺らせる場所が見つかりませんでした（背が低い、または 1 ドットの線や輪郭が"
             "横切っていて、ずらすと途切れる）")
NO_BODY_FAIL = "体の上下を分けて揺らせる場所が見つかりませんでした（絵の背が低い、または絵がありません）"


def _lines_problems(base: np.ndarray, cuts: list[np.ndarray], values: list[int]):
    """線をそのまま使うとき（表の `lines`）に、決まりを守れなかった理由 (決まりの名前, 言葉, 場所) を返す。守れていれば None。"""
    p = _margin(base, cuts)
    padded = np.pad(base, ((0, 0), (p, p), (0, 0))) if p else base
    pcuts = [np.pad(c, (p, p), mode="edge") for c in cuts]
    amount = len(cuts)
    euler = euler_numbers(padded)
    for k in sorted(set(values) | {amount, -amount}):
        f = _apply(padded, pcuts, k)
        if euler_numbers(f) != euler:
            return "euler", f"横に {k:+d} ずらすと{RULE_TEXT['euler']}", problem_pixels(padded, f, "euler")
        got = frame_problems(padded, f)
        if got:
            rules = list(dict.fromkeys(r for r, _ in got))
            return (rules[0], f"横に {k:+d} ずらすと" + "・".join(RULE_TEXT[r] for r in rules),
                    problem_pixels(padded, f, rules[0]))
    return None


def move_poses(base: np.ndarray, split: int | None, amount: int, values: list[int], rng: int = SPLIT_RANGE,
               lines=None):
    """横の位置のポーズ（動きの表の `move` の x）。(値 → フレーム, cuts, costs, 左右に足した列の数, Fail か None)。

    `values` はその列に出てくる値（重ならない並び）。`amount` は揺れ幅（値の大きさのいちばん大きいもの。1 か 2）。
    揺れ幅 2 は分け目を 2 本使う（上の分け目は自動で選ぶ）。今の横揺れと同じく、表に出てこなくても左右いちばん大きく
    動かした形まで、決まり（`check_rules()`）と色ごとのオイラー数で確かめる。`lines` を渡すと分け目を選ばずにその線
    （列ごとの「下から何ドット目」から上が動く、を `amount` 本）を使う。フレームには左右の余白が足してある。
    """
    h = base.shape[0]
    if lines is not None:
        cuts = [h - np.asarray(line, np.intp) for line in lines[:amount]]
        bad = _lines_problems(base, cuts, values)
        if bad is not None:
            rule, text, pixels = bad
            return {}, cuts, [], 0, Fail(rule, text, "書いてある線（lines）で" + text, pixels=pixels)
        out, pad = _frames(base, cuts, values) or (None, 0)
        if out is None:   # 上で確かめたので起きないはず（迷ったら作らない）
            return {}, cuts, [], 0, Fail("parts", "決まりを守れませんでした", "書いてある線では決まりを守れませんでした")
        return dict(zip(values, out)), cuts, [None] * len(cuts), pad, None
    if split is None:
        made = _make(base, None, amount, values)
        if made is None:
            return {}, [], [], 0, Fail("split_not_found", AUTO_FAIL,
                                       "自動では、決まりを守って横に動かせる高さが見つかりませんでした（背が低い、または "
                                       "1 ドットの線や輪郭が横切っていて、ずらすと途切れる）")
    else:
        lim = split_limits(base)
        if lim is None:
            return {}, [], [], 0, Fail("split_not_found", NO_BODY_FAIL,
                                       "絵の背が低い、または絵が無いので、横に動かせる高さがありません")
        if not lim[0] <= split <= lim[1]:
            from .breathe import out_of_limits
            text = out_of_limits(split, lim)
            return {}, [], [], 0, Fail("split_not_found", text, text, rows_from_bottom=[split, split])
        made = _make(base, split, amount, values, rng)
        if made is None:
            from .breathe import near_hint
            near = nearby_splits(base, split, amount, rng=rng)
            a, b = max(lim[0], split - rng), min(lim[1], split + rng)
            legacy = (f"下から {split} ドット目の近く（分け目を下から {a}〜{b} ドット目に収める）では、決まりを守って揺らせませんでした"
                      f"（ずらすと 1 ドットの線や輪郭が途切れる、または部品のつながりが変わる）。{near_hint(near)}")
            text = (f"下から {a}〜{b} ドット目では、決まりを守って横に動かせません"
                    f"（ずらすと 1 ドットの線や輪郭が途切れる、または部品のつながりが変わる）。{near_hint(near)}")
            return {}, [], [], 0, Fail("split_not_found", legacy, text, rows_from_bottom=[a, b], near=near)
    cuts, costs, out, pad = made
    return dict(zip(values, out)), cuts, costs, pad, None


def sway_table(frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
               amount: int = DEFAULT_AMOUNT, split: int | str | None = None) -> dict:
    """横揺れを動きの表で書いたもの。範囲 `upper`（分け目）と、横の位置の列 `upper_x`（+ が右）。

    例: 4 フレーム → 0, +1, 0, −1、8 フレーム・揺れ幅 2 → 0, 1, 2, 1, 0, −1, −2, −1。値が 0 の行には書かない。
    """
    from .base import is_auto
    from .spec import FORMAT, SPEC_VERSION
    durations = check_durations(durations, frames)
    region = {"type": "split", "at": AUTO if is_auto(split) else int(split)}
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "regions": {"upper": region},
            "tracks": {"upper_x": {"type": "move", "region": "upper", "axis": "x"}},
            "frames": [{"ms": d, **({"upper_x": k} if k else {})}
                       for d, k in zip(durations, sway_offsets(frames, amount))]}


def sway(rgba: np.ndarray, frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
         amount: int = DEFAULT_AMOUNT, split: int | str | None = None) -> MotionResult:
    """横揺れのフレームを作る。1 枚目は元のスプライトのまま（左右に余白を足したときは、それを足しただけのもの）。

    `durations` は全フレーム共通の表示時間（ミリ秒）か、フレームごとの並び。`amount` は揺れ幅（1〜2 ドット）。
    `split` は揺らす高さ: None / "auto" なら自動、数なら画像の下から何ドット目のあたりから上を揺らすか（1 = いちばん下の行。
    画像の外ならエラー）。決まりを守って揺らせなければ `skipped` にして警告を返す（指定した高さなら、近くで揺らせる高さも知らせる）。

    中身は「表を埋めるプリセット」: `sway_table()` で表を埋め、表の仕組み（`timeline.py`）で作る。警告の言葉と `info` は前と同じ。
    """
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    check_frames(frames)
    if frames < MIN_FRAMES:
        raise ValueError(f"横揺れのフレーム数は {MIN_FRAMES}〜{MAX_FRAMES} で指定してください（2 では揺れません）: {frames}")
    if isinstance(amount, (bool, np.bool_)) or not isinstance(amount, (int, np.integer)) or not 1 <= amount <= MAX_AMOUNT:
        raise ValueError(f"揺れ幅は 1〜{MAX_AMOUNT} ドットで指定してください: {amount}")
    amount = int(amount)
    durations = check_durations(durations, frames)
    split = check_split(split, rgba.shape[0])
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    offsets = sway_offsets(frames, amount)
    h = base.shape[0]
    spec = parse_spec(sway_table(frames, durations, amount, split))

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"横揺れを付けませんでした: {reason}"], True,
                            {"offsets": offsets, "split": AUTO if split is None else split, "padding": 0},
                            spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, NAME), planned_ms=list(durations))

    built = build(base, spec)
    if built.problems:   # 横の位置は分け目を選ぶときに決まりを確かめるので、問題は「作れなかった」だけ
        return skip(built.problems[0].fail.legacy)
    track = built.regions["upper"]["tracks"]["upper_x"]
    cuts, costs, pad = track["lines"], track["costs"], track["padding"]
    return MotionResult(NAME, built.frames, durations, [], False,
                        {"offsets": offsets, "cuts": [c.tolist() for c in cuts], "costs": costs, "padding": pad,
                         "split": AUTO if split is None else split,
                         "split_from_bottom": [[int(h - c.max()), int(h - c.min())] for c in cuts]},
                        spec=spec_to_dict(spec, built.used))
