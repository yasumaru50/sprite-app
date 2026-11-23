"""呼吸: 体の上の部分がゆっくり 1 ドット下がって戻る、控えめな動き（足元は動かさない）。

方法（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」）:
左端から右端まで、各列から 1 ドットずつを選んでつないだ横向きの「継ぎ目」（シーム）を 1 本決め、各列で継ぎ目のドットを抜いて、
その上のドットを 1 ドット下げる（いちばん上には透明が入る）。継ぎ目より下は 1 ドットも変わらないので、足元は動かず、
抜いた分だけ上が下がるので隙間もできない。色はもとのドットを動かすだけなので新しい色は出ない。

継ぎ目は、次の「目立ち方」の合計が最も小さくなるものを動的計画法で選ぶ:
- 抜くドットが、縦に同じ色が続く部分（3 ドット以上）の 1 つなら 0（同じ色の部分が 1 ドット短くなるだけ）。
  2 ドットの部分（目など）なら少し目立つ（`RUN2_COST`）。1 ドットしかない部分（横向きの輪郭の線、1 ドットの目・口）は抜かない
- 隣の列と継ぎ目の高さが違うと、その間の行では片方の列だけがずれる。両方の列で色が変わる行（線や境目の位置）だと
  段差ができるので、その行の数だけ目立つ（片方の列が同じ色の続きなら、ずれても見た目は変わらない）
- 腰のあたり（上から 7 割の高さ）から離れるほど少しずつ目立つ（`BIAS`）。上下を分ける位置を体の中ほどより下にする

継ぎ目を探す範囲は、絵の上から 35% より下で、下の端（足元）から 12%（2 ドット以上）より上。
抜けるドットの無い形（1 ドットの線が横切っているなど）や、動かしたあとに部品・輪郭のつながりが変わる形には呼吸を付けず、
警告を出す（迷ったら動かさない）。

分ける高さの指定（`split`）: 省くか "auto" なら上のとおり自動で選ぶ。数で指定すると「画像の下から何ドット目のあたりで分けるか」
（1 = いちばん下の行）として扱い、継ぎ目をその高さの上下 `SPLIT_RANGE` ドット以内に限って選ぶ（腰の目安の代わりに、指定した高さから
離れるほど目立つ）。決まり（3 ドット以上続く所からだけ抜く・新しい色を作らない・足元を動かさない・隙間を作らない・輪郭を途切れさせない）は
指定があっても変えない。その範囲で決まりを守った継ぎ目が作れなければ、ほかの高さに勝手に移さず、呼吸を付けずに警告を出し、
近くで分けられる高さを知らせる（理由は docs/sprite-app-plan.md の「フェーズ 2 の方針」）。画像の外の高さはエラー（`ValueError`）。

伸ばす（動きの表の縦の変形の +。`stretch_poses()`）も、ここの継ぎ目の選び方をそのまま使う。抜く代わりに継ぎ目のドットを複製し、
そこから上を 1 ドット上げる（`raise_above_seam()`）。使ってよい所・目立ち方・決まりは縮めると同じ。
小さな部品を守る（`keep=True`、表の `keep_small_parts`）と、同じ色の小さな塊（縦 `SMALL_PART_HEIGHT` 以下・面積 `SMALL_PART_AREA` 以下）を
継ぎ目が通らず、作ったあとも塊が形のまま（全部のドットが同じ量だけ縦に動き、大きさも同じ）であることを確かめる（`parts_kept()`）。
"""

from __future__ import annotations

import math

import numpy as np

from .base import (AUTO, BAND_FOOT, MAX_FRAMES, MIN_HEIGHT, Fail, MotionResult, check_durations, check_frames,
                   check_rules, check_split, color_keys, split_limits)
from .base import body_limits as _limits
from .base import is_auto as _is_auto

NAME = "breathe"
LABEL = "呼吸（体の上の部分が 1 ドット下がって戻る）"
DEFAULT_FRAMES = 2
DEFAULT_DURATION = 500   # 1 フレームの表示時間（ミリ秒）。2 フレームで 1 秒の呼吸
DEFAULT_DEPTH = 1        # 下がる量（ドット）
MAX_DEPTH = 2

BAND_TOP = 0.35          # 継ぎ目を探す範囲の上の端（絵の高さに対する割合）
TARGET = 0.7             # 上下を分けたい高さ（腰のあたり）
BIAS = 0.05              # TARGET から 1 行離れるごとの目立ち方
RUN2_COST = 2.0          # 縦に 2 ドットの部分を 1 ドットにする目立ち方
SPLIT_RANGE = 2          # 高さを指定したとき、継ぎ目が指定した高さから離れてよい量（ドット）
_INF = 1e9

# 縦の変形の向き（動きの表の deform の −・+）
SHRINK = "shrink"        # 継ぎ目のドットを抜いて上を下げる（−）
STRETCH = "stretch"      # 継ぎ目のドットを複製して上を上げる（+）

# 小さな部品を守る（keep_small_parts。既定は切）: 同じ色が上下左右につながった塊で、縦が SMALL_PART_HEIGHT ドット以下・
# 面積が SMALL_PART_AREA ドット以下のもの（目・頬・口など）は、継ぎ目を通さず、形のまま動くか止まるかにする。
# 縦 3 ドットまでにした理由: ai04 の頬の赤みは縦 3 ドット（横 3）で、縦 2 以下だけでは守れないため（docs/sprite-app-plan.md）
SMALL_PART_HEIGHT = 3
SMALL_PART_AREA = 24     # 横揺れの「小さな部品」（PART_MAX）と同じ


def breathe_offsets(frames: int = DEFAULT_FRAMES, depth: int = DEFAULT_DEPTH) -> list[int]:
    """各フレームで下がる量（ドット）。ゆっくりした波（1 − cos）を整数に丸める（ちょうど半分のときは下げない）。

    例: 2 フレーム → [0, 1]、4 フレーム・深さ 2 → [0, 1, 2, 1]、8 フレーム・深さ 1 → [0, 0, 0, 1, 1, 1, 0, 0]。
    """
    out = []
    for i in range(frames):
        v = depth * (1 - math.cos(2 * math.pi * i / frames)) / 2
        out.append(int(math.ceil(v - 0.5 - 1e-9)))
    return out


def _runs(key: np.ndarray) -> np.ndarray:
    """各ドットが入っている、縦に同じ色が続く部分の長さ。"""
    h = key.shape[0]
    up = np.ones(key.shape, np.int32)
    down = np.ones(key.shape, np.int32)
    for y in range(1, h):
        same = key[y] == key[y - 1]
        up[y] = np.where(same, up[y - 1] + 1, 1)
    for y in range(h - 2, -1, -1):
        same = key[y] == key[y + 1]
        down[y] = np.where(same, down[y + 1] + 1, 1)
    return up + down - 1


def _seam_in_band(key: np.ndarray, run: np.ndarray, pref_all: np.ndarray, lo: int, hi: int,
                  target: float, block: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """lo〜hi 行の中で、目立ち方の合計が最も小さい継ぎ目（動的計画法）。抜ける継ぎ目が無ければ None。

    `block`（真のドットは通らない。小さな部品を守るとき）を渡すと、そのドットも 1 ドットの所と同じく使わない。
    """
    w = key.shape[1]
    band = np.arange(lo, hi + 1)
    run = run[lo:hi + 1]                                            # (B, W)
    node = np.where(run == 1, _INF, np.where(run == 2, RUN2_COST, 0.0))
    if block is not None:
        node = np.where(block[lo:hi + 1], _INF, node)
    node = node + BIAS * np.abs(band - target)[:, None]
    pref = pref_all[lo:hi + 1]                                      # (B, W-1)

    cost = node[:, 0].copy()
    back = np.zeros((len(band), w), np.intp)
    for x in range(1, w):
        p = pref[:, x - 1]
        trans = cost[:, None] + np.abs(p[None, :] - p[:, None])     # [前の行, 今の行]
        arg = np.argmin(trans, axis=0)
        back[:, x] = arg
        cost = trans[arg, np.arange(len(band))] + node[:, x]
    j = int(np.argmin(cost))
    total = float(cost[j])
    if total >= _INF:
        return None
    seam = np.zeros(w, np.intp)
    for x in range(w - 1, -1, -1):
        seam[x] = j
        j = int(back[j, x])
    return band[seam], total


def _prepare(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各ドットの縦の同じ色の長さと、隣り合う 2 列で両方の色が変わる行の数の累積（段差の目立ち方の計算用）。"""
    h, w = key.shape
    vary = np.zeros((h, w), bool)
    vary[1:] = key[1:] != key[:-1]
    both = (vary[:, :-1] & vary[:, 1:]).astype(np.float64)          # (H, W-1)
    return _runs(key), np.cumsum(both, axis=0)


def find_seam(rgba: np.ndarray, split: int | str | None = None,
              block: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """抜く継ぎ目（列ごとの行の番号）と、その目立ち方。探す範囲に抜ける継ぎ目が無ければ None。

    `split` を数で指定したときは、その高さの近くの継ぎ目のうち、目立ち方が最も小さいもの（決まりを守るかは確かめない。
    確かめながら選ぶのは `seam_candidates()` と `breathe()`）。
    """
    if not _is_auto(split):
        cands = seam_candidates(rgba, split, block=block)
        return cands[0] if cands else None
    key = color_keys(rgba)
    band = _auto_band(key)
    if band is None:
        return None
    lo, hi, target = band
    run, pref = _prepare(key)
    return _seam_in_band(key, run, pref, lo, hi, target, block)


def _auto_band(key: np.ndarray) -> tuple[int, int, float] | None:
    """自動で継ぎ目を探す範囲 (上の行, 下の行, 腰の目安の行)。探せなければ None。"""
    lim = _limits(key)
    if lim is None:
        return None
    top, bottom, height, foot = lim
    lo = max(1, top + int(round(BAND_TOP * height)))
    hi = foot
    if hi < lo:
        return None
    return lo, hi, top + TARGET * height


def auto_target(rgba: np.ndarray) -> int | None:
    """自動の腰の目安の高さ（下から何ドット目）。作れなかったときに近くの高さを探す起点。"""
    band = _auto_band(color_keys(rgba))
    if band is None:
        return None
    return int(rgba.shape[0] - int(round(band[2])))


def auto_candidates(rgba: np.ndarray, block: np.ndarray | None = None) -> list[tuple[np.ndarray, float]]:
    """自動の範囲の継ぎ目の候補（小さな部品を守るときに順に試す）。範囲全体で最も目立たないものを先に、
    次に範囲の中の高さ 1〜3 行の小さな帯ごとに選んだものを目立ち方の小さい順に（同じものは 1 つ）。"""
    key = color_keys(rgba)
    band = _auto_band(key)
    if band is None:
        return []
    lo_all, hi_all, target = band
    run, pref = _prepare(key)
    first = _seam_in_band(key, run, pref, lo_all, hi_all, target, block)
    found = {}
    for lo in range(lo_all, hi_all + 1):
        for hi in range(lo, min(hi_all, lo + 2) + 1):
            got = _seam_in_band(key, run, pref, lo, hi, target, block)
            if got is not None:
                found.setdefault(tuple(got[0].tolist()), got)
    rest = sorted(found.values(), key=lambda c: (c[1], c[0].tolist()))
    if first is None:
        return rest
    return [first] + [c for c in rest if not np.array_equal(c[0], first[0])]


def seam_candidates(rgba: np.ndarray, split: int, rng: int = SPLIT_RANGE,
                    block: np.ndarray | None = None) -> list[tuple[np.ndarray, float]]:
    """下から `split` ドット目の上下 `rng`（既定 `SPLIT_RANGE`）ドット以内に収まる継ぎ目の候補（目立ち方の小さい順、同じものは 1 つ）。

    範囲全体で最も目立たないもの 1 本だけだと、それが部品のつながりを変える形のときに、少し目立つが決まりを守る別の継ぎ目を
    見落とすので、範囲の中の小さな帯（上下 0〜2 ドットの帯すべて）ごとにも選ぶ。目立ち方は、どれも指定した高さからの離れ方で測る。
    指定できる範囲（`split_limits()`）の外なら空。
    """
    key = color_keys(rgba)
    h = key.shape[0]
    split = check_split(split, h)
    lim = split_limits(rgba)
    if lim is None or not lim[0] <= split <= lim[1]:
        return []
    _, _, _, foot = _limits(key)
    row = h - split                                                 # 指定した高さの行（上から）
    lo_all, hi_all = max(1, row - rng), min(foot, row + rng)
    run, pref = _prepare(key)
    found = {}
    for lo in range(lo_all, hi_all + 1):
        for hi in range(lo, hi_all + 1):
            got = _seam_in_band(key, run, pref, lo, hi, row, block)
            if got is not None:
                found.setdefault(tuple(got[0].tolist()), got)
    return sorted(found.values(), key=lambda c: (c[1], c[0].tolist()))


def lower_above_seam(rgba: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """各列で継ぎ目のドットを抜き、その上を 1 ドット下げる（いちばん上は透明。継ぎ目より下は変えない）。"""
    h, w = rgba.shape[:2]
    r = np.arange(h)[:, None]
    src = np.where(r <= seam[None, :], r - 1, r)                    # 下げる部分は 1 つ上の行から持ってくる
    out = np.take_along_axis(rgba, np.clip(src, 0, h - 1)[..., None], axis=0).copy()
    out[src < 0] = 0
    return out


def raise_above_seam(rgba: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """各列で継ぎ目のドットを複製し、そこから上を 1 ドット上げる（伸ばす。継ぎ目より下は変えない）。

    いちばん上の行のドットは外に出るので、呼ぶ側で上に透明の行を足しておく（`timeline.py`）。
    """
    h = rgba.shape[0]
    r = np.arange(h)[:, None]
    src = np.where(r < seam[None, :], r + 1, r)                     # 上げる部分は 1 つ下の行から持ってくる
    return np.take_along_axis(rgba, np.clip(src, 0, h - 1)[..., None], axis=0).copy()


def check_keep(keep) -> bool:
    """小さな部品を守るかの指定（真か偽）。"""
    if not isinstance(keep, (bool, np.bool_)):
        raise ValueError(f"小さな部品を守るかは true か false で指定してください: {keep!r}")
    return bool(keep)


def _apply(rgba: np.ndarray, seam: np.ndarray, op: str) -> np.ndarray:
    return raise_above_seam(rgba, seam) if op == STRETCH else lower_above_seam(rgba, seam)


# ---- 小さな部品を守る ----

def small_parts(rgba: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """小さな部品（同じ色が上下左右につながった塊で、縦 SMALL_PART_HEIGHT 以下・面積 SMALL_PART_AREA 以下）の
    (色の番号, 行の並び, 列の並び) の並び。透明は数えない。"""
    key = color_keys(rgba)
    h, w = key.shape
    k = key.tolist()
    seen = [[False] * w for _ in range(h)]
    out = []
    for y0 in range(h):
        for x0 in range(w):
            c = k[y0][x0]
            if c < 0 or seen[y0][x0]:
                continue
            seen[y0][x0] = True
            stack, ys, xs = [(y0, x0)], [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy][xx] and k[yy][xx] == c:
                        seen[yy][xx] = True
                        stack.append((yy, xx))
            if len(ys) <= SMALL_PART_AREA and max(ys) - min(ys) + 1 <= SMALL_PART_HEIGHT:
                out.append((int(c), np.array(ys, np.intp), np.array(xs, np.intp)))
    return out


def small_part_mask(rgba: np.ndarray) -> np.ndarray:
    """小さな部品のドット（継ぎ目を通さない所）。"""
    mask = np.zeros(rgba.shape[:2], bool)
    for _, ys, xs in small_parts(rgba):
        mask[ys, xs] = True
    return mask


def parts_kept(base: np.ndarray, pose: np.ndarray, parts=None, reach: int = MAX_DEPTH) -> bool:
    """元の絵の小さな部品が、どれも形のまま（全部のドットが同じ量だけ縦に動いて）ポーズにあるか。

    継ぎ目が部品を通らなくても、部品の列ごとに継ぎ目の上と下に分かれると部品がずれて形が変わるので、それも見つける。
    """
    parts = small_parts(base) if parts is None else parts
    kp = color_keys(pose)
    h = kp.shape[0]
    for c, ys, xs in parts:
        ok = False
        for dy in range(-reach, reach + 1):
            yy = ys + dy
            if (yy.min() >= 0 and yy.max() < h and bool((kp[yy, xs] == c).all())
                    and _piece_size(kp, int(yy[0]), int(xs[0]), len(ys) + 1) == len(ys)):
                ok = True                   # 同じ形が、伸びたり（同じ色が足されたり）せずにある
                break
        if not ok:
            return False
    return True


def _piece_size(key: np.ndarray, y0: int, x0: int, limit: int) -> int:
    """(y0, x0) から上下左右につながった同じ色の塊の大きさ（limit で数えるのをやめる）。"""
    h, w = key.shape
    c = key[y0, x0]
    seen = {(y0, x0)}
    stack = [(y0, x0)]
    while stack and len(seen) < limit:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen and key[yy, xx] == c:
                seen.add((yy, xx))
                stack.append((yy, xx))
    return len(seen)


def _block(pose: np.ndarray, keep: bool) -> np.ndarray | None:
    return small_part_mask(pose) if keep else None


def _steps_auto(base: np.ndarray, n: int, op: str = SHRINK, keep: bool = False):
    """自動の継ぎ目で n 回縮めた（伸ばした）絵の並び (poses, seams, costs, 作れなかった段（1 から）か None)。

    小さな部品を守らないとき（既定）は、自動の継ぎ目を 1 本だけ使う（決まりの確かめは呼ぶ側）。守るときは、部品を通らない
    候補を順に試し、決まりと部品の形を守った最初のものを使う。
    """
    poses, seams, costs = [base], [], []
    parts = small_parts(base) if keep else None
    for j in range(n):
        if keep:
            chosen = None
            for seam, cost in auto_candidates(poses[-1], _block(poses[-1], True)):
                pose = _apply(poses[-1], seam, op)
                if not check_rules(base, [pose]) and parts_kept(base, pose, parts):
                    chosen = (pose, seam, cost)
                    break
            if chosen is None:
                return poses, seams, costs, j + 1
            pose, seam, cost = chosen
        else:
            found = find_seam(poses[-1])
            if found is None:
                return poses, seams, costs, j + 1
            seam, cost = found
            pose = _apply(poses[-1], seam, op)
        poses.append(pose)
        seams.append(seam)
        costs.append(round(cost, 2))
    return poses, seams, costs, None


def _steps_split(base: np.ndarray, split: int, n: int, rng: int = SPLIT_RANGE, op: str = SHRINK, keep: bool = False):
    """指定した高さの近くで n 回縮めた（伸ばした）絵の並び (poses, seams, costs, 作れなかった段（1 から）か None)。

    どの段でも決まりを守る継ぎ目のうち最も目立たないものを選ぶ（小さな部品を守るときは、部品の形も守るもの）。
    """
    poses, seams, costs = [base], [], []
    parts = small_parts(base) if keep else None
    for j in range(n):
        for seam, cost in seam_candidates(poses[-1], split, rng, _block(poses[-1], keep)):
            pose = _apply(poses[-1], seam, op)
            if not check_rules(base, [pose]) and (not keep or parts_kept(base, pose, parts)):
                break
        else:
            return poses, seams, costs, j + 1
        poses.append(pose)
        seams.append(seam)
        costs.append(round(cost, 2))
    return poses, seams, costs, None


def _poses_auto(base: np.ndarray, n: int, op: str = SHRINK, keep: bool = False):
    poses, seams, costs, failed = _steps_auto(base, n, op, keep)
    return None if failed else (poses, seams, costs)


def _poses_split(base: np.ndarray, split: int, n: int, rng: int = SPLIT_RANGE, op: str = SHRINK, keep: bool = False):
    """指定した高さの近くで n 回下げた（上げた）絵の並び。どの段でも決まりを守る継ぎ目のうち最も目立たないものを選ぶ。作れなければ None。"""
    poses, seams, costs, failed = _steps_split(base, split, n, rng, op, keep)
    return None if failed else (poses, seams, costs)


# 作れなかったときの理由（プリセットの今の警告と、表の警告）
AUTO_FAIL = "体の上下を分けられる場所が見つかりませんでした（背が低い、または横向きの 1 ドットの線などが横切っている）"
NO_BODY_FAIL = "体の上下を分けられる場所が見つかりませんでした（絵の背が低い、または絵がありません）"


def out_of_limits(split: int, lim: tuple[int, int]) -> str:
    """指定できる範囲の外の高さの理由（呼吸と横揺れで同じ言葉）。"""
    return (f"下から {split} ドット目では分けられません（足元の {lim[0] - 1} 行は動かさないため、"
            f"この絵で指定できるのは {lim[0]}〜{lim[1]}）")


def near_hint(near: list[int]) -> str:
    return ("近くで分けられる高さ: " + "、".join(str(v) for v in near)) if near else "この絵では分けられる高さがありません"


def shrink_poses(base: np.ndarray, split: int | None, n: int, rng: int = SPLIT_RANGE, lines=None, keep: bool = False):
    """縦に縮めるポーズ（動きの表の `deform` の −）。(poses, seams, costs, Fail か None)。

    `poses[k]` は k ドット縮めた（上を k ドット下げた）絵（`poses[0]` は元の絵）。`split` は None（自動）か下から何ドット目か
    （`check_split()` を通したもの）、`rng` は継ぎ目が指定した高さから離れてよい量。`lines` を渡すと継ぎ目を選ばずにその線
    （列ごとの「下から何ドット目」の並びを n 本。2 本目は 1 本目で下げたあとの絵の上の高さ）をそのまま使う
    （決まりの確かめは呼ぶ側の `timeline.py`）。作れなかった段より前のポーズは `poses` に入っている。
    `keep` は小さな部品を守る（表の `keep_small_parts`）。
    """
    return deform_poses(base, split, n, rng, lines, SHRINK, keep)


def stretch_poses(base: np.ndarray, split: int | None, n: int, rng: int = SPLIT_RANGE, lines=None, keep: bool = False,
                  limits: tuple[int, int] | None = None):
    """縦に伸ばすポーズ（動きの表の `deform` の +）。`base` は上に n 行以上の透明の行がある絵（`timeline.py` が足す）。
    返す形と引数は `shrink_poses()` と同じ（`poses[k]` は k ドット伸ばした絵）。`limits` は指定できる高さの範囲
    （余白を足す前の絵の `split_limits()`。縮めると同じ範囲・同じ言葉にするため）。"""
    return deform_poses(base, split, n, rng, lines, STRETCH, keep, limits)


def deform_poses(base: np.ndarray, split: int | None, n: int, rng: int = SPLIT_RANGE, lines=None, op: str = SHRINK,
                 keep: bool = False, limits: tuple[int, int] | None = None):
    """縮める・伸ばすの共通の中身（`shrink_poses()`・`stretch_poses()`）。"""
    h = base.shape[0]
    if lines is not None:
        poses, seams = [base], []
        for line in lines[:n]:
            seam = h - np.asarray(line, np.intp)
            poses.append(_apply(poses[-1], seam, op))
            seams.append(seam)
        return poses, seams, [None] * len(seams), None
    verb = "伸ばせ" if op == STRETCH else "縮められ"
    hint_more = op == STRETCH or keep    # 縮める（既定）の自動の警告は前の版と同じ言葉のまま
    if split is None:
        poses, seams, costs, failed = _steps_auto(base, n, op, keep)
        if failed is None:
            return poses, seams, costs, None
        near: list[int] = []
        if hint_more:
            t = auto_target(base)
            lim = split_limits(base)
            if t is not None and lim is not None:
                t = min(max(t, lim[0]), lim[1])
                near = sorted(set(([t] if _poses_split(base, t, n, rng, op, keep) is not None else [])
                                  + nearby_splits(base, t, n, rng=rng, op=op, keep=keep)))
        what = ("小さな部品を守ったまま" if keep else "") + f"決まりを守って{verb}る高さが見つかりませんでした"
        text = (f"自動では、{what}（背が低い、または横向きの 1 ドットの線などが横切っている）"
                + (f"（{failed} 本目の継ぎ目）" if failed > 1 else "") + (f"。{near_hint(near)}" if hint_more else ""))
        legacy = AUTO_FAIL if not hint_more else text
        return poses, seams, costs, Fail("split_not_found", legacy, text, depth=failed, near=near)
    lim = split_limits(base) if limits is None else limits
    if lim is None:
        return [base], [], [], Fail("split_not_found", NO_BODY_FAIL,
                                    f"絵の背が低い、または絵が無いので、{verb}る高さがありません")
    if not lim[0] <= split <= lim[1]:
        text = out_of_limits(split, lim)
        return [base], [], [], Fail("split_not_found", text, text, rows_from_bottom=[split, split])
    poses, seams, costs, failed = _steps_split(base, split, n, rng, op, keep)
    if failed is None:
        return poses, seams, costs, None
    near = nearby_splits(base, split, n, rng=rng, op=op, keep=keep)
    a, b = max(lim[0], split - rng), min(lim[1], split + rng)
    why = "（1 ドットの線が横切っている、または分けると部品か輪郭のつながりが変わる" + ("、または小さな部品の形が変わる" if keep else "") + "）"
    legacy = (f"下から {split} ドット目の近く（継ぎ目を下から {a}〜{b} ドット目に収める）では、決まりを守って上下を分けられませんでした"
              f"{why}。{near_hint(near)}")
    text = (f"下から {a}〜{b} ドット目では、決まりを守って{verb}ません" + why
            + (f"（{failed} 本目の継ぎ目）" if failed > 1 else "") + f"。{near_hint(near)}")
    return poses, seams, costs, Fail("split_not_found", legacy, text, depth=failed, rows_from_bottom=[a, b], near=near)


def nearby_splits(rgba: np.ndarray, split: int, depth: int = DEFAULT_DEPTH, count: int = 1,
                  rng: int = SPLIT_RANGE, op: str = SHRINK, keep: bool = False) -> list[int]:
    """`split` の上と下で、決まりを守って分けられる最も近い高さ（下から何ドット目）を、それぞれ `count` 個まで返す（小さい順）。"""
    lim = split_limits(rgba)
    if lim is None:
        return []
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    out = []
    for step in (-1, 1):
        got = 0
        s = split + step
        while lim[0] <= s <= lim[1] and got < count:
            if _poses_split(base, s, depth, rng, op, keep) is not None:
                out.append(s)
                got += 1
            s += step
    return sorted(out)


def breathe_table(frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
                  depth: int = DEFAULT_DEPTH, split: int | str | None = None, keep_small_parts: bool = False) -> dict:
    """呼吸を動きの表（sprite-motion の JSON にする辞書）で書いたもの。範囲 `upper`（分け目）と、縦の変形の列 `upper_y`。

    各行の `upper_y` は −（下がる量）。例: 2 フレーム → 0, −1、4 フレーム・深さ 2 → 0, −1, −2, −1。値が 0 の行には書かない。
    `keep_small_parts` を真にすると、範囲に `"keep_small_parts": true` を書く（切のときは書かない。前の版と同じ表）。
    """
    from .spec import FORMAT, SPEC_VERSION
    durations = check_durations(durations, frames)
    region = {"type": "split", "at": AUTO if _is_auto(split) else int(split)}
    if keep_small_parts:
        region["keep_small_parts"] = True
    return {"format": FORMAT, "version": SPEC_VERSION, "name": NAME, "loop": True,
            "regions": {"upper": region},
            "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
            "frames": [{"ms": d, **({"upper_y": -k} if k else {})}
                       for d, k in zip(durations, breathe_offsets(frames, depth))]}


def breathe(rgba: np.ndarray, frames: int = DEFAULT_FRAMES, durations: int | list[int] = DEFAULT_DURATION,
            depth: int = DEFAULT_DEPTH, split: int | str | None = None,
            keep_small_parts: bool = False) -> MotionResult:
    """呼吸のフレームを作る。1 枚目は元のスプライトのまま。

    `durations` は全フレーム共通の表示時間（ミリ秒）か、フレームごとの並び。付けられない形なら `skipped` にして警告を返す。
    `split` は上下を分ける高さ: None / "auto" なら自動、数なら画像の下から何ドット目のあたりか（1 = いちばん下の行。
    画像の外ならエラー）。指定した高さの近くで決まりを守って分けられなければ、呼吸を付けずに警告し、近くで分けられる高さを知らせる。

    中身は「表を埋めるプリセット」: `breathe_table()` で表を埋め、表の仕組み（`timeline.py`）で作る。警告の言葉と `info` は
    表の仕組みを入れる前と同じ。`spec` に埋めた表（作れたときは実際に使った高さ `"used"` つき）を入れる。
    """
    from .spec import parse_spec, spec_to_dict
    from .timeline import build
    check_frames(frames)
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"下がる量は 1〜{MAX_DEPTH} ドットで指定してください: {depth}")
    durations = check_durations(durations, frames)
    split = check_split(split, rgba.shape[0])
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    offsets = breathe_offsets(frames, depth)
    h = base.shape[0]
    spec = parse_spec(breathe_table(frames, durations, depth, split, check_keep(keep_small_parts)))

    def skip(reason: str) -> MotionResult:
        from .report import problem_entries
        return MotionResult(NAME, [base.copy()], [sum(durations)], [f"呼吸を付けませんでした: {reason}"], True,
                            {"offsets": offsets, "split": AUTO if split is None else split}, spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, NAME), planned_ms=list(durations))

    built = build(base, spec)
    if built.problems:
        fail = next((q.fail for q in built.problems if q.fail is not None), None)
        if fail is not None:
            return skip(fail.legacy)
        out = [base if k == 0 else built.raw_poses[("upper_y", -k)] for k in offsets]
        return skip("動かすと形が崩れるため（" + "、".join(check_rules(base, out)) + "）")
    track = built.regions["upper"]["tracks"]["upper_y"]
    seams, costs = track["lines"], track["costs"]
    return MotionResult(NAME, built.frames, durations, [], False,
                        {"offsets": offsets, "seams": [s.tolist() for s in seams], "costs": costs,
                         "split": AUTO if split is None else split,
                         "seam_from_bottom": [[int(h - s.max()), int(h - s.min())] for s in seams]},
                        spec=spec_to_dict(spec, built.used))
