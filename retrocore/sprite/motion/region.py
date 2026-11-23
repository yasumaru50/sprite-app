"""動きの表の範囲「帯」（`"type": "band"`）と「四角」（`"type": "rect"`）。

形は docs/sprite-app-plan.md の「動きの表（タイムライン）の設計」の 1. と「帯・四角の範囲」。分け目（`"type": "split"`）は
「分け目から上が全部動く」が、帯と四角は**囲んだ中だけ**が動き、外は 1 ドットも動かさない:

    "regions": {
      "head":  {"type": "rect", "left": 7, "right": 34, "bottom": 17, "top": 44},
      "chest": {"type": "band", "bottom": 10, "top": 16}
    }

- 高さ（`bottom`・`top`）は分け目の `at` と同じく、画像の下から何ドット目（1 = いちばん下の行）か、キャラの高さに対する割合（`"40%"`）。
  横（`left`・`right`）は画像の左から何ドット目（1 = いちばん左の列）か、キャラの幅（透明を除いた幅）に対する、キャラの左の端からの割合。
  どれも端の行・列を含む。帯は横いっぱい（`left`・`right` を書かない）
- 範囲は足元（下から 12%、2 ドット以上）を含められない（分け目と同じ `split_limits()`）。含めると作らずに警告する

動かし方（列の種類）ごとの中身:
- **縦の変形**（`deform`）: 範囲の下の端から `range`（省くと 2）ドット以内に継ぎ目を選び（呼吸の継ぎ目と同じ選び方。列は範囲の中だけ）、
  縮める（−）なら継ぎ目のドットを抜いて範囲の中の継ぎ目より上を 1 ドット下げ、伸ばす（+）なら継ぎ目のドットを複製して 1 ドット上げる。
  縮めると範囲のいちばん上の行が空き（下の「空いた所」）、伸ばすと範囲のいちばん上の行が範囲の外に 1 行出る（下の「重なった所」）
- **位置**（`move`）: 範囲の中身を形のまま横（`axis: "x"`、+ が右）か縦（`axis: "y"`、+ が下。体全体のずれ `root` と同じ向き）にずらす。
  1 ドットずつずらし、範囲も中身と一緒に動く。帯・四角でだけ縦（`y`）に使える

範囲の境目の扱い（設計の 5.「隠れていた所・重なり」。作らない寄り）:
- **空いた所**（動いた後ろから見えてくる所）: 範囲の外のすぐ隣（動いた向きの反対側）が透明か、空いた所にもともとあったドットが透明なら透明のまま。
  外の隣の色ともともとのドットの色が同じなら、その色（同じ色の部分が 1 ドット長くなるだけ）。外の隣から反対向きに同じ色が 3 ドット以上
  続くなら、その色の続きで埋める。**それ以外は作らずに警告**（`hidden`。後ろに何の色があったか分からない）
- **重なった所**（範囲の外に出た中身が、外のドットの上に来る所）: 出た中身が透明なら外のドットのまま。外のドットが透明か同じ色なら上に描く。
  **違う色のドットを隠すときは作らずに警告**（`overlap`）
- 作ったポーズは今までと同じ決まり（`frame_problems()`。新しい色・大きさ・足元・部品と輪郭のつながり・縁の色）に加えて、
  横揺れと同じ色ごとのオイラー数（輪郭の輪が切れていない・隙間が無い）で確かめる。範囲の境目で輪郭が途切れると、ここで見つかる
- 作れなければ、範囲の上下左右の端を 1〜3 ドットずらして実際に作って確かめ、作れた範囲を「近くで作れる範囲」として添える
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import RULE_TEXT, Fail, color_keys, frame_problems, problem_pixels, rule_reference, split_limits
from .breathe import RUN2_COST, SHRINK, STRETCH, _prepare, _seam_in_band, parts_kept, small_part_mask, small_parts
from .sway import euler_numbers

BOX_TYPES = ("band", "rect")
EDGES = {"band": ("bottom", "top"), "rect": ("left", "right", "bottom", "top")}
EDGE_LABEL = {"left": "左", "right": "右", "bottom": "下", "top": "上"}
NEAR_STEPS = (1, -1, 2, -2, 3, -3)   # 作れないときに端をずらして試す量（ドット）
NEAR_COUNT = 3                       # 添える「近くで作れる範囲」の数
NEAR_TRIES = 24                      # 試す数の上限（作れないときに時間がかかりすぎないように）


@dataclass(frozen=True)
class Box:
    """範囲を絵の上の位置にしたもの。行は上から、列は左から 0 で数え、両端を含む（`r0` が上の行、`r1` が下の行）。"""

    r0: int
    r1: int
    c0: int
    c1: int

    def shifted_rows(self, dy: int) -> "Box":
        return Box(self.r0 + dy, self.r1 + dy, self.c0, self.c1)

    def shifted_cols(self, dx: int) -> "Box":
        return Box(self.r0, self.r1, self.c0 + dx, self.c1 + dx)


# ---- 横の割合（縦の割合は spec.py の percent_to_height と同じ） ----

def body_cols(rgba: np.ndarray) -> tuple[int, int] | None:
    """(キャラの左の透明の列の数, キャラの幅)。キャラの幅は、いちばん左の不透明な列からいちばん右の不透明な列まで。"""
    cols = np.flatnonzero((rgba[..., 3] > 0).any(axis=0))
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1] - cols[0] + 1)


def _half_up(v: float) -> int:
    return int(math.floor(v + 0.5 + 1e-9))


def percent_to_width(rgba: np.ndarray, text: str) -> int | None:
    """横の割合（"40.0%"）を「左から何ドット目」にする: 左の透明の列の数 + 四捨五入（p / 100 × キャラの幅）（1 より小さければ 1）。"""
    from .spec import parse_percent
    b = body_cols(rgba)
    p = parse_percent(text)
    if b is None or p is None:
        return None
    blank, width = b
    return blank + max(1, _half_up(p / 100 * width))


def width_to_percent(rgba: np.ndarray, n: int) -> str | None:
    """「左から何ドット目」を横の割合にする（小数 1 桁。読み直して同じ数に戻らなければ桁を増やす）。"""
    b = body_cols(rgba)
    if b is None:
        return None
    blank, width = b
    p = (n - blank) / width * 100
    for digits in range(1, 7):
        text = f"{p:.{digits}f}%"
        if percent_to_width(rgba, text) == n:
            return text
    return None


def edge_number(rgba: np.ndarray, edge: str, value) -> int | None:
    """端の値（数か割合）を数（縦は下から、横は左から何ドット目）にする。割合を決められなければ None。"""
    if not isinstance(value, str):
        return int(value)
    from .spec import percent_to_height
    return percent_to_height(rgba, value) if edge in ("bottom", "top") else percent_to_width(rgba, value)


def edge_percent(rgba: np.ndarray, edge: str, n: int) -> str | None:
    from .spec import height_to_percent
    return height_to_percent(rgba, n) if edge in ("bottom", "top") else width_to_percent(rgba, n)


def box_text(edges: dict) -> str:
    """「下 17〜上 44・左 7〜右 34」（帯は「下 10〜上 16」）。"""
    t = f"下 {edges['bottom']}〜上 {edges['top']}"
    if "left" in edges:
        t += f"・左 {edges['left']}〜右 {edges['right']}"
    return t


def box_edges(box: Box, h: int, kind: str) -> dict:
    """位置（`Box`）を表の書き方（下から・左から何ドット目）に戻す。"""
    out = {}
    if kind == "rect":
        out["left"], out["right"] = box.c0 + 1, box.c1 + 1
    out["bottom"], out["top"] = h - box.r1, h - box.r0
    return out


def resolve_box(base: np.ndarray, region) -> tuple[Box | None, dict | None, Fail | None]:
    """範囲の端を絵の上の位置にする。(位置, 数にした端, 作れない理由)。画像の外の数・上下や左右が逆は ValueError。"""
    h, w = base.shape[:2]
    nums = {}
    for edge in EDGES[region.type]:
        value = region.edges[edge]
        n = edge_number(base, edge, value)
        if n is None:
            text = f"絵が無いので、割合の端（{EDGE_LABEL[edge]} {value}）を決められません"
            return None, None, Fail("split_not_found", text, text)
        limit, what = (h, "高さ") if edge in ("bottom", "top") else (w, "幅")
        if not 1 <= n <= limit:
            raise ValueError(f"範囲 {region.name} の {edge}（{EDGE_LABEL[edge]}）{n} は画像の範囲の外です。この画像（{what} {limit} ドット）では "
                             f"1〜{limit}（{'1 = いちばん下の行' if what == '高さ' else '1 = いちばん左の列'}）で指定してください")
        nums[edge] = n
    if nums["bottom"] > nums["top"]:
        raise ValueError(f"範囲 {region.name} の下（{nums['bottom']}）が上（{nums['top']}）より上にあります")
    if region.type == "rect" and nums["left"] > nums["right"]:
        raise ValueError(f"範囲 {region.name} の左（{nums['left']}）が右（{nums['right']}）より右にあります")
    if region.type == "band":
        nums = {"bottom": nums["bottom"], "top": nums["top"]}
    lim = split_limits(base)
    if lim is None:
        text = "絵の背が低い、または絵が無いので、動かせる範囲がありません"
        return None, nums, Fail("split_not_found", text, text)
    if nums["bottom"] < lim[0]:
        near = dict(nums, bottom=lim[0]) if lim[0] <= nums["top"] else None
        text = (f"範囲の下の端（下から {nums['bottom']} ドット目）が足元の中です（足元の {lim[0] - 1} 行は動かさないため、"
                f"この絵で範囲の下の端に指定できるのは {lim[0]} から上）")
        if near is not None:
            text += f"。近くで作れる範囲: {box_text(near)}（作れるかは試していません）"
        return None, nums, Fail("split_not_found", text, text, rows_from_bottom=[nums["bottom"], nums["bottom"]])
    box = Box(h - nums["top"], h - nums["bottom"], nums.get("left", 1) - 1, nums.get("right", w) - 1)
    return box, nums, None


# ---- 1 ドット動かす（空いた所と重なった所の扱い） ----

def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """同じ色か（透明どうしも同じ）。"""
    if a[3] == 0 or b[3] == 0:
        return bool(a[3] == 0 and b[3] == 0)
    return bool(np.array_equal(a, b))


def _inside(img: np.ndarray, y: int, x: int) -> bool:
    return 0 <= y < img.shape[0] and 0 <= x < img.shape[1]


def _fill(cur: np.ndarray, y: int, x: int, dy: int, dx: int) -> np.ndarray | None:
    """(y, x) の中身が (dy, dx) の向きに 1 ドット動いて空いたとき、そこに入れるドット。埋められなければ None。"""
    m = cur[y, x]
    qy, qx = y - dy, x - dx
    if not _inside(cur, qy, qx) or cur[qy, qx, 3] == 0 or m[3] == 0:
        return np.zeros(4, np.uint8)
    q = cur[qy, qx]
    if _same(q, m):
        return q.copy()
    run = [(qy - k * dy, qx - k * dx) for k in (1, 2)]
    if all(_inside(cur, a, b) and _same(cur[a, b], q) for a, b in run):
        return q.copy()
    return None


def _cover(cur: np.ndarray, new: np.ndarray, y: int, x: int, px: np.ndarray) -> bool:
    """範囲の外に出たドット `px` を (y, x) に描く。外の違う色のドットを隠すなら描かずに False。"""
    if px[3] == 0:
        return True
    if not _inside(cur, y, x):
        return False
    t = cur[y, x]
    if t[3] == 0 or _same(t, px):
        new[y, x] = px
        return True
    return False


def move_step(cur: np.ndarray, box: Box, dy: int, dx: int):
    """範囲の中身を (dy, dx)（どちらかが ±1、もう片方は 0）だけずらす。(新しい絵, 動いたあとの範囲, 空いた所の問題, 重なりの問題)。"""
    new = cur.copy()
    hidden, overlap = [], []
    r0, r1, c0, c1 = box.r0, box.r1, box.c0, box.c1
    if dx:
        if c1 > c0:
            if dx > 0:
                new[r0:r1 + 1, c0 + 1:c1 + 1] = cur[r0:r1 + 1, c0:c1]
            else:
                new[r0:r1 + 1, c0:c1] = cur[r0:r1 + 1, c0 + 1:c1 + 1]
        lead, lead_src, trail = (c1 + 1, c1, c0) if dx > 0 else (c0 - 1, c0, c1)
        for y in range(r0, r1 + 1):
            if not _cover(cur, new, y, lead, cur[y, lead_src]):
                overlap.append([lead, y])
            f = _fill(cur, y, trail, 0, dx)
            if f is None:
                hidden.append([trail, y])
            else:
                new[y, trail] = f
        return new, box.shifted_cols(dx), hidden, overlap
    if r1 > r0:
        if dy > 0:
            new[r0 + 1:r1 + 1, c0:c1 + 1] = cur[r0:r1, c0:c1 + 1]
        else:
            new[r0:r1, c0:c1 + 1] = cur[r0 + 1:r1 + 1, c0:c1 + 1]
    lead, lead_src, trail = (r1 + 1, r1, r0) if dy > 0 else (r0 - 1, r0, r1)
    for x in range(c0, c1 + 1):
        if not _cover(cur, new, lead, x, cur[lead_src, x]):
            overlap.append([x, lead])
        f = _fill(cur, trail, x, dy, 0)
        if f is None:
            hidden.append([x, trail])
        else:
            new[trail, x] = f
    return new, box.shifted_rows(dy), hidden, overlap


def deform_step(cur: np.ndarray, box: Box, seam: np.ndarray, op: str):
    """範囲の中で 1 回縮める（継ぎ目のドットを抜いて上を 1 ドット下げる）か伸ばす（複製して上を 1 ドット上げる）。

    `seam` は範囲の列ごとの行（上から）。(新しい絵, 空いた所の問題, 重なりの問題)。範囲の外の列・範囲より下の行は変えない。
    """
    new = cur.copy()
    hidden, overlap = [], []
    r0 = box.r0
    for i, x in enumerate(range(box.c0, box.c1 + 1)):
        s = int(seam[i])
        if op == STRETCH:
            if not _cover(cur, new, r0 - 1, x, cur[r0, x]):
                overlap.append([x, r0 - 1])
            if s > r0:
                new[r0:s, x] = cur[r0 + 1:s + 1, x]
        else:
            if s > r0:
                new[r0 + 1:s + 1, x] = cur[r0:s, x]
            f = _fill(cur, r0, x, 1, 0)
            if f is None:
                hidden.append([x, r0])
            else:
                new[r0, x] = f
    return new, hidden, overlap


# ---- 確かめ ----

class _Check:
    """元の絵（余白を足したもの）と比べてポーズを確かめる（目印は 1 回だけ作る）。"""

    def __init__(self, src: np.ndarray, keep: bool = False):
        self.src = src
        self.ref = rule_reference(src)
        self.euler = euler_numbers(src)
        self.parts = small_parts(src) if keep else None

    def problems(self, pose: np.ndarray) -> list[str]:
        """守れていない決まりの名前の並び（空なら守れている）。"""
        got = [r for r, _ in frame_problems(self.src, pose, ref=self.ref)]
        if not got and euler_numbers(pose) != self.euler:
            got = ["euler"]
        if not got and self.parts is not None and not parts_kept(self.src, pose, self.parts):
            got = ["small_parts"]
        return list(dict.fromkeys(got))


@dataclass
class Outcome:
    """1 つの値（−1・+2 など）のポーズを作った結果。`pose` が None なら作れなかった（`rule`・`pixels` が理由）。"""

    pose: np.ndarray | None
    rule: str = ""
    pixels: list | None = None
    detail: str = ""        # 継ぎ目が見つからなかったなどの理由（決まりの言葉の代わり）


def _seam_candidates(cur: np.ndarray, box: Box, r_top: int, rng: int, block):
    """範囲の列で、範囲の下の端から `rng` ドット以内（`r_top` より下）の継ぎ目の候補（目立ち方の小さい順、同じものは 1 つ）。"""
    key = color_keys(cur)[:, box.c0:box.c1 + 1]
    run, pref = _prepare(key)
    lo_all, hi_all = max(r_top, box.r1 - rng), box.r1
    blk = None if block is None else block[:, box.c0:box.c1 + 1]
    found = {}
    for lo in range(lo_all, hi_all + 1):
        for hi in range(lo, hi_all + 1):
            got = _seam_in_band(key, run, pref, lo, hi, box.r1, blk)
            if got is not None:
                found.setdefault(tuple(got[0].tolist()), got)
    return sorted(found.values(), key=lambda c: (c[1], c[0].tolist()))


def deform_box(src: np.ndarray, box: Box, n: int, rng: int, op: str, keep: bool = False):
    """範囲の中で n 回縮めた（伸ばした）ポーズ。(Outcome の並び（k = 1〜n）, 継ぎ目, 目立ち方)。

    どの段でも、決まりを守る継ぎ目のうち最も目立たないものを選ぶ。作れなかった段から先は作れない（同じ理由）。
    """
    check = _Check(src, keep)
    poses, seams, costs, outs = [src], [], [], []
    for j in range(n):
        cur = poses[-1]
        top = box.r0 - j if op == STRETCH else box.r0      # 伸ばすと、範囲の上の端も 1 行ずつ上に広がる
        b = Box(top, box.r1, box.c0, box.c1)
        cands = _seam_candidates(cur, b, top, rng, small_part_mask(cur) if keep else None)
        first = None
        for seam, cost in cands:
            pose, hidden, overlap = deform_step(cur, b, seam, op)
            if hidden or overlap:
                got = Outcome(None, "hidden" if hidden else "overlap", hidden or overlap)
            else:
                rules = check.problems(pose)
                if not rules:
                    poses.append(pose)
                    seams.append(np.asarray(seam, np.intp))
                    costs.append(round(cost, 2))
                    outs.append(Outcome(pose))
                    break
                got = Outcome(None, rules[0], problem_pixels(src, pose, rules[0] if rules[0] != "small_parts" else "parts"))
                got.detail = "・".join(RULE_TEXT[r] for r in rules)
            first = first or got
        else:
            if first is None:
                h = src.shape[0]
                a, b2 = h - box.r1, h - max(top, box.r1 - rng)
                verb = "伸ばせる" if op == STRETCH else "縮められる"
                first = Outcome(None, "split_not_found",
                                detail=f"範囲の下の端の近く（下から {a}〜{b2} ドット目）に{verb}継ぎ目がありません"
                                       "（1 ドットの線が横切っている" + ("、または小さな部品を通る" if keep else "") + "）")
            outs += [first] * (n - j)
            return outs, seams, costs
    return outs, seams, costs


def move_box(src: np.ndarray, box: Box, value: int, axis: str) -> Outcome:
    """範囲の中身を形のまま `value` ドットずらしたポーズ（`axis` "x" は + が右、"y" は + が下）。"""
    check = _Check(src)
    s = 1 if value > 0 else -1
    cur, b = src, box
    for _ in range(abs(value)):
        cur, b, hidden, overlap = move_step(cur, b, s if axis == "y" else 0, s if axis == "x" else 0)
        if hidden or overlap:
            return Outcome(None, "hidden" if hidden else "overlap", hidden or overlap)
    rules = check.problems(cur)
    if rules:
        out = Outcome(None, rules[0], problem_pixels(src, cur, rules[0]))
        out.detail = "・".join(RULE_TEXT[r] for r in rules)
        return out
    return Outcome(cur)


def move_padding(base: np.ndarray, box: Box, amount: int) -> int:
    """横にずらしてはみ出さないために、左右に足す透明の列の数（横揺れと同じく左右に同じ数）。"""
    op = base[box.r0:box.r1 + 1, box.c0:box.c1 + 1, 3] > 0
    cols = np.flatnonzero(op.any(axis=0))
    if len(cols) == 0:
        return 0
    w = base.shape[1]
    left, right = box.c0 + int(cols[0]), box.c0 + int(cols[-1])
    return max(0, amount - left, right + amount - (w - 1))


def top_need(base: np.ndarray, box: Box, n: int) -> int:
    """範囲の中身を上に n ドット出すとき（伸ばす・上にずらす）、絵の上に足す透明の行の数。"""
    op = base[box.r0:box.r1 + 1, box.c0:box.c1 + 1, 3] > 0
    rows = np.flatnonzero(op.any(axis=1))
    if len(rows) == 0:
        return 0
    return max(0, n - (box.r0 + int(rows[0])))


def outcome_text(out: Outcome, what: str) -> str:
    """「1 ドット縮めると…」の形の言葉（`what` は「1 ドット縮めると」など）。"""
    if out.rule == "split_not_found":
        return out.detail
    return what + (out.detail or RULE_TEXT[out.rule])


def near_boxes(base: np.ndarray, kind: str, nums: dict, works) -> list[dict]:
    """端を 1〜3 ドットずつずらした範囲を実際に作って確かめ、作れたものを `NEAR_COUNT` 個まで返す（近い順）。

    `works(box)` は、その範囲で列の全部の値が作れるか。ずらすのは 1 つの端だけ（下・上・左・右の順）。
    """
    h, w = base.shape[:2]
    lim = split_limits(base)
    if lim is None:
        return []
    out, tries = [], 0
    for step in NEAR_STEPS:
        for edge in EDGES[kind][-2:] + EDGES[kind][:-2]:        # 下・上・左・右の順
            cand = dict(nums)
            cand[edge] += step
            if not (lim[0] <= cand["bottom"] <= cand["top"] <= h):
                continue
            if kind == "rect" and not 1 <= cand["left"] <= cand["right"] <= w:
                continue
            tries += 1
            box = Box(h - cand["top"], h - cand["bottom"], cand.get("left", 1) - 1, cand.get("right", w) - 1)
            if works(box):
                out.append(cand)
                if len(out) >= NEAR_COUNT:
                    return out
            if tries >= NEAR_TRIES:
                return out
    return out


def near_text(near: list[dict]) -> str:
    if not near:
        return f"近くで作れる範囲は見つかりませんでした（端を {max(NEAR_STEPS)} ドットまでずらして試しました）"
    return "近くで作れる範囲: " + "、".join(box_text(n) for n in near)
