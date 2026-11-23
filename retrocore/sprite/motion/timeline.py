"""動きの表（タイムライン）からフレームを作る仕組み。

表の形は `spec.py`、設計は docs/sprite-app-plan.md の「動きの表（タイムライン）の設計」。

    from retrocore.sprite.motion import make_motion_from_spec, load_spec
    res = make_motion_from_spec(sprite, load_spec("bounce.motion.json"))

1 フレームは「その場の形（ポーズ）」と「体全体のずれ」の 2 段で作り、それぞれ今の決まりで確かめる（設計の 3.）:

1. **ポーズを作る**: 元の絵に、その行の位置（`move`）・変形（`deform`）の値を当てる。動かし方の中身は今の動きの関数をそのまま使う
   （縮める = 呼吸の継ぎ目、伸ばす = 同じ継ぎ目のドットを複製して上を上げる、横の位置 = 横揺れの分け目）。伸ばすときは、
   上に出る分の透明の行を先に全部のポーズの上に足す（`padding_y` の上。足元はフレームの下のまま）。範囲に `keep_small_parts` が
   あれば、縮める・伸ばすで目・頬などの小さな部品を形のまま動かす（`breathe.small_parts()`）。範囲は足元（下から 12%、2 ドット以上）を含められないので、
   **ポーズでは足元は動かない**
2. **ポーズを確かめる**: 縮めたポーズは `check_rules()`（新しい色なし・同じ大きさ・足元が同じ・部品と輪郭のつながり・縁の色）。
   横の位置は今の横揺れと同じく、左右いちばん大きく動かした形まで、それに色ごとのオイラー数を足して確かめる
3. **体全体のずれを当てる**: ポーズを縦に `root` の値だけずらす（浮遊の `shift_rows()`）。はみ出す分は先に上下に透明の行を足す
4. **ずれを確かめる**: `check_rules(..., shifts=...)` と同じく「ポーズを縦にずらしただけ（欠けたドットも無い）」と
   「足元が縦にちょうどその量だけ動き、横には動かない」。足元が動いてよいのは、範囲を持たない `root` の列だけ

範囲が帯・四角（`region.py`）のときは、1. と 2. を `region.py` の囲んだ中だけを動かす作り方と確かめ（空いた所・重なり・オイラー数も）にする。
3. と 4. は同じ。

同じ値のポーズは 1 回だけ作って使い回す（8 フレームの呼吸でもポーズは 0 と −1 の 2 つ）。1 行でも作れなければシートは作らず
（途中のフレームだけ抜いたシートは作らない）、全部の行を確かめて、問題をまとめて警告する（行の番号は表と同じ 1 から）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import AUTO, RULE_TEXT, Fail, MotionResult, check_split, frame_problems, problem_pixels, split_limits
from .breathe import parts_kept, shrink_poses, stretch_poses
from .floating import shift_rows
from . import region as region_mod
from .spec import (fmt_value, height_to_percent, parse_percent, parse_spec, percent_to_height,
                   spec_to_dict)
from .sway import move_poses


@dataclass
class Problem:
    """作れなかった行の問題（設計の 6.）。`rows` は表と同じ 1 から数えた行の番号。"""

    rows: list[int]
    track: str | None
    region: str | None
    value: int | None
    rule: str
    message: str
    rows_from_bottom: list | None = None
    pixels: list = field(default_factory=list)
    near: list = field(default_factory=list)
    fail: Fail | None = None        # 動かし方が作れなかったときの理由（プリセットが今の警告を出すのに使う）
    values: list = field(default_factory=list)   # まとめた行の値（`_merge()`）

    def text(self) -> str:
        """「3 行目（upper_y = −1、範囲 upper）: …」の形の 1 文。"""
        rows = "・".join(str(r) for r in self.rows) + " 行目"
        vals = self.values or [self.value]
        what = [] if self.track is None else [f"{self.track} = {'・'.join(fmt_value(v or 0) for v in vals)}"]
        if self.region is not None:
            what.append(f"範囲 {self.region}")
        return f"{rows}（{'、'.join(what)}）: {self.message}" if what else f"{rows}: {self.message}"

    def to_dict(self) -> dict:
        d = {"row": self.rows[0], "rows": list(self.rows), "track": self.track, "region": self.region,
             "value": self.value, "values": list(self.values or [self.value]), "rule": self.rule, "message": self.text()}
        if self.rows_from_bottom is not None:
            d["rows_from_bottom"] = list(self.rows_from_bottom)
        if self.near:
            d["near"] = list(self.near)
        if self.pixels:
            d["pixels"] = [list(p) for p in self.pixels]
        return d


@dataclass
class Built:
    """表から作った結果（`build()`）。`problems` が空ならシートにできる。"""

    frames: list | None
    problems: list[Problem]
    regions: dict                    # 範囲の名前 → {"split": None か int, "tracks": {列の名前: {"lines", "costs", "padding"}}}
    raw_poses: dict                  # (列の名前, 値) → ポーズ（決まりを守れなかったものも入る。プリセットの警告用）
    padding: int = 0                 # 左右に足した透明の列の数（横の位置を使ったとき）
    padding_y: tuple = (0, 0)        # 上と下に足した透明の行の数（体全体のずれを使ったとき）
    offset_y: list | None = None     # 行ごとの体全体のずれ（体全体のずれを使ったとき）
    move_used: bool = False
    root_used: bool = False
    poses: int = 0                   # 作ったポーズの数（元の絵は数えない）
    used: dict = field(default_factory=dict)   # 範囲の名前 → 実際に使った高さと線（表の "used"）
    stretch_top: int = 0             # 伸ばす分として上に足した透明の行の数（padding_y の上に含む）


def _merge(problems: list[Problem], order: list[str]) -> list[Problem]:
    """同じ列・同じ問題（同じ言葉）の行をまとめる（「1・3 行目（upper_y = −1・−2、範囲 upper）: …」）。行の順に並べる。"""
    out: dict = {}
    first: dict = {}                  # まとめた問題 → 値 → その値の最初の行（値は行の順に並べる）
    for q in problems:
        key = (q.track, q.region, q.rule, q.message)
        if key in out:
            m = out[key]
            m.rows = sorted(set(m.rows) | set(q.rows))
        else:
            m = out[key] = q
            first[key] = {}
        first[key][q.value] = min(first[key].get(q.value, q.rows[0]), q.rows[0])
    for key, m in out.items():
        m.values = sorted(first[key], key=first[key].get)
        m.value = m.values[0]
    return sorted(out.values(), key=lambda q: (q.rows[0], order.index(q.track) if q.track in order else -1))


def _resolve_split(base: np.ndarray, region) -> tuple[int | None, Fail | None]:
    """範囲の at を、None（自動）か下から何ドット目かにする。画像の外は ValueError。"""
    at = region.at
    if at == AUTO:
        return None, None
    if isinstance(at, str):
        n = percent_to_height(base, at)
        if n is None:
            text = f"絵が無いので、割合の高さ（{at}）を決められません"
            return None, Fail("split_not_found", text, text)
        at = n
    try:
        return check_split(at, base.shape[0]), None
    except ValueError as e:
        raise ValueError(f"範囲 {region.name} の at: {e}") from None


def _check_lines(lines, need: int, base: np.ndarray, where: str) -> None:
    h, w = base.shape[:2]
    if len(lines) < need:
        raise ValueError(f"{where} の lines は {need} 本書いてください（今は {len(lines)} 本）")
    for line in lines[:need]:
        if len(line) != w:
            raise ValueError(f"{where} の lines の幅（{len(line)}）が、この画像の幅（{w}）と違います")
        if min(line) < 1 or max(line) > h:
            raise ValueError(f"{where} の lines の高さは 1〜{h}（下から何ドット目）で書いてください")


def _rows_with(col: list[int], test) -> list[int]:
    return [i + 1 for i, v in enumerate(col) if test(v)]


def _used_of(base: np.ndarray, tracks: dict) -> dict:
    """範囲で実際に使った高さと線（表の "used"・シートの JSON の "regions_used" の形）。伸ばすの線は「列の名前+」。"""
    h = base.shape[0]
    lines = {name: [(t.get("height", h) - np.asarray(line)).astype(int).tolist() for line in t["lines"]]
             for name, t in tracks.items() if t["lines"]}
    vals = [v for ls in lines.values() for line in ls for v in line]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    at = int(np.floor((lo + hi) / 2 + 0.5))
    return {"at": at, "percent": height_to_percent(base, at), "from_bottom": [lo, hi], "lines": lines}


def _box_used(info: dict) -> dict:
    """帯・四角で実際に使った範囲（表の "used"・シートの JSON の "regions_used"）。端は数、継ぎ目は範囲の左の列からの並び。"""
    out = dict(info["box"])
    lines = {name: [(t["height"] - np.asarray(line)).astype(int).tolist() for line in t["lines"]]
             for name, t in info["tracks"].items() if t["lines"]}
    if lines:
        out["lines"] = lines
    return out


def _build_box(spec, region, rtracks, resolved, base, base_s, s_top, problems, raw, good, move_pad) -> dict:
    """帯・四角の範囲のポーズを作って確かめる（`build()` の 1. と 2.）。作れない値は problems に足す。"""
    box, nums, rfail = resolved
    rname = region.name
    info = {"split": None, "type": region.type, "box": dict(nums or {}), "tracks": {}}
    if rfail is not None:
        for t in rtracks:
            col = spec.column(t.name)
            for v in sorted({v for v in col if v}):
                problems.append(Problem(_rows_with(col, lambda x: x == v), t.name, rname, v, rfail.rule, rfail.text,
                                        rfail.rows_from_bottom, fail=rfail))
        return info
    h = base.shape[0]
    box_s = box.shifted_rows(s_top)
    for t in rtracks:
        col = spec.column(t.name)
        if t.type == "deform":
            for sign, op in ((-1, region_mod.SHRINK), (1, region_mod.STRETCH)):
                need = sorted({sign * v for v in col if sign * v > 0})
                if not need:
                    continue
                n = need[-1]
                key = t.name if sign < 0 else t.name + "+"
                outs, seams, costs = region_mod.deform_box(base_s, box_s, n, region.range, op, region.keep_small_parts)
                info["tracks"][key] = {"type": "deform", "lines": seams, "costs": costs, "padding": 0,
                                       "height": int(base_s.shape[0])}
                word = "縮めると" if sign < 0 else "伸ばすと"
                near = None
                for k in need:
                    out = outs[k - 1]
                    if out.pose is not None:
                        raw[(t.name, sign * k)] = good[(t.name, sign * k)] = out.pose
                        continue
                    if near is None:
                        def works(b, n=n, op=op):
                            return region_mod.deform_box(base_s, b.shifted_rows(s_top), n, region.range, op,
                                                         region.keep_small_parts)[0][-1].pose is not None
                        near = region_mod.near_boxes(base, region.type, nums, works)
                    _box_problem(problems, col, t.name, rname, sign * k, out, f"{k} ドット{word}", near, s_top, 0)
        else:   # move（位置）: 範囲の中身を形のまま横か縦にずらす
            vals = sorted({v for v in col if v})
            amount = max(abs(v) for v in vals)

            def made(b, vals=vals, amount=amount, axis=t.axis):
                pad = region_mod.move_padding(base, b, amount) if axis == "x" else 0
                src = np.pad(base_s, ((0, 0), (pad, pad), (0, 0))) if pad else base_s
                bb = b.shifted_rows(s_top).shifted_cols(pad)
                return pad, {v: region_mod.move_box(src, bb, v, axis) for v in vals}

            pad, outs = made(box)
            info["tracks"][t.name] = {"type": "move", "lines": [], "costs": [], "padding": pad}
            if all(o.pose is not None for o in outs.values()):
                move_pad[t.name] = pad
            near = None
            for v in vals:
                out = outs[v]
                if out.pose is not None:
                    raw[(t.name, v)] = out.pose
                    if t.name in move_pad:
                        good[(t.name, v)] = out.pose
                    continue
                if near is None:
                    near = region_mod.near_boxes(base, region.type, nums,
                                                 lambda b: all(o.pose is not None for o in made(b)[1].values()))
                way = ("横に " if t.axis == "x" else "縦に ") + fmt_value(v) + " ずらすと"
                _box_problem(problems, col, t.name, rname, v, out, way, near, s_top, pad)
    return info


def _box_problem(problems, col, track, rname, v, out, what, near, dy: int, dx: int) -> None:
    """帯・四角で作れなかった値の問題。場所（pixels）は元の絵の上の位置にする（上と左右に足した余白を引く）。"""
    text = region_mod.outcome_text(out, what) + "。" + region_mod.near_text(near)
    fail = Fail(out.rule, text, text, near=list(near))
    pixels = [[x - dx, y - dy] for x, y in (out.pixels or [])][:32]
    problems.append(Problem(_rows_with(col, lambda x: x == v), track, rname, v, out.rule, text,
                            pixels=pixels, near=list(near), fail=fail))


def build(rgba: np.ndarray, spec) -> Built:
    """表からフレームを作り、確かめる。画像に合わない指定（画像の外の高さ・幅の違う線）は ValueError。"""
    spec = parse_spec(spec)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = base.shape[:2]
    problems: list[Problem] = []
    regions: dict = {}
    raw: dict = {}                    # (列, 値) → ポーズ
    good: dict = {}                   # (列, 値) → 確かめを通ったポーズ
    move_pad: dict = {}               # 列 → 左右に足した列の数

    # 帯・四角の範囲を、先に絵の上の位置にする（画像の外の端は ValueError）
    boxes: dict = {}
    for rname, region in spec.regions.items():
        if region.edges is not None and any(spec.used(t.name) for t in spec.tracks.values() if t.region == rname):
            boxes[rname] = region_mod.resolve_box(base, region)

    # 伸ばす（deform の +）: 上に出る分の透明の行を、先に全部のポーズの上に足す（横揺れの左右の余白と同じ考え方。
    # 足元はフレームの下のまま）。足す数は「いちばん大きく伸ばす量 − 絵の上にもともとある透明の行」
    stretch_n = max([v for t in spec.tracks.values() if t.type == "deform" and t.region not in boxes
                     for v in spec.column(t.name)] + [0])
    opaque_rows = np.flatnonzero((base[..., 3] > 0).any(axis=1))
    s_top = max(0, stretch_n - (int(opaque_rows[0]) if len(opaque_rows) else stretch_n))
    # 帯・四角: 中身を上に出す（伸ばす・上にずらす）分が絵の上の端を越えるときだけ足す
    for rname, (box, _, _) in boxes.items():
        for t in spec.tracks.values():
            if t.region != rname or box is None:
                continue
            up = [v for v in spec.column(t.name) if v > 0] if t.type == "deform" else \
                [-v for v in spec.column(t.name) if v < 0] if (t.type, t.axis) == ("move", "y") else []
            if up:
                s_top = max(s_top, region_mod.top_need(base, box, max(up)))
    base_s = np.pad(base, ((s_top, 0), (0, 0), (0, 0))) if s_top else base

    # ---- 1. と 2.: 範囲ごとにポーズを作って確かめる ----
    for rname, region in spec.regions.items():
        rtracks = [t for t in spec.tracks.values() if t.region == rname and spec.used(t.name)]
        if not rtracks:
            continue
        if rname in boxes:
            regions[rname] = _build_box(spec, region, rtracks, boxes[rname], base, base_s, s_top,
                                        problems, raw, good, move_pad)
            continue
        split, rfail = _resolve_split(base, region)
        info = regions[rname] = {"split": split, "tracks": {}}
        for t in rtracks:
            col = spec.column(t.name)
            if rfail is not None:
                for v in sorted({v for v in col if v}):
                    problems.append(Problem(_rows_with(col, lambda x: x == v), t.name, rname, v, rfail.rule,
                                            rfail.text, fail=rfail))
                continue
            lines = (region.lines or {}).get(t.name)
            if t.type == "deform":
                plus_lines = (region.lines or {}).get(t.name + "+")   # 伸ばすの線は「列の名前+」に書く
                keep = region.keep_small_parts
                for sign, src in ((-1, base), (1, base_s)):
                    need = sorted({sign * v for v in col if sign * v > 0})
                    if not need:
                        continue
                    n = need[-1]
                    use = lines if sign < 0 else plus_lines
                    key = t.name if sign < 0 else t.name + "+"
                    if use is not None:
                        _check_lines(use, n, src, f"範囲 {rname} の {key}")
                    if sign < 0:
                        poses, seams, costs, fail = shrink_poses(src, split, n, region.range, use, keep)
                    else:
                        poses, seams, costs, fail = stretch_poses(src, split, n, region.range, use, keep,
                                                                  split_limits(base))
                    info["tracks"][key] = {"type": "deform", "lines": seams, "costs": costs, "padding": 0,
                                           "height": int(src.shape[0])}
                    word = "縮めると" if sign < 0 else "伸ばすと"
                    for k in need:
                        rows = _rows_with(col, lambda x: x == sign * k)
                        if fail is not None and k >= fail.depth:
                            problems.append(Problem(rows, t.name, rname, sign * k, fail.rule, fail.text,
                                                    fail.rows_from_bottom, fail.pixels, fail.near, fail))
                            continue
                        pose = raw[(t.name, sign * k)] = poses[k]
                        got = frame_problems(src, pose)
                        if not got and keep and not parts_kept(src, pose):
                            got = [("small_parts", "")]
                        if got:
                            rules = list(dict.fromkeys(r for r, _ in got))
                            problems.append(Problem(rows, t.name, rname, sign * k, rules[0],
                                                    f"{k} ドット{word}" + "・".join(RULE_TEXT[r] for r in rules),
                                                    pixels=problem_pixels(src, pose, rules[0])))
                        else:
                            good[(t.name, sign * k)] = pose
            else:   # move（横の位置）
                amount = max(abs(v) for v in col)
                values = sorted(set(col))
                if lines is not None:
                    _check_lines(lines, amount, base, f"範囲 {rname} の {t.name}")
                shapes, cuts, costs, pad, fail = move_poses(base, split, amount, values, region.range, lines)
                info["tracks"][t.name] = {"type": "move", "lines": cuts if fail is None else [], "costs": costs,
                                          "padding": pad}
                if fail is not None:
                    for v in sorted({v for v in col if v}):
                        problems.append(Problem(_rows_with(col, lambda x: x == v), t.name, rname, v, fail.rule,
                                                fail.text, fail.rows_from_bottom, fail.pixels, fail.near, fail))
                    continue
                move_pad[t.name] = pad
                for v in values:
                    if v:
                        raw[(t.name, v)] = good[(t.name, v)] = shapes[v]

    # 横の位置で足した余白は全部のポーズでそろえる（今の横揺れと同じく、足元はフレームの真ん中のまま）
    p = max(move_pad.values(), default=0)

    def hpad(a: np.ndarray, have: int) -> np.ndarray:
        e = p - have
        a = np.pad(a, ((0, 0), (e, e), (0, 0))) if e else a
        t = s_top - (a.shape[0] - h)          # 伸ばす分の上の余白（伸ばしたポーズはもう足してある）
        return np.pad(a, ((t, 0), (0, 0), (0, 0))) if t else a

    base_p = hpad(base, 0)
    row_keys, row_poses, padded = [], [], {}
    for row in spec.rows:
        key = next(((k, v) for k, v in row.values.items() if spec.tracks[k].type != "root"), None)
        row_keys.append(key)
        if key is None:
            row_poses.append(base_p)
        elif key in good:
            if key not in padded:
                padded[key] = hpad(good[key], move_pad.get(key[0], 0))
            row_poses.append(padded[key])
        else:
            row_poses.append(None)

    # ---- 3. と 4.: 体全体のずれ ----
    root = next((t for t in spec.tracks.values() if t.type == "root"), None)
    offs = spec.column(root.name) if root is not None else [0] * len(spec.rows)
    root_used = root is not None and any(offs)
    top = bottom = 0
    if root_used:
        made = [pp for pp in row_poses if pp is not None]
        ops = [np.flatnonzero((pp[..., 3] > 0).any(axis=1)) for pp in made]
        ops = [r for r in ops if len(r)]
        if made and not ops:
            for v in sorted({v for v in offs if v}):
                text = "絵が無いので、体全体を動かせません（不透明なドットが無い）"
                problems.append(Problem(_rows_with(offs, lambda x: x == v), root.name, None, v, "no_image", text,
                                        fail=Fail("no_image", "絵がありません（不透明なドットが無い）", text)))
            root_used_ok = False
        else:
            root_used_ok = bool(made)
        if root_used_ok:
            hh = base_p.shape[0]
            first = min(int(r[0]) for r in ops)
            last = max(int(r[-1]) for r in ops)
            top = max(0, max(0, -min(offs)) - first)
            if s_top:
                # 伸ばすと一緒に使うときは、行ごとに「上へのずれ − そのポーズの上の透明の行」を見る
                # （伸ばしたポーズを上にずらす行と、伸ばさないポーズを大きくずらす行が別のとき、余白を足しすぎないように）
                need = [-dy - int(np.flatnonzero((pp[..., 3] > 0).any(axis=1))[0])
                        for pp, dy in zip(row_poses, offs) if pp is not None and (pp[..., 3] > 0).any()]
                top = max([0] + need)
            bottom = max(0, max(0, max(offs)) - (hh - 1 - last))

    def vpad(a: np.ndarray) -> np.ndarray:
        return np.pad(a, ((top, bottom), (0, 0), (0, 0))) if top or bottom else a

    frames, cache, vcache = [], {}, {}
    root_bad: dict = {}
    for i, pose in enumerate(row_poses):
        if pose is None:
            frames.append(None)
            continue
        dy = offs[i]
        ck = (row_keys[i], dy)
        if ck not in cache:
            if row_keys[i] not in vcache:
                vcache[row_keys[i]] = vpad(pose)
            src = vcache[row_keys[i]]
            f = shift_rows(src, dy) if root_used else src
            cache[ck] = f
            if root_used and dy:
                got = frame_problems(src, f, shift=dy)
                if got:
                    root_bad[ck] = (got, src, f)
        if ck in root_bad:
            frames.append(None)
        else:
            frames.append(cache[ck].copy())
    for (key, dy), (got, src, f) in root_bad.items():
        rows = [i + 1 for i in range(len(spec.rows)) if (row_keys[i], offs[i]) == (key, dy)]
        rules = list(dict.fromkeys(r for r, _ in got))
        text = f"体全体を縦に {dy:+d} ずらすと" + "・".join(RULE_TEXT[r] for r in rules)
        problems.append(Problem(rows, root.name, None, dy, rules[0], text, pixels=problem_pixels(src, f, rules[0]),
                                fail=Fail(rules[0], text, text)))

    problems = _merge(problems, list(spec.tracks))
    used = {}
    if not problems:
        for rname, info in regions.items():
            if "box" in info:
                used[rname] = _box_used(info)
                continue
            u = _used_of(base, info["tracks"])
            if u:
                used[rname] = u
    return Built(None if problems else frames, problems, regions, raw, p, (top + s_top, bottom),
                 offs if root_used else None, bool(move_pad), root_used, len(good), used, s_top)


def make_motion_from_spec(rgba: np.ndarray, spec) -> MotionResult:
    """表（辞書・JSON の文字列・`MotionSpec`）から動きのフレームを作る。

    作れた: フレームと表示時間、`info`（`poses`: 作ったポーズの数、`padding`: 左右に足した列の数（横の位置を使ったとき）、
    `padding_y`・`offset_y`（体全体のずれを使ったとき）、`regions_used`（at が "auto" か割合の範囲で実際に使った高さと線と、
    帯・四角で実際に使った範囲（端の数）と継ぎ目））、
    `spec`（読んだ表。範囲に "used" を足したもの）。
    作れない行があった: `skipped`。`warnings` の 1 つ目は見出し、2 つ目からは「N 行目（列 = 値、範囲 名前）: …」。
    `info["problems"]` に同じ問題を形のある辞書で入れる（row は 1 から）。`skips` に作らなかった理由の記録（`report.py`）。
    表の書き間違いは `SpecError`、画像に合わない高さ（画像の外）・幅の違う線は `ValueError`（どちらも何も作らない）。
    """
    spec = parse_spec(spec)
    base = np.ascontiguousarray(rgba, dtype=np.uint8)
    built = build(base, spec)
    durations = [row.ms for row in spec.rows]
    if built.problems:
        bad_rows = len({r for q in built.problems for r in q.rows})
        warnings = [f"動き「{spec.name}」のシートを作りませんでした（作れない行が {bad_rows} 行あります。"
                    f"問題 {len(built.problems)} 件）"] + [q.text() for q in built.problems]
        from .report import problem_entries
        boxes = {n: r["box"] for n, r in built.regions.items() if isinstance(r, dict) and "box" in r}
        return MotionResult(spec.name, [base.copy()], [sum(durations)], warnings, True,
                            {"problems": [q.to_dict() for q in built.problems]}, spec=spec_to_dict(spec),
                            skips=problem_entries(built.problems, boxes=boxes), planned_ms=list(durations))
    info: dict = {"poses": built.poses, "regions": {
        name: {**({"type": r["type"], "box": dict(r["box"])} if "box" in r
                  else {"split": AUTO if r["split"] is None else r["split"]}),
               "tracks": {t: {"from_bottom": [[int(d.get("height", base.shape[0]) - np.max(line)),
                                               int(d.get("height", base.shape[0]) - np.min(line))]
                                              for line in d["lines"]], "padding": d["padding"]}
                          for t, d in r["tracks"].items()}}
        for name, r in built.regions.items()}}
    if built.move_used:
        info["padding"] = int(built.padding)
    if built.root_used or built.stretch_top:
        info["padding_y"] = [int(v) for v in built.padding_y]
    if built.root_used:
        info["offset_y"] = [int(v) for v in built.offset_y]
    # 帯・四角は、実際に使った範囲（端を数にしたもの）と継ぎ目をいつも書く（前の版には無い範囲なので、前の出力は変わらない）
    ru = {name: u for name, u in built.used.items()
          if spec.regions[name].edges is not None
          or (spec.regions[name].lines is None and (spec.regions[name].at == AUTO
                                                    or parse_percent(spec.regions[name].at) is not None))}
    if ru:
        info["regions_used"] = ru
    return MotionResult(spec.name, built.frames, durations, [], False, info, spec=spec_to_dict(spec, built.used))
