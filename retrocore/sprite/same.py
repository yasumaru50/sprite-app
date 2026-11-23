"""コマのシートで「この範囲は 1 コマ目と同じにする」（`--frames --same-as-first x,y,w,h`）。

    from retrocore.sprite.same import SameRange, same_frames
    frames, report = same_frames(res.frames, [SameRange(0, 0, 53, 52)], search=3)

AI が描いた歩き・走りのコマは、頭をコマごとに描き直しているため、頭の輪郭・髪の束が 1 ドット隣にずれて再生するとちらつく。
人が選んだ範囲（1 コマ目の上の四角か帯）の中身を、2 コマ目以降の同じもの（頭）がある位置に、1 コマ目のドットのまま写す。
目・口は「1 コマ目とまったく同じ」か「そのコマのまま」のどちらかで、混ざらない（範囲ごとに全部写すか、何もしないか）。

手順（コマごと・範囲ごと。どれも迷ったら何もしない）:

1. **位置を探す**: 範囲の中身を、上下左右 `search` ドット以内でずらしてコマに重ね、合い方（どちらかが不透明なドットのうち、
   不透明・透明と色が同じドットの割合）がいちばん高いずれを探す（同じなら小さいずれ）。頭が上下・左右に揺れていても、その位置に写す
2. **自信がないときは写さない**: 合い方が `MIN_MATCH` 未満なら写さない（`low_match`）
3. **入ってきた部品は残す**: 範囲の中で、コマにだけある太い（3×3 のかたまりを含む）部品で、範囲の端から外のドットとつながっているもの
   （腕を上げた・武器・範囲の外から伸びる髪）は、写さずにそのコマのドットを残す。残す部品が 1 コマ目の目・口などの小さな部品に触れるか、
   残すドットが多すぎる（`KEEP_MAX` を超える）ときは、目・口が混ざるので写さない（`small_part`・`intrusion`）
4. **写した結果を確かめる**（だめなら写さない）:
   - 境目（`seam`）: 範囲の端をまたぐ所で、写す前はつながっていた（透明どうし・輪郭どうし・ほかの色どうしが隣にある）のに、
     写した後はつながらなくなった所（段差・輪郭の途切れ・透明のすき間）
   - 穴（`hole`）: 外の透明とつながらない透明が新しくできる
   - 足元（`foot`）: いちばん下の不透明な行が変わる
   - キャンバスの外（`edge`）: 写す中身がキャンバスの外に出る
   範囲が足元（いちばん下の不透明な行）にかかるときは、写せても警告する
5. 写せなかった範囲には、範囲の端を 1〜3 ドットずらした範囲・探す幅を広げた・範囲を小さくしたなど、次に試せることを添える
   （端をずらした範囲は実際に試して、写せるコマが増えるものだけを添える）

同じ元のコマは同じ結果になる（「行って戻る」1,2,3,2 の 2 コマ目は 1 回だけ計算する）。基準のコマのドットか、そのコマのドットしか
使わないので、新しい色は作らない。乱数は使わない。`retrocore` の方針どおり NumPy と Pillow だけを使う（np.bincount などは使っていない）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SEARCH = 3          # 位置を探す幅（上下左右のドット）
MIN_SEARCH, MAX_SEARCH = 0, 8
MIN_MATCH = 0.55            # これ未満の合い方のコマは写さない
KEEP_MAX = 0.25             # 残す部品のドットが、範囲の中のどちらかが不透明なドットのこの割合を超えたら写さない
KEEP_MIN = 6                # 入ってきた部品とみなす最小のドット数
MAX_RANGES = 16
SUGGEST_ROUNDS = 4          # 「次に試せること」の下調べでの入れ替えの回数の上限
REFINE_ROUNDS = 12          # 境目の問題を減らす入れ替えの回数の上限
SEAM_BAND = 3               # 継ぎ目を範囲の端から内側に寄せてよい幅（ドット）
NEAR_STEPS = (1, -1, 2, -2, 3, -3)
NEAR_COUNT = 3
NEAR_TRIES = 12             # 写せないときに端をずらして試す範囲の数の上限

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 写さなかった理由（JSON の code）
CODE_LOW = "low_match"        # 合い方が低い（位置に自信がない）
CODE_EMPTY = "empty"          # 1 コマ目の範囲の中に不透明なドットが無い
CODE_EDGE = "edge"            # 写す中身がキャンバスの外に出る
CODE_SEAM = "seam"            # 境目で段差・輪郭の途切れ・透明のすき間ができる
CODE_HOLE = "hole"            # 透明の穴ができる
CODE_FOOT = "foot"            # 足元が動く
CODE_PART = "small_part"      # 入ってきた部品が目・口などに触れる
CODE_INTRUSION = "intrusion"  # 入ってきた部品が大きすぎる
CODE_TEXT = {
    CODE_LOW: "1 コマ目と合う位置が見つかりません（合い方が低い）",
    CODE_EMPTY: "1 コマ目の範囲の中に不透明なドットがありません",
    CODE_EDGE: "写す中身がキャンバスの外に出ます",
    CODE_SEAM: "範囲の境目で段差・輪郭の途切れ・透明のすき間ができます",
    CODE_HOLE: "写すと透明の穴ができます",
    CODE_FOOT: "写すと足元（いちばん下の行）が動きます",
    CODE_PART: "範囲に入ってきた部品（腕・髪など）が目・口などの小さな部品に触れます",
    CODE_INTRUSION: "範囲に入ってきた部品（腕・髪など）が大きすぎます",
}


@dataclass(frozen=True)
class SameRange:
    """基準のコマの上の範囲（ドット。左上が 0, 0。`w`・`h` は幅と高さ）。`kind` は "rect"（四角）か "band"（帯。横いっぱい）。"""

    x: int
    y: int
    w: int
    h: int
    kind: str = "rect"

    def info(self) -> dict:
        return {"type": self.kind, "x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}

    def text(self) -> str:
        if self.kind == "band":
            return f"帯 {self.y}〜{self.y + self.h - 1} 行"
        return f"四角 x {self.x}・y {self.y}・幅 {self.w}・高さ {self.h}"

    def cli(self) -> str:
        if self.kind == "band":
            return f"--same-band {self.y},{self.y + self.h - 1}"
        return f"--same-as-first {self.x},{self.y},{self.w},{self.h}"


def parse_rect(text: str) -> SameRange:
    """"x,y,w,h"（四角）を読む。"""
    try:
        vals = [int(v) for v in str(text).replace(" ", "").split(",")]
    except ValueError:
        raise ValueError(f"範囲は x,y,幅,高さ のように 4 つの整数で書いてください: {text}") from None
    if len(vals) != 4:
        raise ValueError(f"範囲は x,y,幅,高さ のように 4 つの整数で書いてください: {text}")
    x, y, w, h = vals
    if x < 0 or y < 0 or w < 1 or h < 1:
        raise ValueError(f"範囲の x・y は 0 以上、幅・高さは 1 以上にしてください: {text}")
    return SameRange(x, y, w, h)


def parse_band(text: str, width: int = 0) -> SameRange:
    """"上の行,下の行"（帯。両端を含む）を読む。幅はキャンバスに合わせる（`fit_ranges()` で切り詰める）。"""
    try:
        vals = [int(v) for v in str(text).replace(" ", "").split(",")]
    except ValueError:
        raise ValueError(f"帯は 上の行,下の行 のように 2 つの整数で書いてください: {text}") from None
    if len(vals) != 2:
        raise ValueError(f"帯は 上の行,下の行 のように 2 つの整数で書いてください: {text}")
    y0, y1 = vals
    if y0 < 0 or y1 < y0:
        raise ValueError(f"帯の行は 0 以上で、上の行 ≦ 下の行 にしてください: {text}")
    return SameRange(0, y0, max(int(width), 1), y1 - y0 + 1, "band")


def range_from_dict(d: dict) -> SameRange:
    """JSON の {"type": "rect", "x", "y", "w", "h"} / {"type": "band", "y", "h"} から作る（画面から受け取る形）。"""
    if not isinstance(d, dict):
        raise ValueError(f"範囲は {{\"x\": 0, \"y\": 0, \"w\": 10, \"h\": 10}} のような形で渡してください: {d}")
    unknown = sorted(set(d) - {"type", "x", "y", "w", "h"})
    if unknown:
        raise ValueError(f"知らない範囲の項目です: {', '.join(unknown)}")
    kind = d.get("type", "rect")
    if kind not in ("rect", "band"):
        raise ValueError(f"範囲の type は rect か band です: {kind}")
    vals = []
    for k in ("x", "y", "w", "h"):
        v = d.get(k, 0 if k == "x" else (1 if k == "w" else None))
        if v is None or isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"範囲の {k} は整数で指定してください: {v}")
        vals.append(v)
    x, y, w, h = vals
    if x < 0 or y < 0 or w < 1 or h < 1:
        raise ValueError(f"範囲の x・y は 0 以上、幅・高さは 1 以上にしてください: {d}")
    return SameRange(x, y, w, h, kind)


def fit_ranges(ranges, width: int, height: int) -> list[SameRange]:
    """キャンバスに合わせる（帯は横いっぱい、はみ出しは切り詰める）。キャンバスの外の範囲はエラー。"""
    out = []
    if len(ranges) > MAX_RANGES:
        raise ValueError(f"範囲は {MAX_RANGES} 個までです")
    for r in ranges:
        if r.kind == "band":
            r = SameRange(0, r.y, width, r.h, "band")
        if r.x >= width or r.y >= height:
            raise ValueError(f"範囲（{r.text()}）がコマ（{width}×{height} ドット）の外です")
        out.append(SameRange(r.x, r.y, min(r.w, width - r.x), min(r.h, height - r.y), r.kind))
    return out


def validate_search(search) -> None:
    if isinstance(search, bool) or not isinstance(search, (int, np.integer)) or not MIN_SEARCH <= int(search) <= MAX_SEARCH:
        raise ValueError(f"位置を探す幅は {MIN_SEARCH}〜{MAX_SEARCH} の整数で指定してください: {search}")


@dataclass
class RangeResult:
    """1 コマ・1 範囲の結果。"""

    range_index: int
    shift: tuple                 # 見つけたずれ (dx, dy)
    match: float                 # 合い方（0〜1）
    copied: bool
    code: str | None = None      # 写さなかった理由（写したら None）
    changed: int = 0             # 写して変わったドットの数
    kept: int = 0                # 入ってきた部品として残したドットの数
    seam: list = field(default_factory=list)    # 境目の問題の場所 [[x, y], ...]（コマの座標。範囲の内側のドット）
    holes: list = field(default_factory=list)   # 新しくできる穴の場所 [[x, y], ...]
    foot_warning: bool = False   # 範囲が足元にかかる
    kept_mask: np.ndarray | None = None
    pasted_mask: np.ndarray | None = None
    changed_mask: np.ndarray | None = None

    def info(self) -> dict:
        out = {"range": int(self.range_index), "shift": {"x": int(self.shift[0]), "y": int(self.shift[1])},
               "match": round(float(self.match), 3), "copied": bool(self.copied)}
        if self.code is not None:
            out["code"] = self.code
        out["changed_dots"] = int(self.changed)
        out["kept_dots"] = int(self.kept)
        if self.seam:
            out["seam"] = [[int(x), int(y)] for x, y in self.seam]
        if self.holes:
            out["holes"] = [[int(x), int(y)] for x, y in self.holes]
        if self.foot_warning:
            out["touches_foot"] = True
        return out


@dataclass
class SameFrame:
    frame: int                   # 元のコマの番号（1 から）
    ranges: list[RangeResult]

    @property
    def changed(self) -> int:
        return sum(r.changed for r in self.ranges)


@dataclass
class SameReport:
    reference: int
    search: int
    ranges: list[SameRange]
    frames: list[SameFrame]
    warnings: list[str] = field(default_factory=list)
    suggestions: list[dict] = field(default_factory=list)

    def info(self) -> dict:
        """シートの JSON の `same_as_first` に書く辞書。"""
        out = {"reference_frame": int(self.reference), "search": int(self.search),
               "ranges": [r.info() for r in self.ranges], "frames": []}
        for f in self.frames:
            out["frames"].append({"source_frame": int(f.frame), "ranges": [r.info() for r in f.ranges]})
        if self.suggestions:
            out["suggestions"] = self.suggestions
        return out

    def counts(self) -> list[int]:
        """各コマ（元の並び順。基準のコマは 0）で写して変わったドットの数。"""
        by = {f.frame: f.changed for f in self.frames}
        n = max([self.reference] + [f.frame for f in self.frames])
        return [by.get(i, 0) for i in range(1, n + 1)]

    def copied_count(self) -> tuple[int, int]:
        """(写した 範囲×コマ の数, 全部の数)。"""
        rs = [r for f in self.frames for r in f.ranges]
        return sum(r.copied for r in rs), len(rs)


# ---------------------------------------------------------------- 下ごしらえ

def _keys(img: np.ndarray) -> np.ndarray:
    k = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2].astype(np.int64)
    return np.where(img[..., 3] > 0, k, -1)


def _window(img: np.ndarray, x: int, y: int, w: int, h: int, fill=0) -> np.ndarray:
    """img の (x, y) から w×h を切り出す（キャンバスの外は fill。絵なら透明）。"""
    H, W = img.shape[:2]
    out = np.full((h, w) + img.shape[2:], fill, img.dtype)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 < x1 and y0 < y1:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    m = mask.tolist()
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
                if 0 <= yy < h and 0 <= xx < w and m[yy][xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    q.append((yy, xx))
        out.append(pts)
    return out


def _dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
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
    t = ~opaque
    ext = np.zeros(opaque.shape, bool)
    ext[0], ext[-1], ext[:, 0], ext[:, -1] = t[0], t[-1], t[:, 0], t[:, -1]
    while True:
        p = np.pad(ext, 1)
        nxt = t & (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
        if (nxt == ext).all():
            return ext
        ext = nxt


def _bottom(opaque: np.ndarray) -> int:
    rows = np.flatnonzero(opaque.any(1))
    return int(rows[-1]) if len(rows) else -1


def _classes(img: np.ndarray, line_colors: np.ndarray) -> np.ndarray:
    """ドットの種類: 0 = 透明、1 = 輪郭の色、2 = ほかの色。"""
    k = _keys(img)
    out = np.where(k >= 0, 2, 0).astype(np.int8)
    if len(line_colors):
        out[np.isin(k, line_colors)] = 1
    return out


def _shifts(search: int) -> list[tuple[int, int]]:
    s = [(dx, dy) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]
    return sorted(s, key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[1]), abs(d[0]), d[1], d[0]))


# ---------------------------------------------------------------- 1 コマ・1 範囲

def find_shift(patch: np.ndarray, frame: np.ndarray, r: SameRange, search: int) -> tuple[tuple[int, int], float]:
    """範囲の中身（`patch`）がコマにいちばんよく合うずれと合い方。中身がキャンバスの外に出るずれは使わない。"""
    H, W = frame.shape[:2]
    pk = _keys(patch)
    pop = pk >= 0
    ys, xs = np.nonzero(pop)
    best = None
    for dx, dy in _shifts(search):
        if len(ys) and (xs.min() + r.x + dx < 0 or ys.min() + r.y + dy < 0
                        or xs.max() + r.x + dx >= W or ys.max() + r.y + dy >= H):
            continue
        fk = _keys(_window(frame, r.x + dx, r.y + dy, r.w, r.h))
        union = pop | (fk >= 0)
        n = int(union.sum())
        m = (int(((pk == fk) & union).sum()) / n) if n else 0.0
        if best is None or m > best[1] + 1e-12:
            best = ((dx, dy), m)
    if best is None:
        return (0, 0), 0.0
    return best


def _intruders(patch_keys: np.ndarray, win_keys: np.ndarray, outside_opaque: np.ndarray) -> np.ndarray:
    """範囲の中で、コマにだけある太い部品で、範囲の端から外とつながっているもの（残すドット）。

    `outside_opaque` は窓のまわりに 1 ドットずつ足した (h+2)×(w+2) の、コマの不透明（窓の中は偽）。
    """
    h, w = win_keys.shape
    cand = (win_keys >= 0) & (patch_keys < 0)
    keep = np.zeros((h, w), bool)
    for pts in _components(cand):
        if len(pts) < KEEP_MIN:
            continue
        m = np.zeros((h, w), bool)
        ys = np.array([p[0] for p in pts], np.intp)
        xs = np.array([p[1] for p in pts], np.intp)
        m[ys, xs] = True
        p = np.pad(m, 1)
        thick = (m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]).any()
        if not thick:
            continue
        # 範囲の外の不透明なドットに触れているか（8 方向）
        if not (_dilate(np.pad(m, 1)) & outside_opaque).any():
            continue
        keep |= m
    return keep


def _nb(a: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """各ドットの (dy, dx) 隣の値（キャンバスの外は fill）。"""
    h, w = a.shape
    out = np.full((h, w), fill, a.dtype)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def _seam(new: np.ndarray, src: np.ndarray, old: np.ndarray, pasted: np.ndarray, line_colors: np.ndarray) -> list:
    """写した所（`pasted`。コマの座標の真偽）の境目にできた問題のドット [(x, y), ...]。

    `src` は 1 コマ目を写す位置にずらした絵（写した所の元）、`old` は写す前のコマ。次の 2 つを数える:
    - 輪郭のすき間（段差）: 境目をはさんで上下左右に隣り合う 2 つのドットが「輪郭でない色」と「透明」で、1 コマ目でもコマでも
      その 2 か所がそうではなかった（輪郭が切れて、中の色が外に出た）。印は輪郭でない色の方
    - 輪郭の途切れ: 境目のそばの輪郭のドットが、元の絵（写した側は 1 コマ目、残した側はコマ）では境目の向こう側の輪郭と
      つながっていた（8 方向）のに、写した後は向こう側に輪郭が無い
    """
    return _seam_points(_seam_mask(_classes(new, line_colors), _classes(src, line_colors),
                                   _classes(old, line_colors), pasted))


def _seam_points(m: np.ndarray) -> list:
    ys, xs = np.nonzero(m)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]


def _seam_mask(cn: np.ndarray, cs: np.ndarray, co: np.ndarray, pasted: np.ndarray) -> np.ndarray:
    """`_seam()` の判定を真偽の絵で返す（cn・cs・co は写した後・1 コマ目・コマの種類）。"""
    h, w = pasted.shape
    P = np.pad(pasted, 1, constant_values=False)
    V = np.pad(np.ones((h, w), bool), 1, constant_values=False)
    CN = np.pad(cn, 1, constant_values=-1)
    CS = np.pad(cs, 1, constant_values=-1)
    CO = np.pad(co, 1, constant_values=-1)

    def nb(A, dy, dx):
        return A[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    def gap(a, b):
        return ((a == 2) & (b == 0)) | ((a == 0) & (b == 2))

    out = np.zeros((h, w), bool)
    for dy, dx in _N4:
        across = pasted & nb(V, dy, dx) & ~nb(P, dy, dx)      # p は写した所、q は写していない所
        if not across.any():
            continue
        qn = nb(CN, dy, dx)
        bad = across & gap(cn, qn) & ~gap(cs, nb(CS, dy, dx)) & ~gap(co, nb(CO, dy, dx))
        out |= bad & (cn == 2)
        qbad = np.pad(bad & (qn == 2), 1, constant_values=False)   # 輪郭でない色が q の側なら q に印
        out |= nb(qbad, -dy, -dx)
    had = np.zeros((h, w), bool)
    now = np.zeros((h, w), bool)
    for dy, dx in _N8:
        other = nb(V, dy, dx) & (nb(P, dy, dx) != pasted)
        had |= other & np.where(pasted, nb(CS, dy, dx) == 1, nb(CO, dy, dx) == 1)
        now |= other & (nb(CN, dy, dx) == 1)
    out |= (cn == 1) & had & ~now
    return out


def _cut(cost: np.ndarray, band: int) -> np.ndarray:
    """下の端の継ぎ目（列ごとに、上から何行を写すか k。h−band 〜 h）。cost[y, x] はその行を写した最後の行にしたときの費用。
    隣の列とは 1 行までしか変えない。費用が同じなら多く写す方（外側）。"""
    h, w = cost.shape
    ks = list(range(max(1, h - band), h + 1))
    n = len(ks)
    c = np.array([[cost[k - 1, x] - 1e-6 * k for x in range(w)] for k in ks])   # (n, w)
    acc = c[:, 0].copy()
    back = np.zeros((n, w), np.int64)
    for x in range(1, w):
        nxt = np.empty(n)
        for i in range(n):
            lo, hi = max(0, i - 1), min(n, i + 2)
            j = lo + int(np.argmin(acc[lo:hi]))
            back[i, x] = j
            nxt[i] = acc[j] + c[i, x]
        acc = nxt
    i = int(np.argmin(acc))
    out = np.empty(w, np.int64)
    for x in range(w - 1, -1, -1):
        out[x] = ks[i]
        i = int(back[i, x])
    return out


def _paste_mask(pk: np.ndarray, wk: np.ndarray, re: np.ndarray, fe: np.ndarray, band: int, edges,
                parts=()) -> np.ndarray:
    """窓の中で写す所。範囲の端（`edges` の向き）から `band` ドット以内では、継ぎ目を内側に寄せて、境目の問題
    （`_seam()` と同じ考え方）ができない所を通す。`re`・`fe` は 1 コマ目の範囲・コマの窓のまわりに 1 ドットずつ足した種類。
    費用: 境目の問題 1、1 コマ目とコマの色が違う 0.01、小さな部品を横切る 4。同じなら多く写す方（外側）。"""
    h, w = pk.shape
    diff = (pk != wk).astype(np.float64)
    mask = np.ones((h, w), bool)
    if band <= 0:
        return mask
    for edge in edges:
        # 下の端に回して計算し、戻す（継ぎ目は y 行目と y+1 行目の間。y 行目までを写す）
        rot = {"bottom": 0, "right": 1, "top": 2, "left": 3}[edge]
        R = np.rot90(re, rot)
        F = np.rot90(fe, rot)
        hh, ww = R.shape[0] - 2, R.shape[1] - 2
        if hh < 2:
            continue
        p_here, f_here = R[1:hh + 1, 1:ww + 1], F[1:hh + 1, 1:ww + 1]
        r_down, f_down = R[2:hh + 2, 1:ww + 1], F[2:hh + 2, 1:ww + 1]
        gap = (((p_here == 2) & (f_down == 0)) | ((p_here == 0) & (f_down == 2))) \
            & ~(((p_here == 2) & (r_down == 0)) | ((p_here == 0) & (r_down == 2))) \
            & ~(((f_here == 2) & (f_down == 0)) | ((f_here == 0) & (f_down == 2)))
        r_below = (R[2:hh + 2, 0:ww] == 1) | (R[2:hh + 2, 1:ww + 1] == 1) | (R[2:hh + 2, 2:ww + 2] == 1)
        f_below = (F[2:hh + 2, 0:ww] == 1) | (F[2:hh + 2, 1:ww + 1] == 1) | (F[2:hh + 2, 2:ww + 2] == 1)
        f_above = (F[1:hh + 1, 0:ww] == 1) | (F[1:hh + 1, 1:ww + 1] == 1) | (F[1:hh + 1, 2:ww + 2] == 1)
        p_above = (R[1:hh + 1, 0:ww] == 1) | (R[1:hh + 1, 1:ww + 1] == 1) | (R[1:hh + 1, 2:ww + 2] == 1)
        brk = ((p_here == 1) & r_below & ~f_below) | ((f_down == 1) & f_above & ~p_above)
        cr = gap.astype(np.float64) + brk.astype(np.float64) + 0.01 * np.rot90(diff, rot)
        for lab in parts:
            lr = np.rot90(lab, rot)
            cross = np.zeros(cr.shape, bool)
            cross[:-1] = (lr[:-1] >= 0) & (lr[:-1] == lr[1:])
            cr += 4.0 * cross
        ks = _cut(cr, min(band, hh - 1))
        m = np.arange(hh)[:, None] < ks[None, :]
        mask &= np.rot90(m, -rot)
    return mask


def same_range(ref: np.ndarray, frame: np.ndarray, r: SameRange, index: int, search: int,
               line_colors: np.ndarray, ref_parts: np.ndarray, original: np.ndarray | None = None,
               band: int | None = None, refine: int | None = None):
    """1 コマの 1 つの範囲を 1 コマ目と同じにする。戻り値は (直したコマ, RangeResult)。

    `refine` は境目の問題を減らす入れ替えの回数の上限（既定 `REFINE_ROUNDS`。「次に試せること」の下調べでは少なくする。
    回数が多い方が写せるコマが多いか同じ）。

    `original` は位置を探すのに使うコマ（ほかの範囲を写す前。省くと `frame`）。`band` は継ぎ目を寄せる幅（既定 `SEAM_BAND`）。
    """
    H, W = frame.shape[:2]
    band = SEAM_BAND if band is None else int(band)
    patch = ref[r.y:r.y + r.h, r.x:r.x + r.w]
    pk = _keys(patch)
    ref_bottom = _bottom(ref[..., 3] > 0)
    if not (pk >= 0).any():
        return frame, RangeResult(index, (0, 0), 0.0, False, CODE_EMPTY)
    (dx, dy), match = find_shift(patch, frame if original is None else original, r, search)
    tx, ty = r.x + dx, r.y + dy
    res = RangeResult(index, (dx, dy), match, False)
    fop = frame[..., 3] > 0
    foot = _bottom(fop)
    res.foot_warning = r.y + r.h - 1 >= ref_bottom or ty + r.h - 1 >= foot
    if match < MIN_MATCH:
        res.code = CODE_LOW
        return frame, res
    x0, y0, x1, y1 = max(0, tx), max(0, ty), min(W, tx + r.w), min(H, ty + r.h)
    inside = np.zeros((r.h, r.w), bool)
    inside[y0 - ty:y1 - ty, x0 - tx:x1 - tx] = True
    if ((pk >= 0) & ~inside).any():
        res.code = CODE_EDGE
        return frame, res
    win = _window(frame, tx, ty, r.w, r.h)
    wk = _keys(win)
    around = _window(fop.astype(np.uint8), tx - 1, ty - 1, r.w + 2, r.h + 2) > 0
    around[1:-1, 1:-1] = False
    keep = _intruders(pk, wk, around) & inside
    union = int(((pk >= 0) | (wk >= 0)).sum())
    # 継ぎ目: キャンバスの端でない範囲の端だけ寄せる（帯の左右はキャンバスの端）
    edges = [e for e, on in (("bottom", ty + r.h < H), ("top", ty > 0), ("left", tx > 0), ("right", tx + r.w < W)) if on]
    from .steady import small_parts
    ref_win = ref_parts[r.y:r.y + r.h, r.x:r.x + r.w]
    fparts = _window(small_parts(frame), tx, ty, r.w, r.h, fill=-1)
    fe = _classes(_window(frame, tx - 1, ty - 1, r.w + 2, r.h + 2), line_colors)
    re = _classes(_window(ref, r.x - 1, r.y - 1, r.w + 2, r.h + 2), line_colors)
    mask = _paste_mask(pk, wk, re, fe, band, edges,
                       (ref_win, fparts)) & inside & ~keep
    res.kept = int(keep.sum())
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = mask[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    full_keep = np.zeros((H, W), bool)
    full_keep[y0:y1, x0:x1] = keep[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    res.kept_mask = full_keep
    res.pasted_mask = full
    if res.kept > KEEP_MAX * union:
        res.code = CODE_INTRUSION
        return frame, res
    # 目・口などの小さな部品は、全部写すか全部そのままか。継ぎ目が部品を横切るときは、部品を全部写す側に入れる
    # （1 コマ目の部品は全部 1 コマ目のドットに、コマの部品は全部 1 コマ目のドットで上書き）。入れられないときは写さない
    for _ in range(8):
        grow = np.zeros_like(mask)
        for lab in (ref_win, fparts):
            for v in np.unique(lab[mask]):
                if v < 0:
                    continue
                m = lab == v
                if not mask[m].all():
                    grow |= m
        if not grow.any():
            break
        if (grow & (keep | ~inside)).any():
            res.code = CODE_PART
            return frame, res
        mask |= grow
    else:
        res.code = CODE_PART
        return frame, res
    src = np.zeros_like(ref)
    sy0, sy1, sx0, sx1 = max(0, dy), min(H, H + dy), max(0, dx), min(W, W + dx)
    src[sy0:sy1, sx0:sx1] = ref[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx]
    ext0 = _exterior(fop)
    cs_full, co_full = _classes(src, line_colors), _classes(frame, line_colors)

    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = mask[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
    smask = _seam_mask(np.where(full, cs_full, co_full), cs_full, co_full, full)
    seam = _seam_points(smask)
    # 境目の問題が残るときは、継ぎ目を寄せてよい所（端から band ドット以内）で、問題のドットのまわりのドットを
    # 写す・写さないを 1 つずつ入れ替え、問題がいちばん減るものを選ぶ（減らなくなるまで。小さな部品のドットは動かさない）
    zone = np.zeros((r.h, r.w), bool)
    for e in edges:
        if e == "bottom":
            zone[max(0, r.h - band):] = True
        elif e == "top":
            zone[:band] = True
        elif e == "left":
            zone[:, :band] = True
        else:
            zone[:, max(0, r.w - band):] = True
    zone &= inside & ~keep & (ref_win < 0) & (fparts < 0)
    score = len(seam)
    for _ in range((REFINE_ROUNDS if refine is None else int(refine)) if band > 0 else 0):
        if score == 0:
            break
        pts = set(seam)
        cands = set()
        for x, y in pts:
            for yy in range(y - 1, y + 2):
                for xx in range(x - 1, x + 2):
                    wy, wx = yy - ty, xx - tx
                    if 0 <= wy < r.h and 0 <= wx < r.w and zone[wy, wx]:
                        cands.add((("flip", (wy, wx)),))
            # 列（上下の端）・行（左右の端）ごとに、継ぎ目の位置を端から 0〜band ドットに置き直す
            for e in edges:
                for d in (-1, 0, 1):
                    if e in ("bottom", "top"):
                        wx = x - tx + d
                        if not 0 <= wx < r.w:
                            continue
                        cells = [(wy, wx) for wy in (range(r.h - band, r.h) if e == "bottom" else range(band - 1, -1, -1))
                                 if 0 <= wy < r.h]
                    else:
                        wy = y - ty + d
                        if not 0 <= wy < r.h:
                            continue
                        cells = [(wy, wx) for wx in (range(r.w - band, r.w) if e == "right" else range(band - 1, -1, -1))
                                 if 0 <= wx < r.w]
                    for k in range(len(cells) + 1):
                        cands.add(tuple(("set", c, i < k) for i, c in enumerate(cells)))
        best = None
        for cand in sorted(cands, key=repr):
            cells = []
            for c in cand:
                (wy, wx), v = (c[1], c[2]) if c[0] == "set" else (c[1], not mask[c[1]])
                if zone[wy, wx] and mask[wy, wx] != v:
                    cells.append((wy, wx, v))
            if not cells:
                continue
            # 変わるのは入れ替えたドットのまわり 1 ドットの判定だけ（判定は 1 ドット先までしか見ない）
            cy0 = max(0, min(c[0] for c in cells) + ty - 1)
            cy1 = min(H, max(c[0] for c in cells) + ty + 2)
            cx0 = max(0, min(c[1] for c in cells) + tx - 1)
            cx1 = min(W, max(c[1] for c in cells) + tx + 2)
            ey0, ey1, ex0, ex1 = max(0, cy0 - 1), min(H, cy1 + 1), max(0, cx0 - 1), min(W, cx1 + 1)
            f2 = full[ey0:ey1, ex0:ex1].copy()
            for wy, wx, v in cells:
                f2[wy + ty - ey0, wx + tx - ex0] = v
            cs_c, co_c = cs_full[ey0:ey1, ex0:ex1], co_full[ey0:ey1, ex0:ex1]
            loc = _seam_mask(np.where(f2, cs_c, co_c), cs_c, co_c, f2)
            # 切り出しの端（キャンバスの端でない所）の 1 ドットは正しくないので、内側だけを比べる
            inner = loc[cy0 - ey0:cy1 - ey0, cx0 - ex0:cx1 - ex0]
            sc = score - int(smask[cy0:cy1, cx0:cx1].sum()) + int(inner.sum())
            if sc < score and (best is None or sc < best[0]):
                best = (sc, cells, (cy0, cy1, cx0, cx1), inner)
        if best is None:
            break
        score, cells, (cy0, cy1, cx0, cx1), inner = best
        for wy, wx, v in cells:
            mask[wy, wx] = v
            full[wy + ty, wx + tx] = v
        smask[cy0:cy1, cx0:cx1] = inner
        seam = _seam_points(smask)
    res.pasted_mask = full
    new = frame.copy()
    new[full] = src[full]
    holes = (~(new[..., 3] > 0)) & ~_exterior(new[..., 3] > 0) & ext0
    nop = new[..., 3] > 0
    changed = (_keys(new) != _keys(frame))
    res.changed = int(changed.sum())
    res.changed_mask = changed
    res.seam = seam
    res.holes = [(int(x), int(y)) for y, x in zip(*np.nonzero(holes))]
    if _bottom(nop) != foot:
        res.code = CODE_FOOT
    elif res.holes:
        res.code = CODE_HOLE
    elif res.seam:
        res.code = CODE_SEAM
    if res.code is not None:
        res.changed = 0
        return frame, res
    res.copied = True
    return new, res


def _one_frame(ref, frame, ranges, search, line_colors, ref_parts, number):
    out = frame
    results = []
    for i, r in enumerate(ranges):
        out, rr = same_range(ref, out, r, i + 1, search, line_colors, ref_parts, original=frame)
        results.append(rr)
    return out, SameFrame(number, results)


def _suggest(ref, frames, ranges, search, reference, line_colors, ref_parts, report) -> list[dict]:
    """写せなかった範囲に、次に試せることを添える。端をずらした範囲は実際に試し、写せるコマが増えるものだけ。"""
    H, W = ref.shape[:2]
    out = []
    for i, r in enumerate(ranges):
        rs = [f.ranges[i] for f in report.frames]
        failed = [x for x in rs if not x.copied]
        if not failed:
            continue
        codes = sorted({x.code for x in failed})
        tips = []
        if CODE_LOW in codes:
            tips.append("範囲を顔・冠など形の変わらない所に小さくする")
            if search < MAX_SEARCH:
                tips.append(f"位置を探す幅を広げる（--same-search {min(MAX_SEARCH, search + 2)}）")
        if CODE_EDGE in codes:
            tips.append("範囲をキャンバスの端から離す")
        if CODE_FOOT in codes:
            tips.append("範囲の下の端を足元より上にする")
        if CODE_PART in codes or CODE_INTRUSION in codes:
            tips.append("腕・髪などが入ってくる所を範囲から外す（範囲を小さくする・2 つに分ける）")
        if CODE_SEAM in codes or CODE_HOLE in codes:
            tips.append("範囲の端を、透明な所か、コマごとに形が変わらない所に動かす")
        base = sum(x.copied for x in rs)
        near = []
        tries = 0
        for edge in ("bottom", "top", "left", "right"):
            if r.kind == "band" and edge in ("left", "right"):
                continue
            for s in NEAR_STEPS:
                x, y, w, h = r.x, r.y, r.w, r.h
                if edge == "bottom":
                    h += s
                elif edge == "top":
                    y -= s
                    h += s
                elif edge == "left":
                    x -= s
                    w += s
                else:
                    w += s
                if x < 0 or y < 0 or w < 2 or h < 2 or x + w > W or y + h > H:
                    continue
                tries += 1
                if tries > NEAR_TRIES:
                    break
                cand = SameRange(x, y, w, h, r.kind)
                # 写せなかったコマを先に試し、1 つも写せなければほかのコマは試さない
                order = [k for k, x_ in enumerate(rs) if not x_.copied] + [k for k, x_ in enumerate(rs) if x_.copied]
                ok = 0
                for j, k in enumerate(order):
                    f = frames[report.frames[k].frame - 1]
                    _, rr = same_range(ref, f, cand, i + 1, search, line_colors, ref_parts, refine=SUGGEST_ROUNDS)
                    ok += rr.copied
                    if j == len(failed) - 1 and ok == 0:
                        break
                if ok > base:
                    near.append((-ok, abs(s), cand))
        near.sort(key=lambda t: (t[0], t[1]))
        entry = {"range": i + 1, "codes": codes, "tips": tips,
                 "near": [dict(c.info(), copied_frames_at_least=-k, cli=c.cli()) for k, _, c in near[:NEAR_COUNT]]}
        out.append(entry)
    return out


def same_frames(frames: list[np.ndarray], ranges: list[SameRange], search: int = DEFAULT_SEARCH,
                reference: int = 1, suggest: bool = True) -> tuple[list[np.ndarray], SameReport]:
    """そろえたコマ（同じ大きさ）の、範囲の中身を基準のコマと同じにする。戻り値は (コマ, SameReport)。"""
    from .steady import outline_keys, small_parts
    validate_search(search)
    if not frames:
        raise ValueError("コマがありません")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("コマの大きさがそろっていません")
    if not 1 <= int(reference) <= len(frames):
        raise ValueError(f"元にするコマは 1〜{len(frames)} にしてください: {reference}")
    H, W = frames[0].shape[:2]
    ranges = fit_ranges(ranges, W, H)
    ref = frames[reference - 1]
    line_colors = outline_keys(ref)
    ref_parts = small_parts(ref)
    out, rep = [], []
    for i, f in enumerate(frames):
        if i == reference - 1:
            out.append(f.copy())
            continue
        g, sf = _one_frame(ref, f, ranges, int(search), line_colors, ref_parts, i + 1)
        out.append(g.copy())
        rep.append(sf)
    report = SameReport(int(reference), int(search), ranges, rep)
    # 念のため: 新しい色を作っていない（基準のコマかそのコマにある色だけ）
    ref_keys = set(np.unique(_keys(ref)).tolist())
    for f, g in zip(frames, out):
        if not set(np.unique(_keys(g)).tolist()) <= ref_keys | set(np.unique(_keys(f)).tolist()):
            raise RuntimeError("範囲を写す段で新しい色ができました（不具合）")
    # 警告
    for sf in rep:
        for rr in sf.ranges:
            r = ranges[rr.range_index - 1]
            head = f"{sf.frame} コマ目・範囲 {rr.range_index}（{r.text()}）"
            if not rr.copied:
                extra = ""
                if rr.code == CODE_SEAM:
                    extra = "（" + "、".join(f"({x}, {y})" for x, y in rr.seam[:6]) + (" ほか" if len(rr.seam) > 6 else "") + "）"
                elif rr.code == CODE_HOLE:
                    extra = f"（{len(rr.holes)} ドット）"
                elif rr.code == CODE_LOW:
                    extra = f"（合い方 {rr.match * 100:.0f}%、ずれ {rr.shift[0]:+d},{rr.shift[1]:+d}）"
                report.warnings.append(f"{head}: 写しませんでした。{CODE_TEXT[rr.code]}{extra}")
            elif rr.foot_warning:
                report.warnings.append(f"{head}: 範囲が足元（いちばん下の行）にかかっています（足元は動いていません）")
    if suggest and any(not rr.copied for sf in rep for rr in sf.ranges):
        report.suggestions = _suggest(ref, frames, ranges, int(search), int(reference), line_colors, ref_parts, report)
        for s in report.suggestions:
            parts = list(s["tips"])
            if s["near"]:
                parts.append("写せるコマが増える範囲: " + "、".join(f"{n['cli']}（{n['copied_frames_at_least']} コマ以上）" for n in s["near"]))
            if parts:
                report.warnings.append(f"範囲 {s['range']} で次に試せること: " + "／".join(parts))
    return out, report


# ---------------------------------------------------------------- 図

def same_image(before: list[np.ndarray], after: list[np.ndarray], report: SameReport, scale: int = 8) -> np.ndarray:
    """範囲の図（RGBA）。写した後の各コマを scale 倍にし、範囲を写した位置の枠（緑 = 写した、赤 = 写さなかった、
    基準のコマは水色）、橙 = 境目の問題の所・穴、青 = 入ってきた部品として残したドット。上に「#コマ x ずれ,y ずれ 合い方%」。"""
    from PIL import Image, ImageDraw

    from .frames import _draw_label
    h, w = after[0].shape[:2]
    gap = scale
    n = len(after)
    nr = len(report.ranges)
    top = 4 + 12 * max(1, nr)
    W = n * w * scale + (n + 1) * gap
    Hh = h * scale + gap + top
    canvas = Image.new("RGBA", (W, Hh), (200, 200, 200, 255))
    draw = ImageDraw.Draw(canvas)
    by = {f.frame: f for f in report.frames}
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where((((yy + xx) % 2) == 1)[..., None], 222, 236).astype(np.float64)
    lw = max(1, scale // 4)
    for i, f in enumerate(after):
        x0, y0 = gap + i * (w * scale + gap), top
        a = f[..., 3:4].astype(np.float64) / 255
        img = f[..., :3] * a + bg * (1 - a)
        canvas.paste(Image.fromarray(img.astype(np.uint8), "RGB").resize((w * scale, h * scale), Image.NEAREST), (x0, y0))
        sf = by.get(i + 1)
        if sf is None:
            for k, r in enumerate(report.ranges):
                draw.rectangle([x0 + r.x * scale, y0 + r.y * scale, x0 + (r.x + r.w) * scale - 1,
                                y0 + (r.y + r.h) * scale - 1], outline=(0, 170, 255, 255), width=lw)
            _draw_label(draw, x0, 2, f"#{i + 1} (ref)")
            continue
        for k, rr in enumerate(sf.ranges):
            r = report.ranges[k]
            if rr.kept_mask is not None and rr.copied:
                for y, x in zip(*np.nonzero(rr.kept_mask)):
                    draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                                   outline=(40, 90, 255, 255), width=lw)
            for x, y in list(rr.seam) + list(rr.holes):
                draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                               fill=(255, 140, 0, 160), outline=(255, 100, 0, 255), width=lw)
            col = (0, 190, 60, 255) if rr.copied else (255, 30, 30, 255)
            tx, ty = r.x + rr.shift[0], r.y + rr.shift[1]
            draw.rectangle([x0 + tx * scale, y0 + ty * scale, x0 + (tx + r.w) * scale - 1, y0 + (ty + r.h) * scale - 1],
                           outline=col, width=lw)
            _draw_label(draw, x0, 2 + 12 * k, f"#{i + 1} {rr.shift[0]:+d},{rr.shift[1]:+d} {int(round(rr.match * 100))}%"
                        + ("" if rr.copied else " x"), fill=col[:3] + (255,) if not rr.copied else (0, 0, 0, 255))
    return np.asarray(canvas)
