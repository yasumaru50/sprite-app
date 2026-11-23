"""差し替え（まばたきなど）。目を閉じた顔を 1 回描くだけで、動きのシートの中に自然なまばたきを入れる。

ブラウザ版の「ドットを直す」欄の「差し替えを描く」と、CLI の `--swaps` で共通の処理。

差し替えは「1 コマ目（元にする絵）の上に描いた別の形」で、1 コマ目と違うドットだけを持つ（`touchup.Dot` と同じ形。色はシートの
パレットの 1 色か透明）。範囲は違うドットの離れた固まりごとの四角（`rects`。左右の目は別々の四角）で、自動で決まる。
種類（`type`）ごとに形の名前が決まっている:

- `blink`（まばたき）: `closed`（目を閉じた絵。いる）・`half`（半分閉じた絵。なくてもよい）。
  あれば 開く → 半分 → 閉じる → 半分 → 開く、無ければ 開く → 閉じる → 開く

口パクなど別の種類は、`TYPES` に形の名前と並べ方を足せば同じ仕組みで使える（今は blink だけ）。

当て方（`apply_swaps()`。どれも迷ったら当てない）:

1. **何周にするか**: 動き 1 周の時間がまばたきの間隔（既定 3000 ミリ秒）より短いときは、動きを何周か繰り返したシートにして、
   その中に 1 回だけまばたきを入れる（周の数 = 間隔 ÷ 1 周の時間 を四捨五入。1 以上、フレームが `MAX_FRAMES` を超えない数）
2. **位置を探す**: フレームごとに、差し替えの四角のまわり（`MARGIN` ドット広げた所。四角が何個あっても 1 つのずれ）の 1 コマ目の絵が
   いちばんよく合うずれを、元の位置（足した余白と体全体のずれの分を足した所）から上下左右 `search` ドット以内で探す（`same.py` と同じ合い方）
3. **確かめる**（どれかに引っかかったフレームには当てない。迷ったら入れない）:
   - 合い方が `MIN_MATCH` 未満（`low_match`）・キャンバスの外に出る（`edge`）・足元が変わる（`foot`）
   - 四角の境目をまたいで、1 コマ目と違う部品がある（`crossing`。コマごとに動く前髪・手など。置き換えると四角の中だけ 1 コマ目の形に戻る）
   - 外から入ってきた太い部品がある（`intrusion`。`same.py` の「入ってきた部品」と同じ見つけ方）
   - 置き換えると四角の境目で段差・輪郭の途切れができる（`seam`。`same.py` と同じ確かめ）
4. **時間を分ける**: まばたきの時間（閉じる `close_ms` 既定 120、半分 `half_ms` 既定 60 を前後に）を、シートの真ん中あたりの
   フレームの始まりから入れる。かかるフレームは時間で 2 つ（か 3 つ以上）に分ける（全体の時間は変えない）。`MIN_PIECE` ミリ秒より
   短い切れ端は隣の形に足す。かかるフレームのどれかに当てられないときは、次のフレームから試す
5. **当てる**: 四角を、描いた形（1 コマ目に違うドットを重ねたもの）でそのまま置き換える（目の形がコマごとに少し違っても
   残らないように）。新しい色は作らない（形の色はパレットにあることを確かめる）。大きさ・足元は変えない

差し替えの一覧のファイル（`<出力名>_<名前>_swaps.json`）の形:

    {"version": 1, "kind": "sprite_swaps", "name": "frames", "frame_width": 66, "frame_height": 92,
     "palette": ["#rrggbb", ...], "interval_ms": 3000, "close_ms": 120, "half_ms": 60, "search": 3,
     "swaps": [{"type": "blink", "rects": [{"x": 20, "y": 38, "w": 6, "h": 4}, {"x": 40, "y": 38, "w": 6, "h": 4}],
                "closed": [[x, y, 色の番号 か null（透明）, 1 コマ目の色の番号 か null], ...], "half": [...]}]}

`rects` は書き出すときだけ付ける（読むときは無視して、ドットから決め直す）。シートの JSON には、差し替えたときだけ
`swaps`（間隔・1 周の時間・周の数・入れた時刻とフレーム・コマごとの位置と合い方）と、差し替えたフレームの `frames[].swap` を足す。
差し替えが無いときの出力は、今までとバイト単位で同じ。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .touchup import TRANSPARENT, Dot

VERSION = 1
KIND = "sprite_swaps"
TYPES = {"blink": ("closed", "half")}          # 種類 → 形の名前（最初の形はいる）
TYPE_TEXT = {"blink": "まばたき"}
SHAPE_TEXT = {"closed": "閉じた目", "half": "半分閉じた目"}
DEFAULT_INTERVAL = 3000
DEFAULT_CLOSE = 120
DEFAULT_HALF = 60
DEFAULT_SEARCH = 3
MIN_INTERVAL, MAX_INTERVAL = 500, 20000
MIN_CLOSE, MAX_CLOSE = 20, 1000
MIN_SEARCH, MAX_SEARCH = 0, 8
MIN_MATCH = 0.7          # これ未満の合い方のフレームには当てない
MARGIN = 2               # 位置を探すとき・違う部品を確かめるときに、四角のまわりを何ドット広げて見るか
CLUSTER_GAP = 2          # 違うドットどうしがこのドット数以内なら、同じ固まり（同じ四角）にする
MIN_PIECE = 20           # これより短い切れ端は作らない（多くのブラウザは 20 ミリ秒未満の GIF を 100 ミリ秒にする）
MAX_FRAMES = 256         # 繰り返したシートのフレームの数の上限
MAX_DOTS = 200_000

CODE_SIZE = "size"
CODE_PALETTE = "palette"
CODE_OUTSIDE = "outside"
CODE_EMPTY = "empty"
CODE_TEXT = {
    CODE_SIZE: "1 コマ目の大きさが差し替えを描いたときと違う",
    CODE_PALETTE: "差し替えに使った色が今のパレットに無い",
    CODE_OUTSIDE: "差し替えのドットがコマの外にある",
    CODE_EMPTY: "差し替えに 1 コマ目と違うドットがありません",
}
# フレームに当てなかった理由（シートの JSON の swaps.items[].frames_not_used[].code）
SKIP_LOW = "low_match"
SKIP_EDGE = "edge"
SKIP_FOOT = "foot"
SKIP_CROSS = "crossing"
SKIP_INTRUSION = "intrusion"
SKIP_SEAM = "seam"
SKIP_TEXT = {SKIP_LOW: "1 コマ目と合う位置が見つからない", SKIP_EDGE: "キャンバスの外に出る", SKIP_FOOT: "足元が変わる",
             SKIP_CROSS: "四角の境目をまたいで 1 コマ目と違う部品（前髪・手など）がある",
             SKIP_INTRUSION: "外から入ってきた部品（腕・髪など）がある",
             SKIP_SEAM: "置き換えると四角の境目で段差・輪郭の途切れができる"}

_HEX = re.compile(r"#[0-9a-f]{6}")


class SwapError(ValueError):
    """差し替えを当てられない（大きさ・パレットが変わったときなど）。`code` は CODE_TEXT のどれか。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Swap:
    """1 つの差し替え。`shapes` は 形の名前 → 1 コマ目と違うドットの並び（(y, x) の順。同じ場所は 1 つ）。"""

    type: str
    shapes: dict[str, list[Dot]] = field(default_factory=dict)

    def rect(self) -> tuple[int, int, int, int] | None:
        """全部の形の違うドットを囲む四角 (x, y, 幅, 高さ)。"""
        pts = [(d.x, d.y) for v in self.shapes.values() for d in v]
        if not pts:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1

    def rects(self) -> list[tuple[int, int, int, int]]:
        """違うドット（全部の形）を、離れた固まりごとに囲んだ四角 (x, y, 幅, 高さ) の並び（上から、左から）。
        `CLUSTER_GAP` ドット以内のドットは同じ固まりにし、重なる・接する四角は 1 つにまとめる。"""
        pts = sorted({(d.x, d.y) for v in self.shapes.values() for d in v})
        if not pts:
            return []
        # 近いドットどうしをつなぐ（ドットの数は多くても数百なので、素直に数える）
        parent = list(range(len(pts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, (x0, y0) in enumerate(pts):
            for j in range(i + 1, len(pts)):
                x1, y1 = pts[j]
                if abs(x1 - x0) <= CLUSTER_GAP and abs(y1 - y0) <= CLUSTER_GAP:
                    parent[find(j)] = find(i)
        groups: dict[int, list] = {}
        for i, p in enumerate(pts):
            groups.setdefault(find(i), []).append(p)
        boxes = []
        for g in groups.values():
            xs, ys = [p[0] for p in g], [p[1] for p in g]
            boxes.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])
        merged = True
        while merged:   # 重なる・接する四角をまとめる
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        return [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in sorted(boxes, key=lambda b: (b[1], b[0]))]

    def sequence(self, close_ms: int, half_ms: int) -> list[tuple[str, int]]:
        """入れる形と時間の並び（まばたき: 半分 → 閉じる → 半分。半分が無ければ閉じるだけ）。"""
        if self.shapes.get("half"):
            return [("half", half_ms), ("closed", close_ms), ("half", half_ms)]
        return [("closed", close_ms)]


@dataclass
class SwapSet:
    """差し替えの一覧と時間の設定。"""

    swaps: list[Swap] = field(default_factory=list)
    frame_width: int | None = None
    frame_height: int | None = None
    name: str = "frames"
    interval_ms: int = DEFAULT_INTERVAL
    close_ms: int = DEFAULT_CLOSE
    half_ms: int = DEFAULT_HALF
    search: int = DEFAULT_SEARCH

    def empty(self) -> bool:
        return not any(d for s in self.swaps for d in s.shapes.values())

    def validate(self) -> None:
        for k, v, lo, hi in (("interval_ms", self.interval_ms, MIN_INTERVAL, MAX_INTERVAL),
                             ("close_ms", self.close_ms, MIN_CLOSE, MAX_CLOSE),
                             ("half_ms", self.half_ms, MIN_CLOSE, MAX_CLOSE),
                             ("search", self.search, MIN_SEARCH, MAX_SEARCH)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= int(v) <= hi:
                raise ValueError(f"{k} は {lo}〜{hi} の整数で書いてください: {v!r}")


# ---------------------------------------------------------------- 読み書き

def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{what}は整数で書いてください: {v!r}")
    return int(v)


def _hex(v, what: str) -> str:
    if not isinstance(v, str) or not _HEX.fullmatch(v.lower()):
        raise ValueError(f"{what}は \"#rrggbb\" の形で書いてください: {v!r}")
    return v.lower()


def swaps_from_dict(d) -> SwapSet:
    """差し替えの一覧のファイルの辞書（上の形）を読む。形が違えば ValueError。"""
    if not isinstance(d, dict):
        raise ValueError("差し替えの一覧は {\"version\": 1, \"swaps\": [...]} の形で書いてください")
    if d.get("kind", KIND) != KIND:
        raise ValueError(f"差し替えの一覧のファイルではありません（kind が {d.get('kind')!r}）")
    if _int(d.get("version", VERSION), "version") > VERSION:
        raise ValueError(f"新しい版（{d.get('version')}）の差し替えの一覧です。Sprite App を新しくしてください")
    unknown = sorted(set(d) - {"version", "kind", "name", "frame_width", "frame_height", "palette", "interval_ms",
                               "close_ms", "half_ms", "search", "swaps"})
    if unknown:
        raise ValueError(f"差し替えの一覧に知らない項目があります: {', '.join(unknown)}")
    pal_raw = d.get("palette", [])
    if not isinstance(pal_raw, list):
        raise ValueError("palette は色（\"#rrggbb\"）の並びで書いてください")
    palette = [_hex(c, "パレットの色") for c in pal_raw]
    size = {}
    for k in ("frame_width", "frame_height"):
        if d.get(k) is not None:
            size[k] = _int(d[k], k)
            if size[k] < 1:
                raise ValueError(f"{k} は 1 以上で書いてください: {size[k]}")
    name = d.get("name", "frames")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"name は半角の英数字と _ - で書いてください: {name!r}")

    def color(v, what):
        if v is None:
            return None
        i = _int(v, what)
        if not 0 <= i < len(palette):
            raise ValueError(f"{what}の番号 {i} がパレット（{len(palette)} 色）にありません")
        return palette[i]

    raw = d.get("swaps", [])
    if not isinstance(raw, list):
        raise ValueError("swaps は [{\"type\": \"blink\", \"closed\": [...]}] の並びで書いてください")
    swaps: list[Swap] = []
    total = 0
    for e in raw:
        if not isinstance(e, dict):
            raise ValueError("swaps の中は {\"type\": \"blink\", \"closed\": [[x, y, 色の番号], ...]} の形で書いてください")
        kind = e.get("type")
        if kind not in TYPES:
            raise ValueError(f"差し替えの種類は {' / '.join(TYPES)} のどれかです: {kind!r}")
        if any(s.type == kind for s in swaps):
            raise ValueError(f"同じ種類（{kind}）の差し替えは 1 つだけにしてください")
        names = TYPES[kind]
        bad = sorted(set(e) - {"type", "rect", "rects"} - set(names))
        if bad:
            raise ValueError(f"{kind} の差し替えに知らない項目があります: {', '.join(bad)}（形は {' / '.join(names)}）")
        shapes = {}
        for sn in names:
            dots = e.get(sn)
            if dots is None:
                continue
            if not isinstance(dots, list):
                raise ValueError(f"{sn} は [[x, y, 色の番号], ...] の並びで書いてください")
            m: dict[tuple[int, int], Dot] = {}
            for v in dots:
                if not isinstance(v, list) or len(v) not in (3, 4):
                    raise ValueError(f"ドットは [x, y, 色の番号 か null] か [x, y, 色, 1 コマ目の色] で書いてください: {v!r}")
                x, y = _int(v[0], "x"), _int(v[1], "y")
                if x < 0 or y < 0:
                    raise ValueError(f"ドットの位置は 0 以上で書いてください: {x}, {y}")
                was = None
                if len(v) == 4:
                    was = TRANSPARENT if v[3] is None else color(v[3], "1 コマ目の色")
                m[(y, x)] = Dot(x, y, color(v[2], "色"), was)
            total += len(dots)
            if total > MAX_DOTS:
                raise ValueError(f"差し替えのドットが多すぎます（{MAX_DOTS} 個まで）")
            if m:
                shapes[sn] = [m[p] for p in sorted(m)]
        if shapes and names[0] not in shapes:
            raise ValueError(f"{kind} の差し替えには {names[0]}（{SHAPE_TEXT.get(names[0], names[0])}）がいります")
        if shapes:
            swaps.append(Swap(kind, shapes))
    out = SwapSet(swaps=swaps, frame_width=size.get("frame_width"), frame_height=size.get("frame_height"), name=name,
                  interval_ms=d.get("interval_ms", DEFAULT_INTERVAL), close_ms=d.get("close_ms", DEFAULT_CLOSE),
                  half_ms=d.get("half_ms", DEFAULT_HALF), search=d.get("search", DEFAULT_SEARCH))
    out.validate()
    out.interval_ms, out.close_ms, out.half_ms, out.search = (int(out.interval_ms), int(out.close_ms),
                                                              int(out.half_ms), int(out.search))
    return out


def load_swaps(text: str) -> SwapSet:
    try:
        d = json.loads(text)
    except ValueError as e:
        raise ValueError(f"差し替えの一覧の JSON を読めません: {e}") from None
    return swaps_from_dict(d)


def swaps_dict(s: SwapSet, palette: list[str]) -> dict:
    """差し替えの一覧のファイルの辞書（`palette` は番号の元。足りない色は後ろに足す）。"""
    pal = list(palette)
    for sw in s.swaps:
        for dots in sw.shapes.values():
            for d in dots:
                for c in (d.color, d.was):
                    if c is not None and c != TRANSPARENT and c not in pal:
                        pal.append(c)
    index = {c: i for i, c in enumerate(pal)}

    def dot(d: Dot) -> list:
        v = [int(d.x), int(d.y), None if d.color is None else index[d.color]]
        if d.was is not None:
            v.append(None if d.was == TRANSPARENT else index[d.was])
        return v

    items = []
    for sw in s.swaps:
        e: dict = {"type": sw.type}
        rs = sw.rects()
        if rs:
            e["rects"] = [{"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for r in rs]
        for sn in TYPES[sw.type]:
            if sw.shapes.get(sn):
                e[sn] = [dot(d) for d in sw.shapes[sn]]
        items.append(e)
    return {"version": VERSION, "kind": KIND, "name": s.name, "frame_width": s.frame_width,
            "frame_height": s.frame_height, "palette": pal, "interval_ms": int(s.interval_ms),
            "close_ms": int(s.close_ms), "half_ms": int(s.half_ms), "search": int(s.search), "swaps": items}


def swaps_json(s: SwapSet, palette: list[str]) -> str:
    """差し替えの一覧のファイルの中身（1 つのドットを 1 行に書く。差分が見やすいように）。"""
    d = swaps_dict(s, palette)
    lines = ["{"]
    for k, v in d.items():
        if k != "swaps":
            lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "swaps": [')
    for i, e in enumerate(d["swaps"]):
        lines.append("    {" + f'"type": {json.dumps(e["type"])}' + (f', "rects": {json.dumps(e["rects"])}' if "rects" in e else "") + ",")
        shapes = [sn for sn in TYPES[e["type"]] if sn in e]
        for j, sn in enumerate(shapes):
            lines.append(f'     {json.dumps(sn)}: [')
            for k, v in enumerate(e[sn]):
                lines.append("       " + json.dumps(v) + ("," if k + 1 < len(e[sn]) else ""))
            lines.append("     ]" + ("," if j + 1 < len(shapes) else ""))
        lines.append("    }" + ("," if i + 1 < len(d["swaps"]) else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def swaps_file_name(stem: str, name: str) -> str:
    """差し替えの一覧のファイルの名前（`<出力名>_<名前>_swaps.json`）。"""
    return f"{stem}_{name}_swaps.json"


def swap_from_image(base: np.ndarray, drawn: dict[str, np.ndarray], kind: str = "blink") -> Swap:
    """1 コマ目（`base`）と、描いた形の絵（形の名前 → 同じ大きさの RGBA）から、違うドットだけの差し替えを作る。"""
    shapes = {}
    for sn, img in drawn.items():
        if img.shape != base.shape:
            raise SwapError(CODE_SIZE, f"{CODE_TEXT[CODE_SIZE]}（{base.shape[1]}×{base.shape[0]} と "
                                       f"{img.shape[1]}×{img.shape[0]}）")
        a, b = _keys(base), _keys(img)
        ys, xs = np.nonzero(a != b)
        dots = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            dots.append(Dot(x, y, None if b[y, x] < 0 else _hex_of(img[y, x]), _under(base, x, y)))
        if dots:
            shapes[sn] = dots
    return Swap(kind, shapes)


# ---------------------------------------------------------------- 下ごしらえ

def _keys(img: np.ndarray) -> np.ndarray:
    k = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2].astype(np.int64)
    return np.where(img[..., 3] > 0, k, -1)


def _hex_of(p) -> str:
    return "#%02x%02x%02x" % (int(p[0]), int(p[1]), int(p[2]))


def _under(img: np.ndarray, x: int, y: int) -> str:
    return TRANSPARENT if img[y, x, 3] == 0 else _hex_of(img[y, x])


def _bottom(img: np.ndarray) -> int:
    rows = np.flatnonzero((img[..., 3] > 0).any(1))
    return int(rows[-1]) if len(rows) else -1


def _window(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 < x1 and y0 < y1:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = img[y0:y1, x0:x1]
    return out


def _shifts(search: int) -> list[tuple[int, int]]:
    s = [(dx, dy) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]
    return sorted(s, key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[1]), abs(d[0]), d[1], d[0]))


def _shape_patch(ref: np.ndarray, dots: list[Dot], rect: tuple[int, int, int, int]) -> np.ndarray:
    """範囲の四角の中の、1 コマ目に形のドットを重ねた絵。"""
    x, y, w, h = rect
    patch = ref[y:y + h, x:x + w].copy()
    for d in dots:
        if d.color is None:
            patch[d.y - y, d.x - x] = (0, 0, 0, 0)
        else:
            c = d.color
            patch[d.y - y, d.x - x] = (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)
    return patch


def check_swaps(ref: np.ndarray, palette: list[str], s: SwapSet) -> list[str]:
    """当てられるかを確かめる。当てられなければ SwapError。戻り値は警告（1 コマ目の色が描いたときと違うドットの数）。"""
    h, w = ref.shape[:2]
    if (s.frame_width is not None and s.frame_width != w) or (s.frame_height is not None and s.frame_height != h):
        raise SwapError(CODE_SIZE, f"{CODE_TEXT[CODE_SIZE]}（描いたとき {s.frame_width}×{s.frame_height}、いま {w}×{h} ドット）")
    if s.empty():
        raise SwapError(CODE_EMPTY, CODE_TEXT[CODE_EMPTY])
    have = set(palette)
    missing = sorted({d.color for sw in s.swaps for v in sw.shapes.values() for d in v if d.color is not None} - have)
    if missing:
        raise SwapError(CODE_PALETTE, f"{CODE_TEXT[CODE_PALETTE]}（{', '.join(missing)}）")
    changed = 0
    for sw in s.swaps:
        for sn, v in sw.shapes.items():
            for d in v:
                if not (0 <= d.x < w and 0 <= d.y < h):
                    raise SwapError(CODE_OUTSIDE, f"{CODE_TEXT[CODE_OUTSIDE]}（{sn} の {d.x}, {d.y}。コマは {w}×{h}）")
                if d.was is not None and d.was != _under(ref, d.x, d.y):
                    changed += 1
    if changed:
        return [f"差し替えのドットのうち {changed} 個は、1 コマ目の色が描いたときと違います（前の段を変えたため）。"
                "差し替えはそのまま当てています。絵を見て確かめてください"]
    return []


# ---------------------------------------------------------------- 当てる

@dataclass
class Placement:
    """1 フレームでの差し替えの位置。"""

    frame: int            # 元のシートのフレームの番号（0 から）
    shift: tuple[int, int]
    match: float
    ok: bool
    code: str | None = None
    images: dict = field(default_factory=dict)   # 形の名前 → 当てたフレームの絵

    def info(self) -> dict:
        out = {"frame": int(self.frame), "shift": {"x": int(self.shift[0]), "y": int(self.shift[1])},
               "match": round(float(self.match), 3), "used": bool(self.ok)}
        if self.code is not None:
            out["code"] = self.code
        return out


def _crossing(ref: np.ndarray, frame: np.ndarray, rect, at) -> bool:
    """四角の境目をまたいで、1 コマ目と違う部品があるか（四角の中と、まわり `MARGIN` ドットに続く違うドットの固まり）。
    置き換えると、その部品の四角の中の分だけが 1 コマ目の形に戻ってしまう（コマごとに動く前髪・手など）。"""
    from .same import _components
    rx, ry, rw, rh = rect
    tx, ty = at
    w, h = rw + 2 * MARGIN, rh + 2 * MARGIN
    a = _keys(_window(ref, rx - MARGIN, ry - MARGIN, w, h))
    b = _keys(_window(frame, tx - MARGIN, ty - MARGIN, w, h))
    diff = a != b
    if not diff.any():
        return False
    inside = np.zeros((h, w), bool)
    inside[MARGIN:MARGIN + rh, MARGIN:MARGIN + rw] = True
    for pts in _components(diff):
        flags = {bool(inside[y, x]) for y, x in pts}
        if len(flags) == 2:
            return True
    return False


def _intrusion(ref: np.ndarray, frame: np.ndarray, rect, at) -> bool:
    """四角の中に、外から入ってきた太い部品があるか（`same.py` の「入ってきた部品」と同じ見つけ方）。"""
    from .same import _intruders, _window as window_any
    rx, ry, rw, rh = rect
    tx, ty = at
    pk = _keys(ref[ry:ry + rh, rx:rx + rw])
    wk = _keys(_window(frame, tx, ty, rw, rh))
    around = window_any(frame[..., 3] > 0, tx - 1, ty - 1, rw + 2, rh + 2, False)
    around[1:-1, 1:-1] = False
    return bool(_intruders(pk, wk, around).any())


def _place(frame: np.ndarray, index: int, ref: np.ndarray, sw: Swap, origin: tuple[int, int], search: int,
           line_colors: np.ndarray) -> Placement:
    """1 フレームで差し替えの位置を探し、確かめて、形ごとの絵を作る（四角が何個あっても、ずれは 1 つ）。"""
    from .same import _classes, _seam_mask
    rects = sw.rects()
    H, W = ref.shape[:2]
    FH, FW = frame.shape[:2]
    ox, oy = origin
    # 位置を探す窓（四角のまわりを MARGIN ドット広げる。1 コマ目のキャンバスの中だけ）
    wins = []
    for rx, ry, rw, rh in rects:
        wx0, wy0 = max(0, rx - MARGIN), max(0, ry - MARGIN)
        wx1, wy1 = min(W, rx + rw + MARGIN), min(H, ry + rh + MARGIN)
        wins.append((wx0, wy0, wx1 - wx0, wy1 - wy0, _keys(ref[wy0:wy1, wx0:wx1])))
    best = None
    for dx, dy in _shifts(search):
        if any(rx + ox + dx < 0 or ry + oy + dy < 0 or rx + ox + dx + rw > FW or ry + oy + dy + rh > FH
               for rx, ry, rw, rh in rects):
            continue
        n = hit = 0
        for wx0, wy0, ww, wh, rk in wins:
            fk = _keys(_window(frame, wx0 + ox + dx, wy0 + oy + dy, ww, wh))
            union = (rk >= 0) | (fk >= 0)
            n += int(union.sum())
            hit += int(((rk == fk) & union).sum())
        m = hit / n if n else 0.0
        if best is None or m > best[1] + 1e-12:
            best = ((dx, dy), m)
    if best is None:
        return Placement(index, (0, 0), 0.0, False, SKIP_EDGE)
    (dx, dy), m = best
    if m < MIN_MATCH:
        return Placement(index, (dx, dy), m, False, SKIP_LOW)
    ats = [(rx + ox + dx, ry + oy + dy) for rx, ry, rw, rh in rects]
    for rect, at in zip(rects, ats):
        if _crossing(ref, frame, rect, at):
            return Placement(index, (dx, dy), m, False, SKIP_CROSS)
        if _intrusion(ref, frame, rect, at):
            return Placement(index, (dx, dy), m, False, SKIP_INTRUSION)
    p = Placement(index, (dx, dy), m, True)
    foot = _bottom(frame)
    co = _classes(frame, line_colors)
    for sn, dots in sw.shapes.items():
        img = frame.copy()
        pasted = np.zeros((FH, FW), bool)
        for rect, (tx, ty) in zip(rects, ats):
            rx, ry, rw, rh = rect
            mine = [d for d in dots if rx <= d.x < rx + rw and ry <= d.y < ry + rh]
            if not mine:
                continue   # この形はこの四角を変えない（もう片方の目だけ閉じる絵など）
            img[ty:ty + rh, tx:tx + rw] = _shape_patch(ref, mine, rect)
            pasted[ty:ty + rh, tx:tx + rw] = True
        if _bottom(img) != foot:
            return Placement(index, (dx, dy), m, False, SKIP_FOOT)
        # 境目の確かめ（same.py と同じ）: 1 コマ目に形を重ねた絵を、見つけた位置に置いたもの（写した側の元）と比べる
        shaped = ref.copy()
        for d in dots:
            shaped[d.y, d.x] = (0, 0, 0, 0) if d.color is None else \
                (int(d.color[1:3], 16), int(d.color[3:5], 16), int(d.color[5:7], 16), 255)
        src = _window(shaped, -(ox + dx), -(oy + dy), FW, FH)
        cs = _classes(src, line_colors)
        bad = _seam_mask(_classes(img, line_colors), cs, co, pasted)
        if bad.any():
            # 1 コマ目そのものに描いたときにもできる所（描いた人が決めた形）は数えない
            cr = _classes(_window(ref, -(ox + dx), -(oy + dy), FW, FH), line_colors)
            bad &= ~_seam_mask(cs, cs, cr, pasted)
        if bad.any():
            return Placement(index, (dx, dy), m, False, SKIP_SEAM)
        p.images[sn] = img
    return p


def _origins(info: dict, n: int) -> list[tuple[int, int]]:
    """フレームごとの、1 コマ目の左上が置かれている位置（足した余白と体全体のずれ）。"""
    left = int(info.get("padding", 0) or 0)
    top = int(info.get("padding_y", (0, 0))[0]) if "padding_y" in info else 0
    offs = info.get("offset_y")
    if offs is None:
        offs = info.get("offsets")
    offs = list(offs or [])
    return [(left, top + (int(offs[i]) if i < len(offs) else 0)) for i in range(n)]


def _repeats(cycle: int, n: int, interval: int) -> int:
    r = max(1, int(round(interval / cycle))) if cycle > 0 else 1
    return max(1, min(r, MAX_FRAMES // max(n, 1)))


def _segments(starts, durations, window_start, seq) -> list[list[tuple[str | None, int]]]:
    """フレームごとの切れ目（形の名前か None（そのまま）, 時間）の並び。"""
    marks = []   # (始まり, 終わり, 形)
    t = window_start
    for sn, ms in seq:
        marks.append((t, t + ms, sn))
        t += ms
    out = []
    for st, d in zip(starts, durations):
        en = st + d
        pieces: list[list] = []
        cur = st
        for a, b, sn in marks:
            a, b = max(a, st), min(b, en)
            if a >= b:
                continue
            if a > cur:
                pieces.append([None, a - cur])
            pieces.append([sn, b - a])
            cur = b
        if cur < en:
            pieces.append([None, en - cur])
        # 短すぎる切れ端は前（無ければ後ろ）に足す
        i = 0
        while len(pieces) > 1 and i < len(pieces):
            if pieces[i][1] < MIN_PIECE:
                j = i - 1 if i > 0 else i + 1
                if pieces[i][0] is None or pieces[j][0] is not None:
                    pieces[j][1] += pieces[i][1]
                else:   # 形の切れ端を「そのまま」に足すと形が消えるので、形の方を伸ばす
                    pieces[j] = [pieces[i][0], pieces[j][1] + pieces[i][1]]
                del pieces[i]
                i = 0
                continue
            i += 1
        # 同じ形が続いたらまとめる
        merged: list[list] = []
        for p in pieces:
            if merged and merged[-1][0] == p[0]:
                merged[-1][1] += p[1]
            else:
                merged.append(list(p))
        out.append([(p[0], int(p[1])) for p in merged])
    return out


@dataclass
class SwapReport:
    """差し替えの記録（シートの JSON の `swaps`）。"""

    interval_ms: int
    cycle_ms: int
    repeats: int
    items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return any(e.get("applied") for e in self.items)

    def info(self) -> dict:
        return {"interval_ms": int(self.interval_ms), "cycle_ms": int(self.cycle_ms), "repeats": int(self.repeats),
                "items": self.items}


def apply_swaps(anim, ref: np.ndarray, s: SwapSet, palette: list[str] | None = None):
    """動きの結果（`motion.base.MotionResult`。シートの材料）に差し替えを入れた新しい結果と記録を返す（元は変えない）。

    `ref` は差し替えを描いた 1 コマ目（コマが並んだ絵は元の 1 コマ目、1 枚の絵は変換したスプライト）。
    当てられない（大きさ・パレットが違う）ときは SwapError。どのフレームにも当てられないときは、元の結果のまま（周も増やさない）で
    記録に理由を書く。
    """
    import dataclasses

    from .edit import palette_hex
    pal = palette if palette is not None else palette_hex(list(anim.frames))
    warnings = check_swaps(ref, pal, s)
    frames, durations = list(anim.frames), [int(d) for d in anim.durations]
    n = len(frames)
    cycle = int(sum(durations))
    origins = _origins(anim.info, n)
    rep = SwapReport(int(s.interval_ms), cycle, 1, warnings=list(warnings))
    sw = s.swaps[0]   # 今は blink だけ（1 つ）
    seq = sw.sequence(s.close_ms, s.half_ms)
    length = sum(ms for _, ms in seq)
    from .steady import outline_keys
    line_colors = outline_keys(ref)
    places = [_place(frames[i], i, ref, sw, origins[i], s.search, line_colors) for i in range(n)]
    item: dict = {"type": sw.type, "rects": [{"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for r in sw.rects()],
                  "shapes": [sn for sn in TYPES[sw.type] if sw.shapes.get(sn)]}
    not_used = [p for p in places if not p.ok]
    if not_used:
        item["frames_not_used"] = [{"frame": int(p.frame), "code": p.code, "match": round(float(p.match), 3)}
                                   for p in not_used]
    repeats = _repeats(cycle, n, s.interval_ms)
    total = cycle * repeats
    seq_frames = list(range(n)) * repeats
    seq_dur = durations * repeats
    starts = list(np.cumsum([0] + seq_dur[:-1]).astype(int).tolist())
    # 入れ始めるフレーム: シートの真ん中から順に、まばたきがかかるフレームが全部当てられる所
    order = list(range(len(seq_frames) // 2, len(seq_frames))) + list(range(0, len(seq_frames) // 2))
    chosen = None
    for j in order:
        t0 = starts[j]
        if t0 + length > total:
            continue
        cover = [k for k in range(len(seq_frames)) if starts[k] < t0 + length and starts[k] + seq_dur[k] > t0]
        if all(places[seq_frames[k]].ok for k in cover):
            chosen = j
            break
    name = TYPE_TEXT.get(sw.type, sw.type)
    if chosen is None:
        item["applied"] = False
        item["message"] = f"{name}を入れられるフレームがありません（どのフレームも 1 コマ目と合う位置が見つからない）"
        rep.items.append(item)
        rep.warnings.append(item["message"])
        out = dataclasses.replace(anim, info=dict(anim.info, swaps=rep.info()))
        return out, rep
    rep.repeats = repeats
    t0 = starts[chosen]
    segs = _segments(starts, seq_dur, t0, seq)
    new_frames, new_dur, new_src, swap_marks = [], [], [], []
    used_frames = []
    for k, pieces in enumerate(segs):
        i = seq_frames[k]
        for sn, ms in pieces:
            if sn is None:
                new_frames.append(frames[i])
            else:
                new_frames.append(places[i].images[sn])
                used_frames.append(len(new_frames) - 1)
            new_dur.append(ms)
            new_src.append(i)
            swap_marks.append(None if sn is None else {"type": sw.type, "shape": sn})
    item.update({"applied": True, "start_ms": int(t0), "duration_ms": int(length), "frames": used_frames,
                 "placements": [p.info() for p in places]})
    rep.items.append(item)
    if not_used:
        src = anim.info.get("source_frames")
        src = src if isinstance(src, (list, tuple)) and len(src) == n else None
        parts = [f"{p.frame + 1} 枚目" + (f"（元の {src[p.frame]} コマ目）" if src else "") + f": {SKIP_TEXT[p.code]}"
                 for p in not_used]
        rep.warnings.append(f"{name}を入れなかったフレームがあります（迷ったら入れません）。" + "／".join(parts))
    info = dict(anim.info)
    for key in ("source_frames", "offset_y", "offsets"):
        v = info.get(key)
        if isinstance(v, (list, tuple)) and len(v) == n:
            info[key] = [v[i] for i in new_src]
    info["swap_frames"] = swap_marks
    info["swaps"] = rep.info()
    out = dataclasses.replace(anim, frames=new_frames, durations=new_dur, info=info)
    return out, rep


def mirror_swapped(swapped, mirror_anim):
    """差し替えを入れた結果（反転していない方）を左右反転した結果。`mirror_anim` は同じ設定で反転して組み立てた結果
    （名前・pivot などを使う）。差し替えの位置の横のずれは向きを反対にする。"""
    import copy
    import dataclasses
    info = dict(swapped.info)
    info["pivot_x"] = mirror_anim.info.get("pivot_x", info.get("pivot_x"))
    if "swaps" in info:
        sw = copy.deepcopy(info["swaps"])
        for it in sw["items"]:
            for p in it.get("placements", []):
                p["shift"]["x"] = -p["shift"]["x"] if p["shift"]["x"] else 0
            w = swapped.frames[0].shape[1]
            pl = int(info.get("padding", 0) or 0)
            ref_w = w - 2 * pl
            it["rects"] = [dict(r, x=ref_w - r["x"] - r["w"]) for r in it.get("rects", [])]
        info["swaps"] = sw
    frames = [f[:, ::-1].copy() for f in swapped.frames]
    return dataclasses.replace(mirror_anim, frames=frames, durations=list(swapped.durations), info=info)


def swap_summary(rep: SwapReport) -> str:
    """コンソール・画面に出す 1 行の説明。"""
    parts = []
    for it in rep.items:
        name = TYPE_TEXT.get(it["type"], it["type"])
        if it.get("applied"):
            parts.append(f"{name}を {it['start_ms']} ミリ秒から {it['duration_ms']} ミリ秒（フレーム "
                         f"{', '.join(str(i + 1) for i in it['frames'])}）")
        else:
            parts.append(f"{name}は入れませんでした")
    head = f"動き {rep.repeats} 周（{rep.cycle_ms * rep.repeats} ミリ秒）に 1 回" if rep.applied else ""
    return (head + "、" if head else "") + "・".join(parts)
