"""手の直し（コマが並んだ絵の、ドットの差し替え）。

ブラウザ版の「ドットを直す」欄と CLI の `--edits` で共通の処理。直しは「元のコマ（source_frame）ごとの、ドットの差し替えの一覧」で、
各ドットは (x, y, 色) のどれか。色はシートのパレット（直す前のコマで使っている色）の 1 色か、透明。新しい色は作れない。
1,2,3,2 のように同じ元のコマを何度も使う順番でも、直しは元のコマに当てるので、1 回直せば両方に入る。

処理の順番は 変換 → 1 コマ目と同じにする範囲 → ちらつきを減らす → **手の直し** → シート（再生・保存）。
前の段を変えて計算し直したときは、同じ一覧をそのまま上に重ね直す。直したときの下の色（`was`）と今の下の色が違うドットは数えて知らせる。
コマの大きさが違う・コマの数が足りない・直しの色が今のパレットに無いときは当てられない（`TouchupError`）。

直しの一覧のファイル（`<出力名>_<名前>_edits.json`）の形:

    {"version": 1, "kind": "sprite_edits", "name": "frames", "frame_width": 66, "frame_height": 92, "frame_count": 3,
     "palette": ["#rrggbb", ...], "dot_count": 10,
     "frames": [{"source_frame": 1, "dots": [[x, y, 色の番号 か null（透明）, 直したときの下の色の番号 か null], ...]}, ...]}

色の番号は、このファイルの `palette` の何番目か（0 から）。下の色（4 つ目）は省いてもよい（そのときは下の色の確かめをしない）。
シートの JSON には、直しがあるときだけ `edits`（`palette`・`dot_count`・`frames`。ドットは [x, y, 色の番号 か null]）を足す。
直しが無いときの出力は、今までとバイト単位で同じ。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

VERSION = 1
KIND = "sprite_edits"
MAX_DOTS = 200_000   # 1 つのファイルのドットの数の上限（ふつうは数十〜数百）

CODE_SIZE = "size"         # コマの大きさが違う
CODE_FRAME = "frame"       # 元のコマの番号がコマの数を超える
CODE_PALETTE = "palette"   # 直しの色が今のパレットに無い
CODE_OUTSIDE = "outside"   # ドットがコマの外
CODE_TEXT = {
    CODE_SIZE: "コマの大きさが直したときと違う",
    CODE_FRAME: "直したコマが今のコマの数より多い",
    CODE_PALETTE: "直しに使った色が今のパレットに無い",
    CODE_OUTSIDE: "直したドットがコマの外にある",
}

_HEX = re.compile(r"#[0-9a-f]{6}")


class TouchupError(ValueError):
    """直しを当てられない（変換し直して大きさ・パレットが変わったときなど）。`code` は CODE_TEXT のどれか。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Dot:
    x: int
    y: int
    color: str | None            # "#rrggbb" か None（透明）
    was: str | None = None       # 直したときの下の色（"#rrggbb"・透明は "transparent"）。None は分からない


TRANSPARENT = "transparent"


@dataclass
class Touchup:
    """直しの一覧。`frames` は 元のコマの番号（1 から）→ ドットの並び（同じ場所は 1 つだけ。(y, x) の順）。"""

    frames: dict[int, list[Dot]] = field(default_factory=dict)
    frame_width: int | None = None
    frame_height: int | None = None
    frame_count: int | None = None
    name: str = "frames"

    @property
    def dot_count(self) -> int:
        return sum(len(v) for v in self.frames.values())

    def empty(self) -> bool:
        return self.dot_count == 0


@dataclass
class TouchupFrame:
    frame: int                       # 元のコマの番号（1 から）
    dots: list[Dot]
    changed_under: list[tuple[int, int]] = field(default_factory=list)   # 直したときと下の色が違うドット (x, y)
    no_effect: int = 0               # いまの下の色と同じで、見た目が変わらないドットの数


@dataclass
class TouchupReport:
    palette: list[str]               # 直す前のコマのパレット（番号の元）
    frames: list[TouchupFrame]
    warnings: list[str] = field(default_factory=list)

    @property
    def dot_count(self) -> int:
        return sum(len(f.dots) for f in self.frames)

    @property
    def changed_under_count(self) -> int:
        return sum(len(f.changed_under) for f in self.frames)

    def info(self) -> dict:
        """シートの JSON の `edits`（元のコマごとの差し替えの一覧と、直したドットの数）。"""
        index = {c: i for i, c in enumerate(self.palette)}
        return {"palette": list(self.palette), "dot_count": self.dot_count,
                "frames": [{"source_frame": int(f.frame),
                            "dots": [[int(d.x), int(d.y), None if d.color is None else index[d.color]] for d in f.dots]}
                           for f in self.frames]}


# ---------------------------------------------------------------- 読み書き

def _hex(v, what: str) -> str:
    if not isinstance(v, str) or not _HEX.fullmatch(v.lower()):
        raise ValueError(f"{what}は \"#rrggbb\" の形で書いてください: {v!r}")
    return v.lower()


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{what}は整数で書いてください: {v!r}")
    return int(v)


def touchup_from_dict(d) -> Touchup:
    """直しの一覧のファイルの辞書（上の形）を読む。形が違えば ValueError。"""
    if not isinstance(d, dict):
        raise ValueError("直しの一覧は {\"version\": 1, \"frames\": [...]} の形で書いてください")
    if d.get("kind", KIND) != KIND:
        raise ValueError(f"直しの一覧のファイルではありません（kind が {d.get('kind')!r}）")
    if _int(d.get("version", VERSION), "version") > VERSION:
        raise ValueError(f"新しい版（{d.get('version')}）の直しの一覧です。Sprite App を新しくしてください")
    unknown = sorted(set(d) - {"version", "kind", "name", "frame_width", "frame_height", "frame_count", "palette",
                               "dot_count", "frames"})
    if unknown:
        raise ValueError(f"直しの一覧に知らない項目があります: {', '.join(unknown)}")
    pal_raw = d.get("palette", [])
    if not isinstance(pal_raw, list):
        raise ValueError("palette は色（\"#rrggbb\"）の並びで書いてください")
    palette = [_hex(c, "パレットの色") for c in pal_raw]
    size = {}
    for k in ("frame_width", "frame_height", "frame_count"):
        if d.get(k) is not None:
            size[k] = _int(d[k], k)
            if size[k] < 1:
                raise ValueError(f"{k} は 1 以上で書いてください: {size[k]}")
    name = d.get("name", "frames")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"name は半角の英数字と _ - で書いてください: {name!r}")
    fr = d.get("frames", [])
    if not isinstance(fr, list):
        raise ValueError("frames は [{\"source_frame\": 1, \"dots\": [...]}] の並びで書いてください")

    def color(v, what):
        if v is None:
            return None
        i = _int(v, what)
        if not 0 <= i < len(palette):
            raise ValueError(f"{what}の番号 {i} がパレット（{len(palette)} 色）にありません")
        return palette[i]

    out: dict[int, dict[tuple[int, int], Dot]] = {}
    total = 0
    for e in fr:
        if not isinstance(e, dict) or set(e) - {"source_frame", "dots"}:
            raise ValueError("frames の中は {\"source_frame\": 1, \"dots\": [[x, y, 色の番号], ...]} の形で書いてください")
        k = _int(e.get("source_frame"), "source_frame")
        if k < 1:
            raise ValueError(f"source_frame は 1 から数えてください: {k}")
        dots = e.get("dots", [])
        if not isinstance(dots, list):
            raise ValueError("dots は [[x, y, 色の番号], ...] の並びで書いてください")
        cur = out.setdefault(k, {})
        for v in dots:
            if not isinstance(v, list) or len(v) not in (3, 4):
                raise ValueError(f"ドットは [x, y, 色の番号 か null] か [x, y, 色, 下の色] で書いてください: {v!r}")
            x, y = _int(v[0], "x"), _int(v[1], "y")
            if x < 0 or y < 0:
                raise ValueError(f"ドットの位置は 0 以上で書いてください: {x}, {y}")
            was = None
            if len(v) == 4:
                was = TRANSPARENT if v[3] is None else color(v[3], "下の色")
            cur[(y, x)] = Dot(x, y, color(v[2], "色"), was)   # 同じ場所は後のものを使う
        total += len(dots)
        if total > MAX_DOTS:
            raise ValueError(f"直したドットが多すぎます（{MAX_DOTS} 個まで）")
    frames = {k: [m[p] for p in sorted(m)] for k, m in sorted(out.items()) if m}
    return Touchup(frames=frames, frame_width=size.get("frame_width"), frame_height=size.get("frame_height"),
                   frame_count=size.get("frame_count"), name=name)


def load_touchup(text: str) -> Touchup:
    try:
        d = json.loads(text)
    except ValueError as e:
        raise ValueError(f"直しの一覧の JSON を読めません: {e}") from None
    return touchup_from_dict(d)


def touchup_dict(t: Touchup, palette: list[str]) -> dict:
    """直しの一覧のファイルの辞書（`palette` は番号の元。直しの色と下の色が全部入っていること）。"""
    pal = list(palette)
    for dots in t.frames.values():
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

    return {"version": VERSION, "kind": KIND, "name": t.name, "frame_width": t.frame_width,
            "frame_height": t.frame_height, "frame_count": t.frame_count, "palette": pal, "dot_count": t.dot_count,
            "frames": [{"source_frame": int(k), "dots": [dot(d) for d in v]} for k, v in sorted(t.frames.items()) if v]}


def touchup_json(t: Touchup, palette: list[str]) -> str:
    """直しの一覧のファイルの中身（1 つのドットを 1 行に書く。差分が見やすいように）。"""
    d = touchup_dict(t, palette)
    lines = ["{"]
    head = {k: v for k, v in d.items() if k != "frames"}
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "frames": [')
    for i, f in enumerate(d["frames"]):
        lines.append(f'    {{"source_frame": {f["source_frame"]}, "dots": [')
        for j, v in enumerate(f["dots"]):
            lines.append("      " + json.dumps(v) + ("," if j + 1 < len(f["dots"]) else ""))
        lines.append("    ]}" + ("," if i + 1 < len(d["frames"]) else ""))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def edits_file_name(stem: str, name: str) -> str:
    """直しの一覧のファイルの名前（`<出力名>_<名前>_edits.json`）。"""
    return f"{stem}_{name}_edits.json"


# ---------------------------------------------------------------- 当てる

def palette_of(frames: list[np.ndarray]) -> list[str]:
    """直す前のコマのパレット（シートの JSON の palette と同じ形・同じ順）。"""
    from .edit import palette_hex
    return palette_hex(frames)


def _color_at(f: np.ndarray, x: int, y: int) -> str:
    p = f[y, x]
    if p[3] == 0:
        return TRANSPARENT
    return "#%02x%02x%02x" % (int(p[0]), int(p[1]), int(p[2]))


def check_touchup(frames: list[np.ndarray], t: Touchup) -> list[str]:
    """当てられるかを確かめる。当てられなければ TouchupError。戻り値は直す前のパレット。"""
    if not frames:
        raise TouchupError(CODE_FRAME, "コマがありません")
    h, w = frames[0].shape[:2]
    if (t.frame_width is not None and t.frame_width != w) or (t.frame_height is not None and t.frame_height != h):
        raise TouchupError(CODE_SIZE, f"{CODE_TEXT[CODE_SIZE]}（直したとき {t.frame_width}×{t.frame_height}、"
                                      f"いま {w}×{h} ドット）")
    if t.frames and max(t.frames) > len(frames):
        raise TouchupError(CODE_FRAME, f"{CODE_TEXT[CODE_FRAME]}（直したコマ {max(t.frames)}、いまのコマ {len(frames)} 個）")
    palette = palette_of(frames)
    have = set(palette)
    missing = sorted({d.color for v in t.frames.values() for d in v if d.color is not None} - have)
    if missing:
        raise TouchupError(CODE_PALETTE, f"{CODE_TEXT[CODE_PALETTE]}（{', '.join(missing)}）")
    for k, v in t.frames.items():
        for d in v:
            if not (0 <= d.x < w and 0 <= d.y < h):
                raise TouchupError(CODE_OUTSIDE, f"{CODE_TEXT[CODE_OUTSIDE]}（{k} コマ目の {d.x}, {d.y}。コマは {w}×{h}）")
    return palette


def apply_touchup(frames: list[np.ndarray], t: Touchup) -> tuple[list[np.ndarray], TouchupReport]:
    """直しを当てたコマと記録を返す（元のコマは変えない）。当てられなければ TouchupError。"""
    palette = check_touchup(frames, t)
    out = [f.copy() for f in frames]
    rep_frames = []
    for k, dots in sorted(t.frames.items()):
        if not dots:
            continue
        f = out[k - 1]
        base = frames[k - 1]
        tf = TouchupFrame(frame=k, dots=list(dots))
        for d in dots:
            under = _color_at(base, d.x, d.y)
            if d.was is not None and d.was != under:
                tf.changed_under.append((d.x, d.y))
            want = TRANSPARENT if d.color is None else d.color
            if want == under:
                tf.no_effect += 1
            if d.color is None:
                f[d.y, d.x] = (0, 0, 0, 0)
            else:
                f[d.y, d.x] = (int(d.color[1:3], 16), int(d.color[3:5], 16), int(d.color[5:7], 16), 255)
        rep_frames.append(tf)
    rep = TouchupReport(palette=palette, frames=rep_frames)
    n = rep.changed_under_count
    if n:
        where = "、".join(f"{f.frame} コマ目 {len(f.changed_under)}" for f in rep_frames if f.changed_under)
        rep.warnings.append(f"直したドットのうち {n} 個は、前の段を変えたので下の色が直したときと違います（{where}）。"
                            "直しはそのまま重ねています。絵を見て確かめてください")
    return out, rep


def touchup_result(res, t: Touchup | None):
    """コマの変換の結果（`frames.FramesResult`）に直しを当てた結果。`t` が空なら `res` をそのまま返す（今までと同じ）。"""
    import dataclasses
    if t is None or t.empty():
        return res
    frames, rep = apply_touchup(res.frames, t)
    return dataclasses.replace(res, frames=frames, touchup=rep, before_touchup=res.frames)
