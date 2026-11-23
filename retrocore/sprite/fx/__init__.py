"""重ね（漫画的エフェクト）: キャラのシートとは別の小さなスプライトシートで、！・汗・ハート・Zzz・？・キラキラなど（14 種）をキャラに重ねる。

    from retrocore.sprite.fx import make_fx, fx_json, composite
    fx = make_fx("surprise", anim.frames)                        # 置く場所はキャラの 1 コマ目から自動
    fx = make_fx("sweat", mirror_anim.frames, mirror=True, offset=(1, 0), end="hide")
    frames, durations, canvas = composite(anim.frames, anim.durations, fx)   # 重ねた見た目（画面の再生・GIF）

決まりごと（docs/how-it-works.md の「重ね」）:
- キャラのシートには手を入れない（エフェクトを選ばないときの出力は今までと同じ）。エフェクトは別のシート（PNG と JSON）
- 絵と動きは `art.py` の文字の表（画面と CLI で同じもの）。動きは、ずらす・絵を切り替える・消す（点滅）だけ。拡大・縮小はしない
- 色は決まった色。輪郭（`o`）だけは、キャラのシルエットの縁に出ている色のうちいちばん暗い色に合わせる
- 置く場所はキャラの 1 コマ目から自動で決める: 上から 1/4 の行（3 行以上）を頭とみなし、「頭の上」は頭の左右の真ん中・上端の
  1 ドット上、「頭の横」は頭の右端の 1 ドット右（縦は頭の上から 1/3 のあたりにエフェクトの真ん中）。
  「顔」は目の位置が分からないので、頭の範囲からの目安（`face_point()`: 頭の真ん中から幅の 1/4 右・頭の上端から頭の行の 2 倍下）に
  エフェクトの上端の真ん中を置く。「顔の上半分」は同じ目安の行の上に、頭の左右の真ん中でエフェクトの下端をそろえる。
  「体のまわり」は絵全体の四角（`body_box()`）の角・辺の真ん中を基準に部品を置く（キャラの大きさで部品の間が変わる）。
  どれもそこから (x, y) ずらせる
- キャラを左右反転したシートでは、置く場所（ずらした分も）を左右反転する。エフェクトの絵は反転しない（文字が裏返らないように）
- JSON の `position` は、キャラのコマの左上から見たエフェクトのコマの左上（ドット。右と下が +。キャラのコマの外ならマイナスもある）
- 1 回で終わるもの（`loop` が false）は、最後のコマで止めて残す（`end` = "keep"。既定）か、最後に透明のコマを足して消す（"hide"）

NumPy と標準ライブラリだけを使う（Pyodide で動く）。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import art

ENDS = {"keep": "最後の形で止めて残す", "hide": "最後に消す"}
DEFAULT_END = "keep"
ANCHORS = {"top": "頭の上", "side": "頭の横", "face": "顔", "upper_face": "顔の上半分", "around": "体のまわり"}
REFS = ("nw", "n", "ne", "w", "e", "sw", "s", "se")   # 体のまわり（around）の部品の基準（絵全体の四角の角・辺の真ん中）
FX_VERSION = 1
FX_FORMAT = "sprite-fx"
HEAD_RATIO = 0.25        # 上から何割の行を頭とみなすか
HEAD_MIN_ROWS = 3
GAP = 1                  # 頭とエフェクトのあいだに空ける透明のドット
FACE_ROWS = 2            # 顔（目の下）の目安: 頭の上端から、頭とみなした行の数の何倍下か
HIDE_MS = 100            # 「消す」で最後に足す透明のコマの時間
HOLD_MS = 1000           # 重ねた GIF・画面の再生で、1 回で終わるものの最後の形を見せておく時間（この後にキャラの周の切れ目まで）
MAX_COMBINED_MS = 12000  # 重ねた見た目の長さの上限（キャラの周とエフェクトの周の最小公倍数がこれより長いときは切る）
MAX_OFFSET = 64
MAX_PISKEL_FRAMES = 256
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class FxError(ValueError):
    """エフェクトの指定・絵の表が正しくない。"""


@dataclass
class Effect:
    """読み込んだエフェクトの定義（`art.EFFECTS` の 1 つ）。`placements` は部品を置く位置を、全部のコマが入る四角の左上を
    0, 0 にずらしたもの。"""

    name: str
    label: str
    anchor: str
    loop: bool
    parts: dict
    placements: list
    durations: list
    width: int
    height: int
    origin: tuple = (0, 0)   # 体のまわり（around）だけ: 絵全体の四角の左上から見た、エフェクトのコマの左上


@dataclass
class FxResult:
    """作ったエフェクト。`frames` は同じ大きさの RGBA（エフェクトの 1 コマ）、`position` はキャラのコマの左上から見た
    エフェクトのコマの左上 (x, y)。"""

    name: str
    label: str
    anchor: str
    loop: bool
    end: str | None
    frames: list
    durations: list
    position: tuple
    offset: tuple
    mirrored: bool
    char_size: tuple
    outline: tuple
    warnings: list = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return int(w), int(h)


# ---------------------------------------------------------------- 絵の表を読む

def parse_part(text: str, colors: dict | None = None) -> list[str]:
    """部品の表（文字の行）を、行の一覧にする。空の行・前後の空白は除く。行の長さ・知らない文字を確かめる。"""
    colors = art.COLORS if colors is None else colors
    rows = [r.strip() for r in str(text).strip("\n").splitlines()]
    rows = [r for r in rows if r]
    if not rows:
        raise FxError("部品の表が空です")
    if len({len(r) for r in rows}) != 1:
        raise FxError(f"部品の表の行の長さがそろっていません: {[len(r) for r in rows]}")
    for r in rows:
        for ch in r:
            if ch not in (art.TRANSPARENT, art.OUTLINE) and ch not in colors:
                raise FxError(f"部品の表に知らない文字があります: {ch!r}（使えるのは . o {' '.join(colors)}）")
    return rows


def _ref_point(ref: str, box: tuple[int, int]) -> tuple[int, int]:
    """絵全体の四角（幅・高さ）の角・辺の真ん中の点（四角の左上が 0, 0。右・下の辺は四角のすぐ外の線）。"""
    w, h = box
    xs = {"w": 0, "c": w // 2, "e": w}
    ys = {"n": 0, "c": h // 2, "s": h}
    v = ref[0] if ref[0] in "ns" else "c"
    u = ref[-1] if ref[-1] in "we" else "c"
    return xs[u], ys[v]


def load_effect(name: str, effects: dict | None = None, box: tuple[int, int] | None = None) -> Effect:
    """名前のエフェクトを読む（`art.EFFECTS`）。全部のコマが入る四角を計算する。

    `box` は体のまわり（around）のときの絵全体の四角の (幅, 高さ)。基準つきの部品 `(名前, 基準, x, y)` の位置をこれで決める
    （None なら 0×0 の四角として読む。名前・置く場所などを見るだけのとき）。"""
    effects = art.EFFECTS if effects is None else effects
    if name not in effects:
        raise FxError(f"知らないエフェクトです: {name}（使えるのは {', '.join(effects)}）")
    if not NAME_RE.match(name):
        raise FxError(f"エフェクトの名前は英小文字・数字・_ で書いてください: {name}")
    d = effects[name]
    if d.get("anchor") not in ANCHORS:
        raise FxError(f"{name} の anchor は {' / '.join(ANCHORS)} のどれかにしてください: {d.get('anchor')}")
    parts = {k: parse_part(v) for k, v in d["parts"].items()}
    frames = d["frames"]
    if not frames:
        raise FxError(f"{name} にコマがありません")
    boxes = []
    for placed, ms in frames:
        if isinstance(ms, bool) or not isinstance(ms, int) or ms < 10:
            raise FxError(f"{name} のコマの時間は 10 以上の整数（ミリ秒）にしてください: {ms}")
        for item in placed:
            pname, x, y = _place(name, d["anchor"], item, box)
            if pname not in parts:
                raise FxError(f"{name} のコマに知らない部品があります: {pname}")
            rows = parts[pname]
            boxes.append((x, y, x + len(rows[0]), y + len(rows)))
    if not boxes:
        raise FxError(f"{name} のどのコマにも部品がありません")
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    w = max(b[2] for b in boxes) - x0
    h = max(b[3] for b in boxes) - y0
    placements = [[(p, x - x0, y - y0) for p, x, y in (_place(name, d["anchor"], it, box) for it in placed)]
                  for placed, _ in frames]
    return Effect(name, str(d.get("label", name)), d["anchor"], bool(d.get("loop", False)), parts, placements,
                  [int(ms) for _, ms in frames], int(w), int(h), (int(x0), int(y0)))


def _place(name: str, anchor: str, item, box) -> tuple[str, int, int]:
    """置く部品 `(名前, x, y)` か `(名前, 基準, x, y)`（体のまわりだけ）を、`(名前, x, y)` にする。"""
    if len(item) == 3:
        pname, x, y = item
        return pname, int(x), int(y)
    if len(item) != 4:
        raise FxError(f"{name} の置く部品は (名前, x, y) か (名前, 基準, x, y) にしてください: {item}")
    pname, ref, x, y = item
    if anchor != "around":
        raise FxError(f"{name}: 基準つきの部品（{ref}）は、置く場所が around のときだけ使えます")
    if ref not in REFS:
        raise FxError(f"{name} の部品の基準は {' / '.join(REFS)} のどれかにしてください: {ref}")
    px, py = _ref_point(ref, box or (0, 0))
    return pname, px + int(x), py + int(y)


def effect_names() -> list[str]:
    return list(art.EFFECTS)


def fx_spec() -> dict:
    """画面の選択肢（名前・表示名・置く場所・くり返すか）と既定値。build_sprite_web.py が画面に埋め込む。"""
    items = []
    for n in effect_names():
        e = load_effect(n)
        items.append({"name": n, "label": e.label, "anchor": e.anchor, "anchor_label": ANCHORS[e.anchor], "loop": e.loop})
    return {"effects": items, "end": {"default": DEFAULT_END, "choices": [[k, v] for k, v in ENDS.items()]},
            "max_offset": MAX_OFFSET}


# ---------------------------------------------------------------- 色・置く場所

def outline_color(frame: np.ndarray) -> tuple[int, int, int]:
    """キャラのシルエットの縁に出ている色のうち、いちばん暗い色（明るさ = 0.299R + 0.587G + 0.114B。同じなら RGB の小さい方）。
    不透明なドットが無ければ `art.OUTLINE_FALLBACK`。"""
    a = np.asarray(frame, np.uint8)
    opaque = a[..., 3] > 0
    pad = np.pad(opaque, 1)
    edge = opaque & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    if not edge.any():
        return tuple(art.OUTLINE_FALLBACK)
    cols = np.unique(a[edge][:, :3].astype(np.int32), axis=0)
    luma = cols[:, 0] * 299 + cols[:, 1] * 587 + cols[:, 2] * 114
    order = np.lexsort((cols[:, 2], cols[:, 1], cols[:, 0], luma))
    c = cols[order[0]]
    return int(c[0]), int(c[1]), int(c[2])


def head_box(frame: np.ndarray) -> dict | None:
    """頭の範囲: 不透明な所の上端から、キャラの高さの 1/4（3 行以上）の行の、左端・右端。不透明なドットが無ければ None。"""
    opaque = np.asarray(frame)[..., 3] > 0
    rows = np.flatnonzero(opaque.any(axis=1))
    if len(rows) == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    n = max(HEAD_MIN_ROWS, int(round((bottom - top + 1) * HEAD_RATIO)))
    n = min(n, bottom - top + 1)
    cols = np.flatnonzero(opaque[top:top + n].any(axis=0))
    return {"top": top, "rows": n, "left": int(cols[0]), "right": int(cols[-1])}


def body_box(frame: np.ndarray) -> dict | None:
    """絵全体の範囲: 不透明な所を囲むいちばん小さな四角の左・上・幅・高さ。不透明なドットが無ければ None。"""
    opaque = np.asarray(frame)[..., 3] > 0
    rows = np.flatnonzero(opaque.any(axis=1))
    if len(rows) == 0:
        return None
    cols = np.flatnonzero(opaque.any(axis=0))
    return {"left": int(cols[0]), "top": int(rows[0]), "width": int(cols[-1] - cols[0] + 1),
            "height": int(rows[-1] - rows[0] + 1)}


def face_point(head: dict) -> tuple[int, int]:
    """顔（右の目の下のあたり）の目安の点。目の位置は分からないので頭の範囲から決める: 横は頭の真ん中から頭の幅の 1/4 右、
    縦は頭の上端から頭とみなした行の数の 2 倍下（2〜3 頭身のキャラで目の下あたり。合わなければずらす欄で直す）。"""
    cx = (head["left"] + head["right"] + 1) // 2
    width = head["right"] - head["left"] + 1
    return cx + width // 4, head["top"] + FACE_ROWS * head["rows"]


def auto_position(effect: Effect, frame: np.ndarray) -> tuple[int, int]:
    """反転していないキャラの 1 コマ目に対する、エフェクトのコマの左上（ずらす前）。"""
    head = head_box(frame)
    if head is None:
        raise FxError("キャラの絵に不透明なドットが無いので、エフェクトを置く場所を決められません")
    if effect.anchor == "around":
        body = body_box(frame)
        return body["left"] + effect.origin[0], body["top"] + effect.origin[1]
    if effect.anchor == "face":
        fx_, fy = face_point(head)
        return int(fx_ - effect.width // 2), int(fy)
    if effect.anchor == "upper_face":
        _, fy = face_point(head)
        x = (head["left"] + head["right"] + 1 - effect.width) // 2
        return int(x), int(fy - effect.height)
    if effect.anchor == "top":
        x = (head["left"] + head["right"] + 1 - effect.width) // 2
        y = head["top"] - GAP - effect.height
    else:
        x = head["right"] + 1 + GAP
        y = head["top"] + head["rows"] // 3 - effect.height // 2
    return int(x), int(y)


def check_offset(offset) -> tuple[int, int]:
    try:
        dx, dy = offset
    except (TypeError, ValueError):
        raise FxError(f"ずらす量は (x, y) の 2 つの整数にしてください: {offset}")
    for v in (dx, dy):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or abs(int(v)) > MAX_OFFSET:
            raise FxError(f"ずらす量は −{MAX_OFFSET}〜{MAX_OFFSET} の整数（ドット）にしてください: {offset}")
    return int(dx), int(dy)


def check_end(end) -> str:
    if end is None:
        return DEFAULT_END
    if end not in ENDS:
        raise FxError(f"終わり方は {' / '.join(ENDS)} のどれかにしてください: {end}")
    return end


# ---------------------------------------------------------------- エフェクトのコマを作る

def render(effect: Effect, outline=None) -> list[np.ndarray]:
    """エフェクトの各コマ（RGBA。全部同じ大きさ）。後に置いた部品が上になる（透明の所は下を消さない）。"""
    lut = {k: (*v, 255) for k, v in art.COLORS.items()}
    lut[art.OUTLINE] = (*(art.OUTLINE_FALLBACK if outline is None else outline), 255)
    out = []
    for placed in effect.placements:
        img = np.zeros((effect.height, effect.width, 4), np.uint8)
        for pname, x, y in placed:
            for r, row in enumerate(effect.parts[pname]):
                for c, ch in enumerate(row):
                    if ch != art.TRANSPARENT:
                        img[y + r, x + c] = lut[ch]
        out.append(img)
    return out


def make_fx(name: str, char_frames, mirror: bool = False, offset=(0, 0), end=None, effects: dict | None = None) -> FxResult:
    """キャラのシートのコマ（`char_frames`。1 コマ目で置く場所と輪郭の色を決める）に重ねるエフェクトを作る。

    `mirror` は、キャラのシートが左右反転したものか（置く場所を反転する。絵は反転しない）。`end` は 1 回で終わるものの終わり方
    （くり返すものでは使わず、None にする）。"""
    effect = load_effect(name, effects)
    dx, dy = check_offset(offset)
    end = check_end(end) if not effect.loop else None
    if not char_frames:
        raise FxError("キャラのコマがありません")
    first = np.asarray(char_frames[0], np.uint8)
    if first.ndim != 3 or first.shape[2] != 4:
        raise FxError(f"キャラのコマは 高さ×幅×4（RGBA）にしてください: {first.shape}")
    ch, cw = first.shape[:2]
    upright = first[:, ::-1] if mirror else first   # 置く場所は反転していない向きで決めて、あとで反転する
    if effect.anchor == "around":                    # 体のまわり: 部品の間をキャラの大きさで決め直す
        body = body_box(upright)
        if body is not None:
            effect = load_effect(name, effects, (body["width"], body["height"]))
    x, y = auto_position(effect, upright)
    x, y = x + dx, y + dy
    if mirror:
        x = cw - x - effect.width
    outline = outline_color(first)
    frames = render(effect, outline)
    durations = list(effect.durations)
    if end == "hide":
        frames.append(np.zeros_like(frames[0]))
        durations.append(HIDE_MS)
    return FxResult(effect.name, effect.label, effect.anchor, effect.loop, end, frames, durations, (int(x), int(y)),
                    (dx, dy), bool(mirror), (int(cw), int(ch)), outline)


# ---------------------------------------------------------------- 書き出し

_SIZE_TAIL = re.compile(r"_(\d+)x(\d+)_(\d+)f$")


def fx_stem(char_image_name: str, name: str) -> str:
    """エフェクトのファイル名の元（拡張子・大きさなし）: キャラのシートの名前から `_sheet`・`_<幅>x<高さ>_<N>f` を除いて
    `_fx_<名前>` を足す（hero_breathe_sheet.png → hero_breathe_fx_surprise、hero_walk_53x91_4f.png → hero_walk_fx_surprise）。"""
    stem = str(char_image_name).rsplit("/", 1)[-1]
    if stem.lower().endswith(".png"):
        stem = stem[:-4]
    if stem.endswith("_sheet"):
        stem = stem[:-len("_sheet")]
    stem = _SIZE_TAIL.sub("", stem)
    return f"{stem}_fx_{name}"


def fx_file_name(char_image_name: str, fx: FxResult) -> str:
    """エフェクトのシートの PNG の名前。キャラがコマのシート（名前の終わりが `_<幅>x<高さ>_<N>f`）なら、同じ形で
    エフェクトの 1 コマの大きさとコマの数を付ける（Piskel の「Import as spritesheet」に入れる数）。"""
    stem = fx_stem(char_image_name, fx.name)
    tail = str(char_image_name)[:-4] if str(char_image_name).lower().endswith(".png") else str(char_image_name)
    if _SIZE_TAIL.search(tail):
        w, h = fx.size
        stem += f"_{w}x{h}_{len(fx.frames)}f"
    return stem + ".png"


def fx_sheet(fx: FxResult) -> np.ndarray:
    return np.concatenate([np.asarray(f, np.uint8) for f in fx.frames], axis=1)


def fx_info(fx: FxResult, image_name: str, char_image_name: str) -> dict:
    """エフェクトのシートの情報（JSON にする辞書）。"""
    w, h = fx.size
    n = len(fx.frames)
    frames = []
    for i, (f, d) in enumerate(zip(fx.frames, fx.durations)):
        e = {"index": i, "x": i * w, "y": 0, "width": w, "height": h, "duration_ms": int(d)}
        if not (np.asarray(f)[..., 3] > 0).any():
            e["empty"] = True   # 何も見えないコマ（点滅の「消える」・終わり方の「消す」）
        frames.append(e)
    info = {
        "version": FX_VERSION,
        "format": FX_FORMAT,
        "fx": fx.name,
        "label": fx.label,
        "image": image_name,
        "character_image": char_image_name,
        "frame_width": w,
        "frame_height": h,
        "frame_count": n,
        "columns": n,
        "rows": 1,
        "loop": fx.loop,
        "end": fx.end,
        "anchor": fx.anchor,
        "mirrored": fx.mirrored,
        "offset": {"x": fx.offset[0], "y": fx.offset[1]},
        # キャラのコマの左上から見た、エフェクトのコマの左上（ドット。右と下が +）
        "position": {"x": fx.position[0], "y": fx.position[1]},
        "character_frame": {"width": fx.char_size[0], "height": fx.char_size[1]},
        "outline_color": "#%02x%02x%02x" % tuple(fx.outline),
        "total_duration_ms": int(sum(fx.durations)),
        "frames": frames,
    }
    return info


def fx_json(fx: FxResult, image_name: str, char_image_name: str) -> str:
    return json.dumps(fx_info(fx, image_name, char_image_name), ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------- 重ねた見た目（画面の再生・GIF・.piskel）

def combined_length(char_ms: int, fx: FxResult) -> tuple[int, list[str]]:
    """重ねた見た目の長さ（ミリ秒）。くり返すものはキャラの周とエフェクトの周の最小公倍数（長すぎれば切って警告）、
    1 回で終わるものは、エフェクトの長さ＋最後の形を見せる時間（HOLD_MS）を、キャラの周の切れ目まで延ばしたもの。"""
    total = int(sum(fx.durations))
    if fx.loop:
        lcm = char_ms * total // math.gcd(char_ms, total)
        if lcm <= MAX_COMBINED_MS:
            return lcm, []
        n = -(-max(char_ms, total) // char_ms)
        return n * char_ms, [f"キャラの 1 周（{char_ms} ミリ秒）とエフェクトの 1 周（{total} ミリ秒）がそろうまでが長すぎるので、"
                             f"重ねた見た目は {n * char_ms} ミリ秒で切りました（GIF の切れ目でエフェクトが途中から始まります。"
                             "ゲームではそれぞれのシートをくり返すので問題ありません）"]
    n = -(-(total + HOLD_MS) // char_ms)
    return n * char_ms, []


def _starts(durations, length: int, loop: bool) -> list[int]:
    out, t = [], 0
    while t < length:
        for d in durations:
            out.append(t)
            t += int(d)
            if t >= length:
                break
        if not loop:
            break
    return out


def _index_at(durations, t: int, loop: bool) -> int:
    total = int(sum(durations))
    if loop:
        t %= total
    elif t >= total:
        return len(durations) - 1
    acc = 0
    for i, d in enumerate(durations):
        acc += int(d)
        if t < acc:
            return i
    return len(durations) - 1


def composite(char_frames, char_durations, fx: FxResult):
    """キャラのコマとエフェクトを重ねた見た目。(コマの一覧, 時間の一覧, 絵の情報) を返す。

    絵の情報は {"width", "height", "char_x", "char_y", "fx_x", "fx_y", "length_ms", "warnings"}（キャラ・エフェクトの左上が
    重ねた絵のどこにあるか）。同じ絵が続くコマは 1 つにまとめる（時間は足す）。"""
    cf = [np.asarray(f, np.uint8) for f in char_frames]
    ch, cw = cf[0].shape[:2]
    fw, fh = fx.size
    x, y = fx.position
    x0, y0 = min(0, x), min(0, y)
    W, H = max(cw, x + fw) - x0, max(ch, y + fh) - y0
    cx, cy, fxx, fxy = -x0, -y0, x - x0, y - y0
    char_ms = int(sum(int(d) for d in char_durations))
    length, warnings = combined_length(char_ms, fx)
    cuts = sorted(set(_starts(char_durations, length, True)) | set(_starts(fx.durations, length, fx.loop)) | {0})
    cuts = [t for t in cuts if t < length]
    frames, durations = [], []
    for i, t in enumerate(cuts):
        ci = _index_at(char_durations, t, True)
        fi = _index_at(fx.durations, t, fx.loop)
        img = np.zeros((H, W, 4), np.uint8)
        img[cy:cy + ch, cx:cx + cw] = cf[ci]
        f = np.asarray(fx.frames[fi], np.uint8)
        on = f[..., 3] > 0
        region = img[fxy:fxy + fh, fxx:fxx + fw]
        region[on] = f[on]
        d = (cuts[i + 1] if i + 1 < len(cuts) else length) - t
        if frames and np.array_equal(frames[-1], img):
            durations[-1] += d
        else:
            frames.append(img)
            durations.append(d)
    return frames, durations, {"width": int(W), "height": int(H), "char_x": int(cx), "char_y": int(cy),
                               "fx_x": int(fxx), "fx_y": int(fxy), "length_ms": int(length), "warnings": warnings}


def combined_gif(char_frames, char_durations, fx: FxResult, scale: int | None = None, background=None):
    """重ねた見た目の GIF（確かめ用。`gif.make_gif()` を通す）。"""
    from ..motion.gif import DEFAULT_SCALE, make_gif
    frames, durations, view = composite(char_frames, char_durations, fx)
    gif = make_gif(frames, durations, DEFAULT_SCALE if scale is None else scale, background)
    gif.warnings = list(view["warnings"]) + list(gif.warnings)
    return gif


def combined_piskel(char_frames, char_durations, fx: FxResult, name: str):
    """キャラとエフェクトを 2 つのレイヤー（下から「キャラ」「エフェクト」）に分けた .piskel の中身。(文字列か None, メモ) を返す。

    Piskel はコマごとの時間を持たず速さが 1 つだけなので、重ねた見た目の時間をすべて同じ長さのコマに切りそろえる。
    その長さ（時間の最大公約数）で 1 秒が割り切れて 24 FPS 以下・コマが 256 以下のときだけ作る（作れなければ None と理由）。"""
    from ..piskel import piskel_layers_text
    cf = [np.asarray(f, np.uint8) for f in char_frames]
    ch, cw = cf[0].shape[:2]
    fw, fh = fx.size
    x, y = fx.position
    x0, y0 = min(0, x), min(0, y)
    W, H = max(cw, x + fw) - x0, max(ch, y + fh) - y0
    char_ms = int(sum(int(d) for d in char_durations))
    length, warnings = combined_length(char_ms, fx)
    tick = 0
    for d in list(char_durations) + list(fx.durations) + [length]:
        tick = math.gcd(tick, int(d))
    n = length // tick
    if 1000 % tick or 1000 // tick > 24 or n > MAX_PISKEL_FRAMES:
        return None, [f"キャラとエフェクトの時間（{'/'.join(str(int(d)) for d in char_durations)} と "
                      f"{'/'.join(str(d) for d in fx.durations)} ミリ秒）を同じ長さのコマにそろえられないので、"
                      "2 つのレイヤーの .piskel は作りませんでした（Piskel の速さは 1〜24 FPS の 1 つだけ）"]
    layer_c, layer_f = [], []
    for k in range(n):
        t = k * tick
        img = np.zeros((H, W, 4), np.uint8)
        img[-y0:-y0 + ch, -x0:-x0 + cw] = cf[_index_at(char_durations, t, True)]
        layer_c.append(img)
        img = np.zeros((H, W, 4), np.uint8)
        img[y - y0:y - y0 + fh, x - x0:x - x0 + fw] = fx.frames[_index_at(fx.durations, t, fx.loop)]
        layer_f.append(img)
    fps = 1000 // tick
    notes = list(warnings) + [f"キャラとエフェクトを {tick} ミリ秒ずつのコマ（{n} コマ、{fps} FPS）にそろえ、"
                              "レイヤー「キャラ」「エフェクト」に分けました"]
    return piskel_layers_text([("キャラ", layer_c), ("エフェクト", layer_f)], name, fps), notes


# ---------------------------------------------------------------- 画面（Pyodide）

def fx_for_web(sheet_png: bytes, sheet_json_text: str, options_json: str = "{}") -> str:
    """ブラウザ UI から呼ぶ形。キャラのシート（PNG と JSON）と設定 {"fx", "offset": [x, y], "end", "gif": {"scale",
    "background"} か無し, "piskel": true/false} から、エフェクトのシート・JSON・重ねた見た目（再生用）を作る。
    反転はシートの JSON の source.mirrored から決める。中身は CLI と同じ関数なので、同じ設定なら同じバイトになる。"""
    import base64

    from ..convert import decode_image, encode_png, error_detail, friendly_error
    from ..motion.gif import frames_from_sheet
    try:
        opts = json.loads(options_json or "{}")
        if not isinstance(opts, dict):
            raise FxError("設定は {\"fx\": \"surprise\"} のような形で渡してください")
        unknown = sorted(set(opts) - {"fx", "offset", "end", "gif", "piskel"})
        if unknown:
            raise FxError(f"知らない設定です: {', '.join(unknown)}")
        info = json.loads(sheet_json_text)
        char_frames, char_durations = frames_from_sheet(decode_image(sheet_png), info)
        src = info.get("source")
        mirror = bool(isinstance(src, dict) and src.get("mirrored"))
        fx = make_fx(str(opts.get("fx")), char_frames, mirror, tuple(opts.get("offset") or (0, 0)), opts.get("end"))
        char_name = str(info["image"])
        name = fx_file_name(char_name, fx)
        out = {"ok": True, "name": name, "png_base64": base64.b64encode(encode_png(fx_sheet(fx))).decode("ascii"),
               "json": fx_json(fx, name, char_name), "loop": fx.loop, "end": fx.end,
               "position": list(fx.position), "frame_width": fx.size[0], "frame_height": fx.size[1],
               "frame_count": len(fx.frames), "durations": [int(d) for d in fx.durations]}
        frames, durations, view = composite(char_frames, char_durations, fx)
        strip = np.concatenate(frames, axis=1)
        out["view"] = {**view, "png_base64": base64.b64encode(encode_png(strip)).decode("ascii"),
                       "durations": [int(d) for d in durations]}
        out["warnings"] = list(view["warnings"])
        stem = fx_stem(char_name, fx.name)
        if opts.get("gif"):
            g = opts["gif"]
            gif = combined_gif(char_frames, char_durations, fx, g.get("scale"), g.get("background"))
            out["gif_base64"] = base64.b64encode(gif.data).decode("ascii")
            out["gif_name"] = stem + ".gif"
            out["gif"] = gif.summary()
        if opts.get("piskel"):
            text, notes = combined_piskel(char_frames, char_durations, fx, stem)
            out["piskel"] = text
            out["piskel_name"] = stem + ".piskel"
            out["piskel_notes"] = notes
    except (ValueError, KeyError, TypeError) as e:
        return json.dumps({"ok": False, "error": f"エフェクトを作れませんでした: {e}", "detail": error_detail(e)},
                          ensure_ascii=False)
    except Exception as e:  # 画面にそのまま出すので、どんなエラーでも言葉にして返す
        return json.dumps({"ok": False, "error": friendly_error(e), "detail": error_detail(e)}, ensure_ascii=False)
    return json.dumps(out, ensure_ascii=False)
