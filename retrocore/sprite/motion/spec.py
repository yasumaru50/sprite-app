"""動きの表（sprite-motion の JSON、version 1）の読み込み・書き間違いの確かめ・書き出し。

形は docs/sprite-app-plan.md の「動きの表（タイムライン）の設計」の 1.。表の行はフレーム、列はトラック（動かし方）:

    {"format": "sprite-motion", "version": 1, "name": "breathe", "loop": true,
     "regions": {"upper": {"type": "split", "at": "auto"}},
     "tracks": {"upper_y": {"type": "deform", "region": "upper", "axis": "y"}},
     "frames": [{"ms": 500}, {"ms": 500, "upper_y": -1}]}

- 範囲（`regions`）は 3 種類:
  - 分け目 1 本（`"type": "split"`。分け目から上が動く）。`"at"` は `"auto"`・下から何ドット目（整数、
    1 = いちばん下の行）・キャラの高さに対する割合（`"35.0%"`）。`"range"`（省くと 2）・`"lines"`（線をそのまま使う）・
    `"used"`（実際に使った高さ。読むときは使わない）・`"keep_small_parts"`（true で、縦の変形のときに目・頬などの小さな部品の
    長さを変えない。省くと false）
  - 帯（`"type": "band"`。`"bottom"`〜`"top"` の行の間だけが動く）と四角（`"type": "rect"`。`"left"`〜`"right"`・
    `"bottom"`〜`"top"` で囲んだ中だけが動く）。高さは `at` と同じ書き方、横は左から何ドット目（1 = いちばん左の列）か、
    キャラの幅に対する割合。`"range"`・`"keep_small_parts"`・`"used"` も書ける（`"at"`・`"lines"` は書けない）。中身は `region.py`
- トラック（`tracks`）は `move`（位置。`axis: "x"`。帯・四角では `axis: "y"` も。+ が下）・`deform`（変形。`axis: "y"`。
  − が縮める、+ が伸ばす）・`root`（体全体のずれ。`axis: "y"`。範囲を持たない）
- 行（`frames`）は 2〜32 行。`"ms"`（表示時間、1 以上）は必ず書く。書いていない列は 0（動かさない）

ここでは画像に関係なく分かる書き間違いだけを確かめ、見つかったものを全部まとめて `SpecError` にする（何も作らない）。
画像に合わない高さ（画像の外）は作るとき（`timeline.py`）に `ValueError`、決まりを守れない行は警告にする。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .base import AUTO, MAX_FRAMES

FORMAT = "sprite-motion"
SPEC_VERSION = 1
MIN_ROWS = 2
MAX_ROWS = MAX_FRAMES
MAX_VALUE = 2               # 列の値の大きさの上限（ドット）
MAX_ROOT_VALUE = 3          # 体全体のずれ（root）の列の上限（弾むの高さ 3 のため。ずらすだけなので形は崩れない）
DEFAULT_RANGE = 2           # 分け目が at から離れてよい量（今の SPLIT_RANGE）
MAX_RANGE = 8
TRACK_TYPES = {"move": ("x",), "deform": ("y",), "root": ("y",)}
TRACK_LABEL = {"move": "位置", "deform": "変形", "root": "体全体のずれ"}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
REGION_KEYS = {"type", "at", "range", "lines", "used", "aim", "keep_small_parts"}
BOX_EDGES = {"band": ("bottom", "top"), "rect": ("left", "right", "bottom", "top")}   # 帯・四角の端（書き出す順）
BOX_KEYS = {"type", "range", "used", "keep_small_parts"}
TRACK_KEYS = {"type", "region", "axis"}
TOP_KEYS = {"format", "version", "name", "loop", "regions", "tracks", "frames"}


class SpecError(ValueError):
    """表の書き間違い（画像に関係なく分かるもの）。`problems` に全部の問題の言葉が入る。"""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("動きの表に書き間違いがあります:\n" + "\n".join(f"- {p}" for p in self.problems))


@dataclass
class Region:
    name: str
    type: str
    at: object                       # "auto" か int（下から何ドット目）か "35.0%"
    range: int = DEFAULT_RANGE
    lines: dict | None = None        # トラックの名前 → 線の並び（列ごとの下から何ドット目）
    used: dict | None = None
    has_range: bool = False          # 表に range が書いてあったか（書き出すときにそのまま戻すため）
    keep_small_parts: bool = False   # 縦の変形で小さな部品（目・頬など）の長さを変えない
    edges: dict | None = None        # 帯・四角の端（"left"・"right"・"bottom"・"top" → 数か割合）


@dataclass
class Track:
    name: str
    type: str
    axis: str
    region: str | None


@dataclass
class Row:
    ms: int
    values: dict                     # トラックの名前 → 値（0 は書かない）


@dataclass
class MotionSpec:
    name: str
    loop: bool
    regions: dict
    tracks: dict
    rows: list = field(default_factory=list)

    def value(self, row: int, track: str) -> int:
        return int(self.rows[row].values.get(track, 0))

    def column(self, track: str) -> list[int]:
        return [self.value(i, track) for i in range(len(self.rows))]

    def used(self, track: str) -> bool:
        """その列に 0 でない値が 1 つでもあるか。"""
        return any(self.column(track))


def fmt_value(v: int) -> str:
    """警告に出す値（−1・+1・0）。"""
    return "0" if v == 0 else (f"+{v}" if v > 0 else f"−{-v}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_percent(text: str) -> float | None:
    """"35.0%" → 35.0。形が違えば None。"""
    m = PERCENT_RE.match(text) if isinstance(text, str) else None
    return float(m.group(1)) if m else None


def _check_at(at, where: str, problems: list[str]) -> None:
    if at == AUTO or (_is_int(at) and at >= 1):
        return
    p = parse_percent(at)
    if p is not None and 0 < p <= 100:
        return
    problems.append(f"{where} の at は \"auto\" か、下から何ドット目かの 1 以上の整数か、\"35.0%\" のような割合"
                    f"（0 より大きく 100 以下）で書いてください: {json.dumps(at, ensure_ascii=False)}")


def _check_edge(v) -> bool:
    if _is_int(v) and v >= 1:
        return True
    p = parse_percent(v)
    return p is not None and 0 <= p <= 100


def _parse_box(rname: str, r: dict, where: str, problems: list[str]) -> Region | None:
    """帯（band）・四角（rect）の範囲を読む。書き間違いは problems に足す。"""
    kind = r["type"]
    edges = BOX_EDGES[kind]
    unknown = sorted(set(r) - BOX_KEYS - set(edges))
    if unknown:
        extra = ""
        if kind == "band" and {"left", "right"} & set(unknown):
            extra = "（左右を決めるときは \"type\": \"rect\"（四角）にしてください）"
        elif {"at", "lines", "aim"} & set(unknown):
            extra = "（at・lines・aim は分け目（split）の範囲の項目です）"
        problems.append(f"{where}: 知らない項目です: {', '.join(unknown)}{extra}")
    ok = True
    for e in edges:
        if e not in r:
            problems.append(f"{where}: {e} を書いてください（{'帯' if kind == 'band' else '四角'}は "
                            f"{'・'.join(edges)} の {len(edges)} つが要ります）")
            ok = False
        elif not _check_edge(r[e]):
            what = "下から" if e in ("bottom", "top") else "左から"
            problems.append(f"{where}: {e} は{what}何ドット目かの 1 以上の整数か、\"40.0%\" のような割合"
                            f"（0 以上 100 以下）で書いてください: {json.dumps(r[e], ensure_ascii=False)}")
            ok = False
    if ok:
        for lo, hi, text in (("bottom", "top", "下が上より上にあります"), ("left", "right", "左が右より右にあります")):
            if lo in edges and _is_int(r[lo]) and _is_int(r[hi]) and r[lo] > r[hi]:
                problems.append(f"{where}: {text}（{lo} {r[lo]}・{hi} {r[hi]}）")
    rng = r.get("range", DEFAULT_RANGE)
    if not _is_int(rng) or not 0 <= rng <= MAX_RANGE:
        problems.append(f"{where}: range は 0〜{MAX_RANGE} の整数で書いてください: {json.dumps(rng, ensure_ascii=False)}")
        rng = DEFAULT_RANGE
    used = r.get("used")
    if used is not None and not isinstance(used, dict):
        problems.append(f"{where}: used は実際に使った範囲（{{\"bottom\": 14, ...}}）の形で書いてください")
    keep = r.get("keep_small_parts", False)
    if not isinstance(keep, bool):
        problems.append(f"{where}: keep_small_parts は true か false で書いてください: {json.dumps(keep, ensure_ascii=False)}")
        keep = False
    return Region(rname, kind, None, rng, None, used, "range" in r, keep, {e: r.get(e) for e in edges})


def parse_spec(data) -> MotionSpec:
    """表（辞書か JSON の文字列）を読み、書き間違いを確かめる。間違いがあれば全部まとめて `SpecError`。"""
    if isinstance(data, MotionSpec):
        return data
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(data)
        except ValueError as e:
            raise SpecError([f"JSON として読めません: {e}"]) from None
    if not isinstance(data, dict):
        raise SpecError(["表は {\"format\": \"sprite-motion\", ...} の形（JSON のオブジェクト）で書いてください"])
    if data.get("format") != FORMAT:
        if "frame_width" in data or "frame_count" in data:
            raise SpecError(["これは動きの表ではありません（スプライトシートの JSON です）。"
                             f"表には \"format\": \"{FORMAT}\" を書きます（--write-motion-spec で書き出せます）"])
        raise SpecError([f"これは動きの表ではありません（\"format\": \"{FORMAT}\" がありません）"])
    problems: list[str] = []
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        problems.append(f"知らない項目です: {', '.join(unknown)}")
    if data.get("version") != SPEC_VERSION or not _is_int(data.get("version")):
        problems.append(f"version は {SPEC_VERSION} にしてください（この版で読める表は version {SPEC_VERSION} だけです）: "
                        f"{json.dumps(data.get('version'), ensure_ascii=False)}")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        problems.append(f"name は英数字と _・- だけで書いてください（シートのファイル名に使います）: "
                        f"{json.dumps(name, ensure_ascii=False)}")
    loop = data.get("loop", True)
    if loop is not True:
        problems.append(f"loop は今は true だけです: {json.dumps(loop, ensure_ascii=False)}")

    regions: dict[str, Region] = {}
    raw_regions = data.get("regions", {})
    if not isinstance(raw_regions, dict):
        problems.append("regions は {\"名前\": {\"type\": \"split\", \"at\": \"auto\"}} の形で書いてください")
        raw_regions = {}
    for rname, r in raw_regions.items():
        where = f"範囲 {rname}"
        if not NAME_RE.match(rname):
            problems.append(f"{where}: 名前は英数字と _・- だけで書いてください")
        if not isinstance(r, dict):
            problems.append(f"{where}: {{\"type\": \"split\", \"at\": \"auto\"}} の形で書いてください")
            continue
        if r.get("type") in BOX_EDGES:
            box = _parse_box(rname, r, where, problems)
            if box is not None:
                regions[rname] = box
            continue
        unknown = sorted(set(r) - REGION_KEYS)
        if unknown:
            problems.append(f"{where}: 知らない項目です: {', '.join(unknown)}")
        if r.get("type") != "split":
            problems.append(f"{where}: type は \"split\"（分け目 1 本）・\"band\"（帯）・\"rect\"（四角）のどれかです: "
                            f"{json.dumps(r.get('type'), ensure_ascii=False)}")
        if "aim" in r:
            problems.append(f"{where}: aim（自動で狙う高さ）はまだ作っていません（うなずくは首の高さを at に数で書きます）")
        at = r.get("at", AUTO)
        _check_at(at, where, problems)
        rng = r.get("range", DEFAULT_RANGE)
        if not _is_int(rng) or not 0 <= rng <= MAX_RANGE:
            problems.append(f"{where}: range は 0〜{MAX_RANGE} の整数で書いてください: {json.dumps(rng, ensure_ascii=False)}")
            rng = DEFAULT_RANGE
        lines = r.get("lines")
        if lines is not None:
            ok = isinstance(lines, dict) and all(
                isinstance(v, list) and v and all(isinstance(line, list) and line and all(_is_int(x) for x in line)
                                                  for line in v) for v in lines.values())
            if not ok:
                problems.append(f"{where}: lines は {{\"列の名前\": [[列ごとの下から何ドット目, ...], ...]}} の形で書いてください")
                lines = None
        used = r.get("used")
        if used is not None and not isinstance(used, dict):
            problems.append(f"{where}: used は実際に使った高さ（{{\"at\": 14, ...}}）の形で書いてください")
        keep = r.get("keep_small_parts", False)
        if not isinstance(keep, bool):
            problems.append(f"{where}: keep_small_parts は true か false で書いてください: {json.dumps(keep, ensure_ascii=False)}")
            keep = False
        regions[rname] = Region(rname, "split", at, rng, lines, used, "range" in r, keep)

    tracks: dict[str, Track] = {}
    raw_tracks = data.get("tracks")
    if not isinstance(raw_tracks, dict) or not raw_tracks:
        problems.append("tracks（動かし方の列）を 1 つ以上、{\"名前\": {\"type\": \"deform\", ...}} の形で書いてください")
        raw_tracks = {}
    for tname, t in raw_tracks.items():
        where = f"列 {tname}"
        if not NAME_RE.match(tname) or tname == "ms":
            problems.append(f"{where}: 名前は英数字と _・- だけで書いてください（ms は使えません）")
        if not isinstance(t, dict):
            problems.append(f"{where}: {{\"type\": \"deform\", \"region\": \"upper\", \"axis\": \"y\"}} の形で書いてください")
            continue
        unknown = sorted(set(t) - TRACK_KEYS)
        if unknown:
            problems.append(f"{where}: 知らない項目です: {', '.join(unknown)}")
        ttype, axis, region = t.get("type"), t.get("axis"), t.get("region")
        if ttype not in TRACK_TYPES:
            problems.append(f"{where}: 知らない種類です（move・deform・root のどれか）: {json.dumps(ttype, ensure_ascii=False)}")
            continue
        box_region = region in regions and regions[region].type in BOX_EDGES
        if ttype == "move" and axis == "y" and not box_region:
            problems.append(f"{where}: 分け目の範囲では縦の位置（move の axis \"y\"）は使えません。"
                            "上下に動かすのは縦の変形（deform）で書いてください")
        elif axis not in TRACK_TYPES[ttype] and not (ttype == "move" and axis == "y"):
            problems.append(f"{where}: {ttype} の axis は {' / '.join(TRACK_TYPES[ttype])} です: "
                            f"{json.dumps(axis, ensure_ascii=False)}")
        if ttype == "root":
            if region is not None:
                problems.append(f"{where}: 体全体のずれ（root）は範囲を持ちません（region を消してください）")
        elif region not in regions:
            problems.append(f"{where}: 範囲 {json.dumps(region, ensure_ascii=False)} が regions にありません")
        tracks[tname] = Track(tname, ttype, axis, None if ttype == "root" else region)
    # 同じ範囲・同じ種類の列が 2 つあると、同じ行でどちらを先に当てるかが決まらない
    seen: dict = {}
    for t in tracks.values():
        key = (t.region, t.type, t.axis)
        if key in seen:
            label = "縦の位置" if (t.type, t.axis) == ("move", "y") else TRACK_LABEL[t.type]
            what = "体全体のずれ（root）の列" if t.type == "root" else f"範囲 {t.region} の{label}の列"
            problems.append(f"{what}が 2 つあります（{seen[key]}・{t.name}）。1 つにまとめてください")
        seen.setdefault(key, t.name)
    for r in regions.values():
        for tname, v in (r.lines or {}).items():
            tname = tname[:-1] if tname.endswith("+") else tname   # 「列の名前+」は伸ばすの線
            if tname not in tracks or tracks[tname].region != r.name:
                problems.append(f"範囲 {r.name}: lines の {tname} は、この範囲の列ではありません")

    rows: list[Row] = []
    raw_rows = data.get("frames")
    if not isinstance(raw_rows, list) or not MIN_ROWS <= len(raw_rows) <= MAX_ROWS:
        n = len(raw_rows) if isinstance(raw_rows, list) else "なし"
        problems.append(f"frames（行）は {MIN_ROWS}〜{MAX_ROWS} 行で書いてください（今は {n}）")
        raw_rows = raw_rows if isinstance(raw_rows, list) else []
    for i, fr in enumerate(raw_rows, start=1):
        where = f"{i} 行目"
        if not isinstance(fr, dict):
            problems.append(f"{where}: {{\"ms\": 500, \"列の名前\": 値}} の形で書いてください")
            continue
        ms = fr.get("ms")
        if not _is_int(ms) or ms < 1:
            problems.append(f"{where}: ms（表示時間、1 以上のミリ秒の整数）を書いてください: {json.dumps(ms, ensure_ascii=False)}")
        values = {}
        for key, v in fr.items():
            if key == "ms":
                continue
            if key not in tracks:
                problems.append(f"{where}: 知らない列です: {key}" + ("" if key in raw_tracks else "（tracks に書いてください）"))
                continue
            limit = MAX_ROOT_VALUE if tracks[key].type == "root" else MAX_VALUE
            if not _is_int(v) or not -limit <= v <= limit:
                problems.append(f"{where}: {key} の値は −{limit}〜+{limit} の整数で書いてください: "
                                f"{json.dumps(v, ensure_ascii=False)}")
                continue
            if v:
                values[key] = int(v)
        # 最初の版で作れない組み合わせ（順番と確かめ方を決めていない）
        moved = sorted({tracks[k].region for k in values if tracks[k].type != "root"})
        if len(moved) > 1:
            problems.append(f"{where}: 1 つの行で 2 つの範囲（{'・'.join(moved)}）を動かすのは、まだ作れません")
        for rname in moved:
            kinds = {tracks[k].type for k in values if tracks[k].region == rname}
            if {"move", "deform"} <= kinds:
                problems.append(f"{where}: 範囲 {rname} で横の位置と縦の変形を同じ行で一緒に使うのは、まだ作れません")
            if len({tracks[k].axis for k in values if tracks[k].region == rname and tracks[k].type == "move"}) > 1:
                problems.append(f"{where}: 範囲 {rname} で横の位置と縦の位置を同じ行で一緒に使うのは、まだ作れません")
        rows.append(Row(int(ms) if _is_int(ms) else 1, values))
    if problems:
        raise SpecError(problems)
    return MotionSpec(name, True, regions, tracks, rows)


def load_spec(path) -> MotionSpec:
    """表のファイルを読む（UTF-8）。読めない・書き間違いは `SpecError`。"""
    from pathlib import Path
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as e:
        raise SpecError([f"表のファイルを読めません: {e}"]) from None
    return parse_spec(text)


def spec_to_dict(spec: MotionSpec, used: dict | None = None) -> dict:
    """表を JSON にする辞書に戻す（値が 0 の列は書かない）。`used` は範囲の名前 → 実際に使った高さ（あれば `"used"` に書く）。"""
    regions = {}
    for r in spec.regions.values():
        d = {"type": r.type, "at": r.at} if r.edges is None else {"type": r.type, **r.edges}
        if r.has_range or r.range != DEFAULT_RANGE:
            d["range"] = r.range
        if r.keep_small_parts:
            d["keep_small_parts"] = True
        if r.lines is not None:
            d["lines"] = r.lines
        u = (used or {}).get(r.name, r.used)
        if u is not None:
            d["used"] = u
        regions[r.name] = d
    tracks = {}
    for t in spec.tracks.values():
        d = {"type": t.type}
        if t.region is not None:
            d["region"] = t.region
        d["axis"] = t.axis
        tracks[t.name] = d
    frames = [{"ms": row.ms, **{k: row.values[k] for k in spec.tracks if row.values.get(k)}} for row in spec.rows]
    out = {"format": FORMAT, "version": SPEC_VERSION, "name": spec.name, "loop": spec.loop}
    if regions:
        out["regions"] = regions
    out["tracks"] = tracks
    out["frames"] = frames
    return out


def _dump(v, indent: int) -> str:
    """JSON を書く。数だけの並び（線など）と行（frames の 1 行）は 1 行に、ほかは字下げして書く。"""
    pad = "  " * indent
    if isinstance(v, dict) and v:
        items = [f'{pad}  {json.dumps(k, ensure_ascii=False)}: {_dump(x, indent + 1)}' for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list) and v and not all(isinstance(x, (int, float, str)) for x in v):
        return "[\n" + ",\n".join(f"{pad}  {_dump(x, indent + 1)}" for x in v) + "\n" + pad + "]"
    return json.dumps(v, ensure_ascii=False)


def spec_json(data: dict) -> str:
    """表のファイルの中身（文字列。保存するときは UTF-8）。行（frames）は 1 行に 1 フレーム、線は 1 本を 1 行で書く。"""
    head = {k: v for k, v in data.items() if k != "frames"}
    text = _dump(head, 0)
    rows = ",\n".join("    " + json.dumps(f, ensure_ascii=False) for f in data.get("frames", []))
    return text[:-2] + ",\n  \"frames\": [\n" + rows + "\n  ]\n}\n"


# ---- 高さの数と割合（docs/sprite-app-plan.md の 7.「別のキャラで使うときの高さ」） ----

def body_rows(rgba: np.ndarray) -> tuple[int, int] | None:
    """(キャラの下の透明の行の数, キャラの高さ)。キャラの高さは、いちばん上の不透明な行からいちばん下の不透明な行まで。"""
    rows = np.flatnonzero((rgba[..., 3] > 0).any(axis=1))
    if len(rows) == 0:
        return None
    return int(rgba.shape[0] - 1 - rows[-1]), int(rows[-1] - rows[0] + 1)


def _half_up(v: float) -> int:
    return int(math.floor(v + 0.5 + 1e-9))


def percent_to_height(rgba: np.ndarray, text: str) -> int | None:
    """割合（"35.0%"）を「下から何ドット目」にする: 下の透明の行の数 + 四捨五入（p / 100 × キャラの高さ）（1 より小さければ 1）。"""
    b = body_rows(rgba)
    p = parse_percent(text)
    if b is None or p is None:
        return None
    blank, height = b
    return blank + max(1, _half_up(p / 100 * height))


def height_to_percent(rgba: np.ndarray, n: int) -> str | None:
    """「下から何ドット目」を割合にする（小数 1 桁。読み直して同じ数に戻らなければ桁を増やす）。"""
    b = body_rows(rgba)
    if b is None:
        return None
    blank, height = b
    p = (n - blank) / height * 100
    for digits in range(1, 7):
        text = f"{p:.{digits}f}%"
        if percent_to_height(rgba, text) == n:
            return text
    return None


def with_height_style(data: dict, rgba: np.ndarray, style: str) -> dict:
    """書き出す表の高さを数（`number`）か割合（`percent`）にそろえる（`"auto"` と `used` はそのまま）。

    割合にできない高さ（キャラの下の透明の行より下・絵が無い）は数のまま残す。
    """
    if style not in ("number", "percent"):
        raise ValueError(f"高さの書き方は number か percent です: {style}")
    out = json.loads(json.dumps(data))
    for r in out.get("regions", {}).values():
        if r.get("type") in BOX_EDGES:
            _box_style(r, rgba, style)
            continue
        at = r.get("at")
        if style == "number" and isinstance(at, str) and parse_percent(at) is not None:
            n = percent_to_height(rgba, at)
            if n is not None:
                r["at"] = n
        elif style == "percent" and _is_int(at):
            p = height_to_percent(rgba, at)
            if p is not None and parse_percent(p) and 0 < parse_percent(p) <= 100:
                r["at"] = p
    return out


def _box_style(r: dict, rgba: np.ndarray, style: str) -> None:
    """帯・四角の端を数か割合にそろえる（`with_height_style()`）。割合にできない端は数のまま。"""
    from .region import edge_number, edge_percent
    for e in BOX_EDGES[r["type"]]:
        v = r.get(e)
        if style == "number" and isinstance(v, str) and parse_percent(v) is not None:
            n = edge_number(rgba, e, v)
            if n is not None:
                r[e] = n
        elif style == "percent" and _is_int(v):
            p = edge_percent(rgba, e, v)
            if p is not None and 0 <= parse_percent(p) <= 100:
                r[e] = p
