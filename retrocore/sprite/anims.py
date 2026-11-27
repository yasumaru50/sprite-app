"""動きに名前を付けて、1 つのスプライトシートにまとめる（「コマが並んだ絵」の「動きに名前を付けてまとめる」。CLI の --anims）。

    from retrocore.sprite.anims import config_from_dict, convert_anims, assemble_anims
    cfg = config_from_dict({"anims": [{"name": "walk", "files": ["walk.png"]},
                                      {"name": "jump", "files": ["jump.png"], "align_y": "keep", "loop": False}]})
    res = convert_anims([("walk.png", rgba1), ("jump.png", rgba2)], SpriteSettings(), cfg)
    sheet = assemble_anims(res, cfg)     # 1 つの動き = 1 行のシート（AnimsSheet）
    sheet.info                           # JSON の辞書（"animations" に動きごとのフレームの番号・時間・くり返すか・行）

ゲームのエンジン（Godot の SpriteFrames など）で、1 つのキャラに idle・walk・run などのアニメーションを持たせて切り替えやすくするため。
「別々のファイルの絵をつなげる」（`join.py`）を広げたもの:

- **全部の動きで共通**: パレット（全部のコマから 1 つ）、コマの大きさ（いちばん大きい動きに合わせて透明で広げる。拡大・縮小はしない）、
  地面の高さ（`ground_y`。全部の動きでフレームの下端）、横のそろえ方、左右反転、高さ合わせ（`match_height`）。
  グリッドは今までどおりファイルごとに見つける（`join.prepare_files()`）
- **動きごとに決める**: 順番（その動きの中での番号。1,2,3,2r など）、1 コマの時間、くり返す／1 回で終わる、縦のそろえ方
  （足元・頭の高さ・元の絵の高さを保つ）と上げ下げ（`shift_y`）
- 動きごとに `frames.align_frames()` でそろえてから、そろえた基準（pivot の横の位置）と下端（地面）が全部の動きで同じ所に来るように、
  透明の列・行を足して 1 つのキャンバスにする。基準が半ドットの所にある動きと整数の所にある動きが混ざると、半ドットはそろえられない
  （ドットを動かさないため。pivot はいちばん右の動きのものを書く）
- 1 つのファイルは 1 つの動きにだけ使える。まばたき・口パク・重ね・1 コマ目と同じにする範囲・ちらつきを減らす・体の上下（root_y）は
  使えない（`UNSUPPORTED`。CLI は警告して無視する）

`retrocore` の方針どおり NumPy と Pillow と標準ライブラリだけを使う（Pyodide で動く）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from . import frames as F
from .pipeline import StepResult

CONFIG_KIND = "sprite_anims"
CONFIG_VERSION = 1
DEFAULT_NAME = "anims"                 # シートのファイル名と JSON の motion に使う名前（--frame-name の代わり）
NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
NAME_MAX = 32
SUGGESTED = ("idle", "walk", "run", "jump", "turn", "wave")   # 画面の入力の候補
MAX_ANIMS = 16
# 動きをまとめるときに使えないもの（理由は画面にも出す）
UNSUPPORTED = {
    "swaps": "まばたき・口パク（差し替え）は、動きをまとめるときは使えません（どの動きのどのコマに入れるかを決められないため）",
    "fx": "重ね（エフェクト）は、動きをまとめるときは使えません（キャラのシートが 1 つの動きのときだけ）",
    "same": "1 コマ目と同じにする範囲は、動きをまとめるときは使えません（動きごとに 1 コマ目が違うため）",
    "steady": "ちらつきを減らすは、動きをまとめるときは使えません（動きごとに 1 コマ目が違うため）",
    "flicker": "ちらつきの見える化は、動きをまとめるときは作りません（動きごとに 1 コマ目が違うため）",
    "root_y": "体の上下（root_y）は、動きをまとめるときは使えません（動きごとに上下の余白が変わり、地面の高さがそろわなくなるため。"
              "宙に浮くコマは縦のそろえ方「元の絵の高さを保つ」と上げ下げで作ります）",
}


@dataclass
class AnimSpec:
    """1 つの動き（名前・使うファイル・動きごとの設定）。"""

    name: str
    files: list                       # ファイルの名前（並べた順）
    order: str | None = None          # 順番（その動きの中での番号。"1,2,3,2r"。None は 1, 2, …）
    durations: list = field(default_factory=lambda: [F.DEFAULT_DURATION])   # 1 つか、順番と同じ数（ミリ秒）
    loop: bool = True                 # くり返す（False は 1 回で終わる）
    align_y: str = F.DEFAULT_ALIGN_Y  # 縦のそろえ方（foot / match / keep / none）
    shift_y: list | None = None       # 元のコマごとの上げ下げ（ドット。− が上。数はその動きのコマの数）


@dataclass
class AnimsConfig:
    """動きをまとめる設定（`<名前>.anims.json` の中身。CLI の --anims と画面で同じ）。"""

    anims: list                       # AnimSpec の並び（シートの行の順）
    name: str = DEFAULT_NAME
    align: str = F.DEFAULT_ALIGN      # 横のそろえ方（全部の動きで共通）
    grid: str = F.DEFAULT_GRID_MODE   # グリッドの線の並び（--frame-grid）
    match_height: bool = False        # キャラの高さをいちばん小さいファイルに合わせる（--join-match-height）
    mirror: bool = False              # 左右反転のシートも作る（--mirror）
    settings: dict | None = None      # 変換の設定（SpriteSettings の項目。画面で保存したとき。無ければ CLI の引数）
    canvas_multiple: int = 0          # キャンバスを倍数にそろえる（0 はそろえない。canvas.py。--canvas-multiple）
    canvas_align: str = "bottom"      # その並べ方（bottom = 下ぞろえ・左右中央 / center = 上下左右中央。--canvas-align）

    def files(self) -> list:
        return [f for a in self.anims for f in a.files]


# ---------------------------------------------------------------- 設定

def check_name(name, what: str = "動きの名前") -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or len(name) > NAME_MAX:
        raise ValueError(f"{what}は英小文字で始まり、英小文字・数字・_ だけで書いてください（{NAME_MAX} 文字まで。"
                         f"例 {' / '.join(SUGGESTED)}）: {name!r}")
    return name


def group_rows(files: list, names: list) -> list:
    """ファイルの一覧の各行の「動きの名前」から、動きの並び [(名前, [ファイル, ...]), ...] を作る（画面の一覧と同じ決まり）。

    同じ名前のファイルが続けば 1 つの動き。同じ名前が離れて出てきたらエラー（並べ替えを促す）。全部のファイルに名前が要る。
    """
    if len(files) != len(names):
        raise ValueError("ファイルと動きの名前の数が合いません")
    out: list = []
    for f, n in zip(files, names):
        n = (n or "").strip() if isinstance(n, str) else n
        if not n:
            raise ValueError(f"「{f}」に動きの名前がありません（全部のファイルに名前を入れてください）")
        check_name(n)
        if out and out[-1][0] == n:
            out[-1][1].append(f)
        elif any(a == n for a, _ in out):
            raise ValueError(f"動き「{n}」のファイルが離れています（「{f}」）。同じ動きのファイルは続けて並べてください（▲▼ で並べ替え）")
        else:
            out.append((n, [f]))
    return out


def _bool(v, what):
    if not isinstance(v, bool):
        raise ValueError(f"{what}は true か false で書いてください: {v!r}")
    return v


def _ints(v, what, signed=False):
    vals = [v] if isinstance(v, int) and not isinstance(v, bool) else v
    if not isinstance(vals, list) or not vals or any(isinstance(x, bool) or not isinstance(x, int) for x in vals):
        raise ValueError(f"{what}は整数か整数の並びで書いてください: {v!r}")
    if not signed and any(x < 1 for x in vals):
        raise ValueError(f"{what}は 1 以上にしてください: {v!r}")
    return [int(x) for x in vals]


def config_from_dict(d) -> AnimsConfig:
    """設定のファイル（JSON の辞書）を読んで確かめる。"""
    if not isinstance(d, dict):
        raise ValueError("動きをまとめる設定は {\"anims\": [...]} の形で書いてください")
    unknown = sorted(set(d) - {"version", "kind", "name", "settings", "common", "anims"})
    if unknown:
        raise ValueError(f"知らない設定です: {', '.join(unknown)}")
    if d.get("kind", CONFIG_KIND) != CONFIG_KIND:
        raise ValueError(f"この JSON は動きをまとめる設定（kind: {CONFIG_KIND}）ではありません: kind = {d.get('kind')!r}")
    if d.get("version", CONFIG_VERSION) != CONFIG_VERSION:
        raise ValueError(f"動きをまとめる設定の版 {d.get('version')!r} は読めません（読めるのは {CONFIG_VERSION}）")
    name = d.get("name", DEFAULT_NAME)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"名前は半角の英数字と _ - で書いてください: {name!r}")
    common = d.get("common") or {}
    if not isinstance(common, dict):
        raise ValueError("common（全部の動きで共通の設定）は {...} で書いてください")
    unknown = sorted(set(common) - {"align", "grid", "match_height", "mirror", "canvas_multiple", "canvas_align"})
    if unknown:
        raise ValueError(f"知らない共通の設定です: {', '.join(unknown)}")
    settings = d.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("settings（変換の設定）は {...} で書いてください")
    items = d.get("anims")
    if not isinstance(items, list) or not items:
        raise ValueError("anims（動きの並び）が空です")
    if len(items) > MAX_ANIMS:
        raise ValueError(f"動きは {MAX_ANIMS} 個までです（いま {len(items)} 個）")
    anims, seen, used = [], set(), {}
    for i, a in enumerate(items):
        if not isinstance(a, dict):
            raise ValueError(f"{i + 1} 番目の動きは {{\"name\": ..., \"files\": [...]}} の形で書いてください")
        unknown = sorted(set(a) - {"name", "files", "order", "duration_ms", "loop", "align_y", "shift_y"})
        if unknown:
            raise ValueError(f"動き {a.get('name', i + 1)}: 知らない設定です: {', '.join(unknown)}")
        n = check_name(a.get("name"))
        if n in seen:
            raise ValueError(f"動きの名前「{n}」が 2 つあります（名前は重ならないようにしてください）")
        seen.add(n)
        files = a.get("files")
        if isinstance(files, str):
            files = [files]
        if not isinstance(files, list) or not files or any(not isinstance(f, str) or not f for f in files):
            raise ValueError(f"動き {n}: files（使うファイルの名前）を 1 つ以上書いてください")
        for f in files:
            if f in used:
                raise ValueError(f"ファイル「{f}」が動き {used[f]} と {n} の両方にあります（1 つのファイルは 1 つの動きにだけ使えます）")
            used[f] = n
        order = a.get("order")
        if order is not None:
            if isinstance(order, list):
                order = F.order_text(*F.parse_order_flips(order, None))
            if not isinstance(order, str):
                raise ValueError(f"動き {n}: order は \"1,2,3,2\" のように書いてください: {order!r}")
            order = order.replace(" ", "") or None
            if order is not None:
                F.parse_order_flips(order, None)
        durations = _ints(a.get("duration_ms", F.DEFAULT_DURATION), f"動き {n}: duration_ms（1 コマの時間）")
        align_y = a.get("align_y", F.DEFAULT_ALIGN_Y)
        if align_y not in F.ALIGNS_Y:
            raise ValueError(f"動き {n}: 縦のそろえ方は {' / '.join(F.ALIGNS_Y)} のどれかです: {align_y!r}")
        sy = a.get("shift_y")
        if sy is not None:
            sy = _ints(sy, f"動き {n}: shift_y（上げ下げ）", signed=True)
            if any(abs(v) > F.MAX_SHIFT_Y for v in sy):
                raise ValueError(f"動き {n}: 上げ下げは −{F.MAX_SHIFT_Y}〜{F.MAX_SHIFT_Y} ドットで書いてください: {sy}")
        anims.append(AnimSpec(name=n, files=list(files), order=order, durations=durations,
                              loop=_bool(a.get("loop", True), f"動き {n}: loop"), align_y=align_y, shift_y=sy))
    cfg = AnimsConfig(anims=anims, name=name, align=common.get("align", F.DEFAULT_ALIGN),
                      grid=common.get("grid", F.DEFAULT_GRID_MODE),
                      match_height=_bool(common.get("match_height", False), "match_height"),
                      mirror=_bool(common.get("mirror", False), "mirror"), settings=settings)
    from . import canvas as C
    cfg.canvas_multiple, cfg.canvas_align = C.check(common.get("canvas_multiple", 0), common.get("canvas_align"))
    if cfg.align not in F.ALIGNS:
        raise ValueError(f"横のそろえ方は {' / '.join(F.ALIGNS)} のどれかです: {cfg.align!r}")
    if cfg.grid not in F.GRID_MODES:
        raise ValueError(f"グリッドの線の並びは {' / '.join(F.GRID_MODES)} のどれかです: {cfg.grid!r}")
    from .join import MAX_FILES
    if len(cfg.files()) > MAX_FILES:
        raise ValueError(f"ファイルは全部で {MAX_FILES} 個までです（いま {len(cfg.files())} 個）")
    return cfg


def load_config(text) -> AnimsConfig:
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8-sig")
    try:
        d = json.loads(text)
    except ValueError as e:
        raise ValueError(f"動きをまとめる設定の JSON を読めません: {e}") from None
    return config_from_dict(d)


def config_dict(cfg: AnimsConfig) -> dict:
    """設定のファイルの辞書（画面の「設定を保存」と同じ形。既定の値も全部書く）。"""
    out = {"version": CONFIG_VERSION, "kind": CONFIG_KIND, "name": cfg.name}
    if cfg.settings is not None:
        out["settings"] = dict(cfg.settings)
    out["common"] = {"align": cfg.align, "grid": cfg.grid, "match_height": bool(cfg.match_height), "mirror": bool(cfg.mirror)}
    if cfg.canvas_multiple:   # キャンバスを倍数にそろえるときだけ（そろえない設定のファイルは前と同じ）
        out["common"]["canvas_multiple"] = int(cfg.canvas_multiple)
        out["common"]["canvas_align"] = cfg.canvas_align
    out["anims"] = [{"name": a.name, "files": list(a.files), "order": a.order,
                     "duration_ms": a.durations[0] if len(a.durations) == 1 else list(a.durations), "loop": bool(a.loop),
                     "align_y": a.align_y, "shift_y": None if a.shift_y is None else list(a.shift_y)} for a in cfg.anims]
    return out


def config_json(cfg: AnimsConfig) -> str:
    return json.dumps(config_dict(cfg), ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------- 変換

def _round_half(v: float) -> int:
    return int(np.floor(2 * float(v) + 0.5))


def convert_anims(images: list, settings=None, cfg: AnimsConfig | None = None, edits=None, on_step=None):
    """動きごとのファイル（`images` は (ファイルの名前, RGBA) の並び。名前は設定の files と同じ）を、1 つのキャラとして変換し、
    動きごとにそろえてから 1 つのキャンバスに合わせる。戻り値は `frames.FramesResult`（コマは全部のファイルの全部のコマ。
    設定のファイルの順。`anims` に動きごとのコマの範囲）。作らなかったときは `frames.FramesError` か `GridNotFoundError`。
    """
    from .convert import SpriteSettings
    from .join import join_report, prepare_files
    if cfg is None:
        raise ValueError("動きをまとめる設定がありません")
    s = settings or SpriteSettings()
    by_name: dict = {}
    for n, rgba in images:
        key = str(n).replace("\\", "/").rsplit("/", 1)[-1]
        if key in by_name:
            raise ValueError(f"同じ名前のファイルが 2 つあります: {key}（名前を変えてください）")
        by_name[key] = rgba
    missing = [f for f in cfg.files() if f not in by_name]
    if missing:
        raise ValueError(f"設定に書いたファイルがありません: {'、'.join(missing)}")
    extra = [k for k in by_name if k not in cfg.files()]
    if extra:
        raise ValueError(f"どの動きにも入っていないファイルがあります: {'、'.join(extra)}")
    fs = F.FrameSettings(align=cfg.align, grid=cfg.grid, edits=edits)
    p = prepare_files([(f, by_name[f]) for f in cfg.files()], s, fs, on_step, cfg.match_height)
    t0 = time.perf_counter()
    # 動きごとのコマの範囲（ファイルの順に、動きごとに続いている）
    recs, k, fi = [], 1, 0
    for a in cfg.anims:
        fl = p.files[fi:fi + len(a.files)]
        fi += len(a.files)
        n = sum(f.frame_count for f in fl)
        recs.append({"name": a.name, "first": k, "count": n, "files": [f.name for f in fl]})
        k += n
    placed = []
    for a, r in zip(cfg.anims, recs):
        i0, i1 = r["first"] - 1, r["first"] - 1 + r["count"]
        if a.shift_y is not None and len(a.shift_y) != r["count"]:
            raise ValueError(f"動き {a.name}: 上げ下げの数（{len(a.shift_y)}）は、この動きのコマの数（{r['count']}）と同じにしてください")
        groups = [g for g, f in enumerate(r["files"]) for _ in range(next(x.frame_count for x in p.files if x.name == f))]
        got = F.align_frames(p.raw[i0:i1], cfg.align, a.align_y, F.HEAD_RATIO, p.pos[i0:i1], groups, a.shift_y)
        placed.append(got)
    # 基準（pivot の横の位置）と下端をそろえて 1 つのキャンバスにする
    p2 = [_round_half(g[2][0]) for g in placed]
    t2 = max(p2)
    sx = [(t2 - v) // 2 for v in p2]
    H = max(g[0][0].shape[0] for g in placed)
    W = max(x + g[0][0].shape[1] for x, g in zip(sx, placed))
    frames, offs, anchors = [], [], []
    for (fr, of, _pv, an), x in zip(placed, sx):
        h, w = fr[0].shape[:2]
        y = H - h
        for f in fr:
            c = np.zeros((H, W, 4), np.uint8)
            c[y:y + h, x:x + w] = f
            frames.append(c)
        offs += [(int(o[0]) + x, int(o[1]) + y) for o in of]
        anchors += [float(v) + x for v in an]
    pivot = (t2 / 2, float(H))
    p.emit("align", "動きごとにそろえて 1 つの大きさにする（" + "、".join(
        f"{a.name} {r['count']} コマ・縦 {a.align_y}" + (f"・上げ下げ {','.join(map(str, a.shift_y))}" if a.shift_y else "")
        for a, r in zip(cfg.anims, recs)) + f"、{W}×{H} ドット）", np.concatenate(frames, 1), t0)
    warnings, skipped, joins = list(p.warnings), list(p.skipped), []
    for a, r in zip(cfg.anims, recs):
        fl = [f for f in p.files if f.name in r["files"]]
        if len(fl) < 2:
            continue
        out, jw, js = join_report(frames, fl, F.HEAD_RATIO)
        keep = [not e["loop"] or a.loop for e in out]
        joins += [e for e, ok in zip(out, keep) if ok]
        for m, e in zip(jw, js):
            if not e["loop"] or a.loop:
                warnings.append(f"{a.name}: {m}")
                skipped.append(dict(e, message=f"{a.name}: {e['message']}", animation=a.name))
    files = list(p.files)
    res = F.FramesResult(frames=frames, raw=p.raw, boxes=p.boxes, grids=p.grids, dot_size=files[0].dot_size, offsets=offs,
                         pivot=pivot, align=cfg.align, align_y=F.DEFAULT_ALIGN_Y, anchors=anchors, steps=p.steps,
                         warnings=warnings, skipped=skipped, notes=p.notes, aligned=frames, files=files, sources=p.sources,
                         joins=joins, anims=recs)
    return F._with_edits(res, fs, p.emit)


# ---------------------------------------------------------------- シート

@dataclass
class AnimsSheet:
    """まとめたシート（1 つの動き = 1 行）。"""

    name: str                 # シートの名前（JSON の motion。反転は _mirror）
    anims: list               # 動きごとの MotionResult（全部同じ大きさ。name は動きの名前）
    specs: list               # AnimSpec の並び
    info: dict = field(default_factory=dict)   # JSON の辞書（`sheet_file_name()` の名前で作ったもの）
    image_name: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return self.anims[0].size

    @property
    def frames(self) -> list:
        return [f for a in self.anims for f in a.frames]

    @property
    def columns(self) -> int:
        return max(len(a.frames) for a in self.anims)

    def image(self) -> np.ndarray:
        """シートの画像（行ごとに 1 つの動き。余りは透明）。"""
        w, h = self.size
        out = np.zeros((h * len(self.anims), w * self.columns, 4), np.uint8)
        for r, a in enumerate(self.anims):
            for c, f in enumerate(a.frames):
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
        return out


def _global_order(a: AnimSpec, rec: dict) -> list:
    text = a.order or ",".join(str(i + 1) for i in range(rec["count"]))
    try:
        nums, flips = F.parse_order_flips(text, rec["count"])
    except ValueError as e:
        raise ValueError(f"動き {a.name}: {e}") from None
    return [f"{n + rec['first'] - 1}{'r' if fl else ''}" for n, fl in zip(nums, flips)]


def sheet_file_name(stem: str, sheet: AnimsSheet) -> str:
    """`<出力名>_<名前>_<幅>x<高さ>_<N>f.png`（N は全部の動きのフレームの数。コマのシートと同じ形）。"""
    w, h = sheet.size
    return f"{stem}_{sheet.name}_{w}x{h}_{len(sheet.frames)}f.png"


def assemble_anims(res, cfg: AnimsConfig, stem: str = "sprite", mirror: bool = False) -> AnimsSheet:
    """変換の結果（`convert_anims()`）と設定から、まとめたシート（PNG の絵と JSON の辞書）を作る。"""
    if getattr(res, "anims", None) is None or [r["name"] for r in res.anims] != [a.name for a in cfg.anims]:
        raise ValueError("変換したときと動きの並びが違います（もう一度変換してください）")
    ms = []
    for a, rec in zip(cfg.anims, res.anims):
        r = dataclasses.replace(res, align_y=a.align_y, shift_y=None)
        try:
            m = F.assemble_frames(r, _global_order(a, rec), a.durations, None, a.name, mirror)
        except ValueError as e:
            raise ValueError(e if str(e).startswith(f"動き {a.name}") else f"動き {a.name}: {e}") from None
        ms.append(m)
    # 裏返したコマ（r）で列を足した動きがあれば、ほかの動きにも同じだけ足す（基準の横の位置を全部の動きでそろえる）
    L = max(float(m.info["pivot_x"]) for m in ms)
    lefts = [int(round(L - float(m.info["pivot_x"]))) for m in ms]
    W = max(l + m.size[0] for l, m in zip(lefts, ms))
    for m, l in zip(ms, lefts):
        w, h = m.size
        if l or W - l - w:
            m.frames = [np.pad(f, ((0, 0), (l, W - l - w), (0, 0))) for f in m.frames]
        m.info["pivot_x"] = L
    if cfg.canvas_multiple:   # キャンバスを倍数にそろえる（全部の動きのコマに同じ余白。canvas.py）
        from . import canvas as C
        ms = [C.pad_result(m, cfg.canvas_multiple, cfg.canvas_align) for m in ms]
    sheet = AnimsSheet(name=cfg.name + ("_mirror" if mirror else ""), anims=ms, specs=list(cfg.anims))
    sheet.image_name = sheet_file_name(stem, sheet)
    sheet.info = _sheet_info(sheet, res, sheet.image_name)
    return sheet


def _sheet_info(sheet: AnimsSheet, res, image_name: str) -> dict:
    from .motion.base import MotionResult
    ms = sheet.anims
    info = {"source_frames": [k for m in ms for k in m.info["source_frames"]], "align": res.align,
            "align_y": res.align_y, "pivot_x": ms[0].info["pivot_x"], "ground": True}
    if "canvas" in ms[0].info:
        info["canvas"] = ms[0].info["canvas"]
    if any("flipped" in m.info for m in ms):
        info["flipped"] = [fl for m in ms for fl in m.info.get("flipped", [False] * len(m.frames))]
    if res.touchup is not None:
        info["edits"] = res.touchup.info()
    if res.joined:
        info["files"] = [f.info() for f in res.files]
        info["source_files"] = [list(res.sources[k - 1]) for k in info["source_frames"]]
        info["joins"] = list(res.joins or [])
    big = MotionResult(name=sheet.name, frames=sheet.frames, durations=[d for m in ms for d in m.durations], info=info)
    d = F.frames_sheet_info(big, image_name, res.skipped, res.dot_size)
    w, h = sheet.size
    cols = sheet.columns
    anims, i = {}, 0
    for r, (m, a) in enumerate(zip(ms, sheet.specs)):
        n = len(m.frames)
        for c in range(n):
            f = d["frames"][i + c]
            f["x"], f["y"] = c * w, r * h
            f["animation"] = a.name
        ds = [int(v) for v in m.durations]
        base = max(set(ds), key=lambda v: (ds.count(v), -ds.index(v)))   # いちばん多い時間（同じ数なら先に出る方）
        anims[a.name] = {"frames": list(range(i, i + n)), "duration_ms": base, "loop": bool(a.loop), "row": r}
        i += n
    d["columns"], d["rows"] = cols, len(ms)
    d["loop"] = all(a.loop for a in sheet.specs)
    out = {}
    for k, v in d.items():
        if k == "frames":
            out["animations"] = anims
        out[k] = v
    src = out["source"]
    src["frame_count"] = len(res.frames)
    src.pop("align_y", None)
    src["anims"] = [{"name": a.name, "files": list(rec["files"]), "first_frame": int(rec["first"]),
                     "frame_count": int(rec["count"]), "order": F.order_text(*F.parse_order_flips(
                         a.order or ",".join(str(j + 1) for j in range(rec["count"])), rec["count"])),
                     "align_y": a.align_y, "shift_y": None if a.shift_y is None else list(a.shift_y)}
                    for a, rec in zip(sheet.specs, res.anims)]
    return out


def sheet_json(sheet: AnimsSheet) -> str:
    return json.dumps(sheet.info, ensure_ascii=False, indent=2) + "\n"


def _png(rgba: np.ndarray) -> bytes:
    from .convert import encode_png
    return encode_png(rgba)


def anim_stem(info: dict) -> str:
    """まとめたシートの JSON から、動きごとのファイルの名前の頭（<出力名>）を取り出す。"""
    image = PurePosixPath(str(info.get("image") or "sprite.png").replace("\\", "/")).name
    stem = image[:-4] if image.lower().endswith(".png") else image
    from .edit import EDIT_SUFFIX
    if stem.endswith(EDIT_SUFFIX):
        stem = stem[:-len(EDIT_SUFFIX)]
    motion = str(info.get("motion") or "")
    tail = f"_{motion}_{info.get('frame_width')}x{info.get('frame_height')}_{len(info.get('frames') or [])}f"
    if motion and stem.endswith(tail) and len(stem) > len(tail):
        return stem[:-len(tail)]
    return stem


def anim_file_stem(info: dict, anim: str) -> str:
    """動きごとのファイル（GIF・.piskel）の名前（拡張子なし）: <出力名>_<動き>（反転のシートでは _mirror を足す）。"""
    mirror = str(info.get("motion") or "").endswith("_mirror")
    return f"{anim_stem(info)}_{anim}" + ("_mirror" if mirror else "")


def anim_info(info: dict, anim: str, image_name: str | None = None) -> dict:
    """まとめたシートの JSON から、1 つの動きだけのシート（横 1 列。コマのシートと同じ形）の JSON の辞書を作る（読み戻し・GIF 用）。"""
    anims = info.get("animations")
    if not isinstance(anims, dict) or anim not in anims:
        names = " / ".join(anims) if isinstance(anims, dict) else ""
        raise ValueError(f"シートの情報に動き「{anim}」がありません（ある動き: {names}）")
    a = anims[anim]
    fw, fh = int(info["frame_width"]), int(info["frame_height"])
    idx = a.get("frames")
    if not isinstance(idx, list) or not idx or any(isinstance(i, bool) or not isinstance(i, int)
                                                    or not 0 <= i < len(info["frames"]) for i in idx):
        raise ValueError(f"動き「{anim}」のフレームの番号が読めません: {idx!r}")
    frames = []
    for j, i in enumerate(idx):
        f = copy.deepcopy(info["frames"][i])
        f["index"], f["x"], f["y"] = j, j * fw, 0
        f.pop("animation", None)
        frames.append(f)
    out = {"version": info.get("version"), "motion": anim, "image": image_name or f"{anim_file_stem(info, anim)}.png",
           "frame_width": fw, "frame_height": fh, "frame_count": len(frames), "columns": len(frames), "rows": 1,
           "loop": bool(a.get("loop", True)), "pivot": copy.deepcopy(info.get("pivot"))}
    if "ground_y" in info:
        out["ground_y"] = info["ground_y"]
    if "canvas" in info:
        out["canvas"] = copy.deepcopy(info["canvas"])
    out["total_duration_ms"] = int(sum(int(f["duration_ms"]) for f in frames))
    out["frames"] = frames
    if "source" in info:
        out["source"] = copy.deepcopy(info["source"])
    if "palette" in info:
        out["palette"] = list(info["palette"])
    out["skipped"] = []
    return out


def anim_frames(sheet: np.ndarray, info: dict, anim: str) -> tuple[list, list]:
    """まとめたシートの絵から、1 つの動きのフレームと時間を取り出す。"""
    from .motion.gif import frames_from_sheet
    i = info["animations"][anim]["frames"]
    sub = {"frame_width": info["frame_width"], "frame_height": info["frame_height"],
           "frames": [info["frames"][k] for k in i]}
    return frames_from_sheet(sheet, sub)


def anim_gif(sheet: np.ndarray, info: dict, anim: str, scale=None, background=None):
    """1 つの動きの GIF（くり返さない動きは 1 回だけ再生する）。戻り値は (ファイルの名前, GifResult)。"""
    from .motion.gif import DEFAULT_SCALE, make_gif
    frames, durations = anim_frames(sheet, info, anim)
    loop = bool(info["animations"][anim].get("loop", True))
    g = make_gif(frames, durations, DEFAULT_SCALE if scale is None else scale, background, loop=loop)
    return anim_file_stem(info, anim) + ".gif", g


def anim_piskel(sheet: np.ndarray, info: dict, anim: str) -> tuple[str, str, int, list]:
    """1 つの動きの .piskel（Piskel で直す用）。戻り値は (ファイルの名前, 中身, FPS, メモ)。"""
    from .piskel import fps_for_durations, piskel_text
    frames, durations = anim_frames(sheet, info, anim)
    fps, notes = fps_for_durations(durations)
    stem = anim_file_stem(info, anim)
    return stem + ".piskel", piskel_text(frames, stem, fps), fps, notes


def anims_outputs(res, cfg: AnimsConfig, stem: str, piskel: bool = True) -> dict:
    """まとめたシートのファイル（CLI と画面で同じ）。{"sheets": [AnimsSheet（反転は 2 つ目）],
    "files": {ファイルの名前: バイト}（シートの PNG・JSON、反転、.piskel（動きごと））}。GIF は `anim_gif()` で別に作る。"""
    sheets = [assemble_anims(res, cfg, stem)]
    if cfg.mirror:
        sheets.append(assemble_anims(res, cfg, stem, mirror=True))
    files: dict = {}
    for sh in sheets:
        files[sh.image_name] = _png(sh.image())
        files[sh.image_name[:-4] + ".json"] = sheet_json(sh).encode("utf-8")
    if piskel:
        img = sheets[0].image()
        for a in cfg.anims:
            name, text, _, _ = anim_piskel(img, sheets[0].info, a.name)
            files[name] = text.encode("ascii")
    return {"sheets": sheets, "files": files}


# ---------------------------------------------------------------- 読み戻し

def guess_anim(file_name: str, names: list) -> str | None:
    """直したファイルの名前（<出力名>_<動き>.piskel など）から、どの動きかを決める。分からなければ None。"""
    stem = PurePosixPath(str(file_name).replace("\\", "/")).name
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)          # ブラウザが付ける「 (1)」
    for suf in ("_edited", "_mirror"):
        while stem.endswith(suf):
            stem = stem[:-len(suf)]
    hits = [n for n in names if stem == n or stem.endswith("_" + n)]
    if not hits:
        return None
    hits.sort(key=len, reverse=True)
    if len(hits) > 1 and len(hits[0]) == len(hits[1]):
        return None
    return hits[0]


def readback_anims(edited: bytes, edited_name: str, info: dict, original: bytes | None, opt):
    """まとめたシートの 1 つの動きを読み戻す（`edit.readback()` から呼ぶ。直したのは 1 つの動きだけのシートか .piskel）。

    直す前のまとめたシート（`original`）が要る（ほかの動きのコマはそこから使う）。作るのは、まとめたシート（その動きの行を直したもの）と
    JSON、反転（`opt.mirror`）、その動きの GIF・.piskel・確かめた結果。
    """
    from . import edit as E
    names = list(info["animations"])
    anim = opt.anim or guess_anim(edited_name, names)
    if anim is None:
        raise E.EditError("anim_unknown", "どの動きを直したのか、ファイルの名前から分かりません。動きを選んでください"
                          f"（CLI は --anim。ある動き: {' / '.join(names)}）")
    if anim not in names:
        raise E.EditError("anim_unknown", f"シートの情報に動き「{anim}」がありません（ある動き: {' / '.join(names)}）")
    if original is None:
        raise E.EditError("need_original", "まとめたシートの読み戻しには、直す前のまとめたシート（PNG）も要ります"
                          "（ほかの動きのコマはそこから使います。CLI は --original）")
    fw, fh = int(info["frame_width"]), int(info["frame_height"])
    sheet = E._decode_image(original)
    rows, cols = int(info.get("rows") or 0), int(info.get("columns") or 0)
    if sheet.shape[0] != rows * fh or sheet.shape[1] != cols * fw:
        raise E.EditError("bad_size", f"直す前のまとめたシートの大きさ {sheet.shape[1]}×{sheet.shape[0]} が、JSON の "
                          f"{cols * fw}×{rows * fh}（{cols} 列 × {rows} 行、1 コマ {fw}×{fh}）と合いません")
    sub = anim_info(info, anim)
    before, _ = anim_frames(sheet, info, anim)
    sub_opt = dataclasses.replace(opt, mirror=False, anim=None)
    r = E.readback(edited, edited_name, json.dumps(sub, ensure_ascii=False), _png(np.concatenate(before, 1)), sub_opt)
    if not r.made:
        return r
    new = sheet.copy()
    for k, f in zip(info["animations"][anim]["frames"], r.frames):
        x, y = int(info["frames"][k]["x"]), int(info["frames"][k]["y"])
        new[y:y + fh, x:x + fw] = f
    cells = [new[int(f["y"]):int(f["y"]) + fh, int(f["x"]):int(f["x"]) + fw] for f in info["frames"]]
    rec = dict(r.info.get("edited") or {})
    rec = {"animation": anim, **rec}
    stem = E.output_stem(info)
    files: dict = {}
    names_out = dict(r.names)
    out_info = _rebuilt(info, stem + ".png", cells, rec)
    files[stem + ".png"] = _png(new)
    files[stem + ".json"] = E.info_json(out_info).encode("utf-8")
    names_out.update({"sheet": stem + ".png", "json": stem + ".json"})
    if opt.mirror:
        mstem = E.output_stem(info, mirror=True)
        mnew = np.zeros_like(new)
        for f in info["frames"]:
            x, y = int(f["x"]), int(f["y"])
            mnew[y:y + fh, x:x + fw] = new[y:y + fh, x:x + fw][:, ::-1]
        minfo = _rebuilt(info, mstem + ".png", [c[:, ::-1] for c in cells], dict(rec), mirror=True)
        files[mstem + ".png"] = _png(mnew)
        files[mstem + ".json"] = E.info_json(minfo).encode("utf-8")
        names_out.update({"mirror": mstem + ".png", "mirror_json": mstem + ".json"})
    for k, v in r.files.items():
        if k in (r.names.get("sheet"), r.names.get("json")):
            continue   # 1 つの動きだけのシートは書かない（まとめたシートを書き直す）
        files[k] = v
    report = dict(r.report, animation=anim, image=stem + ".png", files=list(files))
    if "check_json" in r.names:
        files[r.names["check_json"]] = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return E.ReadbackResult(True, files=files, report=report, warnings=r.warnings, notes=r.notes, frames=r.frames,
                            info=out_info, names=names_out)


def _rebuilt(info: dict, image_name: str, cells: list, edited: dict, mirror: bool = False) -> dict:
    from . import edit as E
    out = copy.deepcopy(info)
    out["image"] = image_name
    if mirror:
        fw = int(info["frame_width"])
        out["motion"] = out["motion"][:-len("_mirror")] if out["motion"].endswith("_mirror") else out["motion"] + "_mirror"
        pv = out.get("pivot")
        if isinstance(pv, dict) and isinstance(pv.get("x"), (int, float)):
            pv["x"] = fw - pv["x"]
        src = out.get("source")
        if isinstance(src, dict) and "mirrored" in src:
            src["mirrored"] = not bool(src["mirrored"])
    out = E._with_palette(out, E.palette_hex(cells))
    out["edited"] = edited
    return out


def is_anims_info(info) -> bool:
    return isinstance(info, dict) and isinstance(info.get("animations"), dict)
