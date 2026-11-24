"""手直ししたシートの読み戻し: エディタ（Piskel など）で直したシート（PNG か .piskel）と、元のシートの情報（JSON）から、
JSON・GIF・左右反転のシート・横 1 列に並べ直したシート・.piskel を作り直す。

    from retrocore.sprite.edit import ReadbackOptions, readback
    res = readback(edited_bytes, "直した.png", json_text, options=ReadbackOptions(gif=True, mirror=True, piskel=True))
    res.made        # 作れたか（大きさ・コマの数が合わないときなどは作らない。理由は res.errors）
    res.files       # {ファイル名: 中身のバイト}（CLI はこれを書き、ブラウザ版はこれを保存する。同じ関数なのでバイトで同じ）
    res.report      # 確かめた結果（<名前>_check.json と同じもの）

決まりごと（docs/sprite-app-plan.md の「手直しの読み戻し」）:
- **直したドットはそのまま使う**。新しい色・半透明があっても、作って警告し、場所の画像（`_check.png`）を出す。
  近い色に直すのは、利用者が選んだとき（`fix_colors`）だけ
- 時間・順番・足元（pivot）・padding・offset_y・source_frame などは**元の JSON から引き継ぐ**（Piskel はコマごとの時間を持たない）
- シートの並びは、1 コマの大きさ（JSON の frame_width・frame_height）から読む。横 1 列でなくても（Piskel の書き出しの Columns を
  変えて格子状になっていても）、左から右・上から下の順に読む。整数倍に拡大して書き出されていても、ドットの形が崩れていなければ戻す。
  余ったマスは透明だけのときに限って捨てる
- 作らないとき: JSON がシートの情報でない、絵の大きさが 1 コマの大きさで割り切れない、コマの数が合わない、透明だけのコマがある
  （Piskel は透明だけのコマを読み込まないので、コマがずれた可能性が高い）。理由を分かる言葉で返す
- 同じコマを何度も使う順番（1,2,3,2 など）のシートは、**再生する順に 1 コマずつ**並んでいる（Piskel で再生した見た目がそのまま
  ゲームの見た目になる）。読み戻しでは各コマをそのまま使い、同じ元のコマ（`source_frame`）から作ったコマが直したあとで違えば、
  違うドットの数と場所を警告する。そろえるのは、利用者が選んだとき（`same_frames="first"`。先に出てくるコマに合わせる）だけ
- 元の色: JSON の `palette`（これから作るシートの JSON に入る）か、直す前のシート（PNG）。どちらも無ければ新しい色は調べない

`retrocore` の方針どおり NumPy と Pillow と標準ライブラリだけを使う（Pyodide で動く。np.bincount などは使っていない）。
"""

from __future__ import annotations

import copy
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from .motion import gif as gif_mod
from .piskel import PiskelError, fps_for_durations, piskel_text, read_piskel

EDIT_SUFFIX = "_edited"
CHECK_VERSION = 1
MAX_SCALE = 16            # 拡大して書き出されたシートを戻すときに試す倍率の上限
MAX_PIXELS = 64           # 記録に書く場所の数の上限（色・コマごと）
CHECK_SCALE = 8           # 確認の画像の拡大
SAME_FRAMES = {"keep": "そのまま（直したとおりに使う。違えば知らせる）", "first": "先に出てくるコマに合わせる"}
DEFAULT_SAME_FRAMES = "keep"
MARK_NEW = (255, 0, 200, 255)      # 元のパレットに無い色
MARK_SEMI = (255, 140, 0, 255)     # 半透明
MARK_REPEAT = (0, 150, 255, 255)   # 同じ元のコマなのに違うドット
MARK_CHANGED = (255, 0, 200, 255)  # 直す前と違うドット（_changes.png）


class EditError(ValueError):
    """読み戻せない（作らない）。`code` は理由の種類（記録の JSON の code）。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class ReadbackOptions:
    gif: bool = True
    gif_scale: int = gif_mod.DEFAULT_SCALE
    gif_background: str | None = None
    mirror: bool = True
    piskel: bool = True
    same_frames: str = DEFAULT_SAME_FRAMES
    fix_colors: bool = False
    anim: str | None = None   # まとめたシート（JSON に animations）で直した動きの名前（None はファイルの名前から決める）

    @classmethod
    def from_dict(cls, values: dict) -> "ReadbackOptions":
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"知らない読み戻しの設定です: {', '.join(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        gif_mod.check_scale(self.gif_scale)
        gif_mod.check_background(self.gif_background)
        if self.same_frames not in SAME_FRAMES:
            raise ValueError(f"同じコマの扱いは {' / '.join(SAME_FRAMES)} のどれかにしてください: {self.same_frames}")


@dataclass
class ReadbackResult:
    made: bool
    files: dict = field(default_factory=dict)        # {ファイル名: バイト}（書く順）
    report: dict = field(default_factory=dict)       # 確かめた結果（_check.json の中身）
    warnings: list = field(default_factory=list)     # 知らせること（文）
    notes: list = field(default_factory=list)        # 読み方などのメモ（文）
    errors: list = field(default_factory=list)       # 作らなかった理由（文）
    frames: list = field(default_factory=list)       # 並べ直したコマ（RGBA）
    info: dict | None = None                         # 作り直した JSON の辞書
    names: dict = field(default_factory=dict)        # {"sheet", "json", "gif", "piskel", "check", "check_json", "changes", "mirror", ...}


# ---------------------------------------------------------------- 入力を読む

def parse_sheet_info(text) -> dict:
    """シートの情報（JSON の文字かバイト）を読んで確かめる。"""
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise EditError("bad_json", "シートの情報（JSON）の文字を読めません") from None
    try:
        info = json.loads(text)
    except ValueError:
        raise EditError("bad_json", "シートの情報（JSON）を読めません。Sprite App で保存した、シートと同じ名前の .json を選んでください") from None
    if not isinstance(info, dict):
        raise EditError("bad_json", "シートの情報（JSON）の形が違います")
    if info.get("made") is False:
        raise EditError("bad_json", "この JSON は「作らなかった理由」の記録（_skipped.json）です。シートと同じ名前の .json を選んでください")
    need = ("frame_width", "frame_height", "frames")
    if any(k not in info for k in need):
        if "tracks" in info or "regions" in info:
            raise EditError("bad_json", "この JSON は動きの表（.motion.json）です。シートと同じ名前の .json を選んでください")
        raise EditError("bad_json", "シートの情報（JSON）に frame_width・frame_height・frames がありません。"
                                    "Sprite App で保存した、シートと同じ名前の .json を選んでください")
    fw, fh, frames = info["frame_width"], info["frame_height"], info["frames"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (fw, fh)):
        raise EditError("bad_json", f"シートの情報の 1 コマの大きさが読めません: {fw!r}×{fh!r}")
    if not isinstance(frames, list) or not frames:
        raise EditError("bad_json", "シートの情報にコマ（frames）がありません")
    for f in frames:
        d = f.get("duration_ms") if isinstance(f, dict) else None
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise EditError("bad_json", f"シートの情報のコマの時間（duration_ms）が読めません: {f!r}")
    return info


def _is_piskel(data: bytes, name: str) -> bool:
    return str(name).lower().endswith(".piskel") or bytes(data[:16]).lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{")


def _decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(bytes(data))) as im:
            if getattr(im, "n_frames", 1) > 1:
                raise EditError("bad_image", "アニメーションの画像（GIF など）は読めません。Piskel の Export で「PNG」を選んで書き出したシートか、"
                                             "「Save」で保存した .piskel を選んでください")
            im.load()
            if im.mode == "P" and "transparency" in im.info:
                im = im.convert("RGBA")
            return np.asarray(im.convert("RGBA"), np.uint8).copy()
    except EditError:
        raise
    except (OSError, ValueError) as e:
        raise EditError("bad_image", f"直したシートの画像を読めません（PNG か .piskel を選んでください）: {e}") from None


def _cells(img: np.ndarray, fw: int, fh: int) -> list[np.ndarray]:
    rows, cols = img.shape[0] // fh, img.shape[1] // fw
    return [img[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] for r in range(rows) for c in range(cols)]


def _unscale(img: np.ndarray, s: int) -> np.ndarray | None:
    """s 倍に拡大した絵なら 1 倍に戻した絵（s×s のマスが全部同じ色のときだけ）。違えば None。"""
    h, w = img.shape[0] // s, img.shape[1] // s
    small = img[::s, ::s]
    back = np.repeat(np.repeat(small, s, axis=0), s, axis=1)
    return small.copy() if np.array_equal(back, img[:h * s, :w * s]) else None


def _layout(img: np.ndarray, fw: int, fh: int, n: int, scale: int) -> tuple[list[np.ndarray], dict, list[str]]:
    """倍率 scale のつもりで n コマに分ける（合わなければ EditError）。"""
    cols, rows = img.shape[1] // fw, img.shape[0] // fh
    cells = _cells(img, fw, fh)
    if len(cells) < n:
        raise EditError("frame_count",
                        f"直したシートのコマが {len(cells)} 個しかありません（JSON では {n} コマ）。"
                        "Piskel でコマを消していないか、透明だけのコマが読み込みのときに捨てられていないか確かめてください")
    extra = cells[n:]
    used_extra = [n + i + 1 for i, c in enumerate(extra) if c[..., 3].any()]
    if used_extra:
        listed = "・".join(map(str, used_extra[:8])) + (" ほか" if len(used_extra) > 8 else "")
        raise EditError("frame_count",
                        f"直したシートに JSON より多いコマがあります（{listed} 番目のマスにも絵があります。"
                        f"JSON では {n} コマ）。Piskel でコマを足していないか、別のキャラの JSON を選んでいないか確かめてください")
    notes = []
    if rows == 1 and cols == n:
        notes.append(f"横 1 列に {n} コマ並んだシートとして読みました")
    else:
        notes.append(f"{cols} 列 × {rows} 行に並んだシートとして、左から右・上から下の順に読みました"
                     f"（Piskel の書き出しの Columns が {cols} だったようです）")
    if scale > 1:
        notes.append(f"{scale} 倍に拡大して書き出されていたので、1 倍に戻しました（ドットは変えていません）")
    if extra:
        notes.append(f"最後の {len(extra)} マスは透明だけなので捨てました（格子状に書き出すと余るマス）")
    return [c.copy() for c in cells[:n]], {"columns": cols, "rows": rows, "scale": scale}, notes


def split_sheet(img: np.ndarray, fw: int, fh: int, n: int) -> tuple[list[np.ndarray], dict, list[str]]:
    """直したシートの絵を n コマに分ける。返す値は (コマ, 並び {"columns", "rows", "scale"}, メモ)。

    1 倍から順に、1 コマの大きさで割り切れて、コマの数が合う（余ったマスは透明だけ）倍率を探す。2 倍以上は、
    倍率 × 倍率のマスが全部同じ色（ドットの形のまま拡大した絵）のときだけ。
    """
    H, W = img.shape[:2]
    first_error = None
    for s in range(1, MAX_SCALE + 1):
        if W % (fw * s) or H % (fh * s):
            continue
        small = img if s == 1 else _unscale(img, s)
        if small is None:
            continue
        try:
            return _layout(small, fw, fh, n, s)
        except EditError as e:
            first_error = first_error or e
    if first_error is not None:
        raise first_error
    raise EditError("size_mismatch",
                    f"直したシートの大きさ（{W}×{H}）が、1 コマの大きさ {fw}×{fh}（JSON の frame_width・frame_height）で"
                    "割り切れません。Piskel でキャンバスの大きさ（Resize）を変えていないか、Export の Scale が 1 倍か確かめてください")


def read_edited(data: bytes, name: str, info: dict) -> tuple[list[np.ndarray], dict, list[str], list[str]]:
    """直したファイル（PNG か .piskel）を読んでコマに分ける。返す値は (コマ, 並び, メモ, 警告)。"""
    fw, fh, n = int(info["frame_width"]), int(info["frame_height"]), len(info["frames"])
    if _is_piskel(data, name):
        try:
            pk = read_piskel(data)
        except PiskelError as e:
            raise EditError("bad_piskel", str(e)) from None
        if (pk.width, pk.height) != (fw, fh):
            raise EditError("size_mismatch",
                            f".piskel の 1 コマの大きさ（{pk.width}×{pk.height}）が JSON（{fw}×{fh}）と違います。"
                            "Piskel でキャンバスの大きさ（Resize）を変えていないか、別のキャラの JSON を選んでいないか確かめてください")
        if len(pk.frames) != n:
            raise EditError("frame_count",
                            f".piskel のコマの数（{len(pk.frames)}）が JSON（{n} コマ）と違います。"
                            "Piskel でコマを足したり消したりしていないか確かめてください")
        notes = [f".piskel から {n} コマを読みました（Piskel の速さ {pk.fps:g} FPS は使わず、JSON の時間を使います）"]
        return pk.frames, {"kind": "piskel", "columns": n, "rows": 1, "scale": 1, "layers": len(pk.layers)}, notes, list(pk.warnings)
    img = _decode_image(data)
    frames, layout, notes = split_sheet(img, fw, fh, n)
    return frames, {"kind": "png", **layout}, notes, []


# ---------------------------------------------------------------- 色

def _keys(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    return (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2].astype(np.uint32)


def hex_color(key: int) -> str:
    return "#%06x" % int(key)


def palette_hex(frames) -> list[str]:
    """コマで使っている色（見える画素。α > 0）の一覧（"#rrggbb"、小さい順）。シートの JSON の `palette`。"""
    used = [_keys(f[..., :3])[f[..., 3] > 0] for f in frames]
    allk = np.unique(np.concatenate(used)) if used else np.zeros(0, np.uint32)
    return [hex_color(k) for k in allk.tolist()]


def _parse_palette(values) -> np.ndarray | None:
    if not isinstance(values, list):
        return None
    out = []
    for v in values:
        if isinstance(v, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", v):
            out.append(int(v.lstrip("#"), 16))
    return np.array(sorted(set(out)), np.uint32) if out else None


def _norm(f: np.ndarray) -> np.ndarray:
    """透明の画素の色を 0 にした写し（比べるとき用。透明の画素の色は見えないので違いに数えない）。"""
    g = np.asarray(f, np.uint8).copy()
    g[g[..., 3] == 0] = 0
    return g


def _pixels(mask: np.ndarray, frame: int | None = None, limit: int = MAX_PIXELS) -> list:
    ys, xs = np.nonzero(mask)
    pts = list(zip(xs.tolist(), ys.tolist()))[:limit]
    return [[x, y] if frame is None else [frame, x, y] for x, y in pts]


# ---------------------------------------------------------------- 名前

def output_stem(info: dict, mirror: bool = False) -> str:
    """作り直すシートの名前（拡張子なし）。元の image の名前に `_edited` を付ける（反転は `_mirror` を足すか除く）。"""
    image = str(info.get("image") or "sprite.png")
    stem = PurePosixPath(image.replace("\\", "/")).name
    stem = stem[:-4] if stem.lower().endswith(".png") else stem
    if stem.endswith(EDIT_SUFFIX):
        stem = stem[:-len(EDIT_SUFFIX)]
    if mirror:
        motion = info.get("motion")
        tail = (f"_{motion}_{info.get('frame_width')}x{info.get('frame_height')}_{len(info.get('frames') or [])}f"
                if isinstance(motion, str) else None)
        if tail and "source" in info and stem.endswith(tail) and len(stem) > len(tail):
            # コマのシート（<出力名>_<名前>_<幅>x<高さ>_<N>f）: 名前に _mirror を足すか除く（frames.sheet_file_name と同じ形）
            flipped = motion[:-len("_mirror")] if motion.endswith("_mirror") else motion + "_mirror"
            stem = stem[:-len(tail)] + tail.replace(f"_{motion}_", f"_{flipped}_", 1)
        else:
            stem = stem[:-len("_mirror")] if stem.endswith("_mirror") else stem + "_mirror"
    return stem + EDIT_SUFFIX


# ---------------------------------------------------------------- JSON

def _with_palette(info: dict, palette: list[str]) -> dict:
    out = {}
    placed = False
    for k, v in info.items():
        if k == "palette":
            continue
        if k == "skipped" and not placed:
            out["palette"] = palette
            placed = True
        if k == "edited":
            continue
        out[k] = v
    if not placed:
        out["palette"] = palette
    return out


def rebuilt_info(info: dict, image_name: str, frames: list[np.ndarray], edited: dict, mirror: bool = False) -> dict:
    """元の JSON から、並べ直したシートの JSON の辞書を作る（時間・順番・足元などは引き継ぐ）。"""
    fw, fh = int(info["frame_width"]), int(info["frame_height"])
    out = copy.deepcopy(info)
    out["image"] = image_name
    n = len(frames)
    out["frame_count"] = n
    out["columns"] = n
    out["rows"] = 1
    for i, f in enumerate(out["frames"]):
        f["x"], f["y"] = i * fw, 0
        f["width"], f["height"] = fw, fh
        if "index" in f:
            f["index"] = i
    if mirror:
        if isinstance(out.get("motion"), str):
            m = out["motion"]
            out["motion"] = m[:-len("_mirror")] if m.endswith("_mirror") else m + "_mirror"
        pv = out.get("pivot")
        if isinstance(pv, dict) and isinstance(pv.get("x"), (int, float)) and not isinstance(pv.get("x"), bool):
            pv["x"] = type(pv["x"])(fw - pv["x"]) if isinstance(pv["x"], float) else fw - pv["x"]
        pad = out.get("padding")
        if isinstance(pad, dict) and "left" in pad and "right" in pad:
            pad["left"], pad["right"] = pad["right"], pad["left"]
        src = out.get("source")
        if isinstance(src, dict) and "mirrored" in src:
            src["mirrored"] = not bool(src["mirrored"])
    out = _with_palette(out, palette_hex(frames))
    out["edited"] = edited
    return out


def info_json(info: dict) -> str:
    return json.dumps(info, ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------- 確かめる

def check_frames(frames: list[np.ndarray], info: dict, palette: np.ndarray | None,
                 original: list[np.ndarray] | None = None) -> dict:
    """直したコマを確かめる（直さない）。返す値は記録の辞書（_check.json の一部）と、画像に描く場所。"""
    n = len(frames)
    rec_frames = []
    new_color: dict[int, dict] = {}
    semi_pixels, semi_total = [], 0
    marks = []   # コマごとの {"new": mask, "semi": mask, "repeat": mask}
    for i, f in enumerate(frames):
        a = f[..., 3]
        empty = not bool(a.any())
        semi = (a > 0) & (a < 255)
        semi_total += int(semi.sum())
        semi_pixels += _pixels(semi, i + 1, MAX_PIXELS - len(semi_pixels)) if len(semi_pixels) < MAX_PIXELS else []
        newm = np.zeros(a.shape, bool)
        if palette is not None:
            k = _keys(f[..., :3])
            vis = a > 0
            newm = vis & ~np.isin(k, palette)
            for key in np.unique(k[newm]).tolist():
                m = newm & (k == key)
                e = new_color.setdefault(int(key), {"color": hex_color(key), "dots": 0, "frames": [], "pixels": []})
                e["dots"] += int(m.sum())
                e["frames"].append(i + 1)
                if len(e["pixels"]) < MAX_PIXELS:
                    e["pixels"] += _pixels(m, i + 1, MAX_PIXELS - len(e["pixels"]))
        entry = {"frame": i + 1, "empty": empty, "semi_transparent": int(semi.sum()),
                 "new_color_dots": int(newm.sum()) if palette is not None else None}
        if original is not None:
            entry["changed_dots"] = int((_norm(f) != _norm(original[i])).any(-1).sum())
        sf = info["frames"][i].get("source_frame") if i < len(info["frames"]) else None
        if isinstance(sf, int) and not isinstance(sf, bool):
            entry["source_frame"] = sf
            if info["frames"][i].get("flipped") is True:   # 裏返したコマ（順番の r）は、裏返していないコマとは比べない
                entry["flipped"] = True
        sw = info["frames"][i].get("swap") if i < len(info["frames"]) else None
        if isinstance(sw, dict):   # 差し替え（まばたきなど）を入れたフレームは、同じ元のコマでも違ってよい
            entry["swap"] = sw
        rec_frames.append(entry)
        marks.append({"new": newm, "semi": semi, "repeat": np.zeros(a.shape, bool)})
    # 同じ元のコマ（source_frame）から作ったコマの違い
    groups: dict[tuple, list[int]] = {}
    for e in rec_frames:
        if "source_frame" in e and "swap" not in e:
            groups.setdefault((e["source_frame"], e.get("flipped", False)), []).append(e["frame"] - 1)
    repeats = []
    for (sf, fl), idx in groups.items():
        if len(idx) < 2:
            continue
        first = _norm(frames[idx[0]])
        for j in idx[1:]:
            d = (_norm(frames[j]) != first).any(-1)
            if d.any():
                marks[j]["repeat"] |= d
                marks[idx[0]]["repeat"] |= d
                repeats.append({"source_frame": sf, "frames": [idx[0] + 1, j + 1], "dots": int(d.sum()),
                                "pixels": _pixels(d)})
                if fl:
                    repeats[-1]["flipped"] = True
    return {"frames": rec_frames, "new_colors": sorted(new_color.values(), key=lambda e: (-e["dots"], e["color"])),
            "semi_transparent": {"dots": semi_total, "pixels": semi_pixels}, "repeats": repeats,
            "_marks": marks}


def _messages(chk: dict, palette_source: str | None, n: int) -> list[str]:
    out = []
    if chk["new_colors"]:
        cols = chk["new_colors"]
        listed = "、".join(f"{c['color']}（{c['dots']} ドット、{'・'.join(map(str, c['frames']))} コマ目）" for c in cols[:8])
        out.append(f"元のパレットに無い色が {len(cols)} 色あります: {listed}{' ほか' if len(cols) > 8 else ''}。"
                   "そのまま使いました。場所は確認の画像の赤紫の枠です（近い元の色に直すときは「おかしな色を直す」を選びます）")
    semi = chk["semi_transparent"]["dots"]
    if semi:
        fr = [e["frame"] for e in chk["frames"] if e["semi_transparent"]]
        out.append(f"半透明のドットが {semi} 個あります（{'・'.join(map(str, fr))} コマ目）。シートにはそのまま入れました"
                   "（GIF では透明か不透明のどちらかになります）。場所は確認の画像の橙の枠です")
    for r in chk["repeats"]:
        a, b = r["frames"]
        out.append(f"{a} コマ目と {b} コマ目は同じコマ（元の {r['source_frame']}{'r' if r.get('flipped') else ''} コマ目）から作ったものですが、直した結果が "
                   f"{r['dots']} ドット違います。そのまま使いました（片方だけ直し忘れていないか確かめてください。"
                   "場所は確認の画像の青い枠。そろえるときは「同じコマは先に出てくるコマに合わせる」を選びます）")
    return out


def _check_image(frames: list[np.ndarray], marks: list[dict], scale: int = CHECK_SCALE) -> np.ndarray:
    """確認の画像（RGBA）。コマを scale 倍で横に並べ、新しい色（赤紫）・半透明（橙）・同じコマの違い（青）を枠で囲む。"""
    from PIL import ImageDraw

    from .frames import _draw_label
    h, w = frames[0].shape[:2]
    gap = scale
    W = len(frames) * w * scale + (len(frames) + 1) * gap
    Hh = h * scale + 2 * gap + 12
    canvas = Image.new("RGBA", (W, Hh), (200, 200, 200, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (f, m) in enumerate(zip(frames, marks)):
        x0, y0 = gap + i * (w * scale + gap), gap + 12
        canvas.paste(Image.fromarray(_on_checker(f), "RGB").resize((w * scale, h * scale), Image.NEAREST), (x0, y0))
        for key, color in (("repeat", MARK_REPEAT), ("semi", MARK_SEMI), ("new", MARK_NEW)):
            ys, xs = np.nonzero(m[key])
            for y, x in zip(ys.tolist(), xs.tolist()):
                draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                               outline=color, width=max(1, scale // 4))
        label = f"#{i + 1}"
        if m["new"].any():
            label += f" new {int(m['new'].sum())}"
        if m["semi"].any():
            label += f" semi {int(m['semi'].sum())}"
        if m["repeat"].any():
            label += f" diff {int(m['repeat'].sum())}"
        _draw_label(draw, x0, 1, label)
    return np.asarray(canvas)


def _on_checker(f: np.ndarray) -> np.ndarray:
    h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.full((h, w, 3), 236, np.float64)
    bg[((yy + xx) % 2) == 1] = 222
    a = f[..., 3:4].astype(np.float64) / 255
    return (f[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def changes_image(before: list[np.ndarray], after: list[np.ndarray], scale: int = CHECK_SCALE) -> np.ndarray:
    """直す前（上の段）と直した後（下の段）を並べ、違うドットを赤紫の枠で囲んだ画像（RGBA）。"""
    from PIL import ImageDraw

    from .frames import _draw_label
    h, w = after[0].shape[:2]
    gap = scale
    W = len(after) * w * scale + (len(after) + 1) * gap
    Hh = 2 * (h * scale + 12) + 3 * gap
    canvas = Image.new("RGBA", (W, Hh), (200, 200, 200, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (b, a) in enumerate(zip(before, after)):
        d = (_norm(a) != _norm(b)).any(-1)
        x0 = gap + i * (w * scale + gap)
        for row, (img, tag) in enumerate(((b, "before"), (a, "after"))):
            y0 = gap + row * (h * scale + 12 + gap) + 12
            canvas.paste(Image.fromarray(_on_checker(img), "RGB").resize((w * scale, h * scale), Image.NEAREST), (x0, y0))
            ys, xs = np.nonzero(d)
            for y, x in zip(ys.tolist(), xs.tolist()):
                draw.rectangle([x0 + x * scale, y0 + y * scale, x0 + (x + 1) * scale - 1, y0 + (y + 1) * scale - 1],
                               outline=MARK_CHANGED, width=max(1, scale // 4))
            _draw_label(draw, x0, y0 - 11, f"#{i + 1} {tag}" + (f" diff {int(d.sum())}" if row else ""))
    return np.asarray(canvas)


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- 直す（選んだときだけ）

def fix_frames(frames: list[np.ndarray], palette: np.ndarray | None) -> tuple[list[np.ndarray], dict]:
    """新しい色を元のパレットのいちばん近い色に、半透明を透明か不透明（α 128 が境）にする。返す値は (コマ, 直した数)。"""
    from .palette import nearest_in_palette
    out, colors, semi = [], 0, 0
    pal_rgb = None
    if palette is not None and len(palette):
        pal_rgb = np.stack([(palette >> 16) & 255, (palette >> 8) & 255, palette & 255], axis=1).astype(np.uint8)
    for f in frames:
        g = f.copy()
        a = g[..., 3]
        mid = (a > 0) & (a < 255)
        if mid.any():
            semi += int(mid.sum())
            g[..., 3] = np.where(mid, np.where(a >= 128, 255, 0), a).astype(np.uint8)
            g[g[..., 3] == 0] = 0
        if pal_rgb is not None:
            vis = g[..., 3] > 0
            k = _keys(g[..., :3])
            bad = vis & ~np.isin(k, palette)
            if bad.any():
                colors += int(bad.sum())
                idx = nearest_in_palette(g[..., :3][bad], pal_rgb)
                g[..., :3][bad] = pal_rgb[idx]
        out.append(g)
    return out, {"colors": colors, "semi_transparent": semi}


# ---------------------------------------------------------------- まとめ

def readback(edited: bytes, edited_name: str, info_text, original: bytes | None = None,
             options: ReadbackOptions | None = None) -> ReadbackResult:
    """直したファイルと元の JSON（と、あれば直す前のシート）から、作り直すファイルを全部作る（例外は投げない）。"""
    opt = options or ReadbackOptions()
    edited_name = PurePosixPath(str(edited_name).replace("\\", "/")).name or "edited.png"
    try:
        opt.validate()
        info = parse_sheet_info(info_text)
        from .anims import is_anims_info, readback_anims
        if is_anims_info(info):
            # 動きをまとめたシート: 1 つの動きずつ読み戻す（anims.py）
            return readback_anims(edited, edited_name, info, original, opt)
        if opt.anim is not None:
            raise EditError("bad_option", "動きの名前（--anim）は、動きをまとめたシートの JSON のときだけ使えます")
        frames, layout, notes, warnings = read_edited(edited, edited_name, info)
    except EditError as e:
        return ReadbackResult(False, errors=[str(e)], report=_failed_report(e, edited_name))
    except ValueError as e:
        err = EditError("bad_option", str(e))
        return ReadbackResult(False, errors=[str(err)], report=_failed_report(err, edited_name))
    n = len(frames)
    fw, fh = int(info["frame_width"]), int(info["frame_height"])
    empty = [i + 1 for i, f in enumerate(frames) if not f[..., 3].any()]
    if empty:
        err = EditError("empty_frame",
                        f"{'・'.join(map(str, empty))} コマ目が透明だけです。Piskel は透明だけのコマを読み込まないので、"
                        "コマがずれているかもしれません（読み込んだ後に下のコマの一覧で数を確かめてください）。"
                        "わざと空けたコマでも、ゲームでは消えて見えるので作りませんでした")
        return ReadbackResult(False, errors=[str(err)], notes=notes, warnings=warnings,
                              report=_failed_report(err, edited_name, layout))
    # 元の色: JSON の palette か、直す前のシート
    palette, palette_source, before = _parse_palette(info.get("palette")), None, None
    if palette is not None:
        palette_source = "json"
    if original is not None:
        try:
            img = _decode_image(original)
            before, _, _ = split_sheet(img, fw, fh, n)
        except EditError as e:
            warnings.append(f"直す前のシートを読めなかったので、比べていません: {e}")
            before = None
        if before is not None and palette is None:
            used = [_keys(f[..., :3])[f[..., 3] > 0] for f in before]
            palette, palette_source = np.unique(np.concatenate(used)), "original"
    if palette is None:
        notes.append("元の色の一覧（JSON の palette）が無いので、新しい色は調べていません（直す前のシートも選ぶと調べられます）")
    fixed = None
    if opt.fix_colors:
        frames, fixed = fix_frames(frames, palette)
    same = None
    if opt.same_frames == "first":
        seen: dict[tuple, int] = {}
        same = []
        for i, f in enumerate(info["frames"]):
            sf = f.get("source_frame")
            if not isinstance(sf, int) or isinstance(sf, bool) or isinstance(f.get("swap"), dict):
                continue
            sf = (sf, f.get("flipped") is True)   # 裏返したコマ（r）は、裏返していない同じコマとは別に扱う
            if sf in seen:
                j = seen[sf]
                if not np.array_equal(_norm(frames[i]), _norm(frames[j])):
                    same.append([j + 1, i + 1])
                frames[i] = frames[j].copy()
            else:
                seen[sf] = i
    chk = check_frames(frames, info, palette, before)
    marks = chk.pop("_marks")
    warnings += _messages(chk, palette_source, n)
    if fixed is not None:
        if fixed["colors"] or fixed["semi_transparent"]:
            notes.append(f"おかしな色を直しました（新しい色 {fixed['colors']} ドットを近い元の色に、"
                         f"半透明 {fixed['semi_transparent']} ドットを透明か不透明に）")
        if palette is None:
            notes.append("元の色が分からないので、新しい色は直していません（半透明だけ直しました）")
    if same:
        notes.append("同じ元のコマから作ったコマを、先に出てくるコマに合わせました（"
                     + "、".join(f"{b} コマ目 → {a} コマ目と同じに" for a, b in same) + "）")
    stem = output_stem(info)
    names = {"sheet": stem + ".png", "json": stem + ".json", "check": stem + "_check.png",
             "check_json": stem + "_check.json"}
    edited_rec = {"from": edited_name, "read_as": layout["kind"], "columns": layout["columns"], "rows": layout["rows"],
                  "scale": layout["scale"], "palette_source": palette_source,
                  "new_colors": len(chk["new_colors"]) if palette is not None else None,
                  "semi_transparent": chk["semi_transparent"]["dots"],
                  "changed_dots": [e["changed_dots"] for e in chk["frames"]] if before is not None else None,
                  "same_frames": opt.same_frames, "fixed": fixed, "check": names["check_json"]}
    new_info = rebuilt_info(info, names["sheet"], frames, edited_rec)
    files: dict[str, bytes] = {}
    sheet = np.concatenate(frames, axis=1)
    files[names["sheet"]] = _png(sheet)
    files[names["json"]] = info_json(new_info).encode("utf-8")
    durations = [int(f["duration_ms"]) for f in info["frames"]]
    gif_warn = []
    variants = [(False, frames, new_info, stem)]
    if opt.mirror:
        mframes = [f[:, ::-1].copy() for f in frames]
        mstem = output_stem(info, mirror=True)
        names.update({"mirror": mstem + ".png", "mirror_json": mstem + ".json"})
        minfo = rebuilt_info(info, names["mirror"], mframes, dict(edited_rec), mirror=True)
        files[names["mirror"]] = _png(np.concatenate(mframes, axis=1))
        files[names["mirror_json"]] = info_json(minfo).encode("utf-8")
        variants.append((True, mframes, minfo, mstem))
    for is_m, fr, inf, st in variants:
        if opt.gif:
            key = "mirror_gif" if is_m else "gif"
            try:
                g = gif_mod.make_gif(fr, durations, opt.gif_scale, opt.gif_background,
                                     loop=info.get("loop") is not False)
                names[key] = st + ".gif"
                files[names[key]] = g.data
                if not is_m:
                    gif_warn = list(g.warnings)
            except ValueError as e:
                warnings.append(f"GIF を作れませんでした: {e}")
        if opt.piskel:
            key = "mirror_piskel" if is_m else "piskel"
            fps, fps_notes = fps_for_durations(durations)
            names[key] = st + ".piskel"
            files[names[key]] = piskel_text(fr, st, fps).encode("ascii")
            if not is_m:
                notes += fps_notes
    warnings += [f"GIF: {m}" for m in gif_warn]
    report = {"version": CHECK_VERSION, "made": True, "edited": edited_name, "image": names["sheet"],
              "frame_width": fw, "frame_height": fh, "frame_count": n, "layout": layout,
              "palette_source": palette_source, "palette": [hex_color(k) for k in palette.tolist()] if palette is not None else None,
              "frames": chk["frames"], "new_colors": chk["new_colors"], "semi_transparent": chk["semi_transparent"],
              "repeats": chk["repeats"], "fixed": fixed, "same_frames": opt.same_frames,
              "warnings": list(warnings), "notes": list(notes), "files": list(files) + [names["check"], names["check_json"]]}
    if before is not None:
        names["changes"] = stem + "_changes.png"
        report["files"].append(names["changes"])
    files[names["check"]] = _png(_check_image(frames, marks))
    files[names["check_json"]] = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if before is not None:
        files[names["changes"]] = _png(changes_image(before, frames))
    return ReadbackResult(True, files=files, report=report, warnings=warnings, notes=notes, frames=frames,
                          info=new_info, names=names)


def _failed_report(e: EditError, edited_name: str, layout: dict | None = None) -> dict:
    return {"version": CHECK_VERSION, "made": False, "edited": edited_name, "code": e.code, "error": str(e),
            "layout": layout}
