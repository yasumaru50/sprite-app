"""大きさをくらべる: 変換した素材を、同じ倍率・同じ地面の線で横に並べた確かめ用の画像を作る。

    from retrocore.sprite.compare import compare_entries
    res = compare_entries([("hero.png", png_bytes, None), ("walk.png", sheet_png, sheet_json_text)])
    res["png"]   # 確かめ用の画像（PNG のバイト列）

AI は絵ごとに格子の細かさを決めるので、変換した素材どうしの大きさ・線の太さがそろっているかを 1 枚で見られるようにする。

- 絵ごとに透明な余白を切り詰め、足元（描かれている所の下端）を同じ地面の線にそろえて、渡した順に左から並べる
- 絵そのものは拡大・縮小しない。全体を同じ整数倍（`scale`）で拡大する
- マス目（`grid` ドット）は、横の線を地面の線から上下に数えて、縦の線を左端から引く
- 地面の線は赤。1 枚目の絵のいちばん上の高さに、青の点線を横いっぱいに引く（1 枚目に基準のキャラを置く）
- 各絵の下に「名前 幅x高さ」（切り詰めた後のドット数）を、このファイルの中の小さな文字（ドットの表）で書く。
  書けない文字（日本語など）を含む名前は「#1」のような番号にする（`numbered` に番号と名前の対応）
- シート（このアプリの JSON と組）は 1 コマ目だけを使う。JSON が無ければ PNG 全体を 1 枚の絵にする

CLI（`python -m apps.sprite compare`）と画面（`compare_for_web`）は同じこの関数で作るので、同じ PNG（バイト単位）になる。
"""

from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
from PIL import Image

from .image import to_rgba

DEFAULT_SCALE = 4
MIN_SCALE, MAX_SCALE = 1, 16
DEFAULT_GRID = 16
MIN_GRID, MAX_GRID = 1, 256
DEFAULT_GAP = 8
MIN_GAP, MAX_GAP = 0, 256
MAX_ITEMS = 64
MAX_PIXELS = 64_000_000   # できる画像の画素数の上限（ブラウザで扱える大きさ）
NAME_MAX = 32             # これより長い名前は番号にする
OUTPUT_NAME = "compare.png"

BACKGROUND = (200, 200, 206)   # 明るめの灰色（暗い輪郭が見えるように）
GRID_COLOR = (172, 172, 180)   # 背景より少し暗い
GROUND_COLOR = (220, 48, 48)   # 地面の線
TOP_COLOR = (36, 96, 228)      # 1 枚目の高さの点線
LABEL_BACKGROUND = (232, 232, 236)
LABEL_COLOR = (32, 32, 40)

# ---- 小さな文字（5×9 のドットの表。この Sprite App のために作ったもの） ----
# 大文字・数字は上の 7 行、小文字の g・j・p・q・y は下の 2 行まで使う。# は番号の印
GLYPH_W, GLYPH_H, ADVANCE = 5, 9, 6
_GLYPHS = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "a": [".....", ".....", ".###.", "....#", ".####", "#...#", ".####"],
    "b": ["#....", "#....", "####.", "#...#", "#...#", "#...#", "####."],
    "c": [".....", ".....", ".####", "#....", "#....", "#....", ".####"],
    "d": ["....#", "....#", ".####", "#...#", "#...#", "#...#", ".####"],
    "e": [".....", ".....", ".###.", "#...#", "#####", "#....", ".###."],
    "f": ["..##.", ".#...", ".#...", "###..", ".#...", ".#...", ".#..."],
    "g": [".....", ".....", ".####", "#...#", "#...#", "#...#", ".####", "....#", ".###."],
    "h": ["#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#"],
    "i": ["..#..", ".....", ".##..", "..#..", "..#..", "..#..", ".###."],
    "j": ["...#.", ".....", "..##.", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "k": ["#....", "#....", "#..#.", "#.#..", "##...", "#.#..", "#..#."],
    "l": [".##..", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "m": [".....", ".....", "##.#.", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
    "n": [".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#"],
    "o": [".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###."],
    "p": [".....", ".....", "####.", "#...#", "#...#", "#...#", "####.", "#....", "#...."],
    "q": [".....", ".....", ".####", "#...#", "#...#", "#...#", ".####", "....#", "....#"],
    "r": [".....", ".....", "#.##.", "##..#", "#....", "#....", "#...."],
    "s": [".....", ".....", ".####", "#....", ".###.", "....#", "####."],
    "t": [".#...", ".#...", "###..", ".#...", ".#...", ".#..#", "..##."],
    "u": [".....", ".....", "#...#", "#...#", "#...#", "#..##", ".##.#"],
    "v": [".....", ".....", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "w": [".....", ".....", "#...#", "#...#", "#.#.#", "#.#.#", ".#.#."],
    "x": [".....", ".....", "#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "y": [".....", ".....", "#...#", "#...#", "#...#", "#...#", ".####", "....#", ".###."],
    "z": [".....", ".....", "#####", "...#.", "..#..", ".#...", "#####"],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "_": [".....", ".....", ".....", ".....", ".....", ".....", "#####"],
    "-": [".....", ".....", ".....", ".###.", ".....", ".....", "....."],
    ".": [".....", ".....", ".....", ".....", ".....", ".....", "..#.."],
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    "(": ["...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."],
    ")": [".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."],
    "#": [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
    " ": [],
}
# 名前に使える文字（# は番号の印なので名前には使わない）
NAME_CHARS = frozenset(c for c in _GLYPHS if c not in "# ")


def _glyph_masks() -> dict:
    out = {}
    for c, rows in _GLYPHS.items():
        rows = list(rows) + ["." * GLYPH_W] * (GLYPH_H - len(rows))
        assert len(rows) == GLYPH_H and all(len(r) == GLYPH_W for r in rows), c
        out[c] = np.array([[ch == "#" for ch in r] for r in rows], dtype=bool)
    return out


GLYPHS = _glyph_masks()


def text_scale(scale: int) -> int:
    """名前の文字の大きさ（文字の 1 ドットを何画素にするか）。拡大が大きいほど文字も大きくする（1〜3 倍は 1、4〜11 倍は 2、
    12 倍から 3、16 倍で 4）。"""
    scale = int(scale)
    return 1 if scale < 4 else max(2, scale // 4)


def text_width(text: str, t: int) -> int:
    """文字を書くのに使う横の画素数（最後の文字の後ろの 1 列のすき間も含む）。"""
    return len(text) * ADVANCE * t


def draw_text(img: np.ndarray, text: str, x: int, y: int, t: int, color) -> None:
    """img（H×W×3 の uint8）の (x, y) から text を書く（はみ出す所は書かない）。書けない文字は ValueError。"""
    h, w = img.shape[:2]
    col = np.array(color, dtype=np.uint8)
    for i, c in enumerate(text):
        if c not in GLYPHS:
            raise ValueError(f"書けない文字です: {c!r}")
        m = GLYPHS[c]
        if t > 1:
            m = m.repeat(t, axis=0).repeat(t, axis=1)
        gx = x + i * ADVANCE * t
        y0, x0 = max(y, 0), max(gx, 0)
        y1, x1 = min(y + m.shape[0], h), min(gx + m.shape[1], w)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = m[y0 - y:y1 - y, x0 - gx:x1 - gx]
        img[y0:y1, x0:x1][sub] = col


def display_name(file_name: str) -> str:
    """ファイル名から拡張子を除いた名前。"""
    name = str(file_name).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name[1:] else name


def can_write(name: str) -> bool:
    """名前をこの小さな文字でそのまま書けるか（英字・数字・_ - . + ( ) だけで、長すぎない）。"""
    return 0 < len(name) <= NAME_MAX and all(c in NAME_CHARS for c in name)


def crop_to_content(rgba: np.ndarray) -> np.ndarray:
    """透明な余白を切り詰める（描かれている所が無ければそのまま返す）。"""
    alpha = rgba[..., 3] > 0
    if not alpha.any():
        return rgba
    ys = np.flatnonzero(alpha.any(axis=1))
    xs = np.flatnonzero(alpha.any(axis=0))
    return rgba[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]


def first_frame(rgba: np.ndarray, info) -> np.ndarray:
    """シートの JSON（このアプリが書き出したもの）から 1 コマ目を切り出す。"""
    if not isinstance(info, dict):
        raise ValueError("シートの JSON ではありません")
    frames = info.get("frames")
    f = frames[0] if isinstance(frames, list) and frames and isinstance(frames[0], dict) else None
    if f is not None and all(isinstance(f.get(k), int) for k in ("x", "y", "width", "height")):
        x, y, w, h = f["x"], f["y"], f["width"], f["height"]
    elif isinstance(info.get("frame_width"), int) and isinstance(info.get("frame_height"), int):
        x, y, w, h = 0, 0, info["frame_width"], info["frame_height"]
    else:
        raise ValueError("1 コマ目の場所（frames か frame_width・frame_height）が書かれていません")
    if w < 1 or h < 1 or x < 0 or y < 0 or x + w > rgba.shape[1] or y + h > rgba.shape[0]:
        raise ValueError(f"1 コマ目（x {x}, y {y}, {w}x{h}）が PNG（{rgba.shape[1]}x{rgba.shape[0]}）からはみ出します")
    return rgba[y:y + h, x:x + w]


def _check_int(value, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name}は {lo}〜{hi} の整数で指定してください（{value!r}）")
    return value


def _blend(dst: np.ndarray, src: np.ndarray) -> None:
    """dst（H×W×3）の上に src（H×W×4）を重ねる（整数の計算なので、どこで動かしても同じ結果）。"""
    a = src[..., 3:4].astype(np.uint32)
    s = src[..., :3].astype(np.uint32)
    d = dst.astype(np.uint32)
    dst[...] = ((s * a + d * (255 - a) + 127) // 255).astype(np.uint8)


def compare_images(items, scale: int = DEFAULT_SCALE, grid: int = DEFAULT_GRID, gap: int = DEFAULT_GAP):
    """[(名前, RGBA 配列), ...] を並べた確かめ用の画像（H×W×4 の uint8）と、並べ方の情報を返す。

    名前はそのまま書ける名前（`can_write`）か、そうでなければ番号（#1 など）にする。配列は切り詰める前のもの。
    """
    scale = _check_int(scale, "倍率", MIN_SCALE, MAX_SCALE)
    grid = _check_int(grid, "マス目の大きさ", MIN_GRID, MAX_GRID)
    gap = _check_int(gap, "絵と絵の間", MIN_GAP, MAX_GAP)
    if not items:
        raise ValueError("くらべる絵がありません")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"くらべる絵は {MAX_ITEMS} 枚までです（{len(items)} 枚）")
    t = text_scale(scale)
    placed = []
    x = gap
    for i, (name, rgba) in enumerate(items, start=1):
        img = crop_to_content(to_rgba(rgba))
        h, w = img.shape[:2]
        label_name = name if can_write(name) else f"#{i}"
        label = f"{label_name} {w}x{h}"
        label_dots = -(-text_width(label, t) // scale)
        slot = max(w, label_dots)   # 名前の文字が隣と重ならないように、文字の分だけ間を広げる
        placed.append({"index": i, "name": name, "label": label_name, "text": label, "width": w, "height": h,
                       "x": x, "image": img, "slot": slot})
        x += slot + gap
    width = -(-x // grid) * grid
    tallest = max(p["height"] for p in placed)
    ground = -(-(tallest + gap) // grid) * grid   # 地面の線より上（マス目で数えられる高さ）
    below = max(gap, 1)
    height = ground + below

    canvas = np.empty((height, width, 3), np.uint8)
    canvas[...] = BACKGROUND
    for p in placed:
        y = ground - p["height"]
        _blend(canvas[y:ground, p["x"]:p["x"] + p["width"]], p["image"])
    W, H = width * scale, height * scale
    if W * (H + (GLYPH_H + 5) * t) > MAX_PIXELS:
        raise ValueError(f"できる画像が大きすぎます（{W}x{H} 画素）。倍率かマス目を小さくしてください")
    big = canvas.repeat(scale, axis=0).repeat(scale, axis=1)
    # マス目: 背景の所だけに引く（絵の上には引かない）
    paper = np.zeros((height, width), bool)
    paper[...] = True
    for p in placed:
        y = ground - p["height"]
        paper[y:ground, p["x"]:p["x"] + p["width"]] &= p["image"][..., 3] == 0
    paper_big = paper.repeat(scale, axis=0).repeat(scale, axis=1)
    lines = np.zeros((H, W), bool)
    lines[:, [min(gx * scale, W - 1) for gx in range(0, width + 1, grid)]] = True
    rows = [ground - k for k in range(0, ground + 1, grid)] + [ground + k for k in range(grid, below + 1, grid)]
    lines[[min(r * scale, H - 1) for r in rows], :] = True
    big[lines & paper_big] = GRID_COLOR
    thick = max(1, scale // 2)
    big[ground * scale:ground * scale + thick, :] = GROUND_COLOR
    top = (ground - placed[0]["height"]) * scale   # 1 枚目のいちばん上の行の上の端
    dash = max(2, scale)
    on = (np.arange(W) // dash) % 2 == 0
    big[max(top - thick, 0):max(top, thick), on] = TOP_COLOR

    pad = 3 * t
    strip_h = pad + GLYPH_H * t + 2 * t
    strip = np.empty((strip_h, W, 3), np.uint8)
    strip[...] = LABEL_BACKGROUND
    for p in placed:
        draw_text(strip, p["text"], p["x"] * scale, pad, t, LABEL_COLOR)
    rgb = np.concatenate([big, strip], axis=0)
    out = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    info = {"scale": scale, "grid": grid, "gap": gap, "width": int(W), "height": int(out.shape[0]),
            "ground_y": int(ground * scale), "first_top_y": int(top),
            "items": [{k: p[k] for k in ("index", "name", "label", "width", "height", "x")} for p in placed]}
    return out, info


def encode(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(data))) as img:
        img.load()
        return to_rgba(img)


def compare_entries(entries, scale: int = DEFAULT_SCALE, grid: int = DEFAULT_GRID, gap: int = DEFAULT_GAP) -> dict:
    """[(ファイル名, PNG のバイト列, シートの JSON の文字列 か None), ...] から確かめ用の画像を作る（CLI と画面で共通）。

    結果: {"png": バイト列, "items": [...], "numbered": [{"label": "#2", "name": "きのこ"}], "warnings": [...], ...}。
    """
    items, sheets = [], []
    for file_name, data, sheet_text in entries:
        name = display_name(file_name)
        try:
            rgba = _decode(data)
        except Exception as e:
            raise ValueError(f"画像として読み込めません: {file_name}（{type(e).__name__}）") from None
        if sheet_text is not None:
            try:
                info = json.loads(sheet_text)
            except ValueError as e:
                raise ValueError(f"{name} の JSON を読めません（{e}）") from None
            try:
                rgba = first_frame(rgba, info)
            except ValueError as e:
                raise ValueError(f"{name}: {e}") from None
        items.append((name, rgba))
        sheets.append(sheet_text is not None)
    img, info = compare_images(items, scale, grid, gap)
    warnings = []
    for it, sheet in zip(info["items"], sheets):
        it["sheet"] = sheet
    for (name, rgba), it in zip(items, info["items"]):
        if not (rgba[..., 3] > 0).any():
            warnings.append(f"{name}: 描かれている所がありません（透明だけの絵です）")
    info["numbered"] = [{"label": it["label"], "name": it["name"]} for it in info["items"] if it["label"] != it["name"]]
    info["warnings"] = warnings
    info["png"] = encode(img)
    return info


def pair_files(names) -> list:
    """ファイル名の並びを [(PNG の番号, JSON の番号 か None), ...] に組む（JSON は拡張子を除いた名前が同じ PNG と組にする）。

    並びは PNG の順。組になる PNG の無い JSON は ValueError。
    """
    def key(n):
        n = str(n).replace("\\", "/")
        return n.rsplit(".", 1)[0] if "." in n.rsplit("/", 1)[-1][1:] else n

    images = [i for i, n in enumerate(names) if not str(n).lower().endswith(".json")]
    jsons = {}
    for i, n in enumerate(names):
        if str(n).lower().endswith(".json"):
            jsons.setdefault(key(n), i)
    used = set()
    out = []
    for i in images:
        j = jsons.get(key(names[i]))
        if j is not None:
            used.add(j)
        out.append((i, j))
    lost = [names[j] for j in jsons.values() if j not in used]
    if lost:
        raise ValueError("同じ名前の PNG が無い JSON があります: " + "、".join(str(n) for n in lost))
    return out


def summary_lines(res: dict) -> list:
    """CLI と画面に出す説明（並べた絵の大きさと、番号にした名前の対応）。"""
    lines = []
    for it in res["items"]:
        lines.append(f"{it['label']}: {it['width']}x{it['height']}" + ("（シートの 1 コマ目）" if it.get("sheet") else ""))
    for n in res["numbered"]:
        lines.append(f"{n['label']} = {n['name']}")
    return lines


def compare_for_web(datas: list, names: list, options_json: str = "{}") -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。datas は選んだ順のファイルのバイト列（PNG と JSON）、names は同じ順のファイル名。

    結果は JSON 文字列（{"ok": true, "png_base64", "name", "items", "numbered", "lines", "warnings"} か
    {"ok": false, "error"}）。
    """
    try:
        if len(datas) != len(names):
            raise ValueError("ファイルと名前の数が合いません")
        o = json.loads(options_json or "{}")
        entries = []
        for i, j in pair_files(names):
            text = None if j is None else bytes(datas[j]).decode("utf-8-sig")
            entries.append((names[i], bytes(datas[i]), text))
        res = compare_entries(entries, o.get("scale", DEFAULT_SCALE), o.get("grid", DEFAULT_GRID),
                              o.get("gap", DEFAULT_GAP))
    except (ValueError, UnicodeDecodeError) as e:
        return json.dumps({"ok": False, "error": f"くらべる画像を作れませんでした: {e}"}, ensure_ascii=False)
    png = res.pop("png")
    res.update({"ok": True, "png_base64": base64.b64encode(png).decode("ascii"), "name": OUTPUT_NAME,
                "lines": summary_lines(res)})
    return json.dumps(res, ensure_ascii=False)


def spec() -> dict:
    """画面の欄の既定値と範囲（CLI と同じ）。"""
    return {"scale": {"default": DEFAULT_SCALE, "min": MIN_SCALE, "max": MAX_SCALE},
            "grid": {"default": DEFAULT_GRID, "min": MIN_GRID, "max": MAX_GRID},
            "gap": {"default": DEFAULT_GAP, "min": MIN_GAP, "max": MAX_GAP}, "max_items": MAX_ITEMS}
