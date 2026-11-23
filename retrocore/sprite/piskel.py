""".piskel ファイル（ブラウザのドット絵エディタ Piskel の保存の形）の書き出しと読み込み。

    from retrocore.sprite.piskel import piskel_text, read_piskel
    text = piskel_text(frames, "hero_sprite_walk_53x91_4f", fps=7)   # Piskel の「Open」で開ける .piskel の中身（文字列）
    pk = read_piskel(open("直した.piskel", "rb").read())             # pk.frames: 見えるレイヤーを重ねたコマ（RGBA）

形は Piskel のソース（GitHub の piskelapp/piskel の `src/js/utils/serialization/Serializer.js`・`Deserializer.js`、
`src/js/utils/FrameUtils.js` の `createFramesFromChunk`、`src/js/utils/PiskelFileUtils.js`）と、Piskel で保存した本物の
.piskel（ai08_edited.piskel）で確かめた:

- ファイルはそのまま JSON の文字（Base64 にはしていない。Piskel はファイルを data URL で読み、Base64 を戻してから JSON.parse する。
  そのとき文字は 1 バイトずつ読むので、日本語などは \\uXXXX で書く。ここでは `ensure_ascii=True`）
- `{"modelVersion": 2, "piskel": {"name", "description", "fps", "height", "width", "layers": [...], "hiddenFrames": [...]}}`
- `layers` の各要素は **JSON の文字列**で、中身は `{"name", "opacity", "frameCount", "chunks": [{"layout", "base64PNG"}]}`
- chunk の PNG はコマを並べた絵。`layout[i]` は i 列目に上から入っているコマの番号（`[[0], [1], [2], [3]]` なら横 1 列に 4 コマ）。
  1 コマの大きさは PNG の幅 ÷ 列の数・高さ ÷ 行の数
- レイヤーは一覧の先頭がいちばん下。Piskel は `opacity`（0〜1）を掛けて上に重ねる（書き出しの `LayerUtils.flattenFrameAt`）。
  レイヤーを隠す仕組みは無く、不透明度 0 が見えないレイヤー
- `hiddenFrames` は Piskel のプレビューで飛ばすコマの番号。PNG の書き出しには入るので、読むときも捨てない（警告だけ）
- `fps` は 1 つだけ（Piskel の速さのつまみは 0〜24 の整数）。コマごとの時間は持たない。無ければ 12

標準ライブラリ（json・base64）と NumPy・Pillow だけを使う（Pyodide で動く）。
"""

from __future__ import annotations

import base64
import binascii
import io
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MODEL_VERSION = 2
DEFAULT_FPS = 12          # Piskel が fps の無いファイルに使う値
MAX_FPS = 24              # Piskel の速さのつまみの上限（0〜24）
MIN_FPS = 1
LAYER_NAME = "Layer 1"    # Piskel が新しい絵に付けるレイヤーの名前
MAX_PISKEL_FRAMES = 256
MAX_PISKEL_SIDE = 1024
PNG_PREFIX = "data:image/png;base64,"


class PiskelError(ValueError):
    """.piskel を読めない（壊れている・形が違う）。"""


@dataclass
class PiskelFile:
    """読み込んだ .piskel。`frames` は見えるレイヤーを重ねた各コマ（RGBA の uint8、高さ×幅×4）。"""

    name: str
    width: int
    height: int
    fps: float
    frames: list[np.ndarray]
    layers: list[dict] = field(default_factory=list)   # {"name", "opacity", "frames"}（下から）
    hidden_frames: list[int] = field(default_factory=list)   # Piskel のプレビューで飛ばすコマ（1 から）
    warnings: list[str] = field(default_factory=list)
    model_version: int = MODEL_VERSION


# ---------------------------------------------------------------- 速さ

def fps_for_durations(durations) -> tuple[int, list[str]]:
    """コマごとの時間（ミリ秒）から Piskel の速さ（1 秒あたりのコマ数。1〜24 の整数）を決める。

    いちばん多い時間（同じ数なら先に出てくる時間）を 1000 ÷ 時間 の四捨五入にする。返す値は (fps, 説明)。
    """
    ms = [int(d) for d in durations]
    if not ms or any(d < 1 for d in ms):
        raise ValueError(f"表示時間は 1 以上のミリ秒にしてください: {ms}")
    counts = Counter(ms)
    best = max(counts.values())
    common = next(d for d in ms if counts[d] == best)
    fps = int(min(MAX_FPS, max(MIN_FPS, (2000 // common + 1) // 2)))   # 1000 / common の四捨五入（整数だけで計算）
    notes = []
    if len(counts) > 1:
        notes.append(f"コマごとの時間が違う（{'/'.join(map(str, ms))} ミリ秒）ので、Piskel の速さはいちばん多い {common} ミリ秒から決めました"
                     "（Piskel は速さを 1 つしか持てません。本当の時間は JSON に残っています）")
    if 1000 % fps or 1000 // fps != common:
        notes.append(f"Piskel の速さは 1 秒あたりのコマ数（整数）なので、{common} ミリ秒は {fps} FPS（1 コマ約 {round(1000 / fps)} ミリ秒）"
                     "で再生されます。直すときの目安にしてください")
    return fps, notes


# ---------------------------------------------------------------- 書き出し

def _png_data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(buf, format="PNG")
    return PNG_PREFIX + base64.b64encode(buf.getvalue()).decode("ascii")


def piskel_text(frames: list[np.ndarray], name: str, fps: int = DEFAULT_FPS, description: str = "") -> str:
    """コマ（同じ大きさの RGBA）から .piskel の中身を作る（レイヤー 1 つ、chunk 1 つ、横 1 列）。

    Piskel の「Open」（Open Piskel）で開くと、コマ・大きさ・速さが入った状態になる。同じ入力からは同じ文字列になる。
    """
    if not frames:
        raise ValueError("コマがありません")
    shape = np.asarray(frames[0]).shape
    if any(np.asarray(f).shape != shape for f in frames):
        raise ValueError("コマの大きさがそろっていません")
    if len(shape) != 3 or shape[2] != 4:
        raise ValueError(f"コマは 高さ×幅×4（RGBA）の配列にしてください: {shape}")
    h, w = int(shape[0]), int(shape[1])
    strip = np.concatenate([np.asarray(f, np.uint8) for f in frames], axis=1)
    layer = {"name": LAYER_NAME, "opacity": 1, "frameCount": len(frames),
             "chunks": [{"layout": [[i] for i in range(len(frames))], "base64PNG": _png_data_url(strip)}]}
    doc = {"modelVersion": MODEL_VERSION,
           "piskel": {"name": str(name), "description": str(description), "fps": int(fps), "height": h, "width": w,
                      "layers": [json.dumps(layer, ensure_ascii=True, separators=(",", ":"))], "hiddenFrames": []}}
    return json.dumps(doc, ensure_ascii=True, separators=(",", ":"))


def piskel_layers_text(layers: list, name: str, fps: int = DEFAULT_FPS, description: str = "") -> str:
    """いくつかのレイヤー（`[(レイヤーの名前, コマの一覧), ...]`。先頭がいちばん下）の .piskel の中身。
    どのレイヤーもコマの数・大きさは同じにする（Piskel の決まり）。重ね（fx）の「キャラ」「エフェクト」に使う。"""
    if not layers:
        raise ValueError("レイヤーがありません")
    shape = np.asarray(layers[0][1][0]).shape
    count = len(layers[0][1])
    out = []
    for lname, frames in layers:
        if len(frames) != count or any(np.asarray(f).shape != shape for f in frames):
            raise ValueError("レイヤーのコマの数・大きさがそろっていません")
        strip = np.concatenate([np.asarray(f, np.uint8) for f in frames], axis=1)
        layer = {"name": str(lname), "opacity": 1, "frameCount": count,
                 "chunks": [{"layout": [[i] for i in range(count)], "base64PNG": _png_data_url(strip)}]}
        out.append(json.dumps(layer, ensure_ascii=True, separators=(",", ":")))
    h, w = int(shape[0]), int(shape[1])
    doc = {"modelVersion": MODEL_VERSION,
           "piskel": {"name": str(name), "description": str(description), "fps": int(fps), "height": h, "width": w,
                      "layers": out, "hiddenFrames": []}}
    return json.dumps(doc, ensure_ascii=True, separators=(",", ":"))


def piskel_name(file_name: str) -> str:
    """.piskel の `name`（Piskel の画面に出る名前）。ファイル名から拡張子を除いたもの。"""
    stem = str(file_name).rsplit("/", 1)[-1]
    return stem[:-len(stem.rsplit(".", 1)[-1]) - 1] if "." in stem else stem


# ---------------------------------------------------------------- 読み込み

def _decode_png(data_url) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise PiskelError("レイヤーの絵（base64PNG）がありません")
    head, body = data_url.split(",", 1)
    if not head.startswith("data:image/") or ";base64" not in head:
        raise PiskelError("レイヤーの絵（base64PNG）の形が違います")
    try:
        raw = base64.b64decode(body, validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            return np.asarray(im.convert("RGBA"), np.uint8).copy()
    except (binascii.Error, OSError, ValueError) as e:
        raise PiskelError(f"レイヤーの絵（base64PNG）を読めません: {e}") from None


def _layer_frames(layer: dict, width: int, height: int, lname: str) -> list[np.ndarray]:
    count = layer.get("frameCount")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_PISKEL_FRAMES:
        raise PiskelError(f"レイヤー「{lname}」のコマの数（frameCount）が読めません: {count!r}")
    chunks = layer.get("chunks")
    if chunks is None and layer.get("base64PNG"):
        # 古い形（chunk が無く、全部のコマを横に並べた絵が 1 枚）。Piskel の Deserializer.normalizeLayerData_ と同じに読む
        chunks = [{"base64PNG": layer["base64PNG"], "layout": [[i] for i in range(count)]}]
    if not isinstance(chunks, list) or not chunks:
        raise PiskelError(f"レイヤー「{lname}」に絵（chunks）がありません")
    frames: list[np.ndarray | None] = [None] * count
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise PiskelError(f"レイヤー「{lname}」の chunk の形が違います")
        layout = chunk.get("layout")
        if (not isinstance(layout, list) or not layout or not all(isinstance(c, list) and c for c in layout)
                or len({len(c) for c in layout}) != 1):
            raise PiskelError(f"レイヤー「{lname}」の並び（layout）の形が違います: {layout!r}")
        img = _decode_png(chunk.get("base64PNG"))
        cols, rows = len(layout), len(layout[0])
        if img.shape[1] != cols * width or img.shape[0] != rows * height:
            raise PiskelError(f"レイヤー「{lname}」の絵の大きさ（{img.shape[1]}×{img.shape[0]}）が、"
                              f"コマの大きさ {width}×{height} の {cols} 列 × {rows} 行と合いません")
        for i, col in enumerate(layout):
            for j, idx in enumerate(col):
                if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < count:
                    raise PiskelError(f"レイヤー「{lname}」の並び（layout）のコマの番号が違います: {idx!r}")
                frames[idx] = img[j * height:(j + 1) * height, i * width:(i + 1) * width].copy()
    missing = [i + 1 for i, f in enumerate(frames) if f is None]
    if missing:
        raise PiskelError(f"レイヤー「{lname}」の {'・'.join(map(str, missing))} コマ目の絵がありません")
    return frames   # type: ignore[return-value]


def _over(dst: np.ndarray, src: np.ndarray, opacity: float) -> np.ndarray:
    """src を opacity を掛けて dst の上に重ねる（ふつうの重ね方。結果は四捨五入して uint8）。"""
    sa = src[..., 3:4].astype(np.float64) / 255.0 * opacity
    da = dst[..., 3:4].astype(np.float64) / 255.0
    oa = sa + da * (1.0 - sa)
    rgb = src[..., :3].astype(np.float64) * sa + dst[..., :3].astype(np.float64) * da * (1.0 - sa)
    safe = np.where(oa > 0, oa, 1.0)
    out = np.empty(dst.shape, np.uint8)
    out[..., :3] = np.clip(np.rint(np.where(oa > 0, rgb / safe, 0.0)), 0, 255).astype(np.uint8)
    out[..., 3:4] = np.clip(np.rint(oa * 255.0), 0, 255).astype(np.uint8)
    return out


def read_piskel(data: bytes | str) -> PiskelFile:
    """.piskel の中身（バイトか文字列）を読む。見えるレイヤー（不透明度が 0 より大きい）を下から重ねて 1 枚にする。"""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise PiskelError(".piskel の文字を読めません（Piskel で保存したファイルか確かめてください）") from None
    else:
        text = str(data)
    try:
        doc = json.loads(text)
    except ValueError:
        raise PiskelError(".piskel の中身を読めません（Piskel の「Save」で保存したファイルか確かめてください）") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("piskel"), dict):
        raise PiskelError(".piskel の形が違います（\"piskel\" がありません）")
    version = doc.get("modelVersion")
    if version not in (1, 2):
        raise PiskelError(f"この .piskel は古い形（modelVersion {version!r}）なので読めません。"
                          "Piskel で開いて「Save」で保存し直してから選んでください")
    p = doc["piskel"]
    w, h = p.get("width"), p.get("height")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_PISKEL_SIDE for v in (w, h)):
        raise PiskelError(f".piskel の大きさ（width・height）が読めません: {w!r}×{h!r}")
    fps = p.get("fps", DEFAULT_FPS)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        fps = DEFAULT_FPS
    layers = p.get("layers")
    if not isinstance(layers, list) or not layers:
        raise PiskelError(".piskel にレイヤーがありません")
    parsed = []
    for k, entry in enumerate(layers):
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except ValueError:
                raise PiskelError(f"{k + 1} 番目のレイヤーを読めません") from None
        if not isinstance(entry, dict):
            raise PiskelError(f"{k + 1} 番目のレイヤーの形が違います")
        lname = str(entry.get("name") or f"Layer {k + 1}")
        op = entry.get("opacity", 1)
        try:
            op = float(op)
        except (TypeError, ValueError):
            op = 1.0
        if not 0 <= op <= 1:
            op = 1.0   # Piskel の Layer.setOpacity と同じく、範囲の外は 1
        parsed.append((lname, op, _layer_frames(entry, w, h, lname)))
    counts = {len(f) for _, _, f in parsed}
    if len(counts) != 1:
        raise PiskelError("レイヤーごとにコマの数が違います: "
                          + "、".join(f"「{n}」{len(f)} コマ" for n, _, f in parsed))
    n = counts.pop()
    visible = [(name, op, fr) for name, op, fr in parsed if op > 0]
    warnings = []
    if not visible:
        frames = [np.zeros((h, w, 4), np.uint8) for _ in range(n)]
        warnings.append("見えるレイヤーがありません（全部のレイヤーの不透明度が 0）")
    elif len(visible) == 1 and visible[0][1] == 1.0:
        frames = [f.copy() for f in visible[0][2]]
    else:
        frames = []
        for i in range(n):
            acc = np.zeros((h, w, 4), np.uint8)
            for _, op, fr in visible:
                acc = _over(acc, fr[i], op)
            frames.append(acc)
    if len(parsed) > 1:
        hidden = [name for name, op, _ in parsed if op <= 0]
        warnings.append(f"レイヤーが {len(parsed)} 枚あったので、見えるレイヤー {len(visible)} 枚を重ねて 1 枚にしました"
                        + (f"（不透明度 0 の「{'」「'.join(hidden)}」は入れていません）" if hidden else ""))
    faint = [name for name, op, _ in visible if op < 1]
    if faint:
        warnings.append(f"不透明度が 1 より小さいレイヤー（「{'」「'.join(faint)}」）があります。重ねた所が半透明になります")
    hidden_frames = sorted({int(v) + 1 for v in (p.get("hiddenFrames") or [])
                            if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n})
    if hidden_frames:
        warnings.append(f"Piskel のプレビューで飛ばす設定のコマ（{'・'.join(map(str, hidden_frames))} コマ目）がありますが、"
                        "そのまま入れました")
    return PiskelFile(name=str(p.get("name") or ""), width=w, height=h, fps=fps, frames=frames,
                      layers=[{"name": name, "opacity": op, "frames": len(fr)} for name, op, fr in parsed],
                      hidden_frames=hidden_frames, warnings=warnings, model_version=int(version))
