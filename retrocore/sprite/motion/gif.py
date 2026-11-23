"""動きの書き出し: そのまま見られる GIF アニメ（サイトに載せる・人に見せる用。ゲーム用はシートの PNG＋JSON）。

    from retrocore.sprite.motion.gif import save_gif
    gif = save_gif(res, "out/hero_breathe.gif", scale=4)            # 透明の背景
    gif = save_gif(res, "out/hero_breathe.gif", background="#203040")   # 背景色あり

決まりごと:
- 各フレームは、シートの各フレームとまったく同じ絵（新しい色を作らない・ずらさない）。拡大は最近傍だけ（1 ドットを
  scale × scale の正方形にする）
- パレットは、全フレームで使っている色をそのまま並べたもの（Pillow の減色・近い色への置き換えを通さない）。
  透明のときは 0 番を透明の色にする。GIF は 256 色までなので、それを超える絵は作らない（ValueError）
- 透明（α が 0）の画素は、背景を選んでいなければ透明、選んでいればその色。GIF の透明は「透明か不透明か」の 2 段階なので、
  半透明（α が 1〜254）の画素があれば α 128 を境に透明か不透明にして警告する（今の変換の結果には無い）
- 表示時間はシートの JSON の `duration_ms`。GIF は 10 ミリ秒単位なので、始まりからの時刻（累計）を 10 ミリ秒に丸めて
  各フレームの時間にする（1 枚ずつ丸めると合計がずれるため。割り切れないときは警告する）。どのフレームも 10 ミリ秒以上
- 同じ絵のフレームが続くときは、こちらで 1 枚にまとめて時間を足す（Pillow もまとめるが、版で振る舞いが変わらないように
  先にまとめる。合計の時間は変わらない）。まとめたことは `merged` に残す
- ずっと繰り返す（loop = 0）。透明のときは、次のフレームの前に前のフレームを消す（disposal = 2）。背景色ありのときは
  全面が不透明なので、前のフレームの上に変わった所だけを描く（disposal = 1。Pillow が変わった四角だけを書く）
- Pillow の GIF の書き出しは、`optimize=False`（パレットを詰め直さない）で使う。CLI とブラウザ版（Pyodide の Pillow）で
  同じ関数を通る
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .base import MotionResult

MIN_SCALE = 1
MAX_SCALE = 8
DEFAULT_SCALE = 4
GIF_TICK_MS = 10          # GIF の表示時間の単位（1/100 秒）
BROWSER_MIN_MS = 20       # これより短い時間は、多くのブラウザが 100 ミリ秒として表示する
MAX_COLORS = 256


@dataclass
class GifResult:
    """作った GIF。`durations` は書いたフレームごとの時間（まとめたあと）、`frame_durations` はシートのフレームごとの
    丸めたあとの時間（まとめる前。シートのフレームと同じ数）、`frames_of` は書いた各フレームがシートのどのフレーム（0 から）か。"""
    data: bytes
    scale: int
    background: tuple[int, int, int] | None
    durations: list[int]
    frame_durations: list[int]
    frames_of: list[list[int]]
    colors: int
    warnings: list[str] = field(default_factory=list)

    @property
    def merged(self) -> list[list[int]]:
        """1 枚にまとめたシートのフレームの組（0 から）。"""
        return [g for g in self.frames_of if len(g) > 1]

    def summary(self) -> dict:
        """画面と CLI の表示に使う情報（GIF のバイト列は含めない）。"""
        return {"scale": self.scale,
                "background": None if self.background is None else "#%02x%02x%02x" % self.background,
                "durations": list(self.durations), "frame_durations": list(self.frame_durations),
                "frames_of": [list(g) for g in self.frames_of], "merged": self.merged, "colors": self.colors,
                "warnings": list(self.warnings)}


def check_scale(scale) -> int:
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or not MIN_SCALE <= scale <= MAX_SCALE:
        raise ValueError(f"GIF の拡大は {MIN_SCALE}〜{MAX_SCALE} の整数で指定してください: {scale}")
    return int(scale)


def check_background(background) -> tuple[int, int, int] | None:
    """背景の指定（None・"transparent" = 透明、"#rrggbb"・(R, G, B) = その色）を (R, G, B) か None にする。"""
    if background is None:
        return None
    if isinstance(background, str):
        s = background.strip()
        if s.lower() in ("", "transparent", "none"):
            return None
        h = s[1:] if s.startswith("#") else s
        if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
            raise ValueError(f"GIF の背景は #rrggbb の形の色か transparent（透明）で指定してください: {background}")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    c = tuple(background)
    if len(c) != 3 or not all(isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in c):
        raise ValueError(f"GIF の背景は 0〜255 の (R, G, B) で指定してください: {background}")
    return tuple(int(v) for v in c)  # type: ignore[return-value]


def gif_durations(durations) -> tuple[list[int], list[str]]:
    """ミリ秒の表示時間を GIF の 10 ミリ秒単位にする（始まりからの時刻を丸める。合計は 10 で割り切れればそのまま）。

    返す値は (丸めたあとの各フレームの時間, 警告)。どのフレームも 10 ミリ秒以上にする（短すぎて 0 になるときは 10 にし、
    合計が変わることを警告する）。
    """
    ms = [int(d) for d in durations]
    if not ms or any(d < 1 for d in ms):
        raise ValueError(f"表示時間は 1 以上のミリ秒で指定してください: {ms}")
    out, prev, clock, raised = [], 0, 0, False
    for d in ms:
        clock += d
        end = (clock + GIF_TICK_MS // 2) // GIF_TICK_MS * GIF_TICK_MS   # 四捨五入（5 は上へ）
        if end < prev + GIF_TICK_MS:
            end, raised = prev + GIF_TICK_MS, True
        out.append(end - prev)
        prev = end
    warnings = []
    changed = [(i, a, b) for i, (a, b) in enumerate(zip(ms, out)) if a != b]
    if changed:
        listed = "、".join(f"{i + 1} 枚目 {a} → {b}" for i, a, b in changed)
        total = (f"合計 {sum(ms)} → {sum(out)} ミリ秒" if sum(ms) != sum(out)
                 else f"合計 {sum(ms)} ミリ秒は変えていません")
        warnings.append(f"GIF の表示時間は 10 ミリ秒単位なので、近い値にしました（{listed} ミリ秒。{total}）")
    if raised:
        warnings.append(f"GIF では 1 枚を {GIF_TICK_MS} ミリ秒より短くできないので、{GIF_TICK_MS} ミリ秒にした所があります")
    short = [i + 1 for i, d in enumerate(out) if d < BROWSER_MIN_MS]
    if short:
        warnings.append(f"{'・'.join(map(str, short))} 枚目は {BROWSER_MIN_MS} ミリ秒より短いので、多くのブラウザでは"
                        "100 ミリ秒として表示されます")
    return out, warnings


def _binary_alpha(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[str]]:
    """α を 0 か 255 にそろえる（半透明があれば α 128 を境にして警告する）。"""
    out, partial = [], 0
    for f in frames:
        a = np.asarray(f, np.uint8)
        if a.ndim != 3 or a.shape[2] != 4:
            raise ValueError(f"フレームは H×W×4（RGBA）の配列にしてください: {a.shape}")
        alpha = a[..., 3]
        mid = (alpha > 0) & (alpha < 255)
        if mid.any():
            partial += int(mid.sum())
            a = a.copy()
            a[..., 3] = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        out.append(a)
    warnings = []
    if partial:
        warnings.append(f"GIF の透明は「透明か不透明か」だけなので、半透明の画素 {partial} 個を、α 128 を境に透明か不透明にしました")
    return out, warnings


def make_gif(frames: list[np.ndarray], durations, scale: int = DEFAULT_SCALE, background=None) -> GifResult:
    """フレーム（RGBA の配列。シートのフレームと同じもの）と表示時間（ミリ秒）から GIF アニメを作る。"""
    scale = check_scale(scale)
    bg = check_background(background)
    if not frames:
        raise ValueError("フレームがありません")
    shape = np.asarray(frames[0]).shape
    if any(np.asarray(f).shape != shape for f in frames):
        raise ValueError("フレームの大きさがそろっていません")
    ms = [int(d) for d in durations] if not isinstance(durations, (int, np.integer)) else [int(durations)] * len(frames)
    if len(ms) != len(frames):
        raise ValueError(f"表示時間の数（{len(ms)}）とフレーム数（{len(frames)}）が合いません")
    frames, warnings = _binary_alpha(frames)
    tick, tick_warn = gif_durations(ms)
    warnings += tick_warn

    # 色（R・G・B を 1 つの数にしたもの）。透明の画素は数えない。背景色ありのときは背景色も色の 1 つにする
    keys = [((f[..., 0].astype(np.uint32) << 16) | (f[..., 1].astype(np.uint32) << 8) | f[..., 2].astype(np.uint32))
            for f in frames]
    opaque = [f[..., 3] == 255 for f in frames]
    used = [k[o] for k, o in zip(keys, opaque)]
    if bg is not None:
        used.append(np.array([(bg[0] << 16) | (bg[1] << 8) | bg[2]], np.uint32))
    colors = np.unique(np.concatenate(used)) if used else np.zeros(0, np.uint32)
    first = 1 if bg is None else 0   # 透明のときは 0 番を透明に使う
    if len(colors) + first > MAX_COLORS:
        raise ValueError(f"GIF は {MAX_COLORS} 色まで（透明のときは {MAX_COLORS - 1} 色まで）ですが、"
                         f"この動きは {len(colors)} 色を使っています（色数を減らして変換してください）")
    if bg is not None:
        fill = int(np.searchsorted(colors, (bg[0] << 16) | (bg[1] << 8) | bg[2]))
    else:
        fill = 0
    palette = [0, 0, 0] * first
    for c in colors.tolist():
        palette += [(c >> 16) & 255, (c >> 8) & 255, c & 255]

    # 同じ絵が続くフレームは 1 枚にまとめる（時間は足す）
    groups: list[list[int]] = []
    for i in range(len(frames)):
        if groups and np.array_equal(frames[i], frames[groups[-1][0]]):
            groups[-1].append(i)
        else:
            groups.append([i])
    images = []
    for g in groups:
        i = g[0]
        idx = np.full(shape[:2], fill, np.uint8)
        o = opaque[i]
        if len(colors):
            idx[o] = (np.searchsorted(colors, keys[i][o]) + first).astype(np.uint8)
        if scale > 1:
            idx = np.repeat(np.repeat(idx, scale, axis=0), scale, axis=1)
        im = Image.fromarray(np.ascontiguousarray(idx), "P")
        im.putpalette(palette)
        images.append(im)
    gif_ms = [sum(tick[i] for i in g) for g in groups]
    params = {"format": "GIF", "save_all": True, "append_images": images[1:], "duration": gif_ms,
              "loop": 0, "optimize": False, "disposal": 2 if bg is None else 1, "background": fill}
    if bg is None:
        params["transparency"] = 0
    buf = io.BytesIO()
    images[0].save(buf, **params)
    return GifResult(buf.getvalue(), scale, bg, gif_ms, tick, groups, int(len(colors)), warnings)


def frames_from_sheet(sheet: np.ndarray, info: dict) -> tuple[list[np.ndarray], list[int]]:
    """シートの画像と情報（`sheet_info()` の辞書）から、フレームの配列と表示時間を取り出す（ブラウザ版はこちらを通る）。"""
    a = np.asarray(sheet, np.uint8)
    w, h = int(info["frame_width"]), int(info["frame_height"])
    frames, durations = [], []
    for f in info["frames"]:
        x, y = int(f["x"]), int(f["y"])
        if x < 0 or y < 0 or x + w > a.shape[1] or y + h > a.shape[0]:
            raise ValueError(f"シートの情報のフレームの位置がシートの外です: {f}")
        frames.append(a[y:y + h, x:x + w])
        durations.append(int(f["duration_ms"]))
    return frames, durations


def motion_gif(result: MotionResult, scale: int = DEFAULT_SCALE, background=None) -> GifResult:
    """動き（`make_motion()` などの結果）から GIF を作る。作らなかった動き（`skipped`）からは作らない（ValueError）。"""
    if result.skipped:
        raise ValueError("動きを作らなかったので、GIF も作りません（理由は _skipped.json）")
    return make_gif(result.frames, result.durations, scale, background)


def gif_path(sheet_png: str | Path) -> Path:
    """シートの PNG（<出力名>_<動き>_sheet.png）に対する GIF の保存先（<出力名>_<動き>.gif）。"""
    p = Path(sheet_png)
    stem = p.stem[:-len("_sheet")] if p.stem.endswith("_sheet") else p.stem
    return p.with_name(stem + ".gif")


def save_gif(result: MotionResult, path: str | Path, scale: int = DEFAULT_SCALE, background=None) -> GifResult:
    """GIF を作って保存する（親フォルダが無ければ作る）。"""
    gif = motion_gif(result, scale, background)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gif.data)
    return gif
