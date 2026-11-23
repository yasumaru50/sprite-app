"""パレット統合の工程（第一歩）: パレットファイルの読み書きと、指定したパレットに合わせる処理。

目的は、自分のゲームで使う複数のキャラクターの色をそろえること。進め方は次の 2 段階:
1. キャラクターを 1 体変換して、使った色をパレットファイルに保存する（`save_palette()`。CLI `--save-palette`）
2. 次のキャラクターを変換するときに、そのパレットを指定して、その色だけで描く（`PaletteStep`。CLI `--palette`）

**パレットファイル**（外部のドット絵エディタ Aseprite・Piskel・LibreSprite などでも使える形式）:

| 拡張子 | 読み込み | 書き出し | 中身 |
| --- | --- | --- | --- |
| `.hex` | ○ | ○ | 1 行に `RRGGBB` を 1 つ（先頭の `#` は省いてもよい）。Lospec などで使われる形式 |
| `.gpl` | ○ | ○ | GIMP のパレット（`GIMP Palette` の行、`Name:`・`Columns:`、`R G B 名前` の行）。Aseprite も読み書きできる |
| `.png` | ○ | × | パレットの見本画像。不透明な画素（α ≥ 128）の色を、重複なしで、画像の上から・左からの順に読む |

読み込んだ色は、ファイルの順のまま使う（重複は最初のものだけ残す）。書き出すときは `sort_palette()` の順に並べる。

**色の並び順**（`sort_palette()`）: 人が見て分かりやすいように、
1. 色みのほとんど無い色（灰色・黒・白。OKLab の彩度 √(a² + b²) が `GRAY_CHROMA` 未満）を先頭に、暗い色から明るい色へ
2. 残りは色み（OKLab の色相の角度）でグループに分ける。色相の順に並べて、となりとの角度の差が `HUE_GAP`（25°）以上
   空いている所で切る。1 つのグループが `MAX_HUE_SPAN`（90°）より広ければ、その中のいちばん大きなすき間でさらに切る
3. グループは色相の角度の順（0° から。赤 → 橙・肌 → 黄 → 緑 → 水色 → 青 → 紫 → ピンク）、グループの中は暗い色から明るい色へ
固定の区切り（30° ごとなど）にしないのは、同じ素材の明暗（暗い赤 → 赤 → 明るい赤）は色相が少しずつずれるため、
固定の区切りではちょうど境目で別のグループに分かれてしまうことがあるから。

**パレットに合わせる**（`match_palette()`）: 不透明な画素の色を、パレットの中で最も近い色にする。
- 色の近さは単純化と同じ OKLab の距離（色み a・b は 1.5 倍。`simplify._distance()`）。ディザは使わない
- 透明な画素（α < 128）はそのまま。不透明な画素は α を 255 にする（パレットは不透明の色だけなので、出力は透明か不透明の 2 段階）
- 同点はパレットの並びの早い色（乱数なし。同じ入力からは同じ結果）
- 新しい色は足さない。もとは別の色だった部品が、パレットの同じ色に合わさって消えることがある
  （頬のピンクと肌が、パレットの同じ肌色になる、など）。そのときは「同じ色になった組」（`PaletteMatch.collisions`）として
  どの色とどの色が、パレットのどの色になったか、それぞれ何ドットかを返す（CLI は警告を表示する）。
  見分けのつかない色（距離 `merge_distance` 未満。単純化の `merge_close_colors()`）は 1 つの部品として数える。
  色の種類が `MAX_WARN_COLORS`（512）より多いとき（単純化しないとき）は、ドットの多い 512 色だけで調べる
- パレットにはっきり近い色が無い色（距離 `FAR_DISTANCE` 以上）も「遠い色」（`PaletteMatch.far`）として返す

**どの工程で合わせるか**（CLI `--palette`）: 単純化の直後に `PaletteStep` を置く（単純化しないときはグリッド検出・縮小の直後）。
- 単純化で、ぼかしの名残の中間色を除いた「部品の色」（既定 12 色まで）にしてから合わせるので、部品ごとに 1 つずつパレットの色が
  決まり、「同じ色になった組」の警告が部品どうしの話になる（数百色のまま合わせると、中間色どうしの重なりで警告が埋もれる）
- そのあとのクリーンアップは新しい色を作らず、絵の中にある色だけで置き換えるので、パレットの色のまま動く
- 縁取りは `OutlineStep(palette=...)` で、外側に付ける輪郭の色もパレットの中から選ぶ（`outline.py`）。切れ目を埋めるときは
  もともと絵の中の輪郭の色を使う
- 最後に `PaletteStep(final=True)` で、パレットにない色が残っていないかを確かめる（残っていれば最も近い色にする。ふつうは 0 ドット）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import Step
from .simplify import ALPHA_OPAQUE, DEFAULT_MERGE_DISTANCE, _distance, merge_close_colors, rgb_to_oklab

READ_FORMATS = (".hex", ".gpl", ".png")
WRITE_FORMATS = (".hex", ".gpl")
MAX_PALETTE_COLORS = 256
# 並び順
GRAY_CHROMA = 0.025      # OKLab の彩度がこれ未満なら「色みのほとんど無い色」
HUE_GAP = 25.0           # 色相の角度の差がこれ以上空いている所でグループを切る
MAX_HUE_SPAN = 90.0      # 1 つのグループの色相の広がりの上限
# パレットに合わせる
FAR_DISTANCE = 0.1       # パレットの最も近い色との距離がこれ以上なら「遠い色」（0.1 以上ではっきり違う色）
MAX_WARN_COLORS = 512    # 警告を調べる色の数の上限（ドットの多い色から）
GPL_COLUMNS = 8


class PaletteError(ValueError):
    """パレットファイルを読めない・書けないときのエラー。"""


# --- 読み書き -----------------------------------------------------------------------------

def _unique_keep_order(cols: np.ndarray) -> np.ndarray:
    cols = np.asarray(cols, np.uint8).reshape(-1, 3)
    if len(cols) == 0:
        return cols
    key = cols.astype(np.int64) @ np.array([65536, 256, 1])
    _, first = np.unique(key, return_index=True)
    return cols[np.sort(first)]


_HEX_LINE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex(text: str) -> np.ndarray:
    """.hex の中身（1 行に RRGGBB）を k×3 の uint8 にする。空行と `;` で始まる行は読み飛ばす。"""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip().lstrip("﻿")
        if not s or s.startswith(";"):
            continue
        m = _HEX_LINE.match(s)
        if not m:
            raise PaletteError(f".hex の {n} 行目が RRGGBB の形ではありません: {s!r}")
        v = int(m.group(1), 16)
        out.append(((v >> 16) & 255, (v >> 8) & 255, v & 255))
    return _unique_keep_order(np.array(out, np.uint8).reshape(-1, 3))


def parse_gpl(text: str) -> np.ndarray:
    """.gpl（GIMP のパレット）の中身を k×3 の uint8 にする。

    1 行目は `GIMP Palette`。`Name:`・`Columns:`・`Channels:` の行、`#` で始まる行、空行は読み飛ばす。
    色の行は `R G B [名前]`（Aseprite の `Channels: RGBA` では `R G B A [名前]`。α は使わない）。
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or not lines[0].strip().startswith("GIMP Palette"):
        raise PaletteError(".gpl の 1 行目が 'GIMP Palette' ではありません")
    out = []
    for n, line in enumerate(lines[1:], 2):
        s = line.strip()
        if not s or s.startswith("#") or re.match(r"^(Name|Columns|Channels)\s*:", s):
            continue
        parts = s.split()
        try:
            rgb = [int(v) for v in parts[:3]]
        except ValueError:
            raise PaletteError(f".gpl の {n} 行目が 'R G B' の形ではありません: {s!r}")
        if len(rgb) < 3 or not all(0 <= v <= 255 for v in rgb):
            raise PaletteError(f".gpl の {n} 行目の色が 0〜255 の 3 つの数ではありません: {s!r}")
        out.append(rgb)
    return _unique_keep_order(np.array(out, np.uint8).reshape(-1, 3))


def palette_from_image(rgba: np.ndarray) -> np.ndarray:
    """画像の不透明な画素（α ≥ 128）の色を、重複なしで、上から・左からの順に返す。"""
    a = np.asarray(rgba)
    return _unique_keep_order(a[..., :3][a[..., 3] >= ALPHA_OPAQUE])


def load_palette(path: str | Path) -> np.ndarray:
    """パレットファイル（.hex / .gpl / .png）を読み、k×3 の uint8（ファイルの順、重複なし）を返す。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in READ_FORMATS:
        raise PaletteError(f"パレットは {' / '.join(READ_FORMATS)} のファイルで指定してください: {path}")
    try:
        if ext == ".png":
            with Image.open(path) as img:
                img.load()
                cols = palette_from_image(np.asarray(img.convert("RGBA")))
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            cols = parse_hex(text) if ext == ".hex" else parse_gpl(text)
    except OSError as e:
        raise PaletteError(f"パレットを読み込めません: {path}（{e}）")
    if len(cols) == 0:
        raise PaletteError(f"パレットに色がありません: {path}")
    if len(cols) > MAX_PALETTE_COLORS:
        raise PaletteError(f"パレットの色が多すぎます（{len(cols)} 色。{MAX_PALETTE_COLORS} 色まで）: {path}")
    return cols


def format_hex(colors) -> str:
    """.hex の中身（1 行に小文字の rrggbb）。"""
    return "".join(f"{r:02x}{g:02x}{b:02x}\n" for r, g, b in np.asarray(colors, np.uint8).reshape(-1, 3).tolist())


def format_gpl(colors, name: str = "sprite") -> str:
    """.gpl の中身。色の名前は #RRGGBB。"""
    lines = ["GIMP Palette", f"Name: {name}", f"Columns: {GPL_COLUMNS}", "#"]
    for r, g, b in np.asarray(colors, np.uint8).reshape(-1, 3).tolist():
        lines.append(f"{r:3d} {g:3d} {b:3d}\t#{r:02X}{g:02X}{b:02X}")
    return "\n".join(lines) + "\n"


def save_palette(colors, path: str | Path, *, sort: bool = True, name: str | None = None) -> np.ndarray:
    """色をパレットファイル（拡張子で .hex / .gpl を決める）に書き出し、書いた順の色を返す。

    `sort=True`（既定）なら `sort_palette()` の順にする。重複は除く。親フォルダが無ければ作る。
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in WRITE_FORMATS:
        raise PaletteError(f"パレットの保存先は {' / '.join(WRITE_FORMATS)} のファイルで指定してください: {path}")
    cols = _unique_keep_order(np.asarray(colors, np.uint8).reshape(-1, 3))
    if sort:
        cols = sort_palette(cols)
    text = format_hex(cols) if ext == ".hex" else format_gpl(cols, name or path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return cols


# --- 並び順 -------------------------------------------------------------------------------

def _hue_groups(hue: np.ndarray) -> list[np.ndarray]:
    """色相（度）を、すき間で切ったグループ（各グループは色相の順の番号）に分ける。"""
    n = len(hue)
    order = np.argsort(hue, kind="stable")
    h = hue[order]
    if n == 1:
        return [order]
    gaps = np.diff(np.concatenate([h, h[:1] + 360.0]))   # gaps[i]: h[i] → h[i+1]（最後は 1 周して先頭へ）
    cuts = np.flatnonzero(gaps >= HUE_GAP)
    if len(cuts) == 0:
        cuts = np.array([int(np.argmax(gaps))])
    groups = []
    for k, c in enumerate(cuts):
        nxt = cuts[(k + 1) % len(cuts)]
        idx = [(c + 1 + i) % n for i in range((nxt - c) % n or n)]
        groups.append(order[idx])

    def span(g):
        hh = hue[g]
        return float((hh[-1] - hh[0]) % 360.0)

    out = []
    stack = groups[::-1]
    while stack:
        g = stack.pop()
        if len(g) > 1 and span(g) > MAX_HUE_SPAN:
            d = np.diff(hue[g]) % 360.0
            c = int(np.argmax(d))
            stack.extend([g[c + 1:], g[:c + 1]])
        else:
            out.append(g)
    return out


def _circular_mean(deg: np.ndarray) -> float:
    r = np.deg2rad(deg)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)


def sort_palette(colors) -> np.ndarray:
    """人が見て分かりやすい順に並べる（灰色を先頭に暗い→明るい、次に色みのグループごとに暗い→明るい）。

    詳しくはモジュールの説明。乱数なし。同じ明るさは RGB の値の小さい順。
    """
    cols = _unique_keep_order(colors)
    if len(cols) <= 1:
        return cols
    lab = rgb_to_oklab(cols.astype(np.float64))
    L = lab[:, 0]
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.rad2deg(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    key = cols.astype(np.int64) @ np.array([65536, 256, 1])

    def by_light(idx):
        idx = np.asarray(idx)
        return idx[np.lexsort((key[idx], L[idx]))]

    gray = np.flatnonzero(chroma < GRAY_CHROMA)
    chrom = np.flatnonzero(chroma >= GRAY_CHROMA)
    parts = [by_light(gray)] if len(gray) else []
    if len(chrom):
        groups = [chrom[g] for g in _hue_groups(hue[chrom])]
        groups.sort(key=lambda g: (_circular_mean(hue[g]), int(key[g].min())))
        parts.extend(by_light(g) for g in groups)
    return cols[np.concatenate(parts)]


# --- パレットに合わせる -------------------------------------------------------------------

@dataclass
class Collision:
    """もとは別の色だった部品が、パレットの同じ色になった組。"""

    target: tuple                # パレットの色（RGB）
    sources: list                # [(もとの色 RGB, ドット数), ...]（ドットの多い順）

    @property
    def dots(self) -> int:
        """いちばん多い部品以外のドット数（見分けがつかなくなったドットの数の目安）。"""
        return int(sum(n for _, n in self.sources[1:]))

    def message(self) -> str:
        src = " と ".join(f"{_hexs(c)}（{n} ドット）" for c, n in self.sources)
        return f"{src} が、パレットの同じ色 {_hexs(self.target)} になりました"


@dataclass
class FarColor:
    """パレットにはっきり近い色が無かった色。"""

    source: tuple
    target: tuple
    dots: int
    distance: float

    def message(self) -> str:
        return (f"{_hexs(self.source)}（{self.dots} ドット）に近い色がパレットにありません。"
                f"いちばん近い {_hexs(self.target)} にしました（距離 {self.distance:.3f}）")


@dataclass
class PaletteMatch:
    """`match_palette()` の結果。"""

    image: np.ndarray            # パレットの色だけにした RGBA
    palette: np.ndarray          # 指定したパレット（k×3）
    used: np.ndarray             # 使ったパレットの色の番号（パレットの順）
    colors_before: int           # 合わせる前の色数（不透明な部分）
    changed: int                 # 色が変わった不透明なドットの数
    collisions: list = field(default_factory=list)   # [Collision]（ドットの多い順）
    far: list = field(default_factory=list)          # [FarColor]（ドットの多い順）

    @property
    def warnings(self) -> list[str]:
        return [c.message() for c in self.collisions] + [f.message() for f in self.far]


def _hexs(c) -> str:
    r, g, b = (int(v) for v in c)
    return f"#{r:02x}{g:02x}{b:02x}"


def check_palette(palette) -> np.ndarray:
    p = np.asarray(palette)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("パレットは 1 色以上の n×3 の配列で指定してください")
    if not np.all((p >= 0) & (p <= 255) & (np.round(p) == p)):
        raise ValueError("パレットの色は 0〜255 の整数で指定してください")
    return _unique_keep_order(p.astype(np.uint8))


def nearest_in_palette(rgb, palette) -> np.ndarray:
    """各色（n×3）について、パレットで最も近い色の番号（OKLab、色み 1.5 倍。同点はパレットの早い色）。"""
    rgb = np.asarray(rgb, np.float64).reshape(-1, 3)
    pal = np.asarray(palette, np.float64).reshape(-1, 3)
    if len(rgb) == 0:
        return np.zeros(0, np.int64)
    return np.argmin(_distance(rgb_to_oklab(rgb), rgb_to_oklab(pal)), axis=1)


def match_palette(rgba: np.ndarray, palette, merge_distance: float = DEFAULT_MERGE_DISTANCE,
                  far_distance: float = FAR_DISTANCE) -> PaletteMatch:
    """不透明な画素をパレットの中で最も近い色にする（透明はそのまま。ディザなし。新しい色は足さない）。

    別の部品の色（`merge_distance` 以上離れた色）がパレットの同じ色になったら `collisions` に、
    パレットの最も近い色との距離が `far_distance` 以上の色は `far` に入れる。入力の配列は書き換えない。
    """
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4 or a.dtype != np.uint8:
        raise ValueError("RGBA の uint8 配列（高さ × 幅 × 4）が必要です")
    pal = check_palette(palette)
    out = a.copy()
    opaque = a[..., 3] >= ALPHA_OPAQUE
    if not opaque.any():
        return PaletteMatch(out, pal, np.zeros(0, np.int64), 0, 0)
    rgb = a[..., :3][opaque]
    key = rgb.astype(np.int64) @ np.array([65536, 256, 1])
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    lab = rgb_to_oklab(cols.astype(np.float64))
    plab = rgb_to_oklab(pal.astype(np.float64))
    dist = _distance(lab, plab)
    idx = np.argmin(dist, axis=1)
    near = dist[np.arange(len(cols)), idx]
    new = pal[idx]
    out[..., :3][opaque] = new[inv]
    out[..., 3][opaque] = 255
    changed = int(cnt[np.any(new != cols, axis=1)].sum())

    # 部品（見分けのつかない色はまとめて 1 つ）ごとに、どのパレットの色になったか。
    # 色の種類が多い（単純化していない）ときは、ドットの多い MAX_WARN_COLORS 色だけで調べる（距離の表が n×n になるため）
    sel = np.arange(len(cols))
    if len(sel) > MAX_WARN_COLORS:
        sel = np.sort(np.lexsort((uniq, -cnt))[:MAX_WARN_COLORS])
    parts = merge_close_colors(cols[sel], cnt[sel].astype(np.float64), merge_distance)
    k = len(parts.colors)
    part_dots = np.bincount(parts.labels, weights=cnt[sel], minlength=k).astype(np.int64)
    part_target = idx[sel][parts.keep]       # 部品の代表の色が合わさった先
    part_near = near[sel][parts.keep]
    collisions = []
    for t in np.unique(part_target):
        ps = np.flatnonzero(part_target == t)
        if len(ps) < 2:
            continue
        ps = ps[np.lexsort((np.arange(len(ps)), -part_dots[ps]))]
        collisions.append(Collision(tuple(int(v) for v in pal[t]),
                                    [(tuple(int(v) for v in parts.colors[p]), int(part_dots[p])) for p in ps]))
    collisions.sort(key=lambda c: (-c.dots, c.target))
    far = [FarColor(tuple(int(v) for v in parts.colors[p]), tuple(int(v) for v in pal[part_target[p]]),
                    int(part_dots[p]), float(part_near[p]))
           for p in np.flatnonzero(part_near >= far_distance)]
    far.sort(key=lambda f: (-f.dots, f.source))
    return PaletteMatch(out, pal, np.unique(idx), len(cols), changed, collisions, far)


def used_colors(rgba: np.ndarray) -> np.ndarray:
    """画像の不透明な画素（α ≥ 128）で使われている色（重複なし、RGB の値の小さい順。透明は含めない）。"""
    a = np.asarray(rgba)
    rgb = a[..., :3][a[..., 3] >= ALPHA_OPAQUE]
    if len(rgb) == 0:
        return np.zeros((0, 3), np.uint8)
    key = np.unique(rgb.astype(np.int64) @ np.array([65536, 256, 1]))
    return np.stack([(key >> 16) & 255, (key >> 8) & 255, key & 255], axis=1).astype(np.uint8)


@dataclass
class PaletteStep(Step):
    """指定したパレットに合わせる工程。直前の結果は `last`（`PaletteMatch`）に残る。

    `final=True` は最後の確認用（縁取りなどの後に、パレットにない色が残っていないかを確かめる。ふつうは何も変わらない）。
    """

    palette: np.ndarray = None
    merge_distance: float = DEFAULT_MERGE_DISTANCE
    final: bool = False
    name: str = "palette"
    last: PaletteMatch | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.palette = check_palette(self.palette)
        if self.merge_distance < 0:
            raise ValueError(f"まとめる距離は 0 以上で指定してください（{self.merge_distance}）")
        if self.final and self.name == "palette":
            self.name = "palette-check"

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        self.last = match_palette(rgba, self.palette, self.merge_distance)
        return self.last.image

    def describe(self) -> str:
        r = self.last
        if self.final:
            if r is None:
                return "パレットの確認"
            return f"パレットの確認（パレットにない色だったドット {r.changed}）"
        if r is None:
            return "パレットに合わせる"
        return (f"パレットに合わせる（{r.colors_before} 色 → パレット {len(self.palette)} 色のうち {len(r.used)} 色、"
                f"同じ色になった組 {len(r.collisions)}、遠い色 {len(r.far)}）")
