"""Sprite App の動き（フェーズ 2）の確認用の画像を作る。呼吸（breathe）・横揺れ（sway）・浮遊（float）・弾む（bounce）・うなずく（nod）・
伸び（stretch）。

    python tools/compare_sprite_motion.py 入力.png [入力 ...] [-o 出力フォルダ] [--motion breathe|sway|float|bounce|nod|stretch]
                                          [--frames N] [--depth 1] [--amount 1] [--duration MS] [--zoom 8] [--converted]
                                          [--split auto --split 12 ...] [--rows]
                                          [--height 1 --height 3 ...] [--compare-durations] [--vs-float]
                                          [--spec 表.motion.json [--vs-spec 比べる表.motion.json]] [--keep off|on|both]

入力ごとに、CLI（`python -m apps.sprite`）の既定の設定で変換し（`--converted` なら入力をそのまま変換済みのスプライトとして使う）、
動きを付けて、出力フォルダ（既定 `out/sprite/motion/`）に次を保存する。

- `<名前>_<動き>.gif`: 動きを見るための GIF（`--zoom` 倍に最近傍で拡大、ぼかさない、繰り返し再生、背景は無地の色）
- `<名前>_<動き>_frames.png`: 全フレームを横に並べて `--zoom` 倍にしたもの。前のフレーム（1 枚目は最後のフレーム）から
  変わったドットにマゼンタの枠、上下を分けた継ぎ目（呼吸で抜いたドット・横揺れの分け目）の位置に、左右の端の外側に水色の印を付ける。
  浮遊では左に行の番号（元の絵の下から何ドット目か。上下に足した余白の行は灰色）を付け、1 枚目の足元（pivot）の高さに黄色の線を引く
  （絵全体が動くので、変わったドットの枠は付けず、数だけ書く）
- `<名前>_<動き>_sheet.png` と `.json`: CLI の `--motion` と同じ、実際に出力されるスプライトシート（拡大しない）と情報ファイル
- `--split`（分ける高さ。CLI の `--breathe-split` / `--sway-split` と同じ。何度でも指定できる）を付けると、高さごとに
  `<名前>_<動き>_split<高さ>`（自動は `<名前>_<動き>_split-auto`）の名前で上の 3 つを作る
- 弾む（`--motion bounce`）は `--height`（何度でも指定できる）ごとに `<名前>_bounce_h<高さ>_<表示時間>` の名前で上の 3 つを作る
  （表示時間は既定の着地だけ長め `landing300`、`--duration 150` なら `uniform150`）。行の番号と黄色の線（pivot）は浮遊と同じ。
  `--compare-durations` を付けると「全部 150 ミリ秒」と「着地だけ長め」の両方を作り、同じ時計で横に並べた
  `<名前>_bounce_h<高さ>_uniform150_vs_landing300.gif` も作る。`--vs-float` を付けると、浮遊（既定）と弾む（既定）を
  同じ時計で横に並べた `<名前>_float_vs_bounce_h<高さ>.gif` も作る
- うなずく（`--motion nod`）は `--split`（自動の首は `auto`。何度でも）と `--depth`（1・2。何度でも）の組み合わせごとに
  `<名前>_nod_d<深さ>_split<高さ>`（自動は `split-auto`）の名前で上の 3 つと、分けた所に線を引いたフレームの並び
  `<名前>_nod_d<深さ>_split<高さ>_line.png`（水色の線 = 1 本目の継ぎ目の下の縁。ここより下は動かない。2 本目は黄色。
  左に行の番号、自動で見つけた首の行に赤い印）を作る。作れなかったときは警告の文を `<名前>_nod_skipped.txt` に足していく
- 伸び（`--motion stretch`）はうなずくと同じく `--split` と `--depth`（伸びる量 1・2）の組み合わせごとに
  `<名前>_stretch_a<量>_split<高さ>` の名前で上の 3 つと `_line.png` を作る（線は継ぎ目（複製したドット）の下の縁。2 本目は黄色）。
  作れなかったときは `<名前>_stretch_skipped.txt` に足していく
- `--keep on|both`（呼吸・うなずく・伸び）: 小さな部品を守る（CLI の `--keep-small-parts`）。on は名前の後ろに `_keep` を付けて作る。
  both は切と入の両方を作り、切と入を同じ時計で横に並べた `<名前>_keep_off_vs_on.gif` と、線を引いた並びを上下に重ねた
  `<名前>_keep_off_vs_on_line.png`（上が切、下が入）も作る。呼吸で `--keep` を付けたときも `_line.png` を作る
- `--spec 表.motion.json` を付けると、プリセットの代わりに動きの表（CLI の `--motion-spec` と同じ）で作り、
  `<名前>_<表の name>` の名前で上の 3 つと、実際に使った分け目の線（表の "used"）を引いた `_line.png` を作る
  （フレームの並びの上には、行の番号（1 から）とその行の列の値を書く。線は縮める = 水色・黄色、伸ばす = 緑・橙）。
  表に帯・四角の範囲があれば、範囲を色付きの枠で描いたフレームの並び `_regions.png` も作る（範囲ごとに赤・緑・橙・水色。
  作れなかったときは元の絵に範囲の枠と、問題のドット（マゼンタ）と、すすめられた範囲（細い白の枠）を描いて `_regions.png` にし、
  警告の文を `<名前>_spec_skipped.txt` に足していく）
- `--vs-spec 比べる表.motion.json`（`--spec` と一緒に）: 比べる表でも作り、2 つを同じ時計で横に並べた
  `<名前>_<比べる表の name>_vs_<表の name>.gif`（左が比べる表）と、線を引いた並びを上下に重ねた `_vs_..._line.png` も作る
- `--rows` を付けると `<名前>_rows.png`: スプライトを `--zoom` 倍にし、左に「下から何ドット目か」の番号を 5 行ごとに書いた目安の画像
  （`--split` に指定する数を選ぶため。足元の動かさない行は左の目盛りを灰色にする）

動きを付けられなかった画像は、理由を表示してシートと GIF を作らない。確認に使う AI 画像（ai03〜ai05）は
preview/ai-check ブランチの ai-input/ にある（main には入れない）:

    git fetch origin preview/ai-check
    mkdir -p out/ai && git show origin/preview/ai-check:ai-input/ai05.png > out/ai/ai05.png
    python tools/compare_sprite_motion.py out/ai/ai05.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrocore.sprite import GridNotFoundError, SpriteSettings, convert_sprite, load_rgba, save_rgba  # noqa: E402
from retrocore.sprite.motion import MOTIONS, load_spec, make_motion, make_motion_from_spec, save_sheet  # noqa: E402
from retrocore.sprite.motion import bouncing as bounce_mod  # noqa: E402
from retrocore.sprite.motion import breathe as breathe_mod  # noqa: E402
from retrocore.sprite.motion import floating as float_mod  # noqa: E402
from retrocore.sprite.motion import nod as nod_mod  # noqa: E402
from retrocore.sprite.motion import stretch as stretch_mod  # noqa: E402
from retrocore.sprite.motion import sway as sway_mod  # noqa: E402

DEFAULT_OUT = ROOT / "out" / "sprite" / "motion"
BG = (170, 190, 210)        # 動きを見るときの背景（どのキャラクターの色とも離れた、明るい灰青）
SHEET_BG = (40, 40, 48)
FG = (230, 230, 230)
MARK = (255, 0, 255)
SEAM = (0, 200, 255)
GAP = 16
LABEL_H = 16
FOOT = (255, 220, 0)        # 浮遊: 1 枚目の足元の行（ゲームの pivot の高さ）


def on_bg(rgba: np.ndarray, bg=BG) -> np.ndarray:
    """透明の部分を無地の背景色にした RGB（スプライトは 0 か 255 の透明なので、混ぜずに置き換える）。"""
    out = np.empty(rgba.shape[:2] + (3,), np.uint8)
    out[:] = bg
    opaque = rgba[..., 3] > 0
    out[opaque] = rgba[..., :3][opaque]
    return out


def zoom(rgb: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


def save_gif(frames: list[np.ndarray], durations: list[int], path: Path, k: int) -> None:
    """拡大した GIF（ぼかさない、繰り返し再生）。色は元のスプライトの色だけなので、パレットはそのまま作る。"""
    save_rgb_gif([zoom(on_bg(f), k) for f in frames], durations, path)


def save_rgb_gif(rgbs: list[np.ndarray], durations: list[int], path: Path) -> None:
    """RGB の並びをそのままの色で GIF にする（色は 256 まで。ぼかさない）。"""
    cols = np.unique(np.concatenate([r.reshape(-1, 3) for r in rgbs]), axis=0)
    if len(cols) > 256:
        raise ValueError("GIF にするには色が多すぎます")
    pal = cols.flatten().tolist() + [0] * (768 - cols.size)
    ims = []
    for r in rgbs:
        key = (r[..., 0].astype(np.int64) << 16) | (r[..., 1].astype(np.int64) << 8) | r[..., 2]
        ckey = (cols[:, 0].astype(np.int64) << 16) | (cols[:, 1].astype(np.int64) << 8) | cols[:, 2]
        idx = np.searchsorted(ckey, key).astype(np.uint8)
        im = Image.fromarray(idx, "P")
        im.putpalette(pal)
        ims.append(im)
    path.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=durations, loop=0, disposal=1, optimize=False)


def side_by_side_gif(items: list, path: Path, k: int, foot: bool = True) -> None:
    """いくつかの動きを、同じ時計で横に並べた GIF（それぞれ自分の表示時間で繰り返す）。上に名前、足元（pivot）に黄色の線。

    `items` は (名前, MotionResult) の並び。全体の長さは各動きの 1 回の長さの最小公倍数（どれも切れ目なく繰り返す）。
    下の端をそろえて置くので、地面（pivot）の高さが同じ線になる。
    """
    totals = [int(sum(r.durations)) for _, r in items]
    length = int(np.lcm.reduce(np.array(totals, dtype=np.int64)))
    if length > 60000:
        raise ValueError(f"横に並べる GIF が長すぎます（{length} ミリ秒）")
    cuts = set()
    for (_, r), t in zip(items, totals):
        starts = np.concatenate([[0], np.cumsum(r.durations)[:-1]]).astype(int)
        for rep_ in range(length // t):
            cuts.update(int(rep_ * t + s) for s in starts)
    cuts = sorted(cuts) + [length]
    hs = [r.frames[0].shape[0] for _, r in items]
    H = max(hs) * k
    widths = [r.frames[0].shape[1] * k for _, r in items]
    W = sum(widths) + GAP * (len(items) + 1)
    out, durs = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        im = Image.new("RGB", (W, H + LABEL_H + 2 * GAP), BG)
        d = ImageDraw.Draw(im)
        d.fontmode = "1"     # 文字をぼかさない（GIF の色を増やさない）
        x = GAP
        for (label, r), t, w in zip(items, totals, widths):
            pos = a % t
            i = int(np.searchsorted(np.cumsum(r.durations), pos, side="right"))
            f = r.frames[i]
            fh = f.shape[0] * k
            y0 = LABEL_H + GAP + H - fh
            im.paste(Image.fromarray(zoom(on_bg(f), k)), (x, y0))
            if foot:
                bottom = r.info.get("padding_y", [0, 0])[1]
                fy = y0 + fh - bottom * k
                d.line([x, fy, x + w - 1, fy], fill=FOOT)
            d.text((x, 4), label, fill=(20, 20, 30))
            x += w + GAP
        out.append(np.asarray(im))
        durs.append(b - a)
    save_rgb_gif(out, durs, path)


def frames_sheet(res, k: int) -> Image.Image:
    """全フレームを横に並べて拡大し、前のフレームから変わったドットと継ぎ目に印を付けた画像。"""
    frames = res.frames
    h, w = frames[0].shape[:2]
    fw, fh = w * k, h * k
    n = len(frames)
    floating = res.name in (float_mod.NAME, bounce_mod.NAME)   # 体全体が動く（浮遊・弾む）
    left = RULE_W if floating else 0   # 浮遊・弾む: 左に行の番号
    sheet = Image.new("RGB", (left + n * fw + (n + 1) * GAP, fh + 2 * LABEL_H + 2 * GAP), SHEET_BG)
    d = ImageDraw.Draw(sheet)
    if floating:
        top, bottom = res.info["padding_y"]
        y0 = GAP + LABEL_H
        for y in range(h):
            r = h - bottom - y                                      # 元の絵の下から何ドット目（余白は 0 以下）
            cy = y0 + y * k + k // 2
            col = FG if 1 <= r <= h - top - bottom else (110, 110, 120)
            major = r % 5 == 0 or r == 1
            d.line([left + GAP - (10 if major else 4), cy, left + GAP - 1, cy], fill=col)
            if major and r >= 1:
                d.text((2, cy - 6), f"{r:>3}", fill=col)
    offsets = res.info.get("offsets", [0] * n)
    seams = res.info.get("seams") or res.info.get("cuts") or []
    sway = res.name == sway_mod.NAME
    pad = res.info.get("padding", 0)
    table = "poses" in res.info and res.spec is not None   # 動きの表から作った（--spec）
    for i, f in enumerate(frames):
        x0, y0 = left + GAP + i * (fw + GAP), GAP + LABEL_H
        sheet.paste(Image.fromarray(zoom(on_bg(f), k)), (x0, y0))
        if floating:   # 1 枚目の足元の行（pivot の高さ）の下の縁に黄色の線。動いたフレームでは足元がこの線から離れる
            fy = y0 + (h - res.info["padding_y"][1]) * k
            d.line([x0, fy, x0 + fw - 1, fy], fill=FOOT)
        prev = frames[i - 1]
        changed = np.argwhere((f != prev).any(-1))
        for y, x in ([] if floating else changed):   # 浮遊は絵全体が動くので枠は付けない（数だけ書く）
            d.rectangle([x0 + x * k, y0 + y * k, x0 + (x + 1) * k - 1, y0 + (y + 1) * k - 1], outline=MARK)
        if offsets[i] != 0 and seams:
            seam = seams[0]  # 1 本目の継ぎ目（元の絵で抜いた行・揺れ始める行）。左右の端の外側に印
            if sway:   # 横揺れの分け目は余白を足す前の列の並び。余白の列は端の値と同じ
                seam = [seam[0]] * pad + list(seam) + [seam[-1]] * pad
            for x, yy in ((0, seam[0]), (w - 1, seam[-1])):
                sx = x0 - 6 if x == 0 else x0 + fw + 1
                d.rectangle([sx, y0 + yy * k + 1, sx + 4, y0 + (yy + 1) * k - 2], fill=SEAM)
        move = (f"shift {offsets[i]:+d}px" if sway else f"y {offsets[i]:+d}px" if floating
                else f"down {offsets[i]}px")
        if table:   # 動きの表（--spec）: その行の列の値（行は 1 から。表と同じ数え方）
            vals = {k: v for k, v in res.spec["frames"][i].items() if k != "ms"}
            move = f"row {i + 1}: " + (" ".join(f"{k} {v:+d}" for k, v in vals.items()) or "0")
        d.text((x0, GAP), f"#{i}  {move}  {res.durations[i]}ms  changed {len(changed)}", fill=FG)
    fb = res.info.get("seam_from_bottom") or res.info.get("split_from_bottom")
    if fb:   # 継ぎ目の高さ（下から何ドット目）を右下に
        d.text((GAP, fh + LABEL_H + GAP + 2), f"split {res.info.get('split', 'auto')}: {'cut' if sway else 'seam'} at "
               + ", ".join(f"{a}-{b}" for a, b in fb) + " from bottom"
               + (f"  (padding {pad} each side)" if pad else ""), fill=FG)
    if floating:
        top, bottom = res.info["padding_y"]
        d.text((left + GAP, fh + LABEL_H + GAP + 2), f"{res.name}: whole sprite moves, offset_y {offsets} (- = up)  "
               f"padding top {top} / bottom {bottom}  yellow = foot of frame #0 (pivot)", fill=FG)
    return sheet


RULE_W = 44                 # 目安の画像の左の目盛りの幅
GUIDE = (255, 255, 255)


def rows_image(rgba: np.ndarray, k: int) -> Image.Image:
    """スプライトを k 倍にし、左に「下から何ドット目か」（1 = いちばん下の行）の番号を 5 行ごとに書いた目安の画像。

    1 行ごとに短い目盛り、5 行ごとに長い目盛りと番号と、絵の上に点線を引く。足元の動かさない行（`--split` に指定できない）は
    目盛りを灰色にする。
    """
    h, w = rgba.shape[:2]
    lim = breathe_mod.split_limits(rgba)
    im = Image.new("RGB", (RULE_W + w * k + GAP, h * k + 2 * GAP), SHEET_BG)
    im.paste(Image.fromarray(zoom(on_bg(rgba), k)), (RULE_W, GAP))
    d = ImageDraw.Draw(im)
    for y in range(h):
        n = h - y                                                   # 下から何ドット目
        cy = GAP + y * k + k // 2
        ok = lim is not None and lim[0] <= n <= lim[1]
        col = FG if ok else (110, 110, 120)
        major = n % 5 == 0 or n == 1
        d.line([RULE_W - (10 if major else 4), cy, RULE_W - 1, cy], fill=col)
        if major:
            d.text((2, cy - 6), f"{n:>3}", fill=col)
            if n % 5 == 0:
                for x in range(RULE_W, RULE_W + w * k, 4):
                    d.point((x, cy), fill=GUIDE)
    return im


def _split_arg(text: str):
    return text if text == breathe_mod.AUTO or text.endswith("%") else int(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sprite App の動きの確認用の画像（GIF・フレームの並び・シート）を作る")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--motion", choices=list(MOTIONS), default=breathe_mod.NAME)
    p.add_argument("--frames", type=int, default=None, help="フレーム数（既定は動きの既定。呼吸 2、横揺れ 4、浮遊 4）")
    p.add_argument("--depth", type=int, action="append", default=None,
                   help="呼吸で下がる量・うなずく深さ（うなずくは何度でも指定でき、深さごとに作る）")
    p.add_argument("--amount", type=int, default=sway_mod.DEFAULT_AMOUNT, help="横揺れ・浮遊の揺れ幅")
    p.add_argument("--duration", type=int, default=None, help="表示時間（既定は動きの既定。呼吸 500、横揺れ 300、浮遊 400）")
    p.add_argument("--zoom", type=int, default=8)
    p.add_argument("--converted", action="store_true", help="入力を変換済みのスプライト（1 ドット＝1 画素）として使う")
    p.add_argument("--split", type=_split_arg, action="append", default=None, metavar="auto|N",
                   help="呼吸の分ける高さ（下から何ドット目。auto は自動）。何度でも指定でき、高さごとに作る")
    p.add_argument("--rows", action="store_true", help="「下から何ドット目か」の目安の画像（<名前>_rows.png）も作る")
    p.add_argument("--height", type=int, action="append", default=None, metavar="H",
                   help="弾む高さ（1〜3。何度でも指定でき、高さごとに作る。既定 2）")
    p.add_argument("--compare-durations", action="store_true",
                   help="弾む: 全部 150 ミリ秒と着地だけ長めの両方を作り、横に並べた GIF も作る")
    p.add_argument("--vs-float", action="store_true", help="弾む: 浮遊（既定）と横に並べた GIF も作る")
    p.add_argument("--keep", choices=["off", "on", "both"], default="off",
                   help="呼吸・うなずく・伸び: 小さな部品を守る（--keep-small-parts）。both は切と入を並べた GIF と線の画像も作る")
    p.add_argument("--spec", type=Path, default=None, metavar="FILE",
                   help="動きの表（.motion.json。CLI の --motion-spec と同じ）で作る。<名前>_<表の name> の GIF・フレームの並び・シート")
    p.add_argument("--vs-spec", type=Path, default=None, metavar="FILE",
                   help="--spec と一緒に: 比べる表でも作り、横に並べた GIF（左が比べる表）と上下に重ねた線の画像を作る")
    args = p.parse_args(argv)
    failed = 0
    for src in args.inputs:
        rgba = load_rgba(src)
        if not args.converted:
            try:
                rgba = convert_sprite(rgba, SpriteSettings()).image
            except GridNotFoundError as e:
                print(f"{src.name}: 変換できません（{e}）", file=sys.stderr)
                failed += 1
                continue
        if args.rows:
            args.output.mkdir(parents=True, exist_ok=True)
            rows_image(rgba, args.zoom).save(args.output / f"{src.stem}_rows.png")
        save_rgba(rgba, args.output / f"{src.stem}_sprite.png")
        if args.spec is not None:
            res = make_motion_from_spec(rgba, load_spec(args.spec))
            name = f"{src.stem}_{res.name}"
            if res.skipped:
                print(f"{name}: {' / '.join(res.warnings)}")
                if _has_box(res):
                    args.output.mkdir(parents=True, exist_ok=True)
                    skipped_regions(rgba, res, args.zoom).save(args.output / f"{name}_regions.png")
                    with open(args.output / f"{src.stem}_spec_skipped.txt", "a", encoding="utf-8") as fh:
                        fh.write(f"{name}（{args.spec.name}）:\n" + "\n".join(res.warnings) + "\n\n")
                failed += 1
                continue
            _save_all_table(res, args.output, name, args.zoom)
            line = line_sheet(res, args.zoom, table_lines(res), f"table {res.name}")
            line.save(args.output / f"{name}_line.png")
            if _has_box(res):
                region_sheet(res, args.zoom).save(args.output / f"{name}_regions.png")
            if args.vs_spec is not None:
                other = make_motion_from_spec(rgba, load_spec(args.vs_spec))
                oname = f"{src.stem}_{other.name}"
                if other.skipped:
                    print(f"{oname}: {' / '.join(other.warnings)}")
                    failed += 1
                    continue
                _save_all_table(other, args.output, oname, args.zoom)
                oline = line_sheet(other, args.zoom, table_lines(other), f"table {other.name}")
                oline.save(args.output / f"{oname}_line.png")
                vs = f"{oname}_vs_{res.name}"
                side_by_side_gif([(other.name, other), (res.name, res)], args.output / f"{vs}.gif", args.zoom)
                stack = Image.new("RGB", (max(oline.width, line.width), oline.height + line.height), SHEET_BG)
                stack.paste(oline, (0, 0))
                stack.paste(line, (0, oline.height))
                stack.save(args.output / f"{vs}_line.png")
            continue
        floating = args.motion in (float_mod.NAME, bounce_mod.NAME)
        if floating and args.split:
            print("浮遊・弾むには分ける高さ（--split）がありません", file=sys.stderr)
            return 2
        if args.motion == bounce_mod.NAME:
            failed += bounce_images(args, src, rgba)
            continue
        if args.motion in (stretch_mod.NAME, nod_mod.NAME) or (args.motion == breathe_mod.NAME and args.keep != "off"):
            failed += deform_images(args, src, rgba)
            continue
        for split in args.split or [None]:
            options = {} if split is None else {"split": split}
            try:
                extra = ({"amount": args.amount} if args.motion in (sway_mod.NAME, float_mod.NAME)
                         else {"depth": (args.depth or [breathe_mod.DEFAULT_DEPTH])[0]})
                res = make_motion(rgba, args.motion, frames=args.frames, durations=args.duration, **extra, **options)
            except ValueError as e:
                print(f"{src.name}: {e}", file=sys.stderr)
                failed += 1
                continue
            name = f"{src.stem}_{args.motion}" + ("" if split is None else f"_split{'-' if split == 'auto' else ''}{split}")
            if args.motion == sway_mod.NAME and args.amount != sway_mod.DEFAULT_AMOUNT:
                name += f"_amount{args.amount}"
            if floating:   # 浮遊は揺れ幅とフレーム数をいつも名前に入れる（1 と 2 を並べて見比べるため）
                name += f"_amount{args.amount}_f{len(res.frames)}"
            if res.skipped:
                print(f"{name}: {' / '.join(res.warnings)}")
                continue
            save_gif(res.frames, res.durations, args.output / f"{name}.gif", args.zoom)
            frames_sheet(res, args.zoom).save(args.output / f"{name}_frames.png")
            save_sheet(res, args.output / f"{name}_sheet.png")
            seams = res.info.get("seams") or res.info.get("cuts") or []
            rows = f"継ぎ目の行 {min(seams[0])}〜{max(seams[0])}（高さ {rgba.shape[0]}）" if seams else ""
            fb = res.info.get("seam_from_bottom") or res.info.get("split_from_bottom")
            fb = f"、下から {'・'.join(f'{a}〜{b}' for a, b in fb)} ドット目" if fb else ""
            changed = [int((f != res.frames[i - 1]).any(-1).sum()) for i, f in enumerate(res.frames)]
            what = "揺れる量" if args.motion == sway_mod.NAME else "上下のずれ" if floating else "下がる量"
            pad = f"、左右の余白 {res.info['padding']}" if res.info.get("padding") else ""
            if floating:
                pad = f"、上下の余白 {res.info['padding_y']}"
            w, h = res.size
            cost = f"、目立ち方 {res.info['costs']}" if res.info.get("costs") is not None else ""
            print(f"{name}: {w}×{h}、{len(res.frames)} フレーム、{what} {res.info['offsets']}"
                  f"{'、' + rows if rows else ''}{fb}{cost}{pad}、変わったドット {changed}")
    return 1 if failed else 0


def _save_all_table(res, out: Path, name: str, k: int) -> None:
    save_gif(res.frames, res.durations, out / f"{name}.gif", k)
    frames_sheet(res, k).save(out / f"{name}_frames.png")
    save_sheet(res, out / f"{name}_sheet.png")
    w, h = res.size
    print(f"{name}: {w}×{h}、{len(res.frames)} フレーム（表 {len(res.spec['frames'])} 行）、表示時間 {res.durations}、"
          f"作ったポーズ {res.info.get('poses')}、範囲 {res.info.get('regions')}")


REGION_COLORS = [(255, 70, 70), (60, 230, 90), (255, 170, 0), (0, 200, 255)]


def _has_box(res) -> bool:
    return any(r.get("type") in ("band", "rect") for r in ((res.spec or {}).get("regions") or {}).values())


def _box_px(edges: dict, w0: int, h0: int) -> tuple[int, int, int, int]:
    """範囲の端（下から・左から何ドット目）を、元の絵の上の (左の列, 上の行, 右の列, 下の行)（0 から）にする。"""
    left, right = edges.get("left", 1) - 1, edges.get("right", w0) - 1
    return left, h0 - edges["top"], right, h0 - edges["bottom"]


def region_sheet(res, k: int) -> Image.Image:
    """全フレームを並べ、帯・四角の範囲を色付きの枠で描いた画像（枠は元の絵の上の位置。余白・体全体のずれに合わせてずらす）。"""
    frames = res.frames
    h, w = frames[0].shape[:2]
    fw, fh = w * k, h * k
    n = len(frames)
    pad = res.info.get("padding", 0)
    top, bottom = res.info.get("padding_y", [0, 0])
    offs = res.info.get("offset_y") or [0] * n
    h0, w0 = h - top - bottom, w - 2 * pad
    boxes = [(name, r["box"]) for name, r in res.info.get("regions", {}).items() if "box" in r]
    sheet = Image.new("RGB", (n * fw + (n + 1) * GAP, fh + (2 + len(boxes)) * LABEL_H + 2 * GAP), SHEET_BG)
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        x0, y0 = GAP + i * (fw + GAP), GAP + LABEL_H
        sheet.paste(Image.fromarray(zoom(on_bg(f), k)), (x0, y0))
        for j, (_, e) in enumerate(boxes):
            c0, r0, c1, r1 = _box_px(e, w0, h0)
            col = REGION_COLORS[j % len(REGION_COLORS)]
            d.rectangle([x0 + (pad + c0) * k, y0 + (top + r0 + offs[i]) * k,
                         x0 + (pad + c1 + 1) * k - 1, y0 + (top + r1 + 1 + offs[i]) * k - 1], outline=col, width=2)
        vals = {kk: v for kk, v in res.spec["frames"][i].items() if kk != "ms"}
        d.text((x0, GAP), f"row {i + 1}: " + (" ".join(f"{kk} {v:+d}" for kk, v in vals.items()) or "0"), fill=FG)
    for j, (name, e) in enumerate(boxes):
        kind = "rect" if "left" in e else "band"
        where = f"bottom {e['bottom']} - top {e['top']}" + (f", left {e['left']} - right {e['right']}" if kind == "rect" else "")
        d.text((GAP, fh + LABEL_H + GAP + 2 + j * LABEL_H), f"{kind} {name}: {where}  (only inside moves)",
               fill=REGION_COLORS[j % len(REGION_COLORS)])
    return sheet


def skipped_regions(rgba: np.ndarray, res, k: int) -> Image.Image:
    """作れなかった表: 元の絵に、範囲の枠（色）・問題のドット（マゼンタ）・すすめられた範囲（細い白の枠）を描いた画像。"""
    from retrocore.sprite.motion.region import edge_number
    h0, w0 = rgba.shape[:2]
    regs = [(name, r) for name, r in (res.spec.get("regions") or {}).items() if r.get("type") in ("band", "rect")]
    k2 = k * 2
    im = Image.new("RGB", (w0 * k2 + 2 * GAP, h0 * k2 + (3 + len(regs)) * LABEL_H + 2 * GAP), SHEET_BG)
    im.paste(Image.fromarray(zoom(on_bg(rgba), k2)), (GAP, GAP + LABEL_H))
    d = ImageDraw.Draw(im)
    x0, y0 = GAP, GAP + LABEL_H
    problems = res.info.get("problems", [])
    for j, (name, r) in enumerate(regs):
        e = {kk: edge_number(rgba, kk, r[kk]) for kk in ("left", "right", "bottom", "top") if kk in r}
        if None in e.values():
            continue
        col = REGION_COLORS[j % len(REGION_COLORS)]
        c0, r0, c1, r1 = _box_px(e, w0, h0)
        d.rectangle([x0 + c0 * k2, y0 + r0 * k2, x0 + (c1 + 1) * k2 - 1, y0 + (r1 + 1) * k2 - 1], outline=col, width=3)
        for q in problems:
            if q.get("region") != name:
                continue
            for nb in q.get("near") or []:
                if isinstance(nb, dict):
                    c0, r0, c1, r1 = _box_px(nb, w0, h0)
                    d.rectangle([x0 + c0 * k2 + 3, y0 + r0 * k2 + 3, x0 + (c1 + 1) * k2 - 4, y0 + (r1 + 1) * k2 - 4],
                                outline=(240, 240, 240), width=1)
        d.text((GAP, h0 * k2 + LABEL_H + GAP + 2 + j * LABEL_H), f"{r['type']} {name}: "
               + ", ".join(f"{kk} {v}" for kk, v in e.items()) + "  (white = suggested)", fill=col)
    for q in problems:   # 帯・四角の問題の場所は元の絵の上の位置
        for x, y in (q.get("pixels") or []) if q.get("region") in dict(regs) else []:
            d.rectangle([x0 + x * k2, y0 + y * k2, x0 + (x + 1) * k2 - 1, y0 + (y + 1) * k2 - 1], outline=MARK, width=2)
    d.text((GAP, 4), "could not make: " + "; ".join(q["rule"] for q in problems), fill=FG)
    return im


def _report(name: str, res) -> None:
    w, h = res.size
    print(f"{name}: {w}×{h}、{len(res.frames)} フレーム、上下のずれ {res.info['offsets']}、"
          f"表示時間 {res.durations}（1 回 {sum(res.durations)} ミリ秒）、上下の余白 {res.info['padding_y']}")


def _save_all(res, out: Path, name: str, k: int) -> None:
    save_gif(res.frames, res.durations, out / f"{name}.gif", k)
    frames_sheet(res, k).save(out / f"{name}_frames.png")
    save_sheet(res, out / f"{name}_sheet.png")
    _report(name, res)


def bounce_images(args, src: Path, rgba: np.ndarray) -> int:
    """弾む: 高さごとに GIF・フレームの並び・シート。表示時間の比べと、浮遊との比べの GIF も作る。"""
    failed = 0
    for height in args.height or [bounce_mod.DEFAULT_HEIGHT]:
        variants = []
        if args.compare_durations:
            variants = [("uniform150", bounce_mod.AIR_DURATION), (f"landing{bounce_mod.LANDING_DURATION}", None)]
        elif args.duration is not None:
            variants = [(f"uniform{args.duration}", args.duration)]
        else:
            variants = [(f"landing{bounce_mod.LANDING_DURATION}", None)]
        made = []
        for tag, dur in variants:
            try:
                res = make_motion(rgba, bounce_mod.NAME, durations=dur, height=height)
            except ValueError as e:
                print(f"{src.name}: {e}", file=sys.stderr)
                failed += 1
                continue
            name = f"{src.stem}_bounce_h{height}_{tag}"
            if res.skipped:
                print(f"{name}: {' / '.join(res.warnings)}")
                continue
            _save_all(res, args.output, name, args.zoom)
            made.append((tag, res))
        if len(made) == 2:
            path = args.output / f"{src.stem}_bounce_h{height}_{made[0][0]}_vs_{made[1][0]}.gif"
            side_by_side_gif([(f"h{height} {t}", r) for t, r in made], path, args.zoom)
            print(f"{path.name}: 横に並べた GIF（同じ時計）")
        if args.vs_float and made:
            fl = make_motion(rgba, float_mod.NAME)
            bo = next(r for t, r in made if t.startswith("landing")) if any(
                t.startswith("landing") for t, _ in made) else made[0][1]
            path = args.output / f"{src.stem}_float_vs_bounce_h{height}.gif"
            side_by_side_gif([("float 400ms", fl), (f"bounce h{height}", bo)], path, args.zoom)
            print(f"{path.name}: 浮遊と弾むを横に並べた GIF（同じ時計）")
    return failed


NECK = (255, 60, 60)        # うなずく: 自動で見つけた首の行の印
SEAM2 = (255, 220, 0)       # 2 本目の継ぎ目


def nod_line_sheet(res, k: int) -> Image.Image:
    """うなずく: 全フレームを並べ、左に行の番号（下から何ドット目）、継ぎ目の下の縁に線（ここより下は動かない）を引いた画像。"""
    frames = res.frames
    h, w = frames[0].shape[:2]
    fw, fh = w * k, h * k
    n = len(frames)
    sheet = Image.new("RGB", (RULE_W + n * fw + (n + 1) * GAP, fh + 3 * LABEL_H + 2 * GAP), SHEET_BG)
    d = ImageDraw.Draw(sheet)
    y0 = GAP + LABEL_H
    neck = (res.info.get("neck") or {}).get("from_bottom")
    for y in range(h):
        r = h - y
        cy = y0 + y * k + k // 2
        major = r % 5 == 0 or r == 1
        d.line([RULE_W + GAP - (10 if major else 4), cy, RULE_W + GAP - 1, cy], fill=FG)
        if major:
            d.text((2, cy - 6), f"{r:>3}", fill=FG)
        if neck is not None and r == neck:
            d.rectangle([RULE_W + GAP - 14, cy - 2, RULE_W + GAP - 11, cy + 2], fill=NECK)
    seams = res.info.get("seams") or []
    for i, f in enumerate(frames):
        x0 = RULE_W + GAP + i * (fw + GAP)
        sheet.paste(Image.fromarray(zoom(on_bg(f), k)), (x0, y0))
        for j, seam in enumerate(seams):
            col = SEAM if j == 0 else SEAM2
            pts = []
            for x, yy in enumerate(seam):   # 継ぎ目のドットの下の縁（元の絵の行）。ここより下は、どのフレームでも動かない
                yb = y0 + (yy + 1) * k - 1
                pts += [(x0 + x * k, yb), (x0 + (x + 1) * k - 1, yb)]
            d.line(pts, fill=col, width=2)
        d.text((x0, GAP), f"#{i}  down {res.info['offsets'][i]}px  {res.durations[i]}ms", fill=FG)
    at = res.spec["regions"]["upper"]["at"] if res.spec else "?"
    fb = ", ".join(f"{a}-{b}" for a, b in res.info.get("seam_from_bottom") or [])
    text = (f"nod depth {res.info['depth']}: split {res.info['split']} -> at {at}"
            + (f" (neck found at {neck}, red mark)" if neck is not None else "")
            + f"  seam {fb} from bottom. below the cyan line nothing moves" + ("; yellow = 2nd seam" if len(seams) > 1 else ""))
    d.text((GAP, fh + LABEL_H + GAP + 2), text, fill=FG)
    return sheet


PLUS = (80, 220, 120)       # 表の伸ばす（+）の 1 本目の線
PLUS2 = (255, 150, 40)      # 表の伸ばす（+）の 2 本目の線


def line_sheet(res, k: int, lines: list, title: str) -> Image.Image:
    """全フレームを並べ、左に行の番号（元の絵の下から何ドット目）、分け目の線（継ぎ目のドットの下の縁）を引いた画像。

    `lines` は [(色, 列ごとの「下から何ドット目」), ...]。上に足した余白・体全体のずれ（offset_y）に合わせて線をずらす。
    線より下は、そのフレームでは（体全体のずれを除いて）動かない。
    """
    frames = res.frames
    h, w = frames[0].shape[:2]
    fw, fh = w * k, h * k
    n = len(frames)
    top, bottom = res.info.get("padding_y", [0, 0])
    offs = res.info.get("offset_y") or [0] * n
    sheet = Image.new("RGB", (RULE_W + n * fw + (n + 1) * GAP, fh + 3 * LABEL_H + 2 * GAP), SHEET_BG)
    d = ImageDraw.Draw(sheet)
    y0 = GAP + LABEL_H
    neck = (res.info.get("neck") or {}).get("from_bottom")
    for y in range(h):
        r = h - bottom - y                       # 元の絵の下から何ドット目（上に足した余白は元の高さより上）
        cy = y0 + y * k + k // 2
        col = FG if 1 <= r <= h - top - bottom else (110, 110, 120)
        major = r % 5 == 0 or r == 1
        d.line([RULE_W + GAP - (10 if major else 4), cy, RULE_W + GAP - 1, cy], fill=col)
        if major and r >= 1:
            d.text((2, cy - 6), f"{r:>3}", fill=col)
        if neck is not None and r == neck:
            d.rectangle([RULE_W + GAP - 14, cy - 2, RULE_W + GAP - 11, cy + 2], fill=NECK)
    for i, f in enumerate(frames):
        x0 = RULE_W + GAP + i * (fw + GAP)
        sheet.paste(Image.fromarray(zoom(on_bg(f), k)), (x0, y0))
        for col, line in lines:
            pts = []
            for x, v in enumerate(line):
                yy = h - bottom - v + offs[i]            # その行のフレームの中の行
                yb = y0 + (yy + 1) * k - 1
                pts += [(x0 + x * k, yb), (x0 + (x + 1) * k - 1, yb)]
            d.line(pts, fill=col, width=2)
        if res.spec is not None and len(res.spec.get("frames", [])) == n:
            vals = {kk: v for kk, v in res.spec["frames"][i].items() if kk != "ms"}
            move = " ".join(f"{kk} {v:+d}" for kk, v in vals.items()) or "0"
        else:
            move = ""
        d.text((x0, GAP), f"#{i}  {move}  {res.durations[i]}ms", fill=FG)
    d.text((GAP, fh + LABEL_H + GAP + 2), title + "  (below each line nothing moves in the pose)", fill=FG)
    return sheet


def preset_lines(res) -> list:
    """プリセット（呼吸・うなずく・伸び）の継ぎ目を「下から何ドット目」の線にする（1 本目は水色、2 本目は黄色）。"""
    seams = res.info.get("seams") or []
    h = res.frames[0].shape[0] - (res.info.get("padding_y") or [0, 0])[0] if res.name != stretch_mod.NAME else res.frames[0].shape[0]
    return [((SEAM if j == 0 else SEAM2), [h - v for v in seam]) for j, seam in enumerate(seams)]


def table_lines(res) -> list:
    """表で実際に使った線（"used" の lines。縮める = 水色・黄色、伸ばす（列の名前+）= 緑・橙）。"""
    out = []
    for r in (res.spec or {}).get("regions", {}).values():
        for tname, ls in ((r.get("used") or {}).get("lines") or {}).items():
            plus = tname.endswith("+")
            for j, line in enumerate(ls):
                out.append(((PLUS if j == 0 else PLUS2) if plus else (SEAM if j == 0 else SEAM2), line))
    return out


def deform_images(args, src: Path, rgba: np.ndarray) -> int:
    """呼吸（--keep のとき）・うなずく・伸び: 分ける高さ（--split）と量（--depth）ごとに GIF・フレームの並び・線を引いた並び・シート。

    `--keep both` なら、小さな部品を守るの切と入を横に並べた GIF と、線の画像を上下に重ねたものも作る。
    """
    failed = 0
    motion = args.motion
    log = args.output / f"{src.stem}_{motion}_skipped.txt"
    keeps = {"off": [False], "on": [True], "both": [False, True]}[args.keep]
    for depth in args.depth or [1]:
        for split in args.split or [None]:
            tag = "auto" if split in (None, "auto") else str(split)
            q = "a" if motion == stretch_mod.NAME else "d"
            base_name = f"{src.stem}_{motion}_{q}{depth}_split{'-' if tag == 'auto' else ''}{tag}"
            made = {}
            for keep in keeps:
                key = "depth" if motion != stretch_mod.NAME else "amount"
                options = {key: depth, **({} if split is None else {"split": split})}
                if keep:
                    options["keep_small_parts"] = True
                if motion == breathe_mod.NAME:
                    options["frames"] = args.frames or (2 if depth == 1 else 4)
                try:
                    res = make_motion(rgba, motion, durations=args.duration, **options)
                except ValueError as e:
                    print(f"{src.name}: {e}", file=sys.stderr)
                    failed += 1
                    continue
                name = base_name + ("_keep" if keep else "")
                if res.skipped:
                    print(f"{name}: {' / '.join(res.warnings)}")
                    args.output.mkdir(parents=True, exist_ok=True)
                    with log.open("a", encoding="utf-8") as fh:
                        fh.write(f"{name}: {' / '.join(res.warnings)}\n")
                    continue
                save_gif(res.frames, res.durations, args.output / f"{name}.gif", args.zoom)
                frames_sheet(res, args.zoom).save(args.output / f"{name}_frames.png")
                at = res.spec["regions"]["upper"]["at"] if res.spec else "?"
                fb = ", ".join(f"{a}-{b}" for a, b in res.info.get("seam_from_bottom") or [])
                title = (f"{motion} {q}{depth} keep_small_parts {'on' if keep else 'off'}: split {res.info['split']} -> at {at}"
                         f"  seam {fb} from bottom")
                if motion == nod_mod.NAME:
                    nod_line_sheet(res, args.zoom).save(args.output / f"{name}_line.png")
                    line_img = line_sheet(res, args.zoom, preset_lines(res), title)
                else:
                    line_img = line_sheet(res, args.zoom, preset_lines(res), title)
                    line_img.save(args.output / f"{name}_line.png")
                save_sheet(res, args.output / f"{name}_sheet.png")
                made[keep] = (res, line_img)
                neck = res.info.get("neck")
                w, h = res.size
                print(f"{name}: {w}×{h}、{len(res.frames)} フレーム、量 {res.info['offsets']}、表示時間 {res.durations}、"
                      + (f"首（自動）下から {neck['from_bottom']}、" if neck else "")
                      + f"分けた高さ {at}、継ぎ目 下から {res.info['seam_from_bottom']}、目立ち方 {res.info['costs']}"
                      + (f"、上の余白 {res.info['padding_y'][0]}" if res.info.get("padding_y") else ""))
            if len(made) == 2:
                (r0, l0), (r1, l1) = made[False], made[True]
                path = args.output / f"{base_name}_keep_off_vs_on.gif"
                side_by_side_gif([("keep off", r0), ("keep on", r1)], path, args.zoom)
                both = Image.new("RGB", (max(l0.width, l1.width), l0.height + l1.height + GAP), SHEET_BG)
                both.paste(l0, (0, 0))
                both.paste(l1, (0, l0.height + GAP))
                both.save(args.output / f"{base_name}_keep_off_vs_on_line.png")
                print(f"{path.name}: 小さな部品を守る 切（左）と入（右）を横に並べた GIF（同じ時計）")
    return failed


if __name__ == "__main__":
    sys.exit(main())
