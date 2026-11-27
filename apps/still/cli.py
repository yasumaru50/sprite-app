"""Still App の CLI。

    python -m apps.still 入力.png --mode pc98 --out out/still/部屋        # → 部屋.png（インデックスカラー）と 部屋.json
    python -m apps.still 入力.png --width 160                             # グリッド検出の代わりに横 160 マスに縮める
    python -m apps.still 入力.png --colors 8 --no-4bit --max-size 320x200 --canvas 320x200

--mode はプリセット（初めの設定の組み合わせ）で、個別のオプションで上書きできる。変換は retrocore.still（ブラウザ版と共通）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrocore.still import convert as convert_mod
from retrocore.still.colors import MAX_COLORS, MIN_COLORS
from retrocore.still.convert import DEFAULT_MODE, MODES, StillError, StillSettings, parse_size


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apps.still",
                                description="AI が描いたドット絵風の 1 枚絵を、決まった色数・格子のドット絵にそろえる")
    p.add_argument("input", type=Path, help="入力の画像（PNG・JPG）")
    p.add_argument("--out", "-o", default=None,
                   help="出力名（拡張子なし。<出力名>.png と <出力名>.json を書く。既定 out/still/<入力の名前>）")
    p.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                   help="初めの設定の組み合わせ（既定 pc98: " + MODES["pc98"]["label"] + "）。下のオプションで個別に変えられる")
    p.add_argument("--colors", type=int, default=None, help=f"色数の上限（{MIN_COLORS}〜{MAX_COLORS}）")
    p.add_argument("--no-4bit", dest="four_bit", action="store_const", const=False, default=None,
                   help="色を 4096 色（RGB 各 4 ビット）に丸めない")
    p.add_argument("--max-size", default=None, help="大きさの上限 幅x高さ（例 640x400）。超えたら出力せずに知らせる")
    p.add_argument("--width", type=int, default=None,
                   help="グリッド検出の代わりに、横 N マスに縮める（縦は縦横の比から）")
    p.add_argument("--canvas", default=None, help="余白を足してこの大きさ（幅x高さ）にする。余白はいちばん暗い色")
    p.add_argument("--no-keep-darkest", dest="keep_darkest", action="store_const", const=False, default=None,
                   help="いちばん暗い色をパレットに取っておかない")
    p.add_argument("-q", "--quiet", action="store_true", help="結果の表示を減らす")
    return p


def settings_from_args(args) -> StillSettings:
    s = StillSettings.from_mode(
        args.mode, colors=args.colors, four_bit=args.four_bit,
        max_size=None if args.max_size is None else parse_size(args.max_size, "大きさの上限"),
        width=args.width, keep_darkest=args.keep_darkest,
        canvas=None if args.canvas is None else parse_size(args.canvas, "キャンバス"))
    s.validate()
    return s


def out_base(args) -> Path:
    if args.out:
        base = Path(args.out)
        return base.with_suffix("") if base.suffix.lower() in (".png", ".json") else base
    return Path("out") / "still" / args.input.stem


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
    except ValueError as e:
        parser.error(str(e))
    if not args.input.is_file():
        print(f"エラー: 入力のファイルがありません: {args.input}", file=sys.stderr)
        return 1
    try:
        out = convert_mod.convert_still_bytes(args.input.read_bytes(), args.input.name, settings)
    except StillError as e:
        print(f"エラー（{args.input.name}）: {e}", file=sys.stderr)
        if e.candidates:
            print("  --width の候補:", file=sys.stderr)
            for c in e.candidates:
                print(f"    --width {c['width']}（{c['width']}×{c['height']} ドット。{c['note']}）", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"エラー（{args.input.name}）: {convert_mod.friendly_error(e)}", file=sys.stderr)
        return 1
    base = out_base(args)
    base.parent.mkdir(parents=True, exist_ok=True)
    png, js = Path(f"{base}.png"), Path(f"{base}.json")
    png.write_bytes(out["png"])
    js.write_bytes(out["json"].encode("utf-8"))
    info = out["info"]
    for w in info["warnings"]:
        print(f"警告（{args.input.name}）: {w}", file=sys.stderr)
    if not args.quiet:
        d = info["dots"]
        how = (f"グリッド検出（1 ドット {d['dot_size']:.1f}px）" if d["method"] == "grid"
               else f"--width {d['width']}（1 マス {d['cell_size'][0]:.1f}×{d['cell_size'][1]:.1f}px）")
        print(f"{args.input.name}: {how} → {info['picture']['width']}×{info['picture']['height']} ドット、"
              f"{info['colors']['used']} 色" + (f"、キャンバス {info['width']}×{info['height']}" if info["canvas"] else ""))
        print(f"  → {png}\n  → {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
