"""Still App の CLI。

    python -m apps.still 入力.png --mode pc98 --out out/still/部屋        # → 部屋.png（インデックスカラー）と 部屋.json
    python -m apps.still 入力.png --width 160                             # グリッド検出の代わりに横 160 マスに縮める
    python -m apps.still 入力.png --colors 8 --no-4bit --max-size 320x200 --canvas 320x200
    python -m apps.still 入力.png --width 600 --tidy-pattern                # ざらついた範囲を 2 色の模様に整える
    python -m apps.still 入力.png --mode modern --tidy-dots                  # 今風（32 色）。孤立したドットをそろえる
    python -m apps.still 写真.jpg --mode photo --smooth strong --outline     # 写真の下書き（横 320）。輪郭の線も引く
    python -m apps.still 写真.jpg --mode photo --width 640 --colors 16 --4bit --dither 2x2   # 空などに 2×2 の網目

--mode は初めの設定（項目の値の組み合わせ）で、どの値も個別のオプション（項目）で変えられる。
    python -m apps.still 入力.png --mode modern --width 600 --smooth weak     # 今風＋色の塊を弱くまとめる
    python -m apps.still 写真.png --mode photo --width 640 --colors 16 --4bit  # 写真の下書き＋16 色＋4096 色に丸め
変換は retrocore.still（ブラウザ版と共通）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrocore.still import convert as convert_mod
from retrocore.still.colors import MAX_COLORS, MIN_COLORS
from retrocore.still.convert import (DEFAULT_MODE, DITHER_CHOICES, MODES, SMOOTH_CHOICES, StillError, StillSettings,
                                     parse_size)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apps.still",
                                description="AI が描いたドット絵風の 1 枚絵を、決まった色数・格子のドット絵にそろえる")
    p.add_argument("input", type=Path, help="入力の画像（PNG・JPEG。HEIC は読めない）")
    p.add_argument("--out", "-o", default=None,
                   help="出力名（拡張子なし。<出力名>.png と <出力名>.json を書く。既定 out/still/<入力の名前>）")
    p.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                   help="初めの設定の組み合わせ（既定 pc98: " + MODES["pc98"]["label"] + "、modern: "
                   + MODES["modern"]["label"] + "、photo: " + MODES["photo"]["label"]
                   + "）。初めの設定（項目の値の組み合わせ）で、下の項目で 1 つずつ変えられる")
    # ここから下は工程の順（向き → マス目 → 色の塊 → 減色 → 似た色 → ディザ → 孤立したドット → 模様 → 輪郭 → 余白）。
    # どれも初めの設定（--mode）の値を 1 つずつ変える
    g = p.add_argument_group("項目（初めの設定の値を 1 つずつ変える。工程の順）")
    g.add_argument("--orient", dest="orient", action="store_const", const=True, default=None,
                   help="写真の向き（EXIF）を直して読む（写真の下書きでは既定で入）")
    g.add_argument("--no-orient", dest="orient", action="store_const", const=False,
                   help="写真の向き（EXIF）を直さない")
    g.add_argument("--grid", dest="find_grid", action="store_const", const=True, default=None,
                   help="ドットのマス目を探す（98・今風の既定。写真の下書きで使うとき。--width とは一緒に使えない）")
    g.add_argument("--width", type=int, default=None,
                   help="マス目を探さずに、横 N マスに分ける（縦は縦横の比から。写真の下書きの既定は 320）")
    g.add_argument("--max-size", default=None, help="大きさの上限 幅x高さ（例 640x400）。超えたら出力せずに知らせる")
    g.add_argument("--smooth", choices=list(SMOOTH_CHOICES), default=None,
                   help="色の塊をまとめる強さ（off しない・weak 弱・medium 中・strong 強。写真の下書きの既定は medium、"
                   "ほかは off）")
    g.add_argument("--colors", type=int, default=None, help=f"色数の上限（{MIN_COLORS}〜{MAX_COLORS}）")
    g.add_argument("--4bit", dest="four_bit", action="store_const", const=True, default=None,
                   help="色を 4096 色（RGB 各 4 ビット）に丸める（98 の既定）")
    g.add_argument("--no-4bit", dest="four_bit", action="store_const", const=False,
                   help="色を 4096 色に丸めない（今風・写真の下書きの既定）")
    g.add_argument("--keep-darkest", dest="keep_darkest", action="store_const", const=True, default=None,
                   help="いちばん暗い色をパレットに取っておく（どの初めの設定でも既定で入）")
    g.add_argument("--no-keep-darkest", dest="keep_darkest", action="store_const", const=False,
                   help="いちばん暗い色をパレットに取っておかない")
    g.add_argument("--merge-similar", dest="merge_similar", action="store_const", const=True, default=None,
                   help="見た目でほぼ同じ色をまとめる（今風の既定。98・写真の下書きでも使える）")
    g.add_argument("--no-merge-similar", dest="merge_similar", action="store_const", const=False,
                   help="見た目でほぼ同じ色をまとめない")
    g.add_argument("--dither", choices=list(DITHER_CHOICES), default=None,
                   help="なめらかに色が変わる所（空など）で、近い 2 色の間を決まった網目でつなぐ（off しない・2x2 5 段・"
                   "4x4 17 段。どの初めの設定でも既定は off）。新しい色は作らない")
    g.add_argument("--tidy-dots", dest="tidy_dots", action="store_const", const=True, default=None,
                   help="孤立したドット（まわり 8 ドットに同じ色が無く、7 つ以上が同じ色）を、まわりの色にそろえる"
                   "（写真の下書きでは既定で入）")
    g.add_argument("--no-tidy-dots", dest="tidy_dots", action="store_const", const=False,
                   help="孤立したドットをそろえない（写真の下書きで切にするとき）")
    g.add_argument("--tidy-pattern", dest="tidy_pattern", action="store_const", const=True, default=None,
                   help="ざらついた範囲（減色でばらばらの点になったディザ風の所）を、2 色の決まった模様（2×2）に整える")
    g.add_argument("--outline", dest="outline", action="store_const", const=True, default=None,
                   help="明るさの差が大きい色の境目に、暗い方の側へ 1 ドットの線を引く（線はパレットのいちばん暗い色）")
    g.add_argument("--canvas", default=None, help="余白を足してこの大きさ（幅x高さ）にする。余白はいちばん暗い色")
    p.add_argument("-q", "--quiet", action="store_true", help="結果の表示を減らす")
    return p


def settings_from_args(args) -> StillSettings:
    if args.find_grid and args.width is not None:
        raise ValueError(f"--grid（マス目を探す）と --width {args.width}（横のマス数で分ける）は一緒に使えません")
    s = StillSettings.from_mode(
        args.mode, orient=args.orient, find_grid=args.find_grid, width=args.width,
        max_size=None if args.max_size is None else parse_size(args.max_size, "大きさの上限"),
        smooth=args.smooth, colors=args.colors, four_bit=args.four_bit, keep_darkest=args.keep_darkest,
        merge_similar=args.merge_similar, dither=args.dither, tidy_dots=args.tidy_dots, tidy_pattern=args.tidy_pattern,
        outline=args.outline, canvas=None if args.canvas is None else parse_size(args.canvas, "キャンバス"))
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
        if info.get("photo", {}).get("smooth"):
            name = "写真の下書き" if info["settings"]["mode"] == "photo" else "色の塊をまとめる"
            how = f"{name}（まとめの強さ {info['photo']['smooth']['level']}）、" + how
        print(f"{args.input.name}: {how} → {info['picture']['width']}×{info['picture']['height']} ドット、"
              f"{info['colors']['used']} 色" + (f"、キャンバス {info['width']}×{info['height']}" if info["canvas"] else ""))
        if "preset" in info:
            print(f"  初めの設定（{info['preset']['name']}）から変えた項目: {', '.join(info['preset']['changed'])}")
        if "similar" in info:
            print(f"  似た色をまとめた組 {info['similar']['merged']}")
        if "dither" in info:
            print(f"  網目を入れたドット {info['dither']['changed']}（網目の範囲 {info['dither']['area']}）")
        if "tidy_dots" in info:
            print(f"  孤立したドットを置き換えた数 {info['tidy_dots']['changed']}")
        if "outline" in info:
            print(f"  輪郭の線を引いたドット {info['outline']['drawn']}")
        if "pattern" in info:
            t = info["pattern"]
            print(f"  模様を整えた範囲 {t['regions']} か所、変えたドット {t['changed']}")
        print(f"  → {png}\n  → {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
