"""Sprite App のブラウザ版を、1 枚で動く HTML にまとめる。

    python tools/build_sprite_web.py            # apps/sprite/web/sprite-app.html を作り直す
    python tools/build_sprite_web.py --check    # 作り直しが必要なら 1 で終わる（テスト・CI 用）

ひな形（apps/sprite/web/template.html）に、retrocore の Python ファイル（そのまま）と、
画面の設定の選択肢・既定値（retrocore.sprite.convert の既定 = CLI の既定）を埋め込む。
出来上がった HTML はサーバーなしで、ダウンロードしてダブルクリックで開ける（Pyodide はインターネットから読み込む）。
同じ入力からは毎回同じファイルができる（日時などは入れない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrocore.sprite import frames as frames_mod  # noqa: E402
from retrocore.sprite import anims as anims_mod  # noqa: E402
from retrocore.sprite import steady as steady_mod  # noqa: E402
from retrocore.sprite import outline as outline_mod  # noqa: E402
from retrocore.sprite.cleanup import LEVEL_HELP  # noqa: E402
from retrocore.sprite.convert import MAX_COLORS, SpriteSettings, motion_spec  # noqa: E402
from retrocore.sprite.motion import gif as gif_mod  # noqa: E402
from retrocore.sprite.fx import fx_spec  # noqa: E402
from retrocore.sprite import compare as compare_mod  # noqa: E402

WEB = ROOT / "apps" / "sprite" / "web"
TEMPLATE = WEB / "template.html"
OUTPUT = WEB / "sprite-app.html"
# Pyodide の版。NumPy 2.0 と Pillow を含む版に固定する（変えたら画面で動くか確かめる）
PYODIDE_VERSION = "0.27.7"


def python_sources() -> dict[str, str]:
    """retrocore の .py ファイル（パッケージからの相対パス → 中身）。改行は LF にそろえる。

    1 枚絵のアプリ（Still App）だけが使う retrocore/still/ は入れない（Sprite App の HTML を変えないため）。
    """
    files = sorted(p for p in (ROOT / "retrocore").rglob("*.py")
                   if "__pycache__" not in p.parts and p.relative_to(ROOT / "retrocore").parts[0] != "still")
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8").replace("\r\n", "\n") for p in files}


def settings_spec() -> dict:
    """画面で変えられる設定の選択肢と既定値（既定値は SpriteSettings = CLI と同じ）。"""
    d = SpriteSettings()
    return {
        "colors": {"default": d.colors, "min": 1, "max": MAX_COLORS},
        "cleanup": {"default": d.cleanup, "choices": [[k, v] for k, v in LEVEL_HELP.items()]},
        "outline": {"default": d.outline, "choices": [[k, v] for k, v in outline_mod.MODES.items()]},
        "motion": motion_spec(),   # 動き（呼吸）の欄。既定は CLI の --motion-frames などと同じ
        "fx": fx_spec(),   # 重ね（--fx）の選択肢
        "compare": compare_mod.spec(),   # 大きさをくらべる（CLI の compare の --scale・--grid・--gap）
        "gif": {"default": gif_mod.DEFAULT_SCALE, "min": gif_mod.MIN_SCALE, "max": gif_mod.MAX_SCALE},   # GIF の拡大（--gif-scale）
        # コマが並んだ絵（--frames）。画面に出すそろえ方は初心者向け（CLI の center・none は出さない）
        "frames": {"align": {"default": frames_mod.DEFAULT_ALIGN, "choices": [
                       ["match", "頭の形を重ねる（おすすめ）"], ["head", "頭の真ん中に合わせる"]]},
                   "align_y": {"default": frames_mod.DEFAULT_ALIGN_Y, "choices": [
                       ["foot", "足元（歩きなど。おすすめ）"], ["match", "頭の高さ（走り・跳びなど）"],
                       ["keep", "元の絵の高さを保つ（ジャンプなど、宙に浮くコマ）"]]},
                   "steady": {"default": steady_mod.DEFAULT_MODE, "size": steady_mod.DEFAULT_SIZE, "choices": [
                       ["off", "なし（今までと同じ）"], ["on", "あり（小さな違いを 1 コマ目に戻す）"]]},
                   "duration": frames_mod.DEFAULT_DURATION, "name": frames_mod.DEFAULT_NAME,
                   "max_frames": frames_mod.MAX_FRAMES, "max_shift_y": frames_mod.MAX_SHIFT_Y},
        # 動きに名前を付けてまとめる（--anims）。名前の候補・決まり・まとめている間は使えないものの理由
        "anims": {"name": anims_mod.DEFAULT_NAME, "suggested": list(anims_mod.SUGGESTED), "name_max": anims_mod.NAME_MAX,
                  "max_anims": anims_mod.MAX_ANIMS, "unsupported": dict(anims_mod.UNSUPPORTED)},
    }


def _script_json(value) -> str:
    """<script> の中に置ける JSON（`<` を \\u003c にして、</script> などで途切れないようにする）。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("<", "\\u003c")


def build() -> str:
    template = TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")
    sources = python_sources()
    spec = settings_spec()
    digest = hashlib.sha256()
    for part in (template, _script_json(sources), _script_json(spec), PYODIDE_VERSION):
        digest.update(part.encode("utf-8"))
    html = template
    # ひな形の説明コメントは、出来上がった HTML 向けの言葉に差し替える
    head, rest = html.split("<!--", 1)
    _, rest = rest.split("-->", 1)
    html = (head + "<!--\n  Sprite App（ブラウザ版）。このファイルは tools/build_sprite_web.py で作ったもの（直接編集しない）。\n"
            "  元は apps/sprite/web/template.html と retrocore/。ダウンロードしてダブルクリックで開ける。\n-->" + rest)
    for key, value in (("__SETTINGS_SPEC__", _script_json(spec)), ("__PY_SOURCES__", _script_json(sources)),
                       ("__PYODIDE_VERSION__", PYODIDE_VERSION), ("__BUILD_ID__", digest.hexdigest()[:12])):
        if key not in html:
            raise RuntimeError(f"ひな形に {key} がありません")
        html = html.replace(key, value)
    return html


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="作り直しが必要かだけを確かめる（書き換えない）")
    p.add_argument("-o", "--output", type=Path, default=OUTPUT, help=f"出力先（既定 {OUTPUT.relative_to(ROOT)}）")
    args = p.parse_args(argv)
    html = build()
    if args.check:
        current = args.output.read_text(encoding="utf-8").replace("\r\n", "\n") if args.output.exists() else None
        if current != html:
            print(f"{args.output} が古くなっています。python tools/build_sprite_web.py で作り直してください",
                  file=sys.stderr)
            return 1
        print(f"{args.output} は最新です")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(f"→ {args.output}（{len(html.encode('utf-8')) // 1024} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
