"""Still App（1 枚絵）のブラウザ版を、1 枚で動く HTML にまとめる。

    python tools/build_still_web.py            # apps/still/web/still-app.html を作り直す
    python tools/build_still_web.py --check    # 作り直しが必要なら 1 で終わる（テスト・CI 用）

ひな形（apps/still/web/template.html）に、retrocore の Python ファイル（そのまま）と、
画面の設定の選択肢・既定値（retrocore.still.convert のプリセット = CLI の既定）を埋め込む。作り方は Sprite App
（tools/build_sprite_web.py）と同じで、Pyodide の版も同じものを使う。ビルド番号は埋め込んだ中身から決まる
（同じ入力からは毎回同じファイルができる。日時などは入れない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrocore.still.convert import spec as still_spec  # noqa: E402

WEB = ROOT / "apps" / "still" / "web"
TEMPLATE = WEB / "template.html"
OUTPUT = WEB / "still-app.html"
# Pyodide の版（Sprite App と同じ。変えたら画面で動くか確かめる）
PYODIDE_VERSION = "0.27.7"


def python_sources() -> dict[str, str]:
    """retrocore の .py ファイル（パッケージからの相対パス → 中身）。改行は LF にそろえる。

    retrocore.still は retrocore.sprite（グリッド検出・単純化）を使い、retrocore/__init__.py などが
    ほかのモジュールも読み込むので、retrocore を全部入れる。
    """
    files = sorted(p for p in (ROOT / "retrocore").rglob("*.py") if "__pycache__" not in p.parts)
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8").replace("\r\n", "\n") for p in files}


def settings_spec() -> dict:
    """画面で変えられる設定の選択肢と既定値（CLI の既定と同じ）。"""
    return still_spec()


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
    head, rest = html.split("<!--", 1)
    _, rest = rest.split("-->", 1)
    html = (head + "<!--\n  Still App（ブラウザ版）。このファイルは tools/build_still_web.py で作ったもの（直接編集しない）。\n"
            "  元は apps/still/web/template.html と retrocore/。ダウンロードしてダブルクリックで開ける。\n-->" + rest)
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
            print(f"{args.output} が古くなっています。python tools/build_still_web.py で作り直してください",
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
