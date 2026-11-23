"""回帰テスト用のゴールデン出力を生成する。

    python tools/make_golden.py                 # tests/golden/ を作り直す
    python tools/make_golden.py --fixtures      # 合成フィクスチャも作り直す
    python tools/make_golden.py -k presets      # 名前に "presets" を含むケースだけ

ケースの定義は `tests/golden_cases.py`。比較は `tests/test_golden.py`。
**変換結果が意図して変わったときだけ** 実行し、差分を確認してからコミットすること。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import golden_cases as gc  # noqa: E402


def write_fixtures() -> None:
    for rel, img in gc.make_fixture_images().items():
        p = gc.FIXTURES / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, format="PNG", optimize=False)
        print(f"  fixture {rel}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", action="store_true", help="合成フィクスチャも作り直す")
    ap.add_argument("-k", help="名前にこの文字列を含むケースだけ作り直す")
    args = ap.parse_args(argv)

    if args.fixtures:
        write_fixtures()
    names = [n for n in gc.CASES if not args.k or args.k in n]
    if not args.k:
        # 全ケースを作り直すときは古いファイルを残さない（README.md は残す）
        for d in gc.GOLDEN.iterdir() if gc.GOLDEN.exists() else []:
            if d.is_dir():
                shutil.rmtree(d)
    index = gc.load_index() if args.k and gc.INDEX.exists() else {}
    for name in names:
        files = gc.run_case(name)
        index[name] = sorted(files)
        for rel, data in files.items():
            p = gc.GOLDEN / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        print(f"  {name}: {len(files)} 件")
    index = {k: index[k] for k in gc.CASES if k in index}
    gc.INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  -> {gc.GOLDEN.relative_to(ROOT)} ({len(names)} ケース)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
