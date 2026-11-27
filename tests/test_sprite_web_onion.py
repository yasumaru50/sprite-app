"""画面の「ドットを直す」の「重ねるコマ」（どのコマを薄く重ねるか）を確かめる。

template.html から onionFrame() を取り出して Node で動かす（Node が無ければ飛ばす）。
見るためだけの機能なので、出力（PNG・JSON・GIF など）には関係しない。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "apps" / "sprite" / "web" / "template.html"
NODE = shutil.which("node")


def _function_source(name: str) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    start = text.index(f"function {name}(")
    m = re.compile(r"^}\n", re.M).search(text, start)
    return text[start:m.end()]


def _run(cases: list) -> list:
    src = _function_source("onionFrame")
    js = src + "\nconst cases = " + json.dumps(cases) + ";\n" \
        "console.log(JSON.stringify(cases.map(([k, how, o]) => onionFrame(k, how, o))));\n"
    out = subprocess.run([NODE, "-e", js], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _o(order, list_=None, groups=None, pick=None, n=None):
    list_ = list_ or sorted({abs(v) for v in order})
    return {"order": order, "list": list_, "groups": groups, "pick": pick, "n": n or len([v for v in list_ if v > 0])}


pytestmark = pytest.mark.skipif(NODE is None, reason="Node が無い")


def test_prev1_is_unchanged():
    o = _o([1, 2, 3, 4])
    assert _run([[1, "prev1", o], [3, "prev1", o]]) == [4, 2]
    # 再生に使っていないコマは 1 つ小さい番号（1 コマ目なら最後）
    o = _o([1, 2], list_=[1, 2, 3], n=3)
    assert _run([[3, "prev1", o], [1, "prev1", o]]) == [2, 2]


def test_prev2_wraps_from_the_end():
    o = _o([1, 2, 3, 4])
    assert _run([[3, "prev2", o], [1, "prev2", o], [2, "prev2", o]]) == [1, 3, 4]


def test_first_and_same_frame_gives_none():
    o = _o([1, 2, 3, 4])
    assert _run([[3, "first", o], [1, "first", o]]) == [1, None]
    one = _o([1])
    assert _run([[1, "prev2", one], [1, "first", one]]) == [None, None]
    # 2 コマで 2 つ前は自分
    assert _run([[1, "prev2", _o([1, 2])]]) == [None]


def test_pick():
    o = _o([1, 2, 3, 4], pick=2)
    assert _run([[4, "pick", o], [2, "pick", o]]) == [2, None]
    assert _run([[4, "pick", _o([1, 2, 3, 4], pick=9)]]) == [None]
    o = _o([1, 2, 3, -2], list_=[1, 2, 3, -2], pick=-2, n=3)
    assert _run([[2, "pick", o]]) == [-2]


def test_mirrored_frames():
    # 順番 1,2,3,2r: 2r の 2 つ前は 2、1 の 2 つ前は 3
    o = _o([1, 2, 3, -2], list_=[1, 2, 3, -2], n=3)
    assert _run([[-2, "prev2", o], [1, "prev2", o], [-2, "first", o]]) == [2, 3, 1]
    # 再生に使っていない裏返したコマ（直しだけが残っている）は、選べるコマの並びで数える
    o = _o([1, 2, 3], list_=[1, 2, 3, -2], n=3)
    assert _run([[-2, "prev2", o]]) == [2]


def test_anims_count_inside_the_same_animation():
    # 歩き 1〜4、走り 5〜8
    o = _o([1, 2, 3, 4, 5, 6, 7, 8], groups=[[1, 4], [5, 8]])
    assert _run([[5, "prev2", o], [6, "prev2", o], [1, "prev2", o], [7, "first", o], [3, "first", o]]) == [7, 8, 3, 5, 1]
    # 1 つ前は今までと同じ（動きをまたぐ）
    assert _run([[5, "prev1", o]]) == [4]


def test_ui_has_onion_choice():
    text = TEMPLATE.read_text(encoding="utf-8")
    for s in ('id="touch-onion-of"', 'value="prev1" selected', 'value="prev2"', 'value="first"', 'value="pick"', 'id="touch-onion-pick"'):
        assert s in text
