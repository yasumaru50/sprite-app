"""回帰テスト: 現在の変換結果（tests/golden/）とピクセル単位・バイト単位で一致するか。

ケースの定義は `golden_cases.py`。変換結果を意図して変えたときは
`python tools/make_golden.py` で作り直し、差分を確認してからコミットする。
"""

import io
import json

import numpy as np
import pytest
from PIL import Image

import golden_cases as gc


def test_synthetic_fixtures_reproducible():
    """合成フィクスチャはコードから同じピクセルで再生成できる。"""
    for rel, img in gc.make_fixture_images().items():
        saved = Image.open(gc.fixture(rel))
        assert saved.mode == img.mode and saved.size == img.size, rel
        assert np.array_equal(np.asarray(saved), np.asarray(img)), rel


def _compare_png(rel: str, got: bytes, want_path) -> None:
    a = Image.open(io.BytesIO(got))
    b = Image.open(want_path)
    assert (a.mode, a.size) == (b.mode, b.size), f"{rel}: モード・サイズが違います"
    if a.mode == "P":
        # インデックスとパレットの両方を比較
        assert a.getpalette() == b.getpalette(), f"{rel}: パレットが違います"
    av, bv = np.asarray(a), np.asarray(b)
    diff = np.argwhere(av != bv)
    assert len(diff) == 0, (
        f"{rel}: {len(np.unique(diff[:, :2], axis=0))} 画素が違います（最初: y,x = {diff[0][:2].tolist()}）")
    if a.mode == "P":
        assert np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))), rel


def _compare_text(rel: str, got: bytes, want: bytes) -> None:
    if rel.endswith((".gpl", ".json")):
        # save_palette / export_db は Path.write_text（テキストモード）で書くため、
        # Windows では改行が CRLF になる。ゴールデンは LF なので改行だけそろえる
        # （内容は 1 バイトも緩めない。CSV は newline="" で両環境とも CRLF なのでそのまま比較）。
        got = got.replace(b"\r\n", b"\n")
    if rel.endswith(".json"):
        assert json.loads(got) == json.loads(want), f"{rel}: JSON の内容が違います"
    assert got == want, f"{rel}: 内容が違います"


@pytest.mark.parametrize("name", list(gc.CASES))
def test_golden(name):
    files = gc.run_case(name)
    index = gc.load_index()
    assert name in index, f"{name}: ゴールデンがありません（python tools/make_golden.py）"
    # 出力ファイルの一覧も一致すること（export_db の書き出しファイルなど）
    assert sorted(files) == index[name], f"{name}: 出力ファイルの一覧が違います"
    for rel, data in files.items():
        want = gc.GOLDEN / rel
        if rel.endswith(".png"):
            _compare_png(rel, data, want)
        else:
            _compare_text(rel, data, want.read_bytes())


def test_golden_index_matches_files():
    """index.json とケース定義・tests/golden/ のファイルが過不足なく対応する。"""
    index = gc.load_index()
    assert sorted(index) == sorted(gc.CASES)
    listed = {rel for files in index.values() for rel in files}
    on_disk = {p.relative_to(gc.GOLDEN).as_posix() for p in gc.GOLDEN.rglob("*") if p.is_file()}
    assert on_disk - {"index.json", "README.md"} == listed
