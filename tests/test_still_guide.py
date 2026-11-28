"""Still App の「？」の説明の文（retrocore/still/guide.py）。ブラウザ版と CLI の --help で共通。"""

from __future__ import annotations

import re
from pathlib import Path

from apps.still import cli
from retrocore.still import guide
from retrocore.still.convert import ITEMS, MODES, spec

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "apps" / "still" / "web" / "template.html"


def test_every_item_and_mode_has_three_lines():
    assert list(guide.ITEMS) == list(ITEMS)   # 工程の順のまま、全部の項目
    assert list(guide.MODES) == list(MODES) == spec()["guide"]["mode_order"]
    g = spec()["guide"]
    for key, item in g["items"].items():
        heads = [h for h, _ in item["lines"]]
        assert heads[0] in guide.LEAD_LABELS.values(), key
        assert heads[1:] == ["気を付けること", "向いている絵"], key
        assert item["cli"].startswith("--"), key
        for _, text in item["lines"]:
            assert text and len(text) <= 60 and "\n" not in text, (key, text)   # 1 行は短く
    for key, mode in g["modes"].items():
        assert [h for h, _ in mode["lines"]] == ["選ぶと", "気を付けること", "向いている絵"], key


def test_lead_label_by_kind():
    g = spec()["guide"]["items"]
    assert g["orient"]["lines"][0][0] == "オンにすると"
    assert g["dither"]["lines"][0][0] == "選ぶと"
    assert g["width"]["lines"][0][0] == "入れると"


def test_template_has_a_help_for_every_item():
    html = TEMPLATE.read_text(encoding="utf-8")
    keys = re.findall(r'data-guide="([a-z_]+)"', html)
    assert sorted(keys) == sorted(ITEMS)   # 「必要なときに開く」の全部の項目に 1 つずつ
    assert "data-guide-modes" in html


def test_cli_help_uses_same_text():
    text = cli.build_parser().format_help()
    flat = re.sub(r"\s+", "", text)
    for key in ("orient", "width", "aspect", "smooth", "colors", "four_bit", "merge_similar", "dither", "tidy_dots",
                "tidy_pattern", "outline", "canvas"):
        assert re.sub(r"\s+", "", guide.ITEMS[key]["lead"]) in flat, key
    for key in MODES:
        assert re.sub(r"\s+", "", guide.MODES[key]["lead"]) in flat, key
    assert "0〜100 の%" in text   # % が argparse の書式で壊れない
