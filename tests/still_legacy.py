"""Still App の前の版と比べるテストの道具。

初めの設定から項目を変えたときに足した JSON の `preset`（初めの設定の名前・変えた項目・項目の値すべて）を除くと、
前の版と同じバイト列になる。前のハッシュと比べるテストは、`legacy_json()` を通したものを比べる
（初めの設定のままのときは `preset` が無いので、そのままのバイト列と同じ）。
"""

import json

from retrocore.still.convert import info_json_text


def legacy_json(text: str) -> str:
    info = json.loads(text)
    info.pop("preset", None)
    return info_json_text(info)
