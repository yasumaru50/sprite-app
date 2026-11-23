# 回帰テストのフィクスチャ

`tests/test_golden.py`（回帰テスト）の入力です。ケースの定義は `tests/golden_cases.py`。

| パス | 内容 |
| --- | --- |
| `input/synth_photo.png` | コード合成の写真風 RGB 画像（空のグラデーション・肌色の球・丘・ノイズ） |
| `input/synth_alpha.png` | コード合成の RGBA 画像（透明背景。`background` の合成を確認） |
| `reference/synth/synth_cg.png` | `tests/synth_cg.py` の 16 色お手本（条件を満たす） |
| `reference/synth/synth_cg_bad.png` | 同上に 16 段階でない色を加えたもの（`check_reference` で除外される） |

合成画像は `python tools/make_golden.py --fixtures` で同じピクセルに再生成できます
（`test_synthetic_fixtures_reproducible` で確認）。

他の人が作った画像（お手本・写真など）は使いません。
