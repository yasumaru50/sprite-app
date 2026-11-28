# still-05: モードを「1 本の工程と初めの設定」に整理して、新しくできた組み合わせ

モード（`--mode`）は初めの設定（項目の値の組み合わせ）になり、向きを直す・マス目の決め方・色の塊をまとめる も項目になった。
ここでは、前はできなかった組み合わせを 2 つ試した。PNG・JSON は `python -m apps.still` の出力そのもの（CLI）。
JSON の `preset` に、初めの設定の名前・変えた項目・項目の値すべてが入る。

| ファイル | 作り方 |
| --- | --- |
| still-05_ai17_room_fine_modern.png | `ai-input/ai17_room_fine.png --mode modern --width 600`（今風。前からできた） |
| still-05_ai17_room_fine_modern_weak.png | `--mode modern --width 600 --smooth weak`（今風＋色の塊を弱くまとめる。新しい組み合わせ） |
| still-05_ai17_compare_crop.png | 上の 2 枚の一部（左上 200,120〜400,260）を 3 倍で並べたもの（左: 今風、右: ＋弱くまとめる） |
| still-05_photo01_photo640.png | `ai-input/photo01.jpg --mode photo --width 640`（写真の下書き。前からできた） |
| still-05_photo01_photo640_16c_4bit.png | `--mode photo --width 640 --colors 16 --4bit`（写真の下書き＋16 色＋4096 色に丸め。新しい組み合わせ） |
| still-05_photo01_compare.png | 上の 2 枚を並べたもの（左: 写真の下書き、右: ＋16 色＋4096 色） |

| 結果 | 大きさ | 色数 | 孤立したドットを置き換えた数 | 変えた項目（JSON の preset.changed） | 時間（CLI） |
| --- | --- | --- | --- | --- | --- |
| ai17 modern | 600×375 | 32 | - | find_grid, width | 2.2 秒 |
| ai17 modern_weak | 600×375 | 32 | - | find_grid, width, smooth | 2.0 秒 |
| photo01 photo640 | 640×853 | 32 | 443 | width | 6.3 秒 |
| photo01 photo640_16c_4bit | 640×853 | 16 | 395 | width, colors, four_bit | 6.5 秒 |

感想:
- ai17（今風＋弱くまとめる）: 空・山・床のざらざらの点が減って面がまとまり、手直しの出発点として見やすい。電線・手すりなどの細い線はほぼ残るが、
  ところどころ途切れる所がある（細い線を残したいときは、まとめない方がよい）
- photo01（＋16 色＋4096 色）: 壁・床が大きな面になり、98 風の下絵に近い。床の紫がかったタイルは灰色に寄り、色の段は粗くなる。
  4096 色に丸めて重なった色は無かった（16 色のまま）

注意: photo01 は JPEG なので、ブラウザ版では少し違う結果になる（JPEG を画素に戻すライブラリの違い。PNG にした入力ではバイト単位で同じ）。
