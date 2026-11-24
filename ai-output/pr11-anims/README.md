# PR #11 動きに名前を付けて 1 つのシートにまとめる

入力: `ai-input/` の ai07（walk）・ai09（run）・ai13_jump（jump。元の絵の高さを保つ・1 回で終わる）。
設定は `pr11.anims.json`。作り方:

```bash
python -m apps.sprite ai07.png ai09.png ai13_jump.png --anims pr11.anims.json -o out/chara.png --gif --piskel
```

| ファイル | 中身 |
| --- | --- |
| `1_sheet_x4.png` | まとめたシート全体（4 倍、ニアレストネイバー。透明は市松模様）。`1_sheet_x4_transparent.png` は透明のまま |
| `2_rows_labeled.png` | 動きの境目の見える化: 左に行ごとの動きの名前・時間・くり返し・フレームの番号、コマの枠に通し番号と `air`（地面の線から何ドット上か）、赤い線が地面の線 `ground_y` |
| `3_jump.gif` | jump の GIF（1 回だけ再生。4 倍） |
| `4_sheet.json` | まとめたシートの JSON（`animations` 入り） |

## 数

- 1 コマ 62×94 ドット、4 列 × 3 行（walk 4・run 4・jump 4 フレーム）、パレット 16 色（3 つの動きで共通）
- 地面の線 `ground_y` 94（全部の動きでフレームの下端）。`air`: walk・run は全部 0、jump は 0 / 10 / 27 / 0（#10 の比較画像と同じ浮き方）
- pivot (33.0, 94)
- 変換は 3 ファイルで 1 回（CLI で約 6 秒）

```json
"animations": {"walk": {"frames": [0, 1, 2, 3], "duration_ms": 150, "loop": true, "row": 0},
               "run": {"frames": [4, 5, 6, 7], "duration_ms": 100, "loop": true, "row": 1},
               "jump": {"frames": [8, 9, 10, 11], "duration_ms": 150, "loop": false, "row": 2}}
```

## 感想

- いちばん大きい jump（宙に浮く分で縦 94）に合わせて、walk・run は上に透明の行が足されている。キャラは拡大・縮小していないので、walk の方が jump より大きく見えるのは元の絵の大きさの違い（AI がファイルごとに違う大きさで描いた）
- 3 つのファイルでキャラの高さが違う（91・83・68 ドット。jump はしゃがみ・跳びのポーズ）ので「高さが違う」の注意が出る。ポーズの違いなのでそのままでよい
- 横は「頭の形を重ねる」の基準が 3 つの動きで同じ列に来るので、切り替えても頭の位置は横にずれない
