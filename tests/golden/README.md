# 回帰テストのゴールデン出力

`retrocore/` の CG 変換（通常・パレット・トランスファー・スタイル変換・解析）の結果を固定したものです。入力は `tests/fixtures/` のコード合成画像だけです。`tests/test_golden.py` が
同じ条件で再実行し、PNG はピクセル単位（インデックスとパレット）、JSON / CSV / GPL はバイト単位で比較します。

| フォルダ | 内容 |
| --- | --- |
| `normal/` | 通常変換（`mode="normal"`）: 合成の写真風画像を既定・誤差拡散・解像度変更で変換、RGBA の背景合成、プリセットパレット 3 種 × 2 ディザ |
| `transfer/` | パレット・トランスファー: 合成のお手本（`synth_cg.png`）から `extract_palette()` で抽出したパレットを強制（GPL 書き出しも比較）。スタイル変換に固定パレットを渡す場合も含む |
| `style/` | スタイル変換（`mode="style"`）: 既定・ランプをまたいだ混色・RGBA |
| `presets/` | `STYLE_PRESETS` の 5 種 |
| `profile/` | 合成のお手本の解析（`analyze_image` → `build_profile`）、`profile_to_settings()`、その設定での変換、`fit_to_profile()`（評価 8 回） |
| `export_db/` | `tools/export_db.py` の書き出し（`records/*.json`、`palettes/*.gpl`、`index.csv`）そのもの。合成のお手本 2 枚（条件を満たすもの・満たさないもの） |
| `index.json` | ケース名 → 出力ファイルの一覧 |

各変換ケースは `<名前>.png`（変換結果のインデックス画像）と `<名前>.json`（パレット・使用色数・画素比・
`info`（ランプ・ディザ被覆率・ディザマスクの SHA-256 など）・使った `ConvertSettings`）の組です。

乱数を使う処理は k-means の初期化（`palette._kmeans`、`default_rng(seed=0)`）だけで、シードは既定で固定済みです。

## 作り直し

変換結果を **意図して** 変えたときだけ実行し、差分を確認してからコミットしてください。

```bash
python tools/make_golden.py              # 全ケース
python tools/make_golden.py -k presets   # 名前に presets を含むケースだけ
```

ここに入るのは合成画像の変換後の画像と統計データだけです（他の人が作った画像やその変換結果は入れない）。
改行の変換で比較が壊れないよう、`.gitattributes` でこのフォルダを `-text` にしています。
