# CLI のオプションと出力の形

`python -m apps.sprite` のオプションの一覧と、書き出すファイル・JSON の形。使い方の流れは [apps/sprite/README.md](../apps/sprite/README.md)、
工程の仕組みと値を決めた理由は [how-it-works.md](how-it-works.md)。リポジトリのルートで実行する（依存は `requirements.txt`。NumPy と Pillow だけ）。

目次: [変換](#変換のオプション) / [パレット](#パレット) / [動き](#動き1-枚の絵) / [GIF](#gif-アニメ---gif) / [動きの表](#動きの表---motion-spec----write-motion-spec) /
[コマが並んだ絵](#コマが並んだ絵---frames) / [差し替え](#差し替えまばたき---swaps) / [読み戻し](#手直ししたシートの読み戻し---readback) /
[キャンバスを倍数にそろえる](#キャンバスを倍数にそろえる---canvas-multiple) / [大きさをくらべる](#大きさをくらべるcompare) / [シートの JSON](#シートの-json) /
[1 枚絵（Still App）](#1-枚絵still-app)

## 変換のオプション

```bash
python -m apps.sprite 入力.png --preview-scale 10 -o out/sprite/出力.png                     # ドット絵風の画像（既定）
python -m apps.sprite 入力.png --input-type illust --scale 1/4 -o out/sprite/出力.png        # 普通のイラスト
```

| オプション | 内容 |
| --- | --- |
| `入力` | 画像ファイルまたはフォルダ（複数可）。フォルダは中の画像（png / jpg / webp / bmp / gif / tif）をすべて処理 |
| `--input-type` | `pixel`（既定）= ドット絵風（グリッド検出）、`illust` = 普通のイラスト（縮小）。自動判定はしない |
| `--width W` / `--height H` | （`illust` のみ）出力サイズ。片方だけなら縦横比を保つ。両方なら縦横比を無視してそのサイズ |
| `--scale S` | （`illust` のみ）縮小倍率（`0.25` または `1/4`）。幅・高さとは同時に使えない |
| `--dot-size W[xH]` | （`pixel` のみ）ドットの大きさ（画素。`20` または `20x21`、小数可）。省略時は推定。指定すると「見つからない」の判定をしない |
| `--dot-shape` | （`pixel` のみ）`square`（既定）= 正方形とみなし縦横を合わせて推定、`separate` = 縦横を別々に推定（横長・縦長のドット向け） |
| `--grid-origin X,Y` | （`pixel` のみ）グリッドの始まる位置。指定すると線を色の境目に合わせず、等間隔の格子にする |
| `--grid-snap` / `--no-grid-snap` | （`pixel` のみ）グリッド線を近くの色の境目に合わせる（既定で有効）。無効にすると等間隔の格子 |
| `--grid-tolerance R` | （`pixel` のみ）線の間隔をドットの大きさの ±R 倍まで許す（既定 0.2） |
| `--background` / `--no-background` | 背景除去（既定で有効）。外周からつながった背景色の部分を透明にする。入力がすでに透過付きならそのまま通す |
| `--bg-color COLOR` | 背景色（`#RRGGBB` または `R,G,B`）。省略時は画像の外周から推定 |
| `--bg-tolerance T` | 背景色とみなす差（R・G・B の差の最大値、0〜255、既定 32） |
| `--bg-remove-enclosed` | キャラクターに囲まれた背景（髪の束の間、腕と体のすき間など）も消す。背景と同じ色の白い服なども消えるので注意 |
| `--resize-method` | （`illust` のみ。以下の縮小のオプションも同じ）`mode`（既定）/ `median` / `nearest` / `box` / `bilinear` / `lanczos` |
| `--mode-bin N` | `mode` で色をまとめる区間の幅（1〜256、既定 32）。1 なら完全に同じ色だけを数える |
| `--line-priority` / `--no-line-priority` | 線優先（`mode` / `median` のみ。既定で有効）。暗い輪郭線の画素が一定割合以上のブロックは線の色にする。`nearest` などでは無効（明示すればエラー） |
| `--line-ratio R` | 線優先にする暗い画素の割合（0〜1、既定 0.25） |
| `--line-luma L` | 暗い画素とみなす輝度の上限（0〜255、既定 64） |
| `--colors N` | 単純化で残す色の数の上限（1〜256、既定 16。PR #39 まで 12）。透明は数えない |
| `--merge-distance D` | この距離（OKLab。色みは 1.5 倍で測る）より近い色は、色数の上限に関係なく必ず 1 色にまとめる（0〜1、既定 0.03）。`0` でまとめない（以前の既定の動き）。大きくすると（0.06 など）近さで決められる |
| `--keep-large R` | 面積が全体のこの割合以上の色どうしは、距離 0.06 以上の差があれば、ほかにまとめる組が無くなるまでまとめない（0〜1、既定 0.01）。`0` で以前の動き |
| `--simplify` / `--no-simplify` | 単純化（既定で有効）。`--no-simplify` でグリッド検出・縮小の結果をそのまま出す |
| `--cleanup LEVEL` | クリーンアップの強さ。`off` / `light` / `normal`（既定）/ `strong` |
| `--no-cleanup` | `--cleanup off` と同じ |
| `--outline MODE` | 縁取り。`auto`（既定）/ `add` / `thin` / `off` |
| `--no-outline` | `--outline off` と同じ |
| `--outline-color C` | 外側に付ける輪郭の色。`darken`（既定。すぐ内側の色を暗くした色）/ `single`（絵の中でいちばん暗い色の 1 色）/ 色の指定（`#3a2030` または `58,32,48`） |
| `--outline-shape S` | 外側に付ける線の形。`thin`（既定。斜めでつながる 1 ドットの線）/ `thick`（上下左右でつながる線、角も埋める）。`--outline thin` とは別 |
| `--outline-colors N` | `darken` で新しく作る輪郭の色の数の上限（既定 4。近い色はまとめる） |
| `--outline-pad` / `--no-outline-pad` | シルエットが画像の端に接しているとき、その辺に 1 ドットの余白を足してから輪郭を付ける（既定で有効） |
| `--palette FILE` | このパレットの色だけで出力する（`.hex` / `.gpl` / `.png`）。下の「パレット」 |
| `--save-palette FILE` | 最終結果で使った色（透明は含めない）を保存する（`.hex` / `.gpl`。拡張子で決める。何度でも指定できる）。入力が複数のときは `<名前>_<出力名>.<拡張子>` |
| `--transparent-ratio R` | 透過画素（α < 128）がこの割合以上のマス（グリッド検出）・ブロック（`mode` / `median`）を透過にする（既定 0.5） |
| `-o, --output` | `.png` で終わればそのファイル（入力 1 枚のとき）。それ以外はフォルダ（既定 `out/sprite/`、名前は `<入力名>.png`） |
| `--preview-scale N` | 出力を最近傍で N 倍（1〜32）にしたプレビューも `<出力名>_xN.png` に保存する。出力そのものは変わらない |
| `--save-steps` | 工程ごとの中間画像を `<出力名>_steps/01_background.png`、`02_grid.png`（または `02_resize.png`）…に保存 |
| `-q, --quiet` | 進捗を表示しない |

- **`pixel` のサイズ指定**: 出力の大きさはグリッド検出で決まる（1 ドット＝1 画素、キャラクターの大きさに切り詰め）。`--width`・`--height`・`--scale`・
  縮小のオプションを付けるとエラーで止まって理由を出す（黙って無視すると指定が効いたと誤解するため）。大きさを変えたいときは
  `--dot-size 20`（推定が外れる）・`--input-type illust --width 64`（決まった大きさに縮小）・`--preview-scale 10`（見やすく拡大）
- 格子が見つからないときは変換せずに理由を出して止まる（終了コード 1。フォルダ入力では他の画像は続ける）
- 透過 PNG は透過を保ったまま処理する。出力は常に RGBA の PNG

```bash
python -m apps.sprite chara.png --dot-size 20x18 -o out/sprite/chara.png                      # ドットの大きさを指定
python -m apps.sprite chara.png --dot-size 20 --grid-origin 5,3 -o out/sprite/chara.png       # 等間隔の格子
python -m apps.sprite chara.png --colors 16 --merge-distance 0 -o out/sprite/chara.png        # 見分けのつかない色もまとめない
python -m apps.sprite chara.png --colors 32 --merge-distance 0.06 -o out/sprite/chara.png     # 近さで決める
python -m apps.sprite chara.png --cleanup light --preview-scale 8 -o out/sprite/chara.png     # クリーンアップを弱めに
python -m apps.sprite walk.png --frames --frame-count 3 --outline thin -o out/sprite/walk.png # 太い外側の輪郭を 1 ドットに
python -m apps.sprite girl.png --input-type illust --height 48 --outline add --outline-color "#3a2030" -o out/sprite/girl.png
python -m apps.sprite chara.png --input-type illust --bg-color "#d4f6bd" --bg-tolerance 24 --width 64 -o out/sprite/chara64.png
python -m apps.sprite chara.png --input-type illust --width 64 --resize-method nearest -o out/sprite/chara64.png
python -m apps.sprite local_samples --input-type illust --height 64 -o out/sprite               # フォルダをまとめて
```

## パレット

```bash
python -m apps.sprite chara1.png -o out/sprite/chara1.png --save-palette out/sprite/game.hex --save-palette out/sprite/game.gpl
python -m apps.sprite chara2.png -o out/sprite/chara2.png --palette out/sprite/game.gpl
```

| 拡張子 | 読む | 書く | 中身 |
| --- | --- | --- | --- |
| `.hex` | ○ | ○ | 1 行に `RRGGBB` を 1 つ（Lospec などの形式。先頭の `#` は省略可、`;` で始まる行は読み飛ばす） |
| `.gpl` | ○ | ○ | GIMP のパレット（Aseprite も読み書きできる）。`Channels: RGBA` の形式も読める（α は使わない）。色の名前に `#RRGGBB` を入れる |
| `.png` | ○ | × | パレットの見本画像。不透明な画素の色を重複なしで、上から・左からの順に読む（透明は読まない。256 色まで） |

- 保存する並び: 灰色・黒・白を先頭に暗い → 明るい、残りは色相のグループ（赤 → 橙・肌 → 黄 → 緑 → 水色 → 青 → 紫 → ピンク）ごとに暗い → 明るい。
  読み込んだパレットはファイルの順のまま使う
- `--palette` の警告（`-q` でも出す。シートの JSON の `skipped` にも `stage: "palette"` で入る）:

  ```
  警告（chara2.png）: #f7845f（375 ドット） と #ebb39d（170 ドット） が、パレットの同じ色 #fa9a8d になりました
  警告（chara2.png）: #d55e5b（163 ドット）に近い色がパレットにありません。いちばん近い #a45d28 にしました（距離 0.141）
  ```

  色は単純化の後の色（部品の色）。「近い色がない」は最も近い色との距離が 0.1 以上のとき。警告が出たらパレットに色を足す（.hex なら 1 行）か、エディタで直す
- 外部のエディタで使うとき: Aseprite は `.gpl`・`.hex`・`.png` を読める。インデックスカラーにすると先頭（0 番）の色が透明扱いになることがあるので確かめる
  （保存したパレットに透明の色は入れていない）。Unity・Godot はパレットファイルを読まないので、出力の PNG をそのまま使う

## 動き（1 枚の絵）

変換の結果（`-o` の PNG）は `--motion` を付けても変わらない。シートは `<出力名>_<動き>_sheet.png` / `.json`。

| オプション | 内容 |
| --- | --- |
| `--motion NAME` | `breathe`（呼吸）/ `sway`（横揺れ）/ `float`（浮遊）/ `bounce`（弾む）/ `nod`（うなずく）/ `stretch`（伸び） |
| `--motion-frames N` | フレーム数。呼吸 2〜32（既定 2）、横揺れ・浮遊 3〜32（既定 4）。弾む・うなずく・伸びは量で決まるので使えない |
| `--motion-duration MS[,MS...]` | 表示時間（ミリ秒）。1 つなら全フレーム同じ、並べればフレームごと（数はフレーム数と同じ）。既定: 呼吸 500、横揺れ 300、浮遊 400、弾む `300,150,150,150`（着地だけ長め）、うなずく `600,200`、伸び 500 |
| `--breathe-depth 1\|2` | 呼吸の下がる量（既定 1）。4 フレーム・2 → 0, 1, 2, 1 |
| `--breathe-split auto\|N` | 呼吸の上下を分ける高さ。N は**出力の画像の下から何ドット目か**（1 = いちばん下の行） |
| `--sway-amount 1\|2` | 横揺れの揺れ幅（既定 1）。8 フレーム・2 → 0, 1, 2, 1, 0, −1, −2, −1 |
| `--sway-split auto\|N` | 揺れ始める高さ（数え方は `--breathe-split` と同じ） |
| `--float-amount 1\|2` | 浮遊の揺れ幅（既定 1）。8 フレーム・2 → 0, −1, −2, −1, 0, +1, +2, +1（− が上） |
| `--bounce-height 1\|2\|3` | 弾む高さ（既定 2）。1 → 0, −1、2 → 0, −1, −2, −1、3 → 0, −1, −2, −3, −2, −1 |
| `--nod-depth 1\|2` | うなずくの深さ（既定 1）。1 → 0, 1、2 → 0, 1, 2, 1 |
| `--nod-split auto\|N\|P%` | 首の高さ。auto は首を自動で探す。`31.0%` はキャラの高さに対する下からの割合 |
| `--stretch-amount 1\|2` | 伸びる量（既定 1）。1 → 0, +1、2 → 0, +1, +2, +1 |
| `--stretch-split auto\|N\|P%` | 伸びの分ける高さ（auto は呼吸と同じ探し方） |
| `--keep-small-parts` | 目・頬など（同じ色の塊で縦 3 ドット以下・面積 24 ドット以下）の長さを変えない（呼吸・うなずく・伸びだけ。既定は切） |

- `--swaps JSON`: ブラウザ版の「差し替えを描く」で保存した `<出力名>_<動き>_swaps.json` で、シートの中にまばたきを入れる。口パクがあれば、しゃべりのシートも書く（下の「[差し替え](#差し替えまばたき---swaps)」）
- 動きごとのオプションをほかの動きに付けるとエラー（例 `--breathe-depth は --motion sway では使えません`）
- 作れないとき（決まりを守れない形）は動きを付けずに警告し、シートは書かない（変換の結果は保存する）。指定した高さの近くで作れないときは
  「近くで分けられる高さ: 13、16」のように試せる数を出す（勝手に別の高さで作らない）。画像の外の高さはエラー（終了コード 1）
- 代わりに `<出力名>_<動き>_skipped.json` を書く（下の「作らなかった理由」）
- 分ける高さの目安の画像: `python tools/compare_sprite_motion.py chara.png --rows`（左に 5 行ごとの番号。灰色は足元で指定できない）。`--split 12 --split 16` で高さごとの GIF も

```bash
python -m apps.sprite chara.png -o out/sprite/chara.png --motion breathe --motion-frames 4 --breathe-depth 2 --motion-duration 300,200,300,200
python -m apps.sprite chara.png -o out/sprite/chara.png --motion sway --sway-split 10 --motion-frames 8 --sway-amount 2 --motion-duration 150
python -m apps.sprite chara.png -o out/sprite/chara.png --motion float --motion-frames 8 --float-amount 2
python -m apps.sprite chara.png -o out/sprite/chara.png --motion bounce --bounce-height 3 --motion-duration 120
python -m apps.sprite chara.png -o out/sprite/chara.png --motion nod --nod-split 16 --nod-depth 2
python -m apps.sprite chara.png -o out/sprite/chara.png --motion stretch --stretch-amount 2 --keep-small-parts
```

## GIF アニメ（`--gif`）

見せる用（ゲームにはシートの PNG と JSON を使う）。プリセットでも `--motion-spec` でも `--frames` でも使える。

| オプション | 内容 |
| --- | --- |
| `--gif` | `<出力名>_<動き>.gif` も書く（`--frames` はシートと同じ名前の `.gif`） |
| `--gif-scale N` | 拡大 1〜8（既定 4。最近傍） |
| `--gif-background transparent\|#rrggbb` | 既定は透明（1 ビット）。色ならその色で塗る |

- 各フレームはシートのフレームと同じ絵。表示時間は JSON の `duration_ms` を始まりからの時刻で 10 ミリ秒に丸める（合計は保つ。125 × 4 → 130・120・130・120。変えたら警告）。
  20 ミリ秒より短いと多くのブラウザは 100 ミリ秒として表示する（警告）
- 同じ絵が続くフレームは 1 枚にまとめて時間を足す。256 色（透明のとき 255 色）を超えたら GIF は書かない。動きを作らなかったときも書かない

## 動きの表（`--motion-spec`・`--write-motion-spec`）

動きを「行 = フレーム、列 = 動かし方」の表（`.motion.json`）で書く。プリセットはこの表を埋めて同じ仕組みで作る。設計は [sprite-app-plan.md](sprite-app-plan.md) の「動きの表（タイムライン）の設計」。

| オプション | 内容 |
| --- | --- |
| `--motion-spec FILE` | 表で動きを作る。シートは `<出力名>_<表の name>_sheet.png` / `.json` |
| `--write-motion-spec FILE` | プリセット（か読んだ表）が埋めた表を書き出す（シートも書く）。作れたときは範囲ごとに実際に使った高さと線 `"used"` が付く。入力が複数なら `<名前>_<出力の名前>.json` |
| `--spec-height number\|percent` | 書き出す高さを数（このキャラ用、既定）か割合（別のキャラにも使う）にする |

```bash
python -m apps.sprite chara.png -o out/sprite/chara.png --motion breathe --breathe-split 12 --write-motion-spec breathe12.motion.json
python -m apps.sprite chara.png -o out/sprite/chara.png --motion sway --write-motion-spec sway.motion.json --spec-height percent
python -m apps.sprite chara.png -o out/sprite/chara.png --motion-spec tests/motion_specs/bounce.motion.json
python -m apps.sprite chara.png -o out/sprite/chara.png --motion-spec apps/sprite/examples/bounce_stretch.motion.json --gif --gif-scale 2 --gif-background "#203040"
```

表の例（弾む相当。`tests/motion_specs/bounce.motion.json`）:

```json
{
  "format": "sprite-motion",
  "version": 1,
  "name": "bounce",
  "loop": true,
  "tracks": {
    "root_y": {"type": "root", "axis": "y"}
  },
  "frames": [
    {"ms": 150},
    {"ms": 150, "root_y": -1},
    {"ms": 150, "root_y": -2},
    {"ms": 150, "root_y": -1}
  ]
}
```

- **範囲**（`regions`。どこを動かすか）: 名前を付けていくつでも書ける
  - 分け目（`"type": "split"`）: 分け目から上が全部動く。`"at"` は `"auto"`・整数（出力の画像の下から何ドット目）・`"35.0%"`（透明を除いたキャラの高さに対する、
    キャラのいちばん下の行からの割合）。`"range"` は分け目が at から離れてよい量（0〜8、省くと 2）。`"lines"` を書くとその線をそのまま使う（幅が合わなければエラー）
  - 帯（`"type": "band"`）: `"bottom"`〜`"top"` の行の間**だけ**が動く
  - 四角（`"type": "rect"`）: `"left"`〜`"right"`・`"bottom"`〜`"top"` で囲んだ中**だけ**が動く
  - 帯・四角の端は端の行・列を含む。高さは `at` と同じ、横は左から何ドット目（1 = いちばん左）か、透明を除いた**キャラの幅**に対する左の端からの割合（`"15%"`）。
    `"range"`（縦の変形の継ぎ目を範囲の下の端から何ドット以内で選ぶか）・`"keep_small_parts"` も書ける。`"lines"`・`"at"`・`"aim"` は書けない
  - 範囲は足元（下から 12%、2 ドット以上）を含められない（含めると作らずに警告し、使える下の端を添える）
- **列**（`tracks`。どう動かすか）。値は −2〜+2（`root` だけ ±3）
  | 種類 | 意味 | 値 |
  | --- | --- | --- |
  | `move`（`axis: "x"`） | 範囲を横にずらす（横揺れ）。帯・四角では中身を形のまま横にずらす | + が右 |
  | `move`（`axis: "y"`。帯・四角だけ） | 範囲の中身を形のまま縦にずらす | + が下 |
  | `deform`（`axis: "y"`） | 範囲を縦に縮める（呼吸）・伸ばす（伸び） | − が縮める、+ が伸ばす |
  | `root`（`axis: "y"`。範囲なし） | 絵をまるごと縦にずらす（浮遊・弾む） | + が下。**足元が動いてよいのはこの列だけ** |
- **行**（`frames`）: 2〜32 行。`"ms"` は必ず書く。書いていない列は 0。補間は無い（各行がそのまま 1 フレーム）
- 見本: `apps/sprite/examples/head_rect_nod.motion.json`（頭の四角だけでうなずく）、`bounce_head_band.motion.json`（帯の頭の変形＋体全体のずれ）、
  `bounce_stretch.motion.json`（弾む＋着地で縮む・空中で伸びる）

  ```json
  "regions": {"head": {"type": "rect", "left": "15%", "right": "100%", "bottom": "50%", "top": "100%"}},
  "tracks": {"head_y": {"type": "deform", "region": "head", "axis": "y"}},
  "frames": [{"ms": 600}, {"ms": 200, "head_y": -1}]
  ```
- **作れない行が 1 つでもあれば、シートは書かない**。全部の行を確かめてまとめて警告する。行の番号は表と同じ **1 から**（シートの JSON の `index` は 0 から）:

  ```
  警告（chara.png）: 動き「broken」のシートを作りませんでした（作れない行が 3 行あります。問題 2 件）
  警告（chara.png）: 2・4 行目（upper_y = −1、範囲 upper）: 下から 3 ドット目では分けられません（足元の 4 行は動かさないため、この絵で指定できるのは 5〜29）
  警告（ai04.png）: 2 行目（upper_y = −1、範囲 head）: 1 ドット縮めると輪郭の輪が切れるか、絵の中に隙間ができます。近くで作れる範囲: 下 22〜上 44・左 5〜右 34、…
  ```
- 表の書き間違い（知らない列・範囲外の値・version が違うなど）は変換の前に全部まとめてエラー（終了コード 2）。シートの JSON を渡すと「これは動きの表ではありません」。
  画像に合わない高さは「エラー（ファイル名）: …」（終了コード 1）
- まだ作れない組み合わせ（エラー）: 1 つの範囲に同じ行で横の位置と縦の変形（または横と縦の位置）、1 つの行で 2 つの範囲を動かす（`root` は範囲ではないので一緒に使える）、
  分け目の範囲に縦の位置、`aim`
- `--motion-spec` は `--motion`・`--motion-frames`・`--motion-duration`・`--breathe-*`・`--sway-*`・`--float-*`・`--bounce-*`・`--nod-*` と一緒に使えない
- `"used"`: 分け目は `at`・`percent`・`from_bottom`・`lines`、帯・四角は端を数にしたもの（`left`・`right`・`bottom`・`top`）と継ぎ目の `lines`。読むときは使わない（目安）。
  同じ線で作りたいときは `used` の `lines` を範囲の `"lines"` に写す（分け目だけ）。伸ばす分の線は `"upper_y+"` のように `+` を付けた名前
- `--motion-spec` のシートの JSON には、`at` が `"auto"` か割合の範囲と帯・四角で実際に使ったものを、いちばん外の `"regions_used"` に足す。
  プリセットの表を書き出して読み直すと、シートの PNG はバイト単位で同じ、JSON も `regions_used` を除いて同じ

## コマが並んだ絵（`--frames`）

横に何コマも並んだ 1 枚の絵（歩き・走りなど）を、1 つのキャラとしてそろえたシートにする。

```bash
python -m apps.sprite walk.png -o out/sprite/walk.png --frames --frame-count 4 --frame-name walk --gif
python -m apps.sprite front.png -o out/sprite/front.png --frames --frame-count 3 --frame-order 1,2,3,2 --frame-name walk --mirror
python -m apps.sprite run.png -o out/sprite/run.png --frames --frame-count 4 --align-y match --frame-duration 100 --frame-root-y=0,-1,0,-1 --frame-name run --flicker
python -m apps.sprite 歩き.png --frames --frame-count 4 --same-as-first 0,0,53,53
python -m apps.sprite 走り.png --frames --frame-count 4 --align-y match --same-as-first 0,0,62,54 --same-search 4 --steady on
python -m apps.sprite 正面歩き.png --frames --frame-count 3 --frame-order 1,2,3,2 --edits ai08_sprite_frames_edits.json --piskel --gif
python -m apps.sprite dance1.png dance2.png dance3.png --frames --join -o out/sprite/dance.png --frame-name dance --flicker --gif
python -m apps.sprite jump.png --frames --frame-count 4 --align-y keep --frame-name jump --gif
python -m apps.sprite spin.png --frames --frame-count 3 --frame-order 1,2,3,2r --frame-name spin --gif
```

| オプション | 内容 |
| --- | --- |
| `--frames` | コマが並んだ絵として変換する |
| `--join` | 入力のファイル（何枚でも。フォルダなら中の画像を名前の順に）を、書いた順につなげて 1 つの動きにする（下の「別々のファイルの絵をつなげる」）。無いときは今までどおり 1 枚ずつ別々に変換する |
| `--join-match-height` | `--join` で、キャラの高さをいちばん小さいファイルに合わせる（違いが 1 割までのとき）。付けないときは合わせず、高さの違いを警告するだけ |
| `--frame-count N` | コマの数を確かめる（`--join` では全部のファイルの合計）。合わなければ作らずに警告し `<出力名>_<名前>_skipped.json`（`code: frame_count`）を書く（終了コード 1） |
| `--align` | 横のそろえ方。`match`（既定。上から 3 割の形が 1 コマ目といちばん重なる位置）/ `head`（上の方の横の中心）/ `center` / `none` |
| `--align-y` | 縦のそろえ方。`foot`（既定。足元）/ `match`（上の方の形がいちばん重なる高さ。走り・跳び向け）/ `keep`（元の絵の高さを保つ。ジャンプなど宙に浮くコマ。下の「宙に浮くコマ」）/ `none` |
| `--frame-shift-y DY,...` | 元のコマごとに、そろえた位置から上げ下げする（ドット。− が上）。数は元のコマの数（`--join` では合計）。− で始まるときは `--frame-shift-y=0,-3,-6,0` |
| `--frame-grid` | グリッドの線の並び。`auto`（既定。形の似たコマだけ共通）/ `each`（コマごと。以前の動き）/ `shared`（いつも共通） |
| `--frame-order 1,2,3,2` | 再生の順番（元の絵の左から 1, 2, …。同じコマを何度使ってもよい）。番号の後に `r` を付けたコマは左右に裏返して使う（`1,2,3,2r`。下の「コマごとの左右反転」） |
| `--frame-duration MS[,MS...]` | 表示時間（既定 150） |
| `--frame-name NAME` | ファイル名と JSON の `motion`（既定 `frames`） |
| `--frame-root-y DY,...` | フレームごとに体全体を上下にずらす（− が上。表の `root_y` と同じ）。− で始まるときは `--frame-root-y=-1,0,-1,0` |
| `--mirror` | 左右反転したシート `<出力名>_<名前>_mirror_<幅>x<高さ>_<N>f.png / .json` も書く（`--gif` なら GIF も） |
| `--flicker` / `--flicker-rows N` | ちらつきの見える化 `<出力名>_<名前>_flicker.png / .json`（上から半分の行で 1 コマ目と違うドットに赤紫の枠。自動では直さない） |
| `--steady off\|on` | ちらつきを減らす（既定 off）。上の方の小さな違いを 1 コマ目の色に戻す |
| `--steady-size N` / `--steady-rows N` | 戻す点の大きさ（既定 4。細い線はその 3 倍まで）/ 戻す範囲の行（既定は高さの半分） |
| `--same-as-first X,Y,W,H` | 範囲（四角）を 1 コマ目と同じにする（何回でも。座標はそろえたコマの 1 コマ目の上、左上が 0,0） |
| `--same-band Y0,Y1` | 範囲（帯。上の行と下の行、両端を含む） |
| `--same-search N` | 範囲の位置を探す幅（上下左右 N ドット以内。既定 3） |
| `--same-from K` | 元にするコマ（既定 1） |
| `--edits JSON` | ブラウザ版の「ドットを直す」で保存した `<出力名>_<名前>_edits.json` を当てる。コマの大きさ・数・パレットが合わなければ作らずに理由を出す。下の色が変わったドットは数を警告する |
| `--swaps JSON` | ブラウザ版の「差し替えを描く」で保存した `<出力名>_<名前>_swaps.json` で、シートの中にまばたきを入れる（下の「差し替え」） |
| `--piskel` | シートと同じ名前の `.piskel` も書く（Piskel の Import → Load .piskel file で開ける。速さは 1〜24 FPS の整数 1 つなので、いちばん多い時間から決める） |

- 出力: `<出力名>.png`（そろえたコマを元の順に並べた絵）と `<出力名>_<名前>_<幅>x<高さ>_<N>f.png / .json`（幅・高さは 1 フレーム、N はフレームの数。例 `ai07_walk_53x91_4f.png`）。
  名前で Piskel の「Import as spritesheet」に入れる幅・高さが分かる。1 枚の絵の動きのシートの名前は `<出力名>_<動き>_sheet.png` のまま
- 順番: 変換 → 1 コマ目と同じにする範囲 → ちらつきを減らす → 手の直し → 体全体の上下・シート → 差し替え（まばたき）
- 範囲を写さないとき（警告と次に試せることを出す）の code: `low_match`・`seam`・`hole`・`foot`・`small_part`・`intrusion`・`edge`・`empty`。写さなかった範囲は `skipped`（`stage: same_as_first`）にも入る。
  範囲を使うと図 `<出力名>_<名前>_same.png`（緑の枠 = 写した位置、赤 = 写さなかった、橙 = 境目の問題、青 = 残した部品）も書く
- `--mirror` の JSON は `pivot.x` も反転し、`source.mirrored` が true。`--gif` は `<出力名>_<名前>_<幅>x<高さ>_<N>f.gif`（幅・高さは 1 倍の大きさ）
- 使えないもの: `--motion`・`--motion-spec`（体全体の上下は `--frame-root-y`）、`--input-type illust`、`--grid-origin`

### 宙に浮くコマ（ジャンプ。`--align-y keep`・`--frame-shift-y`）

足元でそろえると、跳んだコマも足が地面に着いてしまう。`--align-y keep`（画面の「元の絵の高さを保つ」）は、コマごとに足元をそろえず、
**元の絵での上下の位置**（ドットの格子に乗せたあとの位置）をそのまま使う。

- 1 つのファイルの中では元の絵の上下のまま。ファイルが何枚か（`--join`）のときは、ファイルどうしをそれぞれの**いちばん低い足元**でそろえる
- 自動でうまくいかないとき（AI がコマごとに違う高さに描いたなど）は、`--frame-shift-y`（画面の「コマごとの上げ下げ」）で元のコマごとに上げ下げする（− が上）。
  どのそろえ方でも使え、そろえた後に足す。数は元のコマの数と同じ（順番の数ではない。順番ごとに体を上下させるのは `--frame-root-y`）。
  指定できるのは −256〜256 ドット（キャラの高さより十分大きい。シートは上げ下げの分だけ広がるので、大きすぎる絵にならないように上限を置いた。超えると理由と範囲を表示する）
- 直しの一覧（edits.json）: `keep` か上げ下げを使ったときは、ファイルが 1 つでも各コマを置いた位置（`origin`）を書く。上げ下げを変えて変換し直しても、
  置いた位置の違いだけドットをずらして、キャラの同じ所に付ける（`--edits` でも画面でも）。ずらすとキャンバスの外に出るときは当てない（画面では確認の窓）。
  画面では、変えたのが上げ下げ・そろえ方だけのときに付け直す（色数などを変えるとコマの絵の大きさが変わることがあり、位置だけでは合わせられないため。そのときは今までどおり）
- キャンバスは、いちばん高く跳んだコマが入る高さに広がる（全部のコマが同じ大きさのまま）。横のそろえ方は今までどおり
- JSON に**地面の線** `ground_y`（いちばん低い足元の高さ。`pivot.y` と同じ）と、フレームごとの `air`（足元が地面の線から何ドット上か。着いていれば 0）を書く。
  Godot では、AnimatedSprite2D の `offset.y` を `-ground_y`（と `centered` を切）にすると、地面の線がノードの位置に来る。影は `air` を見て小さくするなど
- `keep` も上げ下げも使わないときの出力は今までと同じ
- 比較画像: ai13（ジャンプ 4 コマ）で、3 コマ目が地面から 27 ドット、2 コマ目が 10 ドット浮く。シートは 58×94（足元でそろえると 56×68）

### コマごとの左右反転（回る動き。`--frame-order 1,2,3,2r`）

正面・横・後ろの 3 コマから「回る」動きを作るとき、反対の横向きは横向きのコマを裏返して使える。順番の番号の後に `r`（大文字でもよい）を付けると、そのコマを左右に裏返す。

- 裏返す軸は、そろえた基準（横のそろえ方の基準。JSON の `pivot.x`）。キャラは横にずれない。基準の左右で同じ幅になるように、全部のフレームの左右に透明の列を足す
  （JSON の `padding.left`・`padding.right`。`pivot.x` は足した分だけ右に）。シートの大きさは全部のフレームで同じ
- JSON の裏返したフレームに `"flipped": true`（`source_frame` は元のコマの番号のまま）
- **左右で違う物（リボン・持ち物など）は反対側に移る**。r を使うと注意を出す。直したいときは、画面の「ドットを直す」で裏返したコマ（`2r`）を開いて直す。
  元のコマの直しは裏返したコマにも裏返して付き、裏返したコマで直したドットは裏返したコマだけに入る（下の「edits.json の形」）
- まばたき・口パク（`--swaps`）は、裏返したコマには裏返して当てる（位置を探すのも、裏返し直した向きで）
- **シート全体の左右反転（`--mirror`）と一緒に使うと**: 反転したシートは、r のコマも含めて全部のフレームを反転したもの（r のコマは元の向きに戻る）。
  `pivot.x` は同じ（基準の左右で同じ幅なので）、`padding.left` と `right` が入れ替わる。`frames[].flipped` はそのまま（「元のコマを裏返したか」の印）
- 重ね（`--fx`）・`.piskel`・読み戻し（`--readback`）でもそのまま使える。読み戻しで「同じ元のコマ」を比べる・そろえる（`--same-frames first`）ときは、
  裏返したコマと裏返していないコマを別のコマとして扱う
- r を使わないときの出力は今までと同じ

### 別々のファイルの絵をつなげる（`--join`）

長い動き（ダンスなど）を AI に何枚かに分けて描かせたときに、ファイルを書いた順につなげて 1 つの動きにする。1 枚の中のコマは 1 つでも何コマでもよい。

- コマの番号（`--frame-order`・`--same-from`・直し・ちらつき）は、1 つ目のファイルの左のコマから 1, 2, … と全部のファイルを通して数える
- グリッド（ドットの大きさ・線の位置）はファイルごとに見つけ、パレットは全部のファイルで 1 つ（`--colors`）、大きさと足元（`--align-y match` なら頭の高さ）は全部のコマでそろえる
- `--frame-grid auto` で線の並びを共通にするのは、2 コマ以上のファイルの全部で共通にできるときだけ（どれかが合わなければ全部コマごと）
- キャラの高さ（ファイルの中でいちばん高いコマ）は、**既定では合わせない**。いちばん小さいファイルより 2 ドット以上高いファイルがあれば警告する（`skipped` の `join_height`）。
  「いちばん小さいファイルに合わせる」ので、しゃがみ・ジャンプのコマだけのファイルが混ざると、ほかのファイルが黙って小さくされてしまうため（迷ったら何もしない）
- `--join-match-height` を付けると、違いが 1 割までのファイルは、ドット絵に戻すときの格子をファイルごとに少し大きくして（等間隔の格子にする）、いちばん小さいファイルの高さに合わせる。
  完成したドット絵の拡大・縮小はしない。1 ドットの違いは合わせない。違いが 1 割より大きいときは付けても合わせずに警告する（AI に描き直させる前提）
- つなぎ目のコマは自動で混ぜない・中割りを作らない。境目（くり返すときの最後 → 最初も）で体の位置（上の方の横の中心・足元・頭の上）が、ファイルの中の隣のコマどうしのずれの最大より大きく、
  3 ドット以上ずれるときは警告する（`skipped` の `join_shift`）。`--flicker` の画像では、ファイルの境目のすき間を橙に塗り、各ファイルの 1 コマ目の上に f1・f2… と書く
- 出力名は 1 つ目のファイルの名前（`-o` で変えられる）。表示に、ファイルごとのコマの数・何コマ目から・ドットの大きさ・高さ（合わせたときは格子を何倍にしたか）を出す
- ファイルが 1 つのとき（`--join` なしと同じ）の出力は、今までとバイト単位で同じ
- 同じ名前のファイルが 2 つ以上あるときは、2 つ目から `名前#2` のように区別する（直しの一覧・JSON の記録の名前）
- 比較画像: `tools/compare_sprite_frames.py`・`compare_sprite_steady.py`・`compare_sprite_same.py`（`--frame-count 4` など）

### edits.json の形（「ドットを直す」の直しの一覧）

```json
{"version": 3, "kind": "sprite_edits", "name": "spin", "frame_width": 77, "frame_height": 95, "frame_count": 3,
 "palette": ["#rrggbb", ...], "dot_count": 3,
 "frames": [{"source_frame": 2, "dots": [[x, y, 色の番号 か null, 直したときの下の色]]},
            {"source_frame": 2, "mirrored": true, "dots": [[x, y, 色の番号 か null, 下の色]]}]}
```

- `source_frame` は元のコマの番号（1 から）。順番 1,2,3,2 の 2 番目と 4 番目は 1 回直せば両方に入る
- `"mirrored": true` の項目は、**裏返したコマ（順番の r）だけの直し**。座標は、元のコマを左右に裏返した絵（大きさは元のコマと同じ。左上が 0,0）の上。
  元のコマの直し（`mirrored` の無い項目）は、裏返したコマにも裏返して付く。同じコマでは、元のコマの直しの後に置く
- 版: 裏返したコマの直しがあれば 3、つなげたとき（`--join` で 2 つ以上のファイル）は 2（`files`・`file`・`file_frame`・`origin`）、ほかは 1。
  ファイルが 1 つでも、縦のそろえ方 `keep` か上げ下げを使ったときは `origin`（そのコマの絵を置いた位置）を書く（版は変えない）。
  前の形（版 1・2。`mirrored` が無い）もそのまま読める。つなげたときの裏返したコマの `origin` は、裏返した絵の中でそのコマの絵を置いた位置

## 動きに名前を付けてまとめる（`--anims`）

「立ち」「歩き」「走り」「ジャンプ」などを **1 つのシート（1 つの動き = 1 行）** にまとめ、JSON の `animations` に動きの名前とフレームの範囲を書く。
ゲームのエンジン（Godot の SpriteFrames など）で、1 つのキャラに idle・walk・run などのアニメーションを持たせて切り替えるとき用。
「つなげる（`--join`）」を広げたもの（処理は `retrocore/sprite/anims.py`）。画面では「コマが並んだ絵」の「つなげるファイル」の **動きに名前を付けてまとめる** に印を付ける。

```bash
python -m apps.sprite --anims hero.anims.json -o out/sprite/hero.png --gif --piskel
python -m apps.sprite idle.png walk.png jump.png --anims hero_sprite_anims.anims.json --gif --piskel   # 画面で保存した設定と同じ画像
python -m apps.sprite hero_walk.piskel --readback hero_anims_62x94_12f.json --original hero_anims_62x94_12f.png --gif --mirror --piskel
```

設定のファイル（画面の「CLI 用の設定を保存（.anims.json）」で保存できる。自分で書いてもよい。まとめたシートの情報（`animations` の入った JSON）は「⑤ まとめたシートの情報を保存」の方）:

```json
{"version": 1, "kind": "sprite_anims", "name": "anims",
 "settings": {"colors": 16, "cleanup": "normal", "outline": "auto"},
 "common": {"align": "match", "grid": "auto", "match_height": false, "mirror": true},
 "anims": [{"name": "idle", "files": ["idle.png"]},
           {"name": "walk", "files": ["walk1.png", "walk2.png"], "order": "1,2,3,4,3,2", "duration_ms": 150},
           {"name": "jump", "files": ["jump.png"], "align_y": "keep", "shift_y": null, "loop": false}]}
```

| 項目 | 内容 |
| --- | --- |
| `anims[]` | 動きの並び（シートの行の順）。`name` は英小文字で始まり、英小文字・数字・_ だけ（32 文字まで。重ならないこと）。`files` はその動きのファイル（並べた順につなげる）。**1 つのファイルは 1 つの動きにだけ**使える |
| `order` | その動きの中での順番（その動きのファイルの左のコマから 1, 2, …。`1,2,3,2` や `1,2,3,2r`）。無ければ 1, 2, … |
| `duration_ms` | 1 コマの時間（ミリ秒。既定 150）。1 つか、順番と同じ数の並び |
| `loop` | くり返す（既定 true）。false は 1 回で終わる（GIF も 1 回だけ再生。最後のコマで止めるのはゲームの側） |
| `align_y` / `shift_y` | 縦のそろえ方（`foot` 既定・`match`・`keep`・`none`）と、元のコマごとの上げ下げ（数はその動きのコマの数） |
| `common` | 全部の動きで共通: 横のそろえ方（`--align`）、グリッドの線の並び（`--frame-grid`）、高さ合わせ（`--join-match-height`）、左右反転（`--mirror`）、キャンバスを倍数にそろえる（`canvas_multiple`・`canvas_align`。そろえるときだけ書く。`--canvas-multiple`）。CLI で付けたものが優先 |
| `settings` | 変換の設定（色数など。画面で保存したときに入る）。あるときは変換の引数（`--colors` など）は使えない（エラー）。無ければ引数 |
| `name` | シートのファイル名と JSON の `motion`（既定 `anims`） |

- 入力の画像は、引数に書けば名前で探す（設定に無い画像があればエラー）。書かなければ設定のファイルのあるフォルダから探す。出力名は 1 つ目のファイルの名前（`-o` で変えられる）
- **全部の動きで共通**: パレット（全部のコマから 1 つ）、コマの大きさ（いちばん大きいものに合わせて透明で広げる。拡大・縮小はしない）、
  地面の線 `ground_y`（全部の動きでフレームの下端）。グリッドは今までどおりファイルごとに見つける
- 動きごとにそろえてから、そろえた基準（pivot の横の位置）と下端が全部の動きで同じ所に来るように透明の列・行を足す。
  基準が半ドットの所にある動きと整数の所にある動きが混ざると、半ドットはそろえない（ドットを動かさないため）
- 出力: `<出力名>.png`（そろえたコマをファイルの順に並べた絵）、`<出力名>_<名前>_<幅>x<高さ>_<N>f.png / .json`（まとめたシート。N は全部の動きのフレームの数）、
  `--mirror`（か `common.mirror`）で `<出力名>_<名前>_mirror_…`、`--gif` で動きごとの `<出力名>_<動き>.gif`（反転は `<出力名>_<動き>_mirror.gif`）、
  `--piskel` で動きごとの `<出力名>_<動き>.piskel`（直す用）
- シートの PNG は 1 つの動き = 1 行。横はいちばんコマの多い動きに合わせ、余りは透明。JSON の `columns`・`rows` に列と行の数
- 直しの一覧（`--edits`）は今までどおり「どのファイルの何コマ目か」と置いた位置（`origin`）で書く。動きの並び・順番を変えても、直したドットは同じ絵の同じ所に付く
- **使えないもの**（警告して無視する）: まばたき・口パク（`--swaps`）、重ね（`--fx`）、1 コマ目と同じにする範囲（`--same-as-first`・`--same-band`）、
  ちらつきを減らす（`--steady on`）、`--flicker`、`--frame-root-y`。動きごとの設定の引数（`--frame-order`・`--frame-duration`・`--frame-count`・`--align-y`・
  `--frame-shift-y`・`--frame-name`・`--join`）は使えない（エラー。設定のファイルに書く）
- 「1 枚の絵」の動き（呼吸など）は入れられない。idle は 1 コマのファイルで作る
- 画面と CLI の出力は、同じ画像・同じ設定のファイルならバイト単位で同じ（`tests/test_sprite_anims.py`）

### まとめたシートの JSON

コマのシートの JSON（下の「シートの JSON」）と同じ項目に、次を足す:

```json
"animations": {"walk": {"frames": [0, 1, 2, 3], "duration_ms": 150, "loop": true, "row": 0},
               "jump": {"frames": [4, 5, 6, 7], "duration_ms": 150, "loop": false, "row": 1}},
"frames": [{"index": 0, "x": 0, "y": 0, …, "source_frame": 1, "air": 0, "animation": "walk"}, …],
"source": {…, "anims": [{"name": "walk", "files": ["walk.png"], "first_frame": 1, "frame_count": 4, "order": "1,2,3,4",
                         "align_y": "foot", "shift_y": null}, …]}
```

- `animations.<名前>.frames` はシート全体での通し番号（`frames[]` の `index`）。`duration_ms` はその動きでいちばん多い時間（コマごとの時間は `frames[].duration_ms`）。
  `row` はシートの行（0 から）
- `ground_y`・`frames[].air` はいつも書く（足元でそろえた動きは `air` が 0）。`source.frame_count` は全部のファイルのコマの数、`source_frame` は全部のファイルを通した番号
- **Godot**: AnimatedSprite2D の SpriteFrames で、`animations` の名前ごとにアニメーションを足し、「スプライトシートから追加」で横 `columns`・縦 `rows` に分けて、
  その動きの `frames` の番号のコマを選ぶ（行 `row` のコマ）。速さ（FPS）は `1000 / duration_ms`、ループは `loop`。コマごとの時間が違うときは、
  各フレームの「長さ」を `frames[].duration_ms / duration_ms` にする。`offset.y` を `-ground_y`（`centered` を切）にすると地面の線がノードの位置に来る
  （プロジェクト・取り込みの設定と、JSON から SpriteFrames を作る GDScript の例は [apps/sprite/README.md](../apps/sprite/README.md) の「ゲームで使うときの設定（Godot 4）」）

### まとめたシートの読み戻し（`--readback`・`--anim`）

Piskel では動きごとの `.piskel` を直す。読み戻しは 1 つの動きずつ:

- 入力は直した `.piskel`（か PNG。その動きだけのシート）、`--readback` はまとめたシートの JSON、`--original` は**直す前のまとめたシート（要る）**
- どの動きかは、直したファイルの名前（`<出力名>_<動き>.piskel`。`_edited` や「 (1)」が付いていてもよい）から決める。分からなければ `--anim walk` で選ぶ（画面は「D. 直した動き」）
- 作るもの: まとめたシート（その動きの行を直したもの）と JSON（`<元の名前>_edited.png / .json`。`edited.animation` に動きの名前）、
  `--mirror` で反転のまとめたシート、その動きの GIF・.piskel・確かめた結果（`<出力名>_<動き>_edited.gif` など）。続けて別の動きを直すときは、
  `_edited` のシートと JSON を元にすればよい（名前は `_edited` のまま）
- 確かめること・作らないときは、ふつうの読み戻しと同じ

## 差し替え（まばたき。`--swaps`）

目を閉じた顔を 1 回描くだけで、動きの中にまばたきを入れる。`--motion`・`--motion-spec`・`--frames` と一緒に使う。

```bash
python -m apps.sprite chara.png --motion breathe --gif --swaps chara_sprite_breathe_swaps.json
python -m apps.sprite 正面歩き.png --frames --frame-count 3 --frame-order 1,2,3,2 --mirror --gif --swaps 正面歩き_sprite_frames_swaps.json
```

- 一覧のファイルはブラウザ版の「差し替えを描く」で保存する（同じファイルなら画面と CLI の出力はバイト単位で同じ）。1 コマ目（1 枚の絵は変換した絵）と違うドットだけが入る:

```json
{"version": 1, "kind": "sprite_swaps", "name": "frames", "frame_width": 66, "frame_height": 92,
 "palette": ["#rrggbb", ...], "interval_ms": 3000, "close_ms": 120, "half_ms": 60, "search": 3,
 "swaps": [{"type": "blink", "rects": [{"x": 22, "y": 39, "w": 2, "h": 1}, {"x": 40, "y": 39, "w": 2, "h": 1}],
            "closed": [[22, 39, 13, 0], ...], "half": [[x, y, 色の番号 か null, 1 コマ目の色の番号 か null], ...]}]}
```

- `closed`（閉じた目）はいる、`half`（半分閉じた目）はなくてもよい。あれば 開く → 半分（`half_ms`）→ 閉じる（`close_ms`）→ 半分 → 開く
- 動き 1 周が `interval_ms`（500〜20000）より短いときは、動きを何周か繰り返したシートにして 1 回だけ入れる（周の数 = 間隔 ÷ 1 周 を四捨五入）。シートの名前のフレームの数も増える
- まばたきはシートの真ん中あたりのフレームの始まりから入れ、かかるフレームを時間で分ける（全体の時間は変えない。20 ミリ秒より短い切れ端は作らない）
- `rects` は違うドットの離れた固まりごとの四角（2 ドット以内のドットは同じ固まり。左右の目は別々）。書き出すときだけ付け、読むときはドットから決め直す
- 各フレームで頭の位置を上下左右 `search` ドット以内で探す（四角が何個あってもずれは 1 つ）。次のフレームには入れず、どのフレームがなぜ入らなかったかを警告する
  （入れられるフレームから始める。迷ったら入れない）: 合い方 0.7 未満（`low_match`）・キャンバスの外（`edge`）・足元が変わる（`foot`）・
  四角の境目をまたいで 1 コマ目と違う部品がある（`crossing`。コマごとに動く前髪・手など）・外から入ってきた太い部品がある（`intrusion`）・
  置き換えると四角の境目で段差・輪郭の途切れができる（`seam`。`intrusion`・`seam` は `--same-as-first` と同じ確かめ）。
  どのフレームにも入れられなければ、元のシートのまま警告する
- 1 コマ目の大きさ・パレットが描いたときと違うと、作らずに理由を出す（終了コード 1）。1 コマ目の色が描いたときと違うドットは数を警告する
- シートの JSON に `swaps`（`interval_ms`・`cycle_ms`・`repeats`・`items`。`items[]` は `type`・`rects`・`shapes`・`applied`・`start_ms`・`duration_ms`・`frames`（入れたフレームの番号。0 から）・
  `placements`（元のフレームごとの `shift`・`match`・`used`）・`frames_not_used`（入れなかったフレームと `code`））と、入れたフレームの `frames[].swap`（`{"type": "blink", "shape": "closed"}`）を足す。
  `.piskel` は速さを 1 つしか持てないので、短いフレームもいちばん多い時間で再生される（本当の時間は JSON）。読み戻しでは、まばたきのフレームを「同じ元のコマが違う」として数えない

### 口パク（しゃべりのシート）

一覧に `{"type": "talk", "open": [...], "half": [...]}`（`open` = 開いた口。いる、`half` = 半分開いた口。なくてもよい）があると、
**ふつうのシートとは別に「しゃべりのシート」も書く**（ふつうのシートは今までと同じ。まばたきだけが入る）。閉じた口は 1 コマ目のまま。

```bash
python -m apps.sprite chara.png --motion breathe --gif --swaps chara_sprite_breathe_swaps.json   # ふつうのシート＋しゃべりのシート
python -m apps.sprite chara.png --gif --swaps chara_sprite_talk_swaps.json                       # 動きなし: 口だけ動く
```

- 出力: 1 枚の絵は `<出力名>_<動き>_talk_sheet.png` / `.json`（`--gif` なら `<出力名>_<動き>_talk.gif`）。`--motion` なし（口パクがある一覧のときだけ使える）は
  `<出力名>_talk_sheet.png`。コマが並んだ絵は `<出力名>_<名前>_talk_<幅>x<高さ>_<N>f.png`（`--mirror` なら `<名前>_talk_mirror`、`--piskel`・`--gif` も同じ名前）
- 口の動き: 半分の口が無ければ 閉じる → 開く、あれば 閉じる → 半分 → 開く → 半分 のくり返し。決まった順番で、乱数は使わない。
  切り替えの時間は一覧の `talk_ms`（40〜1000。`null` は自動で、半分なし 150・半分あり 100 ミリ秒）。`talk_ms` は口パクがあるときだけ書く（まばたきだけの一覧は前と同じ）
- 周の数: シート全体の時間が口の 1 回りでちょうど割り切れるように動きを繰り返す（まばたきもあれば、まばたきの周の数から探す。20 秒まで）。
  割り切れなければ、口の切り替えの時間を 2% まで伸び縮みさせる（どちらでも、シートの終わりと始まりで口がつながる）。口・まばたき・元のフレームの境目で時間を分け、20 ミリ秒未満の切れ端は作らない
- まばたきも入る（ふつうのシートと同じ決め方）。目と口の四角が重なるフレームは、口を優先してまばたきを入れずに警告する（`overlap`）
- 口の当て方はまばたきと同じ（±`search` ドットで探す。`low_match`・`edge`・`foot`・`crossing`・`intrusion`・`seam` のフレームは閉じた口のまま警告）。
  入らないフレームが半分以上なら、しゃべりのシートは書かずに、理由と次に試せること（口のまわりまで広く描く・動きを小さくする・動きなしにする）を警告する（終了コード 0。ふつうのシートは書く）
- JSON の `swaps` は `talk_ms` と、`items[]` に口パク（`type: "talk"`・`step_ms`・`pattern`・`steps`・`step_ms_used`・`frames`・`placements`）とまばたきの記録。
  `frames[].swap` は、そのフレームに入れた形（`{"talk": "open", "blink": "closed"}`。どちらかだけのこともある）

## 重ね（漫画的エフェクト。`--fx`）

！・汗・ハート・Zzz・？・怒りマーク・キラキラなど 14 種を、キャラの頭の上・頭の横・顔の上半分・体のまわりに重ねる。**キャラのシートは今までと同じ**で、エフェクトは別の小さなシート（PNG と JSON）にする。
`--motion`・`--motion-spec`・`--frames` と一緒に使う。一度に選べるのは 1 つ。`--fx` を付けないときの出力は今までとバイト単位で同じ。

```bash
python -m apps.sprite chara.png --motion breathe --fx surprise --gif          # ！（驚き）。重ねた GIF も
python -m apps.sprite chara.png --motion breathe --fx sweat --fx-offset=1,-2 --fx-end hide
python -m apps.sprite walk.png --frames --mirror --piskel --fx sleep         # 左右反転用と、2 レイヤーの .piskel も
```

| オプション | 内容 |
| --- | --- |
| `--fx NAME` | 下の表の 14 種から 1 つ |
| `--fx-offset X,Y` | 自動の置く場所からずらす量（ドット。右と下が +。±64 まで）。− で始まるときは `--fx-offset=-1,2` |
| `--fx-end keep\|hide` | 1 回で終わるもの（surprise・sweat・idea・silence）の終わり方。`keep` = 最後の形で止めて残す（既定）、`hide` = 最後に透明のコマを足して消す |

| 名前 | エフェクト | 置く場所 | 終わり |
| --- | --- | --- | --- |
| `surprise` | ！（驚き） | 頭の上 | 1 回 |
| `sweat` | 汗（たらり） | 頭の横 | 1 回 |
| `heart` | ハート | 頭の上 | くり返し |
| `sleep` | Zzz | 頭の横 | くり返し |
| `question` | ？（疑問）。左右にゆっくり揺れる | 頭の上 | くり返し |
| `anger` | 怒りマーク。大小を交互にドクドク | 頭の横 | くり返し |
| `music` | 音符。揺れながら上へ | 頭の横 | くり返し |
| `sparkle` | キラキラ。2 か所ずつ星がちかちか | 体のまわり | くり返し |
| `pale` | 青ざめの縦線。下の 2 行がゆっくり消えて戻る | 顔の上半分 | くり返し |
| `idea` | 電球（ひらめき）。ポンと出て光が 2 回点滅 | 頭の上 | 1 回 |
| `dizzy` | ぐるぐる（混乱）。頭の上を星 3 つが横長の楕円に回る | 頭の上 | くり返し |
| `steam` | 湯気（怒り・暑さ）。左右 2 本がもくもく上へ | 頭の上 | くり返し |
| `silence` | …（沈黙）。点が 1 つずつ増える | 頭の横 | 1 回 |
| `panic` | 大きな汗（あせあせ）。2 粒が交互に飛ぶ | 頭の横 | くり返し |

- 置く場所「顔の上半分」は、目の位置が分からないので頭の範囲からの目安（ずれていたら `--fx-offset` で合わせる）。「体のまわり」は絵全体の四角の
  角・辺のまわりに置く（キャラが大きければ星の間も広がる）。決め方は [how-it-works.md](how-it-works.md) の「重ね」
- 出力: `<キャラのシートの名前>_fx_<名前>.png` / `.json`。キャラのシートの名前から `_sheet`・`_<幅>x<高さ>_<N>f` を除いて付ける
  （`hero_breathe_sheet.png` → `hero_breathe_fx_surprise.png`、`hero_walk_53x91_4f.png` → `hero_walk_fx_surprise_7x12_5f.png`。コマのシートは
  エフェクトの 1 コマの大きさとコマの数を付ける）。`--mirror` なら左右反転のキャラ用も（`hero_walk_mirror_fx_…`）。しゃべりのシートには重ねない
- `--gif` なら、キャラとエフェクトを重ねた見た目の GIF `<…>_fx_<名前>.gif`（確かめ用。長さは、くり返すものはキャラの 1 周とエフェクトの 1 周がそろう長さ
  （12 秒まで）、1 回で終わるものはエフェクト＋1 秒をキャラの周の切れ目まで延ばした長さ）
- `--frames --piskel` なら、レイヤー「キャラ」「エフェクト」に分けた `<…>_fx_<名前>.piskel`。Piskel は速さが 1 つだけなので、全部の時間の最大公約数の長さの
  コマにそろえる（1 秒を割り切れて 24 FPS 以下・256 コマ以下のときだけ。作れなければ警告）
- 置く場所はキャラのシートの 1 コマ目から自動で決める（仕組みは [how-it-works.md](how-it-works.md) の「重ね」）。左右反転のシートでは置く場所も反転する（エフェクトの絵は反転しない）

### エフェクトの JSON

```json
{"version": 1, "format": "sprite-fx", "fx": "surprise", "label": "！（驚き）",
 "image": "hero_breathe_fx_surprise.png", "character_image": "hero_breathe_sheet.png",
 "frame_width": 6, "frame_height": 16, "frame_count": 5, "columns": 5, "rows": 1,
 "loop": false, "end": "keep", "anchor": "top", "mirrored": false, "offset": {"x": 0, "y": 0},
 "position": {"x": 15, "y": -17}, "character_frame": {"width": 34, "height": 44},
 "outline_color": "#462112", "total_duration_ms": 950,
 "frames": [{"index": 0, "x": 0, "y": 0, "width": 6, "height": 16, "duration_ms": 50}, ...]}
```

- `position`: **キャラのコマの左上から見た、エフェクトのコマの左上**（ドット。右と下が +。上や左にはみ出すときはマイナス）。全部のコマで同じ
  （動きはエフェクトのコマの中で付いている）
- `anchor`: 置く場所（`top` 頭の上・`side` 頭の横・`face` 顔（今は使うエフェクトが無い）・`upper_face` 顔の上半分・`around` 体のまわり）。`around` はキャラの大きさでエフェクトのコマの大きさが変わる
- `loop`: false は 1 回で止める。`end` はそのときの終わり方（くり返すものは `null`）。何も見えないコマ（点滅の消える所・`hide` の最後）は `frames[].empty: true`
- `character_frame` はキャラのシートの 1 コマの大きさ、`outline_color` はエフェクトの輪郭に使った色（キャラのシルエットの縁のいちばん暗い色）
- エフェクトのシートは、キャラのシートとは別の長さ・別の時間（それぞれをくり返す）

Godot での使い方: キャラの `AnimatedSprite2D`（`centered` を切る）の子にエフェクトの `AnimatedSprite2D`（同じく `centered` を切る）を置き、
子の `position` を JSON の `position` にする（ドット＝画素。キャラを拡大しているなら同じ倍率が掛かる）。`centered` を入れたままなら、
子の位置は `position + (frame_width, frame_height) / 2 − (character_frame.width, character_frame.height) / 2`。
エフェクトの SpriteFrames は Hframes = `frame_count`、各コマの時間は `duration_ms`、`loop` が false ならループを切る。
キャラの `offset` を変えたときは、子の `position` にも同じ `offset` を足す（`offset` は絵だけをずらし、子は動かさない）。

## 手直ししたシートの読み戻し（`--readback`）

Piskel などで直したシート（PNG か `.piskel`）と元の JSON から、JSON・GIF・左右反転・横 1 列に並べ直したシート・.piskel を作り直す。**直したドットはそのまま使う**。手順は [sprite-edit.md](sprite-edit.md)。

```bash
python -m apps.sprite 直した.piskel --readback walk_walk_53x91_4f.json --gif --mirror --piskel
python -m apps.sprite 書き出した.png --readback walk_walk_53x91_4f.json --original walk_walk_53x91_4f.png -o out/edited
```

| オプション | 内容 |
| --- | --- |
| `--readback JSON` | 元のシートの JSON。時間・順番・pivot・padding・offset_y・source_frame を引き継ぐ |
| `--original PNG` | 直す前のシート。直したドットに印を付けた画像も書く（古い JSON で `palette` が無いときは、この色と比べる） |
| `--fix-colors` | 元のパレットに無い色をいちばん近い元の色に、半透明を透明か不透明（α 128 が境）に直す（既定は直さない） |
| `--same-frames keep\|first` | 同じ元のコマを何度も使う順番で、後のコマを先のコマと同じにする（既定 `keep`） |
| `--anim NAME` | （動きをまとめたシートのとき）直した動きの名前。省くと直したファイルの名前から決める（下の「まとめたシートの読み戻し」は上の `--anims` の節） |

- 一緒に使えるのは `--gif`（`--gif-scale`・`--gif-background`）・`--mirror`・`--piskel`・`--original`・`--same-frames`・`--fix-colors`・`--anim`・`-o`（書くフォルダ。既定 `out/sprite`）・`-q` だけ
- ファイルの名前は元の JSON の `image` に `_edited` を付けたもの（反転は `_mirror` を足すか除く）。確認の画像・記録は `<…>_edited_check.png` / `.json`
- シートの並びは 1 コマの大きさから読む（格子状の書き出しでも左から右・上から下。余ったマスは透明だけなら捨てる。整数倍に拡大されていても 1 倍に戻す）
- **作らないとき**（終了コード 1）: JSON がシートの情報でない、大きさが 1 コマの大きさで割り切れない、コマの数が違う、透明だけのコマがある、.piskel の大きさ・コマの数が違う
- **作って警告するもの**: 元のパレットに無い色（赤紫の枠）、半透明のドット（橙）、同じ元のコマから作ったコマが直したあとで違う（青）
- `.piskel` の読み込み: 見えるレイヤー（不透明度 > 0）を下から重ねる。飛ばす設定のコマもそのまま入れる（警告だけ）。FPS は使わず JSON の時間を使う

## キャンバスを倍数にそろえる（`--canvas-multiple`）

保存する絵の幅と高さを、決まった数の倍数にする（ゲームのエンジンで扱いやすい 8・16・32 など）。**絵は拡大・縮小せず、透明な余白を足すだけ**（`retrocore/sprite/canvas.py`）。

```bash
python -m apps.sprite chara.png -o out/sprite/chara.png --canvas-multiple 16                        # 1 枚の絵（chara.png と chara.json）
python -m apps.sprite walk.png --frames --frame-count 4 --canvas-multiple 16 --gif --piskel        # コマの大きさを 16 の倍数に
python -m apps.sprite jump.png --frames --align-y keep --canvas-multiple 32 --canvas-align center
```

| オプション | 内容 |
| --- | --- |
| `--canvas-multiple N` | 幅と高さを N の倍数にする（1〜256。縦横同じ倍数）。0 か指定なしでそろえない（出力は前とバイト単位で同じ） |
| `--canvas-align bottom\|center` | `bottom`（既定）= 下ぞろえ・左右中央（足元の高さが変わらない）、`center` = 上下左右中央。`--canvas-multiple` と一緒に使う |

- 余りが奇数のときは、右（`center` では下も）に 1 ドット多く足す。例: 45×37・倍数 16・`bottom` → 48×48（左 1・右 2・上 11・下 0）
- 対象: 1 枚の絵（`<出力名>.png`。一緒に `<出力名>.json` を書く）、動きのシート・コマのシート（**1 コマの大きさ**。全部のコマに同じ余白なので、コマどうしの足元はそろったまま）・
  左右反転・しゃべりのシート・まとめたシート（`--anims`。設定のファイルの `common.canvas_multiple`・`canvas_align` でも。引数が優先）。GIF・.piskel もシートと同じ絵になる。
  コマが並んだ絵の `<出力名>.png`（そろえたコマを元の順に並べた絵）は変えない
- 余白は**最後（保存の直前）**に足す。動きはそろえる前の絵で作り、直し（edits.json）・差し替え（swaps.json）は余白を足す前の座標のまま（意味は変わらない）
- JSON: `pivot` は余白の分ずらす（左に足した列を x に、上に足した行を y に足す）。`pivot.y` と同じ `ground_y` を pivot のすぐ後に、足した余白をいちばん外の `canvas` に書く:
  `"canvas": {"multiple": 16, "align": "bottom", "width": 45, "height": 37, "left": 1, "right": 2, "top": 11, "bottom": 0}`（`width`・`height` は足す前の大きさ）。
  シートのファイル名の大きさ（`_48x48_4f`）は足した後の大きさ
- 1 枚の絵の JSON: `{"version": 1, "kind": "sprite_image", "image": "chara.png", "width": 48, "height": 48, "pivot": {"x": 23.5, "y": 48}, "ground_y": 48, "canvas": {…}}`
  （pivot は足す前の絵の下端の中央）
- エフェクトのシート（`--fx`）は大きさを変えず、JSON の `position` を余白の分ずらす（`character_frame` は足した後のキャラのコマの大きさ）
- 画面では「保存」の欄の「キャンバスを倍数にそろえる」（そろえない／8／16／32／任意の数）と「並べ方」。同じ設定なら CLI と出力はバイト単位で同じ（`tests/test_sprite_canvas.py`）
- 読み戻し（`--readback`）では使えない（読み戻すシートの大きさのまま作る）

## 大きさをくらべる（`compare`）

変換した素材どうしの大きさ・線の太さがそろっているかを、1 枚の確かめ用の PNG で見る（`retrocore/sprite/compare.py`）。
ほかのオプションとは一緒に使えない別のサブコマンドで、最初に `compare` と書く（`compare` という名前のファイルを変換するときは `./compare`）。

```bash
python -m apps.sprite compare hero.png slime.png walk_frames_53x91_4f.png walk_frames_53x91_4f.json -o out/sprite/compare.png
python -m apps.sprite compare hero.png きのこ.png --scale 6 --grid 8 --gap 4
```

| オプション | 内容 |
| --- | --- |
| `files`（並べる順） | くらべる PNG と、シートの JSON。JSON は拡張子を除いた名前が同じ PNG と組にし、そのシートの 1 コマ目（`frames[0]`、無ければ `frame_width`・`frame_height`）だけを使う。組になる PNG の無い JSON はエラー |
| `--scale N` | 全体の拡大（1〜16。既定 4）。絵そのものは拡大・縮小しない |
| `--grid N` | マス目の大きさ（ドット。1〜256。既定 16） |
| `--gap N` | 絵と絵の間（ドット。0〜256。既定 8） |
| `-o` / `--output` / `--out` | 保存先（既定 `out/sprite/compare.png`） |

- 絵ごとに透明な余白を切り詰め、足元（描かれている所の下端）を同じ地面の線（赤）にそろえて、指定した順に左から並べる
- マス目の横の線は地面の線から上下に数え（地面より上はマス目の数ちょうどの高さ）、縦の線は左端から。マス目は絵の上には引かない
- 1 枚目の絵のいちばん上の高さに、青い点線を横いっぱいに引く（1 枚目に主人公などの基準のキャラを置く）
- 各絵の下に「名前 幅x高さ」（切り詰めた後のドット数）。文字はこのアプリの中の小さなドットの文字（英字・数字・`_ - . + ( )`）。
  ほかの文字を含む名前（日本語など）や 33 文字以上の名前は「#2」のような番号にし、番号と名前の対応を表示する。名前が絵より長いときは、隣と重ならないように間を広げる
- 画面の「大きさをくらべる」と同じ関数で作るので、同じファイル・同じ設定なら PNG はバイト単位で同じ（`tests/test_sprite_compare.py`）。ZIP には入れない

## シートの JSON

フレームを左から順にすき間なく横 1 列に並べた透明 PNG（拡大しない）と、同じ名前の JSON。今の版は 2。

```json
{"version": 2, "motion": "breathe", "image": "hero_breathe_sheet.png",
 "frame_width": 49, "frame_height": 45, "frame_count": 2, "columns": 2, "rows": 1, "loop": true,
 "pivot": {"x": 24.5, "y": 45}, "total_duration_ms": 1000,
 "frames": [{"index": 0, "x": 0, "y": 0, "width": 49, "height": 45, "duration_ms": 500},
            {"index": 1, "x": 49, "y": 0, "width": 49, "height": 45, "duration_ms": 500}],
 "skipped": []}
```

| 項目 | 意味 |
| --- | --- |
| `pivot` | 足元（左上が原点、下向きが +y の画素）。動きのシートは (幅 / 2, 高さ − 下に足した行)。`--frames` はそろえた基準の横の位置。キャンバスを倍数にそろえたときは余白の分ずれる |
| `canvas` | （`--canvas-multiple` のとき）倍数・並べ方・足す前の大きさ（`width`・`height`）・足した余白（`left`・`right`・`top`・`bottom`）。このときは `ground_y`（= `pivot.y`）も書く |
| `padding` | 足した透明の行・列（`{"left", "right", "top", "bottom"}`）。横の位置・体全体のずれ・伸ばすを使ったときだけ |
| `frames[].offset_y` | そのフレームの絵が 1 枚目からどれだけ上下にずれたか（下が +）。体全体のずれ（浮遊・弾む・`root_y`・`--frame-root-y`）のときだけ |
| `regions_used` | `--motion-spec` で実際に使った範囲（上の「動きの表」） |
| `frames[].source_frame` | （`--frames`）元の絵の何番目のコマか（1 から） |
| `source` | （`--frames`）`frame_count`・`align`・`align_y`・`mirrored`・`dot_size` |
| `same_as_first` | （`--same-as-first` のとき）範囲・コマごとのずれ・合い方・写したか（`copied`・`code`）・`changed_dots`・`kept_dots` など |
| `steady` | （`--steady on` のとき）`mode`・`size`・`reference_frame`・`rows` と、コマごとの `shift`・`restored_dots`・`kept_dots`・`kept_reasons` |
| `edits` | （`--edits` のとき）`palette`・`dot_count`・元のコマごとの `dots`（[x, y, 色の番号 か null]）。`--join` では `file`・`file_frame` も。裏返したコマだけの直しは `"mirrored": true` |
| `ground_y` / `frames[].air` | （`--align-y keep`・`--frame-shift-y` のとき）地面の線（いちばん低い足元。左上が原点、下向きが +y の画素）と、足元が地面の線から何ドット上か |
| `frames[].flipped` | （順番に `r` があるとき）左右に裏返したコマ |
| `source.shift_y` | （`--frame-shift-y` のとき）元のコマごとの上げ下げ |
| `frames[].source_file` / `source_file_frame` | （`--join` で 2 つ以上のファイル）そのフレームがどのファイルの何コマ目か |
| `source.files` | （`--join` で 2 つ以上のファイル）ファイルごとの `name`・`frame_count`・`first_frame`・`dot_size`（使った大きさ）・`detected_dot_size`・`grid_scale`（格子を何倍にしたか）・`height`・`height_used`・`height_match`（`base` いちばん小さい / `same` / `adjusted` 合わせた / `off` 高さが違うが合わせていない（既定）/ `too_far` 違いが 1 割より大きくて合わせない） |
| `source.joins` | （`--join` で 2 つ以上のファイル）境目ごとの `from_file`・`to_file`・`from_frame`・`to_frame`・`loop`（くり返すときの最後 → 最初）・`shift`（`x`・`foot`・`top`。ドット）・`large`（警告したか） |
| `swaps` / `frames[].swap` | （`--swaps` のとき）まばたきの記録（上の「差し替え」）。しゃべりのシートでは口パクとまばたきの記録（上の「口パク」）。1 枚の絵の動きでは `skipped` の前に入る |
| `animations` / `frames[].animation` / `source.anims` | （`--anims`）動きの名前ごとのフレームの番号・時間・くり返すか・行と、動きごとの設定（上の「まとめたシートの JSON」） |
| `palette` | （`--frames`、PR #40 から）シートで使った色（`"#rrggbb"`、小さい順）。読み戻しで新しい色を見つけるのに使う |
| `skipped` | 作らなかった理由・警告の記録（いちばん外の最後。何もなければ `[]`） |

並びは `source` → `same_as_first` → `steady` → `edits` → `swaps` → `palette` → `skipped`。直し・範囲・ちらつき・差し替えを使わないときの JSON は前の版とバイト単位で同じ。
Windows の CLI は JSON の改行を CRLF、画面は LF で書く（中身は同じ）。

- **Unity**: Sprite Mode = Multiple、Slice の Grid By Cell Size で `frame_width` × `frame_height`。Pivot は Custom で (pivot.x / frame_width, 1 − pivot.y / frame_height)（下端の中央なら (0.5, 0)）
- **Godot**: AnimatedSprite2D の SpriteFrames に「スプライトシートから追加」で横 `frame_count`・縦 1。各フレームの長さは `duration_ms` から。
  テクスチャのフィルターを Nearest・ミップマップなし・整数倍の拡大にする（プロジェクト・取り込みの設定と、JSON から SpriteFrames を作る GDScript の例は [apps/sprite/README.md](../apps/sprite/README.md) の「ゲームで使うときの設定（Godot 4）」）
- 影や当たり判定は pivot のまま動かさないか、`offset_y` の分だけ動かして合わせる

### 作らなかった理由（`<出力名>_<動き>_skipped.json`）

動きを作れなかったときに、シートの代わりに書く（`-q` でも）。形はシートの JSON と同じ版で `image` は `null`、`made: false`、作ろうとしたフレームの数と表示時間、`skipped`:

```json
{"version": 2, "motion": "nod", "image": null, "made": false, "frame_count": 2, "loop": true, "total_duration_ms": 800,
 "frames": [{"index": 0, "duration_ms": 600}, {"index": 1, "duration_ms": 200}],
 "skipped": [{"stage": "motion", "code": "no_neck_found", "message": "うなずくを付けませんでした: …。試せる高さ: 5、9、13",
              "frame": 1, "frames": [1], "row": 2, "rows": [2], "track": "upper_y", "region": "upper",
              "value": -1, "values": [-1], "rows_from_bottom": [4, 15],
              "suggestions": [{"nod_split": 5}, {"nod_split": 9}, {"nod_split": 13}], "result": "held"}]}
```

1 件の項目と code の一覧は [sprite-app-plan.md](sprite-app-plan.md) の「作らなかった理由の記録（skipped）」。

## 1 枚絵（Still App）

`python -m apps.still` は、ドット絵風の 1 枚絵（背景も絵の一部）を、決まった色数・格子のドット絵にそろえる別のアプリ
（`retrocore/still/`。使い方は [apps/still/README.md](../apps/still/README.md)）。ブラウザ版（`apps/still/web/still-app.html`。
`python tools/build_still_web.py` で作る）と同じ関数で、同じ設定なら PNG と JSON はバイト単位で同じ（`tests/test_still_cli.py`）。

```bash
python -m apps.still 入力.png --mode pc98 --out out/still/部屋
python -m apps.still 入力.png --width 160 --colors 8 --canvas 640x400
python -m apps.still 入力.png --width 600 --tidy-pattern            # ざらついた範囲を 2 色の模様に整える
python -m apps.still 入力.png --mode modern --tidy-dots             # 今風。孤立したドットもそろえる
python -m apps.still 写真.jpg --mode photo --smooth strong --outline # 写真の下書き（横 320）。輪郭の線も引く
python -m apps.still 入力.png --mode modern --width 600 --smooth weak # 今風＋色の塊を弱くまとめる
python -m apps.still 写真.png --mode photo --colors 16 --4bit        # 写真の下書き＋16 色＋4096 色に丸め
python -m apps.still 写真.jpg --mode photo --width 640 --colors 16 --4bit --dither 2x2   # 空などに 2×2 の網目
python -m apps.still 写真.jpg --mode photo --aspect 16:9 --anchor 50,20   # 16:9（320×180）に切り抜く（少し上寄り）
python -m apps.still 写真.jpg --mode photo --width 640 --aspect 16:10 --colors 16 --4bit --dither 2x2   # 640×400
```

`--mode` は **初めの設定**（項目の値の組み合わせ）で、工程は 1 本。どの値も下の項目のオプションで 1 つずつ変えられる
（値の一覧は [apps/still/README.md](../apps/still/README.md) の「モード＝初めの設定」の表。コードは `retrocore/still/convert.py` の `MODES`・`ITEMS`）。

| オプション | 内容 |
| --- | --- |
| `input` | 入力の画像（PNG・JPEG）。透明な所は黒で埋める。HEIC は読めない（「JPEG か PNG にしてください」と出して止める） |
| `--out` / `-o` | 出力名（拡張子なし）。`<出力名>.png` と `<出力名>.json` を書く。既定 `out/still/<入力の名前>` |
| `--mode pc98` | 初めの設定（既定）: マス目を探す・上限 640×400・まとめない・16 色・4096 色に丸める・いちばん暗い色を取っておく |
| `--mode modern` | 今風: マス目を探す・上限 1920×1080・まとめない・32 色・4096 色に丸めない・いちばん暗い色・似た色をまとめる（手順 4） |
| `--mode photo` | 写真の下書き: 写真の向き（EXIF）を直す・マス目は探さずに横 320 マス・上限 1920×1080・色の塊をまとめる（中）・減色は今風と同じ（ただし似た色はまとめない）・孤立したドットをそろえる。下の「写真の下書きの工程」 |
| `--orient` / `--no-orient` | 写真の向き（EXIF の Orientation）を直して読む・読まない（写真の下書きだけ既定で入）。切のときの読み方は前の版のまま |
| `--grid` | ドットのマス目を探す（98・今風の既定。写真の下書きで使うとき）。`--width` とは一緒に使えない |
| `--width N` | マス目を探さずに、横 N マスに等分する（縦は縦横の比から四捨五入。写真の下書きの既定は 320） |
| `--aspect W:H` | 比率に合わせて切り抜く（正の整数どうし。例 16:9・16:10・4:3・3:2・1:1・21:9・1200:630。既定は元のまま）。拡大・縮小はせず、切り抜くだけ（余白を足すのは `--canvas`）。下の手順 1b |
| `--anchor X,Y` | 切る位置（横・縦それぞれ 0〜100 の%。既定 50,50 は中央、50,0 は上寄せ、0,50 は左寄せ）。`--aspect` のときだけ使える |
| `--max-size WxH` | 大きさの上限（ドット）。超えたら出力せずに止め、収まる `--width` の候補（いちばん大きい数・その 3/4・半分）を出す |
| `--smooth off\|weak\|medium\|strong` | 色の塊をまとめる強さ（写真の下書きの既定は medium、ほかは off）。マス目を探したときは、見つけたマスごとに 4×4 の画素に縮めてからまとめる（下の「写真の下書きの工程」の 3・4。マスが 4px より小さいときは縮めない） |
| `--colors N` | 色数の上限（2〜256）。似た色が多いときは少なくなる |
| `--4bit` / `--no-4bit` | 色を 4096 色（RGB 各 4 ビット）に丸める・丸めない（98 だけ既定で入） |
| `--keep-darkest` / `--no-keep-darkest` | いちばん暗い色をパレットに取っておく・取っておかない（どの初めの設定でも入） |
| `--merge-similar` / `--no-merge-similar` | 似た色をまとめる・まとめない（今風の既定は入、98・写真の下書きは切。どの初めの設定でも使える）。写真の下書きは、近い色が空などの段として要るので切（まとめると photo03 で 16 色が 11 色に減り、空の段が粗くなった） |
| `--dither off\|2x2\|4x4` | ディザを作る（どの初めの設定でも既定は off）。なめらかに色が変わる所で、近い 2 色の間を Bayer の網目（2x2 は 5 段、4x4 は 17 段）でつなぐ。下の手順 4b |
| `--tidy-dots` / `--no-tidy-dots` | 孤立したドットをそろえる・そろえない（既定は写真の下書きだけ入。下の手順 5） |
| `--outline` | 輪郭の線を引く（どの初めの設定でも既定は切。下の手順 7） |
| `--canvas WxH` | 絵を真ん中に置き、余白を足して W×H にする（奇数の余りは右・下に 1 多く）。余白はパレットのいちばん暗い色（0 番）。絵より小さい・上限より大きいときはエラー |
| `--tidy-pattern` | ざらついた範囲（AI のディザ風の所が減色でばらばらの点になった所）を、2 色の決まった模様（2×2 の Bayer 型・濃さ 5 段）に整える（既定は切。下の手順 6） |
| `-q` | 結果の表示を減らす |

`--help` の項目の説明は、ブラウザ版の「？」の 1 行目と同じ文（`retrocore/still/guide.py`。そのあとに CLI だけの補足（既定の値など）を足す）。

工程:
0. **読み込み**: `--orient` のとき（写真の下書きの既定）は写真の向き（EXIF）を直す（下の「写真の下書きの工程」の 1）
1. **ドットに戻す**: `--grid`（98・今風の既定）は Sprite App と同じグリッド検出（正方形のドット。端で半ドットより細いマスは落とす）。見つからないときは止めて、
   縦・横の境目から別々に推定した 1 ドットの大きさと、1 ドット 6・8・10・12px とした場合の `--width` の候補を出す。
   マスの色は平均ではなく **いちばん多い色**（各チャンネル 16 段階の組で数え、いちばん多い組の中でいちばん多い元の色。縁から 2 割は数えない）
1b. **比率に合わせて切り抜く**（`--aspect` のときだけ。`retrocore/still/aspect.py`）。計算は整数だけで、四捨五入は .5 を上げる:
   - 横のマス数で分けるとき（`--width`・写真の下書き）: 手順 1 の**前に**入力を切り抜く。縦のマス数 H を `横のマス数 × 高さ ÷ 幅` の四捨五入で先に決め、
     入力を ちょうど 横のマス数:H の比の、いちばん大きい四角に切る（片方の辺はそのまま、もう片方は四捨五入）。幅 320 なら 16:9 は 320×180、16:10 は 320×200。
     比が割り切れないとき（例 幅 300・16:9 は 168.75 → 169）は、その分だけ比がずれる（JSON の `aspect.exact` が false）。写真の向きを直した後の画像を切る
   - マス目を探すとき: 手順 1 で見つけたマス目のまま、**マス単位で**切る（格子をずらさない）。長すぎる方の辺のマスの数を `もう一方 × 比` の四捨五入にする
     （例 165×95 マスを 16:9 → 165×93）。マスの色は切り抜く前と同じ。色の塊をまとめるのは切った後のマスだけ
   - 切る量の分け方: 左（上）で `切る量 × 位置% ÷ 100` の四捨五入、残りを右（下）で切る
2. **大きさを確かめる**: 上限を超えたら止める（上の `--max-size`。比率に切り抜いた後の大きさで確かめる）
3. **減色**（`retrocore/still/colors.py`）: いちばん暗い色を 1 色取っておき（全体の 1% 以上ある 16 段階の組のうちいちばん暗い組の色）、
   残りの色を Sprite App の単純化（大きな面を守る割合だけ 0.5%）で選ぶ。選んだ色を 4096 色に丸め（各チャンネルの上下の段のうち、
   明るさの感じ方で近い方）、丸めで重なったら少ない色数のまま（JSON の `merged_by_4bit` と警告）。最後に全部のドットをいちばん近い色
   （OKLab）にする。新しい色は作らない
4. **似た色をまとめる**（今風モードの既定。`retrocore/still/colors.py` の `merge_similar()`）: パレットの色どうしで OKLab の距離が 0.04 未満の組を、
   いちばん近い組から 1 組ずつ、ドット数の少ない方を多い方の色にする（同じなら暗い方を残す）。いちばん暗い色（0 番と取っておいた色）は必ず残す側。
   0.04 は、OKLab でやっと見分けられる差（約 0.02）の 2 倍（並べればやっと分かるが、絵の中ではまず見分けられない）。新しい色は作らない
4b. **ディザを作る**（`--dither 2x2`・`4x4` のときだけ。`retrocore/still/dither.py`）: 色を割り当て直すだけで、パレットはディザなしのときと同じ（新しい色は作らない）。
   - なめらかな所: 減色する前のドットの色（OKLab）で、上下・左右のとなりとの差が 0.03 以上の所が、まわり 5×5 に 1 つも無いドット。
     そのつながりが 64 ドット未満なら外す。輪郭・はっきりした境目・木の葉のような細かい描き込みは差が大きいので入らない
   - 2 色: まわり 5×5 の色の平均にいちばん近い色と、その色から OKLab で 0.15 未満の色のうち、平均を間にはさみ、平均が 2 色を結ぶ線にいちばん近い色。
     平均が線から 2 色の距離の半分より離れているとき・どちらかがいちばん暗い色のときは入れない。ディザなしで 2 色のどちらかだったドットだけを変える
   - 網目: 平均が 2 色の間のどこにあるかを、5 段（2x2）か 17 段（4x4）のいちばん近い段にし、Bayer の表で段の数より小さい所を明るい方の色にする。
     表の位置は絵の左上を基準にした絵全体で共通の座標（手順 6 と同じ考え方）
   - しきい値の理由: photo03（幅 640）で、空のとなりどうしの差は 99.9% が 0.007 以下、木の葉は半分が 0.009 以上・1 割が 0.083 以上だったので、その間の 0.03。
     2 色の上限は、16 色の空の同じ色みの段（0.111）を入れ、空と雪（0.169）は入れない 0.15
   - 網目の範囲のドットは、手順 5（孤立したドット）で変えず、手順 6（模様を整える）の候補にもしない
5. **孤立したドットをそろえる**（`--tidy-dots` のときだけ。`retrocore/still/lone.py`）: 真ん中のドットの色がまわりの 8 ドットに 1 つも無く、
   まわりの 7 つ以上が同じ色 A のとき、そのドットを A にする。いちばん暗い色のドットと絵の端のドットは変えない。1 回だけ（変える前の絵で判断する）。
   使われなくなった色はパレットから外す
6. **模様を整える**（`--tidy-pattern` のときだけ。`retrocore/still/pattern.py`）: ドットごとに、まわり 5×5 の色の入れ替わり（となり合う組のうち色が違う組）が
   3 割以上で、多い 2 色が 8 割以上（2 番目の色も 3 ドット以上）の所を候補にし、多い 2 色の組が同じ候補のつながりを 1 つの範囲にする
   （候補に囲まれた 1〜2 ドットの穴は入れる）。範囲の中の場所ごとに、まわり 7×7 の明るい方の色の割合を 0・25・50・75・100% のいちばん近い段にし、
   2×2 の決まった模様（50% は市松）で塗る。模様の位置は絵の左上を基準にした絵全体で共通の座標なので、範囲の境目でもずれない。
   触らない所: 輪郭（パレットの 0 番と取っておいたいちばん暗い色が 8 近傍で 8 ドット以上つながった所）・平塗りの面（入れ替わりが少ない）・
   64 ドット未満の範囲・3 色目が 1 割以上の範囲（描き込みとみなす。3 色目は範囲とそのまわりで、同じ色が 3〜63 ドットかたまった所だけを数える。
   1〜2 ドットの点はノイズ、64 ドット以上はとなりの面）。3 色目の点は 2 色のうち近い方にする。新しい色は作らず、使われなくなった色はパレットから外す。
   しきい値は中で決めていて、オプションは無い（JSON の `pattern.settings` に書く）。切のときの出力（JSON も）は前とバイト単位で同じ
7. **輪郭の線**（`--outline` のときだけ。`retrocore/still/outline.py`）: 上下左右のとなりより、明るさ（OKLab の L）が 0.25 以上暗いドットを、
   パレットのいちばん暗い色（0 番）にする（境目の暗い方の側に 1 ドットの線）。もともと 0 番のドットは数えない。1 回だけ（引く前の絵で判断する）。
   新しい色は作らず、使われなくなった色はパレットから外す。0.25 は photo01 で 0.2 だと床のごみの縁にも線が付きすぎたため
8. **キャンバス**（`--canvas` のときだけ）→ **PNG**（インデックスカラー。パレットは使った色だけ、暗い順。ビット深度は色数に合わせて 1・2・4・8。
   Pillow の版で圧縮が変わらないよう、標準の zlib で書く）と **JSON**

写真の下書きの工程（`--mode photo` の初めの設定。`retrocore/still/photo.py`）。`--smooth` が off でないとき、上の手順 1 の代わりに 2〜4 を通る（手順 3 からは同じ）。`--grid` と一緒のときは、2 で見つけたマスを 1 つずつ 4×4 の画素に縮め（`smooth_grid_dots()`。マスの区切りと JSON の `dots` はグリッド検出のまま）、`--smooth off` のときは手順 1 の `--width` と同じ:
1. **読み込み**: EXIF の Orientation を見て回す（`ImageOps.exif_transpose`）。HEIC・HEIF は先頭のバイトで見分けて止める
2. **先に縮める**: 写真が仕上がり（横 W × 縦 H マス）の 4 倍より大きいときは、4W × 4H に縮める（マスの画素の平均。整数で四捨五入）。ブラウザ版を重くしないため
3. **色の塊をまとめる**: 桑原（Kuwahara）フィルタ。各画素で、その画素を角に含む 4 つの正方形のうち、明るさのばらつきがいちばん小さいものの平均の色にする。
   境目をまたぐ正方形はばらつきが大きいので選ばれず、境目は残ったまま近所がならされる。強さは正方形の大きさ（半径 r。1 辺 r+1）とくり返す回数:
   weak（弱）r=2・1 回、medium（中）r=4・2 回、strong（強）r=5・3 回（4 倍に縮めた後の画素で数える）。整数だけで計算する（CLI とブラウザ版で同じ結果にするため）
4. **仕上がりの大きさに縮める**: W × H マス（4×4 の画素ずつ）。マスの色はいちばん多い色（上の手順 1 と同じ数え方）

JSON（`<出力名>.json`）の形:

```json
{
  "app": "still", "version": 1, "source": "入力.png",
  "input": {"width": 1672, "height": 941},
  "width": 165, "height": 95,
  "picture": {"width": 165, "height": 95},
  "pixel_aspect": "1:1",
  "dots": {"method": "grid", "cells": [165, 95], "dot_size": 9.935, "gap_range": {"x": [8, 12], "y": [7, 12]},
           "trimmed_edges": {"left": 0, "top": 0, "right": 0, "bottom": 0}, "area": [0, 0, 1672, 941]},
  "canvas": null,
  "colors": {"limit": 16, "used": 16, "chosen": 16, "merged_by_4bit": 0, "darkest_kept": "#111111"},
  "palette": [{"index": 0, "rgb4": [1, 1, 1], "rgb": [17, 17, 17], "hex": "#111111", "count": 2447}, …],
  "settings": {"mode": "pc98", "colors": 16, "four_bit": true, "max_size": [640, 400], "width": null, "canvas": null, "keep_darkest": true},
  "warnings": []
}
```

- `width`・`height` は出力の大きさ（キャンバスを足した後）、`picture` は絵の部分の大きさ
- `dots` は `--width` のとき `{"method": "width", "cells", "width", "cell_size": [横, 縦]（px）, "area"}`
- `--aspect` のときだけ、`settings` に `"aspect": [W, H], "anchor": [X, Y]` と、`dots` の後に `aspect` が入る:
  `{"ratio": [W, H], "anchor": [X, Y], "stage": "input"（入力の段階）か "cells"（マス単位）, "box": [左, 上, 右, 下]（残した範囲。入力の画像の画素）,
  "cut": {"left", "top", "right", "bottom"}（切り落とした画素の数。マス単位のときはマス目の範囲から数える）, "exact": 絵の大きさがちょうど比のとおりか,
  "cut_cells": {"left", "top", "right", "bottom"}（マス単位のときだけ。切り落としたマスの数）}`。`input` は切り抜く前の大きさ、`dots.area` は残した範囲。
  元のままのときの出力（JSON も）は前とバイト単位で同じ（`tests/test_still_aspect.py`・`tests/test_still_items.py`）
- `canvas` は `--canvas` のとき `{"width", "height", "offset": [左, 上], "fill": {"index": 0, "hex"}}`
- `palette` の `rgb4` は RGB 各 0〜15（`--no-4bit` で 4096 色に無い色は `null`）。`count` はその色のドット数（余白を含む）
- `--tidy-pattern` のときだけ、`settings` に `"tidy_pattern": true` と、`palette` の後に `pattern` が入る:
  `{"regions": 整えた範囲の数, "changed": 変えたドット数, "skipped": {"third_color": 3 色目が多くて整えなかった範囲の数, "small": 小さくて整えなかった範囲の数},
  "protected_line_dots": 輪郭として守ったドット数, "settings": {中で決めたしきい値}, "list": [{"box": [左, 上, 右, 下], "size", "colors": ["#暗い方", "#明るい方"], "changed", "third", "skipped"（整えなかったときだけ）}, …]}`。
  座標は絵の部分（キャンバスの余白を足す前）
- `--mode modern`・`--mode photo` のときは `settings` に `"merge_similar": true/false` が入る。似た色をまとめたときは `colors` に
  `"merged_similar": まとめた組の数`、`palette` の後に `similar` が入る: `{"merged": 組の数, "distance": 0.04, "space": "OKLab",
  "list": [{"from": "#まとめた色", "to": "#合わせた色", "count": 移したドット数, "distance": 距離}, …]}`（まとめた順）
- `--dither 2x2`・`4x4` のときだけ、`settings` に `"dither": "2x2"`（か `"4x4"`）と、`palette`（似た色の `similar`）の後に `dither` が入る:
  `{"kind": "2x2", "changed": 網目で色を変えたドット数, "area": 網目の範囲のドット数（2 色の間の段が 0 と最大のどちらでもない所）, "pairs": 網目でつないだ 2 色の組の数,
  "settings": {"tile": "bayer2x2", "levels": 5, "window": 5, "edge_max": 0.03, "min_area": 64, "pair_max": 0.15, "off_line_max": 0.5, "space": "OKLab"}}`。
  off のときの出力（JSON も）は前とバイト単位で同じ（`tests/test_still_dither.py`）
- `--tidy-dots` のときだけ、`settings` に `"tidy_dots": true` と、`tidy_dots`: `{"changed": 置き換えたドット数, "settings": {"min_same": 7, "neighbors": 8, "passes": 1, "keep_darkest": true, "keep_edges": true}}` が入る
- 98 モードで `--tidy-dots` を使わないときの出力（JSON も）は、今風モードを足す前とバイト単位で同じ（`tests/test_still_modern.py`）
- 初めの設定から項目を 1 つでも変えたときは、`settings` の後に `preset` が入る（初めの設定のままなら入らず、前の版とバイト単位で同じ）:
  `{"name": "初めの設定の名前", "changed": {"項目": {"preset": 初めの設定の値, "used": 使った値}, …}, "values": {項目の値すべて（工程の順）}}`。
  項目は `orient`・`find_grid`・`width`・`aspect`・`anchor`・`max_size`・`smooth`・`colors`・`four_bit`・`keep_darkest`・`merge_similar`・`dither`・`tidy_dots`・`tidy_pattern`・`outline`・`canvas`。
  `settings` は前の版と同じ形のまま（初めの設定によって書かない項目がある。全部の値は `preset.values`）。
  比率が元のままのときは `preset.values` に `aspect`・`anchor` を書かない（前と同じバイト列）
- 色の塊をまとめたとき・写真の向きを直したときは、`dots` の後に `photo` が入る（下の `--mode photo` の形。向きを直さないときの `orientation` は null、
  まとめないときの `prescale`・`work_size`・`smooth` は null）。横のマス数で分けてまとめたときの `dots.method` は `"photo"`、マス目を探してまとめたときは `"grid"`
- `--mode photo` のときは `settings` に `"mode": "photo"`・`"width"`・`"tidy_dots": true/false`・`"smooth": "medium"`・`"outline": true/false` が入り、
  `dots` は `{"method": "photo", "cells", "width", "cell_size"（元の写真の px）, "area"}`、`dots` の後に `photo` が入る:
  `{"orientation": EXIF の値（無ければ 1）, "prescale": 4（先に縮めなかったときは null）, "work_size": [まとめた大きさ], "smooth": {"level", "method": "kuwahara", "radius", "passes"}}`。
  `input` は向きを直した後の大きさ
- `--outline` のときは `settings` に `"outline": true` と、`outline`: `{"drawn": 線を引いたドット数, "settings": {"min_diff": 0.25, "space": "OKLab L", "neighbors": 4, "side": "darker", "color": "darkest"}}` が入る。
  98・今風モードで使わないときの出力（JSON も）は前とバイト単位で同じ（`tests/test_still_photo.py`）
