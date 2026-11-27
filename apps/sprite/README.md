# Sprite App の使い方

AI が描いた「ドット絵風の画像」（ドット 1 個が約 20px）から、本物のドット絵のスプライト（1 ドット＝1 画素・平塗り・少ない色）と、
動きのスプライトシートを作る。ブラウザ版と CLI は同じ Python のコード（`retrocore/sprite/`）で動き、同じ設定なら出力はバイト単位で同じ。

- オプションの一覧と出力のファイル・JSON の形: [docs/cli.md](../../docs/cli.md)
- 工程の仕組みと値を決めた理由: [docs/how-it-works.md](../../docs/how-it-works.md)
- Piskel での手直しと読み戻し: [docs/sprite-edit.md](../../docs/sprite-edit.md)

## ブラウザ版

`apps/sprite/web/sprite-app.html` は 1 枚で動く HTML。サーバーもインストールも要らず、画像はブラウザの外に送られない
（インターネットには Pyodide 本体を読み込むためだけにつなぐ）。

1. GitHub で `apps/sprite/web/sprite-app.html` を開き、右上の「Download raw file」を押す
2. ダウンロードした `sprite-app.html` をダブルクリックして、ブラウザ（Chrome・Edge など）で開く
3. 「準備ができました」と出たら（初回は 10〜30 秒）、画像を選んで「② 変換する」→ 下の「保存」の「**まとめて保存（ZIP）**」

入力の種類は「1 枚の絵」（既定）と「コマが並んだ絵」。説明の文はふだん隠してあり、項目や見出しの横の「？」を押すと出る
（いちばん上の「説明をすべて表示」で全部を出せる）。

**画面の並び**（毎回使うものだけを出し、ほかは折りたたんである。閉じていても中の設定は効く）

1. ① 画像を選ぶ・入力の種類
2. 1 枚の絵: 動き（呼吸など）と、その設定 ／ コマが並んだ絵: つなげるファイルの一覧・動きの名前・再生の順番と時間（順番・1 コマの時間・名前）
3. 重ね（エフェクト）
4. 「**必要なときに開く**」（折りたたみ）: 色数・クリーンアップ・縁取り、コマの数・横と縦のそろえ方・コマごとの上げ下げ、分ける高さ
5. 「**細かい設定**」（折りたたみ）: 体の上下、ちらつきを減らす・ちらつきの見える化、1 コマ目と同じにする範囲、小さな部品を守る
6. ② 変換する・設定を最初に戻す
7. 元の画像・変換結果
8. 動きの再生（コマが並んだ絵では「コマの再生」）と、ドットを直す（差し替え = まばたき・口パクもここで描く）
9. 保存: 「**まとめて保存（ZIP）**」、GIF の拡大・背景、「**1 つずつ保存**」（折りたたみ。ファイルを 1 つずつ保存するボタン）
10. 「**外のエディタで直す**」（折りたたみ）: .piskel の保存、直したシートを読み戻す

折りたたんだ欄の中に、最初の値から変えた設定があると、見出しに「**（変更あり）**」と出る
（動きの名前が jump のときに自動で入る値など、アプリが自動で入れた値は数えない）。

**まとめて保存（ZIP）** には、そのとき作れるものが全部入る: ③ の PNG、シートの PNG・JSON・GIF、左右反転のシート・JSON・GIF、
しゃべり・エフェクトのシート（PNG・JSON・GIF）、直していれば `edits.json`、差し替えを描いていれば `swaps.json`、
動きをまとめていれば `.anims.json` と動きごとの GIF。.piskel と、ちらつきの見える化・範囲の図・作らなかった理由などの確かめ用のファイルは入れない
（「外のエディタで直す」「細かい設定」などの欄から 1 つずつ保存する）。ZIP の中のファイルは「1 つずつ保存」で保存したものとバイト単位で同じで、
中の日時は 1980-01-01 に固定してあるので、同じ入力なら同じ ZIP になる。名前は `<出力名>_all.zip`。

### 1 枚の絵に動きを付ける

1. 画像の選択の下の「動き」で、呼吸・横揺れ・浮遊・弾む・うなずく・伸びのどれかを選ぶ → すぐ再生される（変換はやり直さない）
2. 必要なら設定を変える
   - **分ける高さ**（呼吸・うなずく・伸び）・**揺れ始める高さ**（横揺れ。どちらも「必要なときに開く」の中）: 「自動」か、**下から何ドット目か**（1 = いちばん下の行）。
     「行の番号を表示」で左に目盛りが出る（灰色の行は足元で指定できない）
   - フレーム数・表示時間（`500` で全部、`600,400` でフレームごと）・下がる量・揺れ幅・弾む高さ・伸びる量。弾む・うなずく・伸びのフレーム数は量で決まる
   - 「小さな部品を守る」（呼吸・うなずく・伸び。「細かい設定」の中）: 目・頬などの長さを変えない（そのぶん作れない高さが増える）
3. 作れなかったときは黄色い枠に理由が出る。「近くで分けられる高さ: 11」と出たら「11 で試す」を押す。
   「⑤ 作らなかった理由を保存（JSON）」で理由を保存できる
4. 保存: 「まとめて保存（ZIP）」。1 つずつなら「1 つずつ保存」を開いて ④ スプライトシート（PNG）、⑤ シートの情報（JSON。同じフォルダに置く）、⑥ GIF（拡大・背景は「保存」の欄）

| 画面の欄 | CLI |
| --- | --- |
| 呼吸 / 横揺れ / 浮遊 / 弾む / うなずく / 伸び | `--motion breathe` / `sway` / `float` / `bounce` / `nod` / `stretch` |
| 分ける高さ（呼吸）・揺れ始める高さ（横揺れ） | `--breathe-split auto\|N`・`--sway-split auto\|N` |
| 首の高さ・下がる量（うなずく） | `--nod-split auto\|N`・`--nod-depth D` |
| 分ける高さ・伸びる量（伸び） | `--stretch-split auto\|N`・`--stretch-amount A` |
| 弾む高さ | `--bounce-height H` |
| フレーム数・表示時間 | `--motion-frames N`・`--motion-duration MS[,MS...]` |
| 下がる量（呼吸）・揺れ幅（横揺れ・浮遊） | `--breathe-depth D`・`--sway-amount A`・`--float-amount A` |
| 小さな部品を守る | `--keep-small-parts` |
| ⑥ GIF（拡大・背景） | `--gif`（`--gif-scale N`・`--gif-background "#rrggbb"`） |

保存したファイルは CLI の `python -m apps.sprite 入力.png -o <名前>_sprite.png --motion breathe ...` が書く `<名前>_sprite_breathe_sheet.png` / `.json` と同じ中身
（JSON の改行だけは Windows の CLI が CRLF、画面が LF）。

### 表で細かく直す（表で編集・表を読み込む）

慣れてきたら、動きを「数の表」で直せる（CLI の `--motion-spec` と同じ。表の形は [docs/cli.md](../../docs/cli.md) の「動きの表」）。プリセットだけで使うなら押さなくてよい。

1. プリセットを選んで**「表で編集」**（ファイルから始めるなら**「表を読み込む（.motion.json）」**）→ 丸いボタンに「表（自分で編集）」が出る
2. 1 行が 1 フレーム（いちばん左が「行（1 から）」）。列は表示時間・**upper_x**（分け目から上の横。+ が右）・**upper_y**（縮める − / 伸ばす +）・
   **root_y**（体全体の上下。− が上。足元も動くのはこの列だけ）。空のマスは 0。右の「↑ ↓ 複製 削除」と「＋ 行を足す」（2〜32 行）
3. 作れない行は**マスが赤くなり**、黄色い枠に理由と「11 で試す（upper_y）」のボタンが出る。同じ行で upper_x と upper_y を一緒に使うのはまだ作れない
4. **「表を保存…」**→「数（このキャラ用）」か「割合（別のキャラにも使う）」→ `<動きの名前>.motion.json`

**動かす範囲（帯・四角）を描く**: 分け目 1 本だと「分け目から上が全部動く」ので、剣のように分け目をまたぐ物も動く。帯・四角なら囲んだ中だけが動く。

1. 「表で編集」の大きな絵（縦は下から、横は左から数える）の上で**ドラッグして四角を描く**（「＋ 四角」「＋ 帯」でも足せる。足元の斜線の行は入れられない）
2. 白い印（端）をつかむと端だけ、枠の中をつかむと全体が動く。一覧の「下・上・左・右」で 1 ドットずつ直せる。名前・種類・削除も一覧で
3. 「upper_y をこの範囲（head）で動かす」か、列の見出しの「範囲」で、どの列をどの範囲で動かすかを選ぶ
4. 作れないときは「案 1：範囲 head を … にして試す」のボタンと、絵の上の点線の案が出る

例（ai04 の剣を動かさずにうなずく）: 「うなずく」→「表で編集」→ 左上（左 1・上 44）から右下（右 34・下 22）までドラッグ →「upper_y をこの範囲（head）で動かす」。
`apps/sprite/examples/head_rect_nod.motion.json` を数で書いたものと同じ形になる。

### コマが並んだ絵（歩き・走りなど）

同じキャラが横に何コマも並んだ絵を、1 つのキャラとしてそろえたシートにする（CLI の `--frames`）。

1. 画像を選び、「**コマが並んだ絵**」を選ぶ。必要なら「必要なときに開く」の「コマの数」を入れる（空なら自動。入れておくと数え間違えたときに作らずに知らせる）
2. 「必要なときに開く」の「縦のそろえ方」は歩きなら「足元」（既定）、足が地面から離れる走り・跳びは「頭の高さ」、ジャンプで宙に浮くコマがあるときは「元の絵の高さを保つ」
   （うまく出ないコマは「コマごとの上げ下げ」に `0,-3,-6,0` のように入れる。− が上）。「細かい設定」の「ちらつきを減らす」を「あり」にすると、上の方の小さな違いを 1 コマ目の色に戻す
3. 「② 変換する」→ 左に見つけたコマの赤い枠と番号（数が合わないときは橙）、右にそろえたコマ
4. 「コマの再生」で再生する。「順番」（`1,2,3,2` など。「行って戻る」ボタン。`1,2,3,2r` のように r を付けたコマは左右に裏返す）・「1 コマの時間」・「名前」（walk・run）と、
   「細かい設定」の「体の上下」は、変えるとすぐ作り直す
5. **1 コマ目と同じにする範囲**（頭など。`--same-as-first`。「細かい設定」の中）: 拡大した 1 コマ目の上で**ドラッグして頭を囲む**（「描く形」を「帯」にすると横いっぱい）。
   離すと 2 コマ目以降に写し、コマごとの「見つけたずれ」「合い方」「写した／写さなかった（理由）」が出る。
   辺・角をドラッグすると大きさ、内側で位置が変わる。キー（絵をクリックしてから）: 矢印 = 1 ドット動かす、Shift＋矢印 = 右下の角、Delete = 消す、Esc = 選ぶのをやめる、Ctrl＋Z = 元に戻す。
   「数で足す」に `x,y,幅,高さ` を入れても足せる
6. **ドットを直す**（数ドットの直し。`--edits`）: 色見本で色を選び、**左ドラッグで塗る**、**右ドラッグで透明**、**Alt＋クリックで色を拾う**。使える色はシートのパレットだけ。
   キー: B ペン／E 消しゴム／I スポイト／「,」「.」前後のコマ／O 前のコマを薄く重ねる／Ctrl＋Z 元に戻す／Ctrl＋Y やり直し（50 回まで）。拡大は 4・8・12・16 倍。
   直しは**元のコマごと**なので、順番 1,2,3,2 の 2 番目と 4 番目は 1 回直せば両方変わる。
   前の段の設定を変えると直しは上に重ね直し、下の色が変わったドットは橙の枠で知らせる（「確かめた」で消える）。
   当てられなくなるとき（変換し直してコマの大きさが変わった・画像を選び直した・入力の種類を変えた）は確認の窓が出る:
   「直しを保存してから捨てる」（おすすめ）・「保存せずに捨てる」・「新しい画像にも当てる」（コマの大きさが同じときだけ）・「やめる」（前に戻す。Esc でも同じ）。
   「直しの一覧を保存（JSON）」で `<出力名>_<名前>_edits.json` を保存でき、読み込んで戻せる。
   順番に r（例 `1,2,3,2r`。そのコマを左右に裏返して使う）があると、裏返したコマ（`2r`）も開いて、そのコマだけを直せる
7. **差し替えを描く（まばたき。`--swaps`）**: 「ドットを直す」の「差し替えを描く」に印を付けると 1 コマ目が出るので、「描く絵: 閉じた目」で目を閉じた形に描く
   （「半分閉じた目」も描くと 開く → 半分 → 閉じる → 半分 → 開く）。離すとすぐ再生にまばたきが入り、「まばたき: 動き 5 周に 1 回」のように何周にしたかが出る。
   「まばたきの間隔」（既定 3000 ミリ秒）・「閉じる時間」（120）・「半分の時間」（60）も変えられる。「差し替えの一覧を保存（JSON）」で
   `<出力名>_<名前>_swaps.json` を保存でき、読み込んで戻せる。**1 枚の絵**の動き（呼吸など）でも、動きを選ぶと下に「差し替えを描く」が出る。
   **口パク**: 「描く絵: 開いた口」で口を開けた形に描くと、絵の下に「しゃべりのシート（口パク）」が出る（今の動きの上で口が 閉じる → 開く をくり返し、
   まばたきも入る。「半分開いた口」も描くと 閉じる → 半分 → 開く → 半分）。「口の切り替え」（空なら自動: 150、半分ありは 100 ミリ秒）で速さを変えられる。
   しゃべりのシートの PNG・JSON・GIF（コマが並んだ絵は左右反転も）は「まとめて保存（ZIP）」に入る。1 つずつなら「1 つずつ保存」、.piskel は「外のエディタで直す」（ふつうのシートは今までと同じ）
8. 保存: 「**まとめて保存（ZIP）**」。1 つずつなら「1 つずつ保存」を開くと、③ そろえたコマを並べた絵、④ シート（PNG）、⑤ 情報（JSON）、⑥ GIF、⑦⑧⑨ 左右反転のシート・JSON・GIF。
   ちらつきの見える化は「細かい設定」、**Piskel で直す用（.piskel）**は「外のエディタで直す」で保存する
9. **別々のファイルの絵をつなげる**（ダンスなどを何枚かに分けて描かせたとき。`--join`）: 「コマが並んだ絵」でファイルを**何枚も選ぶ**（「＋ ファイルを足す」で後から足せる）と、
   「つなげるファイル」の一覧が出る。**▲▼** で並び順を入れ替え、「外す」で外し、名前を押すと左の「元の画像」に出る。「② 変換する」で上から順につなげる。
   変換すると、ファイルごとにコマの数・何コマ目から・ドットの大きさ・高さ（合わせたときは格子を何倍にしたか）が出る。
   「キャラの高さを合わせる」（最初は印なし＝合わせない）に印を付けると、高さの違いが 1 割までならドット絵に戻すときの格子を少し変えて合わせる。
   印なしでも、高さの違うファイルがあれば注意を出す（しゃがみ・ジャンプなどポーズの違いなら、そのままでよい）。
   つなぎ目で体の位置が大きくずれるときは注意が出る（ちらつきの見える化の橙の線と f1・f2… がファイルの境目）。
   並び順を変えて変換し直しても、直したドットは同じ絵のコマの同じ所に付く
10. **動きに名前を付けてまとめる**（立ち・歩き・走り・ジャンプを 1 つのシートに。`--anims`）: 全部のファイルを選び、「つなげるファイル」の
   「**動きに名前を付けてまとめる**」に印を付ける。一覧の各行に動きの名前（idle・walk・run・jump などの候補が出る。ファイル名に入っていれば最初から入る）を入れる。
   同じ名前のファイルが続けば 1 つの動き。「動きごとの設定」で順番・1 コマの時間・くり返し・縦のそろえ方・上げ下げを決めて「② 変換する」
   （jump は最初から「元の絵の高さを保つ」「1 回で終わる」）。「再生する動き」で選んだ動きを再生し、GIF・.piskel もその動き。
   「④ まとめたシートを保存」「⑤ まとめたシートの情報を保存（JSON。animations 入り）」がまとめたシートと JSON（`<出力名>_anims_<幅>x<高さ>_<N>f.json` に `"animations"` が入る）。
   「CLI 用の設定を保存（.anims.json）」は CLI の `--anims` に使う設定のファイル（シートの情報ではない）。
   「1 回で終わる」動きは、再生の見本で最後のコマに止まり、少し待ってから最初に戻す（ゲームでは最後のコマで止める）。
   印を付けたり外したりしたら、「② 変換する」を押すまで ④⑤ は押せない（前の形のシートを保存しないように）。
   まとめている間は、まばたき・口パク・重ね・範囲・ちらつきを減らすは使えない（欄が隠れ、理由が出る）

ゲーム（Godot 4）で読むときの設定と、まとめたシートの JSON から SpriteFrames を作る例は、下の「[ゲームで使うときの設定（Godot 4）](#ゲームで使うときの設定godot-4)」。

保存したファイルは、画面の下に出る CLI（例 `--frames --frame-count 4 --frame-name walk --mirror --flicker --piskel`）と**バイト単位で同じ**。

### 重ね（！・汗・ハート・Zzz・？・キラキラなど 14 種）

動きの選択（コマが並んだ絵では再生の順番と時間）の下の「**重ね（エフェクト）**」でエフェクトを選ぶと、上の再生がキャラとエフェクトを重ねた見た目になる（CLI の `--fx`）。

- 「置く場所をずらす」（横・縦のドット）と、1 回で終わるもの（！・汗・電球・…）の「終わり方」（残す・消す）を変えるとすぐ作り直す
- 保存: エフェクトのシート（PNG）・情報（JSON）・重ねた GIF（確かめ用）。コマが並んだ絵では左右反転用も。どれも「まとめて保存（ZIP）」に入る（1 つずつは「1 つずつ保存」）。
  レイヤー「キャラ」「エフェクト」に分けた .piskel は「外のエディタで直す」で保存する
- キャラのシートは変わらない（エフェクトは別のシート）。ゲームでの置き方は下の「[ゲームで使うときの設定（Godot 4）](#ゲームで使うときの設定godot-4)」と [docs/cli.md](../../docs/cli.md) の「重ね」
- 入れておく絵は `retrocore/sprite/fx/art.py` の文字の表（描き直したら `python tools/build_sprite_web.py` で画面にも入る）

### 直したシートを読み戻す（Piskel で手直ししたあと）

いちばん下の「**外のエディタで直す**」を開き、「**直したシートを読み戻す**」でファイルを選ぶだけ（CLI の `--readback`）。

1. 「A. 直したファイル」に Piskel で保存した .piskel（Save → Save offline as File）か、書き出した PNG（Export → PNG）
2. 「B. 元のシートの情報（JSON）」（C. 直す前のシート（PNG）は、なくてもよい）
3. 確かめた結果・作り直した動き・確認の画像が出る。保存のボタン: 並べ直したシート・JSON・GIF・続けて直す用（.piskel）・左右反転・確認の画像
4. 「同じコマを 2 回使う順番」・「おかしな色を直す」・「GIF の拡大」を変えると、すぐ作り直す
5. 動きをまとめたシートのときは、A に動きごとの .piskel（`<出力名>_walk.piskel` など）、B にまとめたシートの JSON、C に**直す前のまとめたシート（要る）**。
   どの動きかはファイルの名前から決まる（分からなければ「D. 直した動き」で選ぶ）。まとめたシートのその動きの行だけが直る

## ゲームで使うときの設定（Godot 4）

ドット絵は 1 ドット＝1 画素で作ってあるので、ゲームの側で「ぼかさない・小数の位置に置かない・整数倍で広げる」設定が要る。
設定の名前は Godot 4 の画面の英語の表記（日本語の画面では訳が付く）。

**プロジェクトの設定**（Project → Project Settings。右上の「Advanced Settings」を入にすると全部出る）

| 設定 | 値 | 理由 |
| --- | --- | --- |
| Rendering → Textures → Canvas Textures → **Default Texture Filter** | **Nearest** | Linear のままだとドットの境目がぼやける |
| Rendering → 2D → Snap → **Snap 2D Transforms to Pixel** | 入 | 小数の位置に置かれて 1 ドット震えるのを防ぐ（Snap 2D Vertices to Pixel も入にしてよい） |
| Display → Window → Stretch → **Mode** | `viewport`（または `canvas_items`） | 小さな画面（例 320×180）で描いて、窓に合わせて広げる |
| Display → Window → Stretch → **Scale Mode** | **integer** | 2 倍・3 倍など整数倍だけで広げる（Godot 4.2 から。名前は版によって違うことがあります） |

Display → Window → Size の **Viewport Width / Height** にゲームの画面の大きさ（ドット）を入れ、窓の大きさ（Window Width / Height Override）はその整数倍にする。

**取り込みの設定**（FileSystem で PNG を選び、Import タブで変えて「Reimport」）

| 設定 | 値 | 理由 |
| --- | --- | --- |
| Compress → **Mode** | **Lossless** | VRAM Compressed などは色がにじむ・変わる |
| Mipmaps → **Generate** | 切 | 縮めたときに隣のコマの色が混ざる |

**キャラのシート**: AnimatedSprite2D に置き、`centered` を切って、`offset` を `(-pivot.x, -ground_y)` にすると、足元の真ん中がノードの位置に来る
（`ground_y` が無いシートは `pivot.y`）。保存の欄の「キャンバスを倍数にそろえる」（CLI の `--canvas-multiple 16`）で 16 の倍数などにしたときも同じで、
`centered` を切り `offset` を `(-pivot.x, -pivot.y)` にすれば、ノードの位置に足元が来る（JSON の `pivot` は足した余白の分ずれている）。手で読み込むなら SpriteFrames の「Add frames from sprite sheet」（スプライトシートから追加）で、
横 `columns`・縦 `rows` に分け、まとめたシートなら動きの行（`row`）のコマ（`frames` の番号）を選ぶ。速さ（FPS）は `1000 / duration_ms`、ループは `loop`、
コマごとの時間が違うときは各フレームの長さ（Duration）を `frames[].duration_ms / duration_ms` にする（下の例はこれを自動でする）。

**エフェクトのシート**（重ね）: キャラの AnimatedSprite2D の**子**にエフェクトの AnimatedSprite2D を置き、どちらも `centered` を切って、
子の `position` を JSON の `position`（キャラのコマの左上から見たエフェクトの左上。ドット）にする。
キャラの `offset` を変えたときは、子の `position` にも同じ `offset` を足す（`offset` は絵だけをずらし、子は動かさないため）。
エフェクトのシートはキャラとは別の長さ・別の時間で、`loop` が false なら 1 回で止める。

**よくある失敗**

- ぼやける → Default Texture Filter が Linear のまま（ノードの Texture → Filter が Inherit 以外になっていないかも見る）
- 動くと 1 ドット震える・線の太さが変わる → Snap 2D Transforms to Pixel が切、または Scale Mode が fractional（整数倍でない拡大）
- 端の 1 列が欠ける・隣のコマの色がにじむ → ミップマップや圧縮（Lossless 以外）、または「Add frames from sprite sheet」の横・縦の数がシートと違う

**まとめたシートの JSON から SpriteFrames を作る例**（GDScript）

**本物の Godot ではまだ確かめていません。** まとめたシートの JSON（`"animations"` 入り。`<出力名>_anims_<幅>x<高さ>_<N>f.json`）と PNG を
同じフォルダに入れ、PNG は上の取り込みの設定にしておく。AnimatedSprite2D にこのスクリプトを付けて、`sheet_json` に JSON を選ぶ。

```gdscript
# まとめたシートの JSON（"animations" 入り）から SpriteFrames を作る（Godot 4 用。本物の Godot ではまだ確かめていません）
extends AnimatedSprite2D

@export_file("*.json") var sheet_json := "res://sprites/hero_anims_22x28_8f.json"

func _ready() -> void:
	var data: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(sheet_json))
	var tex: Texture2D = load(sheet_json.get_base_dir().path_join(data["image"]))  # PNG は JSON と同じフォルダ
	var sf := SpriteFrames.new()
	for anim_name in data["animations"]:
		var a: Dictionary = data["animations"][anim_name]
		var base_ms := float(a["duration_ms"])  # その動きでいちばん多い時間
		if not sf.has_animation(anim_name):
			sf.add_animation(anim_name)
		sf.set_animation_speed(anim_name, 1000.0 / base_ms)  # FPS
		sf.set_animation_loop(anim_name, a["loop"])  # false は最後のコマで止まる
		for i in a["frames"]:  # シート全体での通し番号（frames[] の index）
			var f: Dictionary = data["frames"][int(i)]
			var atlas := AtlasTexture.new()
			atlas.atlas = tex
			atlas.region = Rect2(f["x"], f["y"], f["width"], f["height"])
			sf.add_frame(anim_name, atlas, float(f["duration_ms"]) / base_ms)  # コマごとの長さ（倍率）
	if not data["animations"].has("default"):
		sf.remove_animation("default")  # SpriteFrames.new() に最初からある空の動き
	sprite_frames = sf
	centered = false
	offset = Vector2(-float(data["pivot"]["x"]), -float(data.get("ground_y", data["pivot"]["y"])))  # 足元をノードの位置に
	play(data["animations"].keys()[0])  # 最初の動き（idle など）を再生
```

- 動きを切り替えるときは `play("walk")` のように名前で呼ぶ。`loop` が false の動き（jump など）は最後のコマで止まり、`animation_finished` の合図が出る
- 書き出したゲームでは .json がそのままでは入らないことがある。Export の「Resources」の「Filters to export non-resource files」に `*.json` を足す
- 左右反転のシート（`_mirror`）を使う代わりに `flip_h` で裏返すこともできる（左右で違う物（リボン・持ち物など）は反対側に移る。左右反転のシートと同じ）

## CLI

リポジトリのルートで実行する（`pip install -r requirements.txt`）。出力先の `out/` は Git に入らない。

```bash
python -m apps.sprite 入力.png -o out/sprite/出力.png                                  # ドット絵風の画像（グリッド検出）
python -m apps.sprite 入力.png --preview-scale 10 -o out/sprite/出力.png               # 10 倍のプレビューも保存
python -m apps.sprite 入力.png --input-type illust --scale 1/4 -o out/sprite/出力.png  # 普通のイラスト（縮小）
python -m apps.sprite 入力.png --colors 12 --cleanup strong                            # 点々が気になるとき
python -m apps.sprite 入力.png --save-palette game.gpl                                 # 使った色を保存
python -m apps.sprite 次のキャラ.png --palette game.gpl                                # その色だけで描く
python -m apps.sprite 入力.png --motion breathe --gif                                  # 呼吸のシート＋GIF
python -m apps.sprite 入力.png --motion nod --nod-depth 2 --keep-small-parts           # うなずく（頭を 2 ドットまで）
python -m apps.sprite 入力.png --motion breathe --write-motion-spec breathe.motion.json # プリセットの表を書き出す
python -m apps.sprite 入力.png --motion-spec apps/sprite/examples/head_rect_nod.motion.json   # 表で動きを作る
python -m apps.sprite 歩き.png --frames --frame-count 4 --frame-order 1,2,3,2 --gif --mirror --piskel
python -m apps.sprite 走り.png --frames --align-y match --frame-root-y=0,-1,0,-1 --flicker
python -m apps.sprite 歩き.png --frames --frame-count 4 --steady on --same-as-first 0,0,53,53
python -m apps.sprite dance1.png dance2.png --frames --join -o out/sprite/dance.png --gif # 別々のファイルの絵をつなげる
python -m apps.sprite 入力.png --motion breathe --fx surprise --gif                     # ！（驚き）を重ねる（別のシート）
python -m apps.sprite 直した.piskel --readback 元.json --gif --mirror --piskel           # 手直しを読み戻す
```

ほかのオプションは `python -m apps.sprite --help` と [docs/cli.md](../../docs/cli.md)。

## 開発

```bash
python -m pytest                    # すべてのテスト（テスト画像はコードで合成）
python tools/build_sprite_web.py    # retrocore を変えたらブラウザ版の HTML を作り直す（template.html から作る。sprite-app.html は直接編集しない）
```

処理の置き場所は [docs/how-it-works.md](../../docs/how-it-works.md) の「ファイル」。
