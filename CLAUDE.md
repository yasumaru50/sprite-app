# CLAUDE.md

このリポジトリで作業するときの手引き。作業の引き継ぎは `docs/TODO.md` から読み始めること。これまでの PR のまとめは `docs/history.md`。

## 概要

**Sprite App** — AI が描いた「ドット絵風の画像」から、本物のドット絵のスプライトと、その動き（呼吸・横揺れ・浮遊・弾む・うなずく・伸び）の
スプライトシートを作るアプリ。CLI（`python -m apps.sprite`）と、Pyodide で動く 1 枚の HTML のブラウザ版がある。

技術スタック: Python 3.10+ / NumPy / Pillow / Pyodide（ブラウザ版）。使い方は `README.md` と `apps/sprite/README.md`、設計は [docs/sprite-app-plan.md](docs/sprite-app-plan.md)。

このリポジトリは、前のリポジトリ（古いパソコン風の CG 変換アプリと一緒に開発していた所）から Sprite App の分だけを移したもの。
ドキュメントに出てくる PR の番号（#1〜#45）は前のリポジトリのもの。

## 作業の進め方

- 1 つの作業 ＝ 1 つのセッション ＝ 1 つの PR。作業が終わったら PR を作り、報告して止まる
- CI がすべて ✓ になってからマージする
- 確認用の比較画像は `preview/ai-check` ブランチの `ai-output/<作業ごとのフォルダ>/` に置く（README に感想・数を書く）。
  入力の AI 画像は同じブランチの `ai-input/`。このブランチは main にマージしない
- 報告は日本語で、初心者にも分かる言葉で書く
- フェーズ（作業単位）ごとにコミットする

## 構成

| パス | 内容 |
| --- | --- |
| `retrocore/` | 共通ライブラリ（NumPy と Pillow のみ）。Sprite App の処理は `retrocore/sprite/`。ほかのファイル（`pipeline.py`・`palette.py`・`dither.py`・`analyzer.py`・`style.py` など）は古いパソコン風の CG 変換で、Sprite App からは使っていない（ブラウザ版の HTML には入る） |
| `retrocore/sprite/` | Sprite App の処理: 工程（`Step`）を並べて実行する `SpritePipeline`、背景除去（`BackgroundStep`、`background.py`）、グリッド検出（`GridStep`、ドット絵風の入力用の `grid.py`）、縮小（`ResizeStep`、ドット絵向けの `downscale.py`）、単純化（`SimplifyStep`、似た色をまとめる `simplify.py`。大きな面の色を守る `keep_large`）、クリーンアップ（`CleanupStep`、余計なドットを片付ける `cleanup.py`）、縁取り（`OutlineStep`、外側の輪郭を付ける・切れ目を埋める `outline.py`）、パレット（`PaletteStep`、パレットファイルの読み書きと指定したパレットに合わせる `palette.py`）、RGBA の読み書き、CLI とブラウザ UI で共通の変換の入口（`convert_sprite()`、`convert.py`）、横に並んだコマを切り分けて 1 つのキャラとしてそろえる（`convert_frames()`、`frames.py`。CLI `--frames`。形の似たコマはグリッドの線を共通にする `--frame-grid`）、ちらつきを減らす（2 コマ目以降の上の方の小さな違いを 1 コマ目の色に戻す `steady.py`。CLI `--steady on`、既定 off）、人が選んだ範囲（頭など）を 1 コマ目と同じにする（位置を探して 1 コマ目のドットを写す `same.py`。CLI `--same-as-first x,y,w,h`、既定は無し）、動き（フェーズ 2。変換後のスプライトから呼吸・横揺れ・浮遊のフレームとスプライトシートを作る `motion/`。動きは数の表（行 = フレーム、列 = 動かし方）で書け、表の形は `motion/spec.py`、表からフレームを作る仕組みは `motion/timeline.py`。呼吸・横揺れ・浮遊・弾む（`motion/bouncing.py`）・うなずく（`motion/nod.py`。首を探して `upper_y` を埋める）・伸び（`motion/stretch.py`。`upper_y` を + で埋める）は表を埋めるプリセット。縮める・伸ばすの中身と「小さな部品を守る」（`keep_small_parts`）は `motion/breathe.py`。表の範囲の帯（`band`）・四角（`rect`。囲んだ中だけが動く）は `motion/region.py`。作らなかった理由の記録（シートの JSON の `skipped`・`_skipped.json`）は `motion/report.py`。そのまま見られる GIF アニメ（シートと同じ絵・時間）は `motion/gif.py`）、アプリ内の手の直し（元のコマごとのドットの差し替えの一覧を最後に当てる `touchup.py`。CLI `--edits`。ブラウザ版の「ドットを直す」）、手直しの読み戻し（Piskel などで直したシート（PNG か .piskel）と元の JSON から JSON・GIF・反転・並べ直したシートを作り直す `edit.py`。CLI `--readback`）、.piskel の読み書き（`piskel.py`） |
| `apps/sprite/` | Sprite App の CLI（`python -m apps.sprite`）。使い方は `apps/sprite/README.md` |
| `apps/sprite/examples/` | 動きの表の見本（`bounce_stretch.motion.json`: 弾む＋着地で縮む・空中で伸びる、`head_rect_nod.motion.json`: 頭の四角だけでうなずく、`bounce_head_band.motion.json`: 帯の頭の変形＋体全体のずれ） |
| `apps/sprite/web/` | Sprite App のブラウザ版（Pyodide。「表で編集」では動かす範囲（帯・四角）を絵の上で描ける）。`template.html` を `tools/build_sprite_web.py` で 1 枚の `sprite-app.html` にする（直接編集しない） |
| `tools/` | `analyze.py`・`export_db.py`・`compare.py`（retrocore の CG 変換のテストが使う）、`make_golden.py`（回帰テストのゴールデン再生成）、`compare_sprite_downscale.py`（Sprite App の縮小の比較画像）、`compare_sprite_background.py`（Sprite App の背景除去の比較画像）、`compare_sprite_grid.py`（Sprite App のグリッド検出の比較画像）、`compare_sprite_simplify.py`（Sprite App の単純化の比較画像）、`compare_sprite_cleanup.py`（Sprite App のクリーンアップの比較画像）、`compare_sprite_outline.py`（Sprite App の縁取りの比較画像）、`compare_sprite_motion.py`（Sprite App の動き（呼吸・横揺れ・浮遊・弾む・うなずく・伸び・動きの表 `--spec`）の GIF・フレームの並び・シート、分け目の線の画像、分ける高さの目安の画像、`--keep both` で小さな部品を守るの切と入の比べ、範囲（帯・四角）を色付きの枠で描いた `_regions.png`、`--vs-spec` で 2 つの表を横に並べた GIF）、`compare_sprite_frames.py`（Sprite App のコマの切り分け `--frames` の比較画像: 枠とグリッド、そろえる前と後の GIF、並び、ちらつき）、`compare_sprite_steady.py`（ちらつきを減らす `--steady` の off と on の GIF・並び・違いの図・顔の拡大・数の一覧）、`compare_sprite_same.py`（1 コマ目と同じにする範囲 `--same-as-first` のなしとありの GIF・並び・境目の図・顔の拡大・数の一覧）、`build_sprite_web.py`（Sprite App のブラウザ版の HTML を作る） |
| `tests/` | pytest。テスト画像は `tests/synth_sprite.py`（Sprite App 用のキャラクター）・`tests/synth_cg.py` でコード合成。`web_region/` はブラウザ版で範囲を描いて保存した表・シート（`test_sprite_web_region.py`）。retrocore の CG 変換の回帰テストは `test_golden.py`（入力は `fixtures/` の合成画像、期待値は `golden/`） |
| `docs/` | `TODO.md`（引き継ぎ）、`history.md`（これまでの PR のまとめ）、`sprite-app-plan.md`（設計）、`sprite-edit.md`（Piskel での手直しと読み戻しの手順）、`research-2026-11.md`（外部調査）、`compare/sprite/`（合成画像の比較）、`images/`（README の画面の写し） |
| `.github/workflows/tests.yml` | push ごとに int32-check（Pyodide の 32 ビット向けの確かめ）と、Linux ですべてのテスト |

出力先の `out/` は git 管理外。`local_samples/` は手持ちの AI 画像などで確認するためのフォルダで、中身は git 管理外（README のみ管理）。

## 開発コマンド

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m apps.sprite 入力.png -o out/sprite/出力.png   # Sprite App（CLI、ドット絵風の入力。グリッド検出）
python -m apps.sprite 入力.png --input-type illust --scale 1/4 -o out/sprite/出力.png   # 普通のイラスト（縮小）
python -m apps.sprite 入力.png --save-palette game.gpl     # 使った色を保存 → 次のキャラは --palette game.gpl
python -m apps.sprite 入力.png --motion breathe            # 呼吸のスプライトシート（<出力名>_breathe_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion breathe --breathe-split 12   # 下から 12 ドット目のあたりで上下を分ける
python -m apps.sprite 入力.png --motion sway               # 横揺れのスプライトシート（<出力名>_sway_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion float              # 浮遊（全体が上下。足元も動く例外）のシート（<出力名>_float_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion bounce --bounce-height 2   # 弾む（全体が上にだけ跳ねる）のシート（<出力名>_bounce_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion nod                # うなずく（首を自動で探し、首から上が下がる）のシート（<出力名>_nod_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion nod --nod-split 16 --nod-depth 2   # 下から 16 ドット目のあたりで分け、頭を 2 ドットまで下げる
python -m apps.sprite 入力.png --motion stretch --stretch-amount 2   # 伸び（体の上が伸びる。上に透明の行を足す）のシート（<出力名>_stretch_sheet.png / .json）も書く
python -m apps.sprite 入力.png --motion nod --nod-depth 2 --keep-small-parts   # 目・頬などの小さな部品の長さを変えない（呼吸・うなずく・伸び）
python -m apps.sprite 入力.png --motion breathe --write-motion-spec breathe.motion.json   # プリセットが埋めた動きの表も書く（"used" つき）
python -m apps.sprite 入力.png --motion-spec tests/motion_specs/bounce.motion.json   # 動きの表で動きを作る（<出力名>_<表の name>_sheet.png / .json）
python -m apps.sprite 入力.png --motion-spec apps/sprite/examples/head_rect_nod.motion.json   # 範囲（四角）の中だけを動かす表
python -m apps.sprite 入力.png --motion breathe --gif      # シートと一緒に GIF アニメ（<出力名>_breathe.gif。4 倍・透明）も書く（--gif-scale・--gif-background）
python -m apps.sprite 歩き.png --frames --frame-count 4 --frame-order 1,2,3,2 --gif --mirror   # 横に並んだコマを切り分けてそろえたシート（<出力名>_frames_<幅>x<高さ>_<N>f.png / .json。幅・高さは 1 コマ、N はコマの数）
python -m apps.sprite 正面歩き.png --frames --frame-count 3 --frame-order 1,2,3,2 --colors 16   # 淡い色の多いキャラは 16 色がおすすめ（形の似たコマはグリッドの線を自動で共通にする）
python -m apps.sprite 走り.png --frames --align-y match --frame-root-y=0,-1,0,-1 --flicker   # 頭の高さでそろえ、体全体を上下。ちらつきの見える化も書く
python -m apps.sprite 歩き.png --frames --frame-count 4 --piskel   # シートと一緒に Piskel で開ける .piskel も書く
python -m apps.sprite 正面歩き.png --frames --frame-count 3 --steady on   # 上の方の小さな違い（点・短い線）を 1 コマ目の色に戻す（--steady-size 4）
python -m apps.sprite 歩き.png --frames --frame-count 4 --same-as-first 0,0,53,53   # 頭の四角を 1 コマ目と同じにする（位置は ±3 ドットで探す。--same-search・--same-band）
python -m apps.sprite 正面歩き.png --frames --frame-count 3 --frame-order 1,2,3,2 --edits ai08_sprite_frames_edits.json   # 画面の「ドットを直す」で保存した直しを当てる
python -m apps.sprite 直した.piskel --readback 元.json --gif --mirror --piskel   # 手直ししたシート（PNG か .piskel）を読み戻して作り直す（<元の名前>_edited.*）
python tools/build_sprite_web.py   # Sprite App のブラウザ版（apps/sprite/web/sprite-app.html）を作り直す
```

## 開発ルール

- Sprite App の動き（フェーズ 2）は `retrocore/sprite/motion/` に置き、変換の工程には手を入れない（変換後のスプライトが材料）。
  新しい色を作らない・大きさと足元をそろえる・輪郭を途切れさせない・迷ったら動かさない（`check_rules()` で確かめる）。
  足元が動く動き（浮遊・弾む）は例外として、足元が縦に決まった量だけ動くことだけを許す（`check_rules(..., shifts=...)`。ほかの決まりは同じ）
  新しい動きは、なるべく動きの表（`timeline.py`）の列か、表を埋めるプリセットとして足す。足元が動いてよいのは表の `root`（体全体のずれ）の列だけ。
  伸ばす（縦の変形の +）ではみ出す分は上に透明の行を足す（`padding.top`。pivot はフレームの下端のまま）。「小さな部品を守る」は既定で切（入のときだけ表に書く）。
  範囲は分け目（`split`）・帯（`band`）・四角（`rect`）。帯・四角は囲んだ中だけが動き、境目で作れない（空いた所の色が分からない・外の違う色を隠す）ときは作らずに警告する。
  プリセットの出力（PNG の画素・JSON のバイト・警告の言葉）は変えない（`tests/test_sprite_timeline.py` がこの前のハッシュと比べる。範囲を書いていない表も含めて `tests/test_sprite_region.py` も）
  手直しの読み戻し（`edit.py`）は直したドットをそのまま使い、新しい色・半透明などは作って警告する（直すのは利用者が選んだときだけ）。
  作れなかった理由は新しい判定を足さずに記録だけする（`report.py`。code は表の問題の `rule` と同じ名前）。前のハッシュと比べるテストは、版 2 で足した分（`skipped`・`_skipped.json`）を `tests/legacy_output.py` で除いて比べる
- Sprite App の新しい処理は `retrocore/sprite/` に置く。工程は `retrocore/sprite/convert.py` の `build_pipeline()` に足す（CLI と画面で共通）。retrocore を変えたら `python tools/build_sprite_web.py` でブラウザ版の HTML も作り直す（テストで検出される）。既存の関数の挙動は変えない（ゴールデンは既存分を再生成しない）
- アプリ内の手の直し（`retrocore/sprite/touchup.py`、CLI `--edits`、画面の「ドットを直す」）は変換・範囲・ちらつきの後に当て、パレットに無い色は作らない。
  直しが無いときの出力は前とバイト単位で同じ（`tests/test_sprite_touchup.py`）
- `retrocore/` は NumPy と Pillow だけに依存させる（Pyodide で動かすため。PySide6・SciPy・OpenCV も不可。`tests/test_retrocore_deps.py` が確かめる）
- Pyodide は 32 ビット（`np.intp` が int32）。`np.bincount`・`np.repeat`・`reduceat` などに int64 の配列を渡さない（`.astype(np.intp)`）。`tests/test_sprite_int32.py` が確かめる（呼ぶ場所を足したら、その中の一覧 `REVIEWED` も更新）
- retrocore の CG 変換の部分（`retrocore/pipeline.py` など。Sprite App からは使っていない）の挙動は変えない
  （`tests/test_golden.py` が合成画像の変換結果と画素単位で比較する。意図して変えたときだけ `python tools/make_golden.py` で更新）
- 重い処理はベクトル化する（ブラウザ版の Pyodide でも数秒で終わるように）
- 新しいモジュールには pytest を付け、テスト画像はコードで合成する。Sprite App の新しいテストは CI の int32-check でも動くようにし、`.github/workflows/tests.yml` の int32-check の一覧に足す
- プリセット名・コード・UI 文言にソフトハウス名や作品名を使わない
- **他の人が作った画像やゲームの画面をリポジトリに入れない**（比較画像・テストの入力も含む。テスト画像はコードで合成し、
  比較画像は自分の AI 画像か合成画像から作る）。ゲーム名・他の人の絵のファイル名もドキュメント・コード・コミットのメッセージに書かない
- `docs/compare/` には合成画像の変換前・変換後だけを入れる
- ドキュメント・コメント・UI 文言は日本語
