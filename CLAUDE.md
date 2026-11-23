# CLAUDE.md

このリポジトリで作業するときの手引き。

## 概要

**Sprite App** — AI が描いた「ドット絵風の画像」から、本物のドット絵のスプライトと、その動き（呼吸・横揺れ・浮遊・弾む・うなずく・伸び）の
スプライトシートを作るアプリ。CLI（`python -m apps.sprite`）と、Pyodide で動く 1 枚の HTML のブラウザ版がある。
技術スタック: Python 3.10+ / NumPy / Pillow / Pyodide。

前のリポジトリ（古いパソコン風の CG 変換アプリと一緒に開発していた所）から Sprite App の分だけを移したもの。
文書に出てくる PR の番号（#1〜#45）と ai01〜ai09 の比較画像は前のリポジトリのもので、**このリポジトリには前の比較画像は無い**。

## 最初に読むもの

- **作業に要るファイルだけを読む**。文書は下の「ファイルの地図」を見て、必要な所だけを開く（全部を読み通さない）
- 残っている作業は `docs/TODO.md`、決まりと方針は `docs/sprite-app-plan.md`。`docs/archive/` は昔の記録なので、調べ物で要るときだけ

## ファイルの地図

| 知りたいこと | 場所 |
| --- | --- |
| 残っている作業 | `docs/TODO.md` |
| 方針・決まり・動きの表の設計・skipped の code の一覧 | `docs/sprite-app-plan.md` |
| 工程の仕組みと値を決めた理由、処理のファイルの一覧 | `docs/how-it-works.md` |
| CLI のオプション、出力のファイル名、シートの JSON の形 | `docs/cli.md` |
| ブラウザ版の使い方・よく使う CLI | `apps/sprite/README.md`（全体の紹介は `README.md`） |
| Piskel での手直しと読み戻しの手順 | `docs/sprite-edit.md` |
| これまでの PR のまとめ | `docs/history.md` |
| 外部調査 | `docs/research-2026-11.md` |
| 昔の設計メモ・引き継ぎの全文（検討・試した数） | `docs/archive/` |

| 機能 | コード |
| --- | --- |
| CLI | `apps/sprite/cli.py` |
| 共通の入口（CLI と画面）・工程の組み立て `build_pipeline()`・画面用の `*_for_web()` | `retrocore/sprite/convert.py` |
| 背景除去 / グリッド検出 / 縮小 | `retrocore/sprite/background.py` / `grid.py` / `resize.py`・`downscale.py` |
| 単純化 / クリーンアップ / 縁取り / パレット | `retrocore/sprite/simplify.py` / `cleanup.py` / `outline.py` / `palette.py` |
| コマが並んだ絵（`--frames`）/ ちらつきを減らす / 1 コマ目と同じにする範囲 / 手の直し | `retrocore/sprite/frames.py` / `steady.py` / `same.py` / `touchup.py` |
| 差し替え（まばたき。CLI `--swaps`、画面の「差し替えを描く」） | `retrocore/sprite/swap.py` |
| 手直しの読み戻し / .piskel | `retrocore/sprite/edit.py` / `piskel.py` |
| 動き（表の形・表からフレーム・決まり・各プリセット・範囲・skipped・シート・GIF） | `retrocore/sprite/motion/`（`spec.py`・`timeline.py`・`base.py`・`breathe.py` ほか・`region.py`・`report.py`・`sheet.py`・`gif.py`） |
| ブラウザ版 | `apps/sprite/web/template.html`（→ `tools/build_sprite_web.py` で `sprite-app.html` を作る。直接編集しない） |
| 動きの表の見本 | `apps/sprite/examples/*.motion.json`、テスト用は `tests/motion_specs/` |
| 比較画像を作る道具 | `tools/compare_sprite_*.py`（背景・グリッド・縮小・単純化・クリーンアップ・縁取り・動き・コマ・ちらつき・範囲） |
| テスト | `tests/`（テスト画像は `tests/synth_sprite.py`・`tests/synth_cg.py` で合成。`web_region/` は画面で保存した表・シート） |
| 古いパソコン風の CG 変換（Sprite App からは使っていない。ブラウザ版の HTML には入る） | `retrocore/` のほかのファイル（`pipeline.py` など）。テストは `tests/test_golden.py`（期待値 `golden/`、入力 `fixtures/`）。`tools/analyze.py`・`export_db.py`・`compare.py`・`make_golden.py` はそのテストが使う |
| CI | `.github/workflows/tests.yml`（int32-check と、すべてのテスト） |

出力先の `out/` と `local_samples/` の中身（手持ちの AI 画像）は Git に入らない。

## 作業の進め方

- 1 つの作業 ＝ 1 つのセッション ＝ 1 つの PR。作業が終わったら PR を作り、報告して止まる
- **PR を作ったら見張らない・後で見回らない**。CI の結果は利用者が伝える（CI がすべて ✓ になってからマージする）
- **報告は要点だけ短く**。日本語で、初心者にも分かる言葉で書く
- **比較画像は指示で頼まれたものだけ作る**。置き場所は `preview/ai-check` ブランチの `ai-output/<作業ごとのフォルダ>/`（README に感想・数）。
  入力の AI 画像は同じブランチの `ai-input/`。このブランチは main にマージしない（前のリポジトリの `ai-output/` はここには無い）
- **ヘッドレスブラウザでの確かめは、画面（`template.html`）を変えたときだけ**
- フェーズ（作業単位）ごとにコミットする

## 開発コマンド

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m apps.sprite 入力.png -o out/sprite/出力.png   # 例はほかに apps/sprite/README.md、オプションは docs/cli.md
python tools/build_sprite_web.py                        # ブラウザ版の HTML を作り直す（--check で古くないか確かめる）
```

## 開発ルール

- Sprite App の動き（フェーズ 2）は `retrocore/sprite/motion/` に置き、変換の工程には手を入れない（変換後のスプライトが材料）。
  新しい色を作らない・大きさと足元をそろえる・輪郭を途切れさせない・迷ったら動かさない（`check_rules()` で確かめる）。
  足元が動く動き（浮遊・弾む）は例外として、足元が縦に決まった量だけ動くことだけを許す（`check_rules(..., shifts=...)`。ほかの決まりは同じ）
  - 新しい動きは、なるべく動きの表（`timeline.py`）の列か、表を埋めるプリセットとして足す。足元が動いてよいのは表の `root`（体全体のずれ）の列だけ
  - 伸ばす（縦の変形の +）ではみ出す分は上に透明の行を足す（`padding.top`。pivot はフレームの下端のまま）。「小さな部品を守る」は既定で切（入のときだけ表に書く）
  - 範囲は分け目（`split`）・帯（`band`）・四角（`rect`）。帯・四角は囲んだ中だけが動き、境目で作れない（空いた所の色が分からない・外の違う色を隠す）ときは作らずに警告する
  - プリセットの出力（PNG の画素・JSON のバイト・警告の言葉）は変えない（`tests/test_sprite_timeline.py` がこの前のハッシュと比べる。範囲を書いていない表も含めて `tests/test_sprite_region.py` も）
  - 手直しの読み戻し（`edit.py`）は直したドットをそのまま使い、新しい色・半透明などは作って警告する（直すのは利用者が選んだときだけ）
  - 作れなかった理由は新しい判定を足さずに記録だけする（`report.py`。code は表の問題の `rule` と同じ名前。code を足したら `docs/sprite-app-plan.md` の一覧にも書く。
    `tests/test_sprite_skipped.py` がこのファイルを読んで確かめる）。前のハッシュと比べるテストは、版 2 で足した分（`skipped`・`_skipped.json`）を `tests/legacy_output.py` で除いて比べる
- Sprite App の新しい処理は `retrocore/sprite/` に置く。工程は `retrocore/sprite/convert.py` の `build_pipeline()` に足す（CLI と画面で共通）。
  retrocore を変えたら `python tools/build_sprite_web.py` でブラウザ版の HTML も作り直す（テストで検出される）。既存の関数の挙動は変えない（ゴールデンは既存分を再生成しない）
- アプリ内の手の直し（`touchup.py`、CLI `--edits`、画面の「ドットを直す」）は変換・範囲・ちらつきの後に当て、パレットに無い色は作らない。
  直しが無いときの出力は前とバイト単位で同じ（`tests/test_sprite_touchup.py`）
- `retrocore/` は NumPy と Pillow だけに依存させる（Pyodide で動かすため。PySide6・SciPy・OpenCV も不可。`tests/test_retrocore_deps.py` が確かめる）
- Pyodide は 32 ビット（`np.intp` が int32）。`np.bincount`・`np.repeat`・`reduceat` などに int64 の配列を渡さない（`.astype(np.intp)`）。
  `tests/test_sprite_int32.py` が確かめる（呼ぶ場所を足したら、その中の一覧 `REVIEWED` も更新）
- retrocore の CG 変換の部分（`retrocore/pipeline.py` など）の挙動は変えない
  （`tests/test_golden.py` が合成画像の変換結果と画素単位で比較する。意図して変えたときだけ `python tools/make_golden.py` で更新）
- 重い処理はベクトル化する（ブラウザ版の Pyodide でも数秒で終わるように）
- 新しいモジュールには pytest を付け、テスト画像はコードで合成する。Sprite App の新しいテストは CI の int32-check でも動くようにし、`.github/workflows/tests.yml` の int32-check の一覧に足す
- プリセット名・コード・UI 文言にソフトハウス名や作品名を使わない
- **他の人が作った画像やゲームの画面をリポジトリに入れない**（比較画像・テストの入力も含む。テスト画像はコードで合成し、
  比較画像は自分の AI 画像か合成画像から作る）。ゲーム名・他の人の絵のファイル名もドキュメント・コード・コミットのメッセージに書かない
- `docs/compare/` には合成画像の変換前・変換後だけを入れる
- ドキュメント・コメント・UI 文言は日本語
- 文書を変えたら、終わったことは `docs/history.md` に、残ったことは `docs/TODO.md` に書く（TODO には残っていることだけを置く）
