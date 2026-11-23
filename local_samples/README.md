# local_samples/

手持ちの AI 画像などで Sprite App を試すためのフォルダです。**このフォルダの画像は Git に入りません**
（`.gitignore` で除外。この README だけ管理しています）。

```bash
# ドット絵風の AI 画像（既定。グリッドを検出して 1 ドット＝1 画素に戻す）
python -m apps.sprite local_samples --preview-scale 10 -o out/sprite

# 普通のイラスト（高さ 64px に縮小）
python -m apps.sprite local_samples --input-type illust --height 64 -o out/sprite
```

出力先の `out/` も Git 管理外です。
