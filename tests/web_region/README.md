# 画面で作った範囲の fixture

`tests/test_sprite_web_region.py` が使う。ブラウザ版（`apps/sprite/web/sprite-app.html`）をヘッドレスの Chromium で動かし、
合成の騎士（`tests/motion_records.py` の `make_knight()` を 20 倍に拡大した PNG）を変換して、「表で編集」で範囲を描いて保存したもの。

| ファイル | 中身 |
| --- | --- |
| `web_rect.*` | 呼吸の表を開き、絵の上でドラッグして頭の四角（左 5〜右 24・下 10〜上 22）を描き、upper_y をその四角で動かした表（表示時間 600・200） |
| `web_band.*` | さらに「＋ 帯」で帯を足し、入力欄で下 10〜上 22 にし、名前を head_band に変え、列の見出しで upper_y を帯にして、3 行目に root_y −1 を足した表 |
| `*.number.motion.json` / `*.percent.motion.json` | 画面の「表を保存…」で数・割合で保存した表 |
| `*_sheet.png` / `*_sheet.json` | 画面で保存したスプライトシート |
| `make_fixture.js` | 作り直すスクリプト（Node と Playwright。Pyodide をインターネットから読む） |

作り直し方（画面の作りを変えて、保存する中身が変わるはずのときだけ）:

```bash
python -c "import sys; sys.path[:0]=['.','tests']; import numpy as np; from PIL import Image; from motion_records import make_knight; Image.fromarray(np.kron(make_knight(), np.ones((20,20,1),np.uint8)),'RGBA').save('out/knight.png')"
node tests/web_region/make_fixture.js "$PWD/apps/sprite/web/sprite-app.html" "$PWD/out/knight.png" "$PWD/tests/web_region"
```
