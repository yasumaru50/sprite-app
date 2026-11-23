# テスト用の動きの表

`tests/test_sprite_timeline.py` が `--motion-spec` で読ませる表（sprite-motion の JSON、version 1）。プリセットには入れない。
形は `docs/sprite-app-plan.md` の「動きの表（タイムライン）の設計」。

| ファイル | 中身 |
| --- | --- |
| `bounce.motion.json` | 弾む相当: 体全体のずれ 0, −1, −2, −1（上にだけ動く。余白は上だけ） |
| `bounce_squash.motion.json` | 上と同じで、着地の 1 行目だけ縮める（縦の変形 −1）。ポーズの確かめと体全体のずれの確かめの組み合わせ |
| `broken.motion.json` | わざと作れない表: 足元の中の高さ（下から 3 ドット目）で縮める（−1）・伸ばす（+1）。警告の文の確かめ用（伸ばすは作れるようになったので、−1 と同じ問題にまとまる） |
