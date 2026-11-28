"""Still App 用の処理（NumPy と Pillow のみ）。AI が描いた「ドット絵風の 1 枚絵」を、決まった色数・格子の本物のドット絵にそろえる。

- `photo.py`: 写真の下書き（`--mode photo`。向きを直す・先に縮める・桑原フィルタで色の塊をまとめる）
- `outline.py`: 輪郭の線（`--outline`。明るさの差が大きい境目の暗い方に、いちばん暗い色で 1 ドット）
- `dots.py`: 透明を黒で埋め、ドットに戻す（Sprite App と同じグリッド検出か、横のマス数の指定）。マスの色はいちばん多い色
- `colors.py`: 減色（4 ビットの丸め・いちばん暗い色を取っておく・Sprite App の単純化で色を選ぶ）
- `pattern.py`: ざらついた範囲を 2 色の決まった模様（2×2）に整える（`--tidy-pattern`。既定は切）
- `convert.py`: 入口（設定・プリセット `--mode`・大きさの上限・キャンバス・PNG と JSON・ブラウザ版向け）

Sprite App の処理（`retrocore.sprite`）は使うだけで、変えない。
"""

from .colors import Reduced, reduce_colors, round_4bit
from .convert import (MODES, StillError, StillResult, StillSettings, convert_still, convert_still_bytes,
                      encode_indexed_png, still_for_web)
from .dots import Dots, flatten_black, grid_dots, mode_colors, width_dots

__all__ = ["MODES", "Dots", "Reduced", "StillError", "StillResult", "StillSettings", "convert_still",
           "convert_still_bytes", "encode_indexed_png", "flatten_black", "grid_dots", "mode_colors",
           "reduce_colors", "round_4bit", "still_for_web", "width_dots"]
