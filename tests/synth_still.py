"""Still App のテスト用の合成画像（コードで作る。他の人の絵は使わない）。"""

from __future__ import annotations

import numpy as np

from synth_sprite import enlarge_like_ai

SKY = (90, 150, 230)
GROUND = (200, 180, 140)
WALL = (230, 225, 210)
ROOF = (160, 80, 60)
WINDOW = (80, 110, 170)
TREE = (70, 150, 80)
POST = (220, 60, 30)
LINE = (20, 20, 24)
GROUND_DARK = (160, 140, 110)
WALL_DARK = (190, 180, 160)
CLOUD = (240, 240, 235)


def make_scene(w: int = 48, h: int = 30) -> np.ndarray:
    """背景のある小さな 1 枚絵のドット絵（h×w の RGBA、全部不透明。1 ドット＝1 画素）。

    空（雲）・地面（点々）・建物（板目の壁・屋根・窓）・木・小さな赤いポスト。輪郭と枠は 1 ドットの暗い線。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.65)
    rgb[:] = SKY
    rgb[yy >= horizon] = GROUND
    # 地面の点々・雲（ドットの境目を増やす。位置は決まった乱数）
    rng = np.random.default_rng(1)
    dots = (yy >= horizon) & (rng.random((h, w)) < 0.2)
    rgb[dots] = GROUND_DARK
    for cx, cy in ((int(w * 0.2), 3), (int(w * 0.6), 4)):
        rgb[cy:cy + 2, cx:cx + 6] = CLOUD
        rgb[cy - 1, cx + 2:cx + 4] = CLOUD
    bx0, bx1, by0 = int(w * 0.3), int(w * 0.75), int(h * 0.35)
    body = (xx >= bx0) & (xx < bx1) & (yy >= by0) & (yy < horizon)
    rgb[body] = WALL
    rgb[body & (xx % 3 == 0)] = WALL_DARK   # 板目
    roof = (yy >= by0 - 5) & (yy < by0) & (np.abs(xx - (bx0 + bx1) / 2) <= (bx1 - bx0) / 2 + (yy - (by0 - 5)))
    rgb[roof] = ROOF
    for wx in (bx0 + 3, bx1 - 7):
        rgb[by0 + 3:by0 + 7, wx:wx + 4] = WINDOW
    tree = ((xx - int(w * 0.12)) ** 2 + (yy - int(h * 0.4)) ** 2) <= 20
    rgb[tree] = TREE
    rgb[horizon - 4:horizon, int(w * 0.85):int(w * 0.85) + 2] = POST
    # 輪郭: 色が変わる所の、上・左の側に 1 ドットの線（空と地面の境は除く）
    shape = body | roof | tree
    pad = np.pad(shape, 1)
    edge = shape & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    rgb[edge] = LINE
    # 1 ドットの枠（AI の 1 枚絵によくある黒い枠。端のマス目もはっきりする）
    rgb[[0, -1], :] = LINE
    rgb[:, [0, -1]] = LINE
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out


def enlarge_scene(art: np.ndarray, sizes=(9, 11), seed: int = 0) -> np.ndarray:
    """AI のドット絵風の 1 枚絵のように拡大する（ドットの大きさのばらつき・ぼかし・ノイズ。余白なし）。"""
    return enlarge_like_ai(art, sizes=sizes, margin=0, half_shift=0, seed=seed)[0]
