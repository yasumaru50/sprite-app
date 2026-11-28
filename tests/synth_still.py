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


def shuffled_level(h: int, w: int, level: int, seed: int = 0) -> np.ndarray:
    """2×2 のかたまりごとに、明るいドットをちょうど level（0〜4）個、場所はばらばらに置いた真偽の配列。

    濃さは決まった段のまま、並びは格子に乗っていない（AI の「ディザ模様」を減色した後のようなざらつき）。
    """
    rng = np.random.default_rng(seed)
    bh, bw = (h + 1) // 2, (w + 1) // 2
    keys = rng.random((bh, bw, 4))
    order = np.argsort(np.argsort(keys, axis=-1), axis=-1)   # かたまりの中の 4 ドットの順位
    light = (order < level).reshape(bh, bw, 2, 2).transpose(0, 2, 1, 3).reshape(bh * 2, bw * 2)
    return light[:h, :w]


NOISY_PALETTE = np.array([[16, 16, 16], [60, 70, 60], [120, 120, 70], [170, 150, 90], [70, 60, 140],
                          [200, 90, 60]], np.uint8)


def make_noisy_scene(w: int = 64, h: int = 48) -> tuple[np.ndarray, dict]:
    """ざらついた範囲のある 1 枚絵（パレットの番号の配列と、各部分の場所）。1 ドット＝1 画素、NOISY_PALETTE の色。

    - 上の左: 色 2・3 の 25%（ざらつき）。上の右: 色 1・2 の 50%。境目は x=32
    - 下の左: 色 2・3 の 75% の中に、横切る暗い線（色 0）
    - 下の右: 色 1・3 の 50% に、3 色目（色 5）の 2×2 のかたまりが 2 割（描き込み）
    - いちばん下の 6 行: 平塗り（色 4）。外枠は暗い線（色 0）
    """
    idx = np.zeros((h, w), np.uint8)
    top, mid = slice(1, 22), slice(22, h - 7)
    left, right = slice(1, 32), slice(32, w - 1)
    idx[top, left] = np.where(shuffled_level(h, w, 1, seed=1)[top, left], 3, 2)
    idx[top, right] = np.where(shuffled_level(h, w, 2, seed=2)[top, right], 2, 1)
    idx[mid, left] = np.where(shuffled_level(h, w, 3, seed=3)[mid, left], 3, 2)
    idx[mid, right] = np.where(shuffled_level(h, w, 2, seed=4)[mid, right], 3, 1)
    rng = np.random.default_rng(5)
    for y in range(22, h - 8, 2):
        for x in range(32, w - 2, 2):
            if rng.random() < 0.2:
                idx[y:y + 2, x:x + 2] = 5
    idx[h - 7:h - 1, 1:w - 1] = 4
    idx[32, 1:32] = 0                                  # 暗い線（輪郭）
    idx[0, :] = idx[-1, :] = 0
    idx[:, 0] = idx[:, -1] = 0
    parts = {"top_left": (top, left), "top_right": (top, right), "mid_left": (mid, left), "mid_right": (mid, right),
             "flat": (slice(h - 7, h - 1), slice(1, w - 1)), "line": (32, slice(1, 32))}
    return idx, parts


def noisy_scene_rgba() -> np.ndarray:
    idx, _ = make_noisy_scene()
    out = np.empty(idx.shape + (4,), np.uint8)
    out[..., :3] = NOISY_PALETTE[idx]
    out[..., 3] = 255
    return out


# 今風モードのテスト用の色。WALL_NEAR は WALL と、SKY_NEAR は SKY と見た目でほぼ同じ（OKLab の距離 0.04 未満）
MODERN_LINE = (12, 12, 16)
MODERN_WALL = (200, 170, 130)
MODERN_WALL_NEAR = (214, 176, 124)
MODERN_SKY = (90, 140, 210)
MODERN_SKY_NEAR = (104, 140, 196)
MODERN_SPECK = (230, 60, 160)   # にじみの点（どの色とも遠い）
MODERN_FLOOR = (90, 70, 50)


def make_modern_scene(w: int = 64, h: int = 40) -> tuple[np.ndarray, dict]:
    """今風モードのテスト用の 1 枚絵（h×w×3 の RGB。1 ドット＝1 画素）と、置いた点の場所。

    上は空、下は壁、いちばん下は床。空と壁の境目は暗い 1 ドットの線。壁の中に少し違う色（WALL_NEAR）の四角、
    空の中に少し違う色（SKY_NEAR）の小さな四角。孤立した点: 空の中・壁の中の 1 ドット（SPECK）、絵の端の 1 ドット（変えない）、
    空の中の暗い線の色の 1 ドット（変えない）、2 ドットならんだ点（どちらもまわりに同じ色があるので変えない）。
    """
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = MODERN_SKY
    rgb[16:] = MODERN_WALL
    rgb[16] = MODERN_LINE
    rgb[34:] = MODERN_FLOOR
    rgb[22:30, 30:44] = MODERN_WALL_NEAR      # 8×14 = 112 ドット
    rgb[3:6, 40:46] = MODERN_SKY_NEAR         # 3×6 = 18 ドット
    spots = {"sky": (8, 10), "wall": (25, 10), "edge": (0, 20), "dark": (8, 50), "pair": [(10, 25), (10, 26)]}
    for key in ("sky", "wall", "edge"):
        rgb[spots[key]] = MODERN_SPECK
    rgb[spots["dark"]] = MODERN_LINE
    for yx in spots["pair"]:
        rgb[yx] = MODERN_SPECK
    return rgb, spots
