"""Sprite App のテスト用画像を合成する（AI 画像風のキャラクター）。

単色背景の上に、頭（円）・胴（楕円）・脚・腕の簡単なキャラクターを置き、
肌と服には左上からの光のグラデーションを付ける。輪郭は濃い線。
"""

import numpy as np

BG = (90, 170, 120)          # 単色背景（緑）
OUTLINE = (30, 20, 30)
SKIN = np.array([250, 205, 170], float)
CLOTH = np.array([70, 90, 210], float)
LEG = np.array([60, 50, 60], float)


def _shade(base, yy, xx, cy, cx, r):
    """左上からの光: 中心から左上方向ほど明るく、右下ほど暗い（0.55〜1.1 倍）。"""
    d = ((yy - cy) + (xx - cx)) / (2.0 * r)
    k = np.clip(0.85 - 0.3 * d, 0.55, 1.1)
    return np.clip(base[None, None, :] * k[..., None], 0, 255)


def make_character(w: int = 256, h: int = 384, transparent: bool = False) -> np.ndarray:
    """H×W×4 の RGBA uint8。`transparent=True` なら背景を透明にする。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    s = min(w / 256, h / 384)
    cx = w / 2
    rgb = np.empty((h, w, 3), float)
    rgb[:] = BG
    body = np.zeros((h, w), bool)

    def put(mask, color):
        rgb[mask] = color[mask] if color.ndim == 3 else color
        body[mask] = True

    def with_outline(masks, color):
        # 輪郭: 外側の図形を線の色で塗ってから、内側を中身の色で塗る
        outer, inner = masks
        put(outer, np.array(OUTLINE, float))
        put(inner, color)

    # 脚
    for dx in (-22, 22):
        lx = cx + dx * s
        outer = (np.abs(xx - lx) < 14 * s) & (yy > 250 * s) & (yy < 360 * s)
        inner = (np.abs(xx - lx) < 11 * s) & (yy > 253 * s) & (yy < 357 * s)
        with_outline((outer, inner), _shade(LEG, yy, xx, 300 * s, lx, 30 * s))
    # 腕
    for dx in (-62, 62):
        ax = cx + dx * s
        outer = ((xx - ax) / (14 * s)) ** 2 + ((yy - 210 * s) / (55 * s)) ** 2 < 1
        inner = ((xx - ax) / (11 * s)) ** 2 + ((yy - 210 * s) / (52 * s)) ** 2 < 1
        with_outline((outer, inner), _shade(SKIN, yy, xx, 210 * s, ax, 50 * s))
    # 胴
    by = 200 * s
    outer = ((xx - cx) / (55 * s)) ** 2 + ((yy - by) / (70 * s)) ** 2 < 1
    inner = ((xx - cx) / (52 * s)) ** 2 + ((yy - by) / (67 * s)) ** 2 < 1
    with_outline((outer, inner), _shade(CLOTH, yy, xx, by, cx, 60 * s))
    # 頭
    hy, r = 95 * s, 60 * s
    dist = np.hypot(xx - cx, yy - hy)
    with_outline((dist < r, dist < r - 3 * s), _shade(SKIN, yy, xx, hy, cx, r))
    # 目
    for dx in (-20, 20):
        eye = (np.abs(xx - (cx + dx * s)) < 4 * s) & (np.abs(yy - (hy + 5 * s)) < 8 * s)
        put(eye, np.array(OUTLINE, float))

    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.round(rgb).astype(np.uint8)
    out[..., 3] = np.where(body, 255, 0) if transparent else 255
    return out


def character_mask(w: int = 256, h: int = 384) -> np.ndarray:
    """キャラクターの画素（背景以外）の真偽マスク。"""
    return make_character(w, h, transparent=True)[..., 3] > 0


def add_noise(rgba: np.ndarray, amp: int = 6, seed: int = 0) -> np.ndarray:
    """AI 画像のように同じ色がほぼ繰り返さないよう、RGB に ±amp の揺らぎを加える（seed で決定的）。"""
    rng = np.random.default_rng(seed)
    out = rgba.copy()
    n = rng.integers(-amp, amp + 1, rgba.shape[:2] + (3,))
    out[..., :3] = np.clip(rgba[..., :3].astype(int) + n, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- ドット絵風の AI 画像（グリッド検出用）

PIX_BG = (212, 246, 189)        # 背景（薄い緑）
PIX_OUTLINE = (80, 30, 40)      # 輪郭（暗い赤茶）
PIX_CAP = (240, 120, 80)        # かさ
PIX_CAP_DARK = (200, 80, 70)    # かさの影
PIX_SPOT = (250, 240, 210)      # かさの斑点
PIX_BODY = (250, 240, 210)      # 体
PIX_BODY_DARK = (230, 180, 160)  # 体の影
PIX_EYE = (70, 50, 60)          # 目
PIX_CHEEK = (250, 180, 190)     # 頬


def make_pixel_sprite(w: int = 22, h: int = 24) -> np.ndarray:
    """キノコ風の小さなドット絵（約 h×w の RGBA、1 ドット＝1 画素、外側は透明。不透明な部分に切り詰める）。

    かさ（半楕円）に斑点と影、体（楕円）に影、1 ドットの目、2 ドットの頬。輪郭は 1 ドット幅の暗い赤茶。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx = (w - 1) / 2
    cap_h = int(h * 0.55)
    cap = (((xx - cx) / (w / 2 - 0.6)) ** 2 + ((yy - cap_h) / (cap_h - 0.4)) ** 2 < 1) & (yy <= cap_h)
    body = (((xx - cx) / (w * 0.3)) ** 2 + ((yy - (cap_h + (h - cap_h) / 2 - 0.5)) / ((h - cap_h) / 2 + 1.2)) ** 2 < 1) \
        & (yy > cap_h - 2) & (yy < h - 1)
    shape = cap | body
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[body] = PIX_BODY
    rgb[body & (xx > cx + w * 0.15)] = PIX_BODY_DARK
    rgb[cap] = PIX_CAP
    rgb[cap & (yy >= cap_h - 2)] = PIX_CAP_DARK
    for sy, sx in ((3, int(cx) - 4), (5, int(cx) + 3), (8, int(cx) - 1)):
        rgb[sy:sy + 2, sx:sx + 2][cap[sy:sy + 2, sx:sx + 2]] = PIX_SPOT
    ey = cap_h + 2
    for ex in (int(cx) - 2, int(cx) + 3):
        rgb[ey, ex] = PIX_EYE
    rgb[ey + 1, int(cx) - 4:int(cx) - 2] = PIX_CHEEK
    rgb[ey + 1, int(cx) + 4:int(cx) + 6] = PIX_CHEEK
    # 輪郭: 形の内側で、上下左右のどれかが外側の画素
    pad = np.pad(shape, 1)
    edge = shape & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    rgb[edge] = PIX_OUTLINE
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = rgb
    out[..., 3] = np.where(shape, 255, 0)
    out[~shape] = 0
    rows = np.nonzero(shape.any(1))[0]
    cols = np.nonzero(shape.any(0))[0]
    return out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()  # 不透明な部分に切り詰める


def _box_blur(rgb: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return rgb.astype(float)
    a = rgb.astype(float)
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = a.shape[axis]
        a = (np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
             - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return a


def enlarge_like_ai(art: np.ndarray, sizes=(19, 22), margin: int = 3, blur: int = 1, noise: int = 6,
                    half_shift: int = 6, bg=PIX_BG, seed: int = 0, sizes_y=None):
    """ドット絵を、AI が描いたドット絵風の画像のように拡大する。

    - 列の幅・行の高さを `sizes` の範囲（両端を含む）でマスごとにばらつかせる。`sizes_y` を指定すると
      行の高さはその範囲にする（横長・縦長のドット）
    - 周りに `margin` ドット分の背景（不透明な単色）を付ける
    - 輪郭のドットのうち `half_shift` 個を、半ドット分横にずらして描く（輪郭の半ドットのずれ）
    - 半径 `blur` のぼかしと ±`noise` のノイズを加える

    戻り値は (RGB 不透明の RGBA 画像, 列の境目, 行の境目)。境目は元のドットの位置（背景の余白を含む）。
    """
    rng = np.random.default_rng(seed)
    h, w = art.shape[:2]
    full = np.zeros((h + 2 * margin, w + 2 * margin, 4), np.uint8)
    full[margin:margin + h, margin:margin + w] = art
    fh, fw = full.shape[:2]
    cw = rng.integers(sizes[0], sizes[1] + 1, fw)
    sy = sizes if sizes_y is None else sizes_y
    ch = rng.integers(sy[0], sy[1] + 1, fh)
    xs = np.concatenate([[0], np.cumsum(cw)])
    ys = np.concatenate([[0], np.cumsum(ch)])
    rgb = np.repeat(np.repeat(full[..., :3], ch, axis=0), cw, axis=1).astype(np.uint8)
    alpha = np.repeat(np.repeat(full[..., 3], ch, axis=0), cw, axis=1)
    rgb[alpha == 0] = bg
    # 輪郭の半ドットのずれ: 輪郭のドットを、右隣が輪郭でない所で半ドット右へはみ出させる
    outline = (full[..., 3] > 0) & (np.abs(full[..., :3].astype(int) - PIX_OUTLINE).max(-1) == 0)
    cand = np.argwhere(outline[:, :-1] & ~outline[:, 1:])
    if len(cand) and half_shift > 0:
        pick = cand[rng.choice(len(cand), min(half_shift, len(cand)), replace=False)]
        for r, c in pick:
            half = cw[c + 1] // 2 - 1  # 半分より少し少なく（マスの過半数は元の色のまま）
            rgb[ys[r]:ys[r + 1], xs[c + 1]:xs[c + 1] + half] = PIX_OUTLINE
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    blurred = _box_blur(rgb, blur)
    n = rng.integers(-noise, noise + 1, rgb.shape) if noise > 0 else 0
    out[..., :3] = np.clip(np.rint(blurred) + n, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out, xs, ys
