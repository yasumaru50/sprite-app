"""Sprite App の動き（フェーズ 2、retrocore/sprite/motion/）のテスト。今あるのは呼吸だけ。

決まりごと（docs/sprite-app-plan.md の「フェーズ 2 の方針」）: 新しい色を作らない・全フレームの大きさが同じで足元が動かない・
輪郭が途切れない・迷ったら動かさない。
"""

import json

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.motion import (MOTIONS, check_rules, foot_center, make_motion, make_sheet, save_sheet,
                                     shape_signature, sheet_info)
from retrocore.sprite.motion.base import color_keys
from retrocore.sprite.motion.breathe import (DEFAULT_DURATION, DEFAULT_FRAMES, breathe, breathe_offsets,
                                             find_seam, lower_above_seam)
from synth_sprite import PIX_OUTLINE, enlarge_like_ai, make_pixel_sprite

OUT = (60, 30, 40)
BODY = (120, 210, 180)
LIGHT = (250, 245, 220)
EYE = (40, 20, 50)


def _outlined(shape: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """形の内側の縁を輪郭の色にした RGBA。"""
    pad = np.pad(shape, 1)
    edge = shape & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    out = np.zeros(shape.shape + (4,), np.uint8)
    out[..., :3] = rgb
    out[edge, :3] = OUT
    out[..., 3] = np.where(shape, 255, 0)
    out[~shape] = 0
    return out


def make_slime(w: int = 24, h: int = 15) -> np.ndarray:
    """足の無い、下が平らなスライムの形（輪郭・目・お腹の明るい部分つき）。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx = (w - 1) / 2
    shape = (((xx - cx) / (w / 2)) ** 2 + ((yy - (h - 1)) / (h - 0.5)) ** 2 < 1) | \
            ((yy >= h - 4) & (np.abs(xx - cx) < w / 2 - 1))
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = BODY
    rgb[(yy >= h - 4) & (np.abs(xx - cx) < w / 4)] = LIGHT
    for ex in (int(cx) - 3, int(cx) + 3):
        rgb[h // 2 - 1:h // 2 + 1, ex] = EYE
    return _outlined(shape, rgb)


@pytest.fixture(scope="module")
def sprites():
    """呼吸を付ける材料: 合成のキノコ（足あり）、スライム（足なし）、AI 風に拡大して変換し直したキノコ。"""
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "converted": converted}


NAMES = ["mushroom", "slime", "converted"]


# ---- 決まりごと ----

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("frames, depth", [(2, 1), (4, 2), (8, 1)])
def test_breathe_keeps_rules(sprites, name, frames, depth):
    base = sprites[name]
    res = breathe(base, frames=frames, depth=depth)
    assert not res.skipped, res.warnings
    assert len(res.frames) == frames == len(res.durations)
    assert np.array_equal(res.frames[0], base)                    # 1 枚目は元のまま
    base_colors = set(np.unique(color_keys(base)).tolist())
    for f in res.frames:
        assert f.shape == base.shape and f.dtype == np.uint8       # 大きさが同じ
        assert set(np.unique(color_keys(f)).tolist()) <= base_colors   # 新しい色が無い
        assert foot_center(f) == foot_center(base)                 # 足元（下端の中央）が同じ
        assert shape_signature(f) == shape_signature(base)         # 部品・輪郭のつながりが同じ
    assert check_rules(base, res.frames) == []
    # 下がる量はフレームごとの波のとおり。いちばん上の不透明な行が、その分だけ下がる
    top = lambda a: int(np.flatnonzero((a[..., 3] > 0).any(1))[0])  # noqa: E731
    assert [top(f) - top(base) for f in res.frames] == res.info["offsets"]
    assert max(res.info["offsets"]) == depth


@pytest.mark.parametrize("name", NAMES)
def test_breathe_keeps_feet_rows_and_moves_upper_part(sprites, name):
    """継ぎ目より下は 1 ドットも変わらない（足元の行はそのまま）。上の部分は 1 ドット下がる。"""
    base = sprites[name]
    res = breathe(base)
    seam = np.array(res.info["seams"][0])
    low = res.frames[1]
    h = base.shape[0]
    for x in range(base.shape[1]):
        y = seam[x]
        assert np.array_equal(low[y + 1:, x], base[y + 1:, x])
        assert np.array_equal(low[1:y + 1, x], base[:y, x])
    rows = np.flatnonzero((base[..., 3] > 0).any(1))
    assert seam.min() > rows[0] and seam.max() < h - 2             # 上の端でも足元でもない所で分ける
    assert np.array_equal(low[-2:], base[-2:])


def test_breathe_seam_avoids_one_dot_lines():
    """横向きの 1 ドットの線（ベルトなど）は抜かない。線のある行を避けて分ける。"""
    shape = np.zeros((20, 12), bool)
    shape[1:19, 1:11] = True
    rgb = np.zeros((20, 12, 3), np.uint8)
    rgb[:] = BODY
    rgb[12, :] = EYE                                               # 1 ドットの太さの横線
    img = _outlined(shape, rgb)
    seam, _ = find_seam(img)
    assert not np.any(seam == 12)
    res = breathe(img)
    assert not res.skipped
    assert all((f[..., :3] == EYE).all(-1).any(1).sum() == 1 for f in res.frames)   # 線は 1 本のまま


def test_breathe_skips_when_no_place_to_split():
    """1 ドットの横線が探す範囲の全行を横切っていれば、呼吸を付けずに警告する（迷ったら動かさない）。"""
    h, w = 20, 10
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[0::2, :, :3] = BODY
    img[1::2, :, :3] = EYE                                         # 1 行ごとに色が変わる
    res = breathe(img)
    assert res.skipped
    assert len(res.frames) == 1 and np.array_equal(res.frames[0], img)
    assert "呼吸を付けませんでした" in res.warnings[0]


def test_breathe_skips_tiny_sprite():
    res = breathe(make_slime(10, 6))
    assert res.skipped and res.warnings


def test_breathe_skips_empty_image():
    res = breathe(np.zeros((16, 16, 4), np.uint8))
    assert res.skipped


def test_breathe_is_deterministic(sprites):
    a = breathe(sprites["converted"], frames=4, depth=2)
    b = breathe(sprites["converted"], frames=4, depth=2)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


def test_check_rules_finds_problems(sprites):
    base = sprites["mushroom"]
    bad_color = base.copy()
    bad_color[base[..., 3] > 0] = (1, 2, 3, 255)
    assert any("色" in p for p in check_rules(base, [bad_color]))
    assert any("大きさ" in p for p in check_rules(base, [base[:-1]]))
    moved = np.roll(base, 1, axis=1)
    assert any("足元" in p for p in check_rules(base, [moved]))
    broken = base.copy()
    ys, xs = np.nonzero((base[..., :3] == PIX_OUTLINE).all(-1) & (base[..., 3] > 0))
    broken[ys[len(ys) // 2], xs[len(xs) // 2]] = 0                 # 輪郭に穴をあける
    assert check_rules(base, [broken])


def test_lower_above_seam_straight():
    img = np.zeros((4, 2, 4), np.uint8)
    img[:, :, 0] = np.arange(4)[:, None] * 10 + 10
    img[..., 3] = 255
    out = lower_above_seam(img, np.array([2, 1]))
    assert out[:, 0, 0].tolist() == [0, 10, 20, 40]
    assert out[:, 1, 0].tolist() == [0, 10, 30, 40]
    assert out[0, :, 3].tolist() == [0, 0]


# ---- フレーム数・表示時間 ----

def test_breathe_offsets():
    assert breathe_offsets(2, 1) == [0, 1]
    assert breathe_offsets(4, 2) == [0, 1, 2, 1]
    assert breathe_offsets(8, 1) == [0, 0, 0, 1, 1, 1, 0, 0]
    assert breathe_offsets(4, 1) == [0, 0, 1, 0]


def test_breathe_defaults_and_durations(sprites):
    res = make_motion(sprites["slime"], "breathe")
    assert len(res.frames) == DEFAULT_FRAMES and res.durations == [DEFAULT_DURATION] * DEFAULT_FRAMES
    res = make_motion(sprites["slime"], "breathe", frames=2, durations=[600, 400])
    assert res.durations == [600, 400]
    with pytest.raises(ValueError):
        make_motion(sprites["slime"], "breathe", frames=3, durations=[600, 400])
    with pytest.raises(ValueError):
        make_motion(sprites["slime"], "breathe", frames=1)
    with pytest.raises(ValueError):
        make_motion(sprites["slime"], "breathe", depth=3)
    with pytest.raises(ValueError):
        make_motion(sprites["slime"], "wobble")
    assert list(MOTIONS) == ["breathe", "sway", "float", "bounce", "nod", "stretch"]


# ---- 書き出し ----

def test_sheet_and_json(tmp_path, sprites):
    res = breathe(sprites["mushroom"], frames=4, durations=[400, 200, 400, 200], depth=2)
    sheet = make_sheet(res.frames)
    h, w = sprites["mushroom"].shape[:2]
    assert sheet.shape == (h, w * 4, 4)
    for i, f in enumerate(res.frames):
        assert np.array_equal(sheet[:, i * w:(i + 1) * w], f)
    png, js = save_sheet(res, tmp_path / "hero_breathe_sheet.png")
    assert js.name == "hero_breathe_sheet.json"
    assert np.array_equal(np.asarray(Image.open(png).convert("RGBA")), sheet)
    info = json.loads(js.read_text(encoding="utf-8"))
    assert info == sheet_info(res, "hero_breathe_sheet.png")
    assert info["frame_width"] == w and info["frame_height"] == h and info["frame_count"] == 4
    assert info["columns"] == 4 and info["rows"] == 1 and info["loop"] is True
    assert info["pivot"] == {"x": w / 2, "y": h}
    assert [f["duration_ms"] for f in info["frames"]] == [400, 200, 400, 200]
    assert [f["x"] for f in info["frames"]] == [0, w, 2 * w, 3 * w]
    assert info["total_duration_ms"] == 1200


def test_make_sheet_rejects_mixed_sizes():
    with pytest.raises(ValueError):
        make_sheet([np.zeros((4, 4, 4), np.uint8), np.zeros((4, 5, 4), np.uint8)])


# ---- CLI ----

def _ai_input(tmp_path):
    path = tmp_path / "in" / "chara.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_without_motion_is_unchanged(tmp_path):
    """--motion を付けないときは、出力もファイルの数も今までと同じ（動きのファイルを書かない）。"""
    src = _ai_input(tmp_path)
    assert cli.main([str(src), "-o", str(tmp_path / "a" / "chara.png"), "-q"]) == 0
    assert cli.main([str(src), "-o", str(tmp_path / "b" / "chara.png"), "-q", "--motion", "breathe"]) == 0
    a = (tmp_path / "a" / "chara.png").read_bytes()
    assert a == (tmp_path / "b" / "chara.png").read_bytes()        # 変換の結果は --motion で変わらない
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["chara.png"]
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == [
        "chara.png", "chara_breathe_sheet.json", "chara_breathe_sheet.png"]
    expected = convert_sprite(np.asarray(Image.open(src).convert("RGBA")), SpriteSettings()).image
    assert np.array_equal(np.asarray(Image.open(tmp_path / "a" / "chara.png").convert("RGBA")), expected)


def test_cli_motion_options(tmp_path, capsys):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "--motion", "breathe", "--motion-frames", "4",
                     "--breathe-depth", "2", "--motion-duration", "300,200,300,200"]) == 0
    assert "chara_breathe_sheet.png" in capsys.readouterr().out
    info = json.loads((out.parent / "chara_breathe_sheet.json").read_text(encoding="utf-8"))
    assert info["frame_count"] == 4 and [f["duration_ms"] for f in info["frames"]] == [300, 200, 300, 200]
    sheet = np.asarray(Image.open(out.parent / "chara_breathe_sheet.png").convert("RGBA"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    assert sheet.shape == (sprite.shape[0], sprite.shape[1] * 4, 4)
    assert np.array_equal(sheet[:, :sprite.shape[1]], sprite)


@pytest.mark.parametrize("extra, words", [
    (["--motion-frames", "4"], "動きのオプション"),
    (["--breathe-depth", "2"], "動きのオプション"),
    (["--motion", "breathe", "--motion-duration", "300,200,100"], "フレーム数"),
    (["--motion", "breathe", "--motion-frames", "1"], "2〜"),
    (["--motion", "breathe", "--breathe-depth", "3"], "1〜2"),
    (["--motion", "breathe", "--motion-duration", "0"], "1 ミリ秒"),
    (["--motion", "jump"], "invalid choice"),
])
def test_cli_rejects_bad_motion_options(tmp_path, capsys, extra, words):
    src = _ai_input(tmp_path)
    with pytest.raises(SystemExit):
        cli.main([str(src), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_motion_skipped_warns_and_keeps_sprite(tmp_path, capsys):
    """呼吸を付けられない絵では、警告を出してシートを書かない（変換の結果は保存する）。"""
    img = np.zeros((20, 10, 4), np.uint8)
    img[..., 3] = 255
    img[0::2, :, :3] = BODY
    img[1::2, :, :3] = EYE
    args = cli.build_parser().parse_args(["x.png", "--motion", "breathe", "-q"])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, img, dst, dst) == 1
    assert "呼吸を付けませんでした" in capsys.readouterr().err
    assert not (tmp_path / "s_breathe_sheet.png").exists()
