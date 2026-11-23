"""横揺れ（`retrocore/sprite/motion/sway.py`、CLI `--motion sway`）のテスト。

決まりごと（docs/sprite-app-plan.md の「フェーズ 2 の方針」の「横揺れ」）: 新しい色を作らない・全フレームの大きさが同じ・
足元を動かさない・隙間を作らない・輪郭を途切れさせない・迷ったら動かさない。
あわせて、--motion を付けないときと --motion breathe の CLI の出力が、この PR の前（PR #23 の main）と同じことをハッシュで確かめる。
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.motion import MOTIONS, check_rules, foot_center, make_motion, save_sheet, sheet_info
from retrocore.sprite.motion.base import color_keys, shape_signature, split_limits
from retrocore.sprite.motion.sway import (DEFAULT_AMOUNT, DEFAULT_DURATION, DEFAULT_FRAMES, SPLIT_RANGE, euler_numbers,
                                          nearby_splits, shift_above, sway, sway_offsets)
from legacy_output import legacy_bytes, legacy_files
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import BODY, EYE, OUT, _outlined, make_slime


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "converted": converted}


NAMES = ["mushroom", "slime", "converted"]


def diamond(n: int = 21) -> np.ndarray:
    """1 ドットの輪郭だけのひし形。どの行も左右が斜めの線なので、横にずらすと必ず線が途切れる。"""
    img = np.zeros((n, n, 4), np.uint8)
    c = n // 2
    for y in range(n):
        d = c - abs(y - c)
        img[y, [c - d, c + d]] = (*OUT, 255)
    return img


def sash(h: int = 30, w: int = 14) -> np.ndarray:
    """四角い体を、斜めの 1 ドットのたすき（別の色）が横切る絵。たすきのあたりの高さでは揺らせない。"""
    shape = np.zeros((h, w), bool)
    shape[1:h - 1, 1:w - 1] = True
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = BODY
    for i in range(w - 2):
        rgb[8 + i // 2, 1 + i] = EYE
    return _outlined(shape, rgb)


def _pad(base: np.ndarray, p: int) -> np.ndarray:
    return np.pad(base, ((0, 0), (p, p), (0, 0)))


def _check_sway(base: np.ndarray, res) -> None:
    """横揺れの結果が決まりを守っていることを、check_rules() とは別の書き方でも確かめる。"""
    assert not res.skipped, res.warnings
    p = res.info["padding"]
    padded = _pad(base, p)
    assert np.array_equal(res.frames[0], padded)                       # 1 枚目は元のまま（余白を足しただけ）
    assert check_rules(padded, res.frames) == []
    base_colors = set(np.unique(color_keys(base)).tolist())
    euler = euler_numbers(padded)
    h = base.shape[0]
    low = min(res.info["cuts"][0])                                    # 分け目のいちばん上より上だけが動きうる
    for k, f in zip(res.info["offsets"], res.frames):
        assert f.shape == padded.shape and f.dtype == np.uint8          # 全フレームの大きさが同じ
        assert set(np.unique(color_keys(f)).tolist()) <= base_colors   # 新しい色が無い
        assert foot_center(f) == foot_center(padded)                   # 足元（下端の中央）が同じ
        assert shape_signature(f) == shape_signature(padded)           # 部品・輪郭のつながりが同じ
        assert euler_numbers(f) == euler                               # 穴の数も同じ（輪のような線が途切れていない）
        assert np.array_equal(f[max(res.info["cuts"][0]) + 1:], padded[max(res.info["cuts"][0]) + 1:])   # 分け目より下は同じ
        # 隙間が無い: 各行の不透明なドットの数は、分け目の高さが変わる所で 1 つ増えるか減るだけ。行の中に新しい切れ目ができない
        for y in range(h):
            row_b, row_f = padded[y, :, 3] > 0, f[y, :, 3] > 0
            assert abs(int(row_f.sum()) - int(row_b.sum())) <= abs(k)
            assert _gaps(row_f) <= _gaps(row_b)
        if k:
            # 分け目のいちばん上より上の行は、行ごとまるごと k ドットずれている（揺れ幅 2 は上の分け目で 2 ドット）
            top_rows = slice(0, low + 1) if abs(k) == 1 else slice(0, min(res.info["cuts"][1]) + 1)
            assert np.array_equal(f[top_rows], np.roll(padded[top_rows], k, axis=1))


def _gaps(row: np.ndarray) -> int:
    """1 行の中の、不透明な部分の間の透明の切れ目の数。"""
    xs = np.flatnonzero(row)
    return 0 if len(xs) == 0 else int(np.count_nonzero(~row[xs[0]:xs[-1] + 1]))


# ---- 決まりごと ----

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("frames, amount", [(4, 1), (8, 2), (3, 1), (6, 1), (8, 1)])
def test_sway_keeps_rules(sprites, name, frames, amount):
    base = sprites[name]
    res = sway(base, frames=frames, amount=amount)
    assert len(res.frames) == frames == len(res.durations)
    _check_sway(base, res)
    assert res.info["offsets"] == sway_offsets(frames, amount)
    assert len(res.info["cuts"]) == max(abs(k) for k in res.info["offsets"])


@pytest.mark.parametrize("name", NAMES)
def test_sway_moves_left_and_right(sprites, name):
    """既定（4 フレーム）は 中央 → 右 → 中央 → 左。上の部分のいちばん左のドットが +1・0・−1 ずれる。"""
    base = sprites[name]
    res = sway(base)
    assert res.info["offsets"] == [0, 1, 0, -1]
    top = int(np.flatnonzero((base[..., 3] > 0).any(1))[0])
    left = [int(np.flatnonzero(f[top, :, 3] > 0)[0]) for f in res.frames]
    assert [v - left[0] for v in left] == [0, 1, 0, -1]
    assert np.array_equal(res.frames[0], res.frames[2])


def test_padding_only_when_needed():
    """揺れる部分が画像の端に接していなければ余白を足さない。接していれば左右に同じ数だけ足す（足元は真ん中のまま）。"""
    tight = make_slime()                                              # 左右の端まで絵がある
    res = sway(tight)
    assert res.info["padding"] == 1 and res.size == (tight.shape[1] + 2, tight.shape[0])
    roomy = np.pad(tight, ((0, 0), (3, 3), (0, 0)))
    res2 = sway(roomy)
    assert res2.info["padding"] == 0 and res2.size == (roomy.shape[1], roomy.shape[0])
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, sway(_pad(tight, 1)).frames))
    res3 = sway(tight, frames=8, amount=2)
    assert res3.info["padding"] in (1, 2)
    info = sheet_info(res, "s.png")
    assert info["padding"] == {"left": 1, "right": 1, "top": 0, "bottom": 0}
    assert info["pivot"] == {"x": info["frame_width"] / 2, "y": info["frame_height"]}


def test_sway_avoids_diagonal_lines():
    """1 ドットの斜めの線（ひし形の輪郭）はどの行で分けても途切れるので、揺らさずに警告する（迷ったら動かさない）。"""
    img = diamond()
    res = sway(img)
    assert res.skipped and len(res.frames) == 1 and np.array_equal(res.frames[0], img)
    assert "横揺れを付けませんでした" in res.warnings[0]


def test_shift_above_breaks_diagonal_and_rules_catch_it():
    """確かめの仕組みが働くこと: ひし形をまっすぐ分けて横にずらすと、輪郭の輪が切れて穴の数が変わる。"""
    img = diamond()
    cut = np.full(img.shape[1], 12)
    moved = shift_above(img, cut, 1)
    assert euler_numbers(moved) != euler_numbers(img) or check_rules(img, [moved])


def test_shift_above_with_step():
    """分け目の高さが変わる所では、同じ行の左右で、ずれる部分とずれない部分が分かれる（色は元の行のもの）。"""
    img = np.zeros((3, 6, 4), np.uint8)
    img[..., 3] = 255
    img[:, :, 0] = np.arange(6)[None, :] * 10 + np.arange(3)[:, None] * 100
    out = shift_above(img, np.array([0, 0, 0, 1, 1, 1]), 1)
    assert out[0, :, 0].tolist() == [0, 0, 10, 20, 30, 40]           # 1 行目は全部右へ（左端は元のまま残る）
    assert out[1, :, 0].tolist() == [100, 110, 120, 130, 130, 140]   # 2 行目は 4 列目から右だけ（4 列目は元のまま残る）
    assert out[2, :, 0].tolist() == img[2, :, 0].tolist()


@pytest.mark.parametrize("img", [make_slime(10, 6), np.zeros((16, 16, 4), np.uint8)], ids=["tiny", "empty"])
def test_sway_skips_tiny_or_empty(img):
    res = sway(img)
    assert res.skipped and res.warnings


def test_sway_is_deterministic(sprites):
    a = sway(sprites["converted"], frames=8, amount=2)
    b = sway(sprites["converted"], frames=8, amount=2)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


def test_sway_offsets():
    assert sway_offsets(4, 1) == [0, 1, 0, -1]
    assert sway_offsets(8, 2) == [0, 1, 2, 1, 0, -1, -2, -1]
    assert sway_offsets(8, 1) == [0, 1, 1, 1, 0, -1, -1, -1]
    assert sway_offsets(3, 1) == [0, 1, -1]
    assert sway_offsets(6, 1) == [0, 1, 1, 0, -1, -1]


def test_sway_defaults_and_bad_values(sprites):
    res = make_motion(sprites["slime"], "sway")
    assert (DEFAULT_FRAMES, DEFAULT_DURATION, DEFAULT_AMOUNT) == (4, 300, 1)
    assert len(res.frames) == DEFAULT_FRAMES and res.durations == [DEFAULT_DURATION] * DEFAULT_FRAMES
    assert make_motion(sprites["slime"], "sway", frames=4, durations=[100, 200, 300, 400]).durations == [100, 200, 300, 400]
    for kw in ({"frames": 2}, {"frames": 33}, {"amount": 0}, {"amount": 3}, {"amount": True},
               {"frames": 4, "durations": [1, 2]}, {"durations": 0}, {"depth": 1}):
        with pytest.raises((ValueError, TypeError)):
            make_motion(sprites["slime"], "sway", **kw)
    assert MOTIONS["sway"].startswith("横揺れ")


# ---- 分ける高さの指定 ----

@pytest.mark.parametrize("name", NAMES)
def test_split_keeps_rules_and_range(sprites, name):
    base = sprites[name]
    lim = split_limits(base)
    h = base.shape[0]
    done = 0
    for split in range(lim[0], lim[1] + 1, 3):
        res = sway(base, split=split)
        if res.skipped:
            assert "近くで分けられる高さ" in res.warnings[0] or "分けられる高さがありません" in res.warnings[0]
            continue
        _check_sway(base, res)
        lo, hi = res.info["split_from_bottom"][0]
        assert split - SPLIT_RANGE <= lo <= hi <= split + SPLIT_RANGE  # 分け目は指定した高さの上下 2 ドット以内
        assert res.info["split"] == split and [h - c for c in (max(res.info["cuts"][0]), min(res.info["cuts"][0]))] == [lo, hi]
        done += 1
    assert done >= 2


def test_split_impossible_warns_and_suggests():
    """指定した高さの近くで揺らせなければ、ほかの高さに移さずに警告し、近くで揺らせる高さを知らせる（その高さなら作れる）。"""
    img = sash()
    lim = split_limits(img)
    failed = [s for s in range(lim[0], lim[1] + 1) if sway(img, split=s).skipped]
    assert failed, "たすきのあたりで揺らせない高さがあるはず"
    res = sway(img, split=failed[0])
    assert len(res.frames) == 1 and np.array_equal(res.frames[0], img)
    assert "近くで分けられる高さ: " in res.warnings[0]
    near = nearby_splits(img, failed[0])
    assert near and f"近くで分けられる高さ: {'、'.join(map(str, near))}" in res.warnings[0]
    for s in near:
        _check_sway(img, sway(img, split=s))


def test_split_in_feet_is_skipped_and_outside_is_error(sprites):
    base = sprites["mushroom"]
    lim = split_limits(base)
    res = sway(base, split=1)
    assert res.skipped and f"{lim[0]}〜{lim[1]}" in res.warnings[0]
    for bad in (0, base.shape[0] + 1):
        with pytest.raises(ValueError, match="範囲の外"):
            sway(base, split=bad)
    for bad in ("middle", 1.5, True):
        with pytest.raises(ValueError):
            sway(base, split=bad)
    a, b = sway(base), sway(base, split="auto")
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames)) and a.info["split"] == "auto"


# ---- CLI ----

def _ai_input(tmp_path):
    path = tmp_path / "in" / "chara.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_sway(tmp_path, capsys):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "--motion", "sway"]) == 0
    assert "chara_sway_sheet.png" in capsys.readouterr().out
    assert sorted(p.name for p in out.parent.iterdir()) == ["chara.png", "chara_sway_sheet.json", "chara_sway_sheet.png"]
    info = json.loads((out.parent / "chara_sway_sheet.json").read_text(encoding="utf-8"))
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    p = info["padding"]["left"]
    assert info["motion"] == "sway" and info["frame_count"] == 4 and info["padding"]["right"] == p
    assert info["frame_width"] == sprite.shape[1] + 2 * p and info["frame_height"] == sprite.shape[0]
    assert [f["duration_ms"] for f in info["frames"]] == [300] * 4
    sheet = np.asarray(Image.open(out.parent / "chara_sway_sheet.png").convert("RGBA"))
    assert sheet.shape == (sprite.shape[0], info["frame_width"] * 4, 4)
    assert np.array_equal(sheet[:, :info["frame_width"]], _pad(sprite, p))
    expected = sway(sprite)
    assert np.array_equal(sheet, np.concatenate(expected.frames, axis=1))


def test_cli_sway_options(tmp_path):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "sway", "--motion-frames", "8", "--sway-amount", "2",
                     "--motion-duration", "150"]) == 0
    info = json.loads((out.parent / "chara_sway_sheet.json").read_text(encoding="utf-8"))
    assert info["frame_count"] == 8 and info["total_duration_ms"] == 1200
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    split = split_limits(sprite)[0] + 6
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "sway", "--sway-split", str(split)]) == 0


@pytest.mark.parametrize("extra, words", [
    (["--sway-amount", "2"], "動きのオプション"),
    (["--sway-split", "3"], "動きのオプション"),
    (["--motion", "sway", "--breathe-depth", "2"], "--motion sway では使えません"),
    (["--motion", "sway", "--breathe-split", "3"], "--motion sway では使えません"),
    (["--motion", "breathe", "--sway-amount", "1"], "--motion breathe では使えません"),
    (["--motion", "breathe", "--sway-split", "auto"], "--motion breathe では使えません"),
    (["--motion", "sway", "--motion-frames", "2"], "3〜32"),
    (["--motion", "sway", "--sway-amount", "3"], "1〜2"),
    (["--motion", "sway", "--motion-duration", "300,200"], "フレーム数（4）"),
    (["--motion", "sway", "--sway-split", "x"], "auto"),
])
def test_cli_rejects_bad_sway_options(tmp_path, capsys, extra, words):
    src = _ai_input(tmp_path)
    with pytest.raises(SystemExit):
        cli.main([str(src), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_sway_skipped_and_outside_split(tmp_path, capsys):
    """揺らせない絵は警告してシートを書かない。画像の外の高さはエラー（終了コード 1）。変換の結果はどちらも保存する。"""
    args = cli.build_parser().parse_args(["x.png", "--motion", "sway", "-q"])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, diamond(), dst, dst) == 1
    assert "横揺れを付けませんでした" in capsys.readouterr().err
    assert not (tmp_path / "s_sway_sheet.png").exists()
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "sway", "--sway-split", "999"]) == 1
    assert "範囲の外" in capsys.readouterr().err and out.exists() and not (out.parent / "chara_sway_sheet.png").exists()


def test_save_sheet_for_sway(tmp_path, sprites):
    res = sway(sprites["mushroom"], frames=8, amount=2)
    png, js = save_sheet(res, tmp_path / "m_sway_sheet.png")
    info = json.loads(js.read_text(encoding="utf-8"))
    assert info["motion"] == "sway" and info["frame_count"] == 8 and info["image"] == "m_sway_sheet.png"
    assert info["padding"]["left"] == res.info["padding"]


# ---- 今までの結果が変わっていないこと ----

def _digest(path) -> str:
    """ファイルの中身のハッシュ。PNG は画素（大きさと RGBA）で比べる（PNG の圧縮のしかたは Pillow・zlib の版や OS で変わり、
    同じ画素でもファイルのバイトが違うことがあるため。Windows の CI で確かめた）。JSON は改行を LF にそろえて比べる。"""
    if path.suffix == ".png":
        a = np.asarray(Image.open(path).convert("RGBA"))
        data = repr(a.shape).encode() + a.tobytes()
    else:
        data = legacy_bytes(path.read_bytes().replace(b"\r\n", b"\n"))   # 版 2 で足した skipped を除く
    return hashlib.sha256(data).hexdigest()[:16]


# PR #23 の main（この PR の前）の CLI で作ったファイルのハッシュ（`_digest()`。SHA-256 の先頭 16 文字）。
# 入力は enlarge_like_ai(make_pixel_sprite(), seed=3)。(設定の名前, ファイル名, ハッシュ)
BEFORE = [
    ("none", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe", "hero_sprite_breathe_sheet.json", "736eef558dfcfa5e"),
    ("breathe", "hero_sprite_breathe_sheet.png", "a3bffd484ccafac1"),
    ("breathe_f4d2", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe_f4d2", "hero_sprite_breathe_sheet.json", "7272ab6f606082e3"),
    ("breathe_f4d2", "hero_sprite_breathe_sheet.png", "69ac46a9a88e37aa"),
    ("breathe_s12", "hero_sprite.png", "616c81fd24a30eac"),
    ("breathe_s12", "hero_sprite_breathe_sheet.json", "736eef558dfcfa5e"),
    ("breathe_s12", "hero_sprite_breathe_sheet.png", "666a2b73570b5b8e"),
]
ARGS = {"none": [], "breathe": ["--motion", "breathe"],
        "breathe_f4d2": ["--motion", "breathe", "--motion-frames", "4", "--breathe-depth", "2"],
        "breathe_s12": ["--motion", "breathe", "--breathe-split", "12"]}


@pytest.mark.parametrize("tag", list(ARGS))
def test_cli_without_motion_and_breathe_are_unchanged(tmp_path, tag):
    """--motion なし・--motion breathe の CLI の出力（ファイルの名前と中身）が、この PR の前とまったく同じ。"""
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=3)
    src = tmp_path / "hero.png"
    Image.fromarray(img, "RGBA").save(src)
    out = tmp_path / tag
    assert cli.main([str(src), "-o", str(out / "hero_sprite.png"), "-q", *ARGS[tag]]) == 0
    got = sorted((f.name, _digest(f)) for f in legacy_files(out.iterdir()))
    assert got == sorted((name, h) for t, name, h in BEFORE if t == tag)
