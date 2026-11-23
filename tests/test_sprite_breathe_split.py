"""呼吸の「分ける高さ」の指定（`breathe(split=...)`、CLI `--breathe-split`）のテスト。

- 指定しないとき・auto のときは PR #21 の呼吸とまったく同じ（下の HASHES は PR #21 の breathe.py で作ったフレームの SHA-256）
- 高さを指定しても決まり（新しい色を作らない・大きさと足元・隙間・輪郭）を守り、継ぎ目は指定した高さの近くに収まる
- 指定した高さの近くで決まりを守れなければ、動かさずに警告し、近くで分けられる高さを知らせる（迷ったら動かさない）
- 画像の外の高さはエラー
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.motion import check_rules, foot_center, make_motion
from retrocore.sprite.motion.base import color_keys
from retrocore.sprite.motion.breathe import (SPLIT_RANGE, breathe, check_split, find_seam, nearby_splits,
                                             seam_candidates, split_limits)
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import BODY, EYE, _outlined, make_slime


@pytest.fixture(scope="module")
def sprites():
    converted = convert_sprite(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], SpriteSettings()).image
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "converted": converted}


def _digest(frames) -> str:
    return hashlib.sha256(b"".join(np.ascontiguousarray(f).tobytes() for f in frames)).hexdigest()[:16]


# PR #21 の呼吸（自動）で作ったフレームの SHA-256（先頭 16 文字）。(絵, フレーム数, 下がる量, ハッシュ)
HASHES = [
    ("mushroom", 2, 1, "5ed4d89305df435c"),
    ("mushroom", 4, 2, "90b0760dce2d02c9"),
    ("mushroom", 8, 1, "3d3dd74f2590b3b8"),
    ("slime", 2, 1, "faf7d7c3cf16ad7a"),
    ("slime", 4, 2, "45566e35a72e963e"),
    ("slime", 8, 1, "30bec59b35b9f3ba"),
    ("converted", 2, 1, "40304f225c2f1126"),
    ("converted", 4, 2, "bb1e35890d2f1b76"),
    ("converted", 8, 1, "39f72c54b45233ab"),
]


@pytest.mark.parametrize("name, frames, depth, digest", HASHES)
@pytest.mark.parametrize("split", [None, "auto"])
def test_auto_is_same_as_before(sprites, name, frames, depth, digest, split):
    """指定しないとき・auto のときは、PR #21 の呼吸とまったく同じフレームになる。"""
    options = {} if split is None else {"split": split}
    res = breathe(sprites[name], frames=frames, depth=depth, **options)
    assert not res.skipped
    assert _digest(res.frames) == digest
    assert res.info["split"] == "auto"


def test_find_seam_auto_is_unchanged(sprites):
    a = find_seam(sprites["converted"])
    b = find_seam(sprites["converted"], "auto")
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def _check_split_result(base, res, split):
    h = base.shape[0]
    assert not res.skipped, res.warnings
    assert check_rules(base, res.frames) == []                   # 色・大きさ・足元・つながり・縁
    base_colors = set(np.unique(color_keys(base)).tolist())
    for f in res.frames:
        assert f.shape == base.shape
        assert set(np.unique(color_keys(f)).tolist()) <= base_colors
        assert foot_center(f) == foot_center(base)
        # 隙間が無い: 各列の不透明なドットの数は、元と同じか 1〜2 減るだけ（抜いた分）で、列の中に新しい透明の切れ目ができない
        for x in range(base.shape[1]):
            col_b, col_f = base[:, x, 3] > 0, f[:, x, 3] > 0
            assert col_f.sum() <= col_b.sum()
            assert _gaps(col_f) <= _gaps(col_b)
    for seam in res.info["seams"]:
        rows_from_bottom = h - np.asarray(seam)
        assert rows_from_bottom.min() >= split - SPLIT_RANGE and rows_from_bottom.max() <= split + SPLIT_RANGE
    lo, hi = res.info["seam_from_bottom"][0]
    assert split - SPLIT_RANGE <= lo <= hi <= split + SPLIT_RANGE


def _gaps(col: np.ndarray) -> int:
    """列の中の不透明な部分の数（透明で区切られたかたまりの数）。"""
    return int(np.count_nonzero(col[1:] & ~col[:-1]) + col[0])


@pytest.mark.parametrize("name, splits", [("mushroom", [6, 10, 15, 20, 25]), ("slime", [4, 7, 10, 13]),
                                          ("converted", [6, 10, 15, 20, 25])])
@pytest.mark.parametrize("frames, depth", [(2, 1), (4, 2), (8, 1)])
def test_split_keeps_rules(sprites, name, splits, frames, depth):
    """高さを指定しても決まりを守り、継ぎ目は指定した高さの上下 SPLIT_RANGE ドット以内に収まる。足元の行は変わらない。"""
    base = sprites[name]
    for split in splits:
        res = breathe(base, frames=frames, depth=depth, split=split)
        _check_split_result(base, res, split)
        assert res.info["split"] == split
        lo = min(h for h, _ in res.info["seam_from_bottom"])
        keep = base.shape[0] - lo + 1                             # 継ぎ目より下の行は 1 ドットも変わらない
        assert all(np.array_equal(f[keep:], base[keep:]) for f in res.frames)


def test_split_moves_the_seam(sprites):
    """違う高さを指定すると、継ぎ目の高さも変わる（指定が効いている）。"""
    base = sprites["converted"]
    lows = [breathe(base, split=s).info["seam_from_bottom"][0] for s in (8, 14, 22)]
    assert lows[0][1] < lows[1][0] and lows[1][1] < lows[2][0]


def test_split_with_numpy_int(sprites):
    a = breathe(sprites["mushroom"], split=12)
    b = breathe(sprites["mushroom"], split=np.int64(12))
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))


def striped(h=30, w=12, stripes=range(8, 20)) -> np.ndarray:
    """上から 8〜19 行目が 1 行ごとに色の変わる横じまで、ほかは同じ色の箱（しまの所では抜けるドットが無い）。"""
    shape = np.zeros((h, w), bool)
    shape[1:h - 1, 1:w - 1] = True
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = BODY
    for y in stripes:
        if y % 2 == 0:
            rgb[y, :] = EYE
    return _outlined(shape, rgb)


def test_split_impossible_warns_and_suggests():
    """指定した高さの近くに決まりを守る継ぎ目が無ければ、動かさずに警告し、近くで分けられる高さを知らせる。"""
    img = striped()
    h = img.shape[0]
    split = h - 14                                                # しまの真ん中
    res = breathe(img, split=split)
    assert res.skipped
    assert len(res.frames) == 1 and np.array_equal(res.frames[0], img)
    msg = res.warnings[0]
    assert "呼吸を付けませんでした" in msg and f"下から {split} ドット目" in msg and "近くで分けられる高さ" in msg
    near = nearby_splits(img, split)
    assert near and all(str(v) in msg for v in near)
    assert any(v < split for v in near) and any(v > split for v in near)
    for v in near:                                                # 知らせた高さでは本当に付けられる
        _check_split_result(img, breathe(img, split=v), v)
    assert seam_candidates(img, split) == [] or all(
        check_rules(img, [img, _lower(img, s)]) for s, _ in seam_candidates(img, split))


def _lower(img, seam):
    from retrocore.sprite.motion.breathe import lower_above_seam
    return lower_above_seam(img, seam)


def test_split_does_not_fall_back_to_auto():
    """作れないときに、自動の高さや遠くの高さに勝手に移さない（自動なら付けられる絵でも、指定が合わなければ付けない）。"""
    img = striped()
    assert not breathe(img).skipped
    assert breathe(img, split=img.shape[0] - 14).skipped


def test_split_in_feet_or_above_sprite_is_skipped(sprites):
    """画像の中でも、足元（動かさない下の行）や絵より上では分けられないので、指定できる範囲を添えて警告する。"""
    base = sprites["mushroom"]
    lo, hi = split_limits(base)
    assert lo >= 3 and hi <= base.shape[0]
    for split in (1, lo - 1):
        res = breathe(base, split=split)
        assert res.skipped and f"{lo}〜{hi}" in res.warnings[0]
    padded = np.zeros((base.shape[0] + 6,) + base.shape[1:], np.uint8)   # 上に 6 行の透明
    padded[6:] = base
    res = breathe(padded, split=padded.shape[0] - 2)
    assert res.skipped and "指定できる" in res.warnings[0]


@pytest.mark.parametrize("split", [0, -1, 31, 100])
def test_split_outside_image_is_error(sprites, split):
    base = sprites["mushroom"]                                     # 高さ 30
    with pytest.raises(ValueError, match="画像の範囲の外"):
        breathe(base, split=split)
    with pytest.raises(ValueError, match="1〜30"):
        make_motion(base, "breathe", split=split)


@pytest.mark.parametrize("split", ["middle", 1.5, True])
def test_split_bad_type_is_error(sprites, split):
    with pytest.raises(ValueError, match="auto"):
        breathe(sprites["mushroom"], split=split)


def test_check_split():
    assert check_split(None, 10) is None and check_split("auto", 10) is None
    assert check_split(1, 10) == 1 and check_split(10, 10) == 10
    with pytest.raises(ValueError):
        check_split(11, 10)


def test_split_limits_empty_and_tiny():
    assert split_limits(np.zeros((16, 16, 4), np.uint8)) is None
    assert split_limits(make_slime(10, 6)) is None
    res = breathe(make_slime(10, 6), split=3)
    assert res.skipped


# ---- CLI ----

def _ai_input(tmp_path):
    path = tmp_path / "in" / "chara.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(28, 32), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_split_auto_is_same_as_default(tmp_path):
    """--breathe-split auto は、付けないときと同じファイルを書く。"""
    src = _ai_input(tmp_path)
    assert cli.main([str(src), "-o", str(tmp_path / "a" / "chara.png"), "-q", "--motion", "breathe"]) == 0
    assert cli.main([str(src), "-o", str(tmp_path / "b" / "chara.png"), "-q", "--motion", "breathe",
                     "--breathe-split", "auto"]) == 0
    for name in ("chara.png", "chara_breathe_sheet.png", "chara_breathe_sheet.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_cli_split_number(tmp_path):
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "breathe", "--breathe-split", "12",
                     "--motion-frames", "4", "--breathe-depth", "2"]) == 0
    sprite = np.asarray(Image.open(out).convert("RGBA"))
    sheet = np.asarray(Image.open(out.parent / "chara_breathe_sheet.png").convert("RGBA"))
    w = sprite.shape[1]
    expected = breathe(sprite, frames=4, depth=2, split=12)
    assert not expected.skipped
    assert all(np.array_equal(sheet[:, i * w:(i + 1) * w], f) for i, f in enumerate(expected.frames))
    assert json.loads((out.parent / "chara_breathe_sheet.json").read_text(encoding="utf-8"))["frame_count"] == 4


def test_cli_split_outside_image_is_error(tmp_path, capsys):
    """画像の高さを超える指定は、分かりやすいエラーにして終了コード 1（変換の結果は保存する。シートは書かない）。"""
    src = _ai_input(tmp_path)
    out = tmp_path / "o" / "chara.png"
    assert cli.main([str(src), "-o", str(out), "-q", "--motion", "breathe", "--breathe-split", "500"]) == 1
    err = capsys.readouterr().err
    assert "エラー（chara.png）" in err and "画像の範囲の外" in err
    assert out.exists() and not (out.parent / "chara_breathe_sheet.png").exists()


@pytest.mark.parametrize("extra, words", [
    (["--breathe-split", "12"], "動きのオプション"),
    (["--motion", "breathe", "--breathe-split", "0"], "1 以上"),
    (["--motion", "breathe", "--breathe-split", "waist"], "auto"),
])
def test_cli_rejects_bad_split(tmp_path, capsys, extra, words):
    src = _ai_input(tmp_path)
    with pytest.raises(SystemExit):
        cli.main([str(src), "-o", str(tmp_path / "x.png")] + extra)
    assert words in capsys.readouterr().err


def test_cli_split_impossible_warns(tmp_path, capsys):
    img = striped()
    args = cli.build_parser().parse_args(["x.png", "--motion", "breathe", "-q", "--breathe-split",
                                          str(img.shape[0] - 14)])
    dst = tmp_path / "s.png"
    assert cli.write_motion(args, img, dst, dst) == 1
    err = capsys.readouterr().err
    assert "呼吸を付けませんでした" in err and "近くで分けられる高さ" in err
    assert not (tmp_path / "s_breathe_sheet.png").exists()
