"""パレット統合の第一歩（retrocore/sprite/palette.py と CLI の --palette / --save-palette）のテスト。"""

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from retrocore.sprite import (OutlineStep, PaletteStep, SpritePipeline, load_palette, match_palette, outline_sprite,
                              save_palette, sort_palette, used_colors)
from retrocore.sprite.palette import PaletteError, format_gpl, format_hex, parse_gpl, parse_hex
from synth_sprite import (PIX_BODY, PIX_CAP, PIX_CHEEK, PIX_EYE, PIX_OUTLINE, enlarge_like_ai, make_pixel_sprite)

PAL = np.array([[0, 0, 0], [255, 255, 255], [200, 40, 40], [40, 160, 60], [60, 80, 200], [240, 200, 160]], np.uint8)


def _colors_set(rgba):
    return {tuple(int(v) for v in c) for c in used_colors(rgba)}


def _pal_set(p):
    return {tuple(int(v) for v in c) for c in np.asarray(p)}


def _write_pixel_art(path, seed=0):
    img, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(path)
    return path


# --- 読み書き -----------------------------------------------------------------------------

def test_read_hex(tmp_path):
    p = tmp_path / "a.hex"
    p.write_text("ff0000\n#00FF00\n\n; コメント\n0000ff\nff0000\n", encoding="utf-8")
    assert load_palette(p).tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]   # 順はそのまま、重複は除く


def test_read_gpl_including_aseprite_rgba(tmp_path):
    p = tmp_path / "a.gpl"
    p.write_text("GIMP Palette\nName: test\nColumns: 4\n#\n255   0   0\tred\n  0 255   0 green\n0 0 255\n",
                 encoding="utf-8")
    assert load_palette(p).tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    q = tmp_path / "b.gpl"
    q.write_text("GIMP Palette\nChannels: RGBA\n#\n10 20 30 255 Untitled\n40 50 60 128\n", encoding="utf-8")
    assert load_palette(q).tolist() == [[10, 20, 30], [40, 50, 60]]


def test_read_png_unique_colors_without_transparent(tmp_path):
    a = np.zeros((2, 3, 4), np.uint8)
    a[0, 0] = (255, 0, 0, 255)
    a[0, 1] = (0, 255, 0, 255)
    a[0, 2] = (255, 0, 0, 255)       # 重複
    a[1, 0] = (9, 9, 9, 0)           # 透明は読まない
    a[1, 1] = (0, 0, 255, 255)
    a[1, 2] = (0, 255, 0, 255)
    p = tmp_path / "a.png"
    Image.fromarray(a, "RGBA").save(p)
    assert load_palette(p).tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


@pytest.mark.parametrize("ext", [".hex", ".gpl"])
def test_write_and_read_back_same_colors(tmp_path, ext):
    p = tmp_path / f"pal{ext}"
    written = save_palette(PAL, p)
    back = load_palette(p)
    assert np.array_equal(back, written)              # 書いた順のまま読める
    assert _pal_set(back) == _pal_set(PAL)            # 色は同じ（並びは sort_palette の順）
    assert np.array_equal(written, sort_palette(PAL))


def test_formats_text():
    assert format_hex([[255, 0, 16]]) == "ff0010\n"
    g = format_gpl([[255, 0, 16]], "x")
    assert g.startswith("GIMP Palette\nName: x\n") and "255   0  16\t#FF0010" in g
    assert parse_gpl(g).tolist() == [[255, 0, 16]]
    assert parse_hex(format_hex(PAL)).tolist() == PAL.tolist()


@pytest.mark.parametrize("name,text", [("a.hex", "ff00\n"), ("a.gpl", "not a palette\n"), ("a.gpl", "GIMP Palette\n1 2\n"),
                                       ("a.hex", ""), ("a.txt", "ff0000\n")])
def test_bad_palette_files(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    with pytest.raises(PaletteError):
        load_palette(p)


def test_save_rejects_png_and_unknown(tmp_path):
    for name in ("a.png", "a.txt"):
        with pytest.raises(PaletteError):
            save_palette(PAL, tmp_path / name)


def test_sort_order_grays_first_then_hue_groups_dark_to_light():
    cols = np.array([[255, 255, 255], [250, 120, 120], [20, 20, 20], [120, 20, 20], [200, 40, 40],
                     [40, 160, 60], [20, 80, 30], [128, 128, 128], [60, 80, 200]], np.uint8)
    s = [tuple(c) for c in sort_palette(cols).tolist()]
    assert s[:3] == [(20, 20, 20), (128, 128, 128), (255, 255, 255)]      # 灰色: 暗い → 明るい
    assert s[3:6] == [(120, 20, 20), (200, 40, 40), (250, 120, 120)]     # 赤のグループ: 暗い → 明るい
    assert s[6:8] == [(20, 80, 30), (40, 160, 60)]                       # 緑
    assert s[8] == (60, 80, 200)                                         # 青
    # 入力の順に関係なく同じ結果
    assert np.array_equal(sort_palette(cols[::-1]), sort_palette(cols))


# --- パレットに合わせる -------------------------------------------------------------------

def test_match_uses_only_palette_colors_and_keeps_transparent():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    a[..., 3] = np.where(rng.random((20, 30)) < 0.3, 0, 255)
    a[0, 0, 3] = 200                                           # 半透明（不透明とみなす）
    a[0, 1, 3] = 50                                            # 半透明（透明とみなす）
    r = match_palette(a, PAL)
    assert _colors_set(r.image) <= _pal_set(PAL)
    transparent = a[..., 3] < 128
    assert np.array_equal(r.image[transparent], a[transparent])   # 透明はそのまま
    assert (r.image[~transparent][:, 3] == 255).all()


def test_match_warns_when_two_parts_become_one_color():
    a = np.zeros((4, 6, 4), np.uint8)
    a[..., 3] = 255
    a[..., :3] = (240, 200, 160)          # 肌
    a[1, 1:3, :3] = (250, 170, 170)       # 頬（パレットに近いピンクが無い → 肌になる）
    a[3, :, :3] = (0, 0, 0)
    r = match_palette(a, PAL)
    assert len(r.collisions) == 1
    c = r.collisions[0]
    assert c.target == (240, 200, 160)
    assert c.sources == [((240, 200, 160), 16), ((250, 170, 170), 2)]
    assert c.dots == 2
    assert "#faaaaa（2 ドット）" in c.message() and "#f0c8a0" in c.message()
    # パレットに頬の色があれば警告なし
    r2 = match_palette(a, np.vstack([PAL, [[250, 170, 170]]]))
    assert r2.collisions == []


def test_match_does_not_warn_for_indistinguishable_colors():
    a = np.zeros((2, 4, 4), np.uint8)
    a[..., 3] = 255
    a[0, :, :3] = (200, 40, 40)
    a[1, :, :3] = (201, 41, 40)           # 見分けのつかない色（距離 0.03 未満）は 1 つの部品
    assert match_palette(a, PAL).collisions == []


def test_match_far_color():
    a = np.zeros((1, 2, 4), np.uint8)
    a[..., 3] = 255
    a[0, 0, :3] = (255, 0, 255)           # マゼンタはパレットに近い色が無い
    a[0, 1, :3] = (0, 0, 0)
    r = match_palette(a, PAL)
    assert [f.source for f in r.far] == [(255, 0, 255)]


def test_match_is_deterministic_and_does_not_modify_input():
    a, _, _ = enlarge_like_ai(make_pixel_sprite(), seed=1)
    a0 = a.copy()
    r1, r2 = match_palette(a, PAL), match_palette(a, PAL)
    assert np.array_equal(r1.image, r2.image) and np.array_equal(a, a0)
    assert [c.sources for c in r1.collisions] == [c.sources for c in r2.collisions]


def test_match_speed_large_image():
    import time
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (400, 640, 4), dtype=np.uint8)
    a[..., 3] = 255
    a[..., :3] //= 8           # 色の種類を 32^3 以下に
    a[..., :3] *= 8
    pal = rng.integers(0, 256, (32, 3), dtype=np.uint8)
    t = time.perf_counter()
    r = match_palette(a, pal, merge_distance=0)
    assert time.perf_counter() - t < 10
    assert _colors_set(r.image) <= _pal_set(pal)


# --- 縁取り（パレットの色から選ぶ） --------------------------------------------------------

def _blob():
    a = np.zeros((10, 10, 4), np.uint8)
    a[2:8, 2:8] = (240, 200, 160, 255)
    a[4:6, 4:6] = (200, 40, 40, 255)
    return a


@pytest.mark.parametrize("color_mode,color", [("darken", None), ("single", None), ("single", (255, 0, 255))])
def test_outline_add_picks_palette_colors(color_mode, color):
    r = outline_sprite(_blob(), mode="add", color_mode=color_mode, color=color, palette=PAL)
    assert r.count > 0
    assert _colors_set(r.image) <= _pal_set(PAL)
    ring = r.kinds > 0
    # 外側の輪郭は内側（肌）より暗いパレットの色
    assert all(tuple(int(v) for v in c) != (240, 200, 160) for c in r.image[ring][:, :3])


def test_outline_without_palette_unchanged():
    a = _blob()
    r1 = outline_sprite(a, mode="add")
    r2 = OutlineStep(mode="add").apply(a)
    assert np.array_equal(r1.image, r2)
    # 暗くした色はパレットに無い色（パレットを指定しないときは今まで通り新しい色を作る）
    assert not _colors_set(r1.image) <= _pal_set(PAL)


# --- CLI ----------------------------------------------------------------------------------

def test_cli_palette_output_only_palette_colors(tmp_path, capsys):
    src = _write_pixel_art(tmp_path / "pix.png")
    pal = tmp_path / "game.hex"
    pal.write_text(format_hex(PAL), encoding="utf-8")
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "-o", str(dst), "--palette", str(pal), "--outline", "add"]) == 0
    out = np.asarray(Image.open(dst))
    assert _colors_set(out) <= _pal_set(PAL)
    text = capsys.readouterr()
    assert "パレットに合わせる" in text.out and "パレットの確認（パレットにない色だったドット 0）" in text.out
    # 透明のまま（外側は透明、不透明は 255 だけ）
    assert set(np.unique(out[..., 3]).tolist()) <= {0, 255}
    assert out[0, 0, 3] == 0


def test_cli_palette_step_order():
    args = cli.build_parser().parse_args(["x.png"])
    args.palette_colors = PAL
    names = [s.name for s in cli.build_pipeline(args).steps]
    assert names == ["background", "grid", "simplify", "palette", "cleanup", "outline", "palette-check"]
    (ol,) = [s for s in cli.build_pipeline(args).steps if s.name == "outline"]
    assert ol.palette is not None
    args = cli.build_parser().parse_args(["x.png", "--no-simplify"])
    args.palette_colors = PAL
    assert [s.name for s in cli.build_pipeline(args).steps][:3] == ["background", "grid", "palette"]


def test_cli_without_palette_pipeline_unchanged():
    args = cli.build_parser().parse_args(["x.png"])
    steps = cli.build_pipeline(args).steps
    assert [s.name for s in steps] == ["background", "grid", "simplify", "cleanup", "outline"]
    assert [s for s in steps if s.name == "outline"][0].palette is None


def test_cli_without_palette_output_unchanged(tmp_path):
    """--palette を使わないときは、パレットの工程の無い組み立てと出力が同じ。"""
    src = _write_pixel_art(tmp_path / "pix.png")
    dst = tmp_path / "a.png"
    assert cli.main([str(src), "-o", str(dst), "-q", "--save-palette", str(tmp_path / "p.hex")]) == 0
    from retrocore.sprite import (BackgroundStep, CleanupStep, GridStep, SimplifyStep, load_rgba)
    ref = SpritePipeline([BackgroundStep(), GridStep(), SimplifyStep(), CleanupStep(), OutlineStep()]).run(
        load_rgba(src))
    assert np.array_equal(np.asarray(Image.open(dst)), ref)


def test_cli_save_palette_then_reuse(tmp_path, capsys):
    src = _write_pixel_art(tmp_path / "pix.png")
    a = tmp_path / "a.png"
    assert cli.main([str(src), "-o", str(a), "-q", "--save-palette", str(tmp_path / "p.hex"),
                     "--save-palette", str(tmp_path / "p.gpl")]) == 0
    out = np.asarray(Image.open(a))
    used = _colors_set(out)
    ph, pg = load_palette(tmp_path / "p.hex"), load_palette(tmp_path / "p.gpl")
    assert _pal_set(ph) == used and np.array_equal(ph, pg)           # 透明は含めず、使った色だけ
    assert np.array_equal(ph, sort_palette(ph))
    # 同じ画像を自分のパレットで変換すると、同じ出力で警告なし
    b = tmp_path / "b.png"
    capsys.readouterr()
    assert cli.main([str(src), "-o", str(b), "-q", "--palette", str(tmp_path / "p.gpl")]) == 0
    assert np.array_equal(np.asarray(Image.open(b)), out)
    assert "警告" not in capsys.readouterr().err


def test_cli_palette_from_png_and_warning(tmp_path, capsys):
    src = _write_pixel_art(tmp_path / "pix.png")
    # 頬の色が無いパレット（見本画像）: 頬のピンクが別の色に合わさって警告が出る
    cols = [PIX_OUTLINE, PIX_CAP, PIX_BODY, PIX_EYE, (255, 255, 255)]
    img = np.array([[list(c) + [255] for c in cols]], np.uint8)
    pp = tmp_path / "pal.png"
    Image.fromarray(img, "RGBA").save(pp)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "-o", str(dst), "-q", "--palette", str(pp), "--colors", "16"]) == 0
    out = np.asarray(Image.open(dst))
    assert _colors_set(out) <= {tuple(c) for c in cols}
    err = capsys.readouterr().err
    assert "警告（pix.png）" in err and "パレットの同じ色" in err
    assert PIX_CHEEK not in _colors_set(out)


def test_cli_palette_deterministic(tmp_path):
    src = _write_pixel_art(tmp_path / "pix.png", seed=2)
    pal = tmp_path / "game.gpl"
    pal.write_text(format_gpl(PAL), encoding="utf-8")
    outs = []
    for i in range(2):
        dst = tmp_path / f"o{i}.png"
        assert cli.main([str(src), "-o", str(dst), "-q", "--palette", str(pal)]) == 0
        outs.append(np.asarray(Image.open(dst)))
    assert np.array_equal(*outs)


def test_cli_save_palette_many_inputs(tmp_path):
    _write_pixel_art(tmp_path / "in" / "a.png", seed=0)
    _write_pixel_art(tmp_path / "in" / "b.png", seed=1)
    assert cli.main([str(tmp_path / "in"), "-o", str(tmp_path / "out"), "-q",
                     "--save-palette", str(tmp_path / "pal" / "p.hex")]) == 0
    assert sorted(p.name for p in (tmp_path / "pal").iterdir()) == ["p_a.hex", "p_b.hex"]


@pytest.mark.parametrize("extra,words", [
    (["--palette", "missing.hex"], "パレットを読み込めません"),
    (["--palette", "x.txt"], ".hex / .gpl / .png"),
    (["--save-palette", "x.png"], "--save-palette は .hex / .gpl"),
])
def test_cli_rejects_bad_palette_options(tmp_path, capsys, extra, words):
    src = _write_pixel_art(tmp_path / "pix.png")
    extra = [str(tmp_path / e) if e.endswith((".hex", ".txt", ".png")) else e for e in extra]
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), "-q"] + extra)
    assert e.value.code == 2
    assert words in capsys.readouterr().err


def test_palette_step_validates():
    with pytest.raises(ValueError):
        PaletteStep(palette=np.zeros((0, 3), np.uint8))
    s = PaletteStep(palette=PAL, final=True)
    assert s.name == "palette-check"
