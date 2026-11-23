"""apps/sprite の CLI のテスト。"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from synth_sprite import make_character

ROOT = Path(__file__).resolve().parents[1]


def _write(path, w=256, h=384, transparent=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_character(w, h, transparent), "RGBA").save(path)
    return path


def _step(args, name):
    """組み立てた工程のうち、名前が name の工程。"""
    (step,) = [s for s in cli.build_pipeline(args).steps if s.name == name]
    return step


def test_cli_scale_to_file(tmp_path):
    src = _write(tmp_path / "chara.png")
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "--input-type", "illust", "--scale", "1/4", "-o", str(dst), "-q"]) == 0
    img = Image.open(dst)
    assert img.format == "PNG" and img.mode == "RGBA" and img.size == (64, 96)


def test_cli_width_keeps_aspect_and_saves_steps(tmp_path, capsys):
    src = _write(tmp_path / "chara.png", transparent=True)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "--input-type", "illust", "--width", "32", "-o", str(dst),
                     "--save-steps"]) == 0
    assert Image.open(dst).size == (32, 48)
    assert np.asarray(Image.open(dst))[0, 0, 3] == 0
    assert (tmp_path / "out_steps" / "01_background.png").is_file()
    assert (tmp_path / "out_steps" / "02_resize.png").is_file()
    out = capsys.readouterr().out
    assert "背景除去" in out and "縮小" in out


def test_cli_folder_input(tmp_path):
    _write(tmp_path / "in" / "a.png")
    _write(tmp_path / "in" / "b.png", 128, 128)
    Image.fromarray(make_character(64, 64)[..., :3]).save(tmp_path / "in" / "a.jpg")
    (tmp_path / "in" / "note.txt").write_text("x")
    out = tmp_path / "out"
    assert cli.main([str(tmp_path / "in"), "--input-type", "illust", "--height", "32",
                     "-o", str(out), "-q"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a_jpg.png", "a_png.png", "b.png"]
    assert Image.open(out / "b.png").size == (32, 32)


@pytest.mark.parametrize("extra", [["--input-type", "illust"],
                                   ["--input-type", "illust", "--scale", "0.5", "--width", "10"],
                                   ["--input-type", "illust", "--scale", "abc"],
                                   ["--input-type", "illust", "--width", "0"]])
def test_cli_rejects_bad_size(tmp_path, extra):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2


def test_cli_rejects_missing_input_and_png_for_many(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "none.png"), "--input-type", "illust", "--scale", "0.5"])
    a = _write(tmp_path / "a.png", 32, 48)
    b = _write(tmp_path / "b.png", 32, 48)
    with pytest.raises(SystemExit):
        cli.main([str(a), str(b), "--input-type", "illust", "--scale", "0.5", "-o", str(tmp_path / "x.png")])


def test_cli_unreadable_file_returns_1(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert cli.main([str(bad), "--input-type", "illust", "--scale", "0.5",
                     "-o", str(tmp_path / "o"), "-q"]) == 1


def test_python_m_entry_point(tmp_path):
    src = _write(tmp_path / "chara.png", 64, 96)
    dst = tmp_path / "o.png"
    r = subprocess.run([sys.executable, "-m", "apps.sprite", str(src), "--input-type", "illust", "--scale", "0.5",
                        "-o", str(dst), "-q"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert Image.open(dst).size == (32, 48)


@pytest.mark.parametrize("method", ["mode", "median"])
def test_cli_pixel_methods(tmp_path, method):
    src = _write(tmp_path / "chara.png", transparent=True)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "--input-type", "illust", "--width", "64", "--resize-method", method, "--line-priority",
                     "--line-ratio", "0.3", "--line-luma", "70", "--mode-bin", "16",
                     "--transparent-ratio", "0.6", "--no-simplify", "-o", str(dst), "-q"]) == 0
    out = np.asarray(Image.open(dst))
    assert out.shape == (96, 64, 4)
    assert set(np.unique(out[..., 3])) <= {0, 255}
    src_rgb = {tuple(c) for c in make_character(transparent=True)[..., :3].reshape(-1, 3)}
    opaque = out[out[..., 3] == 255][:, :3]
    assert {tuple(c) for c in opaque} <= src_rgb


def test_cli_default_is_mode_with_line_priority():
    args = cli.build_parser().parse_args(["x.png", "--input-type", "illust", "--scale", "0.5"])
    step = _step(args, "resize")
    assert step.method == "mode" and step.line_priority


@pytest.mark.parametrize("extra, method, line", [
    (["--no-line-priority"], "mode", False),
    (["--resize-method", "median"], "median", True),
    (["--resize-method", "nearest"], "nearest", False),
    (["--resize-method", "lanczos"], "lanczos", False),
])
def test_cli_line_priority_follows_method(extra, method, line):
    args = cli.build_parser().parse_args(["x.png", "--input-type", "illust", "--scale", "0.5", *extra])
    step = _step(args, "resize")
    assert step.method == method and step.line_priority is line


def test_cli_default_does_not_change_resize_step_default():
    from retrocore.sprite import ResizeStep
    step = ResizeStep(scale=0.5)
    assert step.method == "nearest" and not step.line_priority


def test_cli_default_matches_explicit_mode_line_priority(tmp_path):
    src = _write(tmp_path / "chara.png", transparent=True)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert cli.main([str(src), "--input-type", "illust", "--width", "64", "-o", str(a), "-q"]) == 0
    assert cli.main([str(src), "--input-type", "illust", "--width", "64", "--resize-method", "mode", "--line-priority",
                     "-o", str(b), "-q"]) == 0
    assert np.array_equal(np.asarray(Image.open(a)), np.asarray(Image.open(b)))


def test_cli_preview_scale(tmp_path, capsys):
    src = _write(tmp_path / "chara.png", transparent=True)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "--input-type", "illust", "--width", "32", "--preview-scale", "4", "-o", str(dst)]) == 0
    out = np.asarray(Image.open(dst))
    pv = np.asarray(Image.open(tmp_path / "out_x4.png"))
    assert pv.shape == (48 * 4, 32 * 4, 4)
    assert np.array_equal(pv[::4, ::4], out)
    assert np.array_equal(pv[3::4, 3::4], out)
    assert "プレビュー" in capsys.readouterr().out


def test_cli_preview_scale_folder_and_absent(tmp_path):
    _write(tmp_path / "in" / "a.png", 64, 64)
    out = tmp_path / "out"
    assert cli.main([str(tmp_path / "in"), "--input-type", "illust", "--height", "16", "-o", str(out), "-q"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.png"]
    assert cli.main([str(tmp_path / "in"), "--input-type", "illust", "--height", "16", "-o", str(out), "-q",
                     "--preview-scale", "2"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "a_x2.png"]
    assert Image.open(out / "a_x2.png").size == (32, 32)


@pytest.mark.parametrize("value", ["0", "33", "1.5", "abc"])
def test_cli_rejects_bad_preview_scale(tmp_path, value):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "--input-type", "illust", "--scale", "0.5", "--preview-scale", value,
                  "-o", str(tmp_path / "o.png")])
    assert e.value.code == 2


@pytest.mark.parametrize("extra", [["--resize-method", "nearest", "--line-priority"],
                                   ["--resize-method", "mode", "--line-ratio", "0"],
                                   ["--resize-method", "mode", "--mode-bin", "300"],
                                   ["--resize-method", "mode", "--line-luma", "-1"],
                                   ["--resize-method", "mode", "--transparent-ratio", "2"]])
def test_cli_rejects_bad_pixel_options(tmp_path, extra):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "--input-type", "illust", "--scale", "0.5", "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2


# --- 背景除去 ---------------------------------------------------------------

def test_cli_background_is_default_and_first():
    from retrocore.sprite import BackgroundStep, CleanupStep, OutlineStep, ResizeStep, SimplifyStep
    from retrocore.sprite.background import DEFAULT_TOLERANCE
    steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png", "--input-type", "illust", "--scale", "0.5"])).steps
    assert [type(s) for s in steps] == [BackgroundStep, ResizeStep, SimplifyStep, CleanupStep,
                                          OutlineStep]   # 縮小の前（元の大きさ）
    bg = steps[0]
    assert bg.color is None and bg.tolerance == DEFAULT_TOLERANCE and not bg.remove_enclosed


def test_cli_background_options():
    args = cli.build_parser().parse_args(["x.png", "--input-type", "illust", "--scale", "0.5", "--bg-color", "#5aaa78",
                                          "--bg-tolerance", "12", "--bg-remove-enclosed"])
    bg = cli.build_pipeline(args).steps[0]
    assert bg.color == (90, 170, 120) and bg.tolerance == 12 and bg.remove_enclosed
    args = cli.build_parser().parse_args(["x.png", "--input-type", "illust", "--scale", "0.5", "--no-background"])
    assert [s.name for s in cli.build_pipeline(args).steps] == ["resize", "simplify", "cleanup", "outline"]


def test_cli_removes_background_before_resize(tmp_path):
    from synth_sprite import BG, character_mask
    src = _write(tmp_path / "chara.png")                    # 不透明（背景は単色の緑）
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "--input-type", "illust", "--width", "64", "-o", str(dst), "-q"]) == 0
    out = np.asarray(Image.open(dst)).astype(int)
    assert set(np.unique(out[..., 3])) == {0, 255}
    op = out[..., 3] == 255
    assert not (op & (np.abs(out[..., :3] - np.array(BG)).max(-1) <= 32)).any()
    ref = character_mask().reshape(96, 4, 64, 4).mean((1, 3)) >= 0.5
    assert (ref == op).mean() > 0.97
    # --no-background なら従来どおり（すべて不透明）
    assert cli.main([str(src), "--input-type", "illust", "--width", "64", "--no-background", "--no-outline",
                     "-o", str(dst), "-q"]) == 0
    assert (np.asarray(Image.open(dst))[..., 3] == 255).all()


@pytest.mark.parametrize("extra", [["--bg-tolerance", "300"], ["--bg-tolerance", "-1"],
                                   ["--bg-tolerance", "x"], ["--bg-color", "red"],
                                   ["--bg-color", "1,2,256"]])
def test_cli_rejects_bad_background_options(tmp_path, extra):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "--input-type", "illust", "--scale", "0.5", "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2


# ---------------------------------------------------------------- 入力の種類（ドット絵風 / 普通のイラスト）

def _write_pixel_art(path, seed=0):
    from synth_sprite import enlarge_like_ai, make_pixel_sprite
    art = make_pixel_sprite()
    img, _, _ = enlarge_like_ai(art, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGBA").save(path)
    return art


def test_cli_default_input_type_is_pixel_and_uses_grid(tmp_path, capsys):
    src = tmp_path / "pix.png"
    art = _write_pixel_art(src)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "-o", str(dst), "--save-steps", "--preview-scale", "4"]) == 0
    out = np.asarray(Image.open(dst))
    assert out.shape == art.shape
    assert (tmp_path / "out_steps" / "02_grid.png").is_file()
    assert np.asarray(Image.open(tmp_path / "out_x4.png")).shape[:2] == (art.shape[0] * 4, art.shape[1] * 4)
    text = capsys.readouterr().out
    assert "ドット絵風" in text and "グリッド検出" in text and "色" in text


def test_cli_pipeline_by_input_type():
    steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png"])).steps
    assert [s.name for s in steps] == ["background", "grid", "simplify", "cleanup", "outline"]
    args = cli.build_parser().parse_args(["x.png", "--dot-size", "20x21", "--grid-origin", "3,4",
                                          "--no-grid-snap", "--grid-tolerance", "0.1",
                                          "--transparent-ratio", "0.6"])
    g = _step(args, "grid")
    assert g.dot_size == (20.0, 21.0) and g.origin == (3.0, 4.0) and g.snap is False
    assert g.snap_tolerance == 0.1 and g.transparent_ratio == 0.6
    steps = cli.build_pipeline(cli.build_parser().parse_args(
        ["x.png", "--input-type", "illust", "--width", "64"])).steps
    assert [s.name for s in steps] == ["background", "resize", "simplify", "cleanup", "outline"]


@pytest.mark.parametrize("extra, words", [
    (["--width", "64"], ["--width", "グリッド検出で決まります", "--input-type illust"]),
    (["--scale", "1/4"], ["--scale"]),
    (["--resize-method", "mode"], ["縮小のオプション"]),
    (["--no-line-priority"], ["--no-line-priority"]),
    (["--input-type", "illust", "--width", "64", "--dot-size", "20"], ["グリッド検出のオプション"]),
    (["--dot-size", "abc"], ["20x21"]),
    (["--dot-size", "1"], ["2 以上"]),
    (["--grid-origin", "3"], ["X,Y"]),
    (["--grid-tolerance", "1.5"], ["1 未満"]),
])
def test_cli_rejects_options_for_other_input_type(tmp_path, capsys, extra, words):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2
    err = capsys.readouterr().err
    for w in words:
        assert w in err


def test_cli_grid_not_found_stops_with_message(tmp_path, capsys):
    src = _write(tmp_path / "chara.png", 256, 384)  # 普通のイラスト（ドット絵風ではない）
    dst = tmp_path / "o.png"
    assert cli.main([str(src), "-o", str(dst), "-q"]) == 1
    assert not dst.exists()
    err = capsys.readouterr().err
    assert "変換できません" in err and "見つかりませんでした" in err and "--input-type illust" in err


def test_cli_grid_not_found_continues_with_other_files(tmp_path):
    _write(tmp_path / "in" / "a_illust.png", 256, 384)
    _write_pixel_art(tmp_path / "in" / "b_pixel.png")
    out = tmp_path / "out"
    assert cli.main([str(tmp_path / "in"), "-o", str(out), "-q"]) == 1
    assert [p.name for p in out.iterdir()] == ["b_pixel.png"]


def test_cli_manual_dot_size_skips_detection(tmp_path):
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (6, 8, 3)).astype(np.uint8)
    big = np.repeat(np.repeat(small, 10, 0), 10, 1)
    src = tmp_path / "noise.png"
    Image.fromarray(big).save(src)
    dst = tmp_path / "o.png"
    assert cli.main([str(src), "--no-background", "--dot-size", "10", "--grid-origin", "0,0", "--no-simplify",
                     "--no-cleanup", "--no-outline", "-o", str(dst), "-q"]) == 0
    assert np.array_equal(np.asarray(Image.open(dst))[..., :3], small)


@pytest.mark.parametrize("extra, square", [([], True), (["--dot-shape", "square"], True),
                                           (["--dot-shape", "separate"], False)])
def test_cli_dot_shape(extra, square):
    g = _step(cli.build_parser().parse_args(["x.png", *extra]), "grid")
    assert g.square is square


def test_cli_dot_shape_only_for_pixel(tmp_path, capsys):
    src = _write(tmp_path / "chara.png", 32, 48)
    with pytest.raises(SystemExit):
        cli.main([str(src), "--input-type", "illust", "--width", "16", "--dot-shape", "separate"])
    assert "グリッド検出のオプション" in capsys.readouterr().err


# ---------------------------------------------------------------- 単純化

def test_cli_simplify_is_default_and_before_cleanup():
    from retrocore.sprite.simplify import DEFAULT_COLORS, DEFAULT_MERGE_DISTANCE
    for extra in ([], ["--input-type", "illust", "--width", "32"]):
        steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png", *extra])).steps
        assert [s.name for s in steps[-3:]] == ["simplify", "cleanup", "outline"]
        assert steps[-3].colors == DEFAULT_COLORS == 16
        assert steps[-3].merge_distance == DEFAULT_MERGE_DISTANCE == 0.03
    s = _step(cli.build_parser().parse_args(["x.png", "--colors", "8", "--merge-distance", "0.05"]), "simplify")
    assert s.colors == 8 and s.merge_distance == 0.05
    assert _step(cli.build_parser().parse_args(["x.png", "--merge-distance", "0"]), "simplify").merge_distance == 0
    assert _step(cli.build_parser().parse_args(["x.png"]), "simplify").keep_large == 0.01
    assert _step(cli.build_parser().parse_args(["x.png", "--keep-large", "0"]), "simplify").keep_large == 0
    assert _step(cli.build_parser().parse_args(["x.png", "--keep-large", "0.05"]), "simplify").keep_large == 0.05
    args = cli.build_parser().parse_args(["x.png", "--no-simplify"])
    assert [s.name for s in cli.build_pipeline(args).steps] == ["background", "grid", "cleanup", "outline"]


def test_cli_simplify_end_to_end(tmp_path, capsys):
    src = tmp_path / "pix.png"
    art = _write_pixel_art(src)
    dst = tmp_path / "out.png"
    assert cli.main([str(src), "-o", str(dst), "--colors", "10", "--save-steps", "--no-outline"]) == 0
    out = np.asarray(Image.open(dst))
    assert out.shape == art.shape
    opaque = out[out[..., 3] == 255][:, :3]
    assert 2 <= len({tuple(c) for c in opaque}) <= 10
    assert (tmp_path / "out_steps" / "03_simplify.png").is_file()
    grid = np.asarray(Image.open(tmp_path / "out_steps" / "02_grid.png"))
    assert np.array_equal(grid[..., 3], out[..., 3])     # 透明な部分は変わらない
    assert "単純化" in capsys.readouterr().out


@pytest.mark.parametrize("extra, words", [
    (["--colors", "0"], ["1〜256"]),
    (["--colors", "300"], ["1〜256"]),
    (["--merge-distance", "-0.1"], ["0〜1"]),
    (["--merge-distance", "1.5"], ["0〜1"]),
    (["--merge-distance", "x"], ["数値"]),
    (["--keep-large", "-0.1"], ["0〜1"]),
    (["--keep-large", "2"], ["0〜1"]),
    (["--keep-large", "x"], ["数値"]),
    (["--no-simplify", "--keep-large", "0"], ["--keep-large", "--no-simplify"]),
    (["--no-simplify", "--colors", "8"], ["--colors", "--no-simplify"]),
])
def test_cli_rejects_bad_simplify_options(tmp_path, capsys, extra, words):
    src = tmp_path / "pix.png"
    _write_pixel_art(src)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert all(w in err for w in words), err


# ---------------------------------------------------------------- クリーンアップ

def test_cli_cleanup_is_default_and_last():
    from retrocore.sprite.cleanup import DEFAULT_LEVEL
    for extra in ([], ["--input-type", "illust", "--width", "32"], ["--no-simplify"]):
        steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png", *extra])).steps
        assert steps[-2].name == "cleanup" and steps[-2].level == DEFAULT_LEVEL == "normal"
        assert steps[-1].name == "outline"
    for lv in ("light", "normal", "strong"):
        assert _step(cli.build_parser().parse_args(["x.png", "--cleanup", lv]), "cleanup").level == lv
    for extra in (["--no-cleanup"], ["--cleanup", "off"], ["--no-cleanup", "--cleanup", "off"]):
        steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png", *extra])).steps
        assert [s.name for s in steps] == ["background", "grid", "simplify", "outline"]


def test_cli_no_cleanup_equals_simplify_output(tmp_path, capsys):
    src = tmp_path / "pix.png"
    _write_pixel_art(src)
    assert cli.main([str(src), "-o", str(tmp_path / "a.png"), "--colors", "10", "--no-cleanup", "--save-steps",
                     "--no-outline"]) == 0
    assert "クリーンアップ" not in capsys.readouterr().out
    assert cli.main([str(src), "-o", str(tmp_path / "b.png"), "--colors", "10", "--cleanup", "strong",
                     "--save-steps"]) == 0
    assert "クリーンアップ（strong" in capsys.readouterr().out
    a = np.asarray(Image.open(tmp_path / "a.png"))
    simp = np.asarray(Image.open(tmp_path / "b_steps" / "03_simplify.png"))
    assert np.array_equal(a, simp)                  # --no-cleanup は単純化の結果のまま
    assert (tmp_path / "b_steps" / "04_cleanup.png").is_file()
    b = np.asarray(Image.open(tmp_path / "b.png"))
    # 新しい色は増えない
    assert {tuple(c) for c in b[b[..., 3] == 255][:, :3]} <= {tuple(c) for c in simp[simp[..., 3] == 255][:, :3]}


def test_cli_rejects_cleanup_with_no_cleanup(tmp_path, capsys):
    src = tmp_path / "pix.png"
    _write_pixel_art(src)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), "--no-cleanup", "--cleanup", "strong"])
    assert e.value.code == 2
    assert "--no-cleanup" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([str(src), "--cleanup", "max"])


# ---------------------------------------------------------------- 縁取り

def test_cli_outline_is_default_and_last():
    from retrocore.sprite.outline import DEFAULT_COLOR_MODE, DEFAULT_MAX_COLORS, DEFAULT_MODE, DEFAULT_SHAPE
    for extra in ([], ["--input-type", "illust", "--width", "32"], ["--no-cleanup"]):
        st = cli.build_pipeline(cli.build_parser().parse_args(["x.png", *extra])).steps[-1]
        assert st.name == "outline" and st.mode == DEFAULT_MODE == "auto"
        assert st.shape == DEFAULT_SHAPE == "thin" and st.color_mode == DEFAULT_COLOR_MODE == "darken"
        assert st.color is None and st.pad and st.max_colors == DEFAULT_MAX_COLORS
    for extra in (["--no-outline"], ["--outline", "off"], ["--no-outline", "--outline", "off"]):
        steps = cli.build_pipeline(cli.build_parser().parse_args(["x.png", *extra])).steps
        assert steps[-1].name == "cleanup"


@pytest.mark.parametrize("extra, expect", [
    (["--outline", "add"], dict(mode="add")),
    (["--outline-color", "single"], dict(color_mode="single", color=None)),
    (["--outline-color", "#3a2030"], dict(color_mode="single", color=(0x3a, 0x20, 0x30))),
    (["--outline-color", "58,32,48"], dict(color=(58, 32, 48))),
    (["--outline-shape", "thick"], dict(shape="thick")),
    (["--outline-colors", "2"], dict(max_colors=2)),
    (["--no-outline-pad"], dict(pad=False)),
])
def test_cli_outline_options(extra, expect):
    st = _step(cli.build_parser().parse_args(["x.png", *extra]), "outline")
    for k, v in expect.items():
        assert getattr(st, k) == v, k


@pytest.mark.parametrize("extra, words", [
    (["--no-outline", "--outline", "add"], ["--no-outline"]),
    (["--no-outline", "--outline-shape", "thick"], ["--outline-shape", "縁取りのオプション"]),
    (["--outline-color", "dark"], ["darken / single"]),
    (["--outline", "max"], ["invalid choice"]),
])
def test_cli_rejects_bad_outline_options(tmp_path, capsys, extra, words):
    src = tmp_path / "pix.png"
    _write_pixel_art(src)
    with pytest.raises(SystemExit) as e:
        cli.main([str(src), "-o", str(tmp_path / "o.png"), *extra])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert all(w in err for w in words), err


def test_cli_outline_end_to_end(tmp_path, capsys):
    src = tmp_path / "pix.png"
    _write_pixel_art(src)
    assert cli.main([str(src), "-o", str(tmp_path / "a.png"), "--colors", "10", "--save-steps"]) == 0
    text = capsys.readouterr().out
    assert "縁取り（auto" in text
    assert (tmp_path / "a_steps" / "05_outline.png").is_file()
    before = np.asarray(Image.open(tmp_path / "a_steps" / "04_cleanup.png"))
    a = np.asarray(Image.open(tmp_path / "a.png"))
    # 合成のドット絵は暗い輪郭を持つので、auto は新しい輪郭を付けない（大きさもシルエットも変わらない）
    assert "輪郭あり" in text and a.shape == before.shape
    assert np.array_equal(a[..., 3], before[..., 3])
    # add は外側に付ける（透明だったドットだけが変わる）
    assert cli.main([str(src), "-o", str(tmp_path / "b.png"), "--colors", "10", "--outline", "add", "-q"]) == 0
    b = np.asarray(Image.open(tmp_path / "b.png"))
    assert (b[..., 3] == 255).sum() > (before[..., 3] == 255).sum()
    # --no-outline はクリーンアップの結果のまま
    assert cli.main([str(src), "-o", str(tmp_path / "c.png"), "--colors", "10", "--no-outline", "-q"]) == 0
    assert np.array_equal(np.asarray(Image.open(tmp_path / "c.png")), before)
