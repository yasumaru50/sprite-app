"""動きの GIF アニメ（`retrocore/sprite/motion/gif.py`、CLI の `--gif`、ブラウザ版の「⑥ GIF を保存」）のテスト。

- GIF を読み直した各フレームが、シートの各フレーム（拡大と背景だけを変えたもの）と画素単位で同じ（新しい色・ずれが無い）
- パレットはシートで使っている色だけ（透明のときは 0 番が透明）
- 表示時間はシートの JSON の duration_ms（10 ミリ秒単位に丸めるときは合計を保って警告）、ずっと繰り返す
- 同じ絵が続くフレームは 1 枚にまとめ、合計の時間は変わらない
- CLI と画面（`gif_for_web()`。材料はシートの PNG と JSON）で同じ設定なら同じバイト列
- 32 ビット（Pyodide）のつもりでも通る（int32-check のジョブでも動かす）
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageSequence

from apps.sprite import cli
from int32_numpy import AVAILABLE, emulate_int32
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import encode_png, gif_for_web, gif_image_bytes, motion_image_bytes
from retrocore.sprite.motion import make_motion, make_sheet, sheet_json
from retrocore.sprite.motion import gif as gif_mod
from retrocore.sprite.motion.gif import frames_from_sheet, gif_durations, gif_path, make_gif, motion_gif
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

BG = (32, 48, 64)


@pytest.fixture(scope="module")
def sprite():
    return convert_sprite(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], SpriteSettings()).image


def decode_gif(data: bytes):
    """Pillow で GIF を読み直す（フレームは重ねたあとの RGBA）。(フレーム, 時間, loop, パレットの色の数)"""
    im = Image.open(io.BytesIO(data))
    loop = im.info.get("loop")
    frames, ms = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(np.asarray(fr.convert("RGBA")).copy())
        ms.append(fr.info.get("duration"))
    return frames, ms, loop


def expected_frames(frames, scale, bg):
    out = []
    for f in frames:
        a = np.repeat(np.repeat(np.asarray(f, np.uint8), scale, 0), scale, 1)
        if bg is not None:
            a = np.where(a[..., 3:] == 0, np.array(list(bg) + [255], np.uint8), a)
        out.append(a)
    return out


def same_picture(a, b) -> bool:
    """透明の画素の R・G・B は比べない（GIF では意味が無い）。"""
    na = np.where(a[..., 3:] == 0, 0, a)
    nb = np.where(b[..., 3:] == 0, 0, b)
    return a.shape == b.shape and np.array_equal(na, nb)


def colors_of(frames) -> set:
    out = set()
    for f in frames:
        o = f[..., 3] == 255
        out |= {tuple(c) for c in f[o][:, :3].tolist()}
    return out


# ---- フレームの絵がシートと同じ ----

@pytest.mark.parametrize("name", ["breathe", "sway", "float", "bounce", "stretch"])
@pytest.mark.parametrize("scale", [1, 3, 4])
@pytest.mark.parametrize("bg", [None, BG])
def test_gif_frames_equal_sheet_frames(sprite, name, scale, bg):
    res = make_motion(sprite, name)
    assert not res.skipped
    gif = motion_gif(res, scale, bg)
    frames, ms, loop = decode_gif(gif.data)
    exp = expected_frames(res.frames, scale, bg)
    assert len(frames) == len(exp)
    assert all(same_picture(a, b) for a, b in zip(frames, exp))
    assert ms == [int(d) for d in res.durations] == gif.durations   # 10 で割り切れる既定の時間はそのまま
    assert loop == 0   # ずっと繰り返す
    # 新しい色を作らない（背景色ありのときは背景色だけが増える）
    used = colors_of(res.frames)
    assert colors_of(frames) <= used | ({tuple(bg)} if bg else set())
    assert gif.colors == len(used | ({tuple(bg)} if bg else set()))
    assert gif.warnings == [] and gif.merged == []


def test_palette_is_the_sheet_colors(sprite):
    res = make_motion(sprite, "breathe")
    im = Image.open(io.BytesIO(motion_gif(res, 1).data))
    pal = im.getpalette()
    used = sorted(colors_of(res.frames))
    got = [tuple(pal[i * 3:i * 3 + 3]) for i in range(1, len(used) + 1)]
    assert got == used                         # 1 番から、使っている色を並べたもの（Pillow の減色を通さない）
    assert im.info.get("transparency") == 0    # 0 番が透明
    assert im.mode == "P"


def test_background_same_as_a_sprite_color(sprite):
    """背景色が絵の色と同じでも、色の数は増えず、絵は同じ。"""
    res = make_motion(sprite, "sway")
    col = sorted(colors_of(res.frames))[0]
    gif = motion_gif(res, 2, col)
    frames, _, _ = decode_gif(gif.data)
    assert gif.colors == len(colors_of(res.frames))
    assert all(same_picture(a, b) for a, b in zip(frames, expected_frames(res.frames, 2, col)))


def test_transparent_gif_does_not_keep_previous_frame():
    """透明のときは前のフレームが残らない（disposal = 2）。上下に動く絵で、前のフレームだけにあった所が透明に戻る。"""
    a = np.zeros((6, 4, 4), np.uint8)
    b = a.copy()
    a[0:3, 1:3] = (200, 50, 50, 255)
    b[3:6, 1:3] = (200, 50, 50, 255)
    data = make_gif([a, b, a], [100, 100, 100], 1).data
    frames, _, _ = decode_gif(data)
    assert all(same_picture(x, y) for x, y in zip(frames, [a, b, a]))
    im = Image.open(io.BytesIO(data))
    im.seek(1)
    assert im.disposal_method == 2


# ---- 表示時間 ----

def test_durations_round_to_10ms_keeping_total():
    ms, warns = gif_durations([125, 125, 125, 125])
    assert ms == [130, 120, 130, 120] and sum(ms) == 500
    assert len(warns) == 1 and "10 ミリ秒単位" in warns[0] and "合計 500 ミリ秒は変えていません" in warns[0]
    assert gif_durations([500, 500]) == ([500, 500], [])
    ms, warns = gif_durations([333, 333, 334])
    assert ms == [330, 340, 330] and sum(ms) == 1000


def test_durations_minimum_10ms_and_browser_warning():
    ms, warns = gif_durations([3, 3, 3])
    assert ms == [10, 10, 10]
    assert any("合計 9 → 30" in w for w in warns)
    assert any("10 ミリ秒より短くできない" in w for w in warns)
    assert any("100 ミリ秒として表示" in w for w in warns)
    with pytest.raises(ValueError):
        gif_durations([0])


def test_rounded_durations_are_written(sprite):
    res = make_motion(sprite, "bounce", durations=[125, 125, 125, 125])
    gif = motion_gif(res, 1)
    _, ms, _ = decode_gif(gif.data)
    assert ms == gif.durations == gif.frame_durations == [130, 120, 130, 120]
    assert gif.warnings and "125 → 130" in gif.warnings[0]


# ---- 同じ絵が続くフレーム ----

def test_identical_frames_are_merged_and_total_kept():
    a = np.zeros((4, 4, 4), np.uint8)
    a[1:3, 1:3] = (10, 20, 30, 255)
    b = a.copy()
    b[0, 0] = (10, 20, 30, 255)
    gif = make_gif([a, a, b, b, b, a], [100, 150, 200, 100, 50, 300], 2)
    frames, ms, _ = decode_gif(gif.data)
    assert gif.frames_of == [[0, 1], [2, 3, 4], [5]] and gif.merged == [[0, 1], [2, 3, 4]]
    assert ms == gif.durations == [250, 350, 300] and sum(ms) == 900
    assert all(same_picture(x, y) for x, y in zip(frames, expected_frames([a, b, a], 2, None)))


def test_all_identical_frames_make_one_frame():
    a = np.zeros((3, 3, 4), np.uint8)
    a[1, 1] = (1, 2, 3, 255)
    gif = make_gif([a, a], [500, 500], 1)
    frames, ms, _ = decode_gif(gif.data)
    assert len(frames) == 1 and gif.durations == [1000]


# ---- 作れないとき・設定の間違い ----

def test_too_many_colors():
    a = np.zeros((1, 256, 4), np.uint8)
    a[0, :, 0] = np.arange(256)
    a[..., 3] = 255
    with pytest.raises(ValueError, match="256 色まで"):
        make_gif([a], [100], 1)                  # 透明のときは 255 色まで
    assert make_gif([a], [100], 1, (0, 0, 0)).colors == 256   # 背景色が絵の色と同じなら 256 色で作れる
    with pytest.raises(ValueError, match="256 色まで"):
        make_gif([a], [100], 1, (1, 1, 1))


def test_partial_alpha_is_thresholded_with_warning():
    a = np.zeros((2, 2, 4), np.uint8)
    a[0, 0] = (50, 60, 70, 255)
    a[0, 1] = (50, 60, 70, 200)
    a[1, 0] = (50, 60, 70, 40)
    gif = make_gif([a], [100], 1)
    assert any("半透明の画素 2 個" in w for w in gif.warnings)
    frames, _, _ = decode_gif(gif.data)
    assert frames[0][0, 1, 3] == 255 and frames[0][1, 0, 3] == 0


@pytest.mark.parametrize("scale", [0, 9, 2.0, True, "4"])
def test_bad_scale(scale):
    a = np.zeros((2, 2, 4), np.uint8)
    with pytest.raises(ValueError, match="拡大"):
        make_gif([a], [100], scale)


def test_background_parse():
    assert gif_mod.check_background(None) is None
    assert gif_mod.check_background("transparent") is None
    assert gif_mod.check_background("#203040") == (32, 48, 64)
    assert gif_mod.check_background("FFffFF") == (255, 255, 255)
    for bad in ["#12345", "red", "#ggg000", (0, 0, 256)]:
        with pytest.raises(ValueError):
            gif_mod.check_background(bad)


def test_skipped_motion_makes_no_gif():
    res = make_motion(make_slime(), "nod")
    assert res.skipped
    with pytest.raises(ValueError, match="作らなかった"):
        motion_gif(res)


def test_gif_path():
    assert gif_path("out/hero_breathe_sheet.png").name == "hero_breathe.gif"
    assert gif_path("out/hero_bounce_stretch_sheet.png").name == "hero_bounce_stretch.gif"


# ---- CLI ----

@pytest.fixture(scope="module")
def hero_png(tmp_path_factory):
    path = tmp_path_factory.mktemp("in") / "hero.png"
    Image.fromarray(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], "RGBA").save(path)
    return path


def test_cli_gif_matches_sheet(tmp_path, hero_png, capsys):
    dst = tmp_path / "o" / "hero.png"
    assert cli.main([str(hero_png), "-o", str(dst), "--motion", "sway", "--gif"]) == 0
    out = capsys.readouterr().out
    gp = dst.parent / "hero_sway.gif"
    assert gp.exists() and "hero_sway.gif（GIF: 4 フレーム" in out and "（4 倍）、透明" in out
    sheet = np.asarray(Image.open(dst.parent / "hero_sway_sheet.png").convert("RGBA"))
    info = json.loads((dst.parent / "hero_sway_sheet.json").read_text(encoding="utf-8"))
    frames, durations = frames_from_sheet(sheet, info)
    got, ms, loop = decode_gif(gp.read_bytes())
    assert all(same_picture(a, b) for a, b in zip(got, expected_frames(frames, 4, None)))
    assert ms == durations and loop == 0
    assert "gif" not in json.dumps(info)   # シートの JSON は GIF の有無で変わらない


def test_cli_gif_scale_background_and_spec(tmp_path, hero_png, capsys):
    dst = tmp_path / "o" / "hero.png"
    spec = cli.Path(__file__).resolve().parents[1] / "apps" / "sprite" / "examples" / "bounce_stretch.motion.json"
    assert cli.main([str(hero_png), "-o", str(dst), "--motion-spec", str(spec), "--gif", "--gif-scale", "2",
                     "--gif-background", "#203040"]) == 0
    gp = dst.parent / "hero_bounce_stretch.gif"
    got, ms, _ = decode_gif(gp.read_bytes())
    sheet = np.asarray(Image.open(dst.parent / "hero_bounce_stretch_sheet.png").convert("RGBA"))
    info = json.loads((dst.parent / "hero_bounce_stretch_sheet.json").read_text(encoding="utf-8"))
    frames, durations = frames_from_sheet(sheet, info)
    assert all(same_picture(a, b) for a, b in zip(got, expected_frames(frames, 2, BG)))
    assert ms == durations
    assert "背景 #203040" in capsys.readouterr().out


def test_cli_gif_warns_on_rounding(tmp_path, hero_png, capsys):
    dst = tmp_path / "o" / "hero.png"
    assert cli.main([str(hero_png), "-o", str(dst), "-q", "--motion", "breathe", "--motion-duration", "125", "--gif"]) == 0
    err = capsys.readouterr().err
    assert "10 ミリ秒単位" in err and "125 → 130" in err
    _, ms, _ = decode_gif((dst.parent / "hero_breathe.gif").read_bytes())
    assert ms == [130, 120]


def test_cli_no_gif_when_skipped(tmp_path, capsys):
    path = tmp_path / "slime.png"
    Image.fromarray(np.kron(make_slime(), np.ones((8, 8, 1), np.uint8)), "RGBA").save(path)
    dst = tmp_path / "o" / "slime.png"
    assert cli.main([str(path), "-o", str(dst), "--dot-size", "8", "--motion", "nod", "--gif"]) == 0
    assert "GIF も書きませんでした" in capsys.readouterr().err
    assert sorted(p.name for p in dst.parent.iterdir()) == ["slime.png", "slime_nod_skipped.json"]


@pytest.mark.parametrize("argv, message", [
    (["--gif"], "--gif は --motion"),
    (["--motion", "breathe", "--gif-scale", "2"], "--gif-scale は GIF のオプション"),
    (["--motion", "breathe", "--gif-background", "#000000"], "--gif-background は GIF のオプション"),
    (["--motion", "breathe", "--gif", "--gif-scale", "9"], "1〜8"),
    (["--motion", "breathe", "--gif", "--gif-background", "red"], "#rrggbb"),
])
def test_cli_gif_option_errors(hero_png, capsys, argv, message):
    with pytest.raises(SystemExit) as e:
        cli.main([str(hero_png)] + argv)
    assert e.value.code == 2 and message in capsys.readouterr().err


# ---- 画面（Pyodide）と CLI で同じ GIF ----

@pytest.mark.parametrize("options, gif_options", [
    ({"motion": "breathe"}, {}),
    ({"motion": "sway", "split": 10}, {"scale": 1}),
    ({"motion": "bounce", "durations": 125}, {"scale": 8, "background": "#203040"}),
])
def test_web_gif_equals_cli(tmp_path, hero_png, options, gif_options):
    dst = tmp_path / "o" / "hero.png"
    argv = [str(hero_png), "-o", str(dst), "-q", "--motion", options["motion"], "--gif"]
    if "split" in options:
        argv += ["--sway-split", str(options["split"])]
    if "durations" in options:
        argv += ["--motion-duration", str(options["durations"])]
    if "scale" in gif_options:
        argv += ["--gif-scale", str(gif_options["scale"])]
    if "background" in gif_options:
        argv += ["--gif-background", gif_options["background"]]
    assert cli.main(argv) == 0
    cli_gif = (dst.parent / f"hero_{options['motion']}.gif").read_bytes()
    m = motion_image_bytes(dst.read_bytes(), dict(options, image_name=f"hero_{options['motion']}_sheet.png"))
    assert m["ok"] and not m["skipped"]
    res = json.loads(gif_for_web(m["sheet_png"], m["sheet_json"], json.dumps(gif_options)))
    assert res["ok"], res
    assert base64.b64decode(res["gif_base64"]) == cli_gif
    assert res["name"] == f"hero_{options['motion']}.gif"
    assert res["scale"] == gif_options.get("scale", 4) and res["background"] == gif_options.get("background")


def test_web_gif_errors():
    a = np.zeros((2, 2, 4), np.uint8)
    png = encode_png(a)
    res = make_motion(make_slime(), "breathe")
    info = sheet_json(res, "s_breathe_sheet.png")
    assert not gif_image_bytes(png, info, {"scale": 4})["ok"]                       # フレームの位置がシートの外
    sheet = encode_png(make_sheet(res.frames))
    assert not gif_image_bytes(sheet, info, {"scale": 0})["ok"]
    assert not gif_image_bytes(sheet, info, {"speed": 2})["ok"]
    assert "知らない" in json.loads(gif_for_web(sheet, info, '{"speed": 2}'))["error"]
    assert not json.loads(gif_for_web(sheet, info, "[]"))["ok"]
    assert gif_image_bytes(sheet, info, {})["ok"]


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
@pytest.mark.parametrize("bg", [None, BG])
def test_gif_has_no_int64_index_arrays(sprite, bg):
    res = make_motion(sprite, "bounce")
    expected = motion_gif(res, 3, bg).data
    with emulate_int32():
        got = motion_gif(res, 3, bg).data
    assert got == expected
