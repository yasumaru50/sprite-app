"""Still App の写真の下書き（--mode photo）・輪郭の線（--outline）のテスト。

テスト用の「写真」はコードで合成する（`make_photo()`。壁のグラデーション・ざらつき・暗い家具・明るい窓）。
98 モード・今風モードの既定の出力が前とバイト単位で同じことも、この機能を足す前の版のハッシュと比べて確かめる。
"""

import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from apps.still import cli
from int32_numpy import AVAILABLE, emulate_int32
from retrocore.still import outline as outline_mod
from retrocore.still import photo as photo_mod
from retrocore.still.convert import (MODES, SMOOTH_LEVELS, StillError, StillSettings, convert_still,
                                     convert_still_bytes, spec, still_for_web)
from still_legacy import legacy_json
from synth_still import enlarge_scene, make_modern_scene, make_scene, noisy_scene_rgba


def make_photo(w=960, h=720, seed=3) -> np.ndarray:
    """部屋の写真の代わり（RGB）。左は明るい窓、右下に暗い家具、壁は縦のグラデーションにざらつきを足す。"""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None, None]
    wall = np.array([200, 190, 160]) * (1 - 0.3 * y) + np.zeros((h, w, 3))
    img = wall.copy()
    img[: h // 2, : w // 4] = [245, 250, 255]                       # 窓
    img[h * 3 // 5:, w // 2: w * 5 // 6] = [40, 35, 50]              # 家具
    img[h * 4 // 5:, :] = [120, 80, 70] * np.ones((1, 1, 3))         # 床
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(rgb, orientation=None) -> bytes:
    buf = io.BytesIO()
    im = Image.fromarray(rgb)
    if orientation is None:
        im.save(buf, format="JPEG", quality=90)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(buf, format="JPEG", quality=90, exif=exif.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="module")
def photo():
    return make_photo()


def _photo(rgb, **kw):
    return convert_still(rgb, StillSettings.from_mode("photo", **kw))


# ---- 98 モード・今風モードの出力は前と同じ（この機能を足す前の版のハッシュ） ----

@pytest.mark.parametrize("name, settings, png_hash, json_hash", [
    ("modern", {"mode": "modern", "width": 64}, "ad8ca90d97e27d3f", "5e31160c20716b86"),
    ("modern", {"mode": "modern", "width": 64, "tidy_dots": True}, "85413aa1f5084839", "f54635cd601575b9"),
    ("modern", {"mode": "modern", "width": 64, "merge_similar": False, "colors": 48}, "b18ea02ed4d784d4",
     "ae16a74feb775c2d"),
    ("scene", {"mode": "modern"}, "a9d46f4a1f461da1", "e6d08d86972bea33"),
    ("noisy", {"mode": "modern", "width": 64, "tidy_pattern": True, "tidy_dots": True}, "e38a3998c11e1aaf",
     "cc7e05efdc07677d"),
    ("modern", {"width": 64}, "1b0d6a73b133d629", "026425d7c23ecd2c"),
    ("modern", {"width": 64, "tidy_dots": True}, "38d585227550ff5c", "c555f6ce2d0dbe4d"),
])
def test_modern_and_pc98_output_is_unchanged(name, settings, png_hash, json_hash):
    data = {"modern": lambda: _png(make_modern_scene()[0]), "scene": lambda: _png(enlarge_scene(make_scene(), seed=1)),
            "noisy": lambda: _png(noisy_scene_rgba())}[name]()
    out = convert_still_bytes(data, f"{name}.png", settings)
    assert hashlib.sha256(out["png"]).hexdigest()[:16] == png_hash
    assert hashlib.sha256(legacy_json(out["json"]).encode("utf-8")).hexdigest()[:16] == json_hash
    info = out["info"]
    assert "photo" not in info and "outline" not in info
    assert "smooth" not in info["settings"] and "outline" not in info["settings"]


# ---- プリセットと設定 ----

def test_photo_preset():
    s = StillSettings.from_mode("photo")
    s.validate()
    assert (s.colors, s.four_bit, s.max_size, s.keep_darkest, s.merge_similar, s.tidy_dots, s.width, s.smooth,
            s.outline, s.tidy_pattern) == (32, False, (1920, 1080), True, True, True, 320, "medium", False, False)
    assert set(MODES) == {"pc98", "modern", "photo"}
    assert SMOOTH_LEVELS == ("weak", "medium", "strong")
    assert [x["value"] for x in spec()["smooth_levels"]] == list(SMOOTH_LEVELS)
    assert spec()["modes"]["photo"]["width"] == 320 and spec()["modes"]["photo"]["tidy_dots"] is True
    # 98・今風は孤立したドット・輪郭とも既定は切
    for mode in ("pc98", "modern"):
        m = StillSettings.from_mode(mode)
        assert (m.tidy_dots, m.outline, m.smooth, m.orient, m.find_grid) == (False, False, "off", False, True)


def test_photo_width_null_means_320():
    s = StillSettings.from_dict({"mode": "photo", "width": None})
    assert s.width == 320 and s.smooth == "medium" and s.tidy_dots is True
    assert StillSettings.from_dict({"mode": "photo", "tidy_dots": False}).tidy_dots is False


@pytest.mark.parametrize("values, words", [
    ({"mode": "photo", "smooth": "max"}, "まとめの強さ"),
    ({"smooth": "max"}, "まとめの強さ"),
    ({"find_grid": True, "width": 100}, "一緒"),
    ({"find_grid": False}, "横のマス数"),
    ({"orient": 1}, "true"),
    ({"mode": "photo", "smooth": 2}, "smooth"),
    ({"outline": "yes"}, "true"),
])
def test_bad_photo_settings(values, words):
    with pytest.raises(ValueError, match=words):
        StillSettings.from_dict(values)


def test_photo_json_records_mode_and_settings(photo):
    info = convert_still_bytes(_png(photo), "p.png", {"mode": "photo", "width": 80})["info"]
    st = info["settings"]
    assert st["mode"] == "photo" and st["smooth"] == "medium" and st["tidy_dots"] is True and st["outline"] is False
    assert st["width"] == 80 and st["colors"] == 32 and st["merge_similar"] is True
    assert info["dots"]["method"] == "photo" and info["dots"]["cells"] == [80, 60]
    assert info["dots"]["cell_size"] == [12.0, 12.0]
    p = info["photo"]
    assert p["orientation"] == 1 and p["prescale"] == 4 and p["work_size"] == [320, 240]
    assert p["smooth"] == {"level": "medium", "method": "kuwahara", **photo_mod.SMOOTH["medium"]}
    assert "tidy_dots" in info and "outline" not in info
    off = convert_still_bytes(_png(photo), "p.png", {"mode": "photo", "width": 80, "tidy_dots": False})["info"]
    assert off["settings"]["tidy_dots"] is False and "tidy_dots" not in off


# ---- 工程 ----

def test_exif_orientation_is_applied(photo):
    # Orientation 6（時計回りに 90 度回して見る）: 横長の写真が縦長になる
    data = _jpeg(photo, orientation=6)
    info = convert_still_bytes(data, "p.jpg", {"mode": "photo", "width": 60})["info"]
    assert info["input"] == {"width": 720, "height": 960}
    assert info["photo"]["orientation"] == 6 and (info["width"], info["height"]) == (60, 80)
    rgb, o = photo_mod.decode_photo(data)
    assert o == 6 and rgb.shape == (960, 720, 3)
    # 窓（明るい所）は回した後の右上に来る
    assert rgb[:200, -200:].mean() > 230
    # 98・今風モードの読み方は前のまま（向きの情報を見ない）
    assert convert_still_bytes(data, "p.jpg", {"mode": "modern", "width": 60})["info"]["input"]["width"] == 960


def test_jpeg_is_read_and_heic_is_rejected(photo):
    out = convert_still_bytes(_jpeg(photo), "p.jpg", {"mode": "photo", "width": 64})
    assert out["info"]["photo"]["orientation"] == 1
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64
    with pytest.raises(StillError, match="JPEG か PNG にしてください"):
        convert_still_bytes(heic, "p.heic", {"mode": "photo"})
    res = json.loads(still_for_web(heic, "p.heic", "{}"))
    assert not res["ok"] and "HEIC" in res["error"] and "JPEG か PNG" in res["error"]
    bad = json.loads(still_for_web(b"not an image", "x.jpg", json.dumps({"mode": "photo"})))
    assert not bad["ok"] and "JPEG か PNG にしてください" in bad["error"]


def test_box_shrink_is_the_rounded_mean():
    a = np.zeros((4, 6, 3), np.uint8)
    a[:2, :3] = [10, 20, 30]
    a[:2, 3:] = [0, 0, 255]
    a[2:, :] = [1, 2, 3]
    a[3, 0] = [2, 2, 2]
    out = photo_mod.box_shrink(a, 2, 2)
    assert out[0, 0].tolist() == [10, 20, 30] and out[0, 1].tolist() == [0, 0, 255]
    # 6 画素のうち 1 つだけ (2,2,2): R は (5×1 + 2)/6 = 1.17 → 1、B は (5×3 + 2)/6 = 2.83 → 3
    assert out[1, 0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        photo_mod.box_shrink(a, 7, 2)


def test_prescale_only_for_large_photos():
    big = make_photo(900, 600)
    r = _photo(big, width=100)
    assert r.photo.prescaled and r.photo.work_size == (400, 268)   # 仕上がり 100×67 のちょうど 4 倍
    small = make_photo(300, 200)
    r2 = _photo(small, width=100)
    assert not r2.photo.prescaled and r2.photo.work_size == (300, 200)
    assert r2.size == (100, 67)


def test_kuwahara_keeps_edges_and_flattens_noise():
    rng = np.random.default_rng(1)
    a = np.zeros((40, 40, 3), np.int64)
    a[:, :20] = 60
    a[:, 20:] = 200
    a += rng.integers(-15, 16, a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    out = photo_mod.kuwahara(a, 3)
    # 境目は残る（左は暗く、右は明るいまま。間の色はできない）
    assert out[:, :20].max() < 100 and out[:, 20:].min() > 160
    # ざらつきは減る
    assert out[:, 3:17].std() < a[:, 3:17].std() / 2
    assert out[:, 23:37].std() < a[:, 23:37].std() / 2
    assert np.array_equal(photo_mod.kuwahara(a, 0), a)


def test_stronger_smooth_gives_flatter_result(photo):
    counts = []
    for level in SMOOTH_LEVELS:
        r = _photo(photo, width=120, smooth=level, tidy_dots=False)
        # となりのドットと色が違う所の数（少ないほど塊が大きい）
        idx = r.index
        counts.append(int((idx[:, 1:] != idx[:, :-1]).sum() + (idx[1:] != idx[:-1]).sum()))
    assert counts[0] > counts[1] > counts[2]


def test_photo_has_no_new_colors_and_palette_is_dark_first(photo):
    r = _photo(photo, width=120, outline=True)
    assert r.index.max() < len(r.palette) and len(r.palette) <= 32
    assert sorted(np.unique(r.index).tolist()) == list(range(len(r.palette)))
    assert r.lone is not None and r.outline is not None


def test_photo_width_errors(photo):
    with pytest.raises(StillError, match="写真の幅") as e:
        _photo(photo, width=2000)
    assert e.value.kind == "width" and e.value.candidates
    tall = make_photo(100, 1000)
    with pytest.raises(StillError, match="上限") as e2:
        _photo(tall, width=100, max_size=(1920, 900))
    assert e2.value.kind == "size"


# ---- 輪郭の線 ----

def test_outline_draws_one_dot_on_the_darker_side():
    pal = np.array([[0, 0, 0], [60, 60, 60], [230, 230, 230]], np.uint8)
    idx = np.full((5, 6), 2, np.uint8)
    idx[:, :3] = 1    # 左は暗い灰、右は明るい
    idx[0, 0] = 0
    o = outline_mod.draw_outline(idx, pal)
    assert o.drawn[:, 2].all() and o.drawn.sum() == 5   # 暗い方（左）の境目の 1 列だけ
    assert (o.index[:, 2] == 0).all() and (o.index[:, 3:] == 2).all()
    assert o.count == 5 and o.settings["min_diff"] == outline_mod.MIN_DIFF


def test_outline_skips_small_brightness_steps_and_darkest():
    pal = np.array([[0, 0, 0], [100, 100, 100], [110, 110, 110]], np.uint8)
    idx = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.uint8)
    assert outline_mod.draw_outline(idx, pal).count == 0
    idx2 = np.array([[0, 0, 2, 2]], np.uint8)
    pal2 = np.array([[0, 0, 0], [5, 5, 5], [250, 250, 250]], np.uint8)
    assert outline_mod.draw_outline(idx2, pal2).count == 0   # もともといちばん暗い色は数えない


def test_outline_in_json_overlay_and_other_modes(photo):
    out = convert_still_bytes(_png(photo), "p.png", {"mode": "photo", "width": 80, "outline": True,
                                                     "canvas": [90, 70]})
    info, res = out["info"], out["result"]
    assert info["outline"]["drawn"] == res.outline.count > 0 and info["settings"]["outline"] is True
    ov = res.outline_overlay()
    assert ov.shape == (70, 90, 4)
    assert int((ov[..., 3] > 0).sum()) == res.outline.count
    assert not ov[:5].any()   # 余白には線を引かない（キャンバスの位置に合わせて重ねる）
    # 98 モードでも使える（既定は切）
    m = convert_still_bytes(_png(make_modern_scene()[0]), "m.png", {"width": 64, "outline": True})["info"]
    assert m["settings"]["outline"] is True and "outline" in m


# ---- CLI とブラウザ版 ----

def test_cli_options():
    p = cli.build_parser()
    s = cli.settings_from_args(p.parse_args(["x.jpg", "--mode", "photo"]))
    assert s == StillSettings.from_mode("photo") or (s.width, s.smooth, s.tidy_dots) == (320, "medium", True)
    s = cli.settings_from_args(p.parse_args(["x.jpg", "--mode", "photo", "--smooth", "strong", "--outline",
                                             "--no-tidy-dots", "--width", "200"]))
    assert (s.smooth, s.outline, s.tidy_dots, s.width) == ("strong", True, False, 200)
    assert cli.settings_from_args(p.parse_args(["x.png", "--outline"])).outline is True
    with pytest.raises(SystemExit):
        p.parse_args(["x.jpg", "--smooth", "max"])


@pytest.mark.parametrize("extra, settings, kind", [
    (["--mode", "photo"], {"mode": "photo"}, "jpeg"),
    (["--mode", "photo", "--smooth", "weak", "--outline"], {"mode": "photo", "smooth": "weak", "outline": True}, "png"),
    (["--mode", "photo", "--smooth", "strong", "--no-tidy-dots", "--colors", "16"],
     {"mode": "photo", "smooth": "strong", "tidy_dots": False, "colors": 16}, "jpeg"),
    (["--mode", "photo", "--width", "100", "--canvas", "120x140", "--outline"],
     {"mode": "photo", "width": 100, "canvas": [120, 140], "outline": True}, "exif"),
    (["--outline", "--width", "64"], {"outline": True, "width": 64}, "png"),
])
def test_web_result_equals_cli_files(tmp_path, photo, extra, settings, kind):
    data = {"jpeg": lambda: _jpeg(photo), "png": lambda: _png(photo), "exif": lambda: _jpeg(photo, 8)}[kind]()
    src = tmp_path / ("p.png" if kind == "png" else "p.jpg")
    src.write_bytes(data)
    assert cli.main([str(src), "--out", str(tmp_path / "c"), "-q", *extra]) == 0
    res = json.loads(still_for_web(src.read_bytes(), src.name, json.dumps(settings)))
    assert res["ok"], res
    assert base64.b64decode(res["png_base64"]) == (tmp_path / "c.png").read_bytes()
    assert res["json_text"].encode("utf-8") == (tmp_path / "c.json").read_bytes()
    if settings.get("outline"):
        assert res["outline"]["drawn"] >= 0 and res["outline_overlay_base64"]
    else:
        assert "outline_overlay_base64" not in res
    if settings.get("mode") == "photo":
        assert res["photo"]["smooth"]["level"] == settings.get("smooth", "medium")


def test_cli_prints_photo_counts(tmp_path, photo, capsys):
    src = tmp_path / "p.jpg"
    src.write_bytes(_jpeg(photo))
    assert cli.main([str(src), "--out", str(tmp_path / "a"), "--mode", "photo", "--width", "80", "--outline"]) == 0
    out = capsys.readouterr().out
    assert "写真の下書き（まとめの強さ medium）" in out and "80×60 ドット" in out
    assert "孤立したドットを置き換えた数" in out and "輪郭の線を引いたドット" in out


def test_cli_heic_message(tmp_path, capsys):
    src = tmp_path / "p.heic"
    src.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
    assert cli.main([str(src), "--mode", "photo", "--out", str(tmp_path / "a")]) == 1
    assert "JPEG か PNG にしてください" in capsys.readouterr().err


# ---- 32 ビット（Pyodide） ----

@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_photo_runs_with_int32_numpy(photo):
    expected = _photo(photo, width=100, outline=True)
    with emulate_int32():
        got = _photo(photo, width=100, outline=True)
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.palette, expected.palette)
    assert got.outline.count == expected.outline.count
