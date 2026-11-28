"""Still App の「ディザを作る」（retrocore/still/dither.py、--dither 2x2・4x4）のテスト。

- 入れるのは、なめらかに色が変わる所の、近い 2 色の間だけ（輪郭・はっきりした境目・細かい描き込みには入れない）
- パレットはディザなしのときと同じ（新しい色は作らない）
- 網目は絵全体で共通の座標の Bayer の表（2×2 は 5 段、4×4 は 17 段）
- 孤立したドットをそろえても、網目のドットは消されない
- off のときは前とバイト単位で同じ。CLI とブラウザ版で同じバイト列
テスト画像はコードで合成する。
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from apps.still import cli
from int32_numpy import AVAILABLE, emulate_int32
from retrocore.still import dither as dither_mod
from retrocore.still.convert import (DITHER_CHOICES, ITEMS, MODES, StillSettings, convert_still, convert_still_bytes,
                                     spec, still_for_web)


def _png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def make_sky(w: int = 96, h: int = 64, seed: int = 3) -> np.ndarray:
    """上半分は空（左から右へ、なめらかに青が明るくなる）、左下は暗い山（はっきりした境目）、右下は細かい描き込み（ざらざらの葉）。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    x = np.linspace(0.0, 1.0, w)[None, :]
    img[..., 0] = np.rint(70 + 90 * x).astype(np.uint8)
    img[..., 1] = np.rint(110 + 90 * x).astype(np.uint8)
    img[..., 2] = np.rint(150 + 90 * x).astype(np.uint8)
    img[h // 2:, : w // 2] = (40, 50, 45)                      # 山（平らな面）
    leaves = rng.integers(0, 3, size=(h - h // 2, w - w // 2))
    img[h // 2:, w // 2:] = np.array([(30, 90, 40), (60, 140, 60), (110, 70, 50)], np.uint8)[leaves]
    img[h // 2 - 1, :] = (10, 10, 10)                          # 地平線の暗い線（輪郭）
    return img


@pytest.fixture(scope="module")
def sky():
    return make_sky()


def _run(rgb, **kw):
    return convert_still(rgb, StillSettings.from_mode("modern", width=rgb.shape[1], merge_similar=False,
                                                      colors=kw.pop("colors", 8), **kw))


# ---- 設定 ----

def test_choices_and_default_off():
    assert DITHER_CHOICES == ("off", "2x2", "4x4")
    assert "dither" in ITEMS and ITEMS.index("merge_similar") < ITEMS.index("dither") < ITEMS.index("tidy_dots")
    for mode in MODES:
        assert MODES[mode]["dither"] == "off"
        assert StillSettings.from_mode(mode).dither == "off"
    assert [c["value"] for c in spec()["dither_choices"]] == list(DITHER_CHOICES)
    with pytest.raises(ValueError, match="dither"):
        StillSettings.from_dict({"dither": "8x8"})
    with pytest.raises(ValueError, match="dither"):
        StillSettings.from_dict({"dither": 2})


def test_photo_does_not_merge_similar_by_default():
    assert MODES["photo"]["merge_similar"] is False
    assert MODES["modern"]["merge_similar"] is True and MODES["pc98"]["merge_similar"] is False


# ---- 網目を入れる所 ----

def test_palette_is_the_same_as_without_dither(sky):
    off = _run(sky)
    for kind in ("2x2", "4x4"):
        on = _run(sky, dither=kind)
        assert np.array_equal(on.palette, off.palette), kind      # 新しい色は作らない・選ばれる色も同じ
        assert on.dither.count > 0
        assert on.dither.changed.sum() == np.count_nonzero(on.index != off.index)


def test_only_smooth_parts_get_dither(sky):
    off = _run(sky)
    on = _run(sky, dither="2x2")
    h, w = sky.shape[:2]
    changed = on.index != off.index
    # 山・葉・地平線の線（とその近く）は変えない
    assert not changed[h // 2 - 3:].any()
    # 空の、2 色の境目のあたりだけが変わる（空のほとんどは変わらない）
    assert changed[: h // 2 - 3].any()
    assert on.dither.area[: h // 2 - 3].sum() < (h // 2 - 3) * w
    # 網目に使う色は、ディザなしで空に使われていた色だけ（ほかの所の色を持ち込まない）
    sky_colors = set(np.unique(off.index[: h // 2 - 1]).tolist())
    assert set(np.unique(on.index[on.dither.changed]).tolist()) <= sky_colors


def test_pattern_follows_global_bayer(sky):
    on = _run(sky, dither="2x2")
    d = on.dither
    pal = on.palette
    # 網目の範囲で、明るい方の色のドットは、Bayer の表の値が段より小さい所
    ys, xs = np.nonzero(d.area)
    assert len(ys) > 0
    bay = dither_mod.BAYER2[ys % 2, xs % 2]
    lum = pal.astype(int).sum(1)
    # 2×2 の 1 ブロックの中で、表の値の小さい所ほど明るい（明るい色は表の値の小さい所から置く）
    for y, x in zip(ys[:200], xs[:200]):
        y0, x0 = y - y % 2, x - x % 2
        blk = on.index[y0:y0 + 2, x0:x0 + 2]
        if blk.shape != (2, 2) or not d.area[y0:y0 + 2, x0:x0 + 2].all() or len(np.unique(blk)) != 2:
            continue
        br = lum[blk]
        order = dither_mod.BAYER2.ravel().argsort()
        vals = br.ravel()[order]
        assert all(vals[i] >= vals[i + 1] for i in range(3))
    assert set(bay.tolist()) <= {0, 1, 2, 3}


def test_4x4_has_more_levels_than_2x2(sky):
    def levels(res, n):
        # n×n のブロックごとの、明るい方の色の数（網目の範囲が全部のブロックだけ）
        d, idx, lum = res.dither, res.index, res.palette.astype(int).sum(1)
        found = set()
        h, w = idx.shape
        for y0 in range(0, h - n + 1, n):
            for x0 in range(0, w - n + 1, n):
                blk = idx[y0:y0 + n, x0:x0 + n]
                if d.area[y0:y0 + n, x0:x0 + n].all() and len(np.unique(blk)) == 2:
                    found.add(int((lum[blk] == lum[blk].max()).sum()))
        return found
    l2 = levels(_run(make_sky(192, 64), dither="2x2"), 2)
    l4 = levels(_run(make_sky(192, 64), dither="4x4"), 4)
    assert l2 <= {1, 2, 3} and len(l4) > len(l2)


def test_far_colors_and_darkest_are_not_mixed():
    # 黒から白へ急に変わる（2 色の距離が大きい）なめらかな面: 混ぜない
    w = 64
    img = np.zeros((32, w, 3), np.uint8)
    img[:, w // 2:] = 255
    res = _run(img, colors=2, dither="2x2")
    assert res.dither.count == 0
    # なめらかな暗い面（いちばん暗い色とその次の色の間）: いちばん暗い色は混ぜない
    g = np.rint(np.linspace(0, 60, w)).astype(np.uint8)
    img = np.repeat(np.repeat(g[None, :, None], 32, 0), 3, 2)
    res = _run(img, colors=3, dither="2x2")
    keep = 0
    assert not (res.dither.changed & ((res.index == keep) | (res.reduced.index == keep))).any()


def test_no_smooth_part_means_no_change():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(40, 40, 3)).astype(np.uint8)
    off = _run(img)
    on = _run(img, dither="4x4")
    assert on.dither.count == 0 and np.array_equal(on.index, off.index)


def test_lone_dots_do_not_erase_dither(sky):
    on = _run(sky, dither="2x2")
    both = _run(sky, dither="2x2", tidy_dots=True)
    assert not (both.lone.changed & on.dither.area).any()
    assert np.array_equal(both.index[on.dither.area], on.index[on.dither.area])


def test_pattern_tidy_skips_dither_area(sky):
    both = _run(sky, dither="2x2", tidy_pattern=True)
    assert not (both.tidied.label[both.dither.area] >= 0).any()


# ---- ファイル ----

def test_off_is_the_same_bytes(sky):
    data = _png(sky)
    base = convert_still_bytes(data, "s.png", {"mode": "modern", "width": 96})
    off = convert_still_bytes(data, "s.png", {"mode": "modern", "width": 96, "dither": "off"})
    assert off["png"] == base["png"] and off["json"] == base["json"]
    assert "dither" not in base["info"] and "dither" not in base["info"]["settings"]


def test_json_records_dither(sky):
    out = convert_still_bytes(_png(sky), "s.png", {"mode": "modern", "width": 96, "dither": "4x4"})
    info = out["info"]
    d = info["dither"]
    assert d["kind"] == "4x4" and d["changed"] > 0 and d["area"] >= d["changed"] and d["pairs"] >= 1
    assert d["settings"]["tile"] == "bayer4x4" and d["settings"]["levels"] == 17
    assert d["settings"]["edge_max"] == dither_mod.EDGE_MAX and d["settings"]["pair_max"] == dither_mod.PAIR_MAX
    assert info["settings"]["dither"] == "4x4"
    assert info["preset"]["changed"]["dither"] == {"preset": "off", "used": "4x4"}
    assert list(info).index("dither") < list(info).index("settings")


def test_cli_and_web_are_the_same(tmp_path, sky):
    data = _png(sky)
    src = tmp_path / "s.png"
    src.write_bytes(data)
    assert cli.main([str(src), "--mode", "photo", "--width", "48", "--colors", "16", "--4bit", "--dither", "2x2",
                     "--out", str(tmp_path / "c"), "-q"]) == 0
    values = {"mode": "photo", "width": 48, "colors": 16, "four_bit": True, "dither": "2x2"}
    web = json.loads(still_for_web(data, "s.png", json.dumps(values)))
    assert web["ok"], web
    assert base64.b64decode(web["png_base64"]) == (tmp_path / "c.png").read_bytes()
    assert web["json_text"] == (tmp_path / "c.json").read_text(encoding="utf-8")
    assert web["dither"]["kind"] == "2x2"
    ov = np.asarray(Image.open(io.BytesIO(base64.b64decode(web["dither_overlay_base64"]))))
    assert ov.shape == (web["height"], web["width"], 4)
    assert (ov[..., 3] == 255).sum() == web["dither"]["changed"]


def test_overlay_marks_changed_and_area():
    changed = np.zeros((4, 4), bool)
    area = np.zeros((4, 4), bool)
    changed[1, 1] = True
    area[1:3, 1:3] = True
    ov = dither_mod.overlay_rgba(changed, area)
    assert ov[1, 1, 3] == 255 and ov[2, 2, 3] == dither_mod.OVERLAY_AREA_ALPHA and ov[0, 0, 3] == 0


@pytest.mark.skipif(not AVAILABLE, reason="intp と int64 を区別できない環境（Windows など）")
def test_dither_runs_with_int32_numpy(sky):
    s = StillSettings.from_mode("photo", width=48, colors=16, dither="4x4")
    expected = convert_still(sky, s)
    with emulate_int32():
        got = convert_still(sky, s)
    assert np.array_equal(got.index, expected.index) and np.array_equal(got.palette, expected.palette)
