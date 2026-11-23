"""動きの表の範囲「帯」（band）と「四角」（rect）（retrocore/sprite/motion/region.py・spec.py・timeline.py）のテスト。

設計は docs/sprite-app-plan.md の「動きの表（タイムライン）の設計」の「帯・四角の範囲」。確かめること:

1. 今あるプリセット（呼吸・横揺れ・浮遊・弾む・うなずく・伸び）と、範囲を書いていない表の出力が、範囲を足す前とまったく同じ
   （`tests/motion_records.py` の記録。フレームとシートの PNG は画素で比べる。画面の呼び方も）
2. 帯・四角の表の読み込み・書き間違いのエラー・書き出し（数と割合）
3. 範囲の外は動かない（剣を持った合成のキャラで、分け目 1 本だと剣が動き、頭の四角だと剣が動かない）。決まりも守る
4. 境目で作れないとき（空いた所の色が分からない・外の違う色を隠す・足元を含む）は作らずに警告し、近くで作れる範囲を添える
   （添えた範囲で作り直すと作れる）
5. 書き出した表とシートの JSON に、実際に使った範囲（"used"・"regions_used"）が付く
6. 範囲を使った表で、CLI と画面（`motion_for_web()`）の出力が同じ。画面で読み込んだ表の範囲が消えない
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apps.sprite import cli
from motion_records import digest_all, make_knight, records
from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import encode_png, motion_for_web, table_for_web
from retrocore.sprite.image import load_rgba
from retrocore.sprite.motion import (SpecError, check_rules, load_spec, make_motion, make_motion_from_spec, parse_spec,
                                     sheet_info, sheet_json, spec_json, with_height_style)
from retrocore.sprite.motion.region import percent_to_width, width_to_percent
from retrocore.sprite.motion.spec import spec_to_dict
from synth_sprite import make_pixel_sprite

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "apps" / "sprite" / "examples"
LOCAL = ROOT / "local_samples"

# ---- 1. 範囲を足す前と同じ ----

# 範囲（帯・四角）を足す前（PR #32 の main）の `motion_records` の記録。絵の名前 → ハッシュ
BEFORE = {"mushroom": "3e0b410ec8345465", "slime": "eb61e4c04cb9e4a0", "pixel": "120541b5a420a6b4",
          "hero": "e7c0f5b5ea476061", "knight": "134044450d296e7f"}
# 手持ちの ai03〜ai05（local_samples/ にあるときだけ確かめる。変換の既定の結果）
AI_BEFORE = {"ai03": "d415f367e9267292", "ai04": "e662e8e8b15e0bb4", "ai05": "468b7b6eaa145f3b"}


def test_presets_and_split_tables_are_unchanged():
    """プリセット 6 つ・範囲を書いていない表・画面の呼び方の出力が、範囲を足す前と同じ（合成の絵 5 枚）。"""
    assert digest_all() == BEFORE


@pytest.mark.parametrize("name", sorted(AI_BEFORE))
def test_presets_and_split_tables_are_unchanged_on_ai_images(name):
    path = LOCAL / f"{name}.png"
    if not path.exists():
        pytest.skip(f"{path} がありません（手持ちの AI 画像で確かめるときだけ。preview/ai-check の ai-input/ にある）")
    from motion_records import _dig
    sp = convert_sprite(load_rgba(path), SpriteSettings(keep_large=0)).image
    assert _dig(json.dumps(records(sp), sort_keys=True).encode()) == AI_BEFORE[name]


# ---- 表の材料 ----

def _table(regions, tracks, frames, name="t") -> dict:
    return {"format": "sprite-motion", "version": 1, "name": name, "loop": True, "regions": regions,
            "tracks": tracks, "frames": frames}


HEAD = {"type": "rect", "left": 5, "right": 24, "bottom": 10, "top": 22}   # make_knight() の頭（剣は左の 3 列）
HEAD_Y = {"head_y": {"type": "deform", "region": "head", "axis": "y"}}
HEAD_X = {"head_x": {"type": "move", "region": "head", "axis": "x"}}
NOD = [{"ms": 600}, {"ms": 200, "head_y": -1}]


@pytest.fixture(scope="module")
def knight():
    return make_knight()


SWORD = (slice(None), slice(0, 3))    # make_knight() の剣の列（左の 3 列）


# ---- 2. 読み込み・書き間違い・書き出し ----

def test_parse_band_and_rect_and_write_back():
    t = _table({"head": {"type": "rect", "left": 5, "right": "100%", "bottom": "45.5%", "top": 22, "range": 1,
                         "keep_small_parts": True},
                "chest": {"type": "band", "bottom": 4, "top": 9}},
               {**HEAD_Y, "chest_y": {"type": "deform", "region": "chest", "axis": "y"},
                "head_v": {"type": "move", "region": "head", "axis": "y"}},
               [{"ms": 100}, {"ms": 100, "head_y": -1}, {"ms": 100, "chest_y": 1}, {"ms": 100, "head_v": 1}])
    spec = parse_spec(t)
    assert spec.regions["head"].type == "rect" and spec.regions["head"].edges == {
        "left": 5, "right": "100%", "bottom": "45.5%", "top": 22}
    d = spec_to_dict(spec)
    # 書き出す順は type・left・right・bottom・top・range・keep_small_parts（at と lines は無い）
    assert list(d["regions"]["head"]) == ["type", "left", "right", "bottom", "top", "range", "keep_small_parts"]
    assert d["regions"]["chest"] == {"type": "band", "bottom": 4, "top": 9}
    assert parse_spec(spec_json(d)) is not None and spec_to_dict(parse_spec(spec_json(d))) == d


@pytest.mark.parametrize("region, words", [
    ({"type": "rect", "left": 1, "right": 5, "bottom": 3}, "top を書いてください"),
    ({"type": "band", "bottom": 3, "top": 9, "left": 1}, "知らない項目です: left（左右を決めるときは \"type\": \"rect\""),
    ({"type": "rect", "left": 1, "right": 5, "bottom": 3, "top": 9, "at": 4}, "at・lines・aim は分け目"),
    ({"type": "band", "bottom": 9, "top": 3}, "下が上より上にあります"),
    ({"type": "rect", "left": 6, "right": 5, "bottom": 3, "top": 9}, "左が右より右にあります"),
    ({"type": "band", "bottom": 0, "top": 9}, "bottom は下から何ドット目"),
    ({"type": "rect", "left": "120%", "right": 5, "bottom": 3, "top": 9}, "left は左から何ドット目"),
    ({"type": "band", "bottom": 3, "top": 9, "range": 20}, "range は 0〜8"),
    ({"type": "band", "bottom": 3, "top": 9, "keep_small_parts": 1}, "keep_small_parts は true か false"),
    ({"type": "circle", "bottom": 3, "top": 9}, "type は \"split\""),
])
def test_box_mistakes_are_errors(region, words):
    t = _table({"head": region}, HEAD_Y, NOD)
    with pytest.raises(SpecError, match=words.replace("(", r"\(").replace(")", r"\)")):
        parse_spec(t)


def test_vertical_move_only_on_band_and_rect():
    """縦の位置（move の y）は帯・四角で使える。分け目では今までどおりエラー。同じ行で横と縦の位置を一緒には使えない。"""
    up = {"head_v": {"type": "move", "region": "head", "axis": "y"}}
    parse_spec(_table({"head": HEAD}, up, [{"ms": 1}, {"ms": 1, "head_v": -1}]))
    with pytest.raises(SpecError, match="縦の位置（move の axis \"y\"）は使えません"):
        parse_spec(_table({"head": {"type": "split", "at": "auto"}}, up, [{"ms": 1}, {"ms": 1, "head_v": -1}]))
    with pytest.raises(SpecError, match="横の位置と縦の位置を同じ行で一緒に使うのは、まだ作れません"):
        parse_spec(_table({"head": HEAD}, {**up, **HEAD_X}, [{"ms": 1}, {"ms": 1, "head_v": -1, "head_x": 1}]))


def test_width_percent_roundtrip(knight):
    w = knight.shape[1]
    for n in range(1, w + 1):
        p = width_to_percent(knight, n)
        assert p is not None and percent_to_width(knight, p) == n
    assert percent_to_width(knight, "0%") == 1 and percent_to_width(knight, "100%") == w


def test_percent_and_number_give_the_same_frames(knight):
    """割合で書いた端は、数で書いた端と同じフレームになる。--spec-height percent で書き出して読み直しても同じ。"""
    num = make_motion_from_spec(knight, _table({"head": HEAD}, HEAD_Y, NOD))
    pct = with_height_style(num.spec, knight, "percent")
    assert all(isinstance(pct["regions"]["head"][e], str) for e in ("left", "right", "bottom", "top"))
    again = make_motion_from_spec(knight, pct)
    assert not again.skipped and all(np.array_equal(a, b) for a, b in zip(num.frames, again.frames))
    back = with_height_style(pct, knight, "number")
    assert {e: back["regions"]["head"][e] for e in HEAD if e != "type"} == {e: v for e, v in HEAD.items() if e != "type"}


# ---- 3. 範囲の外は動かない ----

def _outside_unchanged(base, res, box_rows_cols, extra_rows=0):
    """範囲の外（`extra_rows` は範囲の上の端より上に出てよい行の数）のドットが、どのフレームでも元の絵と同じか。"""
    top, _ = res.info.get("padding_y", [0, 0])
    pad = res.info.get("padding", 0)
    h, w = base.shape[:2]
    r0, r1, c0, c1 = box_rows_cols
    inside = np.zeros((h, w), bool)
    inside[max(0, r0 - extra_rows):r1 + 1, c0:c1 + 1] = True
    for f in res.frames:
        core = f[top:top + h, pad:pad + w]
        if not np.array_equal(core[~inside], base[~inside]):
            return False
    return True


def _box(base, r):
    h = base.shape[0]
    return h - r["top"], h - r["bottom"], r.get("left", 1) - 1, r.get("right", base.shape[1]) - 1


def test_split_moves_the_sword_but_rect_does_not(knight):
    """分け目 1 本のうなずくでは剣の上も一緒に下がる。頭の四角で同じ表を作ると、剣（四角の外）は 1 ドットも動かない。"""
    split = make_motion(knight, "nod")
    assert not split.skipped
    assert not np.array_equal(split.frames[1][SWORD], knight[SWORD])          # 分け目 1 本: 剣が動く
    for frames in (NOD, [{"ms": 1}, {"ms": 1, "head_y": 1}], [{"ms": 1}, {"ms": 1, "head_y": -2}]):
        res = make_motion_from_spec(knight, _table({"head": HEAD}, HEAD_Y, frames))
        assert not res.skipped, res.warnings
        top = res.info.get("padding_y", [0, 0])[0]
        for f in res.frames:
            assert np.array_equal(f[top:, :3], knight[:, :3])                # 剣は動かない
        up = max(0, max(v.get("head_y", 0) for v in frames))
        assert _outside_unchanged(knight, res, _box(knight, HEAD), extra_rows=up)
        assert not check_rules(np.pad(knight, ((top, 0), (0, 0), (0, 0))), res.frames)
        assert any(not np.array_equal(f, res.frames[0]) for f in res.frames[1:])   # 頭は動いている


@pytest.mark.parametrize("region, tracks, frames", [
    ({"type": "band", "bottom": 15, "top": 22}, {"v": {"type": "move", "region": "head", "axis": "y"}},
     [{"ms": 1}, {"ms": 1, "v": -1}, {"ms": 1, "v": 1}]),
    ({"type": "band", "bottom": 15, "top": 22}, {"head_y": {"type": "deform", "region": "head", "axis": "y"}},
     [{"ms": 1}, {"ms": 1, "head_y": -1}, {"ms": 1, "head_y": 2}]),
])
def test_band_keeps_outside(knight, region, tracks, frames):
    res = make_motion_from_spec(knight, _table({"head": region}, tracks, frames))
    assert not res.skipped, res.warnings
    up = 2 if "head_y" in tracks else 1
    assert _outside_unchanged(knight, res, _box(knight, region), extra_rows=up)
    top = res.info.get("padding_y", [0, 0])[0]
    assert not check_rules(np.pad(knight, ((top, 0), (0, 0), (0, 0))), res.frames)


def test_rect_move_x_keeps_outside_and_pads():
    """四角の中身を横にずらす: 外は動かず、はみ出す分は左右に同じ数の余白を足す（足元はフレームの真ん中のまま）。"""
    sp = make_pixel_sprite()
    region = {"type": "rect", "left": 1, "right": 20, "bottom": 13, "top": 22}
    res = make_motion_from_spec(sp, _table({"head": region}, HEAD_X,
                                           [{"ms": 1}, {"ms": 1, "head_x": 1}, {"ms": 1}, {"ms": 1, "head_x": -1}]))
    assert not res.skipped, res.warnings
    assert res.info["padding"] == 1 and res.size == (sp.shape[1] + 2, sp.shape[0])
    assert _outside_unchanged(sp, res, _box(sp, region))
    padded = np.pad(sp, ((0, 0), (1, 1), (0, 0)))
    assert not check_rules(padded, res.frames)
    assert not np.array_equal(res.frames[1], res.frames[0]) and not np.array_equal(res.frames[3], res.frames[0])


def test_two_regions_rect_and_root(knight):
    """頭の四角の縦の変形と、体全体のずれ（root）を同じ行で一緒に使える（範囲 2 つの例）。"""
    t = _table({"head": HEAD}, {**HEAD_Y, "root_y": {"type": "root", "axis": "y"}},
               [{"ms": 300, "head_y": -1}, {"ms": 150, "root_y": -1}, {"ms": 150, "root_y": -2, "head_y": 1},
                {"ms": 150, "root_y": -1}])
    res = make_motion_from_spec(knight, t)
    assert not res.skipped, res.warnings
    assert res.info["offset_y"] == [0, -1, -2, -1]
    info = sheet_info(res, "x.png")
    assert info["padding"]["top"] == res.info["padding_y"][0] and info["regions_used"]["head"]["bottom"] == 10


# ---- 4. 作れないときの警告と、近くで作れる範囲 ----

def _first_problem(res):
    assert res.skipped and res.info["problems"]
    return res.info["problems"][0]


def test_hidden_area_warns_and_suggests_working_boxes(knight):
    """縦に上にずらすと、四角の下の端が空く。外の同じ色の続きで埋められなければ作らずに警告し、作れる範囲を添える。"""
    t = _table({"head": HEAD}, {"v": {"type": "move", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "v": -1}])
    res = make_motion_from_spec(knight, t)
    q = _first_problem(res)
    assert q["rule"] == "hidden" and q["region"] == "head" and q["pixels"]
    assert "動いた後ろから見えてくる所の色が分かりません" in q["message"] and "近くで作れる範囲: " in q["message"]
    assert q["near"] and all(set(n) == {"left", "right", "bottom", "top"} for n in q["near"])
    for n in q["near"]:     # 添えた範囲は、実際に作れる
        again = make_motion_from_spec(knight, _table({"head": {"type": "rect", **n}}, t["tracks"], t["frames"]))
        assert not again.skipped, again.warnings
    # 場所（pixels）は元の絵の上の位置（四角の下の端の行）
    assert all(knight.shape[0] - y == HEAD["bottom"] for _, y in q["pixels"])


def test_overlap_warns():
    """伸ばすと四角の上の端の行が外に出る。外の違う色のドットを隠すなら作らない。"""
    sp = make_pixel_sprite()
    t = _table({"head": {"type": "band", "bottom": 11, "top": 13}}, HEAD_Y, [{"ms": 1}, {"ms": 1, "head_y": 1}])
    q = _first_problem(make_motion_from_spec(sp, t))
    assert q["rule"] == "overlap" and "外の違う色のドットを隠します" in q["message"]


def test_foot_is_not_allowed(knight):
    t = _table({"head": {"type": "band", "bottom": 1, "top": 12}}, HEAD_Y, NOD)
    q = _first_problem(make_motion_from_spec(knight, t))
    assert q["rule"] == "split_not_found" and "足元の中です" in q["message"] and "近くで作れる範囲: 下 4〜上 12" in q["message"]


def test_edges_outside_image_are_errors(knight):
    for bad, words in (({"top": 99}, "top（上）99 は画像の範囲の外です"), ({"right": 99}, "right（右）99 は画像の範囲の外です"),
                       ({"bottom": "90%", "top": "30%"}, "下（20）が上（7）より上")):
        with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
            make_motion_from_spec(knight, _table({"head": {**HEAD, **bad}}, HEAD_Y, NOD))


def test_unused_region_is_not_checked(knight):
    """使っていない（値が全部 0 の）範囲は、画像の外でもエラーにしない（分け目と同じ）。"""
    t = _table({"head": {**HEAD, "top": 99}}, HEAD_Y, [{"ms": 1}, {"ms": 1}])
    assert not make_motion_from_spec(knight, t).skipped


# ---- 5. 実際に使った範囲 ----

def test_used_and_regions_used(knight):
    t = _table({"head": {**HEAD, "bottom": "45.5%"}}, HEAD_Y, NOD)
    res = make_motion_from_spec(knight, t)
    assert not res.skipped
    used = res.spec["regions"]["head"]["used"]
    assert {k: used[k] for k in ("left", "right", "bottom", "top")} == {"left": 5, "right": 24, "bottom": 10, "top": 22}
    assert len(used["lines"]["head_y"]) == 1 and len(used["lines"]["head_y"][0]) == 20    # 四角の幅の数の継ぎ目
    assert json.loads(sheet_json(res, "x.png"))["regions_used"]["head"] == used
    # "used" つきの表を読み直しても同じ（used は読むときに使わない）
    again = make_motion_from_spec(knight, res.spec)
    assert all(np.array_equal(a, b) for a, b in zip(res.frames, again.frames))


# ---- 見本の表 ----

@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.motion.json")), ids=lambda p: p.name)
def test_examples_are_valid(path, knight):
    spec = load_spec(path)
    res = make_motion_from_spec(make_pixel_sprite(), spec)
    assert res.spec is not None
    if any(r.edges is not None for r in spec.regions.values()):
        assert not res.skipped, res.warnings


# ---- 6. CLI と画面 ----

def _write_png(path: Path, rgba: np.ndarray) -> Path:
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def _png_pixels(data: bytes) -> np.ndarray:
    import io
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


@pytest.mark.parametrize("table", [
    _table({"head": HEAD}, HEAD_Y, NOD, "nod_rect"),
    _table({"head": HEAD}, {**HEAD_Y, "root_y": {"type": "root", "axis": "y"}},
           [{"ms": 300, "head_y": -1}, {"ms": 150, "root_y": -1}, {"ms": 150, "root_y": -2, "head_y": 1}], "bounce_rect"),
    _table({"head": {"type": "band", "bottom": "70%", "top": "100%"}}, {"v": {"type": "move", "region": "head", "axis": "y"}},
           [{"ms": 1}, {"ms": 1, "v": 1}], "band_y"),
], ids=["nod_rect", "bounce_rect", "band_y"])
def test_cli_and_web_give_the_same_sheet(tmp_path, knight, table):
    """範囲を使った表で、CLI（--motion-spec --converted 相当: 変換済みの絵を --no-background などで通す代わりに、
    変換の結果を画面と同じ PNG にして渡す）と画面（motion_for_web の表の経路）のシートが同じ。"""
    spec_path = tmp_path / "t.motion.json"
    spec_path.write_text(spec_json(table), encoding="utf-8")
    src = _write_png(tmp_path / "in.png", np.kron(knight, np.ones((20, 20, 1), np.uint8)))
    out = tmp_path / "out" / "s.png"
    code = cli.main([str(src), "-o", str(out), "--motion-spec", str(spec_path), "--write-motion-spec",
                     str(tmp_path / "w.motion.json"), "--quiet"])
    assert code == 0
    sprite = load_rgba(out)
    web = json.loads(motion_for_web(encode_png(sprite), json.dumps({"spec": table, "image_name": f"s_{table['name']}_sheet.png"})))
    assert web["ok"] and not web["skipped"], web
    import base64
    cli_png = np.asarray(Image.open(out.parent / f"s_{table['name']}_sheet.png").convert("RGBA"))
    assert np.array_equal(cli_png, _png_pixels(base64.b64decode(web["sheet_png_base64"])))
    cli_json = (out.parent / f"s_{table['name']}_sheet.json").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert cli_json == web["sheet_json"]
    # 画面で保存する表（数）は、CLI の --write-motion-spec と同じ
    assert (tmp_path / "w.motion.json").read_text(encoding="utf-8").replace("\r\n", "\n") == web["spec_files"]["number"]
    assert "regions_used" in json.loads(cli_json)


def test_web_reads_and_keeps_box_regions():
    """画面の「表を読む」: 帯・四角の入った表を読んでも消えず、そのまま返す（画面はこの形で保存する）。"""
    text = spec_json(_table({"head": {**HEAD, "range": 1, "keep_small_parts": True},
                             "chest": {"type": "band", "bottom": "20%", "top": "40.0%"}},
                            {**HEAD_Y, "c": {"type": "deform", "region": "chest", "axis": "y"}},
                            [{"ms": 1}, {"ms": 1, "head_y": -1}, {"ms": 1, "c": -1}]))
    res = json.loads(table_for_web(json.dumps({"text": text})))
    assert res["ok"]
    assert res["spec"]["regions"]["head"] == {**HEAD, "range": 1, "keep_small_parts": True}
    assert res["spec"]["regions"]["chest"] == {"type": "band", "bottom": "20%", "top": "40.0%"}
    assert spec_json(res["spec"]) == text


def test_web_problem_has_box_suggestions(knight):
    t = _table({"head": HEAD}, {"v": {"type": "move", "region": "head", "axis": "y"}}, [{"ms": 1}, {"ms": 1, "v": -1}])
    res = json.loads(motion_for_web(encode_png(knight), json.dumps({"spec": t})))
    assert res["ok"] and res["skipped"]
    q = res["problems"][0]
    assert q["row"] == 2 and q["track"] == "v" and isinstance(q["near"][0], dict)
