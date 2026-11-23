"""回帰テスト（ゴールデン）のケース定義。

`tests/test_golden.py`（比較）と `tools/make_golden.py`（再生成）の両方から使う。
各ケースは「ファイル名 → 内容 (bytes)」の辞書を返し、`tests/golden/` 以下の同名ファイルと
バイト単位（PNG はピクセル単位）で比較する。

入力は `tests/fixtures/` のコード合成画像だけ（`make_fixture_images()` で生成。著作権の問題がない）。
他の人が作った画像（お手本・写真など）は使わない。

乱数: 変換で乱数を使うのは k-means の初期化（`palette._kmeans`、`np.random.default_rng(seed=0)`）
だけで、シードは既定で固定されている。合成画像の生成も `default_rng` に固定シードを渡す。
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from retrocore.analyzer import analyze_image, build_profile
from retrocore.optimize import fit_to_profile
from retrocore.palette import extract_palette, save_palette
from retrocore.pipeline import ConvertSettings, convert, load_image
from retrocore.style import STYLE_PRESETS, apply_overrides, profile_to_settings

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"
INDEX = GOLDEN / "index.json"   # ケース名 → 出力ファイル（tests/golden/ からの相対パス）

# 合成フィクスチャ（tests/fixtures/ にコミット済み。make_fixture_images() で再生成できる）
SYNTH_PHOTO = "input/synth_photo.png"      # RGB 写真風（グラデーション・肌色の球・ノイズ）
SYNTH_ALPHA = "input/synth_alpha.png"      # RGBA（透明背景）
SYNTH_REF = "reference/synth/synth_cg.png"         # 16 色・16 段階のお手本（条件を満たす）
SYNTH_REF_BAD = "reference/synth/synth_cg_bad.png"  # 条件外（16 段階でない色を含む）

# スタイルプリセット名（日本語）→ ファイル名に使う短い名前
PRESET_SLUGS = {
    "1983 デジタル8色・タイルディザ": "1983_digital8_tile",
    "1987 アナログ16色・ベタ塗り": "1987_analog16_flat",
    "1990 アナログ16色・市松グラデーション": "1990_analog16_checker",
    "1993 アナログ16色・タイルディザ": "1993_analog16_tile",
    "1995 アナログ16色・規則ディザ多用": "1995_analog16_ordered",
}


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


def make_fixture_images() -> dict[str, Image.Image]:
    """合成フィクスチャ画像（相対パス → 画像）。"""
    from synth_cg import rgb

    rng = np.random.default_rng(98)
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # 空: 上から下へ青 → 水色
    t = yy / (h - 1)
    img = np.stack([60 + 120 * t, 110 + 110 * t, 200 + 40 * t], -1)
    # 緑の丘
    hill = yy > 170 + 20 * np.sin(xx / 37.0)
    img[hill] = np.stack([40 + 0.3 * yy, 120 + 0.2 * xx, 50 + 0 * xx], -1)[hill]
    # 肌色の球（左上から光）
    cy, cx, r = 110.0, 130.0, 70.0
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ball = d < r
    nz = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    light = np.clip(0.25 + 0.75 * (nz * 0.7 + ((cx - xx) + (cy - yy)) / (2.8 * r)), 0, 1)
    skin = np.array([250.0, 205.0, 175.0])
    img[ball] = (skin[None, :] * light[ball, None]) + np.array([30.0, 10.0, 10.0]) * (1 - light[ball, None])
    # 赤い四角と黒い枠
    img[40:90, 230:290] = (200, 40, 50)
    img[38:40, 228:292] = img[90:92, 228:292] = 10
    img[38:92, 228:230] = img[38:92, 290:292] = 10
    img += rng.normal(0, 4, img.shape)
    photo = Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8), "RGB")

    # RGBA: 透明背景に陰影つきの球と半透明の帯
    ah, aw = 160, 200
    yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float64)
    rgba = np.zeros((ah, aw, 4), np.float64)
    d = np.sqrt((yy - 80) ** 2 + (xx - 90) ** 2)
    ball = d < 60
    shade = np.clip(1.1 - d / 70 - (xx - 90) / 200, 0.1, 1)
    rgba[ball, :3] = np.array([90.0, 160.0, 230.0]) * shade[ball, None]
    rgba[ball, 3] = 255
    rgba[120:140, :, :3] = (240, 200, 60)
    rgba[120:140, :, 3] = np.linspace(0, 255, aw)[None, :]
    alpha = Image.fromarray(np.clip(np.round(rgba), 0, 255).astype(np.uint8), "RGBA")

    good = Image.fromarray(rgb(), "RGB")
    bad_arr = rgb().copy()
    bad_arr[:5, :5] = (100, 101, 102)
    bad = Image.fromarray(bad_arr, "RGB")
    return {SYNTH_PHOTO: photo, SYNTH_ALPHA: alpha, SYNTH_REF: good, SYNTH_REF_BAD: bad}


def load_index() -> dict[str, list[str]]:
    return json.loads(INDEX.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(rel: str) -> Path:
    return FIXTURES / rel


def _img(path: Path) -> Image.Image:
    return load_image(str(path))


# ---------------------------------------------------------------------------
# 出力の符号化
# ---------------------------------------------------------------------------


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def result_files(name: str, res, settings: ConvertSettings) -> dict[str, bytes]:
    """変換結果 → `<name>.png`（インデックス画像）と `<name>.json`（パレット・情報）。"""
    info = {k: v for k, v in res.info.items() if k != "dither_mask"}
    if "dither_mask" in res.info:
        m = np.asarray(res.info["dither_mask"], bool)
        info["dither_mask_sha256"] = hashlib.sha256(np.packbits(m).tobytes()).hexdigest()
        info["dither_mask_shape"] = list(m.shape)
    meta = {
        "size": list(res.image.size),
        "mode": res.image.mode,
        "palette": _plain(np.asarray(res.palette)),
        "used_colors": res.used_colors,
        "pixel_aspect": res.pixel_aspect,
        "info": _plain(info),
        "settings": _plain(asdict(settings)),
    }
    return {f"{name}.png": png_bytes(res.image), f"{name}.json": json_bytes(meta)}


# ---------------------------------------------------------------------------
# ケース
# ---------------------------------------------------------------------------

Case = Callable[[], dict[str, bytes]]
CASES: dict[str, Case] = {}


def case(name: str):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def _conv(name: str, src: Image.Image, overrides: dict, palette=None) -> dict[str, bytes]:
    s = apply_overrides(ConvertSettings(), overrides)
    return result_files(name, convert(src, s, palette=palette), s)


# --- 通常変換 (mode="normal") ----------------------------------------------

_NORMAL = {
    "default": {},
    "dither_none": {"dither": "none"},
    "dither_bayer2": {"dither": "bayer2"},
    "dither_bayer8": {"dither": "bayer8"},
    "dither_floyd": {"dither": "floyd"},
    "dither_atkinson": {"dither": "atkinson"},
    "dither_sierra_lite": {"dither": "sierra_lite"},
    "dither_jjn": {"dither": "jjn", "dither_strength": 0.5},
    "mediancut_perceptual": {"palette_source": "mediancut", "perceptual": True},
    "colors8_nosnap": {"colors": 8, "snap_4096": False, "dither": "floyd"},
    "res_320x200_fill_lanczos": {"preset": "320×200", "fit": "fill", "resample": "lanczos"},
    "res_640x200_line_double": {"preset": "PC-98 640×200 (ライン倍化)"},
    "res_pixel_size": {"preset": "ピクセルサイズ指定", "pixel_size": 3, "pixel_aspect": 1.0,
                       "resample": "bilinear"},
    "res_custom_stretch_nearest": {"preset": "カスタム", "custom_width": 200, "custom_height": 150,
                                   "fit": "stretch", "resample": "nearest"},
    "adjust": {"brightness": 15, "contrast": 20, "saturation": -30},
}

_PRESET_PALETTES = {
    "preset_pc98_16": "PC-98 標準16色",
    "preset_pc98_digital8": "PC-98 デジタル8色",
    "preset_vga16": "VGA 16色",
}

for _k in ("default", "dither_floyd", "res_320x200_fill_lanczos"):
    case(f"normal/synth_photo_{_k}")(lambda _k=_k: _conv(
        f"synth_photo_{_k}", _img(fixture(SYNTH_PHOTO)), {"mode": "normal", **_NORMAL[_k]}))
case("normal/synth_alpha_background")(lambda: _conv(
    "synth_alpha_background", _img(fixture(SYNTH_ALPHA)),
    {"mode": "normal", "background": (255, 255, 255), "dither": "floyd"}))
for _k, _p in _PRESET_PALETTES.items():
    for _d in ("bayer4", "floyd"):
        case(f"normal/synth_photo_{_k}_{_d}")(lambda _p=_p, _d=_d, _k=_k: _conv(
            f"synth_photo_{_k}_{_d}", _img(fixture(SYNTH_PHOTO)),
            {"mode": "normal", "palette_source": f"preset:{_p}", "dither": _d}))


# --- パレット・トランスファー（合成のお手本から抽出したパレットを強制） ------

def _ref_palette(path: Path, method: str = "auto", n: int = 16, snap: bool = True) -> np.ndarray:
    return extract_palette(_img(path).convert("RGB"), n, method, snap)


@case("transfer/synth_photo_synth_ref_auto")
def _transfer_auto():
    pal = _ref_palette(fixture(SYNTH_REF), "auto")
    out = _conv("synth_photo_synth_ref_auto", _img(fixture(SYNTH_PHOTO)),
                {"mode": "normal", "dither": "bayer4"}, pal)
    # GPL 書き出し（パレットの保存と同じ関数）も比較する
    with tempfile.TemporaryDirectory() as tmp:
        gpl = Path(tmp) / "palette.gpl"
        save_palette(pal, gpl, name="synth_ref")
        out["synth_photo_synth_ref_auto.gpl"] = gpl.read_bytes()
    return out


@case("transfer/synth_photo_synth_ref_mediancut")
def _transfer_mediancut():
    pal = _ref_palette(fixture(SYNTH_REF), "mediancut", 12, False)
    return _conv("synth_photo_synth_ref_mediancut", _img(fixture(SYNTH_PHOTO)),
                 {"mode": "normal", "dither": "sierra_lite"}, pal)


@case("transfer/synth_photo_synth_ref_style")
def _transfer_style():
    # スタイル変換に固定パレットを渡す（infer_ramps でランプ推定）
    pal = _ref_palette(fixture(SYNTH_REF))
    return _conv("synth_photo_synth_ref_style", _img(fixture(SYNTH_PHOTO)),
                 {"mode": "style", "style_ramp_extend": True}, pal)


# --- スタイル変換 (mode="style") -------------------------------------------

_STYLE = {
    "default": {},
    "kuwahara_tile2": {"style_smooth": "kuwahara", "style_smooth_strength": 0.6,
                       "style_pattern": "tile2"},
    "bayer4_outline_color": {"style_pattern": "bayer4", "style_outline": True,
                             "style_outline_color": (0, 0, 0), "style_despeckle": 2},
    "no_smooth_no_dither": {"style_smooth": "none", "style_pattern": "none",
                            "style_despeckle": 0},
    "ramps_steps_fixed": {"style_ramps": 4, "style_steps": 3, "style_hue_separation": False,
                          "style_reserve_skin": False},
    "cross_ramp": {"style_cross_ramp": 0.6, "style_cross_ramp_max_lum": 40.0,
                   "style_cross_ramp_max_hue": 120.0},
}

for _k in ("default", "cross_ramp"):
    case(f"style/synth_photo_{_k}")(lambda _k=_k: _conv(
        f"synth_photo_{_k}", _img(fixture(SYNTH_PHOTO)), {"mode": "style", **_STYLE[_k]}))
case("style/synth_alpha_default")(lambda: _conv(
    "synth_alpha_default", _img(fixture(SYNTH_ALPHA)), {"mode": "style"}))


# --- スタイルプリセット -------------------------------------------------------

for _name, _slug in PRESET_SLUGS.items():
    case(f"presets/synth_photo_{_slug}")(lambda _name=_name, _slug=_slug: _conv(
        f"synth_photo_{_slug}", _img(fixture(SYNTH_PHOTO)), STYLE_PRESETS[_name]))


# --- プロファイル解析 → 設定 → 変換・自動調整（合成のお手本） --------------

@case("profile/synth_ref")
def _profile_case():
    path = fixture(SYNTH_REF)
    prof = build_profile([analyze_image(_img(path), path.name)[0]], "synth")
    out = {"synth_ref_profile.json": json_bytes(prof)}
    for cross in (False, True):
        ov, notes = profile_to_settings(prof, cross_ramp=cross)
        tag = "cross" if cross else "plain"
        out[f"synth_ref_settings_{tag}.json"] = json_bytes({"overrides": _plain(ov), "notes": notes})
        out.update(_conv(f"synth_photo_synth_ref_{tag}", _img(fixture(SYNTH_PHOTO)), ov))
    start = apply_overrides(ConvertSettings(), profile_to_settings(prof)[0])
    src = _img(fixture(SYNTH_PHOTO))
    fit = fit_to_profile(src, prof, start, max_evals=8)
    out["synth_ref_fit.json"] = json_bytes({
        "overrides": _plain(fit.overrides),
        "distance": fit.distance,
        "start_distance": fit.start_distance,
        "evaluations": fit.evaluations,
        "penalty": fit.penalty,
        "history": _plain(fit.history),
    })
    out.update(result_files("synth_photo_synth_ref_fit", convert(src, fit.settings), fit.settings))
    return out


# --- tools/export_db.py の JSON / CSV / GPL ---------------------------------

def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"golden_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@case("export_db/synth")
def _export_db():
    import contextlib

    export_db = _load_tool("export_db")
    paths = [str(fixture(SYNTH_REF)), str(fixture(SYNTH_REF_BAD))]
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            assert export_db.main(paths + ["--out", tmp]) == 0
        base = Path(tmp)
        return {p.relative_to(base).as_posix(): p.read_bytes()
                for p in sorted(base.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------


def run_case(name: str) -> dict[str, bytes]:
    """ケースを実行し、`tests/golden/` からの相対パス → 内容 を返す。"""
    group = name.split("/", 1)[0]
    return {f"{group}/{k}": v for k, v in CASES[name]().items()}
