"""動きの出力の「記録」（ハッシュ）。範囲（帯・四角）を足す前と後で、今あるプリセットと範囲を書いていない表の出力が
変わらないことを確かめるのに使う（`tests/test_sprite_region.py`）。

前の版の retrocore でも動くように、ここでは前からある関数だけを使う（帯・四角の範囲は使わない）。記録するもの:

- プリセット（呼吸・横揺れ・浮遊・弾む・うなずく・伸び）のいくつかの設定: 作れたか・警告・表示時間・info・埋めた表・
  フレームの画素・シートの JSON・シートの PNG の画素
- そのプリセットの表（数と割合の 2 通り）を表の仕組みで作り直したもの、tests/motion_specs/ と見本の表（範囲は分け目だけ）
- 画面（`motion_for_web()`）の呼び方での同じもの（シートの PNG は画素で比べる）
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from legacy_output import legacy_sheet_text, legacy_web

from retrocore.sprite import SpriteSettings, convert_sprite
from retrocore.sprite.convert import encode_png, motion_for_web
from retrocore.sprite.motion import (load_spec, make_motion, make_motion_from_spec, make_sheet, sheet_json, spec_json,
                                     with_height_style)
from retrocore.sprite.motion.base import split_limits
from synth_sprite import enlarge_like_ai, make_pixel_sprite
from test_sprite_motion import make_slime

ROOT = Path(__file__).resolve().parents[1]
# 範囲を書いていない表のファイル（前の版からあるもの）
SPLIT_SPECS = [ROOT / "tests" / "motion_specs" / n for n in ("bounce.motion.json", "bounce_squash.motion.json",
                                                              "broken.motion.json")] + \
              [ROOT / "apps" / "sprite" / "examples" / "bounce_stretch.motion.json"]


def make_knight() -> np.ndarray:
    """剣を持ったキャラ（合成）。make_pixel_sprite() の左に透明の列を 4 つ足し、左の端に縦の剣（3 列）を描く。

    剣は下から 3〜14 ドット目にあり、頭（下から 10 ドット目から上）の高さにまたがる（分け目 1 本で頭を動かすと剣の上も動く）。
    """
    base = np.pad(make_pixel_sprite(), ((0, 0), (4, 0), (0, 0)))
    h = base.shape[0]
    dark = base[h - 1, 4 + 7].copy()            # 輪郭の色（いちばん上の行の色）
    blade = np.array([200, 200, 215, 255], np.uint8)
    for n in range(3, 15):                       # 下から何ドット目
        y = h - n
        base[y, 0] = dark
        base[y, 1] = blade if n < 14 else dark
        base[y, 2] = dark
    return base


def sprites() -> dict:
    return {"mushroom": make_pixel_sprite(28, 32), "slime": make_slime(), "pixel": make_pixel_sprite(),
            "hero": convert_sprite(enlarge_like_ai(make_pixel_sprite(), seed=3)[0], SpriteSettings()).image,
            "knight": make_knight()}


def _dig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _pixels(a: np.ndarray) -> str:
    return _dig(repr(a.shape).encode() + np.ascontiguousarray(a).tobytes())


def _result(r) -> dict:
    d = {"skipped": r.skipped, "warnings": r.warnings, "durations": r.durations, "info": _jsonable(r.info),
         "spec": r.spec, "frames": _dig(b"".join(repr(f.shape).encode() + f.tobytes() for f in r.frames)),
         "json": legacy_sheet_text(sheet_json(r, "x.png"))}
    if not r.skipped:
        d["sheet"] = _pixels(make_sheet(r.frames))
    return d


def preset_options(sp: np.ndarray) -> list:
    lim = split_limits(sp)
    mid = None if lim is None else (lim[0] + lim[1]) // 2
    out = [("breathe", {}), ("breathe", {"frames": 4, "depth": 2}), ("breathe", {"keep_small_parts": True}),
           ("sway", {}), ("sway", {"frames": 8, "amount": 2}), ("float", {}), ("float", {"amount": 2}),
           ("bounce", {}), ("bounce", {"height": 3}), ("nod", {}), ("nod", {"depth": 2}),
           ("nod", {"keep_small_parts": True}), ("stretch", {}), ("stretch", {"amount": 2})]
    if mid is not None:
        out += [("breathe", {"split": mid}), ("sway", {"split": mid}), ("nod", {"split": mid}),
                ("stretch", {"split": mid})]
    return out


def records(sp: np.ndarray) -> dict:
    """1 枚の絵の記録（名前 → ハッシュ）。"""
    out = {}
    png = encode_png(sp)
    for motion, opts in preset_options(sp):
        key = f"{motion}|{json.dumps(opts, sort_keys=True)}"
        try:
            r = make_motion(sp, motion, **opts)
        except ValueError as e:
            out[key] = _dig(str(e).encode())
            continue
        rec = {"preset": _result(r)}
        if r.spec is not None:
            for style in ("number", "percent"):
                table = with_height_style(r.spec, sp, style)
                rec[f"text_{style}"] = spec_json(table)
                rec[f"table_{style}"] = _result(make_motion_from_spec(sp, table))
        out[key] = _dig(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode())
    for path in SPLIT_SPECS:
        r = make_motion_from_spec(sp, load_spec(path))
        out[f"file|{path.name}"] = _dig(json.dumps(_result(r), ensure_ascii=False, sort_keys=True).encode())
    # 画面（motion_for_web）の呼び方: プリセットと表
    web = [{"motion": "breathe"}, {"motion": "sway", "amount": 2, "frames": 8}, {"motion": "nod"},
           {"motion": "stretch"}, {"spec": json.loads(SPLIT_SPECS[1].read_text(encoding="utf-8"))}]
    for opts in web:
        res = legacy_web(json.loads(motion_for_web(png, json.dumps(opts))))
        b64 = res.pop("sheet_png_base64", None)
        if b64 is not None:
            res["sheet_pixels"] = _pixels(np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")))
        res.pop("detail", None)
        out[f"web|{json.dumps(opts, sort_keys=True)[:80]}"] = _dig(json.dumps(res, ensure_ascii=False,
                                                                             sort_keys=True).encode())
    return out


def digest_all() -> dict:
    """絵ごとの記録をまとめたハッシュ（絵の名前 → ハッシュ）。"""
    return {name: _dig(json.dumps(records(sp), sort_keys=True).encode()) for name, sp in sprites().items()}


if __name__ == "__main__":
    print(json.dumps(digest_all(), indent=1))
