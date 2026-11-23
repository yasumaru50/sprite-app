"""プリセットの出力の指紋（ハッシュ）。この PR の前のコードでも同じ計算ができるように、新しい動きには触れない。

`nod_fingerprint()` は、うなずくの make_motion の結果（フレーム・警告・表示時間・埋めた表・シートの JSON。設定の間違いはエラーの文）と、
CLI のファイル（PNG は画素、JSON は改行を LF にしたバイト）・終了コード・表示、画面の motion_for_web・table_for_web の結果の
ハッシュをグループごとに返す（`tests/test_sprite_stretch.py` が、伸びの PR の前に記録した値と比べる）。
"""

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from legacy_output import legacy_bytes, legacy_files, legacy_sheet_text, legacy_stdout, legacy_web


def _dig(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


NOD_COMBOS = [{}, {"depth": 2}, {"split": 10}, {"split": 12, "depth": 2}, {"split": "45.0%"}, {"durations": 300},
              {"depth": 2, "durations": [700, 100, 150, 100]}, {"split": 3}, {"depth": 3}, {"frames": 3}]
NOD_CLI = {"n": ["--motion", "nod"], "n_d2": ["--motion", "nod", "--nod-depth", "2"],
           "n_s10": ["--motion", "nod", "--nod-split", "10"], "n_p": ["--motion", "nod", "--nod-split", "45.0%"]}
NOD_WEB = [{"motion": "nod"}, {"motion": "nod", "depth": 2}, {"motion": "nod", "split": 10, "durations": 300},
           {"motion": "nod", "split": 999}]


def nod_fingerprint(tmp: Path, sprites: dict, hero_rgba: np.ndarray) -> dict:
    from apps.sprite import cli
    from retrocore.sprite.convert import convert_for_web, motion_for_web, table_for_web
    from retrocore.sprite.motion import make_motion, sheet_json
    out = {}
    for name, sp in sprites.items():
        recs = {}
        for kw in NOD_COMBOS:
            key = json.dumps(kw, sort_keys=True)
            try:
                r = make_motion(sp, "nod", **kw)
                recs[key] = _dig(json.dumps({"s": r.skipped, "w": r.warnings, "d": r.durations, "spec": r.spec,
                                             "f": _dig(b"".join(repr(f.shape).encode() + f.tobytes() for f in r.frames)),
                                             "j": legacy_sheet_text(sheet_json(r, "x.png"))},
                                            ensure_ascii=False, sort_keys=True, default=str).encode())
            except (ValueError, TypeError) as e:
                recs[key] = str(e)
        out[f"records|{name}"] = _dig(json.dumps(recs, ensure_ascii=False, sort_keys=True).encode())
    src = tmp / "hero.png"
    Image.fromarray(hero_rgba, "RGBA").save(src)
    for tag, args in NOD_CLI.items():
        o = tmp / tag
        err, so = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
            code = cli.main([str(src), "-o", str(o / "s.png"), *args, "--write-motion-spec", str(o / "t.motion.json")])
        stdout = "\n".join(line for line in so.getvalue().replace(str(tmp), "<T>").replace("\\", "/").splitlines()
                           if "秒）" not in line)
        files = {}
        for f in legacy_files(sorted(o.iterdir())):
            if f.suffix == ".png":
                a = np.asarray(Image.open(f).convert("RGBA"))
                files[f.name] = _dig(repr(a.shape).encode() + a.tobytes())
            else:
                files[f.name] = _dig(legacy_bytes(f.read_bytes().replace(b"\r\n", b"\n")))
        out[f"cli|{tag}"] = _dig(json.dumps({"code": code, "files": files, "stdout": _dig(legacy_stdout(stdout).encode()),
                                             "stderr": err.getvalue()}, ensure_ascii=False, sort_keys=True).encode())
    png = base64.b64decode(json.loads(convert_for_web(src.read_bytes(), "{}"))["png_base64"])
    web = {}
    for o in NOD_WEB:
        for kind, text in (("web", motion_for_web(png, json.dumps(o))),
                           ("table", table_for_web(json.dumps({"preset": o}), png))):
            d = legacy_web(json.loads(text))
            d.pop("detail", None)
            if "sheet_png_base64" in d:
                a = np.asarray(Image.open(io.BytesIO(base64.b64decode(d.pop("sheet_png_base64")))).convert("RGBA"))
                d["sheet_pixels"] = _dig(repr(a.shape).encode() + a.tobytes())
            web[f"{kind}|{json.dumps(o, sort_keys=True)}"] = _dig(json.dumps(d, ensure_ascii=False, sort_keys=True).encode())
    out["web"] = _dig(json.dumps(web, ensure_ascii=False, sort_keys=True).encode())
    return out
