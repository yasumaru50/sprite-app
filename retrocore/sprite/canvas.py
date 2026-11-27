"""キャンバスを倍数にそろえる（保存の直前に、透明な余白を足すだけ。絵は拡大・縮小しない）。

ゲームのエンジンやエディタによっては、画像の大きさが 8・16・32 の倍数だと扱いやすい。シートの 1 フレーム（1 枚の絵はその絵）の
幅と高さを、それぞれ `multiple` の倍数の、いちばん近い大きさまで広げる（縦横同じ倍数。もう倍数なら足さない）。

- 並べ方 `align`: `bottom`（既定）= 下ぞろえ・左右中央（足元の高さが変わらない）/ `center` = 上下左右中央
- 余りが奇数のときは、右（`center` では下も）に 1 ドット多く足す。例: 45×37・倍数 16・`bottom` → 48×48（左 1・右 2・上 11・下 0）
- コマのシートは「コマの大きさ」をそろえ、全部のコマに同じ余白を足す（コマどうしの足元はそろったまま）
- 余白は最後に足す（直し edits.json・差し替え swaps.json は余白を足す前の座標のまま）。JSON の `pivot`（と `ground_y`）は
  足した分だけずらし、いちばん外の `canvas` に足した余白を書く。エフェクトのシートは大きさを変えず、`position` を余白の分ずらす
- そろえない（`multiple` が 0 か None）ときは何もしない（出力は前とバイト単位で同じ）
"""

from __future__ import annotations

import dataclasses

import numpy as np

ALIGNS = {"bottom": "下ぞろえ・左右中央", "center": "上下左右中央"}
DEFAULT_ALIGN = "bottom"
MAX_MULTIPLE = 256
CHOICES = (8, 16, 32)   # 画面の選択肢（ほかは「任意の数」）


def check(multiple, align=None) -> tuple[int, str]:
    """設定を確かめて (倍数, 並べ方) を返す。倍数 0・None は「そろえない」（0）。"""
    if multiple is None:
        multiple = 0
    if isinstance(multiple, bool) or not isinstance(multiple, (int, np.integer)):
        raise ValueError(f"キャンバスの倍数は 0（そろえない）か 1〜{MAX_MULTIPLE} の整数にしてください: {multiple!r}")
    multiple = int(multiple)
    if not 0 <= multiple <= MAX_MULTIPLE:
        raise ValueError(f"キャンバスの倍数は 0（そろえない）か 1〜{MAX_MULTIPLE} にしてください: {multiple}")
    align = DEFAULT_ALIGN if align is None else align
    if align not in ALIGNS:
        raise ValueError(f"キャンバスの並べ方は {' / '.join(ALIGNS)} のどちらかにしてください: {align!r}")
    return multiple, align


def from_dict(d) -> tuple[int, str]:
    """{"multiple": 16, "align": "bottom"}（か None）を読む。"""
    if d is None:
        return 0, DEFAULT_ALIGN
    if not isinstance(d, dict):
        raise ValueError("キャンバスの設定は {\"multiple\": 16, \"align\": \"bottom\"} の形にしてください")
    unknown = sorted(set(d) - {"multiple", "align"})
    if unknown:
        raise ValueError(f"知らないキャンバスの設定です: {', '.join(unknown)}")
    return check(d.get("multiple"), d.get("align"))


def pads(width: int, height: int, multiple: int, align: str = DEFAULT_ALIGN) -> dict:
    """足す余白 {"left", "right", "top", "bottom"}（ドット）。"""
    def up(v):
        return -(-int(v) // multiple) * multiple if multiple else int(v)
    ex, ey = up(width) - width, up(height) - height
    left = ex // 2
    if align == "center":
        top = ey // 2
    else:
        top = ey
    return {"left": left, "right": ex - left, "top": top, "bottom": ey - top}


def record(width: int, height: int, multiple: int, align: str) -> dict:
    """JSON の `canvas` に書く記録（足す前の大きさと、足した余白）。"""
    p = pads(width, height, multiple, align)
    return {"multiple": int(multiple), "align": align, "width": int(width), "height": int(height), **p}


def pad_image(rgba: np.ndarray, rec: dict) -> np.ndarray:
    """1 枚（RGBA）に余白を足す。"""
    if not any(rec[k] for k in ("left", "right", "top", "bottom")):
        return rgba
    return np.pad(np.asarray(rgba, np.uint8), ((rec["top"], rec["bottom"]), (rec["left"], rec["right"]), (0, 0)))


def crop_image(rgba: np.ndarray, rec: dict | None) -> np.ndarray:
    """余白を足した 1 枚から、足す前の絵を取り出す（`rec` が None ならそのまま）。"""
    if not rec:
        return rgba
    return np.asarray(rgba)[rec["top"]:rec["top"] + rec["height"], rec["left"]:rec["left"] + rec["width"]]


def pad_result(res, multiple, align=None):
    """動き・コマのシートの材料（`MotionResult`）の全部のフレームに同じ余白を足した、新しい結果を返す。
    そろえないときは `res` をそのまま返す。足したら `info["canvas"]` に記録し、`info["pivot_x"]`（コマのシート）もずらす。"""
    multiple, align = check(multiple, align)
    if not multiple:
        return res
    w, h = res.size
    rec = record(w, h, multiple, align)
    info = dict(res.info)
    info["canvas"] = rec
    if "pivot_x" in info:
        info["pivot_x"] = float(info["pivot_x"]) + rec["left"]
    return dataclasses.replace(res, frames=[pad_image(f, rec) for f in res.frames], info=info)


def image_info(width: int, height: int, image_name: str, rec: dict) -> dict:
    """1 枚の絵（動きなし）を倍数にそろえたときに一緒に書く JSON の辞書。pivot・ground_y は足元（足す前の絵の下端の中央）。"""
    gy = rec["height"] + rec["top"]
    return {"version": 1, "kind": "sprite_image", "image": image_name, "width": int(width), "height": int(height),
            "pivot": {"x": rec["width"] / 2 + rec["left"], "y": gy}, "ground_y": gy, "canvas": dict(rec)}


def image_json(width: int, height: int, image_name: str, rec: dict) -> str:
    import json
    return json.dumps(image_info(width, height, image_name, rec), ensure_ascii=False, indent=2) + "\n"


def pad_still(rgba: np.ndarray, image_name: str, multiple, align=None):
    """1 枚の絵に余白を足す。戻り値は (足した絵, JSON の文字列)。そろえないときは (rgba, None)。"""
    multiple, align = check(multiple, align)
    if not multiple:
        return rgba, None
    h, w = rgba.shape[:2]
    rec = record(w, h, multiple, align)
    out = pad_image(rgba, rec)
    return out, image_json(out.shape[1], out.shape[0], image_name, rec)


def summary(rec: dict | None) -> str:
    """表示の 1 行（「45×37 → 48×48（左 1・右 2・上 11・下 0）」）。"""
    if not rec:
        return ""
    w = rec["width"] + rec["left"] + rec["right"]
    h = rec["height"] + rec["top"] + rec["bottom"]
    return (f"キャンバスを {rec['multiple']} の倍数に（{ALIGNS[rec['align']]}）: {rec['width']}×{rec['height']} → {w}×{h}"
            f"（左 {rec['left']}・右 {rec['right']}・上 {rec['top']}・下 {rec['bottom']}）")
