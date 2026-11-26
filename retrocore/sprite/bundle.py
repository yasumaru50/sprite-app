"""画面の「まとめて保存（ZIP）」で使う、いくつかのファイルを 1 つの ZIP にまとめる処理。

    from retrocore.sprite.bundle import zip_files
    data = zip_files([("hero_sprite_walk.png", png), ("hero_sprite_walk.json", json_bytes)])

- 中のファイルは渡したバイト列をそのまま入れる（圧縮しない = ZIP_STORED。PNG・GIF はもともと圧縮されていて、
  圧縮のしかたが zlib の版で変わることもないので、どこで作っても同じ ZIP になる）
- 中の日時は 1980-01-01 00:00:00 に固定する。同じファイルを同じ順に渡せば、同じ ZIP（バイト単位）になる
- 同じ名前が 2 回あれば、2 つ目からは入れない（先に渡したものを使う）
"""

from __future__ import annotations

import base64
import io
import json
import zipfile

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)   # ZIP で書ける最も古い日時
FILE_MODE = 0o100644                       # ふつうのファイル（展開したときに読めるように）


def zip_files(files) -> bytes:
    """[(名前, バイト列), ...] を、日時を固定した ZIP のバイト列にする。名前は / を含まないファイル名（日本語も可）。"""
    buf = io.BytesIO()
    seen = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in files:
            name = str(name)
            if not name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError(f"ZIP に入れるファイルの名前が正しくありません: {name!r}")
            if name in seen:
                continue
            seen.add(name)
            info = zipfile.ZipInfo(name, FIXED_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3   # 作った OS（Windows でも同じ ZIP にするため固定）
            info.external_attr = FILE_MODE << 16
            z.writestr(info, bytes(data))
    return buf.getvalue()


def zip_for_web(datas: list, names: list) -> str:
    """ブラウザ UI（Pyodide）から呼ぶ形。datas はバイト列の並び、names は同じ順のファイル名。
    結果は JSON 文字列（{"ok": true, "zip_base64": ..., "names": [入れた名前]}）。"""
    try:
        if len(datas) != len(names):
            raise ValueError("ファイルと名前の数が合いません")
        pairs = list(zip(names, datas))
        data = zip_files(pairs)
        used = []
        for n, _ in pairs:
            if n not in used:
                used.append(n)
        return json.dumps({"ok": True, "zip_base64": base64.b64encode(data).decode("ascii"), "names": used},
                          ensure_ascii=False)
    except (ValueError, TypeError) as e:
        return json.dumps({"ok": False, "error": f"ZIP を作れませんでした: {e}"}, ensure_ascii=False)
