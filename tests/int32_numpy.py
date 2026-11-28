"""テスト用: 64 ビットの環境で、32 ビットの NumPy（Pyodide の wasm32 など）でだけ起きる型のエラーを見つける仕組み。

32 ビットの環境では、配列の番号の型 `np.intp` が int32 になる。`np.bincount`・`np.repeat` などは、受け取った配列を
「安全に」（値が失われない場合だけ）intp に変換するので、int64 の配列を渡すと
`TypeError: Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'` になる。
64 ビットの環境では intp が int64 なので、このエラーは起きない。

見分け方: 64 ビットの環境では「明示した int64」（`np.int64`）と「intp（`np.arange` などの既定の整数）」は同じ型なので、
`retrocore` のモジュールから見える `np` を差し替え、`np.int64` を、大きさは同じで型の文字だけが違う `np.longlong`（'q'）にする。
NumPy の型の昇格では 'q' と intp（'l'）が混ざると 'q' が残るので、32 ビットで int64 になる計算は 'q' になる。
差し替えた `np.bincount` などは、32 ビットで intp（int32）に安全に変換できない型（'q'、符号なしの 32・64 ビットなど）を
受け取ると、32 ビットの環境と同じエラーを出す。計算の結果（値）は変わらない。

確かめる関数（64 ビットで uint64 を渡して「rule 'safe'」になるかで調べたもの。bincount は 32 ビットでの既知のエラー）:
`np.bincount`（1 つ目の引数）、`np.repeat`（repeats）、`np.put`（ind）、`np.choose`（a）、ufunc の `reduceat`（indices）、
`np.take`（indices。Pyodide のブラウザ版で、写真の下書きの int64 の番号がこのエラーになった）。
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import numpy as _np

# 32 ビットの環境で int32（intp）に安全に変換できる整数の型（'l' は intp、32 ビットでは int32 になる）
_SAFE_CHARS = set("bBhHil?")

# この仕組みが使えるか: intp が 'l'（Linux・macOS の 64 ビット）で、int64 の印の 'q'（longlong）と区別できるとき。
# Windows の NumPy 2 は intp が 'q' そのものなので区別できない（CI では Linux のジョブで動かす）
AVAILABLE = _np.dtype(_np.intp).char == "l" and _np.dtype(_np.longlong).char == "q"


class Int32CastError(TypeError):
    pass


# 調べるとき用: None 以外のリストにすると、エラーにせず見つかった場所（retrocore のファイル名:行）を足していく
FOUND: list | None = None


def _check(value, func: str, arg: str) -> None:
    """value を 32 ビットの intp に安全に変換できなければ Int32CastError。"""
    if isinstance(value, (_np.ndarray, _np.generic)):
        dt = value.dtype
        if dt.char not in _SAFE_CHARS:
            name = "int64" if dt.char == "q" else str(dt)
            if FOUND is not None:
                f = sys._getframe(1)
                while f is not None and "retrocore" not in f.f_code.co_filename:
                    f = f.f_back
                if f is not None:
                    FOUND.append(f"{f.f_code.co_filename.split('retrocore')[-1]}:{f.f_lineno} {func}")
                return
            raise Int32CastError(
                f"{func} の {arg} に {name} の配列を渡しています。32 ビットの環境（Pyodide）では "
                f"\"Cannot cast array data from dtype('{name}') to dtype('int32') according to the rule 'safe'\" "
                "になります（.astype(np.intp) などで intp にしてから渡す）")
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check(v, func, arg)


def _wrap(func, name: str, pos: int, kw: str):
    def wrapper(*args, **kwargs):
        if len(args) > pos:
            _check(args[pos], f"np.{name}", kw)
        elif kw in kwargs:
            _check(kwargs[kw], f"np.{name}", kw)
        return func(*args, **kwargs)
    wrapper.__name__ = name
    wrapper.__wrapped__ = func
    return wrapper


class _UFunc:
    """ufunc のふりをして、reduceat の indices を確かめる。"""

    def __init__(self, uf):
        self._uf = uf

    def __call__(self, *args, **kwargs):
        return self._uf(*args, **kwargs)

    def reduceat(self, array, indices, *args, **kwargs):
        _check(indices, f"np.{self._uf.__name__}.reduceat", "indices")
        return self._uf.reduceat(array, indices, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._uf, name)


class _Int32NumPy(types.ModuleType):
    """`numpy` の代わりに retrocore のモジュールに見せるもの（上の差し替え以外は numpy そのもの）。"""

    def __init__(self):
        super().__init__("numpy")
        self.int64 = _np.longlong
        self.bincount = _wrap(_np.bincount, "bincount", 0, "x")
        self.repeat = _wrap(_np.repeat, "repeat", 1, "repeats")
        self.put = _wrap(_np.put, "put", 1, "ind")
        self.choose = _wrap(_np.choose, "choose", 0, "a")
        self.take = _wrap(_np.take, "take", 1, "indices")

    def __getattr__(self, name):
        value = getattr(_np, name)
        if isinstance(value, _np.ufunc):
            return _UFunc(value)
        return value


INT32_NUMPY = _Int32NumPy()


def retrocore_modules():
    return [m for n, m in sorted(sys.modules.items())
            if (n == "retrocore" or n.startswith("retrocore.")) and getattr(m, "np", None) is _np]


@contextmanager
def emulate_int32():
    """この中では、retrocore のモジュールの `np` が 32 ビットの型の確かめ付きになる。"""
    import retrocore.sprite  # noqa: F401  （読み込んでから差し替える）
    mods = retrocore_modules()
    for m in mods:
        m.np = INT32_NUMPY
    try:
        yield
    finally:
        for m in mods:
            m.np = _np
