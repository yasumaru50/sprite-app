"""工程を並べて順に実行するパイプライン。

各工程は `Step` を継承し、`name`（ログ・中間画像のファイル名に使う短い英名）と
`apply(rgba) -> rgba` を実装する。工程は入力配列を書き換えず、新しい配列を返すこと。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .image import to_rgba


class Step:
    """パイプラインの 1 工程。"""

    name: str = "step"

    def apply(self, rgba: np.ndarray) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    def describe(self) -> str:
        """ログ表示用の説明（既定は工程名）。"""
        return self.name


@dataclass
class StepResult:
    """1 工程の実行結果。"""

    name: str
    description: str
    image: np.ndarray
    seconds: float


@dataclass
class SpritePipeline:
    """工程のリスト。`run()` で先頭から順に適用する。"""

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> "SpritePipeline":
        self.steps.append(step)
        return self

    def run(self, rgba: np.ndarray,
            on_step: Callable[[StepResult], None] | None = None) -> np.ndarray:
        """全工程を適用した RGBA 配列を返す。`on_step` には工程ごとの結果を渡す（中間画像の保存など）。"""
        cur = to_rgba(rgba)
        for step in self.steps:
            t0 = time.perf_counter()
            out = to_rgba(step.apply(cur))
            res = StepResult(step.name, step.describe(), out, time.perf_counter() - t0)
            if on_step is not None:
                on_step(res)
            cur = out
        return cur

    def run_all(self, rgba: np.ndarray) -> list[StepResult]:
        """全工程の結果を順に返す（最後の要素が最終結果）。"""
        results: list[StepResult] = []
        self.run(rgba, on_step=results.append)
        return results

    @classmethod
    def of(cls, steps: Iterable[Step]) -> "SpritePipeline":
        return cls(list(steps))
