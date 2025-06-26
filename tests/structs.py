from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from time import perf_counter
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

type Computation = Callable[[NDArray[np.float64]], NDArray[np.float64]]


class Length(IntEnum):
    MIN = 25
    FULL = 250


class Files(StrEnum):
    PRICES = "C:/Users/tibo/python_codes/rustats/tests/prices.parquet"
    SUMMARY = "C:/Users/tibo/python_codes/rustats/tests/benchmark_summary.ndjson"


class Library(StrEnum):
    BOTTLENECK = auto()
    RUSTATS = auto()
    RUSTATS_PARALLEL = auto()
    NUMBAGG = auto()


COLORS: dict[Library, str] = {
    Library.RUSTATS: "orange",
    Library.RUSTATS_PARALLEL: "red",
    Library.NUMBAGG: "blue",
    Library.BOTTLENECK: "green",
}

StatType = Literal[
    "mean",  # 🐍 ✅
    "sum",  # 🦀 ✅
    "var",  # 🐍 ✅
    "std",  # 🦀 ✅
    "max",  # 🦀 ✅
    "min",  # 🦀 ✅
    "median",  # 🦀 ✅
    "rank",  # 🦀 ✅
]


class Result(NamedTuple):
    library: Library
    group: str
    time: float


class StatFunc:
    def __init__(
        self,
        library: Library,
        func: Computation,
    ) -> None:
        self.library: Library = library
        self.func: Computation = func
        self.results: list[Result] = []

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr)


@dataclass(slots=True)
class FuncGroup:
    funcs: list[StatFunc]

    def warmup(self):
        arr: NDArray[np.float64] = np.random.rand(1000, 10).astype(np.float64)
        for func in self.funcs:
            for _ in range(10):
                func(arr)

    def time_group(
        self,
        group_name: str,
        arr: NDArray[np.float64],
        n_passes: int,
    ) -> list[Result]:
        results: list[Result] = []
        total: int = len(self.funcs) * n_passes
        with tqdm(total=total) as pbar:
            for func in self.funcs:
                pbar.set_description(f"Timing {group_name} - {func.library}")
                for _ in range(n_passes):
                    start_time: float = perf_counter()
                    func(arr=arr)
                    elapsed_time: float = (perf_counter() - start_time) * 1000
                    results.append(
                        Result(
                            library=func.library,
                            group=group_name,
                            time=elapsed_time,
                        )
                    )
                    pbar.update(1)
        return results
