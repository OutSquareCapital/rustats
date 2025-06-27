from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

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
    BN_BENCH = f"{BOTTLENECK} - {RUSTATS}"
    NBG_BENCH = f"{NUMBAGG} - {RUSTATS_PARALLEL}"


COLORS: dict[Library, str] = {
    Library.RUSTATS: "yellow",
    Library.RUSTATS_PARALLEL: "red",
    Library.NUMBAGG: "cyan",
    Library.BOTTLENECK: "lime",
}

COLORS_BENCH: dict[Library, str] = {
    Library.BN_BENCH: "lime",
    Library.NBG_BENCH: "cyan",
}

StatType = Literal[
    "mean",
    "sum",
    "var",
    "std",
    "max",
    "min",
    "median",
    "rank",
    "skew",
    "kurt",
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