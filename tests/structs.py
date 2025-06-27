from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from typing import Literal, NamedTuple, Protocol
import numpy as np
from numpy.typing import NDArray
import polars as pl

type Computation = Callable[[NDArray[np.float64]], NDArray[np.float64]]
type RSFunc = Callable[[NDArray[np.float64], int, int, bool], NDArray[np.float64]]
type BNFunc = Callable[[NDArray[np.float64], int, int, int], NDArray[np.float64]]
type NBGFunc = Callable[[NDArray[np.float64], int, int, int], NDArray[np.float64]]


class Length(IntEnum):
    MIN = 25
    FULL = 250


class Files(StrEnum):
    PRICES = "C:/Users/tibo/python_codes/rustats/tests/prices.parquet"
    SUMMARY = "C:/Users/tibo/python_codes/rustats/tests/benchmark_summary.ndjson"


class Library(StrEnum):
    POLARS = auto()
    BOTTLENECK = auto()
    RUSTATS = auto()
    RUSTATS_PARALLEL = auto()
    NUMBAGG = auto()
    BN_BENCH = f"{BOTTLENECK} - {RUSTATS}"
    NBG_BENCH = f"{NUMBAGG} - {RUSTATS_PARALLEL}"
    PL_BENCH_SINGLE = f"{POLARS} - {RUSTATS}"
    PL_BENCH_PARALLEL = f"{POLARS} - {RUSTATS_PARALLEL}"


COLORS: dict[Library, str] = {
    Library.RUSTATS: "yellow",
    Library.RUSTATS_PARALLEL: "red",
    Library.NUMBAGG: "cyan",
    Library.BOTTLENECK: "lime",
    Library.POLARS: "white",
}

COLORS_BENCH: dict[Library, str] = {
    Library.BN_BENCH: "lime",
    Library.NBG_BENCH: "cyan",
    Library.PL_BENCH_SINGLE: "orange",
    Library.PL_BENCH_PARALLEL: "white",
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


class StatFuncProtocol(Protocol):
    library: Library

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]: ...


class StatFunc[T: Callable[..., NDArray[np.float64]] | pl.Expr]:
    library: Library

    def __init__(
        self,
        func: T,
    ) -> None:
        self.func: T = func
        self.results: list[Result] = []

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]: ...


class BnFunc(StatFunc[BNFunc]):
    library = Library.BOTTLENECK

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr, Length.FULL, Length.MIN, 0)


class NbgFunc(StatFunc[Computation]):
    library = Library.NUMBAGG

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr)


class PlFunc(StatFunc[pl.Expr]):
    library = Library.POLARS

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return pl.from_numpy(data=arr).select(self.func).to_numpy().astype(np.float64)


class RSingleFunc(StatFunc[RSFunc]):
    library = Library.RUSTATS

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr, Length.FULL, Length.MIN, False)


class RParallelFunc(StatFunc[RSFunc]):
    library = Library.RUSTATS_PARALLEL

    def __call__(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.func(arr, Length.FULL, Length.MIN, True)


@dataclass(slots=True)
class FuncGroup:
    funcs: list[StatFuncProtocol]

    def warmup(self):
        arr: NDArray[np.float64] = np.random.rand(1000, 10).astype(np.float64)
        for func in self.funcs:
            for _ in range(10):
                func(arr)
