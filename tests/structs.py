from collections.abc import Callable
from enum import StrEnum, auto
from typing import Literal, NamedTuple, Protocol, Any
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
import polars as pl


@dataclass(slots=True)
class BenchmarkConfig:
    array: NDArray[np.float64]
    df: pl.DataFrame
    min_length: int = 25
    length: int = 250
    axis: int = 0
    time_target: int = 20
    limit: float = 0.95

    def set_time_target(self) -> None:
        time_input: str = input(
            "write the time target in seconds, press enter for 20 seconds default>"
        ).strip()
        if not time_input == "":
            self.time_target = int(time_input)


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
    PL_BENCH = f"{POLARS} - {RUSTATS_PARALLEL}"


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
    Library.PL_BENCH: "white",
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

    def __call__(self, config: BenchmarkConfig) -> Any: ...


class StatFunc[T: NDArray[np.float64] | pl.DataFrame]:
    library: Library

    def __init__(
        self,
        func: Callable[..., T],
    ) -> None:
        self.func = func
        self.results: list[Result] = []

    def __call__(self, config: BenchmarkConfig) -> T: ...


class BnFunc(StatFunc[NDArray[np.float64]]):
    library = Library.BOTTLENECK

    def __call__(self, config: BenchmarkConfig) -> NDArray[np.float64]:
        return self.func(
            config.array,
            window=config.length,
            min_count=config.min_length,
            axis=config.axis,
        )


class NbgFunc(StatFunc[NDArray[np.float64]]):
    library = Library.NUMBAGG

    def __call__(self, config: BenchmarkConfig) -> NDArray[np.float64]:
        return self.func(
            config.array,
            window=config.length,
            min_count=config.min_length,
            axis=config.axis,
        )


class PlFunc(StatFunc[pl.DataFrame]):
    library = Library.POLARS

    def __call__(self, config: BenchmarkConfig) -> pl.DataFrame:
        return self.func(config.df, length=config.length, min_length=config.min_length)


class RSingleFunc(StatFunc[NDArray[np.float64]]):
    library = Library.RUSTATS

    def __call__(self, config: BenchmarkConfig) -> NDArray[np.float64]:
        return self.func(config.array, config.length, config.min_length, False)


class RParallelFunc(StatFunc[NDArray[np.float64]]):
    library = Library.RUSTATS_PARALLEL

    def __call__(self, config: BenchmarkConfig) -> NDArray[np.float64]:
        return self.func(config.array, config.length, config.min_length, True)
