from enum import StrEnum, auto
from typing import NamedTuple
import numpy as np
from dataclasses import dataclass
from importlib import metadata
from numpy.typing import NDArray
import polars as pl


class ColNames(StrEnum):
    GROUP = "Group"
    TIME_MS = "Time (ms)"
    LIBRARY = "Library"
    VERSION = auto()
    TIME_TARGET = auto()


@dataclass(slots=True)
class BenchmarkConfig:
    array: NDArray[np.float64]
    df: pl.DataFrame
    min_length: int = 25
    length: int = 250
    axis: int = 0
    time_target: int = 20
    limit: float = 0.95
    version_nb = metadata.version("rustats")

    def set_time_target(self) -> None:
        time_input: str = input(
            f"write the time target in seconds, press enter for {self.time_target} seconds default>"
        ).strip()
        if not time_input == "":
            self.time_target = int(time_input)


class Files(StrEnum):
    BASE_DIR = "C:/Users/tibo/python_codes/rustats/tests/data/"
    PRICES = f"{BASE_DIR}prices.parquet"
    PASSES = f"{BASE_DIR}passes.ndjson"
    BENCH_HISTORY = f"{BASE_DIR}bench_history.ndjson"
    RELATIVE_HISTORY = f"{BASE_DIR}relative_history.ndjson"


class Library(StrEnum):
    POLARS = auto()
    BOTTLENECK = auto()
    RUSTATS = auto()
    RUSTATS_PARALLEL = auto()
    NUMBAGG = auto()
    BN_BENCH = f"{BOTTLENECK} - {RUSTATS}"
    NBG_BENCH = f"{NUMBAGG} - {RUSTATS_PARALLEL}"
    PL_BENCH = f"{POLARS} - {RUSTATS_PARALLEL}"


class StatType(StrEnum):
    MEAN = auto()
    SUM = auto()
    VAR = auto()
    STD = auto()
    MAX = auto()
    MIN = auto()
    MEDIAN = auto()
    RANK = auto()
    SKEW = auto()
    KURT = auto()


TEMPLATE = "plotly_dark"


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


RESULT_SCHEMA = {
    ColNames.LIBRARY.value: pl.String,
    ColNames.GROUP.value: pl.String,
    ColNames.TIME_MS.value: pl.Float64,
}

HISTORY_SCHEMA = {
    ColNames.GROUP.value: pl.String,
    ColNames.LIBRARY.value: pl.String,
    ColNames.VERSION.value: pl.String,
    "median_time": pl.Float64,
    ColNames.TIME_TARGET.value: pl.Int32,
}

PASSES_SCHEMA = {
    ColNames.GROUP.value: pl.String,
    "total_time_secs": pl.Float64,
    "n_passes": pl.Int64,
    "time_per_pass_ms": pl.Float64,
}


class Result(NamedTuple):
    library: Library
    group: str
    time: float
