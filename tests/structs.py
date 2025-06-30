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
    MEDIAN_TIME = auto()


@dataclass(slots=True)
class BenchmarkConfig:
    array: NDArray[np.float64]
    df: pl.DataFrame
    min_length: int = 25
    length: int = 250
    axis: int = 0
    time_target: int = 30
    limit: float = 0.95

    @property
    def version(self) -> int:
        version_str: str = metadata.version("rustats")
        return int(version_str.split(".")[-1])

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


class Colors:
    TEMPLATE = "plotly_dark"

    ABSOLUTE: dict[Library, str] = {
        Library.RUSTATS: "yellow",
        Library.RUSTATS_PARALLEL: "red",
        Library.NUMBAGG: "cyan",
        Library.BOTTLENECK: "lime",
        Library.POLARS: "white",
    }

    RELATIVE: dict[Library, str] = {
        Library.BN_BENCH: "lime",
        Library.NBG_BENCH: "cyan",
        Library.PL_BENCH: "white",
    }


class Schemas:
    stat_enum = pl.Enum(StatType)
    library_enum = pl.Enum(Library)

    RESULT = {
        ColNames.LIBRARY.value: library_enum,
        ColNames.GROUP.value: stat_enum,
        ColNames.TIME_MS.value: pl.Float64,
    }

    HISTORY = {
        ColNames.GROUP.value: stat_enum,
        ColNames.LIBRARY.value: library_enum,
        ColNames.VERSION.value: pl.Int32,
        ColNames.TIME_TARGET.value: pl.Int32,
        ColNames.MEDIAN_TIME.value: pl.Float64,
    }

    PASSES = {
        ColNames.GROUP.value: stat_enum,
        ColNames.VERSION.value: pl.Int32,
        ColNames.TIME_TARGET.value: pl.Int32,
        "total_time_secs": pl.Float64,
        "n_passes": pl.Int64,
        "time_per_pass_ms": pl.Float64,
    }


class Result(NamedTuple):
    library: Library
    group: StatType
    time: float
