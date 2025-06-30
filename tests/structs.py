from enum import StrEnum, auto
from typing import NamedTuple
import polars as pl


class ColNames(StrEnum):
    GROUP = auto()
    TIME_MS = "time(ms)"
    LIBRARY = auto()
    VERSION = auto()
    TIME_TARGET = auto()
    MEDIAN_TIME = auto()
    FUNC_LIB = auto()


class Files(StrEnum):
    NDJSON = auto()
    PARQUET = auto()
    BASE_DIR = "C:/Users/tibo/python_codes/rustats/tests/data/"
    PRICES = f"{BASE_DIR}prices.{PARQUET}"
    PASSES = f"{BASE_DIR}passes.{NDJSON}"
    BENCH_HISTORY = f"{BASE_DIR}bench_history.{NDJSON}"
    RELATIVE_HISTORY = f"{BASE_DIR}relative_history.{NDJSON}"


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
