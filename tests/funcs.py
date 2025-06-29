from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import bottleneck as bn
import numbagg as nbg
import numpy as np
import polars as pl
import rustats as rs
from numpy.typing import NDArray

from structs import BenchmarkConfig, Library


class StatFuncProtocol(Protocol):
    library: Library

    def __call__(self, config: BenchmarkConfig) -> Any: ...


class StatFunc[T: NDArray[np.float64] | pl.DataFrame]:
    library: Library

    def __init__(self, func: Callable[..., T]) -> None:
        self.func = func


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


def move_mean(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(pl.all().rolling_mean(window_size=length, min_samples=min_length))


def move_sum(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(pl.all().rolling_sum(window_size=length, min_samples=min_length))


def move_var(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(
        pl.all().rolling_var(window_size=length, min_samples=min_length, ddof=1)
    )


def move_std(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(
        pl.all().rolling_std(window_size=length, min_samples=min_length, ddof=1)
    )


def move_max(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(pl.all().rolling_max(window_size=length, min_samples=min_length))


def move_min(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(pl.all().rolling_min(window_size=length, min_samples=min_length))


def move_median(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(
        pl.all().rolling_median(window_size=length, min_samples=min_length)
    )


def move_kurtosis(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(
        pl.all().rolling_kurtosis(
            window_size=length, min_samples=min_length, bias=True, fisher=True
        )
    )


def move_skewness(df: pl.DataFrame, length: int, min_length: int) -> pl.DataFrame:
    return df.select(
        pl.all().rolling_skew(window_size=length, min_samples=min_length, bias=True)
    )


class PolarsFunc:
    kurt = PlFunc(func=move_kurtosis)
    skew = PlFunc(func=move_skewness)
    mean = PlFunc(func=move_mean)
    sum = PlFunc(func=move_sum)
    var = PlFunc(func=move_var)
    std = PlFunc(func=move_std)
    max = PlFunc(func=move_max)
    min = PlFunc(func=move_min)
    median = PlFunc(func=move_median)


class BottleneckFuncs:
    mean = BnFunc(bn.move_mean)
    sum = BnFunc(bn.move_sum)
    var = BnFunc(partial(bn.move_var, ddof=1))
    std = BnFunc(partial(bn.move_std, ddof=1))
    max = BnFunc(bn.move_max)
    min = BnFunc(bn.move_min)
    median = BnFunc(bn.move_median)
    rank = BnFunc(bn.move_rank)


class NumbaggFuncs:
    mean = NbgFunc(nbg.move_mean)  # type: ignore
    sum = NbgFunc(nbg.move_sum)  # type: ignore
    var = NbgFunc(nbg.move_var)  # type: ignore
    std = NbgFunc(nbg.move_std)  # type: ignore


class RustatSingleFuncs:
    mean = RSingleFunc(rs.move_mean)
    sum = RSingleFunc(rs.move_sum)
    var = RSingleFunc(rs.move_var)
    std = RSingleFunc(rs.move_std)
    max = RSingleFunc(rs.move_max)
    min = RSingleFunc(rs.move_min)
    median = RSingleFunc(rs.move_median)
    rank = RSingleFunc(rs.move_rank)
    skew = RSingleFunc(rs.move_skewness)
    kurt = RSingleFunc(rs.move_kurtosis)


class RustatParallelFuncs:
    mean = RParallelFunc(rs.move_mean)
    sum = RParallelFunc(rs.move_sum)
    var = RParallelFunc(rs.move_var)
    std = RParallelFunc(rs.move_std)
    max = RParallelFunc(rs.move_max)
    min = RParallelFunc(rs.move_min)
    median = RParallelFunc(rs.move_median)
    rank = RParallelFunc(rs.move_rank)
    skew = RParallelFunc(rs.move_skewness)
    kurt = RParallelFunc(rs.move_kurtosis)
