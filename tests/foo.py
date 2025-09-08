import rustats as rs
import polars as pl
from collections.abc import Callable
from pathlib import Path
import numpy as np
import bottleneck as bn  # type: ignore


def path_src():
    return (
        Path(__file__)
        .parent.joinpath("data")
        .joinpath("prices")
        .with_suffix(".parquet")
    )


def compute(expr: pl.Expr, func: Callable[[np.ndarray, int, int], np.ndarray]):
    return expr.map_batches(
        lambda x: func(x.to_numpy(), 3, 2),
        pl.Float32,
    )


def test():
    return (
        pl.scan_parquet(path_src())  # type: ignore
        .head(100)
        .group_by("ticker")
        .agg(
            pl.col("close").pipe(compute, rs.move_mean).alias("rustats"),
            pl.col("close").pipe(compute, bn.move_mean).alias("bottleneck"),  # type: ignore
            pl.col("close").rolling_mean(3, min_samples=2).alias("polars"),
        )
    )


if __name__ == "__main__":
    print(test().collect())
