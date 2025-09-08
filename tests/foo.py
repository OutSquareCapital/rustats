import rustats as rs
import polars as pl
from collections.abc import Callable
from pathlib import Path
import numpy as np
import bottleneck as bn
from time import perf_counter


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


def compare_speed():
    arr1d = np.random.rand(1000).astype(np.float64)
    arr2d = np.random.rand(1000, 1000).astype(np.float64)
    iterations = 100
    start = perf_counter()
    for _ in range(iterations):
        bn.move_median(arr1d, 300, 2, axis=0)
    print("Bottleneck 1D:", _perf(start, iterations))

    start = perf_counter()
    for _ in range(iterations):
        bn.move_median(arr2d, 300, 2, axis=0)
    print("Bottleneck 2D:", _perf(start, iterations))
    start = perf_counter()
    for _ in range(iterations):
        rs.move_median(arr1d, 300, 2)
    print("Rustats 1D:", _perf(start, iterations))
    start = perf_counter()
    for _ in range(iterations):
        rs.move_median(arr2d, 300, 2)
    print("Rustats 2D:", _perf(start, iterations))


def _perf(start: float, iterations: int):
    return round(((perf_counter() - start) / iterations) * 1000, 6)


if __name__ == "__main__":
    compare_speed()
