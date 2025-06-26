from groups import StatType
from manager import BenchmarkManager
from plots import plot_benchmark_results, plot_function_results
import polars as pl
from numpy.typing import NDArray
import numpy as np


def get_array(df: pl.DataFrame) -> NDArray[np.float64]:
    return (
        df.pivot(
            on="ticker",
            index="date",
            values="pct_return",
        )
        .drop("date")
        .to_numpy()
        .astype(dtype=np.float64)
    )


def check_results(
    manager: BenchmarkManager, array: NDArray[np.float64], group_name: StatType
) -> None:
    check: str = (
        input(
            "write 'y' if you want to check the results of this group, press enter to skip: "
        )
        .strip()
        .lower()
    )
    if check == "y":
        results = {
            func.library: func(array) for func in manager.groups[group_name].funcs
        }
        plot_function_results(results=results, group_name=group_name)


def plot_results(
    manager: BenchmarkManager, array: NDArray[np.float64], group_name: StatType
) -> None:
    time_input: str = input(
        "write the time target in seconds, press enter for 20 seconds default:"
    ).strip()
    if time_input == "":
        time_target = 20
    else:
        time_target = int(time_input)

    plot_benchmark_results(
        array=array,
        manager=manager,
        group_name=group_name,
        time_target=time_target,
        log_y=False,
        limit=95,
    )
