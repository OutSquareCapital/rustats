from structs import COLORS, StatType, Library
from manager import BenchmarkManager
from numpy.typing import NDArray
import numpy as np
import plotly.express as px
import polars as pl
from typing import Literal


def plot_function_results(
    results: dict[Library, NDArray[np.float64]],
    group_name: str,
) -> None:
    data: pl.DataFrame = pl.DataFrame(
        {
            "Library": [
                lib for lib in results.keys() for _ in range(results[lib].shape[0])
            ],
            "Index": [
                i for lib in results.keys() for i in range(results[lib].shape[0])
            ],
            "Values": [value for lib in results.keys() for value in results[lib][:, 0]],
        }
    )

    px.line(  # type: ignore
        data.to_pandas(),
        x="Index",
        y="Values",
        color="Library",
        title=f"Results Check - {group_name}",
        template="plotly_dark",
        color_discrete_map=COLORS,
    ).show()


def plot_group_result(
    avg_data: pl.DataFrame,
    group_name: StatType,
    kind: Literal["box", "violins"],
    log_y: bool,
) -> None:
    if kind == "box":
        px.box(  # type: ignore
            avg_data.to_pandas(),
            y="Time (ms)",
            color="Library",
            points=False,
            title=f"Performance Comparison - {group_name}",
            log_y=log_y,
            template="plotly_dark",
            color_discrete_map=COLORS,
        ).show()
    else:
        px.violin(  # type: ignore
            avg_data.to_pandas(),
            y="Time (ms)",
            color="Library",
            title=f"Performance Comparison - {group_name}",
            log_y=log_y,
            violinmode="overlay",
            template="plotly_dark",
            color_discrete_map=COLORS,
        ).show()


def plot_benchmark_results(
    array: NDArray[np.float64],
    manager: BenchmarkManager,
    group_name: StatType,
    time_target: int,
    log_y: bool,
) -> None:
    data: pl.DataFrame = manager.get_perf_for_group(
        array=array,
        group_name=group_name,
        target_time_secs=time_target,
    )
    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="box",
        log_y=log_y,
    )
    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="violins",
        log_y=log_y,
    )
