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
    kind: Literal["box", "violins", "line"],
    log_y: bool,
    limit: int,
) -> None:
    quantile_limit = limit / 100
    distribution_data = avg_data.join(
        avg_data.group_by("Library").agg(
            pl.col("Time (ms)").quantile(quantile_limit).alias("limit")
        ),
        on="Library",
    ).filter(pl.col("Time (ms)") <= pl.col("limit"))
    line_data = avg_data.with_columns(
        pl.arange(0, avg_data.height, 1).alias("Iteration")
    )
    match kind:
        case "box":
            px.box(  # type: ignore
                distribution_data.to_pandas(),
                y="Time (ms)",
                color="Library",
                points=False,
                title=f"Performance Comparison - {group_name}",
                log_y=log_y,
                template="plotly_dark",
                color_discrete_map=COLORS,
            ).show()
        case "violins":
            px.violin(  # type: ignore
                distribution_data.to_pandas(),
                y="Time (ms)",
                color="Library",
                title=f"Performance Comparison - {group_name}",
                log_y=log_y,
                violinmode="overlay",
                template="plotly_dark",
                color_discrete_map=COLORS,
            ).show()
        case "line":
            px.line(  # type: ignore
                line_data.to_pandas(),
                x="Iteration",
                y="Time (ms)",
                color="Library",
                title=f"Performance Comparison - {group_name} (Line Plot)",
                log_y=log_y,
                template="plotly_dark",
                color_discrete_map=COLORS,
            ).show()


def plot_benchmark_results(
    array: NDArray[np.float64],
    manager: BenchmarkManager,
    group_name: StatType,
    time_target: int,
    log_y: bool,
    limit: int,
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
        limit=limit,
    )
    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="violins",
        log_y=log_y,
        limit=limit,
    )

    plot_group_result(
        avg_data=data,
        group_name=group_name,
        kind="line",
        log_y=log_y,
        limit=limit,
    )
