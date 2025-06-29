from typing import Literal

import plotly.express as px
import polars as pl

import stats as st
from manager import BenchmarkManager
from structs import (
    Colors,
    BenchmarkConfig,
    ColNames,
    Files,
    StatType,
)


def plot_check(
    config: BenchmarkConfig,
    manager: BenchmarkManager,
    group_name: StatType,
) -> None:
    df = st.get_data_check(
        results={
            func.library: func(config) for func in manager.groups[group_name].funcs
        }
    )
    _plot_func_result(df=df, group_name=group_name)


def plot_benchmark_results(
    config: BenchmarkConfig, manager: BenchmarkManager, group_name: StatType
) -> None:
    avg_data: pl.DataFrame = manager.get_perf_for_group(
        config=config, group_name=group_name
    )
    distribution_data = st.get_data_distribution(df=avg_data, limit=config.limit)
    line_data = avg_data.with_columns(
        pl.arange(0, avg_data.height, 1).alias("Iteration")
    )
    _plot_group_bench(df=distribution_data, group_name=group_name, kind="box")
    _plot_group_bench(df=distribution_data, group_name=group_name, kind="violins")
    _plot_iterations(df=line_data, group_name=group_name)


def plot_global_bench(manager: BenchmarkManager, config: BenchmarkConfig) -> None:
    combined_results = manager.get_perf_for_all_groups(config=config)
    bench = st.get_time_diff(combined_results)
    st.save_time_results(df=combined_results, config=config, file=Files.BENCH_HISTORY)
    st.save_time_results(df=bench, config=config, file=Files.RELATIVE_HISTORY)
    _plot_absolute_results(df=combined_results)
    _plot_relative_results(df=bench)


def _plot_absolute_results(df: pl.DataFrame) -> None:
    px.histogram(  # type: ignore
        df,
        x=ColNames.GROUP,
        y=ColNames.TIME_MS,
        color=ColNames.LIBRARY,
        barmode="group",
        title="Log Histogram of Average Execution Times for All Groups",
        template=Colors.TEMPLATE,
        log_y=True,
        color_discrete_map=Colors.ABSOLUTE,
        histfunc="avg",
    ).show()


def _plot_relative_results(df: pl.DataFrame) -> None:
    px.bar(  # type: ignore
        df,
        x=ColNames.GROUP,
        y=ColNames.TIME_MS,
        color=ColNames.LIBRARY,
        barmode="group",
        title="Benchmark Comparisons (Difference in ms). Higher is better.",
        template=Colors.TEMPLATE,
        color_discrete_map=Colors.RELATIVE,
    ).show()


def _plot_group_bench(
    df: pl.DataFrame,
    group_name: StatType,
    kind: Literal["box", "violins"],
) -> None:
    match kind:
        case "box":
            px.box(  # type: ignore
                df,
                y=ColNames.TIME_MS,
                color=ColNames.LIBRARY,
                points=False,
                title=f"Performance Comparison - {group_name}",
                template=Colors.TEMPLATE,
                color_discrete_map=Colors.ABSOLUTE,
            ).show()
        case "violins":
            px.violin(  # type: ignore
                df,
                y=ColNames.TIME_MS,
                color=ColNames.LIBRARY,
                title=f"Performance Comparison - {group_name}",
                violinmode="overlay",
                template=Colors.TEMPLATE,
                color_discrete_map=Colors.ABSOLUTE,
            ).show()


def _plot_iterations(df: pl.DataFrame, group_name: StatType) -> None:
    px.line(  # type: ignore
        df,
        x="Iteration",
        y=ColNames.TIME_MS,
        color=ColNames.LIBRARY,
        title=f"Performance Comparison - {group_name} (Line Plot)",
        template=Colors.TEMPLATE,
        color_discrete_map=Colors.ABSOLUTE,
    ).show()


def _plot_func_result(
    df: pl.DataFrame,
    group_name: str,
) -> None:
    px.line(  # type: ignore
        df,
        x="Index",
        y="Values",
        color=ColNames.LIBRARY,
        title=f"Results Check - {group_name}",
        template=Colors.TEMPLATE,
        color_discrete_map=Colors.ABSOLUTE,
    ).show()
