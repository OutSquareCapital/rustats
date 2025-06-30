from typing import Literal

import plotly.express as px
import polars as pl
import stats as st
from config import BenchmarkConfig
from manager import BenchmarkManager
from structs import (
    ColNames,
    Colors,
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
    px.line(  # type: ignore
        df,
        x="Index",
        y="Values",
        color=ColNames.LIBRARY,
        title=f"Results Check - {group_name}",
        template=Colors.TEMPLATE,
        color_discrete_map=Colors.ABSOLUTE,
    ).show()


def plot_benchmark_results(
    config: BenchmarkConfig,
    manager: BenchmarkManager,
    group_name: StatType,
    time_target: int,
) -> None:
    results = manager.get_perf_for_group(
        config=config, group_name=group_name, time_target=time_target
    )
    avg_data = st.get_formatted_results(results=results)
    distribution_data = st.get_data_distribution(df=avg_data, limit=config.limit)
    line_data = avg_data.with_columns(pl.arange(0, len(results), 1).alias("Iteration"))
    _plot_group_bench(df=distribution_data, group_name=group_name, kind="box")
    _plot_group_bench(df=distribution_data, group_name=group_name, kind="violins")
    _plot_iterations(df=line_data, group_name=group_name)


def plot_global_bench(
    manager: BenchmarkManager, config: BenchmarkConfig, time_target: int
) -> None:
    time_by_group = int(time_target / len(manager.groups))
    combined_results = manager.get_perf_for_all_groups(
        config=config, time_target=time_by_group
    )
    df = st.get_formatted_results(results=combined_results)
    bench = st.get_time_relative(df)
    st.save_history(
        df=df, config=config, file=Files.BENCH_HISTORY, time_target=time_by_group
    )
    st.save_history(
        df=bench, config=config, file=Files.RELATIVE_HISTORY, time_target=time_by_group
    )
    _plot_absolute_results(df=df)
    _plot_relative_results(df=bench)


def plot_3d_history(file: Files, log_scale: bool) -> None:
    px.line_3d(  # type: ignore
        st.get_perf_across_versions(file=file),
        x=ColNames.GROUP,
        y=ColNames.VERSION,
        z=ColNames.MEDIAN_TIME,
        color=ColNames.FUNC_LIB,
        line_group=ColNames.FUNC_LIB,
        title="3D Line Plot of Benchmark History",
        log_z=log_scale,
        template=Colors.TEMPLATE,
    ).show()


def plot_2d_history(file: Files, log_scale: bool, group: StatType) -> None:
    px.line(  # type: ignore
        st.get_perf_across_versions(file=file, group=group),
        x=ColNames.VERSION,
        y=ColNames.MEDIAN_TIME,
        color=ColNames.LIBRARY,
        title=f"Line Plot of {group} Benchmark History",
        log_y=log_scale,
        template=Colors.TEMPLATE,
    ).show()


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
