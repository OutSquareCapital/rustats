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
        results=manager.get_results(config=config, group_name=group_name)
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


def plot_group_bench(
    config: BenchmarkConfig,
    manager: BenchmarkManager,
    group_name: StatType,
    time_target: int,
) -> None:
    results = manager.get_perf_for_group(
        config=config, group_name=group_name, time_target=time_target
    )
    absolute = st.get_absolute_results(
        results=results, config=config, time_target=time_target
    )
    relative = st.get_relative_results(absolute, config=config, time_target=time_target)
    abs_distribution = st.get_data_distribution(df=absolute, limit=config.limit)
    relative_distribution = st.get_data_distribution(
        df=relative, limit=config.limit
    )
    line_data = st.get_line_check(df=absolute, iterations=len(results))
    _plot_group_bench(
        df=abs_distribution, group_name=group_name, kind="box", log_scale=True
    )
    _plot_group_bench(
        df=abs_distribution, group_name=group_name, kind="violins", log_scale=True
    )
    _plot_group_bench(
        df=relative_distribution, group_name=group_name, kind="box", log_scale=False
    )
    _plot_group_bench(
        df=relative_distribution, group_name=group_name, kind="violins", log_scale=False
    )
    _plot_iterations(df=line_data.collect(), group_name=group_name)


def plot_global_bench(
    manager: BenchmarkManager, config: BenchmarkConfig, time_target: int
) -> None:
    time_by_group = int(time_target / len(manager.groups))
    combined_results = manager.get_perf_for_all_groups(
        config=config, time_target=time_by_group
    )
    absolute = st.get_absolute_results(
        results=combined_results, config=config, time_target=time_by_group
    )
    relative = st.get_relative_results(
        absolute, config=config, time_target=time_by_group
    )
    px.histogram(  # type: ignore
        absolute,
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

    px.bar(  # type: ignore
        relative,
        x=ColNames.GROUP,
        y=ColNames.TIME_MS,
        color=ColNames.LIBRARY,
        barmode="group",
        title="Benchmark Comparisons (Difference in ms). Higher is better.",
        template=Colors.TEMPLATE,
        color_discrete_map=Colors.RELATIVE,
    ).show()


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


def _plot_group_bench(
    df: pl.DataFrame,
    group_name: StatType,
    kind: Literal["box", "violins"],
    log_scale: bool,
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
                log_y=log_scale,
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
                log_y=log_scale,
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
