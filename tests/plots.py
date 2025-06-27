from structs import COLORS, StatType, Library, COLORS_BENCH, BenchmarkConfig
from manager import BenchmarkManager
from numpy.typing import NDArray
import numpy as np
import plotly.express as px
import polars as pl
from typing import Literal


def plot_global_bench(
    manager: BenchmarkManager, config: BenchmarkConfig
) -> None:
    combined_results = manager.get_perf_for_all_groups(config=config)
    bench = (
        combined_results.group_by(["Group", "Library"])
        .agg(pl.col("Time (ms)").mean().alias("avg_time"), maintain_order=True)
        .pivot(values="avg_time", index="Group", on="Library")
        .with_columns(
            [
                (pl.col(Library.BOTTLENECK) - pl.col(Library.RUSTATS)).alias(
                    Library.BN_BENCH
                ),
                (pl.col(Library.NUMBAGG) - pl.col(Library.RUSTATS_PARALLEL)).alias(
                    Library.NBG_BENCH
                ),
                (pl.col(Library.POLARS) - pl.col(Library.RUSTATS_PARALLEL)).alias(
                    Library.PL_BENCH
                ),
            ]
        )
        .unpivot(
            on=[Library.BN_BENCH, Library.NBG_BENCH, Library.PL_BENCH],
            index="Group",
            value_name="Diff",
            variable_name="Comparison",
        )
    )
    px.histogram(  # type: ignore
        combined_results.to_pandas(),
        x="Group",
        y="Time (ms)",
        color="Library",
        barmode="group",
        title="Log Histogram of Average Execution Times for All Groups",
        template="plotly_dark",
        log_y=True,
        color_discrete_map=COLORS,
        histfunc="avg",
    ).show()

    px.bar(  # type: ignore
        bench.to_pandas(),
        x="Group",
        y="Diff",
        color="Comparison",
        barmode="group",
        title="Benchmark Comparisons (Difference in ms). Higher is better.",
        template="plotly_dark",
        color_discrete_map=COLORS_BENCH,
    ).show()

def plot_group_bench(
    avg_data: pl.DataFrame,
    group_name: StatType,
    kind: Literal["box", "violins", "line"],
    limit: float,
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
                template="plotly_dark",
                color_discrete_map=COLORS,
            ).show()
        case "violins":
            px.violin(  # type: ignore
                distribution_data.to_pandas(),
                y="Time (ms)",
                color="Library",
                title=f"Performance Comparison - {group_name}",
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
                template="plotly_dark",
                color_discrete_map=COLORS,
            ).show()


def plot_check(
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


def plot_benchmark_results(
    config: BenchmarkConfig,
    manager: BenchmarkManager,
    group_name: StatType,
) -> None:
    data: pl.DataFrame = manager.get_perf_for_group(
        config=config,
        group_name=group_name,
    )
    plot_group_bench(
        avg_data=data, group_name=group_name, kind="box", limit=config.limit
    )
    plot_group_bench(
        avg_data=data, group_name=group_name, kind="violins", limit=config.limit
    )

    plot_group_bench(
        avg_data=data, group_name=group_name, kind="line", limit=config.limit
    )
