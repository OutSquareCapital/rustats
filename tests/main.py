from groups import AGG_FUNCS, ROLLING_FUNCS
from plots import plot_benchmark_results, plot_function_results
from manager import BenchmarkManager
import polars as pl
from numpy.typing import NDArray
import numpy as np
from structs import Files


def _get_array(df: pl.DataFrame) -> NDArray[np.float64]:
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


if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    agg = BenchmarkManager(groups=AGG_FUNCS)
    import plotly.io as pio

    array = _get_array(pl.read_parquet(source=Files.PRICES))

    pio.renderers.default = "browser"  # type: ignore
    while True:
        group_name: str = input("enter the group to test: ").strip()
        if group_name not in rolling.groups:
            print(f"Group '{group_name}' not found in rolling functions.")
            continue
        time_input: str = input(
            "enter the time target in seconds(default 20 seconds):"
        ).strip()
        if time_input == "":
            time_target = 20
        else:
            time_target = int(time_input)
        plot_benchmark_results(
            array=array,
            manager=rolling,
            group_name=group_name,
            time_target=time_target,
            log_y=False,
        )
        results = {
            func.library: func(array) for func in rolling.groups[group_name].funcs
        }
        plot_function_results(results=results, group_name=group_name)
