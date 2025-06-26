from groups import AGG_FUNCS, ROLLING_FUNCS
from manager import BenchmarkManager
import polars as pl
from ui import check_results, plot_results, get_array
from structs import Files


if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    agg = BenchmarkManager(groups=AGG_FUNCS)
    import plotly.io as pio

    array = get_array(pl.read_parquet(source=Files.PRICES))

    pio.renderers.default = "browser"  # type: ignore
    while True:
        group_name: str = input("enter the group to test: ").strip()
        if group_name not in rolling.groups:
            print(f"Group '{group_name}' not found in rolling functions.")
            continue
        check_results(manager=rolling, array=array, group_name=group_name)
        plot_results(manager=rolling, array=array, group_name=group_name)
