from groups import ROLLING_FUNCS, StatType
import stats as st
import polars as pl
from plots import (
    BenchmarkManager,
    plot_global_bench,
    plot_benchmark_results,
    plot_check,
)
from structs import Files, BenchmarkConfig


def main(manager: BenchmarkManager, config: BenchmarkConfig) -> None:
    while True:
        _display_menu()
        choice: str = input("Enter your choice (1-4)> ").strip()
        match choice:
            case "1":
                config.set_time_target()
                plot_global_bench(manager=manager, config=config)
            case "2":
                group_name: str = _get_group_name(manager=manager)
                config.set_time_target()
                plot_benchmark_results(
                    config=config,
                    manager=manager,
                    group_name=group_name,
                )
            case "3":
                group_name = _get_group_name(manager=manager)
                plot_check(config=config, manager=manager, group_name=group_name)
            case "4":
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")


def _get_group_name(manager: BenchmarkManager) -> StatType:
    group_name: str = input("Enter the group to test> ").strip()
    if group_name not in manager.groups:
        print(f"Group '{group_name}' not found.")
        return _get_group_name(manager=manager)
    else:
        return StatType[group_name]


def _display_menu() -> None:
    print("\n--- Menu ---")
    print("1. Perform a global performance test for all groups")
    print("2. Test performance for a specific group")
    print("3. Check results for a specific group")
    print("4. Exit")


if __name__ == "__main__":
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    array = st.get_array(pl.read_parquet(source=Files.PRICES))
    config = BenchmarkConfig(array=array, df=pl.from_numpy(array))

    main(manager=rolling, config=config)
