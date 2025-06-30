from groups import ROLLING_FUNCS, StatType
import stats as st
import polars as pl
import plots as pt
from structs import Files
from config import BenchmarkConfig, ChronoBar, set_time_target
from manager import BenchmarkManager


def main(manager: BenchmarkManager, config: BenchmarkConfig) -> None:
    while True:
        _display_menu()
        choice: str = input("Enter your choice> ").strip()
        match choice:
            case "1":
                time_target = set_time_target()
                bar = ChronoBar(time_target=time_target)
                bar.start()
                pt.plot_global_bench(
                    manager=manager, config=config, time_target=time_target
                )
                bar.stop()
            case "2":
                group_name: str = _get_group_name(manager=manager)
                time_target: int = set_time_target()
                bar = ChronoBar(time_target=time_target)
                bar.start()
                pt.plot_group_bench(
                    config=config,
                    manager=manager,
                    group_name=group_name,
                    time_target=time_target,
                )
                bar.stop()
            case "3":
                group_name = _get_group_name(manager=manager)
                pt.plot_check(config=config, manager=manager, group_name=group_name)
            case "4":
                pt.plot_3d_history(file=Files.BENCH_HISTORY, log_scale=True)
                pt.plot_3d_history(file=Files.RELATIVE_HISTORY, log_scale=False)
            case "5":
                group_name = _get_group_name(manager=manager)
                pt.plot_2d_history(
                    file=Files.BENCH_HISTORY, log_scale=True, group=group_name
                )
                pt.plot_2d_history(
                    file=Files.RELATIVE_HISTORY, log_scale=False, group=group_name
                )
            case "6":
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
        return StatType[group_name.upper()]


def _display_menu() -> None:
    print("\n--- Menu ---")
    print("1. Perform a global performance test for all groups")
    print("2. Test performance for a specific group")
    print("3. Check results for a specific group")
    print("4. View global performance history across versions")
    print("5. View specific group performance history across versions")
    print("6. Exit")


if __name__ == "__main__":
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    array = st.get_array(Files.PRICES)
    config = BenchmarkConfig(array=array, df=pl.from_numpy(array))

    main(manager=rolling, config=config)
