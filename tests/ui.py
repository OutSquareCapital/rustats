from manager import get_array
import polars as pl
from plots import (
    BenchmarkManager,
    plot_global_bench,
    plot_benchmark_results,
    plot_check,
)
from structs import Files, BenchmarkConfig


def display_menu() -> None:
    print("\n--- Menu ---")
    print("1. Perform a global performance test for all groups")
    print("2. Test performance for a specific group")
    print("3. Check results for a specific group")
    print("4. Exit")


def main(manager: BenchmarkManager) -> None:
    while True:
        array = get_array(pl.read_parquet(source=Files.PRICES))
        config = BenchmarkConfig(array=array, df=pl.from_numpy(array))
        display_menu()
        choice: str = input("Enter your choice (1-4)> ").strip()
        match choice:
            case "1":
                config.set_time_target()
                plot_global_bench(manager=manager, config=config)
            case "2":
                group_name: str = input("Enter the group to test> ").strip()
                if group_name not in manager.groups:
                    print(f"Group '{group_name}' not found in rolling functions.")
                    continue

                config.set_time_target()
                plot_benchmark_results(
                    config=config, manager=manager, group_name=group_name
                )
            case "3":
                group_name: str = input(
                    "Enter the group to check results for> "
                ).strip()
                if group_name not in manager.groups:
                    print(f"Group '{group_name}' not found in rolling functions.")
                    continue
                results = {
                    func.library: func(config)
                    for func in manager.groups[group_name].funcs
                }
                plot_check(results=results, group_name=group_name)
            case "4":
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")
