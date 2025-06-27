from manager import get_array
import polars as pl
from plots import (
    BenchmarkManager,
    plot_histograms_for_all_groups,
    plot_benchmark_results,
    plot_function_results,
)
from structs import Files


def display_menu() -> None:
    print("\n--- Menu ---")
    print("1. Perform a global performance test for all groups")
    print("2. Test performance for a specific group")
    print("3. Check results for a specific group")
    print("4. Exit")


def get_time_target() -> int:
    time_input: str = input(
        "write the time target in seconds, press enter for 20 seconds default>"
    ).strip()
    if time_input == "":
        return 20
    else:
        return int(time_input)


def main(manager: BenchmarkManager) -> None:
    while True:
        array = get_array(pl.read_parquet(source=Files.PRICES))
        display_menu()
        choice: str = input("Enter your choice (1-4)> ").strip()
        match choice:
            case "1":
                plot_histograms_for_all_groups(
                    manager=manager, array=array, time_target=get_time_target()
                )
            case "2":
                group_name: str = input("Enter the group to test> ").strip()
                if group_name not in manager.groups:
                    print(f"Group '{group_name}' not found in rolling functions.")
                    continue
                plot_benchmark_results(
                    array=array,
                    manager=manager,
                    group_name=group_name,
                    time_target=get_time_target(),
                    limit=95,
                )
            case "3":
                group_name: str = input(
                    "Enter the group to check results for> "
                ).strip()
                if group_name not in manager.groups:
                    print(f"Group '{group_name}' not found in rolling functions.")
                    continue
                results = {
                    func.library: func(array)
                    for func in manager.groups[group_name].funcs
                }
                plot_function_results(results=results, group_name=group_name)
            case "4":
                print("Exiting...")
                break
            case _:
                print("Invalid choice. Please try again.")
