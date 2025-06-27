from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from structs import FuncGroup, Result, Files, StatType
from tqdm import tqdm
from time import perf_counter


@dataclass(slots=True)
class BenchmarkManager:
    groups: dict[StatType, FuncGroup]

    def get_perf_for_group(
        self,
        array: NDArray[np.float64],
        group_name: StatType,
        time_target: float,
    ) -> pl.DataFrame:
        group = self.groups.get(group_name)
        if not group:
            raise KeyError(f"Group '{group_name}' not found.")
        group.warmup()
        n_passes: int = self.get_n_passes(
            time_target=time_target, group_name=group_name
        )
        total: int = len(group.funcs) * n_passes
        results: list[Result] = []
        with tqdm(total=total, desc=f"Timing {group_name}") as pbar:
            for func in group.funcs:
                for _ in range(n_passes):
                    start_time: float = perf_counter()
                    func(arr=array)
                    elapsed_time: float = (perf_counter() - start_time) * 1000
                    results.append(
                        Result(
                            library=func.library,
                            group=group_name,
                            time=elapsed_time,
                        )
                    )
                    pbar.update(1)

        _save_total_time(group_name=group_name, results=results, n_passes=n_passes)
        return _get_formatted_results(results=results)

    def get_perf_for_all_groups(
        self, array: NDArray[np.float64], time_target: int
    ) -> pl.DataFrame:
        combined_results: list[Result] = []
        time_by_group = int(time_target / len(self.groups))
        passes: dict[str, int] = {
            group_name: self.get_n_passes(time_target=time_by_group, group_name=group_name)
            for group_name in self.groups.keys()
        }
        total_passes: int = sum(
            passes[group_name] * len(group.funcs)
            for group_name, group in self.groups.items()
        )
        with tqdm(total=total_passes, desc="Timing all groups") as pbar:
            for group_name, group in self.groups.items():
                group.warmup()
                n_passes: int = passes[group_name]
                results: list[Result] = []
                for func in group.funcs:
                    for _ in range(n_passes):
                        start_time: float = perf_counter()
                        func(arr=array)
                        elapsed_time: float = (perf_counter() - start_time) * 1000
                        results.append(
                            Result(
                                library=func.library,
                                group=group_name,
                                time=elapsed_time,
                            )
                        )
                        pbar.update(1)
                combined_results.extend(results)

        return pl.DataFrame(
            {
                "Library": [result.library for result in combined_results],
                "Group": [result.group for result in combined_results],
                "Time (ms)": [result.time for result in combined_results],
            }
        )

    def get_n_passes(self, time_target: float, group_name: str) -> int:
        group_data = pl.read_ndjson(Files.SUMMARY).filter(pl.col("group") == group_name)

        if group_data.is_empty():
            return 20
        else:
            avg_time_per_pass = (
                group_data.select(pl.col("time_per_pass_ms")).mean().item()
            )
            return max(1, int((time_target * 1000) / avg_time_per_pass))


def get_array(df: pl.DataFrame) -> NDArray[np.float64]:
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


def _get_formatted_results(results: list[Result]) -> pl.DataFrame:
    return pl.DataFrame(
        data={
            "Library": [r.library for r in results],
            "Group": [r.group for r in results],
            "Time (ms)": [r.time for r in results],
        },
        schema=["Library", "Group", "Time (ms)"],
        orient="row",
    )


def _save_total_time(group_name: str, results: list[Result], n_passes: int) -> None:
    new_data = pl.DataFrame(
        data={
            "group": group_name,
            "total_time_secs": round(sum(r.time for r in results) / 1000, 3),
            "n_passes": n_passes,
            "time_per_pass_ms": round(sum(r.time for r in results) / n_passes, 3),
        }
    )

    pl.read_ndjson(Files.SUMMARY).filter(pl.col("group") != group_name).vstack(
        new_data
    ).write_ndjson(Files.SUMMARY)
