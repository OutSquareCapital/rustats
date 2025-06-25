from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from structs import FuncGroup, Result, Files, StatType


@dataclass(slots=True)
class BenchmarkManager:
    groups: dict[StatType, FuncGroup]

    def get_perf_for_group(
        self,
        array: NDArray[np.float64],
        group_name: StatType,
        target_time_secs: float,
    ) -> pl.DataFrame:
        group = self.groups.get(group_name)
        if not group:
            raise KeyError(f"Group '{group_name}' not found.")
        group.warmup()
        n_passes: int = _get_n_passes(
            target_time_secs=target_time_secs, group_name=group_name
        )

        results: list[Result] = group.time_group(
            group_name=group_name, arr=array, n_passes=n_passes
        )
        _save_total_time(group_name, results, n_passes)
        return _get_formatted_results(results=results)


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


def _get_n_passes(target_time_secs: float, group_name: str) -> int:
    summary_data: pl.DataFrame = pl.read_ndjson(Files.SUMMARY)
    group_data = summary_data.filter(pl.col("group") == group_name)

    if group_data.is_empty():
        return 20
    else:
        avg_time_per_pass = group_data.select(pl.col("time_per_pass_ms")).mean().item()
        return max(1, int((target_time_secs * 1000) / avg_time_per_pass))


def _save_total_time(group_name: str, results: list[Result], n_passes: int) -> None:
    new_data = pl.DataFrame(
        data={
            "group": group_name,
            "total_time_secs": round(sum(r.time for r in results) / 1000, 3),
            "n_passes": n_passes,
            "time_per_pass_ms": round(sum(r.time for r in results) / n_passes, 3),
        }
    )
    summary_data: pl.DataFrame = pl.read_ndjson(Files.SUMMARY)
    updated_data: pl.DataFrame = summary_data.filter(
        pl.col("group") != group_name
    ).vstack(new_data)
    updated_data.write_ndjson(Files.SUMMARY)
