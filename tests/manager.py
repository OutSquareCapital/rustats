from dataclasses import dataclass
from time import perf_counter

import polars as pl
from tqdm import tqdm

import stats as st
from funcs import StatFuncProtocol
from structs import Schemas, BenchmarkConfig, Result, StatType


@dataclass(slots=True)
class FuncGroup:
    funcs: list[StatFuncProtocol]

    def warmup(self, config: BenchmarkConfig) -> None:
        for func in self.funcs:
            for _ in range(10):
                func(config)


@dataclass(slots=True)
class BenchmarkManager:
    groups: dict[StatType, FuncGroup]

    def get_perf_for_group(
        self, config: BenchmarkConfig, group_name: StatType
    ) -> pl.DataFrame:
        group = self.groups.get(group_name)
        if not group:
            raise KeyError(f"Group '{group_name}' not found.")
        group.warmup(config=config)
        n_passes: int = st.get_n_passes(
            time_target=config.time_target, group_name=group_name
        )
        total: int = len(group.funcs) * n_passes
        results: list[Result] = []
        with tqdm(total=total, desc=f"Timing {group_name}") as pbar:
            for func in group.funcs:
                for _ in range(n_passes):
                    start_time: float = perf_counter()
                    func(config)
                    elapsed_time: float = (perf_counter() - start_time) * 1000
                    results.append(
                        Result(
                            library=func.library,
                            group=group_name,
                            time=elapsed_time,
                        )
                    )
                    pbar.update(1)

        st.save_group_time(
            group_name=group_name, results=results, n_passes=n_passes, config=config
        )
        return st.get_formatted_results(results=results)

    def get_perf_for_all_groups(self, config: BenchmarkConfig) -> pl.DataFrame:
        combined_results: list[Result] = []
        time_by_group = int(config.time_target / len(self.groups))
        passes: dict[str, int] = {
            group_name: st.get_n_passes(
                time_target=time_by_group, group_name=group_name
            )
            for group_name in self.groups.keys()
        }
        total_passes: int = sum(
            passes[group_name] * len(group.funcs)
            for group_name, group in self.groups.items()
        )
        with tqdm(total=total_passes, desc="Timing all groups") as pbar:
            for group_name, group in self.groups.items():
                group.warmup(config=config)
                n_passes: int = passes[group_name]
                results: list[Result] = []
                for func in group.funcs:
                    for _ in range(n_passes):
                        start_time: float = perf_counter()
                        func(config)
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
            data={
                "Library": [result.library for result in combined_results],
                "Group": [result.group for result in combined_results],
                "Time (ms)": [result.time for result in combined_results],
            },
            schema=Schemas.RESULT,
        )
