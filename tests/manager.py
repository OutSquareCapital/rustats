from dataclasses import dataclass
from time import perf_counter

import numpy as np
import stats as st
from config import BenchmarkConfig
from funcs import StatFuncProtocol
from numpy.typing import NDArray
from structs import Library, Result, StatType


@dataclass(slots=True)
class FuncGroup:
    funcs: list[StatFuncProtocol]

    def warmup(self, config: BenchmarkConfig) -> None:
        for func in self.funcs:
            for _ in range(10):
                func(config)

    def get_perf(
        self, n_passes: int, config: BenchmarkConfig, group_name: StatType
    ) -> list[Result]:
        self.warmup(config=config)
        results: list[Result] = []
        for func in self.funcs:
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
        return results


@dataclass(slots=True)
class BenchmarkManager:
    groups: dict[StatType, FuncGroup]

    def get_group(self, group_name: StatType) -> FuncGroup:
        group = self.groups.get(group_name)
        if not group:
            raise KeyError(f"Group '{group_name}' not found.")
        return group

    def get_perf_for_group(
        self, config: BenchmarkConfig, group_name: StatType, time_target: int
    ) -> list[Result]:
        group: FuncGroup = self.get_group(group_name)
        n_passes: int = st.get_n_passes(time_target=time_target, group_name=group_name)
        results: list[Result] = group.get_perf(
            n_passes=n_passes, config=config, group_name=group_name
        )
        st.save_group_passes(
            group_name=group_name,
            results=results,
            n_passes=n_passes,
            config=config,
            time_target=time_target,
        )
        return results

    def get_perf_for_all_groups(
        self, config: BenchmarkConfig, time_target: int
    ) -> list[Result]:
        combined_results: list[Result] = []
        for group_name in self.groups.keys():
            results = self.get_perf_for_group(
                config=config, group_name=group_name, time_target=time_target
            )
            combined_results.extend(results)
        return combined_results

    def get_results(
        self, config: BenchmarkConfig, group_name: StatType
    ) -> dict[Library, NDArray[np.float64]]:
        return {func.library: func(config) for func in self.groups[group_name].funcs}
