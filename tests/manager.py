from dataclasses import dataclass
from time import perf_counter

import stats as st
from funcs import StatFuncProtocol
from structs import BenchmarkConfig, Result, StatType
import time
import sys
import threading


class ChronoBar:
    def __init__(self, time_target: int) -> None:
        self.time_target: int = time_target
        self._start_time: float = 0.0
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        print()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            elapsed: float = time.time() - self._start_time
            left: float = max(0, self.time_target - elapsed)
            sys.stdout.write(f"\r⏱️  Elapsed: {elapsed:.1f}s | Left: {left:.1f}s")
            sys.stdout.flush()
            if elapsed >= self.time_target:
                break
            time.sleep(1)


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
        bar = ChronoBar(time_target=time_target)
        bar.start()
        results: list[Result] = group.get_perf(
            n_passes=n_passes, config=config, group_name=group_name
        )
        bar.stop()
        st.save_group_time(
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
        time_by_group = int(time_target / len(self.groups))
        bar = ChronoBar(time_target=time_target)
        bar.start()
        for group_name in self.groups.keys():
            results = self.get_perf_for_group(
                config=config, group_name=group_name, time_target=time_by_group
            )
            combined_results.extend(results)
        bar.stop()
        return combined_results
