import time
import sys
import threading
import numpy as np
from dataclasses import dataclass
from importlib import metadata
from numpy.typing import NDArray
import polars as pl

TIME_DEFAULT = 30


def set_time_target() -> int:
    time_input: str = input(
        f"write the time target in seconds, press enter for {TIME_DEFAULT} seconds default> "
    ).strip()
    if not time_input == "":
        return int(time_input)
    else:
        return TIME_DEFAULT


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

    def _run(self) -> None:
        while not self._stop_event.is_set():
            elapsed: float = time.time() - self._start_time
            left: float = max(0, self.time_target - elapsed)
            sys.stdout.write(f"\r⏱️  Elapsed: {elapsed:.1f}s | Left: {left:.1f}s")
            sys.stdout.flush()
            if elapsed >= self.time_target:
                print("\nDone!")
                break
            time.sleep(1)


@dataclass(slots=True)
class BenchmarkConfig:
    array: NDArray[np.float64]
    df: pl.DataFrame
    min_length: int = 25
    length: int = 250
    axis: int = 0
    limit: float = 0.95

    @property
    def version(self) -> int:
        version_str: str = metadata.version("rustats")
        return int(version_str.split(".")[-1])
