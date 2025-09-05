import rustats as rs
import numpy as np
from pathlib import Path

SRC: Path = Path().joinpath("data").joinpath("prices").with_suffix(".parquet")
print(SRC.exists())
print(
    rs.move_mean(
        np.random.rand(10).reshape(-1, 1).astype(dtype=np.float32),
        length=3,
        min_length=1,
        parallel=True,
    )
)
