import rustats as rs
import polars as pl
from pathlib import Path


def path_src():
    return (
        Path(__file__)
        .parent.joinpath("data")
        .joinpath("prices")
        .with_suffix(".parquet")
    )


def test():
    """
    shape: (100, 2)
    ┌────────┬───────────┐
    │ ticker ┆ close     │
    │ ---    ┆ ---       │
    │ enum   ┆ f64       │
    ╞════════╪═══════════╡
    │ SPY    ┆ null      │
    │ SPY    ┆ 24.991655 │
    │ SPY    ┆ 25.003249 │
    │ SPY    ┆ 25.032235 │
    │ SPY    ┆ 24.96847  │
    │ …      ┆ …         │
    │ SPY    ┆ 25.14584  │
    │ SPY    ┆ 25.099941 │
    │ SPY    ┆ 25.047145 │
    │ SPY    ┆ 25.088207 │
    │ SPY    ┆ 25.117536 │
    └────────┴───────────┘
    """
    return (
        pl.scan_parquet(path_src())
        .head(100)
        .select(
            pl.col("ticker"),
            pl.col("close")
            .cast(pl.Float64)
            .map_batches(
                lambda x: rs.move_mean(x.to_numpy().reshape(-1, 1), 3, 2, True),
                pl.List(pl.Float64),
            )
            .over(pl.col("ticker"))
            .list.explode()
            .fill_nan(None),
        )
    )


if __name__ == "__main__":
    print(test().collect())
