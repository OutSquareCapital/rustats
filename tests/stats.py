import numpy as np
import polars as pl
from numpy.typing import NDArray
from structs import (
    Schemas,
    BenchmarkConfig,
    ColNames,
    Files,
    Library,
    Result,
    StatType,
)


def get_n_passes(time_target: int, group_name: StatType) -> int:
    group_data = pl.read_ndjson(Files.PASSES, schema=Schemas.PASSES).filter(
        pl.col(ColNames.GROUP) == group_name
    )

    if group_data.is_empty():
        return 20
    else:
        avg_time_per_pass = group_data.select(pl.col("time_per_pass_ms")).mean().item()
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


def get_formatted_results(results: list[Result]) -> pl.DataFrame:
    return pl.DataFrame(
        data={
            ColNames.LIBRARY: [r.library for r in results],
            ColNames.GROUP: [r.group for r in results],
            ColNames.TIME_MS: [r.time for r in results],
        },
        orient="row",
        schema=Schemas.RESULT,
    )


def save_group_time(
    group_name: StatType, results: list[Result], n_passes: int, config: BenchmarkConfig
) -> None:
    new_data = pl.DataFrame(
        data={
            ColNames.GROUP: group_name,
            ColNames.VERSION: config.version,
            ColNames.TIME_TARGET: config.time_target,
            "total_time_secs": round(sum(r.time for r in results) / 1000, 3),
            "n_passes": n_passes,
            "time_per_pass_ms": round(sum(r.time for r in results) / n_passes, 3),
        },
        schema=Schemas.PASSES,
    )

    pl.read_ndjson(Files.PASSES, schema=Schemas.PASSES).filter(
        ~(
            (pl.col(ColNames.GROUP) == group_name)
            & (
                (pl.col(ColNames.VERSION) < config.version)
                | (
                    (pl.col(ColNames.VERSION) == config.version)
                    & (pl.col(ColNames.TIME_TARGET) <= config.time_target)
                )
            )
        )
    ).extend(new_data).sort(by=ColNames.GROUP).write_ndjson(Files.PASSES)


def get_data_check(results: dict[Library, NDArray[np.float64]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            ColNames.LIBRARY: [
                lib for lib in results.keys() for _ in range(results[lib].shape[0])
            ],
            "Index": [
                i for lib in results.keys() for i in range(results[lib].shape[0])
            ],
            "Values": [value for lib in results.keys() for value in results[lib][:, 0]],
        }
    )


def get_data_distribution(df: pl.DataFrame, limit: float) -> pl.DataFrame:
    quantile_limit: float = limit / 100
    return (
        df.join(
            df.group_by(ColNames.LIBRARY).agg(
                pl.col(ColNames.TIME_MS).quantile(quantile_limit).alias("limit")
            ),
            on=ColNames.LIBRARY,
        )
        .filter(pl.col(ColNames.TIME_MS) <= pl.col("limit"))
        .sort(by=[ColNames.GROUP, ColNames.LIBRARY])
    )


def save_history(df: pl.DataFrame, config: BenchmarkConfig, file: str) -> None:
    current_data = pl.read_ndjson(file, schema=Schemas.HISTORY)

    new_data = get_time_results(df, config)

    data_to_add = (
        new_data.join(
            current_data,
            on=[ColNames.GROUP, ColNames.LIBRARY],
            how="left",
            suffix="_old",
        )
        .filter(
            pl.col(ColNames.VERSION + "_old").is_null()
            | (pl.col(ColNames.VERSION) != pl.col(ColNames.VERSION + "_old"))
            | (
                (pl.col(ColNames.VERSION) == pl.col(ColNames.VERSION + "_old"))
                & (
                    pl.col(ColNames.TIME_TARGET)
                    >= pl.col(ColNames.TIME_TARGET + "_old")
                )
            )
        )
        .select(new_data.columns)
    )

    if data_to_add.is_empty():
        return
    to_remove = data_to_add.join(
        current_data, on=[ColNames.GROUP, ColNames.LIBRARY, ColNames.VERSION]
    ).filter(pl.col(ColNames.TIME_TARGET) >= pl.col(ColNames.TIME_TARGET + "_right"))

    if not to_remove.is_empty():
        current_data = current_data.join(
            to_remove.select([ColNames.GROUP, ColNames.LIBRARY, ColNames.VERSION]),
            on=[ColNames.GROUP, ColNames.LIBRARY, ColNames.VERSION],
            how="anti",
        )

    pl.concat([current_data, data_to_add]).sort(
        by=[ColNames.VERSION, ColNames.GROUP, ColNames.LIBRARY]
    ).write_ndjson(file)


def get_time_results(df: pl.DataFrame, config: BenchmarkConfig) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.lit(value=config.version, dtype=pl.Int32).alias(ColNames.VERSION),
            pl.lit(value=config.time_target, dtype=pl.Int32).alias(
                ColNames.TIME_TARGET
            ),
        )
        .group_by(
            [ColNames.GROUP, ColNames.LIBRARY, ColNames.VERSION, ColNames.TIME_TARGET]
        )
        .agg(pl.col(ColNames.TIME_MS).median().round(2).alias(ColNames.MEDIAN_TIME))
        .drop_nulls()
        .sort(by=[ColNames.GROUP, ColNames.LIBRARY])
    )


def get_time_relative(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by([ColNames.GROUP, ColNames.LIBRARY])
        .agg(pl.col(ColNames.TIME_MS).mean().alias("avg_time"), maintain_order=True)
        .pivot(values="avg_time", index=ColNames.GROUP, on=ColNames.LIBRARY)
        .with_columns(
            [
                (pl.col(name=Library.BOTTLENECK).sub(other=Library.RUSTATS)).alias(
                    name=Library.BN_BENCH
                ),
                (
                    pl.col(name=Library.NUMBAGG).sub(other=Library.RUSTATS_PARALLEL)
                ).alias(name=Library.NBG_BENCH),
                (pl.col(name=Library.POLARS).sub(other=Library.RUSTATS_PARALLEL)).alias(
                    name=Library.PL_BENCH
                ),
            ]
        )
        .unpivot(
            on=[Library.BN_BENCH, Library.NBG_BENCH, Library.PL_BENCH],
            index=ColNames.GROUP,
            value_name=ColNames.TIME_MS,
            variable_name=ColNames.LIBRARY,
        )
        .with_columns(pl.col(ColNames.LIBRARY).cast(Schemas.library_enum))
    ).sort(by=[ColNames.GROUP, ColNames.LIBRARY])
