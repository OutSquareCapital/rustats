from functools import partial

import bottleneck as bn
import numbagg as nbg
import rustats as rs

from structs import FuncGroup, Library, StatFunc, StatType, Length

ROLLING_FUNCS: dict[StatType, FuncGroup] = {
    "mean": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_mean, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_mean,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_mean,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(
                    nbg.move_mean, window=Length.FULL, min_count=Length.MIN, axis=0
                ),
            ),
        ],
    ),
    "sum": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_sum, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_sum,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_sum,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_sum, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
        ],
    ),
    "var": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(
                    bn.move_var,
                    window=Length.FULL,
                    min_count=Length.MIN,
                    axis=0,
                    ddof=1,
                ),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_var,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_var,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_var, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
        ],
    ),
    "std": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(
                    bn.move_std,
                    window=Length.FULL,
                    min_count=Length.MIN,
                    axis=0,
                    ddof=1,
                ),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_std,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_std,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.move_std, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
        ],
    ),
    "max": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_max, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_max,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_max,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
        ],
    ),
    "min": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_min, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_min,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_min,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
        ],
    ),
    "median": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(
                    bn.move_median, window=Length.FULL, min_count=Length.MIN, axis=0
                ),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_median,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_median,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
        ],
    ),
    "rank": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.move_rank, window=Length.FULL, min_count=Length.MIN, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(
                    rs.move_rank,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=False,
                ),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(
                    rs.move_rank,
                    length=Length.FULL,
                    min_length=Length.MIN,
                    parallel=True,
                ),
            ),
        ],
    ),
}

AGG_FUNCS: dict[StatType, FuncGroup] = {
    "mean": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmean, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_mean, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_mean, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanmean, axis=0),
            ),
        ],
    ),
    "sum": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nansum, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_sum, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_sum, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nansum, axis=0),
            ),
        ],
    ),
    "var": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanvar, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_var, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_var, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanvar, axis=0),
            ),
        ],
    ),
    "std": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanstd, axis=0, ddof=1),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_std, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_std, parallel=True),
            ),
            StatFunc(
                Library.NUMBAGG,
                partial(nbg.nanstd, axis=0),
            ),
        ],
    ),
    "max": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmax, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_max, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_max, parallel=True),
            ),
        ],
    ),
    "min": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmin, axis=0),
            ),
            StatFunc(
                Library.RUSTATS,
                partial(rs.agg_min, parallel=False),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_min, parallel=True),
            ),
        ],
    ),
    "median": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.nanmedian, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_median),
            ),
        ],
    ),
    "rank": FuncGroup(
        funcs=[
            StatFunc(
                Library.BOTTLENECK,
                partial(bn.rankdata, axis=0),
            ),
            StatFunc(
                Library.RUSTATS_PARALLEL,
                partial(rs.agg_rank),
            ),
        ],
    ),
}
