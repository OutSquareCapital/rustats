import polars as pl
from structs import Length
from functools import partial
import bottleneck as bn
import numbagg as nbg
import rustats as rs
from structs import PlFunc, BnFunc, NbgFunc, RSingleFunc, RParallelFunc


class PolarsFunc:
    kurt = PlFunc(
        func=pl.all().rolling_kurtosis(
            window_size=Length.FULL, min_samples=Length.MIN, bias=False
        ),
    )

    skew = PlFunc(
        func=pl.all().rolling_skew(
            window_size=Length.FULL, min_samples=Length.MIN, bias=False
        ),
    )

    mean = PlFunc(
        func=pl.all().rolling_mean(window_size=Length.FULL, min_samples=Length.MIN),
    )

    sum = PlFunc(
        func=pl.all().rolling_sum(window_size=Length.FULL, min_samples=Length.MIN),
    )

    var = PlFunc(
        func=pl.all().rolling_var(
            window_size=Length.FULL, min_samples=Length.MIN, ddof=1
        ),
    )

    std = PlFunc(
        func=pl.all().rolling_std(
            window_size=Length.FULL, min_samples=Length.MIN, ddof=1
        ),
    )

    max = PlFunc(
        func=pl.all().rolling_max(window_size=Length.FULL, min_samples=Length.MIN),
    )

    min = PlFunc(
        func=pl.all().rolling_min(window_size=Length.FULL, min_samples=Length.MIN),
    )

    median = PlFunc(
        func=pl.all().rolling_median(window_size=Length.FULL, min_samples=Length.MIN),
    )


class BottleneckFuncs:
    mean = BnFunc(bn.move_mean)
    sum = BnFunc(bn.move_sum)
    var = BnFunc(partial(bn.move_var, ddof=1))
    std = BnFunc(partial(bn.move_std, ddof=1))
    max = BnFunc(bn.move_max)
    min = BnFunc(bn.move_min)
    median = BnFunc(bn.move_median)
    rank = BnFunc(bn.move_rank)


class NumbaggFuncs:
    mean = NbgFunc(
        partial(nbg.move_mean, window=Length.FULL, min_count=Length.MIN, axis=0)
    )
    sum = NbgFunc(
        partial(nbg.move_sum, window=Length.FULL, min_count=Length.MIN, axis=0)
    )
    var = NbgFunc(
        partial(nbg.move_var, window=Length.FULL, min_count=Length.MIN, axis=0)
    )
    std = NbgFunc(
        partial(nbg.move_std, window=Length.FULL, min_count=Length.MIN, axis=0)
    )


class RustatSingleFuncs:
    mean = RSingleFunc(rs.move_mean)
    sum = RSingleFunc(rs.move_sum)
    var = RSingleFunc(rs.move_var)
    std = RSingleFunc(rs.move_std)
    max = RSingleFunc(rs.move_max)
    min = RSingleFunc(rs.move_min)
    median = RSingleFunc(rs.move_median)
    rank = RSingleFunc(rs.move_rank)
    skew = RSingleFunc(rs.move_skewness)
    kurt = RSingleFunc(rs.move_kurtosis)


class RustatParallelFuncs:
    mean = RParallelFunc(rs.move_mean)
    sum = RParallelFunc(rs.move_sum)
    var = RParallelFunc(rs.move_var)
    std = RParallelFunc(rs.move_std)
    max = RParallelFunc(rs.move_max)
    min = RParallelFunc(rs.move_min)
    median = RParallelFunc(rs.move_median)
    rank = RParallelFunc(rs.move_rank)
    skew = RParallelFunc(rs.move_skewness)
    kurt = RParallelFunc(rs.move_kurtosis)
