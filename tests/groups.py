import funcs as fn
from manager import FuncGroup
from structs import StatType

ROLLING_FUNCS: dict[StatType, FuncGroup] = {
    StatType.MEAN: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.mean,
            fn.RustatSingleFuncs.mean,
            fn.RustatParallelFuncs.mean,
            fn.NumbaggFuncs.mean,
            fn.PolarsFunc.mean,
        ],
    ),
    StatType.SUM: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.sum,
            fn.RustatSingleFuncs.sum,
            fn.RustatParallelFuncs.sum,
            fn.NumbaggFuncs.sum,
            fn.PolarsFunc.sum,
        ],
    ),
    StatType.VAR: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.var,
            fn.RustatSingleFuncs.var,
            fn.RustatParallelFuncs.var,
            fn.NumbaggFuncs.var,
            fn.PolarsFunc.var,
        ],
    ),
    StatType.STD: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.std,
            fn.RustatSingleFuncs.std,
            fn.RustatParallelFuncs.std,
            fn.NumbaggFuncs.std,
            fn.PolarsFunc.std,
        ],
    ),
    StatType.MAX: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.max,
            fn.RustatSingleFuncs.max,
            fn.RustatParallelFuncs.max,
            fn.PolarsFunc.max,
        ],
    ),
    StatType.MIN: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.min,
            fn.RustatSingleFuncs.min,
            fn.RustatParallelFuncs.min,
            fn.PolarsFunc.min,
        ],
    ),
    StatType.MEDIAN: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.median,
            fn.RustatSingleFuncs.median,
            fn.RustatParallelFuncs.median,
            fn.PolarsFunc.median,
        ],
    ),
    StatType.RANK: FuncGroup(
        funcs=[
            fn.BottleneckFuncs.rank,
            fn.RustatSingleFuncs.rank,
            fn.RustatParallelFuncs.rank,
        ],
    ),
    StatType.SKEW: FuncGroup(
        funcs=[
            fn.RustatSingleFuncs.skew,
            fn.RustatParallelFuncs.skew,
            fn.PolarsFunc.skew,
        ],
    ),
    StatType.KURT: FuncGroup(
        funcs=[
            fn.RustatSingleFuncs.kurt,
            fn.RustatParallelFuncs.kurt,
            fn.PolarsFunc.kurt,
        ],
    ),
}
