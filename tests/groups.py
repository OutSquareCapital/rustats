from structs import FuncGroup, StatType
import funcs as fn

ROLLING_FUNCS: dict[StatType, FuncGroup] = {
    "mean": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.mean,
            fn.RustatSingleFuncs.mean,
            fn.RustatParallelFuncs.mean,
            fn.NumbaggFuncs.mean,
            fn.PolarsFunc.mean,
        ],
    ),
    "sum": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.sum,
            fn.RustatSingleFuncs.sum,
            fn.RustatParallelFuncs.sum,
            fn.NumbaggFuncs.sum,
            fn.PolarsFunc.sum,
        ],
    ),
    "var": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.var,
            fn.RustatSingleFuncs.var,
            fn.RustatParallelFuncs.var,
            fn.NumbaggFuncs.var,
            fn.PolarsFunc.var,
        ],
    ),
    "std": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.std,
            fn.RustatSingleFuncs.std,
            fn.RustatParallelFuncs.std,
            fn.NumbaggFuncs.std,
            fn.PolarsFunc.std,
        ],
    ),
    "max": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.max,
            fn.RustatSingleFuncs.max,
            fn.RustatParallelFuncs.max,
            fn.PolarsFunc.max,
        ],
    ),
    "min": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.min,
            fn.RustatSingleFuncs.min,
            fn.RustatParallelFuncs.min,
            fn.PolarsFunc.min,
        ],
    ),
    "median": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.median,
            fn.RustatSingleFuncs.median,
            fn.RustatParallelFuncs.median,
            fn.PolarsFunc.median,
        ],
    ),
    "rank": FuncGroup(
        funcs=[
            fn.BottleneckFuncs.rank,
            fn.RustatSingleFuncs.rank,
            fn.RustatParallelFuncs.rank,
        ],
    ),
    "skew": FuncGroup(
        funcs=[
            fn.RustatSingleFuncs.skew,
            fn.RustatParallelFuncs.skew,
            fn.PolarsFunc.skew,
        ],
    ),
    "kurt": FuncGroup(
        funcs=[
            fn.RustatSingleFuncs.kurt,
            fn.RustatParallelFuncs.kurt,
            fn.PolarsFunc.kurt,
        ],
    ),
}
