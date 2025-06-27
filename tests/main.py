from groups import ROLLING_FUNCS
from manager import BenchmarkManager
from ui import main

if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore
    main(manager=rolling)
