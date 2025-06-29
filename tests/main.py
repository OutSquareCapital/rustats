from manager import BenchmarkManager
from ui import main
from groups import ROLLING_FUNCS

if __name__ == "__main__":
    rolling = BenchmarkManager(groups=ROLLING_FUNCS)
    import plotly.io as pio

    pio.renderers.default = "browser"  # type: ignore
    main(manager=rolling)
