"""Render bench.report tables and SVGs with repeatable Matplotlib element IDs."""
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import style

from bench.report import main


if __name__ == "__main__":
    with style.context("default"):
        mpl.rcParams["svg.hashsalt"] = "metal-megakernel-attention-report"
        main()
