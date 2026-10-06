"""Equity curve and drawdown charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import drawdown  # noqa: E402


def plot_equity(equity: pd.Series, benchmark: pd.Series | None = None, title: str = "Equity curve",
                path: str | None = None, log_scale: bool = False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity.index, equity.values, label="Strategy", color="#2a6fdb", lw=1.6)
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, label="Buy & Hold", color="#888888", lw=1.2, alpha=0.9)
    if log_scale:
        ax1.set_yscale("log")
    ax1.set_title(title)
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", frameon=False)

    dd = drawdown(equity)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d1495b", alpha=0.35, label="Strategy")
    if benchmark is not None:
        bdd = drawdown(benchmark)
        ax2.plot(bdd.index, bdd.values * 100, color="#888888", lw=1, label="Buy & Hold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    plt.close(fig)
    return fig
