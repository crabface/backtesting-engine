"""Strategy-vs-benchmark comparison."""

from __future__ import annotations

import pandas as pd

from .engine import Backtest, BacktestResult
from .metrics import format_value, relative_metrics
from .strategy import BuyAndHoldStrategy

ROWS = ["total_return", "cagr", "annual_volatility", "sharpe", "sortino", "max_drawdown",
        "max_drawdown_duration_bars", "calmar", "num_trades", "win_rate", "profit_factor", "final_equity"]


def run_benchmark(backtest: Backtest) -> BacktestResult:
    """Buy-and-hold on the same data, capital, weights and costs as `backtest`."""
    bench = Backtest(backtest.data, BuyAndHoldStrategy, initial_capital=backtest.initial_capital,
                     commission=backtest.commission, slippage=backtest.slippage, weights=backtest.weights,
                     risk_free=backtest.risk_free, periods_per_year=backtest.periods_per_year)
    return bench.run()


def comparison_table(result: BacktestResult, benchmark: BacktestResult, risk_free=0.0,
                     periods_per_year: int = 252) -> pd.DataFrame:
    rows = {k: [format_value(k, result.metrics.get(k)), format_value(k, benchmark.metrics.get(k))] for k in ROWS}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["Strategy", "Buy & Hold"])
    rel = relative_metrics(result.returns, benchmark.returns, risk_free, periods_per_year)
    for k, v in rel.items():
        table.loc[k] = [format_value(k, v), ""]
    return table


def text_report(result: BacktestResult, benchmark: BacktestResult, risk_free=0.0, periods_per_year: int = 252) -> str:
    m = result.metrics
    header = (f"{result.strategy_name} {result.params}\n"
              f"Period: {format_value('start', m['start'])} -> {format_value('end', m['end'])}\n"
              f"Costs paid: commission {result.total_commission:,.2f}, slippage {result.total_slippage:,.2f}\n")
    return header + comparison_table(result, benchmark, risk_free, periods_per_year).to_string()
