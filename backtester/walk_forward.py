"""Walk-forward optimization.

Split history into consecutive (train, test) windows. For each window, grid
search parameters on TRAIN only, then run the winner on the following unseen
TEST window. Stitching the test windows together gives an out-of-sample
equity curve - a far more honest estimate than the best in-sample backtest.

Each test run gets `warmup_bars` of prior history so indicators are primed,
but the portfolio only trades/records from the test window's first bar.
Positions are not carried between windows (each test starts flat).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PandasDataHandler
from .engine import Backtest
from .metrics import compute_metrics


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    metrics: dict


def _grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def walk_forward(frames: dict[str, pd.DataFrame], strategy_cls, param_grid: dict[str, list],
                 train_bars: int = 756, test_bars: int = 252, warmup_bars: int = 252,
                 anchored: bool = False, objective: str = "sharpe", initial_capital: float = 100_000.0,
                 risk_free: float | pd.Series = 0.0, periods_per_year: int = 252,
                 verbose: bool = True, **backtest_kwargs) -> WalkForwardResult:
    index = PandasDataHandler(frames).index
    combos = _grid(param_grid)

    def make_run(lo: int, hi: int, params: dict):
        first = max(0, lo - warmup_bars)
        sliced = {s: df.loc[(df.index >= index[first]) & (df.index <= index[hi - 1])] for s, df in frames.items()}
        dh = PandasDataHandler(sliced, start=index[lo], end=index[hi - 1])
        return Backtest(dh, strategy_cls, params, initial_capital=initial_capital, risk_free=risk_free,
                        periods_per_year=periods_per_year, **backtest_kwargs).run()

    rows, oos_returns, oos_trades = [], [], []
    test_lo = train_bars
    while test_lo < len(index):
        test_hi = min(test_lo + test_bars, len(index))
        if test_hi - test_lo < 20:  # too short to be meaningful
            break
        train_lo = 0 if anchored else test_lo - train_bars

        best_params, best_score = None, -np.inf
        for params in combos:
            try:
                res = make_run(train_lo, test_lo, params)
            except ValueError:  # invalid combination, e.g. short >= long window
                continue
            score = res.metrics.get(objective, np.nan)
            if np.isfinite(score) and score > best_score:
                best_params, best_score = params, score
        if best_params is None:
            best_params = next(p for p in combos if _valid(strategy_cls, p))

        test = make_run(test_lo, test_hi, best_params)
        oos_returns.append(test.returns)
        # Windows end with positions still open; count them as trades marked at the window's last close.
        oos_trades.append(pd.concat([test.trades, test.open_trades], ignore_index=True))
        rows.append({
            "train_start": index[train_lo], "train_end": index[test_lo - 1],
            "test_start": index[test_lo], "test_end": index[test_hi - 1],
            "params": best_params, f"is_{objective}": best_score,
            f"oos_{objective}": test.metrics.get(objective), "oos_return": test.metrics["total_return"],
        })
        if verbose:
            print(f"  test {index[test_lo].date()} -> {index[test_hi - 1].date()}  params={best_params}  "
                  f"IS {objective}={best_score:.2f}  OOS {objective}={test.metrics.get(objective, np.nan):.2f}")
        test_lo = test_hi

    if not rows:
        raise ValueError("Not enough data for a single train/test window")
    returns = pd.concat(oos_returns)
    equity = initial_capital * (1 + returns).cumprod()
    trades = pd.concat(oos_trades, ignore_index=True)
    metrics = compute_metrics(equity, trades, risk_free, periods_per_year, initial_capital)
    return WalkForwardResult(pd.DataFrame(rows), equity, trades, metrics)


def _valid(strategy_cls, params) -> bool:
    try:
        strategy_cls(_DummyData(), None, **params)
        return True
    except ValueError:
        return False


class _DummyData:
    symbols: list[str] = []
