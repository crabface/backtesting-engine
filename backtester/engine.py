"""The event loop that wires DataHandler, Strategy, Portfolio and ExecutionHandler together."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import pandas as pd

from .data import DataHandler
from .events import EventType, MarketEvent
from .execution import CommissionModel, SimulatedExecutionHandler, SlippageModel
from .metrics import compute_metrics, format_value
from .portfolio import Portfolio
from .strategy import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    open_trades: pd.DataFrame
    metrics: dict
    strategy_name: str
    params: dict = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0

    @property
    def equity(self) -> pd.Series:
        return self.equity_curve["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve["returns"]

    def summary(self) -> str:
        lines = [f"{self.strategy_name} {self.params or ''}".rstrip()]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<28}{format_value(k, v)}")
        lines.append(f"  {'total_commission':<28}{self.total_commission:,.2f}")
        lines.append(f"  {'total_slippage':<28}{self.total_slippage:,.2f}")
        return "\n".join(lines)


class Backtest:
    def __init__(self, data: DataHandler, strategy_cls: type[Strategy], strategy_params: dict[str, Any] | None = None,
                 initial_capital: float = 100_000.0, commission: CommissionModel | None = None,
                 slippage: SlippageModel | None = None, weights: dict[str, float] | None = None,
                 risk_free: float | pd.Series = 0.0, periods_per_year: int = 252):
        self.data = data
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.weights = weights
        self.risk_free = risk_free
        self.periods_per_year = periods_per_year

    def run(self) -> BacktestResult:
        self.data.reset()
        events: Queue = Queue()
        strategy = self.strategy_cls(self.data, events, **self.strategy_params)
        portfolio = Portfolio(self.data, events, self.initial_capital, self.weights)
        execution = SimulatedExecutionHandler(self.data, events, self.commission, self.slippage)

        while self.data.update_bars():
            events.put(MarketEvent(self.data.current_time))
            while True:
                try:
                    event = events.get_nowait()
                except Empty:
                    break
                if event.type is EventType.MARKET:
                    execution.on_market(event)        # fill yesterday's orders at today's open
                    strategy.calculate_signals(event)  # decide using data up to today's close
                elif event.type is EventType.SIGNAL:
                    portfolio.on_signal(event)
                elif event.type is EventType.ORDER:
                    execution.on_order(event)          # queued for the next bar
                elif event.type is EventType.FILL:
                    portfolio.on_fill(event)
            portfolio.update_timeindex()

        curve = portfolio.equity_curve()
        trades = portfolio.trades()
        metrics = compute_metrics(curve["equity"], trades, self.risk_free,
                                  self.periods_per_year, self.initial_capital)
        return BacktestResult(
            equity_curve=curve, trades=trades, fills=portfolio.fills(),
            open_trades=portfolio.open_trades(), metrics=metrics,
            strategy_name=self.strategy_cls.__name__, params=dict(self.strategy_params),
            total_commission=portfolio.total_commission, total_slippage=portfolio.total_slippage,
        )
