"""Execution: turns OrderEvents into FillEvents with commission and slippage.

Orders created at bar t's close are filled at bar t+1's OPEN. Filling at the
same close the signal was computed from is a classic look-ahead bug: you
couldn't have known the close and traded at it in the same instant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue

import numpy as np

from .data import DataHandler
from .events import FillEvent, MarketEvent, OrderEvent


# ---------------------------------------------------------------- commissions

class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, quantity: int, price: float) -> float: ...


class NoCommission(CommissionModel):
    def calculate(self, quantity, price):
        return 0.0


class PerShareCommission(CommissionModel):
    """IBKR-style: $/share with a per-order minimum and a % of notional cap."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0, max_pct: float = 0.01):
        self.per_share, self.minimum, self.max_pct = per_share, minimum, max_pct

    def calculate(self, quantity, price):
        q = abs(quantity)
        fee = max(self.minimum, q * self.per_share)
        return min(fee, q * price * self.max_pct)


class PercentCommission(CommissionModel):
    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def calculate(self, quantity, price):
        return abs(quantity) * price * self.rate


# ---------------------------------------------------------------- slippage

class SlippageModel(ABC):
    @abstractmethod
    def fill_price(self, price: float, quantity: int, data: DataHandler, symbol: str) -> float: ...


class NoSlippage(SlippageModel):
    def fill_price(self, price, quantity, data, symbol):
        return price


class FixedBpsSlippage(SlippageModel):
    """Pay `bps` basis points worse than the reference price, in the trade's direction."""

    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def fill_price(self, price, quantity, data, symbol):
        return price * (1 + np.sign(quantity) * self.bps / 1e4)


class VolumeShareSlippage(SlippageModel):
    """Impact grows with the order's share of the bar's volume (square-root model)."""

    def __init__(self, base_bps: float = 2.0, impact: float = 0.1):
        self.base_bps, self.impact = base_bps, impact

    def fill_price(self, price, quantity, data, symbol):
        volume = data.get_latest_bar_value(symbol, "volume")
        participation = abs(quantity) / volume if volume > 0 else 1.0
        cost = self.base_bps / 1e4 + self.impact * np.sqrt(participation)
        return price * (1 + np.sign(quantity) * cost)


# ---------------------------------------------------------------- handlers

class ExecutionHandler(ABC):
    @abstractmethod
    def on_order(self, order: OrderEvent) -> None: ...

    @abstractmethod
    def on_market(self, event: MarketEvent) -> None: ...


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, data: DataHandler, events: Queue,
                 commission: CommissionModel | None = None,
                 slippage: SlippageModel | None = None):
        self.data = data
        self.events = events
        self.commission = commission or NoCommission()
        self.slippage = slippage or NoSlippage()
        self._pending: list[OrderEvent] = []

    def on_order(self, order):
        self._pending.append(order)

    def on_market(self, event):
        """Called at the start of each bar, before the strategy runs."""
        pending, self._pending = self._pending, []
        for order in pending:
            open_ = self.data.get_latest_bar_value(order.symbol, "open")
            if not np.isfinite(open_) or open_ <= 0:
                self._pending.append(order)  # no tradable price yet; retry next bar
                continue
            price = self.slippage.fill_price(open_, order.quantity, self.data, order.symbol)
            commission = self.commission.calculate(order.quantity, price)
            self.events.put(FillEvent(
                symbol=order.symbol, timestamp=event.timestamp, quantity=order.quantity,
                fill_price=price, commission=commission,
                slippage_cost=abs(price - open_) * abs(order.quantity),
            ))
