"""Strategies turn market data into SignalEvents.

A Strategy only sees data through the DataHandler, so it can't read bars that
haven't happened yet. Signals are decided at a bar's close; the engine fills
the resulting orders at the *next* bar's open.

To write a new strategy, subclass Strategy and implement `calculate_signals`,
calling `self.target(symbol, Direction.X)` with the position you want. The
base class only emits a signal when the target changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue

import numpy as np

from .data import DataHandler
from .events import Direction, MarketEvent, SignalEvent


class Strategy(ABC):
    def __init__(self, data: DataHandler, events: Queue):
        self.data = data
        self.events = events
        self._current: dict[str, Direction] = {s: Direction.EXIT for s in data.symbols}

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None: ...

    def target(self, symbol: str, direction: Direction, strength: float = 1.0) -> None:
        """Request a position. No-op if unchanged or during warmup."""
        if self.data.in_warmup or self._current[symbol] == direction:
            return
        self._current[symbol] = direction
        self.events.put(SignalEvent(symbol, self.data.current_time, direction, strength))


class BuyAndHoldStrategy(Strategy):
    """Goes long every symbol on the first tradable bar and never sells."""

    def calculate_signals(self, event):
        for s in self.data.symbols:
            self.target(s, Direction.LONG)


class MovingAverageCrossStrategy(Strategy):
    """Long when the short SMA is above the long SMA, flat (or short) otherwise."""

    def __init__(self, data, events, short_window: int = 50, long_window: int = 200,
                 allow_short: bool = False):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        super().__init__(data, events)
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short

    def calculate_signals(self, event):
        for s in self.data.symbols:
            closes = self.data.get_latest_bars_values(s, "close", self.long_window)
            if len(closes) < self.long_window:
                continue
            short_ma = closes[-self.short_window:].mean()
            long_ma = closes.mean()
            if short_ma > long_ma:
                self.target(s, Direction.LONG)
            else:
                self.target(s, Direction.SHORT if self.allow_short else Direction.EXIT)


class MomentumStrategy(Strategy):
    """Time-series momentum: long if the trailing `lookback` return is positive."""

    def __init__(self, data, events, lookback: int = 126, threshold: float = 0.0):
        super().__init__(data, events)
        self.lookback = lookback
        self.threshold = threshold

    def calculate_signals(self, event):
        for s in self.data.symbols:
            closes = self.data.get_latest_bars_values(s, "close", self.lookback + 1)
            if len(closes) <= self.lookback:
                continue
            ret = closes[-1] / closes[0] - 1
            self.target(s, Direction.LONG if ret > self.threshold else Direction.EXIT)


class BollingerReversionStrategy(Strategy):
    """Mean reversion: buy a close below the lower band, exit at the mean."""

    def __init__(self, data, events, window: int = 20, num_std: float = 2.0):
        super().__init__(data, events)
        self.window = window
        self.num_std = num_std

    def calculate_signals(self, event):
        for s in self.data.symbols:
            closes = self.data.get_latest_bars_values(s, "close", self.window)
            if len(closes) < self.window:
                continue
            mean, std = closes.mean(), closes.std(ddof=1)
            price = closes[-1]
            if np.isfinite(std) and price < mean - self.num_std * std:
                self.target(s, Direction.LONG)
            elif price >= mean:
                self.target(s, Direction.EXIT)


STRATEGIES = {
    "buyhold": BuyAndHoldStrategy,
    "sma": MovingAverageCrossStrategy,
    "momentum": MomentumStrategy,
    "bollinger": BollingerReversionStrategy,
}
