"""Portfolio: sizes signals into orders, books fills, tracks equity and trades."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue

import numpy as np
import pandas as pd

from .data import DataHandler
from .events import Direction, FillEvent, OrderEvent, SignalEvent


@dataclass
class _OpenTrade:
    symbol: str
    entry_time: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_notional: float
    pnl: float = 0.0


class Portfolio:
    """Target-weight portfolio.

    A LONG signal targets `weight * strength` of current equity in a symbol,
    SHORT the negative of that, EXIT zero. Default weights are equal (1/N).
    Order size uses the latest *close* (what we know at decision time); the
    order actually fills at the next open, so cash can dip slightly negative.
    """

    def __init__(self, data: DataHandler, events: Queue, initial_capital: float = 100_000.0,
                 weights: dict[str, float] | None = None):
        self.data = data
        self.events = events
        self.initial_capital = float(initial_capital)
        n = len(data.symbols)
        self.weights = weights or {s: 1.0 / n for s in data.symbols}

        self.cash = self.initial_capital
        self.positions = {s: 0 for s in data.symbols}
        self.avg_cost = {s: 0.0 for s in data.symbols}
        self.pending = {s: 0 for s in data.symbols}
        self.total_commission = 0.0
        self.total_slippage = 0.0

        self._history: list[dict] = []
        self._fills: list[dict] = []
        self._trades: list[dict] = []
        self._open_trades: dict[str, _OpenTrade] = {}

    # ------------------------------------------------------------ valuation

    def _price(self, symbol: str) -> float:
        return self.data.get_latest_bar_value(symbol, "close")

    def equity(self) -> float:
        return self.cash + sum(q * self._price(s) for s, q in self.positions.items() if q)

    # ------------------------------------------------------------ signals -> orders

    def on_signal(self, signal: SignalEvent) -> None:
        if self.data.in_warmup:
            return
        sign = {Direction.LONG: 1, Direction.SHORT: -1, Direction.EXIT: 0}[signal.direction]
        price = self._price(signal.symbol)
        target_value = self.equity() * self.weights[signal.symbol] * signal.strength * sign
        target_qty = int(target_value / price)  # truncates toward zero
        current = self.positions[signal.symbol] + self.pending[signal.symbol]
        delta = target_qty - current
        if delta != 0:
            self.pending[signal.symbol] += delta
            self.events.put(OrderEvent(signal.symbol, signal.timestamp, delta))

    # ------------------------------------------------------------ fills

    def on_fill(self, fill: FillEvent) -> None:
        s, qty, px = fill.symbol, fill.quantity, fill.fill_price
        self.pending[s] -= qty
        self.cash -= qty * px + fill.commission
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self._fills.append({
            "timestamp": fill.timestamp, "symbol": s, "quantity": qty, "price": px,
            "commission": fill.commission, "slippage": fill.slippage_cost,
        })
        self._book_trade(fill)

    def _book_trade(self, fill: FillEvent) -> None:
        """Average-cost accounting; a 'trade' is a round trip from flat to flat."""
        s, qty, px = fill.symbol, fill.quantity, fill.fill_price
        pos, avg = self.positions[s], self.avg_cost[s]
        remaining_commission = fill.commission

        if pos != 0 and np.sign(qty) != np.sign(pos):
            closing = int(np.sign(pos)) * min(abs(qty), abs(pos))
            trade = self._open_trades[s]
            share = abs(closing) / abs(qty)
            trade.pnl += closing * (px - avg) - fill.commission * share
            remaining_commission -= fill.commission * share
            pos -= closing
            qty += closing
            if pos == 0:
                self._close_trade(s, fill.timestamp)
                avg = 0.0

        if qty != 0:
            if pos == 0:
                self._open_trades[s] = _OpenTrade(s, fill.timestamp, int(np.sign(qty)), 0.0)
                avg = 0.0
            trade = self._open_trades[s]
            avg = (pos * avg + qty * px) / (pos + qty)
            trade.entry_notional += abs(qty) * px
            trade.pnl -= remaining_commission
            pos += qty

        self.positions[s], self.avg_cost[s] = pos, avg

    def _close_trade(self, symbol: str, exit_time: pd.Timestamp) -> None:
        t = self._open_trades.pop(symbol)
        self._trades.append({
            "symbol": symbol, "direction": "LONG" if t.direction > 0 else "SHORT",
            "entry_time": t.entry_time, "exit_time": exit_time, "pnl": t.pnl,
            "return": t.pnl / t.entry_notional if t.entry_notional else np.nan,
        })

    # ------------------------------------------------------------ bookkeeping

    def update_timeindex(self) -> None:
        """Mark to market at the current bar's close."""
        if self.data.in_warmup:
            return
        row = {"timestamp": self.data.current_time, "cash": self.cash}
        holdings = 0.0
        for s, q in self.positions.items():
            v = q * self._price(s)
            row[f"pos_{s}"] = q
            row[f"value_{s}"] = v
            holdings += v
        row["holdings"] = holdings
        row["equity"] = self.cash + holdings
        row["commission"] = self.total_commission
        self._history.append(row)

    def equity_curve(self) -> pd.DataFrame:
        df = pd.DataFrame(self._history).set_index("timestamp")
        df["returns"] = df["equity"].pct_change().fillna(df["equity"].iloc[0] / self.initial_capital - 1)
        df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
        return df

    def trades(self) -> pd.DataFrame:
        cols = ["symbol", "direction", "entry_time", "exit_time", "pnl", "return"]
        return pd.DataFrame(self._trades, columns=cols)

    def open_trades(self) -> pd.DataFrame:
        """Still-open round trips, marked to the current close (pnl includes unrealized)."""
        rows = []
        for t in self._open_trades.values():
            pnl = t.pnl + self.positions[t.symbol] * (self._price(t.symbol) - self.avg_cost[t.symbol])
            rows.append({"symbol": t.symbol, "direction": "LONG" if t.direction > 0 else "SHORT",
                         "entry_time": t.entry_time, "exit_time": self.data.current_time, "pnl": pnl,
                         "return": pnl / t.entry_notional if t.entry_notional else np.nan})
        return pd.DataFrame(rows, columns=["symbol", "direction", "entry_time", "exit_time", "pnl", "return"])

    def fills(self) -> pd.DataFrame:
        return pd.DataFrame(self._fills, columns=["timestamp", "symbol", "quantity", "price", "commission", "slippage"])
