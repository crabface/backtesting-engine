"""Event types passed through the engine's queue.

Flow for every bar:
    MarketEvent -> (ExecutionHandler fills pending orders at this bar's open)
                -> Strategy emits SignalEvent
    SignalEvent -> Portfolio sizes it into an OrderEvent
    OrderEvent  -> ExecutionHandler queues it for the NEXT bar's open
    FillEvent   -> Portfolio updates cash / positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


@dataclass
class MarketEvent:
    timestamp: pd.Timestamp
    type: EventType = field(default=EventType.MARKET, init=False)


@dataclass
class SignalEvent:
    symbol: str
    timestamp: pd.Timestamp
    direction: Direction
    strength: float = 1.0
    type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent:
    symbol: str
    timestamp: pd.Timestamp  # when the order was created (decision time)
    quantity: int  # signed: +buy / -sell
    order_type: str = "MKT"
    type: EventType = field(default=EventType.ORDER, init=False)


@dataclass
class FillEvent:
    symbol: str
    timestamp: pd.Timestamp  # when the order was filled
    quantity: int  # signed
    fill_price: float
    commission: float
    slippage_cost: float
    type: EventType = field(default=EventType.FILL, init=False)
