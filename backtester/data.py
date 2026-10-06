"""DataHandlers: the only gateway a Strategy/Portfolio has to market data.

The key look-ahead guard lives here: every accessor slices data up to and
including the *current* bar and never beyond. There is deliberately no
"peek next bar" method.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]


class DataHandler(ABC):
    symbols: list[str]

    @abstractmethod
    def reset(self) -> None:
        """Rewind to before the first bar."""

    @abstractmethod
    def update_bars(self) -> bool:
        """Advance one bar. Returns False when data is exhausted."""

    @property
    @abstractmethod
    def current_time(self) -> pd.Timestamp: ...

    @property
    @abstractmethod
    def in_warmup(self) -> bool:
        """True while bars are only being fed to build indicator history."""

    @abstractmethod
    def get_latest_bars_values(self, symbol: str, field: str, n: int = 1) -> np.ndarray:
        """Last `n` values of `field`, ending at the current bar (inclusive)."""

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame: ...

    def get_latest_bar_value(self, symbol: str, field: str) -> float:
        return float(self.get_latest_bars_values(symbol, field, 1)[-1])


class PandasDataHandler(DataHandler):
    """Iterates bar-by-bar over a dict of OHLCV DataFrames (one per symbol).

    Symbols are aligned on the union of their timestamps and forward-filled,
    and rows before every symbol has data are dropped.

    `start`: bars before this timestamp are warmup - strategies can read them to
    build indicators, but the portfolio neither trades nor records equity.
    `end`: bars after this timestamp are discarded entirely.
    """

    def __init__(self, data: dict[str, pd.DataFrame], start=None, end=None):
        if not data:
            raise ValueError("No data supplied")
        frames = {sym: _normalize(df, sym) for sym, df in data.items()}
        index = frames[next(iter(frames))].index
        for df in frames.values():
            index = index.union(df.index)
        aligned = {}
        for sym, df in frames.items():
            df = df.reindex(index)
            df["volume"] = df["volume"].fillna(0)
            aligned[sym] = df.ffill()
        valid = np.logical_and.reduce([df["close"].notna().to_numpy() for df in aligned.values()])
        index = index[valid]
        if end is not None:
            index = index[index <= pd.Timestamp(end)]
        if len(index) == 0:
            raise ValueError("No overlapping bars across symbols in the requested range")

        self.symbols = list(aligned)
        self._index = index
        self._frames = {s: df.loc[index] for s, df in aligned.items()}
        self._arrays = {}
        for s, df in self._frames.items():
            self._arrays[s] = {}
            for f in FIELDS:
                arr = df[f].to_numpy(dtype=float).copy()
                arr.setflags(write=False)  # strategies can't mutate history
                self._arrays[s][f] = arr
        self._start = pd.Timestamp(start) if start is not None else None
        self._cursor = -1

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return self._frames

    def reset(self) -> None:
        self._cursor = -1

    def update_bars(self) -> bool:
        if self._cursor + 1 >= len(self._index):
            return False
        self._cursor += 1
        return True

    def _check(self):
        if self._cursor < 0:
            raise RuntimeError("update_bars() has not been called yet")

    @property
    def current_time(self) -> pd.Timestamp:
        self._check()
        return self._index[self._cursor]

    @property
    def in_warmup(self) -> bool:
        return self._start is not None and self.current_time < self._start

    @property
    def bars_seen(self) -> int:
        return self._cursor + 1

    def get_latest_bars_values(self, symbol, field, n=1):
        self._check()
        lo = max(0, self._cursor - n + 1)
        return self._arrays[symbol][field][lo : self._cursor + 1]

    def get_latest_bars(self, symbol, n=1):
        self._check()
        lo = max(0, self._cursor - n + 1)
        return self._frames[symbol].iloc[lo : self._cursor + 1].copy()


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[FIELDS].astype(float)


# --------------------------------------------------------------------------
# Loaders - each returns dict[symbol, OHLCV DataFrame]
# --------------------------------------------------------------------------

def load_yfinance(symbols: Iterable[str], start, end=None, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Daily (or intraday, limited history) bars from Yahoo Finance.

    Uses auto_adjust=True so splits/dividends are folded into OHLC. Note that
    adjusted prices are themselves computed with hindsight; that's fine for
    returns, but don't use them for anything price-level-dependent.
    """
    import yfinance as yf

    out = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"yfinance returned no data for {sym}")
        out[sym] = df
    return out


def load_alpaca(symbols: Iterable[str], start, end=None, timeframe: str = "1Day",
                feed: str = "iex") -> dict[str, pd.DataFrame]:
    """Bars from Alpaca Market Data (needs ALPACA_API_KEY / ALPACA_SECRET_KEY).

    timeframe: e.g. "1Day", "1Hour", "15Min". Free accounts use the "iex" feed.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")

    units = {"Min": TimeFrameUnit.Minute, "Hour": TimeFrameUnit.Hour,
             "Day": TimeFrameUnit.Day, "Week": TimeFrameUnit.Week}
    amount = int("".join(ch for ch in timeframe if ch.isdigit()) or 1)
    unit = units["".join(ch for ch in timeframe if ch.isalpha())]

    symbols = list(symbols)
    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame(amount, unit),
                           start=pd.Timestamp(start), end=pd.Timestamp(end) if end else None,
                           adjustment=Adjustment.ALL, feed=DataFeed(feed))
    bars = client.get_stock_bars(req).df
    return {sym: bars.xs(sym, level="symbol") for sym in symbols}


def load_csv(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    """CSV files with a date column first and open/high/low/close/volume columns."""
    return {sym: pd.read_csv(p, index_col=0, parse_dates=True) for sym, p in paths.items()}


def generate_synthetic(symbols: Iterable[str], start="2010-01-01", end="2024-12-31",
                       seed: int = 42, mu: float = 0.08, sigma: float = 0.2) -> dict[str, pd.DataFrame]:
    """Geometric Brownian motion OHLCV - for offline demos and tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    dt = 1 / 252
    out = {}
    for sym in symbols:
        rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), len(idx))
        close = 100 * np.exp(np.cumsum(rets))
        open_ = close * np.exp(rng.normal(0, sigma * np.sqrt(dt) / 3, len(idx)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
        vol = rng.integers(1_000_000, 5_000_000, len(idx))
        out[sym] = pd.DataFrame({"open": open_, "high": high, "low": low,
                                 "close": close, "volume": vol}, index=idx)
    return out
