# Event-Driven Backtesting Engine

A Python backtester built around four pluggable components: `DataHandler`, `Strategy`, `Portfolio` and `ExecutionHandler`. It models commissions and slippage, reports Sharpe, Sortino, max drawdown, CAGR and win rate, plots an equity curve against a buy-and-hold benchmark, and supports multi-asset portfolios and walk-forward optimization.

See [docs/DESIGN.md](docs/DESIGN.md) for how the engine works and how it prevents look-ahead bias.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Optional environment variables:

| Variable | Used for |
|---|---|
| `FRED_API_KEY` | Official FRED API. Without it, `--rf fred` uses FRED's public CSV endpoint. |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | `--source alpaca` (intraday bars). Also run `pip install alpaca-py`. |

## Usage

```bash
# 50/200 SMA crossover on SPY, risk-free rate from FRED (3-month T-bill)
python run_backtest.py --symbols SPY --strategy sma --params short_window=50 long_window=200 --rf fred

# Multi-asset momentum, 10 bps slippage, 0.1% commission
python run_backtest.py --symbols SPY QQQ TLT GLD --strategy momentum --params lookback=126 \
    --commission percent --slippage-bps 10

# Walk-forward: optimize on 3 years, test on the next 1, roll forward
python run_backtest.py --symbols SPY QQQ TLT --strategy sma --walk-forward \
    --grid short_window=20,50 long_window=100,200

# Offline, with synthetic prices
python run_backtest.py --source synthetic --symbols AAA BBB --strategy bollinger

# Hourly bars from Alpaca
python run_backtest.py --source alpaca --interval 1Hour --symbols AAPL --start 2024-01-01 \
    --strategy sma --params short_window=10 long_window=50
```

Each run prints a strategy-vs-buy-and-hold table that includes alpha, beta and information ratio. It saves `equity_curve.png`, `equity_curve.csv` and `trades.csv` to `output/`.

Built-in strategies: `buyhold`, `sma`, `momentum`, `bollinger`.

## Writing a strategy

```python
from backtester import Strategy, Direction, Backtest, PandasDataHandler, load_yfinance
from backtester import PerShareCommission, FixedBpsSlippage, run_benchmark, text_report

class RSIStrategy(Strategy):
    def __init__(self, data, events, period=14, low=30, high=70):
        super().__init__(data, events)
        self.period, self.low, self.high = period, low, high

    def calculate_signals(self, event):
        for s in self.data.symbols:
            closes = self.data.get_latest_bars_values(s, "close", self.period + 1)
            if len(closes) <= self.period:
                continue
            diff = closes[1:] - closes[:-1]
            gain, loss = diff.clip(min=0).mean(), -diff.clip(max=0).mean()
            rsi = 100 if loss == 0 else 100 - 100 / (1 + gain / loss)
            if rsi < self.low:
                self.target(s, Direction.LONG)
            elif rsi > self.high:
                self.target(s, Direction.EXIT)

data = PandasDataHandler(load_yfinance(["SPY"], "2010-01-01"))
bt = Backtest(data, RSIStrategy, {"period": 14}, commission=PerShareCommission(), slippage=FixedBpsSlippage(5))
result = bt.run()
print(text_report(result, run_benchmark(bt)))
```

`self.target(symbol, direction)` declares the position you want. The base class only emits a signal when the target changes, and stays silent during warmup.

## Layout

```
backtester/
  events.py        Market / Signal / Order / Fill events
  data.py          DataHandler, PandasDataHandler, yfinance / Alpaca / CSV / synthetic loaders
  strategy.py      Strategy base class and example strategies
  portfolio.py     Position sizing, cash, average-cost trade accounting, equity curve
  execution.py     Simulated broker, commission and slippage models
  engine.py        The event loop (Backtest) and BacktestResult
  metrics.py       Sharpe, Sortino, drawdown, CAGR, win rate, alpha/beta
  risk_free.py     FRED risk-free rate
  report.py        Buy-and-hold benchmark comparison
  walk_forward.py  Rolling or anchored walk-forward optimization
  plotting.py      Equity and drawdown chart
tests/             Includes look-ahead bias regression tests
run_backtest.py    Command-line interface
```

## Known simplifications

- Market orders only. Orders fill in full at the next bar's open plus slippage, with no partial fills.
- Orders are sized from the last close but fill at the next open, so cash can dip slightly below zero. The engine allows this as a small margin balance.
- Short positions don't pay borrow costs, and uninvested cash earns no interest.
- yfinance prices are split- and dividend-adjusted, which is fine for measuring returns. Survivorship bias isn't handled: today's tickers are the ones that survived.
- In walk-forward, every test window starts flat. Positions still open when a window ends count as trades, marked at that window's last close.
