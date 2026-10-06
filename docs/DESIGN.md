# Design notes

## Why event-driven?

A **vectorized** backtest computes signals for every day at once with pandas, for example `signal = sma50 > sma200`, `returns = signal.shift(1) * price.pct_change()`. It's fast and short, but:

- Nothing stops you from using future rows. A missing `.shift(1)` or a centered rolling window silently leaks the future.
- Path-dependent logic is awkward. Position sizing that depends on current equity, stops, partial fills and per-trade commissions are hard to express.
- The research code has nothing in common with the code that would trade live.

An **event-driven** backtest replays history one bar at a time. Each component reacts to messages on a queue, the same way it would in live trading:

```
                    ┌──────────────┐
  update_bars() ───▶│  DataHandler │──── MarketEvent ────┐
                    └──────────────┘                     ▼
                                              ┌───────────────────┐
                    ┌─────────── FillEvent ───│ ExecutionHandler  │◀── OrderEvent ──┐
                    ▼                         └───────────────────┘                 │
             ┌────────────┐                                                         │
             │ Portfolio  │◀── SignalEvent ── Strategy ◀── MarketEvent              │
             └────────────┘                                                         │
                    └─────────────────────────── OrderEvent ────────────────────────┘
```

Order of operations for each bar `t` (see `engine.py`):

1. **DataHandler** advances to bar `t` and puts a `MarketEvent` on the queue.
2. **ExecutionHandler** fills orders placed at `t-1` at bar `t`'s open, adding slippage and commission, and emits `FillEvent`s.
3. **Strategy** reads history up to bar `t`'s close and emits `SignalEvent`s.
4. **Portfolio** books the fills, then converts the signals into sized `OrderEvent`s.
5. **ExecutionHandler** queues those orders for bar `t+1`.
6. **Portfolio** marks positions to market at bar `t`'s close and records equity.

Each component depends only on the interfaces of the others. A new strategy is a single subclass. Swapping yfinance for Alpaca means changing the loader. Replacing `SimulatedExecutionHandler` with a broker adapter turns the engine into a live or paper-trading system.

## Look-ahead bias

Look-ahead bias means using information at time `t` that wouldn't have been available until later. It makes backtests look much better than live trading. The engine guards against it in these places:

| Where | Bug it prevents | How |
|---|---|---|
| `DataHandler` | Strategy reads tomorrow's bar | Accessors slice `[: cursor+1]`. There is no "next bar" method. |
| `DataHandler` | Strategy mutates history | Arrays are returned read-only (`setflags(write=False)`). |
| `ExecutionHandler` | Trading at the close the signal was computed from | Orders fill at the **next** bar's open. |
| `Portfolio` | Sizing with the fill price | Quantity is sized from the last known close. The fill price is learned later. |
| `metrics.periodic_risk_free` | Using a rate before it was published | As-of join: forward-fill the last published FRED value. |
| `walk_forward` | Choosing parameters with the test data | Grid search runs on the train window only. The winner is then run on unseen data. |
| `walk_forward` | Indicators "starting" with knowledge | Warmup bars before the test window feed indicators but are never traded. |

The regression test `test_changing_future_data_does_not_change_past_decisions` tampers with every price after a cutoff date. It then asserts that equity and fills before the cutoff are byte-identical. Any future leak breaks that test.

### Subtler sources this engine doesn't fully solve

- **Survivorship bias.** Backtesting today's S&P 500 members excludes companies that went bankrupt or were delisted.
- **Adjusted prices.** Dividend and split adjustment is computed with hindsight. It's fine for returns, but price-level rules like "buy when below $10" become wrong.
- **Data revisions.** Economic series (GDP, CPI) are revised. Use ALFRED vintages for point-in-time data.
- **Timestamp alignment.** A daily bar labeled with a date isn't known until that day's close. Mixing daily and intraday data, or data from different time zones, can leak a few hours.
- **Overfitting.** Trying many strategies and reporting the best one is look-ahead at the research level. Walk-forward testing helps, and so does keeping a final hold-out period you never touch.

## Metrics

All metrics are computed from the per-bar equity curve. `periods_per_year` is 252 for daily bars.

- **Sharpe** = √N · mean(r − r_f) / std(r − r_f), using the FRED 3-month T-bill (DTB3) or a constant rate.
- **Sortino** uses the same numerator over the downside deviation √mean(min(r − r_f, 0)²).
- **Max drawdown** is the minimum of equity / running peak − 1. Its duration is the longest stretch below a prior peak.
- **CAGR** = (final / initial)^(1/years) − 1, with years measured in calendar days / 365.25.
- **Win rate** is the fraction of closed round trips (flat → flat) with positive P&L net of commission. Trade P&L uses average-cost accounting, so scaling in and out and reversing direction are handled.
- **Alpha, beta, information ratio** come from regressing strategy excess returns on benchmark excess returns.
