import numpy as np
import pandas as pd
import pytest

from backtester import (Backtest, BuyAndHoldStrategy, Direction, FixedBpsSlippage,
                        MovingAverageCrossStrategy, PandasDataHandler, PerShareCommission, Strategy,
                        compute_metrics, generate_synthetic, walk_forward)


def _frame(closes, opens=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes),
                         "close": closes, "volume": 1e6}, index=idx)


# ---------------------------------------------------------------- look-ahead bias

def test_data_handler_never_exposes_future_bars():
    dh = PandasDataHandler({"X": _frame(range(1, 11))})
    dh.update_bars()
    dh.update_bars()
    assert list(dh.get_latest_bars_values("X", "close", 100)) == [1, 2]
    assert dh.get_latest_bars("X", 100).index[-1] == dh.current_time


def test_history_is_read_only():
    dh = PandasDataHandler({"X": _frame(range(1, 11))})
    dh.update_bars()
    with pytest.raises(ValueError):
        dh.get_latest_bars_values("X", "close", 1)[0] = 999


def test_changing_future_data_does_not_change_past_decisions():
    """If results before day T depend on data after T, we have look-ahead bias."""
    frames = generate_synthetic(["X"], "2015-01-01", "2020-12-31", seed=1)
    cutoff = frames["X"].index[800]
    tampered = {"X": frames["X"].copy()}
    tampered["X"].loc[tampered["X"].index > cutoff, ["open", "high", "low", "close"]] *= 3.0

    params = {"short_window": 20, "long_window": 60}
    a = Backtest(PandasDataHandler(frames), MovingAverageCrossStrategy, params).run()
    b = Backtest(PandasDataHandler(tampered), MovingAverageCrossStrategy, params).run()

    pd.testing.assert_series_equal(a.equity[:cutoff], b.equity[:cutoff])
    fa, fb = a.fills[a.fills.timestamp <= cutoff], b.fills[b.fills.timestamp <= cutoff]
    pd.testing.assert_frame_equal(fa.reset_index(drop=True), fb.reset_index(drop=True))


def test_orders_fill_at_next_bar_open():
    closes = [10, 10, 10, 10, 10]
    opens = [10, 11, 12, 13, 14]
    dh = PandasDataHandler({"X": _frame(closes, opens)})
    res = Backtest(dh, BuyAndHoldStrategy, initial_capital=1000).run()
    fill = res.fills.iloc[0]
    assert fill.timestamp == dh.index[1]   # signal on bar 0, fill on bar 1
    assert fill.price == 11                # at bar 1's open, not bar 0's close
    assert fill.quantity == 100            # sized from bar 0's close: 1000 / 10


# ---------------------------------------------------------------- costs

def test_commission_and_slippage_are_charged():
    dh = PandasDataHandler({"X": _frame([100] * 5)})
    res = Backtest(dh, BuyAndHoldStrategy, initial_capital=10_000,
                   commission=PerShareCommission(0.01, minimum=1.0), slippage=FixedBpsSlippage(10)).run()
    fill = res.fills.iloc[0]
    assert fill.price == pytest.approx(100 * 1.001)
    assert fill.commission == pytest.approx(1.0)  # 100 shares * 0.01 = 1.0
    expected_cash = 10_000 - 100 * 100.1 - 1.0
    assert res.equity_curve["cash"].iloc[-1] == pytest.approx(expected_cash)
    assert res.equity.iloc[-1] < 10_000


# ---------------------------------------------------------------- trade accounting

class ScriptedStrategy(Strategy):
    def __init__(self, data, events, script):
        super().__init__(data, events)
        self.script = script

    def calculate_signals(self, event):
        d = self.script.get(self.data.bars_seen - 1)
        if d is not None:
            self.target("X", d)


def test_round_trip_pnl_and_win_rate():
    # buy signal bar0 -> fill bar1 @10; exit signal bar2 -> fill bar3 @12 (win)
    # buy signal bar4 -> fill bar5 @12; exit signal bar6 -> fill bar7 @9 (loss)
    opens = [10, 10, 11, 12, 12, 12, 10, 9, 9]
    dh = PandasDataHandler({"X": _frame(opens, opens)})
    script = {0: Direction.LONG, 2: Direction.EXIT, 4: Direction.LONG, 6: Direction.EXIT}
    res = Backtest(dh, ScriptedStrategy, {"script": script}, initial_capital=1000).run()
    assert list(res.trades.pnl) == pytest.approx([200.0, -300.0])
    assert res.metrics["win_rate"] == 0.5
    assert res.equity.iloc[-1] == pytest.approx(900.0)


def test_short_and_reversal():
    opens = [10, 10, 10, 8, 8, 8]
    dh = PandasDataHandler({"X": _frame([10, 10, 10, 10, 10, 10], opens)})
    script = {0: Direction.SHORT, 2: Direction.LONG}
    res = Backtest(dh, ScriptedStrategy, {"script": script}, initial_capital=1000).run()
    # short 100 @10, cover @8 -> +200, then long opens at 8
    assert res.trades.pnl.iloc[0] == pytest.approx(200.0)
    assert res.trades.direction.iloc[0] == "SHORT"
    assert len(res.open_trades) == 1


# ---------------------------------------------------------------- metrics

def test_metrics_on_known_series():
    idx = pd.bdate_range("2020-01-01", periods=253)
    equity = pd.Series(100 * 1.001 ** np.arange(253), index=idx)
    m = compute_metrics(equity, risk_free=0.0)
    assert m["max_drawdown"] == 0
    assert m["total_return"] == pytest.approx(1.001 ** 252 - 1)
    eq2 = pd.Series([100, 120, 90, 110], index=idx[:4], dtype=float)
    assert compute_metrics(eq2)["max_drawdown"] == pytest.approx(-0.25)


def test_multi_asset_and_walk_forward_run():
    frames = generate_synthetic(["A", "B"], "2012-01-01", "2018-12-31", seed=3)
    res = Backtest(PandasDataHandler(frames), MovingAverageCrossStrategy,
                   {"short_window": 10, "long_window": 50}).run()
    assert {"pos_A", "pos_B"} <= set(res.equity_curve.columns)
    wf = walk_forward(frames, MovingAverageCrossStrategy,
                      {"short_window": [10, 20], "long_window": [50, 100]},
                      train_bars=504, test_bars=252, warmup_bars=100, verbose=False)
    assert len(wf.windows) >= 4
    assert wf.oos_equity.index[0] == wf.windows.test_start.iloc[0]
