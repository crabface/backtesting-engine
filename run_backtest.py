"""Command-line entry point.

Examples:
    python run_backtest.py --symbols SPY --strategy sma --params short_window=50 long_window=200
    python run_backtest.py --symbols SPY QQQ TLT --strategy momentum --params lookback=126 --rf fred
    python run_backtest.py --symbols SPY --strategy sma --walk-forward \\
        --grid short_window=20,50 long_window=100,150,200
    python run_backtest.py --source synthetic --symbols AAA BBB --strategy bollinger
"""

from __future__ import annotations

import argparse
import ast
import os

from backtester import (STRATEGIES, Backtest, FixedBpsSlippage, NoCommission, PandasDataHandler,
                        PercentCommission, PerShareCommission, fetch_risk_free, generate_synthetic,
                        load_alpaca, load_yfinance, run_benchmark, text_report, walk_forward)
from backtester.metrics import format_value
from backtester.plotting import plot_equity

PERIODS = {"1d": 252, "1Day": 252, "1h": 252 * 7, "1Hour": 252 * 7, "1wk": 52, "1Week": 52}


def parse_kv(items: list[str], as_list: bool = False) -> dict:
    out = {}
    for item in items or []:
        key, val = item.split("=", 1)
        vals = [_literal(v) for v in val.split(",")]
        out[key] = vals if as_list else vals[0]
    return out


def _literal(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    p = argparse.ArgumentParser(description="Event-driven backtester")
    p.add_argument("--symbols", nargs="+", default=["SPY"])
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--source", choices=["yfinance", "alpaca", "synthetic"], default="yfinance")
    p.add_argument("--interval", default="1d", help="yfinance: 1d/1h/1wk; alpaca: 1Day/1Hour/15Min")
    p.add_argument("--strategy", choices=list(STRATEGIES), default="sma")
    p.add_argument("--params", nargs="*", help="strategy params, e.g. short_window=50 long_window=200")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--commission", choices=["none", "per_share", "percent"], default="per_share")
    p.add_argument("--commission-rate", type=float, default=None,
                   help="$/share for per_share (default 0.005) or fraction for percent (default 0.001)")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--rf", default="0.0", help="annual risk-free rate (e.g. 0.02) or 'fred'")
    p.add_argument("--walk-forward", action="store_true")
    p.add_argument("--grid", nargs="*", help="walk-forward grid, e.g. short_window=20,50 long_window=100,200")
    p.add_argument("--train-bars", type=int, default=756)
    p.add_argument("--test-bars", type=int, default=252)
    p.add_argument("--out", default="output")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.symbols} from {args.source}...")
    if args.source == "yfinance":
        frames = load_yfinance(args.symbols, args.start, args.end, args.interval)
    elif args.source == "alpaca":
        frames = load_alpaca(args.symbols, args.start, args.end, args.interval)
    else:
        frames = generate_synthetic(args.symbols, args.start, args.end or "2024-12-31")

    risk_free = fetch_risk_free(args.start, args.end) if args.rf == "fred" else float(args.rf)
    if args.commission == "per_share":
        commission = PerShareCommission(args.commission_rate or 0.005)
    elif args.commission == "percent":
        commission = PercentCommission(args.commission_rate or 0.001)
    else:
        commission = NoCommission()
    slippage = FixedBpsSlippage(args.slippage_bps)
    ppy = PERIODS.get(args.interval, 252)
    strategy_cls = STRATEGIES[args.strategy]

    if args.walk_forward:
        grid = parse_kv(args.grid, as_list=True)
        if not grid:
            p.error("--walk-forward requires --grid")
        print(f"Walk-forward: train={args.train_bars} bars, test={args.test_bars} bars, grid={grid}")
        wf = walk_forward(frames, strategy_cls, grid, args.train_bars, args.test_bars,
                          initial_capital=args.capital, risk_free=risk_free, periods_per_year=ppy,
                          commission=commission, slippage=slippage)
        dh = PandasDataHandler(frames, start=wf.oos_equity.index[0])
        bench = run_benchmark(Backtest(dh, strategy_cls, initial_capital=args.capital, commission=commission,
                                       slippage=slippage, risk_free=risk_free, periods_per_year=ppy))
        print("\nOut-of-sample (stitched) metrics:")
        for k, v in wf.metrics.items():
            print(f"  {k:<28}{format_value(k, v):>14}   buy&hold: {format_value(k, bench.metrics.get(k))}")
        wf.windows.to_csv(f"{args.out}/walk_forward_windows.csv", index=False)
        plot_equity(wf.oos_equity, bench.equity, f"{args.strategy} walk-forward (out-of-sample)",
                    f"{args.out}/walk_forward_equity.png")
        print(f"\nSaved {args.out}/walk_forward_equity.png and walk_forward_windows.csv")
        return

    params = parse_kv(args.params)
    dh = PandasDataHandler(frames)
    bt = Backtest(dh, strategy_cls, params, args.capital, commission, slippage,
                  risk_free=risk_free, periods_per_year=ppy)
    result = bt.run()
    bench = run_benchmark(bt)

    print()
    print(text_report(result, bench, risk_free, ppy))
    result.equity_curve.to_csv(f"{args.out}/equity_curve.csv")
    result.trades.to_csv(f"{args.out}/trades.csv", index=False)
    plot_equity(result.equity, bench.equity, f"{args.strategy} {params} vs buy & hold {args.symbols}",
                f"{args.out}/equity_curve.png")
    print(f"\nSaved {args.out}/equity_curve.png, equity_curve.csv, trades.csv")


if __name__ == "__main__":
    main()
