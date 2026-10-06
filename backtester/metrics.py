"""Performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def periodic_risk_free(index: pd.DatetimeIndex, risk_free: float | pd.Series, periods_per_year: int) -> pd.Series:
    """Convert an annualized rate (constant or time series) to a per-period rate on `index`."""
    if isinstance(risk_free, pd.Series):
        # Use the last published rate at or before each bar (asof join = no look-ahead).
        rf = risk_free.sort_index().reindex(risk_free.index.union(index)).ffill().reindex(index)
        rf = rf.fillna(risk_free.dropna().iloc[0] if risk_free.notna().any() else 0.0)
    else:
        rf = pd.Series(float(risk_free), index=index)
    return (1 + rf) ** (1 / periods_per_year) - 1


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest stretch (in bars) spent below a prior peak."""
    underwater = (equity < equity.cummax()).to_numpy()
    longest = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return longest


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None,
                    risk_free: float | pd.Series = 0.0, periods_per_year: int = 252,
                    initial_capital: float | None = None) -> dict:
    equity = equity.dropna()
    start_value = initial_capital if initial_capital is not None else equity.iloc[0]
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / start_value - 1
    excess = returns - periodic_risk_free(returns.index, risk_free, periods_per_year)

    n = len(returns)
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else n / periods_per_year
    total_return = equity.iloc[-1] / start_value - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan

    ann = np.sqrt(periods_per_year)
    std = excess.std(ddof=1)
    sharpe = ann * excess.mean() / std if std > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    sortino = ann * excess.mean() / downside if downside > 0 else np.nan

    dd = drawdown(equity)
    max_dd = dd.min()

    m = {
        "start": equity.index[0], "end": equity.index[-1],
        "final_equity": equity.iloc[-1],
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": returns.std(ddof=1) * ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_bars": max_drawdown_duration(equity),
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
    }

    if trades is not None and len(trades):
        wins, losses = trades.loc[trades.pnl > 0, "pnl"], trades.loc[trades.pnl <= 0, "pnl"]
        m.update({
            "num_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_win": wins.mean() if len(wins) else 0.0,
            "avg_loss": losses.mean() if len(losses) else 0.0,
            "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        })
    else:
        m.update({"num_trades": 0, "win_rate": np.nan, "avg_win": np.nan,
                  "avg_loss": np.nan, "profit_factor": np.nan})
    return m


def relative_metrics(returns: pd.Series, benchmark_returns: pd.Series, risk_free: float | pd.Series = 0.0,
                     periods_per_year: int = 252) -> dict:
    """Alpha/beta/information ratio of a strategy versus a benchmark."""
    df = pd.concat([returns, benchmark_returns], axis=1, keys=["s", "b"]).dropna()
    rf = periodic_risk_free(df.index, risk_free, periods_per_year)
    s, b = df["s"] - rf, df["b"] - rf
    var_b = b.var(ddof=1)
    beta = s.cov(b) / var_b if var_b > 0 else np.nan
    alpha = (s.mean() - beta * b.mean()) * periods_per_year
    active = df["s"] - df["b"]
    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    return {
        "alpha_annual": alpha,
        "beta": beta,
        "correlation": df["s"].corr(df["b"]),
        "tracking_error": te,
        "information_ratio": active.mean() * periods_per_year / te if te > 0 else np.nan,
    }


PCT = {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate",
       "alpha_annual", "tracking_error"}


def format_value(key: str, v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if key in PCT:
        return f"{v:.2%}"
    if key == "final_equity" or key.startswith("avg_"):
        return f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.2f}"
