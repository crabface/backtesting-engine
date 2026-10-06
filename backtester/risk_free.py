"""Risk-free rate from FRED (St. Louis Fed).

Default series DTB3 = 3-month Treasury bill, secondary market, in percent.
With FRED_API_KEY set we use the official API; otherwise the public
fredgraph CSV endpoint (no key needed).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd


def fetch_risk_free(start, end=None, series_id: str = "DTB3", api_key: str | None = None) -> pd.Series:
    """Annualized risk-free rate as a decimal (0.05 == 5%), indexed by date."""
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    end = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y-%m-%d")
    api_key = api_key or os.environ.get("FRED_API_KEY")

    if api_key:
        params = urllib.parse.urlencode({
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "observation_start": start, "observation_end": end,
        })
        with urllib.request.urlopen(f"https://api.stlouisfed.org/fred/series/observations?{params}", timeout=30) as r:
            obs = json.load(r)["observations"]
        s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
        df = pd.read_csv(url)
        s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))

    s = pd.to_numeric(s, errors="coerce").ffill().dropna() / 100.0
    s.name = series_id
    return s
