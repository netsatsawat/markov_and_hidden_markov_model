"""Rebuild data/market_macro.csv from public sources.

The notebooks read the bundled CSV so they run without any network access
and give the same numbers every time. Run this script only when you want a
fresh snapshot.

Sources:
- SPY daily adjusted close from Yahoo Finance via yfinance
- Three macro series from FRED's public CSV endpoint (no API key needed):
    VIXCLS   CBOE volatility index
    T10Y2Y   10-year minus 2-year Treasury spread
    T10Y3M   10-year minus 3-month Treasury spread

Two notes on history. The original 2020 version of this tutorial used the
TED spread (TEDRATE), which FRED discontinued in January 2022, and an ICE
BofA high yield series, which FRED now serves only as a short licensed
window. Both are replaced here with fully open series so the dataset can
always be rebuilt in full.
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
FRED_SERIES = ["VIXCLS", "T10Y2Y", "T10Y3M"]
START = "2010-01-01"


def fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = urllib.request.urlopen(url, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw), na_values=".")
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[series_id]


def main() -> None:
    # Fetch the full history and slice locally. Asking Yahoo for a start
    # date sometimes returns a truncated window; period="max" is reliable.
    spy = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    px = spy["Close"]
    if isinstance(px, pd.DataFrame):  # newer yfinance returns a 1-col frame
        px = px.iloc[:, 0]
    out = pd.DataFrame({"SPY": px})
    out.index = out.index.tz_localize(None)
    out = out.loc[START:]
    if len(out) < 3000:
        raise RuntimeError(f"only {len(out)} rows returned; retry the download")

    for sid in FRED_SERIES:
        s = fetch_fred(sid)
        out[sid] = s.reindex(out.index).ffill()

    out["sret"] = np.log(out["SPY"] / out["SPY"].shift(1))
    out = out.dropna()
    dest = HERE / "market_macro.csv"
    out.to_csv(dest, float_format="%.6f")
    print(f"wrote {dest}: {len(out)} rows, {out.index.min().date()} "
          f"to {out.index.max().date()}")


if __name__ == "__main__":
    main()
