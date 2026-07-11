from __future__ import annotations

import pandas as pd


_KLINE_PERIOD_ALIASES = {
    "日K": None,
    "周K": "W-FRI",
    "月K": "M",
    "年K": "Y",
}


def aggregate_price_bars(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Aggregate daily OHLCV rows into the selected display period."""
    if period not in _KLINE_PERIOD_ALIASES:
        raise ValueError(f"unsupported K-line period: {period}")

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for column in required[1:]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if period == "日K" or work.empty:
        return work

    work["_period"] = work["date"].dt.to_period(_KLINE_PERIOD_ALIASES[period])
    return (
        work.groupby("_period", sort=True, observed=True)
        .agg(
            date=("date", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )
