from __future__ import annotations

import pandas as pd


_KLINE_PERIOD_ALIASES = {
    "日K": None,
    "周K": "W-FRI",
    "月K": "M",
    "年K": "Y",
}
_DEFAULT_VISIBLE_BARS = {
    "日K": 90,
    "周K": 78,
    "月K": 60,
    "年K": 30,
}


def default_visible_bar_range(df: pd.DataFrame, period: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return a readable initial viewport while preserving access to all bars."""
    if period not in _DEFAULT_VISIBLE_BARS:
        raise ValueError(f"unsupported K-line period: {period}")
    dates = pd.Series(pd.to_datetime(df.get("date"), errors="coerce")).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    visible_count = _DEFAULT_VISIBLE_BARS[period]
    if len(dates) <= visible_count:
        return None

    steps = dates.diff().dropna()
    padding = (steps.median() / 2) if not steps.empty else pd.Timedelta(hours=12)
    return dates.iloc[-visible_count] - padding, dates.iloc[-1] + padding


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
