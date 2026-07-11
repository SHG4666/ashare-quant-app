from __future__ import annotations

import pandas as pd


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
