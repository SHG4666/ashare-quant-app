from datetime import timedelta

import pandas as pd

from ashare_quant import data


def test_recent_qfq_request_refreshes_baostock_instead_of_using_unadjusted_realtime(monkeypatch, tmp_path):
    today = data.date.today()
    start = today - timedelta(days=7)
    old_cache = tmp_path / "baostock_600522_20200101_20200102_qfq.csv"
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    # Exact old-style cache for the same requested range should be ignored for recent data.
    stale_cache = tmp_path / f"baostock_600522_{start:%Y%m%d}_{today:%Y%m%d}_qfq.csv"
    pd.DataFrame(
        {
            "date": pd.to_datetime([today - timedelta(days=2)]),
            "open": [300.0],
            "high": [320.0],
            "low": [290.0],
            "close": [310.0],
            "volume": [1000.0],
        }
    ).to_csv(stale_cache, index=False)

    refreshed = pd.DataFrame(
        {
            "date": pd.to_datetime([today]),
            "open": [45.85],
            "high": [47.28],
            "low": [43.28],
            "close": [47.14],
            "volume": [339200056.0],
        }
    )
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: refreshed)
    monkeypatch.setattr(
        data,
        "fetch_ashare_lightweight_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("qfq must not use unadjusted history")),
    )

    result = data.fetch_ashare_daily("600522", start, today, "qfq")

    assert result.iloc[-1]["close"] == 47.14
    assert result["date"].max().date() == today
    assert result.attrs["source_name"] == "baostock"


def test_summarize_price_data_status_keeps_requested_adjustment_label():
    today = data.date.today()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([today]),
            "open": [45.85],
            "high": [47.28],
            "low": [43.28],
            "close": [47.14],
            "volume": [339200056.0],
        }
    )
    status = data.summarize_price_data_status(
        df,
        symbol="600522",
        start=today - timedelta(days=7),
        end=today,
        adjust="qfq",
        source_name="Ashare新浪/腾讯实时接口",
    )

    assert status["is_stale"] is False
    assert status["adjust_label"] == "前复权"
    assert "可能滞后" not in status["caption"]
