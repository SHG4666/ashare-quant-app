from datetime import date, datetime, timezone

import pandas as pd

from ashare_quant import data


def test_market_session_only_includes_weekday_trading_window():
    assert data.is_mainland_market_session(datetime(2026, 7, 10, 10, 0)) is True
    assert data.is_mainland_market_session(datetime(2026, 7, 10, 21, 0)) is False
    assert data.is_mainland_market_session(datetime(2026, 7, 11, 10, 0)) is False


def test_cloud_utc_time_is_converted_to_china_market_time():
    utc_time = datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc)

    assert data.china_market_now(utc_time) == datetime(2026, 7, 10, 10, 0)
    assert data.is_mainland_market_session(utc_time) is True
    assert data.china_market_today(datetime(2026, 7, 10, 16, 30, tzinfo=timezone.utc)) == date(2026, 7, 11)


def test_expected_latest_business_day_uses_friday_for_weekend_and_preopen_monday():
    assert data.expected_latest_business_day(
        date(2026, 7, 11), now=datetime(2026, 7, 11, 10, 0)
    ) == date(2026, 7, 10)
    assert data.expected_latest_business_day(
        date(2026, 7, 13), now=datetime(2026, 7, 13, 8, 30)
    ) == date(2026, 7, 10)


def test_price_history_freshness_accepts_friday_close_for_saturday_request():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-09", "2026-07-10"]), "close": [10.0, 10.5]})

    assert data.is_price_history_fresh(
        df, end=date(2026, 7, 11), now=datetime(2026, 7, 11, 10, 0)
    ) is True


def test_status_does_not_mark_friday_history_stale_during_monday_market_hours():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-10"]), "close": [10.5]})

    status = data.summarize_price_data_status(
        df,
        symbol="000001",
        start=date(2026, 7, 1),
        end=date(2026, 7, 13),
        adjust="qfq",
        source_name="baostock",
        now=datetime(2026, 7, 13, 10, 0),
    )

    assert status["is_stale"] is False
    assert status["staleness_days"] == 0


def test_require_recent_price_history_rejects_stale_live_provider(monkeypatch):
    df = pd.DataFrame({"date": pd.to_datetime(["2026-08-14"]), "close": [34.75]})
    monkeypatch.setattr(data, "is_recent_market_request", lambda end, now=None: True)

    try:
        data.require_recent_price_history(
            df,
            end=date(2026, 8, 19),
            source_name="baostock",
            now=datetime(2026, 8, 19, 18, 30),
        )
    except RuntimeError as exc:
        assert "仅到 2026-08-14" in str(exc)
        assert "应至少到 2026-08-19" in str(exc)
    else:
        raise AssertionError("expected stale live history to be rejected")


def test_summarize_price_data_status_reports_latest_trade_day_and_adjustment_label():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [1000, 1200],
        }
    )

    status = data.summarize_price_data_status(
        df,
        symbol="000001",
        start=date(2024, 1, 1),
        end=date(2024, 1, 7),
        adjust="qfq",
        source_name="baostock",
    )

    assert status == {
        "symbol": "000001",
        "source_name": "baostock",
        "adjust_label": "前复权",
        "latest_trade_day": "2024-01-05",
        "requested_range": "2024-01-01 至 2024-01-07",
        "row_count": 2,
        "is_stale": False,
        "staleness_days": 0,
        "caption": "000001｜baostock｜前复权｜最新交易日 2024-01-05｜请求区间 2024-01-01 至 2024-01-07｜2 条记录",
    }
