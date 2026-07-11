from datetime import date

import pandas as pd

from ashare_quant import data


def test_fetch_ashare_daily_uses_symbol_cache_when_online_sources_fail(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        }
    )
    cached.to_csv(cache_dir / "000001_20230101_20230131_qfq.csv", index=False)

    monkeypatch.setattr(data, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(data, "SEQUOIA_DB_PATH", tmp_path / "missing_sequoia.db")
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("baostock unavailable")))
    monkeypatch.setattr(data, "fetch_ashare_lightweight_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ashare unavailable")))

    class FakeAk:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            raise RuntimeError("akshare unavailable")

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAk)
    monkeypatch.setattr(data, "fetch_eastmoney_daily_with_curl", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("curl unavailable")))

    result = data.fetch_ashare_daily("000001", date(2024, 2, 1), date(2024, 2, 28), "qfq")

    assert len(result) == 1
    assert result.loc[0, "close"] == 10.5


def test_fetch_ashare_daily_raises_concise_error_without_cache_when_online_sources_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "SEQUOIA_DB_PATH", tmp_path / "missing_sequoia.db")
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("baostock unavailable")))
    monkeypatch.setattr(data, "fetch_ashare_lightweight_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ashare unavailable")))

    class FakeAk:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            raise RuntimeError("akshare unavailable")

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAk)
    monkeypatch.setattr(data, "fetch_eastmoney_daily_with_curl", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("curl unavailable")))

    try:
        data.fetch_ashare_daily("000001", date(2024, 2, 1), date(2024, 2, 28), "qfq")
    except RuntimeError as exc:
        assert "在线数据源暂时不可用" in str(exc)
        assert "akshare unavailable" not in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_ashare_daily_uses_fresh_covering_cache_outside_market_hours(monkeypatch, tmp_path):
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-10"]),
            "open": [10.0, 10.4],
            "high": [10.5, 10.8],
            "low": [9.8, 10.2],
            "close": [10.4, 10.7],
            "volume": [1000.0, 1200.0],
        }
    )
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "is_mainland_market_session", lambda: False)
    monkeypatch.setattr(data, "load_fresh_baostock_cache", lambda *args, **kwargs: sample)
    monkeypatch.setattr(
        data,
        "fetch_baostock_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fresh cache should avoid baostock")),
    )

    result = data.fetch_ashare_daily("000001", date(2024, 7, 11), date(2026, 7, 11), "qfq")

    assert result.iloc[-1]["close"] == 10.7


def test_load_cloud_seed_daily_slices_history_and_labels_source(tmp_path):
    history_dir = tmp_path / "watchlist_history"
    history_dir.mkdir()
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-10"]),
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.8, 10.2, 10.8],
            "close": [10.4, 10.9, 11.3],
            "volume": [1000, 1200, 1400],
        }
    )
    sample.to_csv(history_dir / "600522.csv", index=False)

    result = data.load_cloud_seed_daily(
        "600522",
        date(2026, 7, 9),
        date(2026, 7, 10),
        history_dir,
    )

    assert len(result) == 2
    assert result.iloc[-1]["close"] == 11.3
    assert "云端种子行情" in result.attrs["source_name"]
    assert result.attrs["price_verified"] is False
