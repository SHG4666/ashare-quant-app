from datetime import date

import pandas as pd

from ashare_quant import data


def test_fetch_ashare_daily_does_not_use_unadjusted_ashare_for_qfq(monkeypatch, tmp_path):
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09"]),
            "open": [45.85],
            "high": [47.28],
            "low": [43.28],
            "close": [47.14],
            "volume": [339200056.0],
        }
    )
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("baostock down")))
    monkeypatch.setattr(
        data,
        "fetch_ashare_lightweight_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("qfq must not use unadjusted Ashare history")),
    )

    class FakeAk:
        @staticmethod
        def stock_zh_a_hist(**kwargs):
            return sample.rename(
                columns={"date": "日期", "open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
            )

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAk)
    monkeypatch.setattr(data, "fetch_eastmoney_daily_with_curl", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not reach eastmoney")))

    df = data.fetch_ashare_daily("600522", date(2026, 7, 1), date(2026, 7, 9), "qfq")

    assert len(df) == 1
    assert df.loc[0, "close"] == 47.14
    assert df.attrs["source_name"] == "AkShare/东方财富备用链路"
