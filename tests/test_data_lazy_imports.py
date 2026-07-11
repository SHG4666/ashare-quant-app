import builtins
from datetime import date

import pandas as pd

from ashare_quant import data


def test_fetch_ashare_daily_does_not_import_akshare_when_baostock_succeeds(monkeypatch, tmp_path):
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-02"]),
            "open": [52.65],
            "high": [54.31],
            "low": [50.60],
            "close": [51.58],
            "volume": [279141616],
        }
    )
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: sample)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "akshare":
            raise AssertionError("akshare should not be imported when baostock succeeds")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = data.fetch_ashare_daily("600522", date(2026, 7, 1), date(2026, 7, 2), "")

    assert result.loc[0, "close"] == 51.58
