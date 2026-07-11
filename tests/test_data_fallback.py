import json
import subprocess
from datetime import date

import pandas as pd

from ashare_quant.data import fetch_eastmoney_daily_with_curl, normalize_eastmoney_klines


def test_normalize_eastmoney_klines_parses_daily_rows():
    payload = {
        "data": {
            "klines": [
                "2024-01-02,10.00,10.50,10.80,9.90,12345,888888,0,0,0,0",
                "2024-01-03,10.50,10.20,10.70,10.10,22345,999999,0,0,0,0",
            ]
        }
    }

    df = normalize_eastmoney_klines(payload)

    assert list(df.columns[:6]) == ["date", "open", "close", "high", "low", "volume"]
    assert df.loc[0, "close"] == 10.5
    assert df.loc[1, "volume"] == 22345


def test_fetch_eastmoney_daily_with_curl_uses_curl_fallback(monkeypatch):
    payload = {"data": {"klines": ["2024-01-02,10,11,12,9,100,1000,0,0,0,0"]}}

    def fake_run(cmd, capture_output, text, timeout, check):
        assert cmd[0] == "curl"
        assert "push2his.eastmoney.com" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    df = fetch_eastmoney_daily_with_curl("000001", date(2024, 1, 1), date(2024, 1, 10), "qfq")

    assert len(df) == 1
    assert df.loc[0, "open"] == 10
    assert df.loc[0, "close"] == 11
