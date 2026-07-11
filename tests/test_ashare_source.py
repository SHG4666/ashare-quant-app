import json
import subprocess
from datetime import date

import pandas as pd

from ashare_quant import data


def test_ashare_code_accepts_plain_a_share_symbols():
    assert data.ashare_code("600522") == "sh600522"
    assert data.ashare_code("000001") == "sz000001"
    assert data.ashare_code("300750") == "sz300750"


def test_normalize_ashare_price_frame_converts_index_to_date_column():
    raw = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "volume": [123456.0],
        },
        index=pd.to_datetime(["2026-07-09"]),
    )

    normalized = data.normalize_ashare_price_frame(raw, date(2026, 7, 1), date(2026, 7, 10))

    assert list(normalized.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert normalized.loc[0, "date"] == pd.Timestamp("2026-07-09")
    assert normalized.loc[0, "close"] == 10.5


def test_fetch_ashare_lightweight_daily_uses_external_get_price(monkeypatch):
    raw = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.8, 11.8],
            "low": [9.8, 10.8],
            "close": [10.5, 11.5],
            "volume": [1000.0, 2000.0],
        },
        index=pd.to_datetime(["2026-07-08", "2026-07-09"]),
    )
    calls = []

    def fake_get_price(code, end_date="", count=10, frequency="1d"):
        calls.append((code, end_date, count, frequency))
        return raw

    monkeypatch.setattr(data, "load_ashare_get_price", lambda: fake_get_price)

    df = data.fetch_ashare_lightweight_daily("600522", date(2026, 7, 1), date(2026, 7, 9))

    assert calls[0][0] == "sh600522"
    assert calls[0][1] == "2026-07-09"
    assert calls[0][3] == "1d"
    assert len(df) == 2
    assert df.iloc[-1]["close"] == 11.5


def test_fetch_latest_market_quote_is_explicitly_unadjusted(monkeypatch):
    payload = {
        "data": {
            "f43": 4591,
            "f44": 4978,
            "f45": 4589,
            "f46": 4734,
            "f47": 318389244,
            "f57": "600522",
            "f58": "中天科技",
            "f59": 2,
            "f60": 4714,
            "f86": 1783666740,
        }
    }

    def fake_run(cmd, capture_output, text, timeout, check):
        assert cmd[0] == "curl"
        assert "push2.eastmoney.com" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(data.subprocess, "run", fake_run)

    quote = data.fetch_latest_market_quote("600522")

    assert quote["price"] == 45.91
    assert quote["name"] == "中天科技"
    assert quote["previous_close"] == 47.14
    assert quote["price_basis"] == "不复权实时价"


def test_normalize_tencent_stock_names_parses_multi_symbol_response():
    raw = (
        'v_sh600522="1~中天科技~600522~45.91~47.14";\n'
        'v_sz002747="51~埃斯顿~002747~18.80~18.62";\n'
        'v_sz300750="51~宁德时代~300750~260.00~258.50";'
    )

    names = data.normalize_tencent_stock_names(raw)

    assert names == {"600522": "中天科技", "002747": "埃斯顿", "300750": "宁德时代"}


def test_normalize_tencent_stock_quotes_parses_normal_prices_and_ignores_invalid_price():
    raw = (
        'v_sh600522="1~中天科技~600522~45.91~47.14";\n'
        'v_sz002747="51~埃斯顿~002747~0~18.62";'
    )

    quotes = data.normalize_tencent_stock_quotes(raw)

    assert quotes["600522"]["price"] == 45.91
    assert quotes["600522"]["previous_close"] == 47.14
    assert quotes["600522"]["price_basis"] == "正常市场价格"
    assert quotes["002747"]["price"] is None
