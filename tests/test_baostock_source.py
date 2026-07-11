from datetime import date

import pandas as pd

from ashare_quant import data


def test_baostock_code_uses_exchange_prefix():
    assert data.baostock_code("600522") == "sh.600522"
    assert data.baostock_code("000001") == "sz.000001"
    assert data.baostock_code("300750") == "sz.300750"


def test_baostock_adjust_flag_matches_ui_adjustment():
    assert data.baostock_adjust_flag("") == "3"      # 不复权
    assert data.baostock_adjust_flag("qfq") == "2"   # 前复权
    assert data.baostock_adjust_flag("hfq") == "1"   # 后复权


def test_normalize_baostock_rows_parses_ohlcv():
    rows = [["2026-07-02", "52.65", "54.31", "50.60", "51.58", "279141616", "14677258175.9"]]
    df = data.normalize_baostock_rows(rows)

    assert list(df.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
    assert df.loc[0, "close"] == 51.58
    assert df.loc[0, "volume"] == 279141616
