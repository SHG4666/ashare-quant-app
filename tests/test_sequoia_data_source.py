import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_quant import data


def test_fetch_sequoia_sqlite_daily_normalizes_local_database_rows(tmp_path):
    db = tmp_path / "sequoia_v2.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE stock_daily (
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, turnover REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO stock_daily VALUES ('000001','2024-01-02',10,11,9,10.5,1000,2000)"
        )
        conn.commit()

    df = data.fetch_sequoia_sqlite_daily("000001", date(2024, 1, 1), date(2024, 1, 31), db)

    assert len(df) == 1
    assert list(df.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]
    assert df.loc[0, "close"] == 10.5


def test_fetch_ashare_daily_uses_sequoia_sqlite_only_for_hfq_after_baostock_failure(monkeypatch, tmp_path):
    db = tmp_path / "sequoia_v2.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE stock_daily (
                symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, turnover REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO stock_daily VALUES ('000001','2024-01-02',10,11,9,10.5,1000,2000)"
        )
        conn.commit()

    monkeypatch.setattr(data, "SEQUOIA_DB_PATH", db)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "fetch_baostock_daily", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("baostock unavailable")))

    df = data.fetch_ashare_daily("000001", date(2024, 1, 1), date(2024, 1, 31), "hfq")

    assert len(df) == 1
    assert df.loc[0, "close"] == 10.5
    assert "后复权" in df.attrs["source_name"]
