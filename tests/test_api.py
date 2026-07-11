from datetime import date

import pandas as pd
from fastapi.testclient import TestClient


def test_health_endpoint_reports_service_name_and_version():
    from ashare_quant.api import app

    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "ashare-quant-api"


def test_daily_price_endpoint_returns_normalized_records(monkeypatch):
    import ashare_quant.api as api

    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "open": [50.0, 51.0],
            "high": [52.0, 53.0],
            "low": [49.5, 50.5],
            "close": [51.5, 52.5],
            "volume": [1000000, 1200000],
        }
    )
    monkeypatch.setattr(api, "fetch_ashare_daily", lambda symbol, start, end, adjust: sample)

    client = TestClient(api.app)
    response = client.get("/api/stocks/600522/daily?start=2026-07-01&end=2026-07-02&adjust=qfq")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "600522"
    assert payload["adjust"] == "qfq"
    assert payload["rows"] == 2
    assert payload["data"][0] == {
        "date": "2026-07-01",
        "open": 50.0,
        "high": 52.0,
        "low": 49.5,
        "close": 51.5,
        "volume": 1000000.0,
    }


def test_ma_cross_backtest_endpoint_returns_summary_and_trades(monkeypatch):
    import ashare_quant.api as api

    sample = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "open": [10 + i * 0.1 for i in range(40)],
            "high": [10.5 + i * 0.1 for i in range(40)],
            "low": [9.5 + i * 0.1 for i in range(40)],
            "close": [10 + i * 0.1 for i in range(40)],
            "volume": [1000000 + i for i in range(40)],
        }
    )
    monkeypatch.setattr(api, "fetch_ashare_daily", lambda symbol, start, end, adjust: sample)

    client = TestClient(api.app)
    response = client.post(
        "/api/backtests/ma-cross",
        json={
            "symbol": "600522",
            "start": "2026-01-01",
            "end": "2026-02-09",
            "adjust": "qfq",
            "short_window": 3,
            "long_window": 8,
            "initial_cash": 100000,
            "fee_rate": 0.0003,
            "slippage_rate": 0.0005,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "600522"
    assert payload["strategy"] == "ma_cross"
    assert payload["summary"]["final_equity"] > 0
    assert "total_return" in payload["summary"]
    assert isinstance(payload["trades"], list)
