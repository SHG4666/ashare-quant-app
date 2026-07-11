import pandas as pd
from fastapi.testclient import TestClient


def test_intraday_endpoint_returns_ashare_minute_bars(monkeypatch):
    import ashare_quant.api as api

    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09 14:45:00", "2026-07-09 15:00:00"]),
            "open": [46.98, 47.12],
            "high": [47.19, 47.28],
            "low": [46.81, 46.98],
            "close": [47.12, 47.14],
            "volume": [16749885.0, 21872703.0],
        }
    )
    monkeypatch.setattr(api, "fetch_ashare_lightweight_bars", lambda symbol, frequency, count: sample)

    client = TestClient(api.app)
    response = client.get("/api/stocks/600522/intraday?frequency=15m&count=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "600522"
    assert payload["frequency"] == "15m"
    assert payload["rows"] == 2
    assert payload["data"][-1]["date"] == "2026-07-09T15:00:00"
    assert payload["data"][-1]["close"] == 47.14
