import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from scripts import build_cloud_seed


def make_history(day: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([day]),
            "open": [close],
            "high": [close + 0.2],
            "low": [close - 0.2],
            "close": [close],
            "volume": [1000.0],
        }
    )


def test_verified_history_falls_back_when_baostock_is_stale(monkeypatch):
    quote = {"quote_time": pd.Timestamp("2026-08-19 15:01"), "price": 11.0}
    monkeypatch.setattr(
        build_cloud_seed,
        "fetch_baostock_daily",
        lambda *args, **kwargs: make_history("2026-08-18", 10.5),
    )
    monkeypatch.setattr(
        build_cloud_seed,
        "fetch_eastmoney_daily_with_curl",
        lambda *args, **kwargs: make_history("2026-08-19", 11.0),
    )

    history, source_name = build_cloud_seed._fetch_verified_history(
        "000001",
        date(2026, 1, 1),
        date(2026, 8, 19),
        quote,
    )

    assert history["date"].max() == pd.Timestamp("2026-08-19")
    assert source_name == "东方财富K线"


def test_verified_history_accepts_prior_session_before_provider_ready():
    history = make_history("2026-08-20", 10.5)
    quote = {"quote_time": pd.Timestamp("2026-08-21 15:23"), "price": 11.0}

    checked = build_cloud_seed._validate_history_against_quote(
        "000001",
        history,
        quote,
        expected_date=date(2026, 8, 20),
    )

    assert checked["date"].max() == pd.Timestamp("2026-08-20")


def test_cloud_seed_updates_symbols_independently_and_preserves_failed_file(monkeypatch, tmp_path):
    cloud_dir = tmp_path / "cloud_data"
    history_dir = cloud_dir / "watchlist_history"
    history_dir.mkdir(parents=True)
    old_failed_history = make_history("2026-08-14", 20.0)
    old_failed_history.to_csv(history_dir / "000002.csv", index=False)

    monkeypatch.setattr(build_cloud_seed, "CLOUD_DATA_DIR", cloud_dir)
    monkeypatch.setattr(build_cloud_seed, "CLOUD_HISTORY_DIR", history_dir)
    monkeypatch.setattr(build_cloud_seed, "load_watchlist", lambda: ["000001", "000002"])
    monkeypatch.setattr(
        build_cloud_seed,
        "fetch_tencent_stock_quotes",
        lambda symbols: {
            "000001": {"quote_time": pd.Timestamp("2026-08-19 15:01"), "price": 11.0},
            "000002": {"quote_time": pd.Timestamp("2026-08-19 15:01"), "price": 21.0},
        },
    )

    def fake_fetch(symbol, start, end, quote):
        if symbol == "000002":
            raise RuntimeError("provider lag")
        return make_history("2026-08-19", 11.0), "baostock"

    monkeypatch.setattr(build_cloud_seed, "_fetch_verified_history", fake_fetch)

    result = build_cloud_seed.build_watchlist_histories(run_date=date(2026, 8, 19))

    assert result["updated_count"] == 1
    assert result["failed_count"] == 1
    assert pd.read_csv(history_dir / "000001.csv")["date"].iloc[-1] == "2026-08-19"
    assert pd.read_csv(history_dir / "000002.csv")["date"].iloc[-1] == "2026-08-14"

    manifest = json.loads((cloud_dir / "watchlist_history_manifest.json").read_text(encoding="utf-8"))
    assert manifest["symbols"]["000001"]["status"] == "updated"
    assert manifest["symbols"]["000002"]["status"] == "stale"
    assert manifest["symbols"]["000002"]["latest_date"] == "2026-08-14"
    assert manifest["failed_symbols"]["000002"] == "provider lag"


def test_market_update_workflow_runs_after_close_with_a_retry():
    source = Path(".github/workflows/update-cloud-market-data.yml").read_text(encoding="utf-8")

    assert 'cron: "30 10,12 * * 1-5"' in source
    assert "timeout-minutes: 30" in source


def test_cloud_builder_uses_delayed_provider_publication_cutoff():
    assert build_cloud_seed._expected_cloud_history_date(
        date(2026, 8, 21), now=datetime(2026, 8, 21, 15, 23)
    ) == date(2026, 8, 20)
    assert build_cloud_seed._expected_cloud_history_date(
        date(2026, 8, 21), now=datetime(2026, 8, 21, 18, 30)
    ) == date(2026, 8, 21)
