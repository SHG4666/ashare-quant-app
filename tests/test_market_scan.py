from pathlib import Path

import pandas as pd

from ashare_quant.watchlist import load_watchlist_entries


def test_load_market_symbols_from_sequoia_reads_distinct_symbols(tmp_path):
    import sqlite3
    from ashare_quant.market_scan import load_market_symbols_from_sequoia

    db = tmp_path / "market.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table stock_daily(symbol text, date text, close real)")
        conn.executemany(
            "insert into stock_daily(symbol, date, close) values(?,?,?)",
            [("600522", "2026-07-09", 47.14), ("000001", "2026-07-09", 10.0), ("600522", "2026-07-08", 44.86)],
        )

    assert load_market_symbols_from_sequoia(db) == ["000001", "600522"]


def test_select_top_candidates_sorts_by_selection_score_and_limits_to_top_n():
    from ashare_quant.market_scan import select_top_candidates

    candidates = pd.DataFrame(
        {
            "symbol": ["000001", "600522", "300750"],
            "selection_score": [70.0, 93.5, 88.0],
            "close": [10.0, 47.14, 200.0],
        }
    )

    top = select_top_candidates(candidates, top_n=2)

    assert top["symbol"].tolist() == ["600522", "300750"]


def test_merge_top_candidates_into_watchlist_preserves_existing_and_adds_metadata(tmp_path):
    from ashare_quant.market_scan import merge_top_candidates_into_watchlist

    watchlist = tmp_path / "watchlist.txt"
    watchlist.write_text("000001 平安银行 | 行业:银行 | 标签:原有\n", encoding="utf-8")
    top = pd.DataFrame(
        {
            "symbol": ["600522", "300750"],
            "selection_score": [93.5, 88.0],
            "score_label": ["强势候选", "稳健候选"],
            "score_explanation": ["趋势强", "动量强"],
            "name": ["中天科技", "宁德时代"],
        }
    )

    merge_top_candidates_into_watchlist(top, watchlist)
    entries = load_watchlist_entries(watchlist)

    assert [entry.symbol for entry in entries] == ["000001", "600522", "300750"]
    added = {entry.symbol: entry for entry in entries}
    assert added["600522"].name == "中天科技"
    assert added["600522"].tags == ()
    assert added["000001"].tags == ()
    assert "评分93.50" in added["600522"].note


def test_score_local_market_symbol_accepts_liquid_uptrend_and_rejects_weak_trend():
    from datetime import date

    from ashare_quant.market_scan import score_local_market_symbol

    dates = pd.bdate_range("2026-03-01", periods=80)
    strong_close = pd.Series([10 + index * 0.12 for index in range(80)])
    strong = pd.DataFrame(
        {
            "symbol": ["600001"] * 80,
            "date": dates,
            "close": strong_close,
            "volume": [2_000_000.0] * 79 + [3_000_000.0],
            "turnover": [200_000_000.0] * 80,
        }
    )
    weak = strong.copy()
    weak["close"] = list(reversed(strong_close.tolist()))

    accepted = score_local_market_symbol(strong, latest_date=dates[-1].date())
    rejected = score_local_market_symbol(weak, latest_date=dates[-1].date())

    assert accepted is not None
    assert accepted["selection_score"] > 0
    assert rejected is None


def test_scan_local_market_candidates_returns_ranked_candidates(tmp_path):
    import sqlite3

    from ashare_quant.market_scan import scan_local_market_candidates

    db = tmp_path / "market_scan.db"
    dates = pd.bdate_range("2026-03-01", periods=80)
    rows = []
    for symbol, daily_gain in [("600001", 0.14), ("000002", 0.08), ("300003", -0.04)]:
        for index, trade_date in enumerate(dates):
            close = 10 + index * daily_gain
            rows.append((symbol, trade_date.date().isoformat(), close, close, close, close, 2_000_000.0, 200_000_000.0))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "create table stock_daily(symbol text, date text, open real, high real, low real, close real, volume real, turnover real)"
        )
        conn.executemany("insert into stock_daily values(?,?,?,?,?,?,?,?)", rows)

    result = scan_local_market_candidates(db, top_n=2, min_turnover=100_000_000.0, lookback_rows=80)

    assert result["symbol"].tolist() == ["600001", "000002"]
    assert result["rank"].tolist() == [1, 2]
    assert result.attrs["scan_summary"]["eligible_symbols"] == 2


def test_load_cloud_market_candidates_filters_turnover_and_restores_summary(tmp_path):
    import json

    from ashare_quant.market_scan import load_cloud_market_candidates

    snapshot = tmp_path / "market_candidates.csv"
    metadata = tmp_path / "market_snapshot.json"
    pd.DataFrame(
        {
            "rank": [1, 2],
            "symbol": ["600001", "000002"],
            "date": ["2026-07-10", "2026-07-10"],
            "selection_score": [90.0, 80.0],
            "ret20_pct": [20.0, 10.0],
            "volume_ratio": [1.5, 1.2],
            "turnover": [200_000_000.0, 80_000_000.0],
        }
    ).to_csv(snapshot, index=False)
    metadata.write_text(json.dumps({"total_symbols": 5207, "latest_date": "2026-07-10"}), encoding="utf-8")

    result = load_cloud_market_candidates(snapshot, metadata, top_n=10, min_turnover=100_000_000.0)

    assert result["symbol"].tolist() == ["600001"]
    assert result["rank"].tolist() == [1]
    assert result.attrs["scan_summary"]["total_symbols"] == 5207


def test_format_feishu_market_scan_message_contains_top10_and_risk_disclaimer():
    from ashare_quant.market_scan import format_feishu_market_scan_message

    top = pd.DataFrame(
        {
            "symbol": ["600522"],
            "selection_score": [93.5],
            "score_label": ["强势候选"],
            "close": [47.14],
            "score_explanation": ["趋势强；动量强"],
        }
    )

    message = format_feishu_market_scan_message(top, total_symbols=5207, scanned_symbols=5207, failed_count=3)

    assert "全市场A股扫描Top10" in message
    assert "600522" in message
    assert "93.50" in message
    assert "不构成投资建议" in message
