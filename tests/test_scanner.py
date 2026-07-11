import pandas as pd

from ashare_quant.backtest import run_ma_cross_backtest
from ashare_quant.scanner import apply_latest_quotes_to_candidates, scan_stock_pool


def _ohlcv(symbol_closes):
    dates = pd.date_range("2024-01-01", periods=len(symbol_closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": symbol_closes,
            "high": symbol_closes,
            "low": symbol_closes,
            "close": symbol_closes,
            "volume": [100] * len(symbol_closes),
        }
    )


def test_scan_stock_pool_returns_candidates_with_signal_reason_and_key_metrics():
    data_by_symbol = {
        "000001": _ohlcv([10, 10, 10, 10, 11, 12, 13, 14]),
        "600519": _ohlcv([14, 13, 12, 11, 10, 10, 10, 10]),
    }

    candidates = scan_stock_pool(
        data_by_symbol,
        run_ma_cross_backtest,
        initial_cash=10000,
        short_window=2,
        long_window=4,
    )

    assert candidates["symbol"].tolist() == ["000001"]
    row = candidates.iloc[0]
    assert row["candidate_action"] == "关注/可按计划买入"
    assert "信号为多头" in row["signal_reason"]
    assert row["close"] == 14
    assert row["latest_signal"] == 1
    assert row["strategy_return_pct"] > 0


def test_scan_stock_pool_continues_after_symbol_failure_and_records_failure_details():
    data_by_symbol = {
        "000001": _ohlcv([10, 10, 10, 10, 11, 12, 13, 14]),
        "000002": _ohlcv([10, 11, 12, 13]),
    }

    def strategy_runner(df, initial_cash, **kwargs):
        if len(df) == 4:
            raise RuntimeError("history too short")
        return run_ma_cross_backtest(df, initial_cash=initial_cash, **kwargs)

    candidates = scan_stock_pool(
        data_by_symbol,
        strategy_runner,
        initial_cash=10000,
        short_window=2,
        long_window=4,
    )

    assert candidates["symbol"].tolist() == ["000001"]
    assert candidates.attrs["failed_symbols"] == [
        {"symbol": "000002", "stage": "strategy", "error": "history too short"}
    ]
    assert candidates.attrs["scan_summary"] == {
        "total_symbols": 2,
        "scanned_symbols": 1,
        "candidate_count": 1,
        "failed_count": 1,
    }


def test_scan_stock_pool_scores_candidates_by_trend_momentum_volume_and_risk():
    data_by_symbol = {
        "000001": _ohlcv([10, 10, 10, 10, 11, 12, 13, 14]),
        "000002": _ohlcv([10, 10, 10, 10, 10.5, 11, 11.5, 12]),
    }
    data_by_symbol["000001"].loc[7, "volume"] = 300
    data_by_symbol["000002"].loc[7, "volume"] = 80

    candidates = scan_stock_pool(
        data_by_symbol,
        run_ma_cross_backtest,
        initial_cash=10000,
        short_window=2,
        long_window=4,
    )

    assert candidates["symbol"].tolist() == ["000001", "000002"]
    top = candidates.iloc[0]
    assert top["selection_score"] > candidates.iloc[1]["selection_score"]
    assert top["trend_score"] > 0
    assert top["momentum_score"] > 0
    assert top["volume_score"] > 0
    assert top["risk_score"] >= 0
    assert "趋势" in top["score_explanation"]
    assert "量能" in top["score_explanation"]


def test_scan_stock_pool_adds_score_label_and_action_hint_for_candidate_priority():
    data_by_symbol = {
        "000001": _ohlcv([10, 10, 10, 10, 11, 12, 13, 14]),
        "000002": _ohlcv([10, 10, 10, 10, 10.5, 11, 11.5, 12]),
    }
    data_by_symbol["000001"].loc[7, "volume"] = 300
    data_by_symbol["000002"].loc[7, "volume"] = 80

    candidates = scan_stock_pool(
        data_by_symbol,
        run_ma_cross_backtest,
        initial_cash=10000,
        short_window=2,
        long_window=4,
    )

    assert candidates["score_label"].tolist() == ["强势候选", "稳健候选"]
    assert candidates.iloc[0]["score_action_hint"] == "优先纳入明日交易计划，等待回踩或突破确认。"
    assert candidates.iloc[1]["score_action_hint"] == "保留观察，需结合行业强弱与仓位预算筛选。"


def test_apply_latest_quotes_replaces_scan_basis_price_and_preserves_scan_metadata():
    candidates = pd.DataFrame(
        [{"symbol": "600522", "close": 313.0, "price_verified": False, "price_source": "本地扫描序列"}]
    )
    candidates.attrs["scan_summary"] = {"candidate_count": 1}

    result = apply_latest_quotes_to_candidates(
        candidates,
        {"600522": {"price": 45.91, "source_name": "腾讯批量行情（不复权）"}},
    )

    assert result.iloc[0]["close"] == 45.91
    assert bool(result.iloc[0]["price_verified"]) is True
    assert result.iloc[0]["price_source"] == "腾讯批量行情（不复权）"
    assert result.attrs["scan_summary"] == {"candidate_count": 1}
    assert result.attrs["quote_summary"] == {
        "candidate_count": 1,
        "verified_count": 1,
        "unverified_count": 0,
    }


def test_apply_latest_quotes_leaves_unverified_candidate_blocked_when_quote_missing():
    candidates = pd.DataFrame([{"symbol": "600522", "close": 313.0, "price_verified": False}])

    result = apply_latest_quotes_to_candidates(candidates, {})

    assert result.iloc[0]["close"] == 313.0
    assert bool(result.iloc[0]["price_verified"]) is False
    assert result.attrs["quote_summary"]["unverified_count"] == 1
