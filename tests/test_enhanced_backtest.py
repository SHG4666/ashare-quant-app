import pandas as pd

from ashare_quant.backtest import (
    analyze_drawdown_periods,
    compare_strategy_backtests,
    extract_trades,
    monthly_return_table,
    optimize_ma_parameters,
    run_equal_weight_portfolio_backtest,
    run_ma_cross_backtest,
    summarize_backtest,
    yearly_return_table,
)


def _trend_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = [10, 10, 10, 10, 11, 12, 13, 12, 11, 10]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000] * len(close),
        }
    )


def test_backtest_supports_fee_and_slippage_columns():
    result = run_ma_cross_backtest(
        _trend_data(),
        short_window=2,
        long_window=4,
        initial_cash=10000,
        fee_rate=0.001,
        slippage_rate=0.002,
    )

    assert "trade_cost" in result.columns
    assert "turnover" in result.columns
    assert result["trade_cost"].sum() > 0
    assert result["equity"].iloc[-1] < run_ma_cross_backtest(
        _trend_data(), short_window=2, long_window=4, initial_cash=10000
    )["equity"].iloc[-1]


def test_optimize_ma_parameters_ranks_valid_window_pairs_by_return():
    result = optimize_ma_parameters(
        _trend_data(),
        short_windows=[2, 3, 4],
        long_windows=[3, 4, 5],
        initial_cash=10000,
    )

    assert list(result.columns) == [
        "short_window",
        "long_window",
        "final_equity",
        "total_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "trade_count",
    ]
    assert not result.empty
    assert (result["short_window"] < result["long_window"]).all()
    assert result["total_return_pct"].is_monotonic_decreasing


def test_extract_trades_returns_buy_and_sell_rows():
    result = run_ma_cross_backtest(_trend_data(), short_window=2, long_window=4, initial_cash=10000)

    trades = extract_trades(result)

    assert not trades.empty
    assert set(trades["action"]).issubset({"BUY", "SELL"})
    assert "price" in trades.columns
    assert "equity" in trades.columns


def test_summary_includes_benchmark_return_and_excess_return():
    result = run_ma_cross_backtest(_trend_data(), short_window=2, long_window=4, initial_cash=10000)

    summary = summarize_backtest(result, initial_cash=10000)

    assert "benchmark_return_pct" in summary
    assert "excess_return_pct" in summary
    assert isinstance(summary["benchmark_return_pct"], float)


def test_equal_weight_portfolio_backtest_averages_member_strategy_returns():
    rising = _trend_data()
    falling = _trend_data().copy()
    falling["close"] = list(reversed(falling["close"].tolist()))
    for col in ["open", "high", "low"]:
        falling[col] = falling["close"]

    portfolio, members = run_equal_weight_portfolio_backtest(
        {"RISING": rising, "FALLING": falling},
        run_ma_cross_backtest,
        initial_cash=10000,
        short_window=2,
        long_window=4,
    )

    assert list(members["symbol"]) == ["RISING", "FALLING"]
    assert {"date", "strategy_return", "equity", "benchmark_equity"}.issubset(portfolio.columns)
    expected_return = pd.concat(
        [
            members.loc[members["symbol"] == "RISING", "result"].iloc[0].set_index("date")["strategy_return"],
            members.loc[members["symbol"] == "FALLING", "result"].iloc[0].set_index("date")["strategy_return"],
        ],
        axis=1,
    ).fillna(0).mean(axis=1)
    pd.testing.assert_series_equal(
        portfolio.set_index("date")["strategy_return"],
        expected_return,
        check_names=False,
    )
    assert portfolio["equity"].iloc[0] == 10000


def test_compare_strategy_backtests_returns_ranked_summary_table():
    def buy_and_hold_runner(df, initial_cash=10000, **kwargs):
        result = df.copy()
        result["signal"] = 1
        result["position"] = 1.0
        result["turnover"] = 0.0
        result["action"] = ""
        result.loc[result.index[0], "turnover"] = 1.0
        result.loc[result.index[0], "action"] = "BUY"
        result["daily_return"] = result["close"].pct_change().fillna(0)
        result["strategy_return"] = result["daily_return"]
        result["equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
        result["benchmark_equity"] = result["equity"]
        result["trade_cost"] = 0.0
        return result

    def cash_runner(df, initial_cash=10000, **kwargs):
        result = df.copy()
        result["signal"] = 0
        result["position"] = 0.0
        result["turnover"] = 0.0
        result["action"] = ""
        result["daily_return"] = result["close"].pct_change().fillna(0)
        result["strategy_return"] = 0.0
        result["equity"] = initial_cash
        result["benchmark_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()
        result["trade_cost"] = 0.0
        return result

    rising_data = _trend_data()
    rising_data["close"] = [10, 10, 10, 10, 11, 12, 13, 14, 15, 16]
    for col in ["open", "high", "low"]:
        rising_data[col] = rising_data["close"]

    comparison = compare_strategy_backtests(
        rising_data,
        {
            "空仓": (cash_runner, {}),
            "买入持有策略": (buy_and_hold_runner, {}),
        },
        initial_cash=10000,
    )

    assert list(comparison["strategy"]) == ["买入持有策略", "空仓"]
    assert "rank" in comparison.columns
    assert comparison.loc[0, "rank"] == 1
    assert comparison.loc[0, "total_return_pct"] > comparison.loc[1, "total_return_pct"]


def test_ma_backtest_stop_loss_exits_position_and_blocks_reentry_until_signal_resets():
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    close = [10, 10, 10, 11, 12, 14, 10.5, 14, 15]
    data = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000] * len(close),
        }
    )

    result = run_ma_cross_backtest(data, short_window=2, long_window=3, initial_cash=10000, stop_loss_pct=10)

    stop_row = result.loc[result["risk_exit"] == "STOP_LOSS"].iloc[0]
    assert stop_row["action"] == "SELL"
    assert stop_row["position"] == 1.0
    assert result.loc[stop_row.name + 1, "position"] == 0.0
    assert result.loc[stop_row.name + 1, "risk_blocked"]


def test_period_return_tables_group_equity_by_month_and_year():
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-29", "2025-01-31"]),
            "equity": [10000.0, 11000.0, 9900.0, 11880.0],
        }
    )

    monthly = monthly_return_table(result)
    yearly = yearly_return_table(result)

    assert monthly[["period", "start_equity", "end_equity", "return_pct"]].to_dict("records") == [
        {"period": "2024-01", "start_equity": 10000.0, "end_equity": 11000.0, "return_pct": 10.0},
        {"period": "2024-02", "start_equity": 11000.0, "end_equity": 9900.0, "return_pct": -10.0},
        {"period": "2025-01", "start_equity": 9900.0, "end_equity": 11880.0, "return_pct": 20.0},
    ]
    assert yearly[["period", "start_equity", "end_equity", "return_pct"]].to_dict("records") == [
        {"period": "2024", "start_equity": 10000.0, "end_equity": 9900.0, "return_pct": -1.0},
        {"period": "2025", "start_equity": 9900.0, "end_equity": 11880.0, "return_pct": 20.0},
    ]


def test_analyze_drawdown_periods_returns_ranked_peak_trough_recovery_windows():
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
            "equity": [10000.0, 12000.0, 9000.0, 11000.0, 12500.0, 10000.0],
        }
    )

    drawdowns = analyze_drawdown_periods(result)

    assert drawdowns.iloc[0].to_dict() == {
        "peak_date": pd.Timestamp("2024-01-02"),
        "trough_date": pd.Timestamp("2024-01-03"),
        "recovery_date": pd.Timestamp("2024-01-05"),
        "max_drawdown_pct": -25.0,
        "duration_days": 3,
        "recovered": True,
    }
    assert drawdowns.iloc[1]["recovered"] is False
