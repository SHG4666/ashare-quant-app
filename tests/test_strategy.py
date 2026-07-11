import pandas as pd

from ashare_quant import backtest, indicators
from ashare_quant.indicators import add_moving_averages
from ashare_quant.backtest import run_ma_cross_backtest, summarize_backtest


def test_add_moving_averages_adds_expected_columns_and_values():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})

    result = add_moving_averages(df, short_window=2, long_window=3)

    assert "ma_short" in result.columns
    assert "ma_long" in result.columns
    assert result.loc[2, "ma_short"] == 2.5
    assert result.loc[2, "ma_long"] == 2.0


def test_add_rsi_reports_overbought_after_persistent_gains():
    df = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15, 16]})

    result = indicators.add_rsi(df, window=3)

    assert "rsi" in result.columns
    assert result.loc[:2, "rsi"].isna().all()
    assert result.loc[3:, "rsi"].eq(100.0).all()


def test_run_ma_cross_backtest_buys_when_short_ma_crosses_above_long_ma():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": [10, 10, 10, 10, 11, 12, 13, 14],
            "high": [10, 10, 10, 10, 11, 12, 13, 14],
            "low": [10, 10, 10, 10, 11, 12, 13, 14],
            "close": [10, 10, 10, 10, 11, 12, 13, 14],
            "volume": [100] * 8,
        }
    )

    result = run_ma_cross_backtest(df, short_window=2, long_window=4, initial_cash=10000)

    assert "signal" in result.columns
    assert "equity" in result.columns
    assert result["signal"].max() == 1
    assert result["equity"].iloc[-1] > 10000


def test_run_rsi_backtest_enters_after_oversold_signal():
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    close = [10, 9, 8, 7, 8, 9, 10, 11, 12]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = backtest.run_rsi_backtest(df, window=3, oversold=35, overbought=65, initial_cash=10000)

    assert "rsi" in result.columns
    assert "signal" in result.columns
    assert result["signal"].max() == 1
    assert result["position"].max() == 1
    assert "BUY" in set(result["action"])


def test_rsi_signal_holds_between_oversold_entry_and_overbought_exit():
    close = [10, 9, 8, 7, 7.5, 8, 8.5, 9.5, 10.5]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close)),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = backtest.run_rsi_backtest(df, window=3, oversold=35, overbought=65)
    entered_at = result.index[result["signal"].eq(1)][0]
    exited_at = result.index[(result.index > entered_at) & result["signal"].eq(0)][0]

    assert result.loc[entered_at:exited_at - 1, "signal"].eq(1).all()


def test_run_macd_backtest_enters_after_dif_crosses_above_signal_line():
    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    close = [10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = backtest.run_macd_backtest(
        df,
        fast_window=3,
        slow_window=6,
        signal_window=3,
        initial_cash=10000,
    )

    assert {"macd_dif", "macd_dea", "macd_hist", "signal", "position", "equity"}.issubset(result.columns)
    assert result["signal"].max() == 1
    assert result["position"].max() == 1
    assert "BUY" in set(result["action"])



def test_run_bollinger_backtest_enters_after_close_breaks_lower_band():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = [10, 10, 10, 10, 10, 7, 9, 10, 11, 12]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = backtest.run_bollinger_backtest(
        df,
        window=5,
        num_std=1.5,
        initial_cash=10000,
    )

    assert {"bb_middle", "bb_upper", "bb_lower", "signal", "position", "equity"}.issubset(result.columns)
    assert result["signal"].max() == 1
    assert result["position"].max() == 1
    assert "BUY" in set(result["action"])


def test_bollinger_signal_holds_until_price_recovers_to_middle_band():
    close = [10, 10, 10, 10, 10, 7, 7.5, 8, 10, 11]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close)),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(close),
        }
    )

    result = backtest.run_bollinger_backtest(df, window=5, num_std=1.5)
    entry = result.index[result["signal"].eq(1)][0]
    exit_ = result.index[(result.index > entry) & result["signal"].eq(0)][0]

    assert result.loc[entry:exit_ - 1, "signal"].eq(1).all()



def test_summarize_backtest_reports_core_metrics():
    df = pd.DataFrame({"equity": [10000, 10500, 10200, 11000], "position": [0, 1, 1, 0]})

    summary = summarize_backtest(df, initial_cash=10000)

    assert summary["final_equity"] == 11000
    assert summary["total_return_pct"] == 10.0
    assert summary["max_drawdown_pct"] < 0
    assert summary["trading_days"] == 4
