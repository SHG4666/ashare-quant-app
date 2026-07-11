import pandas as pd

from ashare_quant.ui_helpers import (
    aggregate_price_bars,
    indicator_display_columns,
    normalize_ashare_symbol,
    parse_portfolio_symbols,
    strategy_guide,
    strategy_report_slug,
)


def test_aggregate_price_bars_builds_monthly_ohlcv_from_daily_rows():
    daily = pd.DataFrame(
        {
            "date": ["2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04"],
            "open": [10.0, 11.0, 20.0, 21.0],
            "high": [12.0, 13.0, 22.0, 24.0],
            "low": [9.0, 10.0, 19.0, 18.0],
            "close": [11.0, 12.0, 21.0, 23.0],
            "volume": [100.0, 200.0, 300.0, 400.0],
        }
    )

    monthly = aggregate_price_bars(daily, "月K")

    assert len(monthly) == 2
    assert monthly.loc[0, "date"] == pd.Timestamp("2025-01-31")
    assert monthly.loc[0, ["open", "high", "low", "close", "volume"]].tolist() == [10.0, 13.0, 9.0, 12.0, 300.0]
    assert monthly.loc[1, ["open", "high", "low", "close", "volume"]].tolist() == [20.0, 24.0, 18.0, 23.0, 700.0]


def test_aggregate_price_bars_uses_last_trading_day_for_yearly_bar_date():
    daily = pd.DataFrame(
        {
            "date": ["2025-12-31", "2026-01-02"],
            "open": [10.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [10.5, 12.5],
            "volume": [100.0, 200.0],
        }
    )

    yearly = aggregate_price_bars(daily, "年K")

    assert yearly["date"].tolist() == [pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-02")]


def test_parse_portfolio_symbols_accepts_newlines_commas_and_removes_duplicates_preserving_order():
    symbols = parse_portfolio_symbols("000001 平安银行\n600519, 300750\n000001\n，600519 贵州茅台")

    assert symbols == ["000001", "600519", "300750"]


def test_indicator_display_columns_returns_recent_signal_columns_for_each_strategy():
    assert indicator_display_columns("双均线交叉") == [
        "date",
        "close",
        "ma_short",
        "ma_long",
        "signal",
        "position",
        "action",
        "risk_exit",
        "risk_blocked",
        "equity",
        "benchmark_equity",
    ]
    assert indicator_display_columns("MACD金叉")[2:5] == ["macd_dif", "macd_dea", "macd_hist"]
    assert indicator_display_columns("布林带均值回归")[2:5] == ["bb_middle", "bb_upper", "bb_lower"]


def test_strategy_report_slug_maps_ui_strategy_names_to_download_filenames():
    assert strategy_report_slug("双均线交叉") == "ma"
    assert strategy_report_slug("RSI超卖反弹") == "rsi"
    assert strategy_report_slug("MACD金叉") == "macd"
    assert strategy_report_slug("布林带均值回归") == "bollinger"


def test_normalize_ashare_symbol_accepts_code_with_name_and_exchange_suffix():
    assert normalize_ashare_symbol("600519 贵州茅台") == "600519"
    assert normalize_ashare_symbol("sz.000001") == "000001"


def test_strategy_guide_explains_entry_exit_and_risk():
    guide = strategy_guide("RSI超卖反弹")

    assert guide["style"] == "均值回归"
    assert "超卖" in guide["entry"]
    assert "超买" in guide["exit"]
    assert guide["risk"]
