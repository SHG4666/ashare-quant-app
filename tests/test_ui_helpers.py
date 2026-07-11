from ashare_quant.ui_helpers import (
    indicator_display_columns,
    normalize_ashare_symbol,
    parse_portfolio_symbols,
    strategy_guide,
    strategy_report_slug,
)


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
