from pathlib import Path


def test_app_displays_data_status_latest_trade_day_and_price_basis():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "summarize_price_data_status" in source
    assert "最新交易日" in source
    assert "价格口径" in source
    assert "aq-status-strip" in source


def test_app_price_chart_uses_china_market_colors_and_always_shows_ma10():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'UP_COLOR = "#e53935"' in source
    assert 'DOWN_COLOR = "#0a9b68"' in source
    assert "increasing_fillcolor=UP_COLOR" in source
    assert "decreasing_fillcolor=DOWN_COLOR" in source
    assert 'name=f"MA{window}"' in source
    assert "for window in (5, 10, 20)" in source
    assert 'side="right"' in source
    assert '"bounds": ["sat", "mon"]' in source
    assert '["日K", "周K", "月K", "年K"]' in source
    assert "aggregate_price_bars(result, period)" in source
    assert "default_visible_bar_range(chart_data, period)" in source
    assert 'dragmode="pan"' in source
    assert '"thickness": 0.075' in source


def test_app_uses_readable_equal_spaced_axes_for_aggregated_kline_periods():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'AGGREGATED_VISIBLE_BARS = {"周K": 52, "月K": 36, "年K": 15}' in source
    assert 'x_values = dates.dt.strftime(date_format)' in source
    assert '"type": "date" if period == "日K" else "category"' in source
    assert 'moving_average.notna().sum() < 2' in source
    assert 'period == "年K" and len(chart_data) < 8' in source


def test_app_uses_china_market_colors_for_metric_deltas():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'stMetricDeltaIcon-Up' in source
    assert 'stMetricDeltaIcon-Down' in source
    assert 'delta_color="inverse"' in source
    assert 'f"{summary[\'excess_return_pct\']:+.2f}% 超额"' in source
    assert '当前信号 · {signal_action}' in source


def test_app_exposes_human_friendly_watchlist_navigation_and_refresh_feedback():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "快速选择自选股" in source
    assert "刷新行情与分析" in source
    assert "section_header" in source
    assert "default_tab_once" in source


def test_app_exposes_ma_parameter_optimizer_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "参数优化" in source
    assert "优化双均线参数" in source
    assert "optimize_ma_parameters" in source
    assert "优化结果 Top" in source


def test_app_exposes_equal_weight_portfolio_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "多股票组合回测" in source
    assert "组合股票代码" in source
    assert "run_equal_weight_portfolio_backtest" in source
    assert "组合成分表现" in source


def test_app_exposes_stop_loss_and_take_profit_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "止损/止盈风控" in source
    assert "启用止损" in source
    assert "启用止盈" in source
    assert "stop_loss_pct" in source
    assert "take_profit_pct" in source


def test_app_exposes_strategy_comparison_report_table():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "策略对比报告" in source
    assert "启用策略对比" in source
    assert "compare_strategy_backtests" in source
    assert "strategy_comparison" in source


def test_app_exposes_html_backtest_report_download():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "build_backtest_html_report" in source
    assert "下载 HTML 回测报告" in source
    assert "text/html" in source
    assert "_backtest_report.html" in source


def test_app_exposes_watchlist_management_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "自选股/股票池" in source
    assert "添加到股票池" in source
    assert "移出股票池" in source
    assert "save_watchlist_entries" in source
    assert "load_watchlist_entries" in source


def test_app_exposes_post_market_stock_pool_scanner():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "自选股策略扫描" in source
    assert "enable_stock_scanner" in source
    assert "scan_stock_pool" in source
    assert "候选股" in source
    assert "signal_reason" in source
    assert "keep_stock_pool_tab_active" in source


def test_app_exposes_local_market_auto_selection_and_watchlist_addition():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "全市场自动选股" in source
    assert "开始全市场自动筛选" in source
    assert "scan_local_market_candidates" in source
    assert "将选中候选加入股票池" in source


def test_app_exposes_next_day_trade_plan_from_scanner_candidates():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "明日交易计划" in source
    assert "generate_trade_plan" in source
    assert "plan_candidates" in source
    assert "下载明日交易计划 CSV" in source


def test_app_exposes_review_journal_and_system_score_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "复盘日志/交易体系评分" in source
    assert "append_review_entry" in source
    assert "load_review_journal" in source
    assert "system_score" in source
    assert "保存复盘记录" in source
