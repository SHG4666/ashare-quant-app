from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ashare_quant.backtest import (
    analyze_drawdown_periods,
    compare_strategy_backtests,
    extract_trades,
    monthly_return_table,
    optimize_ma_parameters,
    run_bollinger_backtest,
    run_equal_weight_portfolio_backtest,
    run_ma_cross_backtest,
    run_macd_backtest,
    run_rsi_backtest,
    summarize_backtest,
    yearly_return_table,
)
from ashare_quant.charting import aggregate_price_bars
from ashare_quant.data import (
    fetch_ashare_daily,
    fetch_latest_market_quote,
    fetch_sequoia_sqlite_daily,
    fetch_stock_names,
    fetch_tencent_stock_quotes,
    is_mainland_market_session,
    load_cloud_seed_daily,
    load_fresh_baostock_cache,
    make_demo_data,
    summarize_price_data_status,
)
from ashare_quant.market_scan import merge_top_candidates_into_watchlist, scan_local_market_candidates
from ashare_quant.report import build_backtest_html_report
from ashare_quant.review_journal import (
    DEFAULT_REVIEW_JOURNAL_PATH,
    append_review_entry,
    load_review_journal,
    summarize_review_journal,
)
from ashare_quant.scanner import apply_latest_quotes_to_candidates, scan_stock_pool
from ashare_quant.trade_plan import generate_trade_plan
from ashare_quant.ui_helpers import (
    indicator_display_columns,
    normalize_ashare_symbol,
    parse_portfolio_symbols,
    strategy_guide,
    strategy_report_slug,
)
from ashare_quant.watchlist import (
    DEFAULT_WATCHLIST_PATH,
    WatchlistEntry,
    annotate_scan_candidates_with_watchlist,
    load_watchlist_entries,
    remove_watchlist_entries,
    save_watchlist_entries,
    upsert_watchlist_entry,
    without_watchlist_tags,
)


STRATEGY_NAMES = ["双均线交叉", "RSI超卖反弹", "MACD金叉", "布林带均值回归"]
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
TABLE_LABELS = {
    "symbol": "代码",
    "name": "名称",
    "industry": "行业",
    "note": "备注",
    "rank": "排名",
    "date": "日期",
    "close": "收盘价",
    "price_verified": "价格已校准",
    "price_source": "价格来源",
    "candidate_action": "建议动作",
    "signal_reason": "入选原因",
    "selection_score": "综合分",
    "score_label": "候选等级",
    "score_action_hint": "下一步",
    "trend_score": "趋势分",
    "momentum_score": "动量分",
    "volume_score": "量能分",
    "risk_score": "风险分",
    "score_explanation": "评分说明",
    "ret5_pct": "5日收益(%)",
    "ret20_pct": "20日收益(%)",
    "ret60_pct": "60日收益(%)",
    "volume_ratio": "量比",
    "turnover": "成交额",
    "volatility20_pct": "20日波动(%)",
    "distance_to_high60_pct": "距60日高点(%)",
    "strategy_return_pct": "策略收益(%)",
    "benchmark_return_pct": "基准收益(%)",
    "max_drawdown_pct": "最大回撤(%)",
    "sharpe": "夏普",
    "trade_count": "交易次数",
    "planned_action": "计划动作",
    "entry_reference_price": "参考价",
    "suggested_position_pct": "建议仓位",
    "suggested_capital": "建议金额",
    "suggested_shares": "建议股数",
    "stop_loss_price": "止损价",
    "target_price": "目标价",
    "block_reason": "限制原因",
    "plan_reason": "计划依据",
    "risk_note": "风险说明",
    "action": "动作",
    "position": "仓位",
    "signal": "信号",
    "equity": "策略权益",
    "benchmark_equity": "基准权益",
}


st.set_page_config(page_title="A股量化研究台", page_icon="📊", layout="wide", initial_sidebar_state="auto")
st.markdown(
    """
    <style>
    :root {
        --ink: #182026;
        --muted: #66727c;
        --line: #dce2e6;
        --line-strong: #cbd3d8;
        --paper: #ffffff;
        --canvas: #f3f5f6;
        --accent: #b42318;
        --accent-dark: #8f1d15;
        --positive: #b42318;
        --negative: #087a55;
        --blue: #2457a6;
        --amber: #a15c00;
        --soft-blue: #edf3fb;
        --soft-red: #fff1f0;
        --soft-green: #eaf7f1;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { max-width: 1460px; padding-top: 1.15rem; padding-bottom: 3.5rem; }
    section[data-testid="stSidebar"] {
        background: #fbfcfc;
        border-right: 1px solid var(--line);
        box-shadow: 8px 0 24px rgba(29, 42, 51, 0.025);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: var(--ink) !important; }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.82rem;
        font-weight: 650;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.05rem; }
    p { line-height: 1.55; }
    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.82rem 0.95rem;
        min-height: 102px;
        box-shadow: 0 1px 2px rgba(24, 32, 38, 0.025);
    }
    [data-testid="stMetric"]:hover { border-color: var(--line-strong); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.48rem; font-weight: 720; }
    .aq-brand { font-size: 1.08rem; font-weight: 750; color: var(--ink); margin-bottom: 0.08rem; }
    .aq-brand-mark {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 2px;
        margin-right: 8px;
        background: var(--accent);
        box-shadow: 10px 0 0 var(--blue), 20px 0 0 var(--negative);
    }
    .aq-workbench-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.25rem;
        padding: 0.35rem 0 1rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--line);
    }
    .aq-kicker { color: var(--accent); font-size: 0.72rem; font-weight: 760; margin-bottom: 0.22rem; }
    .aq-title { color: var(--ink); font-size: 1.72rem; font-weight: 760; line-height: 1.18; margin: 0; }
    .aq-subtitle { color: var(--muted); font-size: 0.88rem; margin-top: 0.28rem; }
    .aq-head-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.45rem; }
    .aq-head-meta span {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 0.2rem 0.58rem;
        border: 1px solid var(--line);
        border-radius: 4px;
        background: var(--paper);
        color: #46525b;
        font-size: 0.76rem;
        font-weight: 620;
    }
    .aq-guide {
        background: var(--paper);
        border: 1px solid var(--line);
        border-left: 3px solid var(--blue);
        border-radius: 6px;
        padding: 0.72rem 0.9rem;
        margin: 0.55rem 0 0.8rem;
    }
    .aq-guide strong { color: var(--ink); }
    .aq-guide span { color: var(--muted); font-size: 0.86rem; }
    .aq-stock-heading { margin: 1rem 0 0.65rem; }
    .aq-stock-heading h2 { margin: 0; font-size: 1.3rem; font-weight: 760; }
    .aq-stock-heading p { margin: 0.18rem 0 0; color: var(--muted); font-size: 0.82rem; }
    .aq-status-strip,
    .aq-scan-summary {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem 0.7rem;
        margin: 0.72rem 0 0.9rem;
        padding: 0.62rem 0.78rem;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--paper);
        color: #53606a;
        font-size: 0.78rem;
    }
    .aq-status-strip strong { color: var(--ink); }
    .aq-status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--negative); }
    .aq-status-dot.stale { background: var(--amber); }
    .aq-status-pill {
        padding: 0.14rem 0.42rem;
        border-radius: 3px;
        background: #f0f3f5;
        color: #43515b;
        font-weight: 620;
    }
    .aq-section-head { margin: 0.2rem 0 0.72rem; }
    .aq-section-step { color: var(--accent); font-size: 0.7rem; font-weight: 760; margin-bottom: 0.12rem; }
    .aq-section-title { color: var(--ink); font-size: 1.06rem; font-weight: 740; }
    .aq-section-subtitle { color: var(--muted); font-size: 0.82rem; margin-top: 0.16rem; }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.45rem;
        border-radius: 5px;
        font-weight: 650;
    }
    div[data-baseweb="tab-list"] {
        gap: 1.25rem;
        border-bottom: 1px solid var(--line);
        background: var(--canvas);
    }
    button[data-baseweb="tab"] { padding: 0.55rem 0.08rem 0.62rem; font-weight: 650; }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
        background: var(--paper);
    }
    [data-testid="stExpander"] {
        border-color: var(--line) !important;
        border-radius: 6px !important;
        background: rgba(255, 255, 255, 0.65);
    }
    [data-baseweb="input"], [data-baseweb="select"] > div {
        border-color: var(--line) !important;
        border-radius: 5px !important;
    }
    [data-testid="stAlert"] { border-radius: 6px; }
    @media (max-width: 760px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        .aq-workbench-head { align-items: flex-start; flex-direction: column; gap: 0.7rem; }
        .aq-title { font-size: 1.35rem; }
        .aq-head-meta { justify-content: flex-start; }
        [data-testid="stMetric"] { min-height: 88px; }
        .aq-status-strip { align-items: flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_table(df: pd.DataFrame, columns: list[str] | None = None, height: int | None = None) -> None:
    shown = df.copy()
    if columns is not None:
        shown = shown[[column for column in columns if column in shown.columns]]
    shown = shown.rename(columns=TABLE_LABELS)
    table_config: dict[str, object] = {}
    for column in shown.columns:
        if column in {"代码", "名称"}:
            width = "small" if column == "代码" else "medium"
            table_config[column] = st.column_config.TextColumn(width=width)
        elif column == "排名":
            table_config[column] = st.column_config.NumberColumn(format="%d", width="small")
        elif column == "价格已校准":
            table_config[column] = st.column_config.CheckboxColumn(width="small", disabled=True)
        elif column in {"收盘价", "参考价", "止损价", "目标价"}:
            table_config[column] = st.column_config.NumberColumn(format="¥ %.2f", width="small")
        elif column in {"综合分", "趋势分", "动量分", "量能分", "风险分", "夏普"}:
            table_config[column] = st.column_config.NumberColumn(format="%.2f", width="small")
        elif column.endswith("(%)") or column in {"策略收益(%)", "基准收益(%)", "最大回撤(%)"}:
            table_config[column] = st.column_config.NumberColumn(format="%.2f%%", width="small")
    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        height=height,
        row_height=36,
        column_config=table_config,
    )


def section_header(title: str, subtitle: str = "", step: str = "") -> None:
    step_html = f'<div class="aq-section-step">{escape(step)}</div>' if step else ""
    subtitle_html = f'<div class="aq-section-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="aq-section-head">{step_html}<div class="aq-section-title">{escape(title)}</div>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def select_watchlist_symbol() -> None:
    selected = str(st.session_state.get("quick_watchlist_symbol", "")).strip()
    if selected:
        st.session_state["analysis_symbol_input"] = selected


def sync_manual_symbol_input() -> None:
    selected = str(st.session_state.get("quick_watchlist_symbol", "")).strip()
    manual = str(st.session_state.get("analysis_symbol_input", "")).strip()
    if selected and manual != selected:
        st.session_state["quick_watchlist_symbol"] = ""


def keep_stock_pool_tab_active() -> None:
    """Keep scan results visible across the button-triggered Streamlit rerun."""
    st.session_state["default_tab_once"] = "股票池与计划"


@st.cache_data(show_spinner=False)
def load_data_cached(symbol: str, start: date, end: date, adjust: str, use_demo: bool) -> pd.DataFrame:
    if use_demo:
        demo = make_demo_data()
        demo.attrs["source_name"] = "离线演示数据"
        return demo
    return fetch_ashare_daily(symbol, start, end, adjust)


@st.cache_data(ttl=30, show_spinner=False)
def load_quote_cached(symbol: str) -> dict[str, object]:
    return fetch_latest_market_quote(symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def run_auto_market_scan_cached(top_n: int, min_turnover: float) -> pd.DataFrame:
    return scan_local_market_candidates(top_n=top_n, min_turnover=min_turnover)


def current_strategy_runner_and_kwargs() -> tuple[object, dict[str, object]]:
    common = {
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "stop_loss_pct": float(stop_loss_pct) if enable_stop_loss else None,
        "take_profit_pct": float(take_profit_pct) if enable_take_profit else None,
    }
    if strategy_name == "双均线交叉":
        return run_ma_cross_backtest, {**common, "short_window": short_window, "long_window": long_window}
    if strategy_name == "RSI超卖反弹":
        return run_rsi_backtest, {
            **common,
            "window": rsi_window,
            "oversold": float(rsi_oversold),
            "overbought": float(rsi_overbought),
        }
    if strategy_name == "MACD金叉":
        return run_macd_backtest, {
            **common,
            "fast_window": macd_fast_window,
            "slow_window": macd_slow_window,
            "signal_window": macd_signal_window,
        }
    return run_bollinger_backtest, {**common, "window": boll_window, "num_std": float(boll_num_std)}


def run_current_backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    runner, kwargs = current_strategy_runner_and_kwargs()
    result = runner(data, initial_cash=float(initial_cash), **kwargs)
    return result, summarize_backtest(result, float(initial_cash)), extract_trades(result)


def build_strategy_comparison_configs() -> dict[str, tuple[object, dict[str, object]]]:
    common = {
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "stop_loss_pct": float(stop_loss_pct) if enable_stop_loss else None,
        "take_profit_pct": float(take_profit_pct) if enable_take_profit else None,
    }
    return {
        "双均线交叉": (run_ma_cross_backtest, {**common, "short_window": 5, "long_window": 20}),
        "RSI超卖反弹": (run_rsi_backtest, {**common, "window": 14, "oversold": 30.0, "overbought": 70.0}),
        "MACD金叉": (run_macd_backtest, {**common, "fast_window": 12, "slow_window": 26, "signal_window": 9}),
        "布林带均值回归": (run_bollinger_backtest, {**common, "window": 20, "num_std": 2.0}),
    }


def run_stock_pool_scan_for_symbols(symbols: list[str], progress_callback=None) -> pd.DataFrame:
    data_by_symbol: dict[str, pd.DataFrame] = {}
    fetch_warnings: list[dict[str, str]] = []
    total_symbols = len(symbols)
    for index, item in enumerate(symbols, start=1):
        if progress_callback is not None:
            progress_callback(index - 1, total_symbols, item)
        try:
            if use_demo:
                item_data = load_data_cached(item, start, end, adjust, True)
                item_data.attrs["price_verified"] = False
            else:
                item_data = load_fresh_baostock_cache(item, start, end, adjust)
                if item_data is None:
                    try:
                        item_data = fetch_sequoia_sqlite_daily(item, start, end)
                        item_data.attrs["source_name"] = "Sequoia-X本地库（扫描信号）"
                        item_data.attrs["price_verified"] = False
                    except Exception:
                        try:
                            item_data = load_cloud_seed_daily(item, start, end)
                        except Exception:
                            item_data = fetch_ashare_daily(item, start, end, adjust)
                            item_data.attrs["price_verified"] = adjust != "hfq"
                else:
                    item_data.attrs["price_verified"] = adjust != "hfq"
            data_by_symbol[item] = item_data
        except Exception as exc:
            fetch_warnings.append({"symbol": item, "stage": "data_fetch", "error": str(exc)})
        if progress_callback is not None:
            progress_callback(index, total_symbols, item)
    if not data_by_symbol:
        empty = pd.DataFrame(columns=["symbol"])
        empty.attrs["failed_symbols"] = fetch_warnings
        empty.attrs["scan_summary"] = {
            "total_symbols": len(symbols),
            "scanned_symbols": 0,
            "candidate_count": 0,
            "failed_count": len(fetch_warnings),
        }
        return empty
    runner, kwargs = current_strategy_runner_and_kwargs()
    scan_result = scan_stock_pool(data_by_symbol, runner, initial_cash=float(initial_cash), **kwargs)
    quote_warning = ""
    unverified_symbols = scan_result.loc[
        ~scan_result.get("price_verified", pd.Series(False, index=scan_result.index)).astype(bool), "symbol"
    ].astype(str).tolist()
    if unverified_symbols and not use_demo:
        if progress_callback is not None:
            progress_callback(total_symbols, total_symbols, "批量校准候选价格")
        try:
            scan_result = apply_latest_quotes_to_candidates(
                scan_result,
                fetch_tencent_stock_quotes(unverified_symbols),
            )
        except Exception as exc:
            quote_warning = str(exc)
            scan_result = apply_latest_quotes_to_candidates(scan_result, {})
    else:
        scan_result = apply_latest_quotes_to_candidates(scan_result, {})
    scan_result.attrs["failed_symbols"] = [*fetch_warnings, *scan_result.attrs.get("failed_symbols", [])]
    scan_result.attrs["quote_warning"] = quote_warning
    scan_result.attrs["scan_summary"] = {
        **scan_result.attrs.get("scan_summary", {}),
        "total_symbols": len(symbols),
        "failed_count": len(scan_result.attrs["failed_symbols"]),
    }
    return scan_result


def build_price_figure(result: pd.DataFrame, trades: pd.DataFrame, period: str = "日K") -> go.Figure:
    up_color = "#e53935"
    down_color = "#0a9b68"
    ma_colors = {5: "#d99100", 10: "#2468d8", 20: "#9b51c7"}
    chart_data = aggregate_price_bars(result, period)
    dates = pd.to_datetime(chart_data["date"])
    closes = pd.to_numeric(chart_data["close"], errors="coerce")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.78, 0.22])
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name=period,
            increasing_line_color=up_color,
            increasing_fillcolor=up_color,
            decreasing_line_color=down_color,
            decreasing_fillcolor=down_color,
            whiskerwidth=0.35,
        ),
        row=1,
        col=1,
    )

    for window in (5, 10, 20):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=closes.rolling(window, min_periods=window).mean(),
                mode="lines",
                name=f"MA{window}",
                line={"color": ma_colors[window], "width": 1.45},
                hovertemplate=f"MA{window} %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if strategy_name == "布林带均值回归" and period == "日K":
        fig.add_trace(go.Scatter(x=result["date"], y=result["bb_middle"], name="中轨", line={"color": "#2457a6", "width": 1.4}), row=1, col=1)
        fig.add_trace(go.Scatter(x=result["date"], y=result["bb_upper"], name="上轨", line={"color": "#9aa0a6", "dash": "dot"}), row=1, col=1)
        fig.add_trace(go.Scatter(x=result["date"], y=result["bb_lower"], name="下轨", line={"color": "#9aa0a6", "dash": "dot"}), row=1, col=1)
    if period == "日K" and not trades.empty:
        buys = trades[trades["action"] == "BUY"]
        sells = trades[trades["action"] == "SELL"]
        fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"], mode="markers", name="买入", marker={"symbol": "triangle-up", "size": 10, "color": up_color}), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["date"], y=sells["price"], mode="markers", name="卖出", marker={"symbol": "triangle-down", "size": 10, "color": down_color}), row=1, col=1)
    volume_colors = [up_color if close >= open_ else down_color for open_, close in zip(chart_data["open"], chart_data["close"])]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=chart_data["volume"],
            name="成交量",
            marker_color=volume_colors,
            opacity=0.72,
            hovertemplate="%{x|%Y-%m-%d}<br>成交量 %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    latest_close = float(closes.iloc[-1])
    latest_color = up_color if float(chart_data["close"].iloc[-1]) >= float(chart_data["open"].iloc[-1]) else down_color
    fig.add_hline(
        y=latest_close,
        row=1,
        col=1,
        line={"color": latest_color, "width": 1, "dash": "dot"},
        annotation_text=f"{latest_close:.2f}",
        annotation_position="right",
        annotation_font={"color": latest_color, "size": 11},
    )

    market_dates = pd.DatetimeIndex(dates.dt.normalize().unique())
    weekday_dates = pd.date_range(market_dates.min(), market_dates.max(), freq="B")
    market_holidays = weekday_dates.difference(market_dates)
    fig.update_layout(
        template="plotly_white",
        height=610,
        margin={"l": 12, "r": 62, "t": 32, "b": 12},
        xaxis_rangeslider_visible=False,
        hovermode="x",
        hoverdistance=40,
        spikedistance=-1,
        legend={"orientation": "h", "y": 1.025, "x": 0, "font": {"size": 12}},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#38434d"},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": "#cfd6dc", "font": {"color": "#182026"}},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        bargap=0.18,
    )
    fig.update_xaxes(
        rangebreaks=[{"bounds": ["sat", "mon"]}, {"values": market_holidays}],
        showgrid=True,
        gridcolor="#edf0f2",
        gridwidth=1,
        showspikes=True,
        spikecolor="#7f8b94",
        spikethickness=1,
        spikedash="dot",
        spikesnap="cursor",
        tickformat={"日K": "%m-%d", "周K": "%m-%d", "月K": "%Y-%m", "年K": "%Y"}[period],
    )
    fig.update_yaxes(title_text="价格", side="right", tickformat=".2f", row=1, col=1, gridcolor="#e7eaed", zeroline=False, fixedrange=False)
    fig.update_yaxes(title_text="成交量", side="right", tickformat="~s", row=2, col=1, gridcolor="#eef0f2", zeroline=False, fixedrange=False)
    return fig


def build_indicator_figure(result: pd.DataFrame) -> go.Figure | None:
    fig = go.Figure()
    if strategy_name == "RSI超卖反弹":
        fig.add_trace(go.Scatter(x=result["date"], y=result["rsi"], name=f"RSI{rsi_window}", line={"color": "#2457a6"}))
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="#b42318", annotation_text="超卖")
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="#137333", annotation_text="超买")
        fig.update_yaxes(range=[0, 100])
    elif strategy_name == "MACD金叉":
        colors = ["#b42318" if value >= 0 else "#137333" for value in result["macd_hist"]]
        fig.add_trace(go.Bar(x=result["date"], y=result["macd_hist"], name="MACD柱", marker_color=colors))
        fig.add_trace(go.Scatter(x=result["date"], y=result["macd_dif"], name="DIF", line={"color": "#2457a6"}))
        fig.add_trace(go.Scatter(x=result["date"], y=result["macd_dea"], name="DEA", line={"color": "#d97706"}))
    else:
        return None
    fig.update_layout(template="plotly_white", height=260, margin={"l": 16, "r": 16, "t": 16, "b": 10}, hovermode="x unified")
    return fig


watchlist_entries = without_watchlist_tags(load_watchlist_entries(DEFAULT_WATCHLIST_PATH))
watchlist_symbols = [item.symbol for item in watchlist_entries]
st.session_state.setdefault("analysis_symbol_input", "600522")
if st.session_state.get("quick_watchlist_symbol", "") not in {"", *watchlist_symbols}:
    st.session_state["quick_watchlist_symbol"] = ""


with st.sidebar:
    st.markdown('<div class="aq-brand"><span class="aq-brand-mark"></span>A股量化研究台</div>', unsafe_allow_html=True)
    st.caption("盘后研究与交易计划")
    st.divider()
    if watchlist_symbols:
        sidebar_names = {item.symbol: item.name for item in watchlist_entries}
        st.selectbox(
            "快速选择自选股",
            options=[""] + watchlist_symbols,
            format_func=lambda value: "请选择" if not value else f"{value} {sidebar_names.get(value, '')}".strip(),
            key="quick_watchlist_symbol",
            on_change=select_watchlist_symbol,
        )
    symbol_input = st.text_input(
        "股票代码",
        key="analysis_symbol_input",
        on_change=sync_manual_symbol_input,
        help="输入 6 位 A 股代码，可附带股票名称。",
    )
    date_left, date_right = st.columns(2)
    with date_left:
        start = st.date_input("开始日期", value=date.today() - timedelta(days=365 * 2))
    with date_right:
        end = st.date_input("结束日期", value=date.today())
    adjust = st.selectbox(
        "回测价格口径",
        ["qfq", "", "hfq"],
        format_func=lambda value: {"qfq": "前复权（推荐）", "": "不复权", "hfq": "后复权"}[value],
        help="实时行情卡始终显示不复权价格；图表和回测严格使用这里选择的口径。",
    )

    with st.expander("资金与交易成本", expanded=True):
        initial_cash = st.number_input("初始资金", min_value=1_000, max_value=100_000_000, value=100_000, step=10_000)
        fee_rate = st.number_input("单边手续费率", min_value=0.0, max_value=0.05, value=0.0003, step=0.0001, format="%.4f")
        slippage_rate = st.number_input("单边滑点率", min_value=0.0, max_value=0.05, value=0.0005, step=0.0001, format="%.4f")

    with st.expander("止损/止盈风控", expanded=False):
        enable_stop_loss = st.toggle("启用止损", value=True)
        stop_loss_pct = st.number_input("止损幅度(%)", min_value=0.1, max_value=90.0, value=8.0, step=0.5, disabled=not enable_stop_loss)
        enable_take_profit = st.toggle("启用止盈", value=False)
        take_profit_pct = st.number_input("止盈幅度(%)", min_value=0.1, max_value=500.0, value=20.0, step=0.5, disabled=not enable_take_profit)

    with st.expander("数据选项", expanded=False):
        use_demo = st.checkbox("使用离线演示数据", value=False, help="只在查看软件功能时使用；演示数据不会冒充真实股票。")
        st.caption("在线失败时不会自动替换成模拟行情。")
    run = st.button("刷新行情与分析", type="primary", width="stretch", icon=":material/refresh:")
    if run:
        load_data_cached.clear()
        load_quote_cached.clear()
    st.caption("参数修改后自动重新计算。仅用于研究学习，不构成投资建议。")


try:
    symbol = normalize_ashare_symbol(symbol_input)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if start >= end:
    st.error("开始日期必须早于结束日期。")
    st.stop()


adjustment_context_label = {"qfq": "前复权", "": "不复权", "hfq": "后复权"}[adjust]
st.markdown(
    f'''
    <div class="aq-workbench-head">
        <div>
            <div class="aq-kicker">研究工作区</div>
            <div class="aq-title">A股量化研究台</div>
            <div class="aq-subtitle">行情校验、策略回测、股票筛选与交易计划</div>
        </div>
        <div class="aq-head-meta">
            <span>标的 {escape(symbol)}</span>
            <span>{start.isoformat()} 至 {end.isoformat()}</span>
            <span>{adjustment_context_label}</span>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

strategy_name = st.segmented_control("策略模板", STRATEGY_NAMES, default="双均线交叉", selection_mode="single") or "双均线交叉"
guide = strategy_guide(strategy_name)
st.markdown(
    f'<div class="aq-guide"><strong>{guide["style"]} · {strategy_name}</strong><br>'
    f'<span>{guide["summary"]}<br>进入：{guide["entry"]}　退出：{guide["exit"]}　主要风险：{guide["risk"]}</span></div>',
    unsafe_allow_html=True,
)

with st.expander("调整策略参数", expanded=False):
    if strategy_name == "双均线交叉":
        p1, p2 = st.columns(2)
        short_window = p1.slider("短均线窗口", 2, 60, 5)
        long_window = p2.slider("长均线窗口", 5, 250, 20)
        strategy_params = (short_window, long_window)
        if short_window >= long_window:
            st.error("短均线窗口必须小于长均线窗口。")
            st.stop()
    elif strategy_name == "RSI超卖反弹":
        p1, p2, p3 = st.columns(3)
        rsi_window = p1.slider("RSI窗口", 2, 60, 14)
        rsi_oversold = p2.slider("RSI超卖阈值", 5, 50, 30)
        rsi_overbought = p3.slider("RSI超买阈值", 50, 95, 70)
        strategy_params = (rsi_window, rsi_oversold, rsi_overbought)
        if rsi_oversold >= rsi_overbought:
            st.error("RSI超卖阈值必须小于超买阈值。")
            st.stop()
    elif strategy_name == "MACD金叉":
        p1, p2, p3 = st.columns(3)
        macd_fast_window = p1.slider("MACD快线EMA窗口", 2, 60, 12)
        macd_slow_window = p2.slider("MACD慢线EMA窗口", 5, 120, 26)
        macd_signal_window = p3.slider("MACD信号线窗口", 2, 60, 9)
        strategy_params = (macd_fast_window, macd_slow_window, macd_signal_window)
        if macd_fast_window >= macd_slow_window:
            st.error("MACD快线EMA窗口必须小于慢线EMA窗口。")
            st.stop()
    else:
        p1, p2 = st.columns(2)
        boll_window = p1.slider("布林带窗口", 5, 120, 20)
        boll_num_std = p2.slider("标准差倍数", 0.5, 4.0, 2.0, step=0.1)
        strategy_params = (boll_window, boll_num_std)


current_params = (
    symbol,
    start,
    end,
    adjust,
    use_demo,
    strategy_name,
    *strategy_params,
    float(initial_cash),
    float(fee_rate),
    float(slippage_rate),
    bool(enable_stop_loss),
    float(stop_loss_pct),
    bool(enable_take_profit),
    float(take_profit_pct),
)
if run or st.session_state.get("last_params") != current_params:
    try:
        with st.spinner("正在校验行情并运行策略..."):
            data = load_data_cached(symbol, start, end, adjust, use_demo)
            result, summary, trades = run_current_backtest(data)
    except Exception as exc:
        st.error(f"{symbol} 行情获取失败：{exc}")
        st.info("为避免价格和标的错配，本次没有使用模拟数据代替。可稍后重试，或明确勾选“使用离线演示数据”。")
        st.stop()
    st.session_state.update(
        last_result=result,
        last_summary=summary,
        last_trades=trades,
        last_data=data,
        last_params=current_params,
    )
    if run:
        st.toast("行情与策略分析已刷新", icon=":material/check_circle:")

result = st.session_state["last_result"]
summary = st.session_state["last_summary"]
trades = st.session_state["last_trades"]
data = st.session_state["last_data"]
data_status = summarize_price_data_status(
    data,
    symbol=symbol,
    start=start,
    end=end,
    adjust=adjust,
    source_name=data.attrs.get("source_name", "离线演示数据" if use_demo else "未知数据源"),
)

quote: dict[str, object] | None = None
quote_error = ""
if not use_demo and is_mainland_market_session():
    try:
        quote = load_quote_cached(symbol)
    except Exception as exc:
        quote_error = str(exc)
elif not use_demo:
    quote_error = "当前为非交易时段"

watchlist_name = next((item.name for item in watchlist_entries if item.symbol == symbol and item.name), "")
stock_name = str(quote.get("name", "")) if quote else ""
stock_name = stock_name or watchlist_name or str(data.attrs.get("stock_name", "")) or "A股标的"

st.markdown(
    f'<div class="aq-stock-heading"><h2>{escape(stock_name)} · {escape(symbol)}</h2>'
    f'<p>{escape(strategy_name)} · {escape(data_status["adjust_label"])}价格口径</p></div>',
    unsafe_allow_html=True,
)
if quote:
    quote_price = float(quote["price"])
    quote_day = pd.Timestamp(quote["quote_time"]).normalize()
    reference_close = float(quote.get("previous_close") or 0)
    if not reference_close:
        history_dates = pd.to_datetime(data["date"]).dt.normalize()
        earlier = data.loc[history_dates < quote_day, "close"]
        reference_close = float(earlier.iloc[-1]) if not earlier.empty else float(data["close"].iloc[-2] if len(data) > 1 else data["close"].iloc[-1])
    quote_change_pct = (quote_price / reference_close - 1) * 100 if reference_close else 0.0
    quote_time_label = pd.Timestamp(quote["quote_time"]).strftime("%Y-%m-%d %H:%M")
else:
    quote_price = float(data["close"].iloc[-1])
    quote_change_pct = float(data["close"].pct_change().iloc[-1] * 100) if len(data) > 1 else 0.0
    quote_time_label = data_status["latest_trade_day"]

latest = result.iloc[-1]
signal_label = "持有 / 关注" if int(latest.get("signal", 0)) == 1 else "空仓 / 等待"
metric_cols = st.columns(5)
metric_cols[0].metric("实时价格" if quote else "最新历史收盘", f"¥{quote_price:,.2f}", f"{quote_change_pct:+.2f}%")
metric_cols[1].metric("当前信号", signal_label, str(latest.get("action", "")) or "无新动作")
metric_cols[2].metric("策略总收益", f"{summary['total_return_pct']:.2f}%", f"超额 {summary['excess_return_pct']:+.2f}%")
metric_cols[3].metric("最大回撤", f"{summary['max_drawdown_pct']:.2f}%")
metric_cols[4].metric("夏普比率", f"{summary['sharpe']:.2f}", f"{summary['trade_count']} 次交易")
status_label = "数据可能滞后" if data_status["is_stale"] else "数据已校验"
status_tone = "stale" if data_status["is_stale"] else ""
quote_mode = "实时行情" if quote else "历史收盘"
quote_detail = (
    f'{quote.get("price_basis")} · {quote.get("source_name")} · {quote_time_label}'
    if quote
    else f'{quote_error or "实时行情暂不可用"} · 当前使用最近收盘'
)
st.markdown(
    f'<div class="aq-status-strip"><span class="aq-status-dot {status_tone}"></span>'
    f'<strong>{escape(status_label)}</strong><span class="aq-status-pill">{escape(quote_mode)}</span>'
    f'<span>{escape(quote_detail)}</span><span class="aq-status-pill">{escape(data_status["adjust_label"])}</span>'
    f'<span>最新交易日 {escape(data_status["latest_trade_day"])}</span>'
    f'<span>{escape(data_status["source_name"])} · {data_status["row_count"]} 条</span></div>',
    unsafe_allow_html=True,
)
if data_status["is_stale"]:
    st.warning(f"历史行情距请求结束日约 {data_status['staleness_days']} 个工作日，请先确认数据源是否已经收盘更新。")


watchlist_save_toast = st.session_state.pop("watchlist_save_toast", "")
if watchlist_save_toast:
    st.toast(watchlist_save_toast, icon=":material/check_circle:")
default_tab = st.session_state.pop("default_tab_once", "研究总览")
overview_tab, backtest_tab, scanner_tab, review_tab = st.tabs(
    ["研究总览", "回测明细", "股票池与计划", "复盘日志"],
    default=default_tab,
)

with overview_tab:
    st.subheader("价格、指标与买卖点")
    chart_period = st.segmented_control(
        "K线周期",
        ["日K", "周K", "月K", "年K"],
        default="日K",
        selection_mode="single",
        key="price_chart_period",
    ) or "日K"
    st.plotly_chart(build_price_figure(result, trades, chart_period), use_container_width=True, config=PLOT_CONFIG)
    indicator_fig = build_indicator_figure(result)
    if indicator_fig is not None:
        st.plotly_chart(indicator_fig, use_container_width=True, config=PLOT_CONFIG)

    st.subheader("策略权益与基准")
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=result["date"], y=result["equity"], name="策略权益", line={"color": "#b42318", "width": 2}))
    equity_fig.add_trace(go.Scatter(x=result["date"], y=result["benchmark_equity"], name="买入持有", line={"color": "#687078", "dash": "dash"}))
    equity_fig.update_layout(template="plotly_white", height=330, margin={"l": 16, "r": 16, "t": 16, "b": 10}, hovermode="x unified")
    st.plotly_chart(equity_fig, use_container_width=True, config=PLOT_CONFIG)

    st.subheader("近期信号")
    display_cols = indicator_display_columns(strategy_name)
    display_table(result.tail(30), columns=display_cols, height=380)


with backtest_tab:
    st.subheader("收益、风险与交易质量")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("最终权益", f"{summary['final_equity']:,.2f}")
    k2.metric("年化收益", f"{summary['annual_return_pct']:.2f}%")
    k3.metric("买入持有收益", f"{summary['benchmark_return_pct']:.2f}%")
    k4.metric("交易成本", f"{summary['total_trade_cost']:,.2f}")
    k5.metric("样本交易日", f"{summary['trading_days']}")

    action_col, optimizer_col = st.columns(2)
    with action_col:
        st.markdown("#### 策略对比报告")
        enable_strategy_comparison = st.button("启用策略对比并运行", key="enable_strategy_comparison", width="stretch")
        if enable_strategy_comparison:
            st.session_state["strategy_comparison"] = compare_strategy_backtests(
                data, build_strategy_comparison_configs(), initial_cash=float(initial_cash)
            )
        strategy_comparison = st.session_state.get("strategy_comparison")
        if strategy_comparison is not None:
            display_table(strategy_comparison, height=260)
    with optimizer_col:
        st.markdown("#### 参数优化")
        enable_optimizer = st.checkbox("优化双均线参数", value=False)
        opt_left, opt_right = st.columns(2)
        opt_short_max = opt_left.number_input("短均线最大值", 3, 60, 12, disabled=not enable_optimizer)
        opt_long_max = opt_right.number_input("长均线最大值", 10, 250, 60, disabled=not enable_optimizer)
        if st.button("运行参数优化", disabled=not enable_optimizer, width="stretch"):
            st.session_state["optimization"] = optimize_ma_parameters(
                data,
                short_windows=range(3, int(opt_short_max) + 1),
                long_windows=range(15, int(opt_long_max) + 1),
                initial_cash=float(initial_cash),
                fee_rate=float(fee_rate),
                slippage_rate=float(slippage_rate),
            )
        optimization = st.session_state.get("optimization")
        if optimization is not None:
            st.caption("优化结果 Top 20（样本内结果，不代表未来表现）")
            display_table(optimization.head(20), height=260)

    st.divider()
    st.subheader("月度/年度收益与回撤区间")
    period_left, period_mid, period_right = st.columns(3)
    with period_left:
        st.caption("月度收益")
        display_table(monthly_return_table(result).tail(24), height=310)
    with period_mid:
        st.caption("年度收益")
        display_table(yearly_return_table(result), height=310)
    with period_right:
        st.caption("主要回撤区间")
        drawdown_periods = analyze_drawdown_periods(result)
        if drawdown_periods.empty:
            st.info("当前权益曲线未形成回撤区间。")
        else:
            display_table(drawdown_periods.head(10), height=310)

    trade_left, trade_right = st.columns(2)
    with trade_left:
        st.subheader("交易记录")
        if trades.empty:
            st.info("当前参数没有产生买卖交易。")
        else:
            display_table(trades, height=340)
    with trade_right:
        st.subheader("最近信号")
        display_table(result.tail(50), columns=indicator_display_columns(strategy_name), height=340)

    with st.expander("多股票组合回测", expanded=False):
        enable_portfolio = st.checkbox("启用等权组合回测", value=False)
        portfolio_symbols_text = st.text_area("组合股票代码", value="000001\n600519\n300750", disabled=not enable_portfolio)
        if st.button("运行组合回测", disabled=not enable_portfolio):
            portfolio_symbols = parse_portfolio_symbols(portfolio_symbols_text)
            if len(portfolio_symbols) < 2:
                st.warning("请输入至少 2 个组合股票代码。")
            else:
                portfolio_data: dict[str, pd.DataFrame] = {}
                portfolio_errors: list[str] = []
                for item in portfolio_symbols:
                    try:
                        portfolio_data[item] = load_data_cached(item, start, end, adjust, use_demo)
                    except Exception as exc:
                        portfolio_errors.append(f"{item}: {exc}")
                if len(portfolio_data) < 2:
                    st.error("可用行情不足 2 只，无法运行组合回测。" + "；".join(portfolio_errors))
                else:
                    runner, kwargs = current_strategy_runner_and_kwargs()
                    st.session_state["portfolio_result"], st.session_state["portfolio_members"] = run_equal_weight_portfolio_backtest(
                        portfolio_data, runner, initial_cash=float(initial_cash), **kwargs
                    )
        if "portfolio_result" in st.session_state:
            portfolio_result = st.session_state["portfolio_result"]
            portfolio_members = st.session_state["portfolio_members"]
            portfolio_summary = summarize_backtest(portfolio_result, float(initial_cash))
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("组合总收益", f"{portfolio_summary['total_return_pct']:.2f}%")
            pc2.metric("组合最大回撤", f"{portfolio_summary['max_drawdown_pct']:.2f}%")
            pc3.metric("组合夏普", f"{portfolio_summary['sharpe']:.2f}")
            st.caption("组合成分表现")
            display_table(portfolio_members.drop(columns=["result"]), height=260)

    st.subheader("导出研究结果")
    strategy_slug = strategy_report_slug(strategy_name)
    html_report = build_backtest_html_report(title=f"{symbol} {strategy_name} 回测报告", summary=summary, result=result, trades=trades)
    d1, d2, d3 = st.columns(3)
    d1.download_button("下载 HTML 回测报告", html_report.encode("utf-8"), file_name=f"{symbol}_{strategy_slug}_backtest_report.html", mime="text/html", width="stretch")
    d2.download_button("下载完整回测结果 CSV", result.to_csv(index=False).encode("utf-8-sig"), file_name=f"{symbol}_{strategy_slug}_backtest.csv", mime="text/csv", width="stretch")
    d3.download_button("下载交易记录 CSV", trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"{symbol}_trades.csv", mime="text/csv", width="stretch", disabled=trades.empty)


with scanner_tab:
    section_header("自选股/股票池管理", "维护自选股票及公司名称，扫描与交易计划会直接使用这里的股票。", "步骤 1")
    watchlist_save_message = st.session_state.pop("watchlist_save_message", "")
    if watchlist_save_message:
        st.success(watchlist_save_message)
    watchlist_entries = without_watchlist_tags(load_watchlist_entries(DEFAULT_WATCHLIST_PATH))
    watchlist_symbols = [item.symbol for item in watchlist_entries]

    with st.expander("添加或更新股票", expanded=not watchlist_entries, icon=":material/add_circle:"):
        with st.form("add_watchlist_stock", clear_on_submit=True):
            add_code_col, add_name_col = st.columns(2)
            new_stock_code = add_code_col.text_input("股票代码", placeholder="例如 600519")
            new_stock_name = add_name_col.text_input("公司名称（可选）", placeholder="留空将自动补全")
            add_industry_col, add_note_col = st.columns(2)
            new_stock_industry = add_industry_col.text_input("行业（可选）")
            new_stock_note = add_note_col.text_input("备注（可选）")
            add_stock = st.form_submit_button(
                "添加到股票池",
                type="primary",
                width="stretch",
                icon=":material/add:",
            )

    if add_stock:
        try:
            new_symbol = normalize_ashare_symbol(new_stock_code)
        except ValueError as exc:
            st.error(str(exc))
        else:
            resolved_name = new_stock_name.strip()
            if not resolved_name:
                try:
                    with st.spinner("正在查询公司名称..."):
                        resolved_name = fetch_stock_names([new_symbol]).get(new_symbol, "")
                except Exception as exc:
                    st.warning(f"公司名称暂时无法自动获取，已先保存股票代码：{exc}")
            already_exists = new_symbol in watchlist_symbols
            watchlist_entries = upsert_watchlist_entry(
                watchlist_entries,
                WatchlistEntry(
                    symbol=new_symbol,
                    name=resolved_name,
                    industry=new_stock_industry.strip(),
                    note=new_stock_note.strip(),
                ),
            )
            save_watchlist_entries(DEFAULT_WATCHLIST_PATH, watchlist_entries)
            action = "已更新" if already_exists else "已添加"
            display_identity = f"{new_symbol} {resolved_name}".strip()
            st.session_state["watchlist_save_message"] = f"{action}：{display_identity}"
            st.session_state["watchlist_save_toast"] = f"{display_identity} {action}"
            keep_stock_pool_tab_active()
            st.rerun()

    st.markdown(f"#### 当前股票 · {len(watchlist_symbols)} 只")
    if watchlist_entries:
        display_table(
            pd.DataFrame(
                [
                    {
                        "symbol": item.symbol,
                        "name": item.name,
                        "industry": item.industry,
                        "note": item.note,
                    }
                    for item in watchlist_entries
                ]
            ),
            height=330,
        )
        names_by_symbol = {item.symbol: item.name for item in watchlist_entries}
        delete_select_col, delete_action_col = st.columns([4, 1])
        delete_symbols = delete_select_col.multiselect(
            "批量移除",
            options=watchlist_symbols,
            format_func=lambda item: f"{item} {names_by_symbol.get(item, '')}".strip(),
            placeholder="选择需要移除的股票",
            label_visibility="collapsed",
        )
        delete_selected = delete_action_col.button(
            "移出股票池",
            disabled=not delete_symbols,
            width="stretch",
            icon=":material/delete_outline:",
        )
        if delete_selected:
            watchlist_entries = remove_watchlist_entries(watchlist_entries, delete_symbols)
            save_watchlist_entries(DEFAULT_WATCHLIST_PATH, watchlist_entries)
            st.session_state["watchlist_save_message"] = f"已删除 {len(delete_symbols)} 只股票。"
            st.session_state["watchlist_save_toast"] = f"已删除 {len(delete_symbols)} 只自选股"
            st.session_state.pop("scanner_candidates", None)
            st.session_state.pop("plan_candidates", None)
            keep_stock_pool_tab_active()
            st.rerun()
    else:
        st.info("股票池为空，请先添加一只股票。")

    st.divider()
    section_header(
        "全市场自动选股",
        "使用本地全市场日线，过滤趋势、流动性和过热风险，再按多维指标评分。",
        "步骤 2 · 可选",
    )
    auto_control_left, auto_control_mid, auto_control_right = st.columns([1, 1.2, 1])
    auto_top_n = auto_control_left.number_input("候选数量", min_value=5, max_value=30, value=10, step=5)
    auto_min_turnover_yi = auto_control_mid.selectbox(
        "最低日成交额",
        [0.5, 1.0, 3.0, 5.0],
        index=1,
        format_func=lambda value: f"{value:g} 亿元",
    )
    auto_exclude_st = auto_control_right.toggle("排除 ST / 退市股", value=True)
    run_auto_selection = st.button(
        "开始全市场自动筛选",
        type="primary",
        width="stretch",
        icon=":material/filter_alt:",
        on_click=keep_stock_pool_tab_active,
    )

    if run_auto_selection:
        try:
            with st.spinner("正在扫描本地全市场行情并计算综合评分..."):
                scan_limit = max(int(auto_top_n) * 3, 30)
                auto_candidates = run_auto_market_scan_cached(
                    top_n=scan_limit,
                    min_turnover=float(auto_min_turnover_yi) * 100_000_000,
                )
                scan_summary = dict(auto_candidates.attrs.get("scan_summary", {}))
                names: dict[str, str] = {}
                if not auto_candidates.empty:
                    try:
                        names = fetch_stock_names(auto_candidates["symbol"].astype(str).tolist())
                    except Exception:
                        pass
                    auto_candidates.insert(2, "name", auto_candidates["symbol"].map(names).fillna(""))
                    if auto_exclude_st and names:
                        normalized_names = auto_candidates["name"].str.upper()
                        auto_candidates = auto_candidates[
                            ~normalized_names.str.contains("ST", regex=False)
                            & ~auto_candidates["name"].str.contains("退", regex=False)
                        ]
                    auto_candidates = auto_candidates.head(int(auto_top_n)).reset_index(drop=True)
                    auto_candidates["rank"] = range(1, len(auto_candidates) + 1)
                scan_summary["candidate_count"] = len(auto_candidates)
                scan_summary["name_lookup_ok"] = bool(names)
        except Exception as exc:
            st.session_state.pop("auto_market_candidates", None)
            st.session_state.pop("auto_market_scan_summary", None)
            st.error(f"全市场自动选股失败：{exc}")
        else:
            st.session_state["auto_market_candidates"] = auto_candidates
            st.session_state["auto_market_scan_summary"] = scan_summary
            st.toast(f"自动筛选完成，得到 {len(auto_candidates)} 只候选", icon=":material/check_circle:")

    auto_candidates = st.session_state.get("auto_market_candidates")
    if auto_candidates is not None:
        auto_summary = st.session_state.get("auto_market_scan_summary", {})
        a1, a2, a3 = st.columns(3)
        a1.metric("数据截止", auto_summary.get("latest_date", "--"))
        a2.metric("全市场股票", f"{int(auto_summary.get('total_symbols', 0)):,} 只")
        a3.metric("通过硬条件", f"{int(auto_summary.get('eligible_symbols', 0)):,} 只")
        if auto_candidates is not None and not auto_candidates.empty and not auto_summary.get("name_lookup_ok", True):
            st.warning("公司名称服务暂时不可用，本轮先显示股票代码；综合评分不受影响。")
        if auto_candidates.empty:
            st.info("当前筛选条件下没有候选股，可适当降低成交额门槛后重试。")
        else:
            display_table(
                auto_candidates,
                columns=[
                    "rank",
                    "symbol",
                    "name",
                    "selection_score",
                    "score_label",
                    "ret5_pct",
                    "ret20_pct",
                    "ret60_pct",
                    "volume_ratio",
                    "turnover",
                    "volatility20_pct",
                    "score_explanation",
                    "risk_note",
                ],
                height=390,
            )
            existing_symbols = {item.symbol for item in watchlist_entries}
            available_auto_symbols = [
                item for item in auto_candidates["symbol"].astype(str).tolist() if item not in existing_symbols
            ]
            candidate_names = dict(zip(auto_candidates["symbol"].astype(str), auto_candidates["name"].astype(str)))
            if available_auto_symbols:
                selected_auto_symbols = st.multiselect(
                    "选择加入股票池的候选",
                    options=available_auto_symbols,
                    default=available_auto_symbols,
                    format_func=lambda item: f"{item} {candidate_names.get(item, '')}".strip(),
                )
                add_auto_candidates = st.button(
                    "将选中候选加入股票池",
                    disabled=not selected_auto_symbols,
                    icon=":material/playlist_add:",
                )
                if add_auto_candidates:
                    selected_rows = auto_candidates[auto_candidates["symbol"].astype(str).isin(selected_auto_symbols)]
                    merge_top_candidates_into_watchlist(selected_rows, DEFAULT_WATCHLIST_PATH)
                    st.session_state["watchlist_save_message"] = f"已加入 {len(selected_rows)} 只自动筛选候选。"
                    st.session_state["watchlist_save_toast"] = f"{len(selected_rows)} 只候选已加入股票池"
                    st.session_state.pop("scanner_candidates", None)
                    st.session_state.pop("plan_candidates", None)
                    keep_stock_pool_tab_active()
                    st.rerun()
            else:
                st.info("本轮候选已经全部在股票池中。")

    st.divider()
    section_header(
        "策略扫描与交易计划",
        "按当前策略扫描股票池，价格校准后可直接生成下一交易日计划。",
        "步骤 3",
    )
    scan_left, scan_right = st.columns([1, 1.35])
    with scan_left:
        st.markdown("#### 自选股策略扫描")
        st.write(f"使用当前策略：**{strategy_name}**")
        st.caption(f"当前股票池 {len(watchlist_symbols)} 只，只展示最新信号仍为多头的标的。")
        enable_stock_scanner = st.button(
            "扫描当前股票池候选股",
            type="primary",
            disabled=not watchlist_symbols,
            width="stretch",
            on_click=keep_stock_pool_tab_active,
            icon=":material/manage_search:",
        )
    with scan_right:
        with st.expander("明日交易计划参数", expanded=True):
            c1, c2, c3 = st.columns(3)
            plan_max_position_pct = c1.number_input("单票仓位上限(%)", 1, 100, 20)
            plan_total_position_cap_pct = c2.number_input("总仓位上限(%)", 1, 100, 50)
            plan_risk_budget_pct = c3.number_input("单笔风险预算(%)", 0.1, 10.0, 1.0, step=0.1)
            c4, c5, c6 = st.columns(3)
            plan_total_risk_budget_pct = c4.number_input("总风险预算(%)", 0.1, 20.0, 3.0, step=0.1)
            plan_stop_loss_pct = c5.number_input("计划止损幅度(%)", 0.1, 90.0, 8.0, step=0.5)
            plan_target_profit_pct = c6.number_input("计划目标收益(%)", 0.1, 500.0, 16.0, step=0.5)

    if enable_stock_scanner:
        scan_progress = st.progress(0.0, text="正在准备股票池扫描...")

        def update_scan_progress(done: int, total: int, current_symbol: str) -> None:
            fraction = done / total if total else 1.0
            scan_progress.progress(
                min(max(fraction, 0.0), 1.0),
                text=f"正在读取 {current_symbol} · {done}/{total}",
            )

        with st.spinner("正在读取本地行情、计算信号并批量校准价格..."):
            try:
                scanner_candidates = annotate_scan_candidates_with_watchlist(
                    run_stock_pool_scan_for_symbols(watchlist_symbols, update_scan_progress), watchlist_entries
                )
            finally:
                scan_progress.empty()
        st.session_state.pop("plan_candidates", None)
        st.session_state["scanner_candidates"] = scanner_candidates

    scanner_candidates = st.session_state.get("scanner_candidates")
    if scanner_candidates is not None:
        scan_summary = scanner_candidates.attrs.get("scan_summary", {})
        failed_symbols = scanner_candidates.attrs.get("failed_symbols", [])
        quote_summary = scanner_candidates.attrs.get("quote_summary", {})
        unverified_count = int(quote_summary.get("unverified_count", 0))
        st.markdown(
            f'<div class="aq-scan-summary"><strong>本轮扫描</strong>'
            f'<span class="aq-status-pill">总数 {scan_summary.get("total_symbols", len(watchlist_symbols))}</span>'
            f'<span class="aq-status-pill">成功 {scan_summary.get("scanned_symbols", 0)}</span>'
            f'<span class="aq-status-pill">候选 {len(scanner_candidates)}</span>'
            f'<span class="aq-status-pill">失败 {len(failed_symbols)}</span></div>',
            unsafe_allow_html=True,
        )
        if failed_symbols:
            with st.expander("查看扫描失败明细"):
                display_table(pd.DataFrame(failed_symbols))
        if unverified_count:
            st.warning(f"有 {unverified_count} 只候选的正常市场价格尚未校准，仅展示策略信号，暂不能生成交易计划。")
        if scanner_candidates.empty:
            st.info("当前股票池没有最新多头信号候选股。")
        else:
            display_table(
                scanner_candidates,
                columns=["symbol", "name", "industry", "selection_score", "score_label", "close", "price_verified", "price_source", "signal_reason", "score_action_hint", "max_drawdown_pct", "sharpe", "note"],
                height=360,
            )
            verified_candidates = scanner_candidates[
                scanner_candidates.get("price_verified", pd.Series(False, index=scanner_candidates.index)).astype(bool)
            ]
            candidate_download_col, candidate_plan_col = st.columns(2)
            candidate_download_col.download_button(
                "下载候选股 CSV",
                scanner_candidates.to_csv(index=False).encode("utf-8-sig"),
                file_name="stock_pool_scan_candidates.csv",
                mime="text/csv",
                width="stretch",
                icon=":material/download:",
            )
            enable_trade_plan = candidate_plan_col.button(
                "生成明日交易计划",
                type="primary",
                disabled=verified_candidates.empty,
                on_click=keep_stock_pool_tab_active,
                width="stretch",
                icon=":material/task_alt:",
            )
            if enable_trade_plan:
                plan_candidates = generate_trade_plan(
                    verified_candidates,
                    total_capital=float(initial_cash),
                    max_position_pct=float(plan_max_position_pct) / 100,
                    total_position_cap_pct=float(plan_total_position_cap_pct) / 100,
                    risk_budget_pct=float(plan_risk_budget_pct) / 100,
                    total_risk_budget_pct=float(plan_total_risk_budget_pct) / 100,
                    stop_loss_pct=float(plan_stop_loss_pct),
                    target_profit_pct=float(plan_target_profit_pct),
                )
                st.session_state["plan_candidates"] = plan_candidates
                st.toast(f"已生成 {len(plan_candidates)} 条交易计划", icon=":material/check_circle:")
            plan_candidates = st.session_state.get("plan_candidates")
            if plan_candidates is not None:
                st.subheader("明日交易计划")
                st.caption("股数按 100 股一手取整，并同时受单票仓位、总仓位和风险预算约束。")
                display_table(
                    plan_candidates,
                    columns=["symbol", "planned_action", "entry_reference_price", "suggested_position_pct", "suggested_capital", "suggested_shares", "stop_loss_price", "target_price", "block_reason", "risk_note"],
                    height=360,
                )
                st.download_button(
                    "下载明日交易计划 CSV",
                    plan_candidates.to_csv(index=False).encode("utf-8-sig"),
                    file_name="next_day_trade_plan.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )


with review_tab:
    st.subheader("复盘日志/交易体系评分")
    form_col, history_col = st.columns([0.9, 1.4])
    with form_col:
        with st.form("review_form", clear_on_submit=False):
            review_date = st.date_input("复盘日期", value=date.today())
            review_symbol = st.text_input("复盘标的", value=symbol)
            review_planned_action = st.text_input("原计划动作", value="按交易计划观察/执行")
            review_actual_action = st.text_input("实际动作", value="未记录")
            review_executed_as_planned = st.checkbox("是否按计划执行", value=True)
            s1, s2, s3 = st.columns(3)
            review_discipline_score = s1.number_input("纪律", 0, 10, 8)
            review_execution_score = s2.number_input("执行", 0, 10, 8)
            review_review_score = s3.number_input("复盘", 0, 10, 8)
            review_summary = st.text_area("复盘总结", placeholder="记录偏差原因、有效动作和下一次调整。")
            save_review_entry = st.form_submit_button("保存复盘记录", type="primary", width="stretch")
        if save_review_entry:
            append_review_entry(
                DEFAULT_REVIEW_JOURNAL_PATH,
                trade_date=str(review_date),
                symbol=normalize_ashare_symbol(review_symbol),
                planned_action=review_planned_action,
                actual_action=review_actual_action,
                executed_as_planned=bool(review_executed_as_planned),
                discipline_score=int(review_discipline_score),
                execution_score=int(review_execution_score),
                review_score=int(review_review_score),
                summary=review_summary,
            )
            st.success("复盘记录已保存。")
    with history_col:
        review_journal = load_review_journal(DEFAULT_REVIEW_JOURNAL_PATH)
        if review_journal.empty:
            st.info("暂无复盘记录。完成第一条记录后，这里会形成纪律与执行统计。")
        else:
            review_summary_stats = summarize_review_journal(review_journal)
            r1, r2, r3 = st.columns(3)
            r1.metric("计划执行率", f"{review_summary_stats['plan_execution_win_rate_pct']:.1f}%")
            r2.metric("平均纪律评分", f"{review_summary_stats['average_discipline_score']:.1f}/10")
            r3.metric("体系平均评分", f"{review_summary_stats['average_system_score']:.1f}/10")
            display_table(review_journal.tail(30), height=420)
            st.download_button("下载复盘日志 CSV", review_journal.to_csv(index=False).encode("utf-8-sig"), file_name="review_journal.csv", mime="text/csv")
