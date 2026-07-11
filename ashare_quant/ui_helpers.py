from __future__ import annotations

import re

import pandas as pd


_SYMBOL_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


_TOKEN_SEPARATORS = (",", "，")
_STRATEGY_INDICATOR_COLUMNS = {
    "双均线交叉": ["ma_short", "ma_long"],
    "RSI超卖反弹": ["rsi"],
    "MACD金叉": ["macd_dif", "macd_dea", "macd_hist"],
    "布林带均值回归": ["bb_middle", "bb_upper", "bb_lower"],
}
_STRATEGY_REPORT_SLUGS = {
    "双均线交叉": "ma",
    "RSI超卖反弹": "rsi",
    "MACD金叉": "macd",
    "布林带均值回归": "bollinger",
}
_STRATEGY_GUIDES = {
    "双均线交叉": {
        "style": "趋势跟随",
        "summary": "短均线上穿并保持在长均线上方时持有，适合方向明确的趋势行情。",
        "entry": "短均线高于长均线",
        "exit": "短均线回落至长均线下方",
        "risk": "震荡市容易频繁反复，需关注交易成本和回撤。",
    },
    "RSI超卖反弹": {
        "style": "均值回归",
        "summary": "RSI 进入超卖区后建仓，回到超买区退出，适合有支撑的区间行情。",
        "entry": "RSI 低于超卖阈值",
        "exit": "RSI 达到超买阈值",
        "risk": "单边下跌中可能持续超卖，止损比参数优化更重要。",
    },
    "MACD金叉": {
        "style": "趋势跟随",
        "summary": "DIF 高于 DEA 时持有，用中期动量确认方向。",
        "entry": "DIF 上穿并高于 DEA",
        "exit": "DIF 回落至 DEA 下方",
        "risk": "信号有滞后，快速反转时可能回吐利润。",
    },
    "布林带均值回归": {
        "style": "均值回归",
        "summary": "价格跌破下轨后建仓，回到中轨退出，捕捉偏离后的修复。",
        "entry": "收盘价低于布林下轨",
        "exit": "收盘价回到布林中轨",
        "risk": "趋势性下跌会让下轨持续下移，必须限制单笔风险。",
    },
}
_RECENT_SIGNAL_BASE_COLUMNS = [
    "date",
    "close",
    "signal",
    "position",
    "action",
    "risk_exit",
    "risk_blocked",
    "equity",
    "benchmark_equity",
]
_KLINE_PERIOD_ALIASES = {
    "日K": None,
    "周K": "W-FRI",
    "月K": "M",
    "年K": "Y",
}


def aggregate_price_bars(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Aggregate daily OHLCV rows into the selected display period."""
    if period not in _KLINE_PERIOD_ALIASES:
        raise ValueError(f"unsupported K-line period: {period}")

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for column in required[1:]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if period == "日K" or work.empty:
        return work

    work["_period"] = work["date"].dt.to_period(_KLINE_PERIOD_ALIASES[period])
    return (
        work.groupby("_period", sort=True, observed=True)
        .agg(
            date=("date", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )


def parse_portfolio_symbols(text: str) -> list[str]:
    """Parse portfolio symbols from pasted text while preserving first-seen order.

    Accepts newline/comma separated values and tolerates labels such as
    ``000001 平安银行`` by using the first whitespace-delimited token.
    """
    normalized = text
    for separator in _TOKEN_SEPARATORS:
        normalized = normalized.replace(separator, "\n")

    symbols: list[str] = []
    seen: set[str] = set()
    for raw_item in normalized.splitlines():
        item = raw_item.strip()
        if not item:
            continue
        symbol = item.split()[0].strip()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def normalize_ashare_symbol(value: str) -> str:
    """Extract and validate a six-digit mainland A-share symbol."""
    match = _SYMBOL_RE.search(str(value).strip())
    if not match:
        raise ValueError("请输入 6 位 A 股代码，例如 000001 或 600519。")
    return match.group(1)


def strategy_guide(strategy_name: str) -> dict[str, str]:
    """Return consistent human-facing entry, exit and risk language."""
    if strategy_name not in _STRATEGY_GUIDES:
        raise ValueError(f"unsupported strategy: {strategy_name}")
    return dict(_STRATEGY_GUIDES[strategy_name])


def indicator_display_columns(strategy_name: str) -> list[str]:
    """Return the recent-signal table columns for a UI strategy name."""
    if strategy_name not in _STRATEGY_INDICATOR_COLUMNS:
        raise ValueError(f"unsupported strategy: {strategy_name}")
    return [
        *_RECENT_SIGNAL_BASE_COLUMNS[:2],
        *_STRATEGY_INDICATOR_COLUMNS[strategy_name],
        *_RECENT_SIGNAL_BASE_COLUMNS[2:],
    ]


def strategy_report_slug(strategy_name: str) -> str:
    """Return the stable download filename slug for a UI strategy name."""
    if strategy_name not in _STRATEGY_REPORT_SLUGS:
        raise ValueError(f"unsupported strategy: {strategy_name}")
    return _STRATEGY_REPORT_SLUGS[strategy_name]
