from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from .backtest import summarize_backtest


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_priority_text(selection_score: float) -> tuple[str, str]:
    """Return a concise score label and next-step hint for scanner results."""
    if selection_score >= 90:
        return "强势候选", "优先纳入明日交易计划，等待回踩或突破确认。"
    if selection_score >= 75:
        return "稳健候选", "保留观察，需结合行业强弱与仓位预算筛选。"
    if selection_score >= 60:
        return "观察候选", "仅作为备选，等待趋势、量能或风险评分改善。"
    return "低优先级", "暂不优先执行，除非出现新的催化或更高胜率信号。"


def _score_candidate(price_history: pd.DataFrame, backtest_result: pd.DataFrame) -> dict[str, object]:
    """Score one scan candidate across trend, momentum, volume and risk dimensions."""
    close = price_history["close"].astype(float)
    latest_close = float(close.iloc[-1])
    ma_window = min(20, len(close))
    recent_window = min(5, len(close) - 1)

    ma = float(close.tail(ma_window).mean()) if ma_window else latest_close
    recent_return_pct = float((latest_close / close.iloc[-1 - recent_window] - 1) * 100) if recent_window else 0.0
    trend_premium_pct = float((latest_close / ma - 1) * 100) if ma else 0.0
    trend_score = _bounded(15 + trend_premium_pct * 2 + recent_return_pct, 0, 30)
    momentum_score = _bounded(10 + recent_return_pct * 2, 0, 25)

    if "volume" in price_history.columns and len(price_history) > 1:
        volume = price_history["volume"].astype(float)
        volume_avg = float(volume.iloc[:-1].tail(min(20, len(volume) - 1)).mean())
        volume_ratio = float(volume.iloc[-1] / volume_avg) if volume_avg > 0 else 1.0
    else:
        volume_ratio = 1.0
    volume_score = _bounded(8 + (volume_ratio - 1) * 8, 0, 20)

    daily_returns = close.pct_change().dropna()
    volatility_pct = float(daily_returns.tail(20).std() * 100) if not daily_returns.empty else 0.0
    risk_score = _bounded(25 - abs(float(backtest_result["equity"].div(backtest_result["equity"].cummax()).sub(1).min()) * 100) - volatility_pct, 0, 25)
    selection_score = trend_score + momentum_score + volume_score + risk_score
    score_label, score_action_hint = _score_priority_text(selection_score)

    return {
        "selection_score": round(selection_score, 2),
        "score_label": score_label,
        "score_action_hint": score_action_hint,
        "trend_score": round(trend_score, 2),
        "momentum_score": round(momentum_score, 2),
        "volume_score": round(volume_score, 2),
        "risk_score": round(risk_score, 2),
        "score_explanation": (
            f"趋势{trend_score:.1f}/30：收盘较均线{trend_premium_pct:+.1f}%；"
            f"动量{momentum_score:.1f}/25：近{recent_window}日{recent_return_pct:+.1f}%；"
            f"量能{volume_score:.1f}/20：最新量/均量{volume_ratio:.2f}；"
            f"风险{risk_score:.1f}/25：波动{volatility_pct:.1f}%"
        ),
    }


def _format_signal_reason(latest: pd.Series) -> str:
    if int(latest.get("signal", 0)) == 1:
        action = str(latest.get("action", ""))
        if action == "BUY":
            return "最新交易日信号为多头，且刚出现买入动作。"
        return "最新交易日信号为多头，可纳入明日交易计划观察。"
    return "最新交易日无多头信号。"


def apply_latest_quotes_to_candidates(
    candidates: pd.DataFrame,
    quotes_by_symbol: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Replace scan-only adjusted closes with verified normal market prices."""
    result = candidates.copy()
    original_attrs = dict(candidates.attrs)
    if "price_verified" not in result.columns:
        result["price_verified"] = False
    if "price_source" not in result.columns:
        result["price_source"] = ""

    verified_count = 0
    for index, row in result.iterrows():
        if bool(row.get("price_verified", False)):
            verified_count += 1
            continue
        quote = quotes_by_symbol.get(str(row["symbol"]), {})
        try:
            price = float(quote.get("price", 0))
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            continue
        result.at[index, "close"] = price
        result.at[index, "price_verified"] = True
        result.at[index, "price_source"] = str(quote.get("source_name", "腾讯批量行情（不复权）"))
        verified_count += 1

    result.attrs.update(original_attrs)
    result.attrs["quote_summary"] = {
        "candidate_count": int(len(result)),
        "verified_count": int(verified_count),
        "unverified_count": int(len(result) - verified_count),
    }
    return result


def scan_stock_pool(
    data_by_symbol: Mapping[str, pd.DataFrame],
    strategy_runner: Callable[..., pd.DataFrame],
    initial_cash: float = 100_000.0,
    **strategy_kwargs,
) -> pd.DataFrame:
    """Run a strategy across a stock pool and return latest long-signal candidates.

    The scanner is intended for post-market review: each symbol is backtested
    using only the supplied historical data, and symbols whose latest strategy
    signal is long are returned with a human-readable reason and key metrics.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if not data_by_symbol:
        raise ValueError("data_by_symbol must not be empty")

    rows: list[dict[str, object]] = []
    failed_symbols: list[dict[str, str]] = []
    scanned_symbols = 0
    for symbol, df in data_by_symbol.items():
        if df.empty:
            continue
        try:
            result = strategy_runner(df, initial_cash=initial_cash, **strategy_kwargs)
        except Exception as exc:
            failed_symbols.append({"symbol": symbol, "stage": "strategy", "error": str(exc)})
            continue
        scanned_symbols += 1
        if result.empty:
            continue
        latest = result.iloc[-1]
        latest_signal = int(latest.get("signal", 0))
        if latest_signal != 1:
            continue
        metrics = summarize_backtest(result, initial_cash)
        score = _score_candidate(df, result)
        rows.append(
            {
                "symbol": symbol,
                "candidate_action": "关注/可按计划买入",
                "signal_reason": _format_signal_reason(latest),
                **score,
                "date": latest.get("date"),
                "close": float(latest["close"]),
                "price_verified": bool(df.attrs.get("price_verified", True)),
                "price_source": str(df.attrs.get("source_name", "")),
                "latest_signal": latest_signal,
                "latest_position": float(latest.get("position", 0.0)),
                "strategy_return_pct": metrics["total_return_pct"],
                "benchmark_return_pct": metrics["benchmark_return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "sharpe": metrics["sharpe"],
                "trade_count": metrics["trade_count"],
            }
        )

    columns = [
        "symbol",
        "candidate_action",
        "signal_reason",
        "selection_score",
        "score_label",
        "score_action_hint",
        "trend_score",
        "momentum_score",
        "volume_score",
        "risk_score",
        "score_explanation",
        "date",
        "close",
        "price_verified",
        "price_source",
        "latest_signal",
        "latest_position",
        "strategy_return_pct",
        "benchmark_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "trade_count",
    ]
    if not rows:
        result_df = pd.DataFrame(columns=columns)
    else:
        result_df = pd.DataFrame(rows, columns=columns).sort_values(
            ["selection_score", "strategy_return_pct", "sharpe", "symbol"], ascending=[False, False, False, True], ignore_index=True
        )
    result_df.attrs["failed_symbols"] = failed_symbols
    result_df.attrs["scan_summary"] = {
        "total_symbols": int(len(data_by_symbol)),
        "scanned_symbols": int(scanned_symbols),
        "candidate_count": int(len(result_df)),
        "failed_count": int(len(failed_symbols)),
    }
    return result_df
