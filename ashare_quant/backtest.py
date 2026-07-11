from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .indicators import add_bollinger_bands, add_macd, add_moving_averages, add_rsi


def _validate_risk_params(stop_loss_pct: float | None, take_profit_pct: float | None) -> None:
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive when provided")
    if take_profit_pct is not None and take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive when provided")


def _entry_exit_signal(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Build a long-only stateful signal from explicit entry and exit events."""
    in_market = False
    values: list[int] = []
    for should_enter, should_exit in zip(entry.fillna(False), exit_.fillna(False)):
        if in_market and bool(should_exit):
            in_market = False
        elif not in_market and bool(should_enter):
            in_market = True
        values.append(int(in_market))
    return pd.Series(values, index=entry.index, dtype=int)


def _apply_position_and_risk_controls(
    result: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Apply previous-day signal positions plus optional close-based risk exits."""
    _validate_risk_params(stop_loss_pct, take_profit_pct)
    if stop_loss_pct is None and take_profit_pct is None:
        result["position"] = result["signal"].shift(1).fillna(0).astype(float)
        result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
        result["action"] = result["position"].diff().fillna(result["position"]).map({1.0: "BUY", -1.0: "SELL"}).fillna("")
        result["risk_exit"] = ""
        result["risk_blocked"] = False
        return result

    desired_positions = result["signal"].shift(1).fillna(0).astype(int).tolist()
    closes = result["close"].astype(float).tolist()
    positions: list[float] = []
    turnovers: list[float] = []
    actions: list[str] = []
    risk_exits: list[str] = []
    risk_blocked: list[bool] = []
    in_market = False
    entry_price: float | None = None
    blocked_until_signal_reset = False

    for desired, close in zip(desired_positions, closes):
        action = ""
        risk_exit = ""
        turnover = 0.0

        if desired == 0:
            blocked_until_signal_reset = False
            if in_market:
                action = "SELL"
                turnover = 1.0
            in_market = False
            entry_price = None
            positions.append(0.0)
            turnovers.append(turnover)
            actions.append(action)
            risk_exits.append(risk_exit)
            risk_blocked.append(False)
            continue

        if not in_market and not blocked_until_signal_reset:
            in_market = True
            entry_price = close
            action = "BUY"
            turnover = 1.0

        position = 1.0 if in_market else 0.0
        if in_market and entry_price is not None:
            pnl_pct = (close / entry_price - 1) * 100
            if stop_loss_pct is not None and pnl_pct <= -stop_loss_pct:
                action = "SELL"
                risk_exit = "STOP_LOSS"
                turnover = max(turnover, 1.0)
                in_market = False
                entry_price = None
                blocked_until_signal_reset = True
            elif take_profit_pct is not None and pnl_pct >= take_profit_pct:
                action = "SELL"
                risk_exit = "TAKE_PROFIT"
                turnover = max(turnover, 1.0)
                in_market = False
                entry_price = None
                blocked_until_signal_reset = True

        positions.append(position)
        turnovers.append(turnover)
        actions.append(action)
        risk_exits.append(risk_exit)
        risk_blocked.append(blocked_until_signal_reset and desired == 1 and not risk_exit)

    result["position"] = positions
    result["turnover"] = turnovers
    result["action"] = actions
    result["risk_exit"] = risk_exits
    result["risk_blocked"] = risk_blocked
    return result


def run_ma_cross_backtest(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Long-only moving-average cross backtest.

    Signal is 1 when short MA is above long MA, otherwise 0. The strategy is
    invested in the stock while the previous day's signal is 1. Fee and
    slippage are charged on position changes.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")

    result = add_moving_averages(df, short_window, long_window)
    result["signal"] = (result["ma_short"] > result["ma_long"]).astype(int)
    result.loc[result["ma_long"].isna(), "signal"] = 0
    result = _apply_position_and_risk_controls(result, stop_loss_pct, take_profit_pct)
    result["daily_return"] = result["close"].pct_change().fillna(0)
    result["benchmark_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    cost_rate = fee_rate + slippage_rate
    gross_return = result["position"] * result["daily_return"]
    result["strategy_return"] = gross_return - result["turnover"] * cost_rate
    result["equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["trade_cost"] = result["equity"].shift(1).fillna(initial_cash) * result["turnover"] * cost_rate
    return result


def run_rsi_backtest(
    df: pd.DataFrame,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Long-only RSI mean-reversion backtest.

    Signal is 1 while RSI is below the oversold threshold and 0 once RSI is
    at or above the overbought threshold. The strategy is invested using the
    previous day's signal to avoid look-ahead bias.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")
    if not 0 <= oversold < overbought <= 100:
        raise ValueError("RSI thresholds must satisfy 0 <= oversold < overbought <= 100")

    result = add_rsi(df, window)
    result["signal"] = _entry_exit_signal(result["rsi"] < oversold, result["rsi"] >= overbought)
    result = _apply_position_and_risk_controls(result, stop_loss_pct, take_profit_pct)
    result["daily_return"] = result["close"].pct_change().fillna(0)
    result["benchmark_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    cost_rate = fee_rate + slippage_rate
    gross_return = result["position"] * result["daily_return"]
    result["strategy_return"] = gross_return - result["turnover"] * cost_rate
    result["equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["trade_cost"] = result["equity"].shift(1).fillna(initial_cash) * result["turnover"] * cost_rate
    return result


def run_macd_backtest(
    df: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Long-only MACD trend-following backtest.

    Signal is 1 when DIF is above DEA, otherwise 0. The strategy is invested
    using the previous day's signal to avoid look-ahead bias.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")

    result = add_macd(df, fast_window, slow_window, signal_window)
    result["signal"] = (result["macd_dif"] > result["macd_dea"]).astype(int)
    result = _apply_position_and_risk_controls(result, stop_loss_pct, take_profit_pct)
    result["daily_return"] = result["close"].pct_change().fillna(0)
    result["benchmark_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    cost_rate = fee_rate + slippage_rate
    gross_return = result["position"] * result["daily_return"]
    result["strategy_return"] = gross_return - result["turnover"] * cost_rate
    result["equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["trade_cost"] = result["equity"].shift(1).fillna(initial_cash) * result["turnover"] * cost_rate
    return result


def run_bollinger_backtest(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Long-only Bollinger Band mean-reversion backtest.

    Enter when close breaks below the lower band and stay invested until close
    recovers to the middle band. The previous day's signal controls exposure to
    avoid look-ahead bias.
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("fee_rate and slippage_rate must be non-negative")

    result = add_bollinger_bands(df, window, num_std)
    result["signal"] = _entry_exit_signal(
        result["close"] < result["bb_lower"],
        result["close"] >= result["bb_middle"],
    )
    result = _apply_position_and_risk_controls(result, stop_loss_pct, take_profit_pct)
    result["daily_return"] = result["close"].pct_change().fillna(0)
    result["benchmark_equity"] = initial_cash * (1 + result["daily_return"]).cumprod()

    cost_rate = fee_rate + slippage_rate
    gross_return = result["position"] * result["daily_return"]
    result["strategy_return"] = gross_return - result["turnover"] * cost_rate
    result["equity"] = initial_cash * (1 + result["strategy_return"]).cumprod()
    result["trade_cost"] = result["equity"].shift(1).fillna(initial_cash) * result["turnover"] * cost_rate
    return result


def optimize_ma_parameters(
    df: pd.DataFrame,
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> pd.DataFrame:
    """Rank valid moving-average parameter pairs by backtest return."""
    rows = []
    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue
            result = run_ma_cross_backtest(
                df,
                short_window=int(short_window),
                long_window=int(long_window),
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
            summary = summarize_backtest(result, initial_cash)
            rows.append(
                {
                    "short_window": int(short_window),
                    "long_window": int(long_window),
                    "final_equity": summary["final_equity"],
                    "total_return_pct": summary["total_return_pct"],
                    "benchmark_return_pct": summary["benchmark_return_pct"],
                    "excess_return_pct": summary["excess_return_pct"],
                    "max_drawdown_pct": summary["max_drawdown_pct"],
                    "sharpe": summary["sharpe"],
                    "trade_count": summary["trade_count"],
                }
            )

    columns = [
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
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["total_return_pct", "sharpe", "final_equity"], ascending=False, ignore_index=True
    )


def compare_strategy_backtests(
    df: pd.DataFrame,
    strategies: Mapping[str, tuple[Callable[..., pd.DataFrame], Mapping[str, object]]],
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Run multiple strategy configurations and return a ranked metrics table."""
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if not strategies:
        raise ValueError("strategies must not be empty")

    rows = []
    for strategy_name, (runner, kwargs) in strategies.items():
        result = runner(df, initial_cash=initial_cash, **dict(kwargs))
        rows.append({"strategy": strategy_name, **summarize_backtest(result, initial_cash)})

    comparison = pd.DataFrame(rows).sort_values(
        ["total_return_pct", "sharpe", "final_equity"], ascending=False, ignore_index=True
    )
    comparison.insert(0, "rank", range(1, len(comparison) + 1))
    return comparison



def run_equal_weight_portfolio_backtest(
    data_by_symbol: Mapping[str, pd.DataFrame],
    strategy_runner: Callable[..., pd.DataFrame],
    initial_cash: float = 100_000.0,
    **strategy_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one strategy across symbols and combine returns as equal-weight portfolio."""
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if not data_by_symbol:
        raise ValueError("data_by_symbol must not be empty")

    member_rows = []
    strategy_returns = []
    benchmark_returns = []
    for symbol, df in data_by_symbol.items():
        if df.empty:
            raise ValueError(f"data for {symbol} is empty")
        member_result = strategy_runner(df, initial_cash=initial_cash, **strategy_kwargs)
        member_rows.append(
            {
                "symbol": symbol,
                **summarize_backtest(member_result, initial_cash),
                "result": member_result,
            }
        )
        indexed = member_result.set_index("date")
        strategy_returns.append(indexed["strategy_return"].rename(symbol))
        benchmark_returns.append(indexed["daily_return"].rename(symbol))

    portfolio_returns = pd.concat(strategy_returns, axis=1).sort_index().fillna(0).mean(axis=1)
    benchmark_portfolio_returns = pd.concat(benchmark_returns, axis=1).sort_index().fillna(0).mean(axis=1)
    portfolio = pd.DataFrame(
        {
            "date": portfolio_returns.index,
            "strategy_return": portfolio_returns.values,
            "benchmark_return": benchmark_portfolio_returns.reindex(portfolio_returns.index).fillna(0).values,
        }
    )
    portfolio["equity"] = initial_cash * (1 + portfolio["strategy_return"]).cumprod()
    portfolio["benchmark_equity"] = initial_cash * (1 + portfolio["benchmark_return"]).cumprod()
    portfolio["position"] = 1.0
    portfolio["action"] = ""
    if not portfolio.empty:
        portfolio.loc[portfolio.index[0], "action"] = "BUY"
    members = pd.DataFrame(member_rows)
    return portfolio, members


def _period_return_table(result: pd.DataFrame, freq: str, period_format: str) -> pd.DataFrame:
    """Group an equity curve into period return rows."""
    if result.empty:
        return pd.DataFrame(columns=["period", "start_date", "end_date", "start_equity", "end_equity", "return_pct"])
    if "date" not in result.columns or "equity" not in result.columns:
        raise KeyError("result must contain 'date' and 'equity' columns")

    ordered = result[["date", "equity"]].copy().sort_values("date").reset_index(drop=True)
    ordered["date"] = pd.to_datetime(ordered["date"])
    previous_equity = ordered["equity"].astype(float).shift(1)
    previous_equity.iloc[0] = float(ordered["equity"].iloc[0])
    ordered["period_start_equity"] = previous_equity
    ordered["period"] = ordered["date"].dt.to_period(freq)

    rows = []
    for period, group in ordered.groupby("period", sort=True):
        start_equity = float(group["period_start_equity"].iloc[0])
        end_equity = float(group["equity"].iloc[-1])
        rows.append(
            {
                "period": period.strftime(period_format),
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "start_equity": round(start_equity, 2),
                "end_equity": round(end_equity, 2),
                "return_pct": round((end_equity / start_equity - 1) * 100, 2) if start_equity else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["period", "start_date", "end_date", "start_equity", "end_equity", "return_pct"])


def monthly_return_table(result: pd.DataFrame) -> pd.DataFrame:
    """Return monthly strategy returns from a backtest equity curve."""
    return _period_return_table(result, "M", "%Y-%m")


def yearly_return_table(result: pd.DataFrame) -> pd.DataFrame:
    """Return yearly strategy returns from a backtest equity curve."""
    return _period_return_table(result, "Y", "%Y")


def analyze_drawdown_periods(result: pd.DataFrame) -> pd.DataFrame:
    """Return ranked drawdown windows with peak, trough and recovery dates."""
    columns = ["peak_date", "trough_date", "recovery_date", "max_drawdown_pct", "duration_days", "recovered"]
    if result.empty:
        return pd.DataFrame(columns=columns)
    if "date" not in result.columns or "equity" not in result.columns:
        raise KeyError("result must contain 'date' and 'equity' columns")

    ordered = result[["date", "equity"]].copy().sort_values("date").reset_index(drop=True)
    ordered["date"] = pd.to_datetime(ordered["date"])
    equity = ordered["equity"].astype(float).tolist()
    dates = ordered["date"].tolist()

    rows = []
    peak_index = 0
    index = 1
    while index < len(equity):
        if equity[index] >= equity[peak_index]:
            peak_index = index
            index += 1
            continue

        trough_index = index
        recovery_index = None
        scan = index
        while scan < len(equity) and equity[scan] < equity[peak_index]:
            if equity[scan] < equity[trough_index]:
                trough_index = scan
            scan += 1
        if scan < len(equity):
            recovery_index = scan

        end_index = recovery_index if recovery_index is not None else len(equity) - 1
        rows.append(
            {
                "peak_date": dates[peak_index],
                "trough_date": dates[trough_index],
                "recovery_date": dates[recovery_index] if recovery_index is not None else pd.NaT,
                "max_drawdown_pct": round((equity[trough_index] / equity[peak_index] - 1) * 100, 2),
                "duration_days": int(end_index - peak_index),
                "recovered": recovery_index is not None,
            }
        )
        peak_index = recovery_index if recovery_index is not None else len(equity) - 1
        index = peak_index + 1

    drawdown_df = pd.DataFrame(rows, columns=columns).sort_values("max_drawdown_pct", ignore_index=True)
    if not drawdown_df.empty:
        drawdown_df["recovered"] = drawdown_df["recovered"].astype(object)
    return drawdown_df


def extract_trades(result: pd.DataFrame) -> pd.DataFrame:
    """Extract BUY/SELL rows from a backtest result."""
    if "action" not in result.columns:
        position_change = result["position"].diff().fillna(result["position"])
        actions = position_change.map({1.0: "BUY", -1.0: "SELL"}).fillna("")
    else:
        actions = result["action"]

    trades = result.loc[actions.isin(["BUY", "SELL"])].copy()
    if trades.empty:
        return pd.DataFrame(columns=["date", "action", "price", "equity", "trade_cost"])
    trades["action"] = actions.loc[trades.index]
    trades["price"] = trades["close"]
    cols = [c for c in ["date", "action", "price", "equity", "trade_cost", "signal", "position"] if c in trades.columns]
    return trades[cols].reset_index(drop=True)


def summarize_backtest(result: pd.DataFrame, initial_cash: float) -> dict:
    """Calculate core metrics for a backtest result dataframe."""
    if result.empty:
        raise ValueError("result dataframe is empty")
    equity = result["equity"].astype(float)
    final_equity = float(equity.iloc[-1])
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    total_return_pct = (final_equity / initial_cash - 1) * 100

    if "benchmark_equity" in result.columns:
        benchmark_final = float(result["benchmark_equity"].iloc[-1])
    elif "close" in result.columns:
        first_close = float(result["close"].iloc[0])
        last_close = float(result["close"].iloc[-1])
        benchmark_final = initial_cash * last_close / first_close
    else:
        benchmark_final = initial_cash
    benchmark_return_pct = (benchmark_final / initial_cash - 1) * 100

    daily = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    annual_return_pct = ((final_equity / initial_cash) ** (252 / max(len(equity), 1)) - 1) * 100
    sharpe = 0.0
    if len(daily) > 1 and daily.std() != 0:
        sharpe = float((daily.mean() / daily.std()) * np.sqrt(252))

    return {
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "annual_return_pct": round(float(annual_return_pct), 2),
        "benchmark_return_pct": round(float(benchmark_return_pct), 2),
        "excess_return_pct": round(float(total_return_pct - benchmark_return_pct), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "sharpe": round(sharpe, 2),
        "trading_days": int(len(result)),
        "in_market_days": int(result.get("position", pd.Series(dtype=int)).sum()),
        "trade_count": int((result.get("action", pd.Series(dtype=str)).isin(["BUY", "SELL"])).sum()),
        "total_trade_cost": round(float(result.get("trade_cost", pd.Series(dtype=float)).sum()), 2),
    }
