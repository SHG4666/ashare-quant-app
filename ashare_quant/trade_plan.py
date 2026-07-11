from __future__ import annotations

import math

import pandas as pd


def generate_trade_plan(
    candidates: pd.DataFrame,
    total_capital: float,
    max_position_pct: float = 0.2,
    stop_loss_pct: float = 8.0,
    target_profit_pct: float = 16.0,
    risk_budget_pct: float | None = None,
    total_position_cap_pct: float | None = None,
    total_risk_budget_pct: float | None = None,
) -> pd.DataFrame:
    """Convert scanner candidates into a next-trading-day risk-defined plan."""
    if total_capital <= 0:
        raise ValueError("total_capital must be positive")
    if not 0 < max_position_pct <= 1:
        raise ValueError("max_position_pct must be between 0 and 1")
    if total_position_cap_pct is not None and not 0 < total_position_cap_pct <= 1:
        raise ValueError("total_position_cap_pct must be between 0 and 1")
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive")
    if target_profit_pct <= 0:
        raise ValueError("target_profit_pct must be positive")
    if risk_budget_pct is not None and not 0 < risk_budget_pct <= 1:
        raise ValueError("risk_budget_pct must be between 0 and 1")
    if total_risk_budget_pct is not None and not 0 < total_risk_budget_pct <= 1:
        raise ValueError("total_risk_budget_pct must be between 0 and 1")

    columns = [
        "symbol",
        "planned_action",
        "entry_reference_price",
        "suggested_position_pct",
        "suggested_capital",
        "suggested_shares",
        "risk_budget",
        "capital_at_risk",
        "cumulative_position_pct",
        "position_cap_used_pct",
        "remaining_position_capital",
        "risk_budget_used_pct",
        "remaining_risk_budget",
        "block_reason",
        "stop_loss_price",
        "target_price",
        "plan_reason",
        "risk_note",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    total_capital = float(total_capital)
    total_position_capital = total_capital * (
        float(total_position_cap_pct) if total_position_cap_pct is not None else 1.0
    )
    total_risk_budget = (
        total_capital * float(total_risk_budget_pct) if total_risk_budget_pct is not None else None
    )
    used_position_capital = 0.0
    used_risk_budget = 0.0

    rows: list[dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        close = float(candidate["close"])
        remaining_position_capital_before = max(0.0, total_position_capital - used_position_capital)
        remaining_risk_budget_before = (
            max(0.0, total_risk_budget - used_risk_budget) if total_risk_budget is not None else None
        )
        max_capital = min(total_capital * float(max_position_pct), remaining_position_capital_before)
        stop_loss_price = close * (1 - float(stop_loss_pct) / 100)
        target_price = close * (1 + float(target_profit_pct) / 100)
        risk_budget = total_capital * float(risk_budget_pct) if risk_budget_pct is not None else None
        if risk_budget is not None and remaining_risk_budget_before is not None:
            risk_budget = min(risk_budget, remaining_risk_budget_before)
        if risk_budget is None:
            suggested_capital = max_capital
            suggested_shares = 0
            capital_at_risk = 0.0
        else:
            per_share_risk = close - stop_loss_price
            risk_limited_shares = math.floor(risk_budget / per_share_risk / 100) * 100 if per_share_risk > 0 else 0
            capital_limited_shares = math.floor(max_capital / close / 100) * 100
            suggested_shares = max(0, min(risk_limited_shares, capital_limited_shares))
            suggested_capital = suggested_shares * close
            capital_at_risk = suggested_shares * per_share_risk
        used_position_capital += suggested_capital
        used_risk_budget += capital_at_risk
        position_pct = suggested_capital / total_capital
        cumulative_position_pct = used_position_capital / total_capital
        remaining_position_capital = max(0.0, total_position_capital - used_position_capital)
        remaining_risk_budget = max(0.0, total_risk_budget - used_risk_budget) if total_risk_budget is not None else None
        position_cap_used_pct = used_position_capital / total_position_capital if total_position_capital else 0.0
        risk_budget_used_pct = used_risk_budget / total_risk_budget if total_risk_budget else None

        is_blocked = suggested_shares == 0 and (remaining_position_capital_before <= 0 or remaining_risk_budget_before == 0)
        block_reasons = []
        if is_blocked and remaining_position_capital_before <= 0:
            block_reasons.append("总仓位上限已用尽")
        if is_blocked and remaining_risk_budget_before == 0:
            block_reasons.append("组合风险预算已用尽")
        block_reason = "；".join(block_reasons)
        planned_action = "暂不买入/等待预算释放" if is_blocked else "明日按计划观察/买入"
        plan_reason = str(candidate.get("signal_reason", "盘后扫描入选候选股。"))
        if is_blocked:
            plan_reason = f"{plan_reason} 当前候选不新增风险敞口，等待总仓位或风险预算释放后再评估。"

        risk_note_parts = []
        if risk_budget_pct is not None:
            risk_note_parts.append(f"单笔风险预算{risk_budget_pct:.2%}")
        if total_position_cap_pct is not None:
            risk_note_parts.append(f"总仓位上限{total_position_cap_pct:.0%}")
        risk_note_parts.append(f"单票仓位不超过{max_position_pct:.0%}")
        if remaining_risk_budget is not None:
            risk_note_parts.append(f"组合剩余风险预算{remaining_risk_budget:.2f}")
        if is_blocked:
            risk_note_parts.append("总仓位或组合风险预算已不足")
        risk_note = "；".join(risk_note_parts) + "；跌破止损价先执行风控，未触发条件则不追高。"
        rows.append(
            {
                "symbol": candidate["symbol"],
                "planned_action": planned_action,
                "entry_reference_price": round(close, 4),
                "suggested_position_pct": round(position_pct, 4),
                "suggested_capital": round(suggested_capital, 2),
                "suggested_shares": int(suggested_shares),
                "risk_budget": round(risk_budget, 2) if risk_budget is not None else None,
                "capital_at_risk": round(capital_at_risk, 2),
                "cumulative_position_pct": round(cumulative_position_pct, 4),
                "position_cap_used_pct": round(position_cap_used_pct, 4),
                "remaining_position_capital": round(remaining_position_capital, 2),
                "risk_budget_used_pct": round(risk_budget_used_pct, 4) if risk_budget_used_pct is not None else None,
                "remaining_risk_budget": round(remaining_risk_budget, 2) if remaining_risk_budget is not None else None,
                "block_reason": block_reason,
                "stop_loss_price": round(stop_loss_price, 4),
                "target_price": round(target_price, 4),
                "plan_reason": plan_reason,
                "risk_note": risk_note,
            }
        )
    return pd.DataFrame(rows, columns=columns)
