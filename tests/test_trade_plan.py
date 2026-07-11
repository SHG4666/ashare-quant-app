import pandas as pd

from ashare_quant.trade_plan import generate_trade_plan


def test_generate_trade_plan_turns_scanner_candidates_into_risk_defined_orders():
    candidates = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "candidate_action": "关注/可按计划买入",
                "signal_reason": "最新交易日信号为多头，且刚出现买入动作。",
                "close": 10.0,
                "strategy_return_pct": 12.5,
                "max_drawdown_pct": -6.0,
                "sharpe": 1.2,
            },
            {
                "symbol": "600519",
                "candidate_action": "关注/可按计划买入",
                "signal_reason": "最新交易日信号为多头，可纳入明日交易计划观察。",
                "close": 100.0,
                "strategy_return_pct": 5.0,
                "max_drawdown_pct": -15.0,
                "sharpe": 0.5,
            },
        ]
    )

    plan = generate_trade_plan(
        candidates,
        total_capital=100_000,
        max_position_pct=0.2,
        stop_loss_pct=8,
        target_profit_pct=16,
    )

    assert plan["symbol"].tolist() == ["000001", "600519"]
    first = plan.iloc[0]
    assert first["planned_action"] == "明日按计划观察/买入"
    assert first["entry_reference_price"] == 10.0
    assert first["suggested_position_pct"] == 0.2
    assert first["suggested_capital"] == 20_000.0
    assert first["stop_loss_price"] == 9.2
    assert first["target_price"] == 11.6
    assert "多头" in first["plan_reason"]
    assert "单票仓位不超过20%" in first["risk_note"]


def test_generate_trade_plan_sizes_board_lot_shares_from_risk_budget_and_stop_distance():
    candidates = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "signal_reason": "风险预算仓位测试。",
                "close": 10.0,
            },
            {
                "symbol": "600519",
                "signal_reason": "高价股仓位测试。",
                "close": 95.0,
            },
        ]
    )

    plan = generate_trade_plan(
        candidates,
        total_capital=100_000,
        max_position_pct=0.2,
        stop_loss_pct=5,
        target_profit_pct=10,
        risk_budget_pct=0.01,
    )

    assert plan["risk_budget"].tolist() == [1_000.0, 1_000.0]
    assert plan["suggested_shares"].tolist() == [2000, 200]
    assert plan["suggested_capital"].tolist() == [20_000.0, 19_000.0]
    assert plan["suggested_position_pct"].tolist() == [0.2, 0.19]
    assert plan["capital_at_risk"].tolist() == [1_000.0, 950.0]
    assert "单笔风险预算1.00%" in plan.iloc[0]["risk_note"]


def test_generate_trade_plan_caps_total_position_and_reports_remaining_risk_budget():
    candidates = pd.DataFrame(
        [
            {"symbol": "000001", "signal_reason": "第一候选。", "close": 10.0},
            {"symbol": "000002", "signal_reason": "第二候选。", "close": 20.0},
            {"symbol": "000003", "signal_reason": "第三候选。", "close": 30.0},
        ]
    )

    plan = generate_trade_plan(
        candidates,
        total_capital=100_000,
        max_position_pct=0.2,
        total_position_cap_pct=0.3,
        stop_loss_pct=10,
        target_profit_pct=20,
        risk_budget_pct=0.02,
        total_risk_budget_pct=0.03,
    )

    assert plan["suggested_capital"].tolist() == [20_000.0, 10_000.0, 0.0]
    assert plan["suggested_shares"].tolist() == [2000, 500, 0]
    assert plan["capital_at_risk"].tolist() == [2_000.0, 1_000.0, 0.0]
    assert plan["cumulative_position_pct"].tolist() == [0.2, 0.3, 0.3]
    assert plan["remaining_position_capital"].tolist() == [10_000.0, 0.0, 0.0]
    assert plan["remaining_risk_budget"].tolist() == [1_000.0, 0.0, 0.0]
    assert "总仓位上限30%" in plan.iloc[1]["risk_note"]
    assert "组合剩余风险预算0.00" in plan.iloc[1]["risk_note"]


def test_generate_trade_plan_marks_candidates_blocked_when_position_or_risk_budget_exhausted():
    candidates = pd.DataFrame(
        [
            {"symbol": "000001", "signal_reason": "第一候选。", "close": 10.0},
            {"symbol": "000002", "signal_reason": "预算耗尽后的候选。", "close": 20.0},
        ]
    )

    plan = generate_trade_plan(
        candidates,
        total_capital=100_000,
        max_position_pct=0.2,
        total_position_cap_pct=0.2,
        stop_loss_pct=10,
        target_profit_pct=20,
        risk_budget_pct=0.02,
        total_risk_budget_pct=0.02,
    )

    blocked = plan.iloc[1]
    assert blocked["suggested_shares"] == 0
    assert blocked["planned_action"] == "暂不买入/等待预算释放"
    assert "总仓位或组合风险预算已不足" in blocked["risk_note"]
    assert "当前候选不新增风险敞口" in blocked["plan_reason"]


def test_generate_trade_plan_reports_budget_utilization_and_block_reason():
    candidates = pd.DataFrame(
        [
            {"symbol": "000001", "signal_reason": "第一候选。", "close": 10.0},
            {"symbol": "000002", "signal_reason": "预算耗尽后的候选。", "close": 20.0},
        ]
    )

    plan = generate_trade_plan(
        candidates,
        total_capital=100_000,
        max_position_pct=0.2,
        total_position_cap_pct=0.2,
        stop_loss_pct=10,
        target_profit_pct=20,
        risk_budget_pct=0.02,
        total_risk_budget_pct=0.02,
    )

    assert plan["position_cap_used_pct"].tolist() == [1.0, 1.0]
    assert plan["risk_budget_used_pct"].tolist() == [1.0, 1.0]
    assert plan["block_reason"].tolist() == ["", "总仓位上限已用尽；组合风险预算已用尽"]
