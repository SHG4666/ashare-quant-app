from pathlib import Path

import pandas as pd

from ashare_quant.review_journal import append_review_entry, load_review_journal, summarize_review_journal


def test_append_review_entry_persists_plan_execution_scores_and_summary(tmp_path: Path):
    path = tmp_path / "review_journal.csv"

    appended = append_review_entry(
        path,
        trade_date="2026-07-06",
        symbol="000001",
        planned_action="明日按计划观察/买入",
        actual_action="未买入",
        executed_as_planned=False,
        discipline_score=7,
        execution_score=8,
        review_score=9,
        summary="按计划等待，没有追高。",
    )
    loaded = load_review_journal(path)

    assert len(appended) == 1
    assert loaded["trade_date"].tolist() == ["2026-07-06"]
    row = loaded.iloc[0]
    assert row["symbol"] == "000001"
    assert row["executed_as_planned"] is False
    assert row["discipline_score"] == 7
    assert row["execution_score"] == 8
    assert row["review_score"] == 9
    assert row["system_score"] == 8.0
    assert row["summary"] == "按计划等待，没有追高。"


def test_summarize_review_journal_reports_execution_rate_scores_and_common_bias_reasons():
    journal = pd.DataFrame(
        [
            {
                "trade_date": "2026-07-01",
                "symbol": "000001",
                "planned_action": "按计划低吸",
                "actual_action": "追高买入",
                "executed_as_planned": False,
                "discipline_score": 6,
                "execution_score": 5,
                "review_score": 8,
                "system_score": 6.33,
                "summary": "偏差原因：追高；下次等待回踩。",
            },
            {
                "trade_date": "2026-07-02",
                "symbol": "600519",
                "planned_action": "持有",
                "actual_action": "按计划持有",
                "executed_as_planned": True,
                "discipline_score": 9,
                "execution_score": 8,
                "review_score": 9,
                "system_score": 8.67,
                "summary": "按计划执行。",
            },
            {
                "trade_date": "2026-07-03",
                "symbol": "300750",
                "planned_action": "等待突破",
                "actual_action": "提前买入",
                "executed_as_planned": False,
                "discipline_score": 7,
                "execution_score": 6,
                "review_score": 8,
                "system_score": 7.0,
                "summary": "偏差原因：追高；仓位过重。",
            },
        ]
    )

    summary = summarize_review_journal(journal)

    assert summary["total_entries"] == 3
    assert summary["plan_execution_win_rate_pct"] == 33.33
    assert summary["average_discipline_score"] == 7.33
    assert summary["average_system_score"] == 7.33
    assert summary["common_bias_reasons"] == [{"reason": "追高", "count": 2}]
