from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]
DEFAULT_REVIEW_JOURNAL_PATH = Path("data_cache/review_journal.csv")

COLUMNS = [
    "trade_date",
    "symbol",
    "planned_action",
    "actual_action",
    "executed_as_planned",
    "discipline_score",
    "execution_score",
    "review_score",
    "system_score",
    "summary",
]


def _validate_score(name: str, value: int | float) -> float:
    score = float(value)
    if not 0 <= score <= 10:
        raise ValueError(f"{name} must be between 0 and 10")
    return score


def _format_score(value: float) -> int | float:
    return int(value) if value.is_integer() else round(value, 2)


def load_review_journal(path: PathLike = DEFAULT_REVIEW_JOURNAL_PATH) -> pd.DataFrame:
    """Load the local review journal CSV, returning a stable empty schema when absent."""
    journal_path = Path(path)
    if not journal_path.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(journal_path, dtype={"trade_date": str, "symbol": str})
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[COLUMNS]
    if not df.empty:
        df["executed_as_planned"] = df["executed_as_planned"].map(lambda value: bool(value)).astype(object)
    return df


def append_review_entry(
    path: PathLike,
    trade_date: str,
    symbol: str,
    planned_action: str,
    actual_action: str,
    executed_as_planned: bool,
    discipline_score: int | float,
    execution_score: int | float,
    review_score: int | float,
    summary: str,
) -> pd.DataFrame:
    """Append one plan-execution review row and persist it to local CSV."""
    discipline = _validate_score("discipline_score", discipline_score)
    execution = _validate_score("execution_score", execution_score)
    review = _validate_score("review_score", review_score)
    system_score = round((discipline + execution + review) / 3, 2)

    row = {
        "trade_date": str(trade_date),
        "symbol": str(symbol).strip(),
        "planned_action": str(planned_action),
        "actual_action": str(actual_action),
        "executed_as_planned": bool(executed_as_planned),
        "discipline_score": _format_score(discipline),
        "execution_score": _format_score(execution),
        "review_score": _format_score(review),
        "system_score": _format_score(system_score),
        "summary": str(summary),
    }

    df = pd.concat([load_review_journal(path), pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(journal_path, index=False, encoding="utf-8-sig")
    return df


def _extract_bias_reason(summary: object) -> str | None:
    text = str(summary or "").strip()
    marker = "偏差原因"
    if marker not in text:
        return None
    reason = text.split(marker, 1)[1]
    reason = reason.lstrip("：: ")
    for separator in ["；", ";", "。", ".", "\n", "，", ","]:
        reason = reason.split(separator, 1)[0]
    reason = reason.strip()
    return reason or None


def summarize_review_journal(journal: pd.DataFrame, top_n: int = 3) -> dict[str, object]:
    """Summarize review journal execution discipline and recurring bias reasons."""
    if journal.empty:
        return {
            "total_entries": 0,
            "plan_execution_win_rate_pct": 0.0,
            "average_discipline_score": 0.0,
            "average_system_score": 0.0,
            "common_bias_reasons": [],
        }

    executed = journal["executed_as_planned"].astype(bool)
    reasons = Counter(
        reason
        for reason in journal.loc[~executed, "summary"].map(_extract_bias_reason).dropna().tolist()
        if reason
    )
    common_bias_reasons = [
        {"reason": reason, "count": count}
        for reason, count in reasons.most_common(top_n)
    ]
    return {
        "total_entries": int(len(journal)),
        "plan_execution_win_rate_pct": round(float(executed.mean() * 100), 2),
        "average_discipline_score": round(float(journal["discipline_score"].astype(float).mean()), 2),
        "average_system_score": round(float(journal["system_score"].astype(float).mean()), 2),
        "common_bias_reasons": common_bias_reasons,
    }
