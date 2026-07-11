from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_quant.market_scan import send_feishu_text
from ashare_quant.watchlist import (
    DEFAULT_WATCHLIST_PATH,
    WatchlistEntry,
    load_watchlist_entries,
    save_watchlist_entries,
    without_watchlist_tags,
)

SEQUOIA_DB = Path("/Users/sunhuaiguo/CODE/Sequoia-X/data/sequoia_v2.db")


def load_recent_sequoia_data(db_path: Path = SEQUOIA_DB, lookback_rows: int = 120) -> pd.DataFrame:
    query = """
    WITH ranked AS (
        SELECT
            symbol, date, open, high, low, close, volume, turnover,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
        FROM stock_daily
        WHERE close IS NOT NULL AND volume IS NOT NULL AND volume > 0
    )
    SELECT symbol, date, open, high, low, close, volume, turnover
    FROM ranked
    WHERE rn <= ?
    ORDER BY symbol, date
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(lookback_rows,), parse_dates=["date"])
    return df


def score_symbol(group: pd.DataFrame) -> dict[str, object] | None:
    if len(group) < 60:
        return None
    group = group.sort_values("date").copy()
    close = group["close"].astype(float)
    volume = group["volume"].astype(float)
    latest_close = float(close.iloc[-1])
    latest_date = group["date"].iloc[-1].date()
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ret5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
    ret60 = float((close.iloc[-1] / close.iloc[-61] - 1) * 100)
    high20 = float(close.tail(20).max())
    high60 = float(close.tail(60).max())
    drawdown20 = float((latest_close / high20 - 1) * 100) if high20 else 0.0
    vol_ratio = float(volume.iloc[-1] / volume.iloc[:-1].tail(20).mean()) if len(volume) > 21 and volume.iloc[:-1].tail(20).mean() > 0 else 1.0
    turnover = float(group["turnover"].iloc[-1]) if "turnover" in group.columns and pd.notna(group["turnover"].iloc[-1]) else 0.0

    trend_score = max(0.0, min(35.0, 15 + (latest_close / ma20 - 1) * 120 + (ma20 / ma60 - 1) * 80)) if ma20 and ma60 else 0.0
    momentum_score = max(0.0, min(30.0, 10 + ret5 * 1.4 + ret20 * 0.45 + ret60 * 0.15))
    volume_score = max(0.0, min(20.0, 8 + (vol_ratio - 1) * 8))
    breakout_score = 15.0 if latest_close >= high60 * 0.995 else max(0.0, 12 + drawdown20 * 0.8)
    liquidity_penalty = 0.0 if turnover >= 100_000_000 else 8.0
    score = trend_score + momentum_score + volume_score + breakout_score - liquidity_penalty

    if latest_close < ma20 or ma20 < ma60:
        score -= 12.0
    if ret5 < -3:
        score -= 10.0

    label = "强势候选" if score >= 80 else "稳健候选" if score >= 65 else "观察候选" if score >= 50 else "低优先级"
    return {
        "symbol": str(group["symbol"].iloc[-1]),
        "date": latest_date.isoformat(),
        "close": latest_close,
        "selection_score": round(score, 2),
        "score_label": label,
        "ret5_pct": round(ret5, 2),
        "ret20_pct": round(ret20, 2),
        "ret60_pct": round(ret60, 2),
        "vol_ratio": round(vol_ratio, 2),
        "turnover": turnover,
        "score_explanation": f"趋势{trend_score:.1f}/35；动量{momentum_score:.1f}/30；量能{volume_score:.1f}/20；突破{breakout_score:.1f}/15；5日{ret5:+.1f}%、20日{ret20:+.1f}%、量比{vol_ratio:.2f}",
    }


def run_scan(top_n: int = 10) -> tuple[pd.DataFrame, dict[str, object]]:
    df = load_recent_sequoia_data()
    latest_date = df["date"].max().date().isoformat()
    rows = []
    for _, group in df.groupby("symbol", sort=True):
        row = score_symbol(group)
        if row is not None and row["date"] == latest_date:
            rows.append(row)
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored, {"latest_date": latest_date, "total_symbols": int(df["symbol"].nunique()), "scored_symbols": 0}
    top = scored.sort_values(["selection_score", "ret20_pct", "vol_ratio", "symbol"], ascending=[False, False, False, True]).head(top_n).reset_index(drop=True)
    return top, {"latest_date": latest_date, "total_symbols": int(df["symbol"].nunique()), "scored_symbols": int(len(scored))}


def merge_watchlist(top: pd.DataFrame) -> None:
    existing = without_watchlist_tags(load_watchlist_entries(DEFAULT_WATCHLIST_PATH))
    seen = {entry.symbol for entry in existing}
    merged = list(existing)
    for _, row in top.iterrows():
        symbol = str(row["symbol"])
        if symbol in seen:
            continue
        merged.append(
            WatchlistEntry(
                symbol=symbol,
                note=f"自动选股{row.get('score_label', '候选')}，评分{float(row['selection_score']):.2f}；{row.get('score_explanation', '')}",
            )
        )
        seen.add(symbol)
    save_watchlist_entries(DEFAULT_WATCHLIST_PATH, merged)


def format_message(top: pd.DataFrame, summary: dict[str, object]) -> str:
    lines = [
        "📈 Sequoia-X 思路全市场A股扫描Top10",
        f"数据源：Sequoia-X 本地 SQLite（后复权日线）",
        f"数据截止：{summary['latest_date']}｜股票池：{summary['scored_symbols']}/{summary['total_symbols']} 只",
        "",
    ]
    for idx, row in top.iterrows():
        lines.append(
            f"{idx + 1}. {row['symbol']}｜评分{float(row['selection_score']):.2f}｜{row['score_label']}｜收盘{float(row['close']):.2f}"
        )
        lines.append(f"   {row['score_explanation']}")
    lines.extend(["", "说明：本次按 Sequoia-X 的本地库/盘后批量选股思路生成；仅用于研究学习和交易体系打磨，不构成投资建议。"])
    return "\n".join(lines)


def main() -> int:
    top, summary = run_scan(top_n=10)
    merge_watchlist(top)
    message = format_message(top, summary)
    print(message)
    webhook = os.environ.get("FEISHU_WEBHOOK_URL") or os.environ.get("LARK_WEBHOOK_URL")
    if webhook:
        send_feishu_text(webhook, message)
        print("FEISHU_SENT")
    else:
        print("FEISHU_SKIPPED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
