from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

from .backtest import run_ma_cross_backtest
from .data import CLOUD_DATA_DIR, SEQUOIA_DB_PATH, fetch_ashare_daily
from .scanner import scan_stock_pool
from .watchlist import (
    DEFAULT_WATCHLIST_PATH,
    WatchlistEntry,
    load_watchlist_entries,
    save_watchlist_entries,
    without_watchlist_tags,
)


def load_market_symbols_from_sequoia(db_path: str | Path = SEQUOIA_DB_PATH) -> list[str]:
    """Load the available A-share universe from Sequoia-X's stock_daily table."""
    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Sequoia-X database not found: {source}")
    with sqlite3.connect(source) as conn:
        rows = conn.execute("select distinct symbol from stock_daily order by symbol").fetchall()
    return [str(row[0]) for row in rows if row and str(row[0]).isdigit() and len(str(row[0])) == 6]


def select_top_candidates(candidates: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top candidates sorted by selection score and secondary quality metrics."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if candidates.empty:
        return candidates.copy()
    sort_columns = [column for column in ["selection_score", "strategy_return_pct", "sharpe", "symbol"] if column in candidates.columns]
    ascending = [False if column != "symbol" else True for column in sort_columns]
    return candidates.sort_values(sort_columns, ascending=ascending, ignore_index=True).head(top_n)


def load_recent_market_history(
    db_path: str | Path = SEQUOIA_DB_PATH,
    lookback_rows: int = 120,
) -> pd.DataFrame:
    """Load a bounded recent history for every symbol from the local market DB."""
    if lookback_rows < 61:
        raise ValueError("lookback_rows must be at least 61")
    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Sequoia-X database not found: {source}")
    query = """
        WITH ranked AS (
            SELECT
                symbol, date, open, high, low, close, volume, turnover,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS row_number
            FROM stock_daily
            WHERE close IS NOT NULL AND close > 0 AND volume IS NOT NULL AND volume > 0
        )
        SELECT symbol, date, open, high, low, close, volume, turnover
        FROM ranked
        WHERE row_number <= ?
        ORDER BY symbol, date
    """
    with sqlite3.connect(source) as conn:
        return pd.read_sql_query(query, conn, params=(lookback_rows,), parse_dates=["date"])


def score_local_market_symbol(
    group: pd.DataFrame,
    latest_date: date,
    min_turnover: float = 100_000_000.0,
) -> dict[str, object] | None:
    """Apply quality gates, then score trend, momentum, volume and risk."""
    if len(group) < 61:
        return None
    group = group.sort_values("date")
    if pd.Timestamp(group["date"].iloc[-1]).date() != latest_date:
        return None
    close = pd.to_numeric(group["close"], errors="coerce")
    volume = pd.to_numeric(group["volume"], errors="coerce")
    if close.isna().any() or len(close) < 61:
        return None

    latest_close = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ret5 = float((latest_close / close.iloc[-6] - 1) * 100)
    ret20 = float((latest_close / close.iloc[-21] - 1) * 100)
    ret60 = float((latest_close / close.iloc[-61] - 1) * 100)
    turnover = float(group["turnover"].iloc[-1]) if pd.notna(group["turnover"].iloc[-1]) else 0.0
    previous_volume = float(volume.iloc[:-1].tail(20).mean())
    volume_ratio = float(volume.iloc[-1] / previous_volume) if previous_volume > 0 else 1.0
    daily_returns = close.pct_change().dropna().tail(20)
    volatility20 = float(daily_returns.std() * 100) if not daily_returns.empty else 0.0
    high60 = float(close.tail(60).max())
    distance_to_high60 = float((latest_close / high60 - 1) * 100) if high60 else 0.0

    # Quality gates keep weak, illiquid and already overheated names out before ranking.
    if (
        turnover < min_turnover
        or latest_close < ma20
        or ma20 < ma60
        or ret20 <= 0
        or ret5 < -3
        or ret5 > 25
        or ret20 > 60
    ):
        return None

    price_above_ma20 = (latest_close / ma20 - 1) * 100
    ma_alignment = (ma20 / ma60 - 1) * 100
    trend_score = max(0.0, min(35.0, 18 + price_above_ma20 * 1.8 + ma_alignment * 1.4))
    momentum_score = max(0.0, min(25.0, 10 + ret5 * 0.8 + ret20 * 0.35 + ret60 * 0.08))
    volume_score = max(0.0, min(15.0, 7 + (volume_ratio - 1) * 6))
    breakout_score = max(0.0, min(15.0, 15 + distance_to_high60 * 0.6))
    chasing_penalty = max(ret5 - 12, 0) * 0.35 + max(ret20 - 35, 0) * 0.15
    risk_score = max(0.0, min(10.0, 10 - max(volatility20 - 2.0, 0) * 1.8 - chasing_penalty))
    score = trend_score + momentum_score + volume_score + breakout_score + risk_score
    label = "重点观察" if score >= 80 else "稳健观察" if score >= 68 else "普通观察"
    return {
        "symbol": str(group["symbol"].iloc[-1]),
        "date": latest_date.isoformat(),
        "selection_score": round(score, 2),
        "score_label": label,
        "ret5_pct": round(ret5, 2),
        "ret20_pct": round(ret20, 2),
        "ret60_pct": round(ret60, 2),
        "volume_ratio": round(volume_ratio, 2),
        "turnover": round(turnover, 2),
        "volatility20_pct": round(volatility20, 2),
        "distance_to_high60_pct": round(distance_to_high60, 2),
        "score_explanation": (
            f"趋势{trend_score:.1f}/35；动量{momentum_score:.1f}/25；"
            f"量能{volume_score:.1f}/15；突破{breakout_score:.1f}/15；风险{risk_score:.1f}/10"
        ),
        "risk_note": "仅为量价筛选结果，需结合公告、基本面和次日价格确认。",
    }


def load_cloud_market_candidates(
    snapshot_path: str | Path = CLOUD_DATA_DIR / "market_candidates.csv",
    metadata_path: str | Path = CLOUD_DATA_DIR / "market_snapshot.json",
    top_n: int = 10,
    min_turnover: float = 100_000_000.0,
) -> pd.DataFrame:
    """Load the compact full-market candidate snapshot used in cloud hosting."""
    source = Path(snapshot_path)
    if not source.exists():
        raise FileNotFoundError(f"云端全市场候选快照不存在：{source}")
    candidates = pd.read_csv(source, dtype={"symbol": str})
    if "turnover" in candidates.columns:
        turnover = pd.to_numeric(candidates["turnover"], errors="coerce").fillna(0.0)
        candidates = candidates[turnover >= float(min_turnover)]
    eligible_count = len(candidates)
    candidates = candidates.sort_values(
        ["selection_score", "ret20_pct", "volume_ratio", "symbol"],
        ascending=[False, False, False, True],
        ignore_index=True,
    ).head(top_n)
    if "rank" in candidates.columns:
        candidates = candidates.drop(columns=["rank"])
    candidates.insert(0, "rank", range(1, len(candidates) + 1))

    metadata: dict[str, object] = {}
    metadata_source = Path(metadata_path)
    if metadata_source.exists():
        metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    latest_date = str(candidates["date"].max()) if not candidates.empty and "date" in candidates else str(metadata.get("latest_date", ""))
    candidates.attrs["scan_summary"] = {
        "latest_date": latest_date,
        "total_symbols": int(metadata.get("total_symbols", 0)),
        "eligible_symbols": int(eligible_count),
        "candidate_count": int(len(candidates)),
        "min_turnover": float(min_turnover),
        "source_name": "云端全市场候选快照",
    }
    return candidates


def scan_local_market_candidates(
    db_path: str | Path = SEQUOIA_DB_PATH,
    top_n: int = 10,
    min_turnover: float = 100_000_000.0,
    lookback_rows: int = 120,
) -> pd.DataFrame:
    """Scan the local A-share universe and return ranked, quality-gated candidates."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if min_turnover < 0:
        raise ValueError("min_turnover must be non-negative")
    if not Path(db_path).exists():
        return load_cloud_market_candidates(top_n=top_n, min_turnover=min_turnover)
    history = load_recent_market_history(db_path, lookback_rows=lookback_rows)
    if history.empty:
        result = pd.DataFrame()
        result.attrs["scan_summary"] = {"latest_date": "", "total_symbols": 0, "eligible_symbols": 0}
        return result

    latest_date = pd.Timestamp(history["date"].max()).date()
    rows: list[dict[str, object]] = []
    eligible_prefixes = ("0", "3", "6")
    for symbol, group in history.groupby("symbol", sort=True):
        symbol = str(symbol)
        if not symbol.startswith(eligible_prefixes) or len(symbol) != 6:
            continue
        row = score_local_market_symbol(group, latest_date=latest_date, min_turnover=min_turnover)
        if row is not None:
            rows.append(row)
    scored = pd.DataFrame(rows)
    if scored.empty:
        result = scored
    else:
        result = scored.sort_values(
            ["selection_score", "ret20_pct", "volume_ratio", "symbol"],
            ascending=[False, False, False, True],
            ignore_index=True,
        ).head(top_n)
        result.insert(0, "rank", range(1, len(result) + 1))
    result.attrs["scan_summary"] = {
        "latest_date": latest_date.isoformat(),
        "total_symbols": int(history["symbol"].nunique()),
        "eligible_symbols": int(len(scored)),
        "candidate_count": int(len(result)),
        "min_turnover": float(min_turnover),
    }
    return result


def merge_top_candidates_into_watchlist(
    top_candidates: pd.DataFrame,
    watchlist_path: str | Path = DEFAULT_WATCHLIST_PATH,
) -> Path:
    """Append Top10 market-scan symbols to the watchlist while preserving existing rows."""
    existing = without_watchlist_tags(load_watchlist_entries(watchlist_path))
    existing_symbols = {entry.symbol for entry in existing}
    merged: list[WatchlistEntry] = list(existing)
    for _, row in top_candidates.iterrows():
        symbol = str(row["symbol"])
        if symbol in existing_symbols:
            continue
        score = float(row.get("selection_score", 0.0))
        label = str(row.get("score_label", "候选"))
        explanation = str(row.get("score_explanation", ""))
        merged.append(
            WatchlistEntry(
                symbol=symbol,
                name=str(row.get("name", "")),
                note=f"自动选股{label}，评分{score:.2f}；{explanation}",
            )
        )
        existing_symbols.add(symbol)
    return save_watchlist_entries(watchlist_path, merged)


def format_feishu_market_scan_message(
    top_candidates: pd.DataFrame,
    total_symbols: int,
    scanned_symbols: int,
    failed_count: int,
) -> str:
    """Build a concise Feishu text message for market scan results."""
    lines = [
        "📈 全市场A股扫描Top10",
        f"扫描范围：{scanned_symbols}/{total_symbols} 只；失败：{failed_count} 只",
        "",
    ]
    if top_candidates.empty:
        lines.append("本次没有筛出候选股。")
    else:
        for rank, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            lines.append(
                f"{rank}. {row['symbol']}｜评分{float(row.get('selection_score', 0.0)):.2f}｜"
                f"{row.get('score_label', '')}｜收盘{float(row.get('close', 0.0)):.2f}"
            )
            explanation = str(row.get("score_explanation", ""))
            if explanation:
                lines.append(f"   {explanation}")
    lines.extend(["", "仅用于盘后研究和交易体系打磨，不构成投资建议。"])
    return "\n".join(lines)


def send_feishu_text(webhook_url: str, text: str) -> None:
    """Send a plain text message to a Feishu/Lark bot webhook."""
    if not webhook_url:
        raise ValueError("webhook_url is required")
    response = requests.post(
        webhook_url,
        json={"msg_type": "text", "content": {"text": text}},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    code = payload.get("code", payload.get("StatusCode", 0))
    if code not in (0, "0"):
        raise RuntimeError(f"Feishu webhook returned error: {payload}")


def scan_market_universe(
    symbols: Iterable[str],
    fetcher: Callable[[str, date, date, str], pd.DataFrame] = fetch_ashare_daily,
    start: date | None = None,
    end: date | None = None,
    adjust: str = "qfq",
    max_symbols: int | None = None,
    initial_cash: float = 100_000.0,
) -> pd.DataFrame:
    """Fetch data for a universe and run the current default MA scanner.

    The default fetcher is `fetch_ashare_daily`, which now prefers the Ashare
    Sina/Tencent real-time interface for recent requests before falling back to
    slower/staler sources.
    """
    end = end or date.today()
    start = start or end - timedelta(days=365)
    selected = list(symbols)[:max_symbols] if max_symbols else list(symbols)
    data_by_symbol: dict[str, pd.DataFrame] = {}
    failed_symbols: list[dict[str, str]] = []
    for symbol in selected:
        try:
            df = fetcher(symbol, start, end, adjust)
            if not df.empty:
                data_by_symbol[symbol] = df
        except Exception as exc:
            failed_symbols.append({"symbol": symbol, "stage": "data_fetch", "error": str(exc)})
    if not data_by_symbol:
        result = pd.DataFrame()
        result.attrs["failed_symbols"] = failed_symbols
        result.attrs["scan_summary"] = {
            "total_symbols": len(selected),
            "scanned_symbols": 0,
            "candidate_count": 0,
            "failed_count": len(failed_symbols),
        }
        return result
    result = scan_stock_pool(data_by_symbol, run_ma_cross_backtest, initial_cash=initial_cash, short_window=5, long_window=20)
    result.attrs["failed_symbols"] = [*failed_symbols, *result.attrs.get("failed_symbols", [])]
    result.attrs["scan_summary"] = {
        **result.attrs.get("scan_summary", {}),
        "total_symbols": len(selected),
        "failed_count": len(result.attrs["failed_symbols"]),
    }
    return result
