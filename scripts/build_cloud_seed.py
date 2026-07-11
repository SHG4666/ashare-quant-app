from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from ashare_quant.data import (
    CLOUD_DATA_DIR,
    CLOUD_HISTORY_DIR,
    SEQUOIA_DB_PATH,
    fetch_sequoia_sqlite_daily,
    fetch_tencent_stock_quotes,
)
from ashare_quant.market_scan import scan_local_market_candidates
from ashare_quant.watchlist import load_watchlist


def build_watchlist_histories() -> dict[str, object]:
    symbols = load_watchlist()
    quotes = fetch_tencent_stock_quotes(symbols)
    CLOUD_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    start = date.today() - timedelta(days=365 * 3)
    end = date.today()
    row_count = 0

    for symbol in symbols:
        quote = quotes.get(symbol, {})
        price = float(quote.get("price") or 0)
        if price <= 0:
            raise RuntimeError(f"无法获取 {symbol} 的正常市场价格")
        history = fetch_sequoia_sqlite_daily(symbol, start, end, SEQUOIA_DB_PATH)
        latest_adjusted_close = float(history["close"].iloc[-1])
        scale = price / latest_adjusted_close
        for column in ["open", "high", "low", "close"]:
            history[column] = (history[column].astype(float) * scale).round(4)
        history.to_csv(CLOUD_HISTORY_DIR / f"{symbol}.csv", index=False)
        row_count += len(history)

    return {"symbol_count": len(symbols), "row_count": row_count}


def build_market_snapshot() -> dict[str, object]:
    candidates = scan_local_market_candidates(
        SEQUOIA_DB_PATH,
        top_n=500,
        min_turnover=50_000_000.0,
        lookback_rows=120,
    )
    summary = dict(candidates.attrs.get("scan_summary", {}))
    CLOUD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CLOUD_DATA_DIR / "market_candidates.csv", index=False)
    metadata = {
        "latest_date": summary.get("latest_date", ""),
        "total_symbols": int(summary.get("total_symbols", 0)),
        "eligible_symbols": int(summary.get("eligible_symbols", len(candidates))),
        "snapshot_candidates": len(candidates),
        "generated_on": date.today().isoformat(),
    }
    (CLOUD_DATA_DIR / "market_snapshot.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    if not SEQUOIA_DB_PATH.exists():
        raise FileNotFoundError(f"Sequoia-X database not found: {SEQUOIA_DB_PATH}")
    histories = build_watchlist_histories()
    market = build_market_snapshot()
    print({"watchlist": histories, "market": market})
