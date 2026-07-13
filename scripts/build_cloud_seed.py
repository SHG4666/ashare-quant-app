from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

from ashare_quant.data import (
    CLOUD_DATA_DIR,
    CLOUD_HISTORY_DIR,
    SEQUOIA_DB_PATH,
    fetch_baostock_daily,
    fetch_tencent_stock_quotes,
)
from ashare_quant.watchlist import load_watchlist


def build_watchlist_histories() -> dict[str, object]:
    symbols = load_watchlist()
    quotes = fetch_tencent_stock_quotes(symbols)
    CLOUD_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    start = date.today() - timedelta(days=365 * 3)
    end = date.today()
    row_count = 0
    manifest_symbols: dict[str, dict[str, object]] = {}

    for symbol in symbols:
        quote = quotes.get(symbol, {})
        price = float(quote.get("price") or 0)
        if price <= 0:
            raise RuntimeError(f"无法获取 {symbol} 的正常市场价格")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                history = fetch_baostock_daily(symbol, start, end, "qfq")
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"{symbol} baostock 前复权下载失败") from last_error
        latest_close = float(history["close"].iloc[-1])
        latest_date = history["date"].max().date()
        quote_time = quote.get("quote_time")
        if quote_time is None or quote_time.date() != latest_date:
            raise RuntimeError(
                f"{symbol} baostock 末日 {latest_date.isoformat()} 与腾讯行情日期不一致"
            )
        if abs(latest_close - price) > 0.011:
            raise RuntimeError(
                f"{symbol} 前复权末日收盘 {latest_close:.4f} 与腾讯最新价 {price:.4f} 不一致"
            )
        history.to_csv(CLOUD_HISTORY_DIR / f"{symbol}.csv", index=False)
        row_count += len(history)
        manifest_symbols[symbol] = {
            "rows": len(history),
            "first_date": history["date"].min().date().isoformat(),
            "latest_date": history["date"].max().date().isoformat(),
            "latest_close": latest_close,
        }

    manifest = {
        "source": "baostock",
        "adjust": "qfq",
        "generated_on": date.today().isoformat(),
        "symbols": manifest_symbols,
    }
    (CLOUD_DATA_DIR / "watchlist_history_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {"symbol_count": len(symbols), "row_count": row_count}


def build_market_snapshot() -> dict[str, object]:
    from ashare_quant.market_scan import scan_local_market_candidates

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified cloud market-data backups")
    parser.add_argument(
        "--histories-only",
        action="store_true",
        help="Update watchlist histories without requiring the local Sequoia-X database",
    )
    args = parser.parse_args()
    histories = build_watchlist_histories()
    if args.histories_only:
        market: dict[str, object] = {"skipped": True}
    else:
        if not SEQUOIA_DB_PATH.exists():
            raise FileNotFoundError(f"Sequoia-X database not found: {SEQUOIA_DB_PATH}")
        market = build_market_snapshot()
    print({"watchlist": histories, "market": market})


if __name__ == "__main__":
    main()
