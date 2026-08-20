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


def _load_previous_manifest() -> dict[str, object]:
    path = CLOUD_DATA_DIR / "watchlist_history_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_watchlist_histories() -> dict[str, object]:
    symbols = load_watchlist()
    quotes = fetch_tencent_stock_quotes(symbols)
    CLOUD_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    start = date.today() - timedelta(days=365 * 3)
    end = date.today()
    row_count = 0
    updated_count = 0
    failures: dict[str, str] = {}
    previous_manifest = _load_previous_manifest()
    previous_symbols = previous_manifest.get("symbols", {})
    if not isinstance(previous_symbols, dict):
        previous_symbols = {}
    manifest_symbols: dict[str, dict[str, object]] = {}

    for symbol in symbols:
        try:
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

            if history.empty:
                raise RuntimeError(f"{symbol} baostock 返回空行情")

            latest_close = float(history["close"].iloc[-1])
            latest_date = history["date"].max().date()
            quote_time = quote.get("quote_time")

            # Tencent quotes can advance before baostock has published the same
            # trading day's adjusted history. Treat that as a temporary source
            # lag instead of failing the entire watchlist refresh.
            if quote_time is None:
                raise RuntimeError(f"{symbol} 腾讯行情缺少时间")
            if quote_time.date() < latest_date:
                raise RuntimeError(
                    f"{symbol} 腾讯行情日期 {quote_time.date().isoformat()} 早于 baostock {latest_date.isoformat()}"
                )
            if quote_time.date() == latest_date and abs(latest_close - price) > 0.011:
                raise RuntimeError(
                    f"{symbol} 前复权末日收盘 {latest_close:.4f} 与腾讯最新价 {price:.4f} 不一致"
                )

            history.to_csv(CLOUD_HISTORY_DIR / f"{symbol}.csv", index=False)
            row_count += len(history)
            updated_count += 1
            manifest_symbols[symbol] = {
                "rows": len(history),
                "first_date": history["date"].min().date().isoformat(),
                "latest_date": latest_date.isoformat(),
                "latest_close": latest_close,
            }
        except Exception as exc:
            failures[symbol] = str(exc)
            previous = previous_symbols.get(symbol)
            if isinstance(previous, dict):
                manifest_symbols[symbol] = previous
            print(f"WARNING: keeping previous cloud history for {symbol}: {exc}")

    if updated_count == 0:
        raise RuntimeError(
            "所有自选股云端历史更新均失败；为避免提交无效数据，本次任务终止。"
            + (f" failures={failures}" if failures else "")
        )

    manifest = {
        "source": "baostock",
        "adjust": "qfq",
        "generated_on": date.today().isoformat(),
        "symbols": manifest_symbols,
        "refresh": {
            "requested_symbols": len(symbols),
            "updated_symbols": updated_count,
            "failed_symbols": len(failures),
            "failures": failures,
        },
    }
    (CLOUD_DATA_DIR / "watchlist_history_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "symbol_count": len(symbols),
        "updated_count": updated_count,
        "failed_count": len(failures),
        "row_count": row_count,
    }


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
