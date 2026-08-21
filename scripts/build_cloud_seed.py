from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, time as clock_time, timedelta

import pandas as pd

from ashare_quant.data import (
    CLOUD_DATA_DIR,
    CLOUD_HISTORY_DIR,
    SEQUOIA_DB_PATH,
    china_market_today,
    expected_latest_business_day,
    fetch_baostock_daily,
    fetch_eastmoney_daily_with_curl,
    fetch_latest_market_quote,
    fetch_tencent_stock_quotes,
    validate_ohlcv_frame,
)
from ashare_quant.watchlist import load_watchlist

CLOUD_HISTORY_READY_TIME = clock_time(18, 0)


def _expected_cloud_history_date(end: date, now: datetime | None = None) -> date:
    """Return the session a delayed end-of-day provider should have published."""
    return expected_latest_business_day(
        end,
        now=now,
        data_ready_time=CLOUD_HISTORY_READY_TIME,
    )


def _load_previous_manifest() -> dict[str, object]:
    path = CLOUD_DATA_DIR / "watchlist_history_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _quote_is_complete(quote: object) -> bool:
    if not isinstance(quote, dict):
        return False
    try:
        price = float(quote.get("price") or 0)
        quote_time = pd.to_datetime(quote.get("quote_time"), errors="coerce")
    except (TypeError, ValueError):
        return False
    return price > 0 and not pd.isna(quote_time)


def _validate_history_against_quote(
    symbol: str,
    history: pd.DataFrame,
    quote: dict[str, object],
    expected_date: date,
) -> pd.DataFrame:
    """Return normalized history only when it reaches the latest completed session."""
    checked = validate_ohlcv_frame(history, symbol)
    latest_date = checked["date"].max().date()
    if latest_date < expected_date:
        raise RuntimeError(
            f"{symbol} 历史最新交易日 {latest_date.isoformat()}，"
            f"预期至少到 {expected_date.isoformat()}"
        )

    quote_time = pd.to_datetime(quote.get("quote_time"), errors="coerce")
    price = float(quote.get("price") or 0)
    if pd.isna(quote_time) or price <= 0:
        raise RuntimeError(f"无法获取 {symbol} 的完整正常市场行情")

    quote_date = quote_time.date()
    if quote_date < latest_date:
        raise RuntimeError(
            f"{symbol} 实时行情日期 {quote_date.isoformat()} "
            f"早于历史行情 {latest_date.isoformat()}"
        )

    # The latest qfq close should agree with the close quote when both are for
    # the same completed day. If the quote is intraday for a newer day, the
    # completed-session freshness check above is the applicable guard instead.
    if quote_date == latest_date:
        latest_close = float(checked["close"].iloc[-1])
        tolerance = max(0.02, abs(price) * 0.001)
        if abs(latest_close - price) > tolerance:
            raise RuntimeError(
                f"{symbol} 前复权末日收盘 {latest_close:.4f} "
                f"与最新价 {price:.4f} 不一致"
            )
    return checked


def _fetch_verified_history(
    symbol: str,
    start: date,
    end: date,
    quote: dict[str, object],
) -> tuple[pd.DataFrame, str]:
    """Fetch a verified qfq series, falling back when one provider is stale."""
    expected_date = _expected_cloud_history_date(end)
    provider_errors: list[str] = []
    providers = (
        ("baostock", fetch_baostock_daily, 3),
        ("东方财富K线", fetch_eastmoney_daily_with_curl, 2),
    )
    for source_name, fetcher, attempts in providers:
        history: pd.DataFrame | None = None
        for attempt in range(attempts):
            try:
                history = fetcher(symbol, start, end, "qfq")
                break
            except Exception as exc:
                provider_errors.append(f"{source_name}: {exc}")
                if attempt < attempts - 1:
                    time.sleep(2 * (attempt + 1))
        if history is None:
            continue
        try:
            return (
                _validate_history_against_quote(
                    symbol,
                    history,
                    quote,
                    expected_date,
                ),
                source_name,
            )
        except Exception as exc:
            # A stale provider will not become current after an immediate retry;
            # move to the next independent daily-K-line source instead.
            provider_errors.append(f"{source_name}: {exc}")

    detail = "；".join(provider_errors[-5:]) or "没有可用数据源"
    raise RuntimeError(f"{symbol} 前复权历史更新失败：{detail}")


def _history_metadata(
    symbol: str,
    history: pd.DataFrame,
    *,
    source_name: str,
    status: str,
    error: str = "",
) -> dict[str, object]:
    checked = validate_ohlcv_frame(history, symbol)
    metadata: dict[str, object] = {
        "rows": len(checked),
        "first_date": checked["date"].min().date().isoformat(),
        "latest_date": checked["date"].max().date().isoformat(),
        "latest_close": float(checked["close"].iloc[-1]),
        "source": source_name,
        "status": status,
    }
    if error:
        metadata["error"] = error
    return metadata


def _load_existing_history(symbol: str) -> pd.DataFrame | None:
    source = CLOUD_HISTORY_DIR / f"{symbol}.csv"
    if not source.exists():
        return None
    try:
        return pd.read_csv(source, parse_dates=["date"])
    except Exception:
        return None


def build_watchlist_histories(run_date: date | None = None) -> dict[str, object]:
    """Refresh symbols independently and preserve the last valid cloud snapshot."""
    symbols = load_watchlist()
    try:
        quotes = fetch_tencent_stock_quotes(symbols)
    except Exception:
        # A failed batch quote must not prevent per-symbol fallback requests.
        quotes = {}

    CLOUD_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = run_date or china_market_today()
    start = today - timedelta(days=365 * 3)
    end = today
    expected_date = _expected_cloud_history_date(end)
    previous_manifest = _load_previous_manifest()
    previous_symbols = previous_manifest.get("symbols", {})
    if not isinstance(previous_symbols, dict):
        previous_symbols = {}

    manifest_symbols: dict[str, dict[str, object]] = {}
    updated_symbols: list[str] = []
    failures: dict[str, str] = {}
    source_counts: dict[str, int] = {}

    for symbol in symbols:
        try:
            batch_quote = quotes.get(symbol)
            quote = batch_quote if _quote_is_complete(batch_quote) else fetch_latest_market_quote(symbol)
            history, source_name = _fetch_verified_history(symbol, start, end, quote)

            target = CLOUD_HISTORY_DIR / f"{symbol}.csv"
            temporary = target.with_suffix(".csv.tmp")
            history.to_csv(temporary, index=False)
            temporary.replace(target)

            manifest_symbols[symbol] = _history_metadata(
                symbol,
                history,
                source_name=source_name,
                status="updated",
            )
            updated_symbols.append(symbol)
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
        except Exception as exc:
            error = str(exc)
            failures[symbol] = error
            existing = _load_existing_history(symbol)
            if existing is not None:
                try:
                    manifest_symbols[symbol] = _history_metadata(
                        symbol,
                        existing,
                        source_name="既有云端备份",
                        status="stale",
                        error=error,
                    )
                except Exception:
                    manifest_symbols[symbol] = {"status": "invalid", "error": error}
            else:
                previous = previous_symbols.get(symbol)
                if isinstance(previous, dict):
                    stale_metadata = dict(previous)
                    stale_metadata["status"] = "stale"
                    stale_metadata["error"] = error
                    stale_metadata.setdefault("source", "既有云端备份")
                    manifest_symbols[symbol] = stale_metadata
                else:
                    manifest_symbols[symbol] = {"status": "missing", "error": error}
            print(f"WARNING: keeping previous cloud history for {symbol}: {error}")

    if not updated_symbols:
        raise RuntimeError(
            "所有自选股云端历史更新均失败；为避免提交无效数据，本次任务终止。"
            + (f" failures={failures}" if failures else "")
        )

    row_count = sum(int(item.get("rows", 0)) for item in manifest_symbols.values())
    manifest = {
        "source": "baostock/eastmoney",
        "adjust": "qfq",
        "generated_on": today.isoformat(),
        "expected_latest_date": expected_date.isoformat(),
        "updated_symbols": updated_symbols,
        "failed_symbols": failures,
        "symbols": manifest_symbols,
        "refresh": {
            "requested_symbols": len(symbols),
            "updated_symbols": len(updated_symbols),
            "failed_symbols": len(failures),
            "source_counts": source_counts,
            "failures": failures,
        },
    }
    (CLOUD_DATA_DIR / "watchlist_history_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "symbol_count": len(symbols),
        "updated_count": len(updated_symbols),
        "failed_count": len(failures),
        "failed_symbols": failures,
        "row_count": row_count,
        "source_counts": source_counts,
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
        "generated_on": china_market_today().isoformat(),
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
