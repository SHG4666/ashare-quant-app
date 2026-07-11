from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

from ashare_quant.market_scan import (
    format_feishu_market_scan_message,
    load_market_symbols_from_sequoia,
    merge_top_candidates_into_watchlist,
    scan_market_universe,
    select_top_candidates,
    send_feishu_text,
)
from ashare_quant.watchlist import DEFAULT_WATCHLIST_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan all A-shares, select Top10 and optionally send to Feishu.")
    parser.add_argument("--days", type=int, default=365, help="Historical days to fetch for each symbol")
    parser.add_argument("--top-n", type=int, default=10, help="Number of candidates to select")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit symbols for test runs")
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST_PATH, help="Watchlist path to update")
    parser.add_argument("--send-feishu", action="store_true", help="Send result to Feishu if FEISHU_WEBHOOK_URL is set")
    parser.add_argument("--no-update-watchlist", action="store_true", help="Do not write Top candidates into watchlist")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)
    symbols = load_market_symbols_from_sequoia()
    scan_result = scan_market_universe(symbols, start=start, end=end, max_symbols=args.max_symbols)
    top = select_top_candidates(scan_result, top_n=args.top_n)
    summary = scan_result.attrs.get("scan_summary", {})
    total_symbols = int(summary.get("total_symbols", len(symbols if args.max_symbols is None else symbols[: args.max_symbols])))
    scanned_symbols = int(summary.get("scanned_symbols", 0))
    failed_count = int(summary.get("failed_count", 0))
    message = format_feishu_market_scan_message(top, total_symbols, scanned_symbols, failed_count)

    if not args.no_update_watchlist:
        merge_top_candidates_into_watchlist(top, args.watchlist)
        print(f"WATCHLIST_UPDATED {args.watchlist}")

    print(message)

    if args.send_feishu:
        webhook = os.environ.get("FEISHU_WEBHOOK_URL") or os.environ.get("LARK_WEBHOOK_URL")
        if webhook:
            send_feishu_text(webhook, message)
            print("FEISHU_SENT")
        else:
            print("FEISHU_SKIPPED missing FEISHU_WEBHOOK_URL or LARK_WEBHOOK_URL")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
