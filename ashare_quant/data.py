from __future__ import annotations

import json
import importlib.util
import os
import sqlite3
import subprocess
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
CLOUD_DATA_DIR = PROJECT_ROOT / "cloud_data"
CLOUD_HISTORY_DIR = CLOUD_DATA_DIR / "watchlist_history"
SEQUOIA_DB_PATH = Path(
    os.environ.get("ASHARE_SEQUOIA_DB", str(Path.home() / "CODE/Sequoia-X/data/sequoia_v2.db"))
)
ASHARE_MODULE_PATH = Path(
    os.environ.get("ASHARE_MODULE_PATH", str(Path.home() / "CODE/Ashare/Ashare.py"))
)


def normalize_akshare_hist(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize akshare A-share historical dataframe to English columns."""
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = raw.rename(columns=column_map).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns from data source: {missing}")
    df = df[required + [c for c in ["amount", "pct_change", "turnover"] if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def baostock_code(symbol: str) -> str:
    """Convert a plain A-share code to baostock exchange-prefixed code."""
    symbol = symbol.strip()
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}.{symbol}"


def ashare_code(symbol: str) -> str:
    """Convert a plain A-share code to Ashare's sh/sz-prefixed code."""
    symbol = symbol.strip().replace(".XSHG", "").replace(".XSHE", "")
    if symbol.startswith(("sh", "sz")):
        return symbol
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}{symbol}"


def load_ashare_get_price():
    """Load `/Users/sunhuaiguo/CODE/Ashare/Ashare.py` without requiring installation."""
    if not ASHARE_MODULE_PATH.exists():
        raise FileNotFoundError(f"Ashare.py not found: {ASHARE_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("external_ashare", ASHARE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Ashare module from {ASHARE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_price


def normalize_ashare_price_frame(raw: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Normalize mpquant/Ashare DataFrame to this app's OHLCV schema."""
    if raw.empty:
        raise ValueError("Ashare returned no rows")
    df = raw.copy()
    df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Ashare missing required columns: {missing}")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    return df[required].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_ashare_lightweight_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV using the lightweight mpquant/Ashare Sina/Tencent wrapper."""
    df = fetch_ashare_lightweight_bars(symbol, frequency="1d", count=max((end - start).days + 10, 10), end_date=end.isoformat())
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def fetch_ashare_lightweight_bars(
    symbol: str,
    frequency: str = "1d",
    count: int = 120,
    end_date: str = "",
) -> pd.DataFrame:
    """Fetch daily or intraday OHLCV using mpquant/Ashare's get_price API."""
    get_price = load_ashare_get_price()
    raw = get_price(ashare_code(symbol), end_date=end_date, count=count, frequency=frequency)
    # Use a very wide range here; callers can slice if needed. This keeps the
    # same normalizer for daily and minute bars while preserving timestamps.
    return normalize_ashare_price_frame(raw, date(1900, 1, 1), date(2100, 1, 1))


def baostock_adjust_flag(adjust: str) -> str:
    """Map UI adjustment option to baostock adjustflag.

    baostock: 1=后复权, 2=前复权, 3=不复权.
    """
    return {"hfq": "1", "qfq": "2", "": "3"}.get(adjust, "3")


def adjustment_label(adjust: str) -> str:
    """Return the human-readable price adjustment label used in the UI."""
    return {"qfq": "前复权", "hfq": "后复权", "": "不复权"}.get(adjust, "不复权")


def with_data_source(df: pd.DataFrame, source_name: str, *, price_verified: bool = True) -> pd.DataFrame:
    """Attach source and verification metadata for UI/API status display."""
    df.attrs["source_name"] = source_name
    df.attrs["price_verified"] = price_verified
    return df


def validate_ohlcv_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reject malformed bundled history before it can reach charts or backtests."""
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"行情缺少字段：{missing}")

    checked = df.copy()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    if checked[required].isna().any().any():
        raise ValueError(f"{symbol} 行情包含空值或非数字字段")
    if checked["date"].duplicated().any():
        raise ValueError(f"{symbol} 行情包含重复交易日")
    if (checked[["open", "high", "low", "close"]] <= 0).any().any() or (checked["volume"] < 0).any():
        raise ValueError(f"{symbol} 行情包含无效价格或成交量")

    price_max = checked[["open", "close"]].max(axis=1)
    price_min = checked[["open", "close"]].min(axis=1)
    invalid_bar = (checked["high"] < price_max) | (checked["low"] > price_min) | (checked["high"] < checked["low"])
    if invalid_bar.any():
        raise ValueError(f"{symbol} 行情包含不符合 OHLC 关系的数据")
    return checked.sort_values("date").reset_index(drop=True)


def is_recent_market_request(end: date, freshness_days: int = 3) -> bool:
    """Return True when the user asks for data ending close to today."""
    return 0 <= (date.today() - end).days <= freshness_days


def is_mainland_market_session(now: datetime | None = None) -> bool:
    """Return whether Shanghai/Shenzhen continuous trading may be active."""
    current = now or datetime.now()
    if current.weekday() >= 5:
        return False
    current_time = current.time()
    return time(9, 15) <= current_time <= time(15, 5)


def _business_days_between(start: date, end: date) -> int:
    """Count weekdays after start through end, without flagging weekends as stale."""
    if end <= start:
        return 0
    return len(pd.bdate_range(start=pd.Timestamp(start) + pd.Timedelta(days=1), end=pd.Timestamp(end)))


def expected_latest_business_day(end: date, now: datetime | None = None) -> date:
    """Estimate the latest completed trading day without treating weekends as stale."""
    current = now or datetime.now()
    expected = end
    if end >= current.date() and current.weekday() < 5 and current.time() < time(15, 5):
        expected = current.date() - timedelta(days=1)
    while expected.weekday() >= 5:
        expected -= timedelta(days=1)
    return expected


def is_price_history_fresh(df: pd.DataFrame, end: date, now: datetime | None = None) -> bool:
    """Return whether cached history reaches the latest completed business day."""
    if df.empty or "date" not in df.columns:
        return False
    latest = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(latest):
        return False
    return latest.date() >= expected_latest_business_day(end, now=now)


def summarize_price_data_status(
    df: pd.DataFrame,
    symbol: str,
    start: date,
    end: date,
    adjust: str,
    source_name: str,
) -> dict:
    """Summarize data freshness and price adjustment for display."""
    latest = pd.to_datetime(df["date"]).max().date()
    staleness_days = _business_days_between(latest, end)
    is_stale = staleness_days > 0
    requested_range = f"{start.isoformat()} 至 {end.isoformat()}"
    label = adjustment_label(adjust)
    caption = (
        f"{symbol}｜{source_name}｜{label}｜最新交易日 {latest.isoformat()}｜"
        f"请求区间 {requested_range}｜{len(df)} 条记录"
    )
    if is_stale:
        caption += f"｜可能滞后 {staleness_days} 天"
    return {
        "symbol": symbol,
        "source_name": source_name,
        "adjust_label": label,
        "latest_trade_day": latest.isoformat(),
        "requested_range": requested_range,
        "row_count": len(df),
        "is_stale": is_stale,
        "staleness_days": staleness_days,
        "caption": caption,
    }


def fetch_latest_market_quote(symbol: str) -> dict[str, object]:
    """Return the latest unadjusted market bar for a compact current-price card.

    A quote is deliberately separate from adjusted history. Mixing an unadjusted
    realtime series into a qfq/hfq backtest can create artificial jumps around
    dividends and splits.
    """
    params = {
        "secid": eastmoney_secid(symbol),
        "fields": "f43,f44,f45,f46,f47,f57,f58,f59,f60,f86",
    }
    url = "https://push2.eastmoney.com/api/qt/stock/get?" + urlencode(params)
    completed = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "--max-time", "8", url],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Eastmoney quote lookup failed")
    payload = json.loads(completed.stdout).get("data") or {}
    decimals = int(payload.get("f59") or 2)
    scale = 10**decimals

    def price(field: str) -> float:
        raw = payload.get(field)
        if raw in (None, "", "-"):
            raise ValueError(f"Eastmoney quote missing {field} for {symbol}")
        return float(raw) / scale

    timestamp = payload.get("f86")
    quote_time = pd.to_datetime(int(timestamp), unit="s") if timestamp else pd.Timestamp.now()
    return {
        "symbol": symbol,
        "name": str(payload.get("f58") or "").strip(),
        "quote_time": quote_time,
        "price": price("f43"),
        "open": price("f46"),
        "high": price("f44"),
        "low": price("f45"),
        "previous_close": price("f60"),
        "volume": float(payload.get("f47") or 0),
        "source_name": "东方财富实时行情",
        "price_basis": "不复权实时价",
    }


def fetch_eastmoney_stock_name(symbol: str) -> str:
    """Fetch a stock's short name without downloading the full market list."""
    params = {"secid": eastmoney_secid(symbol), "fields": "f57,f58"}
    url = "https://push2.eastmoney.com/api/qt/stock/get?" + urlencode(params)
    completed = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "--max-time", "8", url],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Eastmoney name lookup failed")
    name = str((json.loads(completed.stdout).get("data") or {}).get("f58") or "").strip()
    if not name:
        raise ValueError(f"Eastmoney returned no name for {symbol}")
    return name


def normalize_baostock_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Normalize baostock query rows to the app OHLCV schema."""
    columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    if rows and len(rows[0]) == 8:
        columns.append("tradestatus")
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise ValueError("baostock returned no rows")
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"].astype(str) == "1"].drop(columns="tradestatus")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date").reset_index(drop=True)


def fetch_baostock_daily(symbol: str, start: date, end: date, adjust: str = "") -> pd.DataFrame:
    """Fetch daily A-share OHLCV from baostock.

    This is the same primary data family used by Sequoia-X, but it respects the
    UI adjustment option so charts can match normal market software prices.
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            baostock_code(symbol),
            "date,open,high,low,close,volume,amount,tradestatus",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag=baostock_adjust_flag(adjust),
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock 查询失败: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = normalize_baostock_rows(rows)
        query_stock_basic = getattr(bs, "query_stock_basic", None)
        if callable(query_stock_basic):
            try:
                basic = query_stock_basic(code=baostock_code(symbol))
                if basic.error_code == "0" and basic.next():
                    basic_row = basic.get_row_data()
                    if len(basic_row) > 1:
                        df.attrs["stock_name"] = str(basic_row[1]).strip()
            except Exception:
                pass
        return df
    finally:
        bs.logout()


def fetch_baostock_stock_names(symbols: list[str]) -> dict[str, str]:
    """Resolve A-share company names in one baostock session."""
    import baostock as bs

    clean_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    if not clean_symbols:
        return {}
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        names: dict[str, str] = {}
        symbols_by_code = {baostock_code(symbol): symbol for symbol in clean_symbols}
        query_all_stock = getattr(bs, "query_all_stock", None)
        if callable(query_all_stock):
            result = query_all_stock(day=date.today().isoformat())
            if result.error_code == "0":
                while result.next():
                    row = result.get_row_data()
                    if len(row) < 3:
                        continue
                    symbol = symbols_by_code.get(str(row[0]).strip())
                    name = str(row[2]).strip()
                    if symbol and name:
                        names[symbol] = name

        # Keep compatibility with older baostock versions that do not expose
        # the all-stock snapshot, while avoiding per-symbol calls in normal use.
        if not callable(query_all_stock):
            for symbol in clean_symbols:
                result = bs.query_stock_basic(code=baostock_code(symbol))
                if result.error_code != "0" or not result.next():
                    continue
                row = result.get_row_data()
                if len(row) > 1 and str(row[1]).strip():
                    names[symbol] = str(row[1]).strip()
        return names
    finally:
        bs.logout()


def normalize_tencent_stock_names(raw: str) -> dict[str, str]:
    """Parse Tencent's compact multi-symbol quote response into code/name pairs."""
    return {
        symbol: str(quote["name"])
        for symbol, quote in normalize_tencent_stock_quotes(raw).items()
        if str(quote.get("name", "")).strip()
    }


def normalize_tencent_stock_quotes(raw: str) -> dict[str, dict[str, object]]:
    """Parse one Tencent response into normal-price snapshots by stock code."""
    quotes: dict[str, dict[str, object]] = {}
    for line in raw.splitlines():
        _, separator, quoted = line.partition('="')
        if not separator:
            continue
        fields = quoted.rstrip('";').split("~")
        if len(fields) < 3:
            continue
        name = str(fields[1]).strip()
        symbol = str(fields[2]).strip()
        if len(symbol) != 6 or not symbol.isdigit():
            continue

        def parse_price(index: int) -> float | None:
            if len(fields) <= index:
                return None
            try:
                value = float(fields[index])
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None

        quotes[symbol] = {
            "symbol": symbol,
            "name": name,
            "price": parse_price(3),
            "previous_close": parse_price(4),
            "source_name": "腾讯批量行情（不复权）",
            "price_basis": "正常市场价格",
        }
    return quotes


def fetch_tencent_stock_quotes(symbols: list[str]) -> dict[str, dict[str, object]]:
    """Resolve names and current prices for many stocks in one request."""
    clean_symbols = list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))
    if not clean_symbols:
        return {}
    quote_codes = ",".join(ashare_code(symbol) for symbol in clean_symbols)
    completed = subprocess.run(
        [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--max-time",
            "8",
            "-H",
            "Referer: https://finance.qq.com/",
            f"https://qt.gtimg.cn/q={quote_codes}",
        ],
        capture_output=True,
        timeout=12,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or "Tencent stock-quote lookup failed")
    return normalize_tencent_stock_quotes(completed.stdout.decode("gb18030", errors="replace"))


def fetch_tencent_stock_names(symbols: list[str]) -> dict[str, str]:
    """Resolve many stock names with one lightweight Tencent quote request."""
    return {
        symbol: str(quote["name"])
        for symbol, quote in fetch_tencent_stock_quotes(symbols).items()
        if str(quote.get("name", "")).strip()
    }


def fetch_stock_names(symbols: list[str]) -> dict[str, str]:
    """Resolve stock names through a fast batch source with baostock fallback."""
    try:
        names = fetch_tencent_stock_names(symbols)
        if names:
            return names
    except Exception:
        pass
    return fetch_baostock_stock_names(symbols)


def fetch_sequoia_sqlite_daily(
    symbol: str,
    start: date,
    end: date,
    db_path: str | Path = SEQUOIA_DB_PATH,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Sequoia-X local SQLite database.

    Sequoia-X maintains a local `stock_daily` table using baostock. Reading this
    database avoids fragile online Eastmoney/AkShare calls during research.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Sequoia-X database not found: {db_path}")
    query = """
        SELECT date, open, high, low, close, volume, turnover
        FROM stock_daily
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params=(symbol, start.isoformat(), end.isoformat()))
    if df.empty:
        raise ValueError(f"Sequoia-X 本地库没有 {symbol} 在 {start} 到 {end} 的数据")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def load_cloud_seed_daily(
    symbol: str,
    start: date,
    end: date,
    history_dir: str | Path = CLOUD_HISTORY_DIR,
) -> pd.DataFrame:
    """Load a static baostock qfq backup bundled for cloud deployment."""
    source = Path(history_dir) / f"{str(symbol).strip()}.csv"
    if not source.exists():
        raise FileNotFoundError(f"云端静态备份不存在：{symbol}")
    df = pd.read_csv(source, parse_dates=["date"])
    df = validate_ohlcv_frame(df, str(symbol).strip())
    sliced = df[
        (pd.to_datetime(df["date"]) >= pd.Timestamp(start))
        & (pd.to_datetime(df["date"]) <= pd.Timestamp(end))
    ].copy()
    if sliced.empty:
        raise ValueError(f"云端静态备份没有 {symbol} 在 {start} 到 {end} 的数据")
    sliced = sliced.reset_index(drop=True)
    sliced.attrs["source_name"] = "静态备份（baostock 前复权）"
    sliced.attrs["price_verified"] = False
    sliced.attrs["is_static_backup"] = True
    return sliced


def eastmoney_secid(symbol: str) -> str:
    """Return Eastmoney secid for common A-share symbols."""
    symbol = symbol.strip()
    if symbol.startswith(("6", "9")):
        return f"1.{symbol}"  # Shanghai
    return f"0.{symbol}"  # Shenzhen / Beijing fallback used by EM for common 0/3 codes


def normalize_eastmoney_klines(payload: dict) -> pd.DataFrame:
    """Normalize Eastmoney kline JSON payload returned by push2his endpoint."""
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        parts = item.split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6] if len(parts) > 6 else None,
            }
        )
    if not rows:
        raise ValueError("Eastmoney returned no kline rows")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_eastmoney_daily_with_curl(symbol: str, start: date, end: date, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch A-share daily data from Eastmoney using system curl as requests fallback.

    In this macOS environment curl can reach Eastmoney while requests/akshare may
    fail with RemoteDisconnected/ProxyError. This function intentionally uses
    curl as a robust fallback for historical daily K-line data.
    """
    fqt = {"": "0", "qfq": "1", "hfq": "2"}.get(adjust, "1")
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": fqt,
        "secid": eastmoney_secid(symbol),
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    completed = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "--max-time", "20", url],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"curl Eastmoney fallback failed: {completed.stderr.strip()}")
    return normalize_eastmoney_klines(json.loads(completed.stdout))


def load_latest_symbol_cache(symbol: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """Load the newest cached CSV for a symbol/adjust pair, if any exists."""
    if not CACHE_DIR.exists():
        return None
    candidates = sorted(CACHE_DIR.glob(f"{symbol}_*_{adjust}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return pd.read_csv(candidates[0], parse_dates=["date"])


def load_fresh_baostock_cache(
    symbol: str,
    start: date,
    end: date,
    adjust: str = "qfq",
    now: datetime | None = None,
) -> pd.DataFrame | None:
    """Load the newest cache that covers the request and reaches the last close."""
    if not CACHE_DIR.exists():
        return None
    suffix = adjust or "none"
    candidates = sorted(
        CACHE_DIR.glob(f"baostock_{symbol}_*_{suffix}.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    expected_start = pd.Timestamp(start)
    coverage_tolerance = expected_start + pd.offsets.BDay(5)
    for candidate in candidates:
        try:
            cached = pd.read_csv(candidate, parse_dates=["date"])
        except Exception:
            continue
        if cached.empty:
            continue
        earliest = pd.to_datetime(cached["date"], errors="coerce").min()
        if pd.isna(earliest) or earliest > coverage_tolerance:
            continue
        if not is_price_history_fresh(cached, end=end, now=now):
            continue
        sliced = cached[
            (pd.to_datetime(cached["date"]) >= pd.Timestamp(start))
            & (pd.to_datetime(cached["date"]) <= pd.Timestamp(end))
        ].reset_index(drop=True)
        if not sliced.empty:
            return with_data_source(sliced, "本地baostock缓存（最近收盘）")
    return None


def fetch_ashare_daily(symbol: str, start: date, end: date, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch A-share daily data using baostock first, with resilient fallbacks.

    Priority: exact-adjust cache -> baostock live -> (hfq only) Sequoia-X SQLite
    -> (unadjusted only) mpquant/Ashare -> AkShare -> curl/Eastmoney -> exact-basis
    symbol cache -> concise error.

    The cache key intentionally includes `baostock` so old Sequoia-X 后复权
    cache files are not reused for 前复权/不复权 display.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"baostock_{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{adjust or 'none'}.csv"

    if not is_mainland_market_session():
        cached = load_fresh_baostock_cache(symbol, start, end, adjust)
        if cached is not None:
            return cached

    if cache_file.exists() and not is_recent_market_request(end):
        return with_data_source(pd.read_csv(cache_file, parse_dates=["date"]), "本地baostock缓存")

    try:
        df = fetch_baostock_daily(symbol, start, end, adjust)
        df.to_csv(cache_file, index=False)
        return with_data_source(df, "baostock")
    except Exception:
        pass

    # Sequoia-X stores the project's historical series in a post-adjusted
    # price basis. Using it for qfq/none was the main cause of 10x-100x prices.
    if adjust == "hfq":
        try:
            df = fetch_sequoia_sqlite_daily(symbol, start, end, SEQUOIA_DB_PATH)
            return with_data_source(df, "Sequoia-X本地库（后复权）")
        except Exception:
            pass

    # The lightweight Sina/Tencent wrapper is unadjusted and is therefore only
    # valid as historical input when the user explicitly asks for no adjustment.
    if adjust == "":
        try:
            df = fetch_ashare_lightweight_daily(symbol, start, end)
            df.to_csv(cache_file, index=False)
            return with_data_source(df, "Ashare新浪/腾讯实时接口")
        except Exception:
            pass

    try:
        import akshare as ak

        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust,
        )
        df = normalize_akshare_hist(raw)
    except Exception:
        try:
            df = fetch_eastmoney_daily_with_curl(symbol, start, end, adjust)
        except Exception:
            cached = load_latest_symbol_cache(symbol, adjust)
            if cached is not None:
                return with_data_source(cached, "本地历史缓存")
            if adjust == "qfq":
                try:
                    return load_cloud_seed_daily(symbol, start, end)
                except Exception:
                    pass
            raise RuntimeError("在线数据源暂时不可用，且本地没有该股票缓存；请稍后重试或勾选离线演示数据。")
    df.to_csv(cache_file, index=False)
    return with_data_source(df, "AkShare/东方财富备用链路")


def make_demo_data(days: int = 260) -> pd.DataFrame:
    """Offline demo data when network/API is unavailable."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    trend = np.linspace(0, 0.35, days)
    cycle = 0.08 * np.sin(np.linspace(0, 8 * np.pi, days))
    noise = rng.normal(0, 0.012, days).cumsum()
    close = 100 * (1 + trend + cycle + noise)
    open_ = close * (1 + rng.normal(0, 0.004, days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, days))
    volume = rng.integers(50_000, 300_000, days)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
