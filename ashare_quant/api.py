from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ashare_quant.backtest import extract_trades, run_ma_cross_backtest, summarize_backtest
from ashare_quant.data import fetch_ashare_daily, fetch_ashare_lightweight_bars

app = FastAPI(
    title="A股量化交易研究台 API",
    description="统一封装行情数据、策略回测、股票池扫描和交易计划能力。仅用于研究学习，不构成投资建议。",
    version="0.1.0",
)


class DailyBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class DailyPriceResponse(BaseModel):
    symbol: str
    start: date
    end: date
    adjust: str
    rows: int
    data: List[DailyBar]


class IntradayResponse(BaseModel):
    symbol: str
    frequency: str
    rows: int
    data: List[DailyBar]


class MaCrossBacktestRequest(BaseModel):
    symbol: str = Field(..., examples=["600522"])
    start: date
    end: date
    adjust: Literal["qfq", "hfq", ""] = "qfq"
    short_window: int = Field(5, ge=2, le=250)
    long_window: int = Field(20, ge=3, le=500)
    initial_cash: float = Field(100000.0, gt=0)
    fee_rate: float = Field(0.0003, ge=0, le=0.1)
    slippage_rate: float = Field(0.0005, ge=0, le=0.1)
    stop_loss_pct: Optional[float] = Field(None, gt=0, le=100)
    take_profit_pct: Optional[float] = Field(None, gt=0, le=1000)


class BacktestResponse(BaseModel):
    symbol: str
    strategy: str
    adjust: str
    summary: Dict[str, Any]
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]


def _df_to_daily_bars(df: pd.DataFrame) -> List[DailyBar]:
    records: List[DailyBar] = []
    for row in df[["date", "open", "high", "low", "close", "volume"]].to_dict("records"):
        records.append(
            DailyBar(
                date=pd.Timestamp(row["date"]).date().isoformat(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return records


def _df_to_timestamp_bars(df: pd.DataFrame) -> List[DailyBar]:
    records: List[DailyBar] = []
    for row in df[["date", "open", "high", "low", "close", "volume"]].to_dict("records"):
        records.append(
            DailyBar(
                date=pd.Timestamp(row["date"]).isoformat(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return records


def _json_safe_records(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in df[columns].to_dict("records"):
        cleaned: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                cleaned[key] = value.date().isoformat()
            elif pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, float):
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ashare-quant-api", "version": app.version}


@app.get("/api/stocks/{symbol}/daily", response_model=DailyPriceResponse)
def get_daily_prices(
    symbol: str,
    start: date = Query(..., description="开始日期，格式 YYYY-MM-DD"),
    end: date = Query(..., description="结束日期，格式 YYYY-MM-DD"),
    adjust: Literal["qfq", "hfq", ""] = Query("qfq", description="复权方式：qfq=前复权，hfq=后复权，空字符串=不复权"),
) -> DailyPriceResponse:
    if start > end:
        raise HTTPException(status_code=400, detail="start must be <= end")
    try:
        df = fetch_ashare_daily(symbol, start, end, adjust)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DailyPriceResponse(symbol=symbol, start=start, end=end, adjust=adjust, rows=len(df), data=_df_to_daily_bars(df))


@app.get("/api/stocks/{symbol}/intraday", response_model=IntradayResponse)
def get_intraday_prices(
    symbol: str,
    frequency: Literal["1m", "5m", "15m", "30m", "60m"] = Query("15m", description="分钟线周期"),
    count: int = Query(120, ge=1, le=800, description="返回最近 N 根 K 线"),
) -> IntradayResponse:
    try:
        df = fetch_ashare_lightweight_bars(symbol, frequency=frequency, count=count)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return IntradayResponse(symbol=symbol, frequency=frequency, rows=len(df), data=_df_to_timestamp_bars(df))


@app.post("/api/backtests/ma-cross", response_model=BacktestResponse)
def run_ma_cross_backtest_api(request: MaCrossBacktestRequest) -> BacktestResponse:
    if request.short_window >= request.long_window:
        raise HTTPException(status_code=400, detail="short_window must be < long_window")
    try:
        prices = fetch_ashare_daily(request.symbol, request.start, request.end, request.adjust)
        result = run_ma_cross_backtest(
            prices,
            short_window=request.short_window,
            long_window=request.long_window,
            initial_cash=request.initial_cash,
            fee_rate=request.fee_rate,
            slippage_rate=request.slippage_rate,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    summary = summarize_backtest(result, request.initial_cash)
    if "total_return_pct" in summary and "total_return" not in summary:
        summary["total_return"] = summary["total_return_pct"]
    trades = extract_trades(result)
    equity_curve = _json_safe_records(result, ["date", "equity", "benchmark_equity", "position", "signal"])
    return BacktestResponse(
        symbol=request.symbol,
        strategy="ma_cross",
        adjust=request.adjust,
        summary=summary,
        trades=_json_safe_records(trades, list(trades.columns)) if not trades.empty else [],
        equity_curve=equity_curve,
    )
