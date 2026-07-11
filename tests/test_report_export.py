import pandas as pd

from ashare_quant.report import build_backtest_html_report


def test_build_backtest_html_report_contains_summary_trades_and_result_table():
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [10.0, 11.0],
            "equity": [10000.0, 11000.0],
            "benchmark_equity": [10000.0, 10800.0],
        }
    )
    summary = {
        "final_equity": 11000.0,
        "total_return_pct": 10.0,
        "benchmark_return_pct": 8.0,
        "excess_return_pct": 2.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 1.23,
        "trade_count": 2,
    }
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "action": ["BUY", "SELL"],
            "price": [10.0, 11.0],
            "equity": [10000.0, 11000.0],
        }
    )

    html = build_backtest_html_report(
        title="000001 双均线交叉回测报告",
        summary=summary,
        result=result,
        trades=trades,
    )

    assert html.startswith("<!doctype html>")
    assert "000001 双均线交叉回测报告" in html
    assert "核心指标" in html
    assert "最终权益" in html
    assert "11,000.00" in html
    assert "交易记录" in html
    assert "BUY" in html
    assert "回测明细" in html
    assert "benchmark_equity" in html
