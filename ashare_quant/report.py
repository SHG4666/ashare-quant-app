from __future__ import annotations

from html import escape
from typing import Mapping

import pandas as pd


_SUMMARY_LABELS = {
    "final_equity": "最终权益",
    "total_return_pct": "策略总收益(%)",
    "annual_return_pct": "年化收益(%)",
    "benchmark_return_pct": "买入持有收益(%)",
    "excess_return_pct": "超额收益(%)",
    "max_drawdown_pct": "最大回撤(%)",
    "sharpe": "夏普比率",
    "trade_count": "交易次数",
    "total_trade_cost": "交易成本",
}


def _format_metric(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.2f}"
    return escape(str(value))


def _table_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    display = df.copy()
    if max_rows is not None:
        display = display.head(max_rows)
    return display.to_html(index=False, escape=True, border=0, classes="data-table")


def build_backtest_html_report(
    title: str,
    summary: Mapping[str, object],
    result: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    max_result_rows: int = 500,
) -> str:
    """Build a self-contained HTML report for a backtest result.

    The report intentionally uses plain HTML/CSS so it can be downloaded from
    Streamlit without extra rendering dependencies.
    """
    if result.empty:
        raise ValueError("result dataframe is empty")
    if max_result_rows <= 0:
        raise ValueError("max_result_rows must be positive")

    safe_title = escape(title)
    summary_rows = []
    for key, value in summary.items():
        label = _SUMMARY_LABELS.get(key, key)
        summary_rows.append(
            f"<tr><th>{escape(str(label))}</th><td>{_format_metric(value)}</td></tr>"
        )
    trades_html = "<p>当前参数没有产生买卖交易。</p>"
    if trades is not None and not trades.empty:
        trades_html = _table_html(trades)

    result_note = ""
    if len(result) > max_result_rows:
        result_note = f"<p class=\"note\">回测明细仅展示前 {max_result_rows} 行，共 {len(result)} 行。</p>"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #222; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #666; margin-top: 0; }}
    section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .summary {{ max-width: 760px; }}
    .summary th {{ width: 220px; }}
    .note {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <p class="subtitle">由 A股量化交易研究台生成，仅用于研究学习，不构成投资建议。</p>

  <section>
    <h2>核心指标</h2>
    <table class="summary"><tbody>{''.join(summary_rows)}</tbody></table>
  </section>

  <section>
    <h2>交易记录</h2>
    {trades_html}
  </section>

  <section>
    <h2>回测明细</h2>
    {result_note}
    {_table_html(result, max_rows=max_result_rows)}
  </section>
</body>
</html>
"""
