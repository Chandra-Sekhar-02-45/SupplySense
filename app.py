"""Minimal app entrypoint for agent deployment portals.

Provides a single function run_agent() that:
1. Loads a small sample of sales data (or expects data/sample_sales.csv).
2. Executes the pipeline (forecast -> safety stock -> report).
3. Prints JSON to stdout for portal verification.

This allows environments that look specifically for app.py to invoke
`python app.py` and get a deterministic output.
"""

from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

from agents.forecast_agent import ForecastAgent
from agents.safety_stock_agent import SafetyStockAgent
from agents.report_agent import ReportAgent
from utils.preprocess import clean_sales


def run_agent(input_path: str = "data/sample_sales.csv", horizon_days: int = 30, service_level: float = 0.95, lead_time_days: int = 7):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(p)
    df = clean_sales(df)

    forecast_agent = ForecastAgent()
    safety_agent = SafetyStockAgent(default_lead_time_days=lead_time_days, service_level=service_level)
    report_agent = ReportAgent()

    forecast_df = forecast_agent.forecast(df, horizon_days=horizon_days)
    safety_df = safety_agent.compute(df)
    last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
    current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
    report_df = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon_days)

    # Shape into output JSON similar to ProcessResponse
    rows = []
    for _, r in report_df.iterrows():
        rows.append({
            "sku": r["sku"],
            "current_stock": float(r["current_stock"]),
            "reorder_point": float(r["reorder_point"]),
            "safety_stock": float(r["safety_stock"]),
            "forecast_30d": float(r["forecast_30d"]),
            "recommended_order_qty": int(r["recommended_order_qty"]),
        })
    out = {"count": len(rows), "rows": rows}
    print(json.dumps(out))


if __name__ == "__main__":
    run_agent()
