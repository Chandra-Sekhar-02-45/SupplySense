"""Inference helper module for deployment portals.

Provides a function `predict(records, horizon_days, service_level, lead_time_days)`
that mirrors the FastAPI /api/process logic but without web server code.
"""
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd

from agents.forecast_agent import ForecastAgent
from agents.safety_stock_agent import SafetyStockAgent
from agents.report_agent import ReportAgent
from utils.preprocess import clean_sales


def predict(records: List[Dict[str, Any]], horizon_days: int = 30, service_level: float = 0.95, lead_time_days: int = 7) -> Dict[str, Any]:
    if not records:
        return {"count": 0, "rows": []}
    df = pd.DataFrame(records)
    df = clean_sales(df)
    forecast_agent = ForecastAgent()
    safety_agent = SafetyStockAgent(default_lead_time_days=lead_time_days, service_level=service_level)
    report_agent = ReportAgent()

    forecast_df = forecast_agent.forecast(df, horizon_days=horizon_days)
    safety_df = safety_agent.compute(df)
    last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
    current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
    report_df = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon_days)

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
    return {"count": len(rows), "rows": rows}

__all__ = ["predict"]
