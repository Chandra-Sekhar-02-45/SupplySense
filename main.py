"""CLI / programmatic orchestrator for SupplySense (non-Streamlit)."""
from __future__ import annotations

import pandas as pd
from agents import ForecastAgent, SafetyStockAgent, ReportAgent
from utils.data_loader import load_sales_csv
from utils.preprocess import clean_sales
from utils.config import (
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    PACK_SIZE_DEFAULT,
)


def run_pipeline(csv_path: str) -> pd.DataFrame:
    sales = load_sales_csv(csv_path)
    sales = clean_sales(sales)

    forecast_agent = ForecastAgent()
    safety_agent = SafetyStockAgent(default_lead_time_days=DEFAULT_LEAD_TIME_DAYS, service_level=DEFAULT_SERVICE_LEVEL)
    report_agent = ReportAgent(pack_size_default=PACK_SIZE_DEFAULT)

    forecast_df = forecast_agent.forecast(sales, horizon_days=DEFAULT_FORECAST_HORIZON_DAYS)
    safety_df = safety_agent.compute(sales)
    # Derive current stock (simplistic: assume last 7 days mean * lead time as proxy if none given)
    # Average of last 7 days per SKU as a proxy for current stock needs
    last7 = sales.sort_values(["sku", "date"]).groupby("sku").tail(7)
    current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")

    report = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=DEFAULT_FORECAST_HORIZON_DAYS)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SupplySense pipeline")
    parser.add_argument("csv", help="Path to sales CSV")
    parser.add_argument("--out", default="outputs/reorder_report.csv", help="Output CSV path")
    args = parser.parse_args()

    df = run_pipeline(args.csv)
    df.to_csv(args.out, index=False)
    print(f"Report written to {args.out}")
