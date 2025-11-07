"""Streamlit app (Form Mode) for manual SKU data input."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is on sys.path when running from subfolder or IDE
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from agents.forecast_agent import ForecastAgent  # noqa: E402
from agents.safety_stock_agent import SafetyStockAgent  # noqa: E402
from agents.report_agent import ReportAgent  # noqa: E402
from utils.preprocess import clean_sales  # noqa: E402

st.title("SupplySense - Form Input")

st.sidebar.header("Settings")
horizon = st.sidebar.number_input("Forecast Horizon (days)", 1, 90, 30)
service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95)
lead_time = st.sidebar.number_input("Default Lead Time (days)", 1, 60, 7)

st.subheader("Enter Data Rows")
rows = st.number_input("Number of rows", 1, 50, 5)
data_entries = []
for i in range(rows):
	cols = st.columns(4)
	sku = cols[0].text_input(f"SKU {i+1}", value=f"SKU{i+1}")
	date = cols[1].date_input(f"Date {i+1}")
	units = cols[2].number_input(f"Units Sold {i+1}", 0, 10000, 10)
	data_entries.append({"sku": sku, "date": date, "units_sold": units})

if st.button("Run Forecast"):
	df = pd.DataFrame(data_entries)
	df["date"] = pd.to_datetime(df["date"])  # ensure dtype
	df = clean_sales(df)
	forecast_agent = ForecastAgent()
	safety_agent = SafetyStockAgent(default_lead_time_days=lead_time, service_level=service_level)
	report_agent = ReportAgent()
	forecast_df = forecast_agent.forecast(df, horizon_days=horizon)
	safety_df = safety_agent.compute(df)
	last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
	current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
	report_df = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon)
	st.success("Completed")
	st.dataframe(report_df)
	st.download_button("Download Report CSV", report_df.to_csv(index=False), file_name="reorder_report.csv")
