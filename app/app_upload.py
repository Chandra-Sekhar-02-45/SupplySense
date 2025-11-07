"""Streamlit app (CSV Upload Mode)."""
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

st.title("SupplySense - CSV Upload")

uploaded = st.file_uploader("Upload sales CSV", type=["csv"])  # expecting date,sku,units_sold
if uploaded:
	raw = pd.read_csv(uploaded)
	st.write("Preview", raw.head())
	horizon = st.sidebar.number_input("Horizon (days)", 1, 90, 30)
	service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95)
	lead_time = st.sidebar.number_input("Default Lead Time (days)", 1, 60, 7)
	if st.button("Process"):
		df = clean_sales(raw)
		forecast_agent = ForecastAgent()
		safety_agent = SafetyStockAgent(default_lead_time_days=lead_time, service_level=service_level)
		report_agent = ReportAgent()
		forecast_df = forecast_agent.forecast(df, horizon_days=horizon)
		safety_df = safety_agent.compute(df)
		last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
		current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
		report_df = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon)
		st.dataframe(report_df)
		st.download_button("Download Report CSV", report_df.to_csv(index=False), file_name="reorder_report.csv")
else:
	st.info("Awaiting file upload.")
