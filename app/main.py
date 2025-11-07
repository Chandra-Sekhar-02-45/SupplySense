"""Unified Streamlit app with both CSV Upload and Manual Form modes."""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is on path for package imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from agents.forecast_agent import ForecastAgent  # noqa: E402
from agents.safety_stock_agent import SafetyStockAgent  # noqa: E402
from agents.report_agent import ReportAgent  # noqa: E402
from utils.preprocess import clean_sales  # noqa: E402


st.set_page_config(page_title="SupplySense", layout="wide")
st.title("SupplySense – Inventory Intelligence")

with st.sidebar:
	st.header("Mode")
	mode = st.radio("Choose input method", ["CSV Upload", "Manual Form"], index=0)
	st.header("Settings")
	horizon = st.number_input("Forecast Horizon (days)", 1, 90, 30)
	service_level = st.slider("Service Level", 0.80, 0.99, 0.95)
	lead_time = st.number_input("Default Lead Time (days)", 1, 60, 7)

forecast_agent = ForecastAgent()
safety_agent = SafetyStockAgent(default_lead_time_days=lead_time, service_level=service_level)
report_agent = ReportAgent()

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
	df = clean_sales(df)
	forecast_df = forecast_agent.forecast(df, horizon_days=horizon)
	safety_df = safety_agent.compute(df)
	last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
	current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
	return report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon)

if mode == "CSV Upload":
	st.subheader("File Upload (CSV or Excel)")
	uploaded = st.file_uploader(
		"Upload sales file (CSV or Excel .xlsx : Rows(date(day/month/year) > sku > units_sold))",
		type=["csv", "xlsx"],
	)
	if uploaded is not None:
		# Read based on file extension
		name = uploaded.name.lower()
		if name.endswith(".csv"):
			raw = pd.read_csv(uploaded)
		elif name.endswith(".xlsx"):
			raw = pd.read_excel(uploaded, engine="openpyxl")
		else:
			st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
			st.stop()
		st.write("Preview", raw.head())
		if st.button("Process CSV"):
			try:
				report_df = run_pipeline(raw)
			except ValueError as e:
				st.error(f"Upload error: {e}")
			else:
				st.dataframe(report_df, use_container_width=True)
				st.download_button("Download Report CSV", report_df.to_csv(index=False), file_name="reorder_report.csv")
	else:
		st.info("Upload a CSV to continue.")
else:
	st.subheader("Manual Form")
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
		try:
			report_df = run_pipeline(df)
		except ValueError as e:
			st.error(f"Form data error: {e}")
		else:
			st.success("Completed")
			st.dataframe(report_df, use_container_width=True)
			st.download_button("Download Report CSV", report_df.to_csv(index=False), file_name="reorder_report.csv")
