"""Unified Streamlit app with both CSV Upload and Manual Form modes.

Redesigned UI with tabs, custom styling, KPIs, and charts.
"""

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
from utils import visualizer  # noqa: E402


st.set_page_config(page_title="SupplySense", page_icon="📦", layout="wide")

# ---- Custom CSS ----
def load_css():
	css_path = ROOT / "templates" / "style.css"
	try:
		with open(css_path, "r", encoding="utf-8") as f:
			st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
	except FileNotFoundError:
		pass

load_css()

st.markdown(
		"""
		<div class="ss-header">
			<div class="ss-title">SupplySense – Inventory Intelligence</div>
			<div class="ss-subtitle">Forecast · Safety Stock · Reorder Recommendations</div>
		</div>
		""",
		unsafe_allow_html=True,
)

with st.sidebar:
	st.markdown("### Mode")
	mode = st.radio("Choose input method", ["CSV Upload", "Manual Form"], index=0)
	st.markdown("### Settings")
	horizon = st.number_input("Forecast Horizon (days)", 1, 90, 30)
	service_level = st.slider("Service Level", 0.80, 0.99, 0.95)
	lead_time = st.number_input("Default Lead Time (days)", 1, 60, 7)

forecast_agent = ForecastAgent()
safety_agent = SafetyStockAgent(default_lead_time_days=lead_time, service_level=service_level)
report_agent = ReportAgent()

def run_pipeline(df: pd.DataFrame):
	df = clean_sales(df)
	forecast_df = forecast_agent.forecast(df, horizon_days=horizon)
	safety_df = safety_agent.compute(df)
	last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
	current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
	report_df = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon)
	return report_df, {"sales": df, "forecast": forecast_df, "safety": safety_df}
tabs = st.tabs(["Input", "Results", "About"])  # UI sections

with tabs[0]:
	st.markdown("<div class='ss-section-title'>Provide Data</div>", unsafe_allow_html=True)
	if mode == "CSV Upload":
		with st.container():
			uploaded = st.file_uploader(
				"Upload .csv or .xlsx (must include sku + a date/quantity column)",
				type=["csv", "xlsx"],
			)
		if uploaded is not None:
			name = uploaded.name.lower()
			if name.endswith(".csv"):
				raw = pd.read_csv(uploaded)
			elif name.endswith(".xlsx"):
				raw = pd.read_excel(uploaded, engine="openpyxl")
			else:
				st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
				st.stop()
			with st.container():
				st.dataframe(raw.head(20), use_container_width=True)
			# Action bar
			st.markdown("<div class='ss-action-bar'>", unsafe_allow_html=True)
			col_a, col_b = st.columns([1,1])
			file_ok = col_a.button("Process File", type="primary")
			clear_file = col_b.button("Clear Upload", help="Remove current file from memory")
			st.markdown("</div>", unsafe_allow_html=True)
			if clear_file:
				st.session_state.pop("report_df", None)
				st.session_state.pop("ctx", None)
				st.experimental_rerun()
			if file_ok:
				try:
					report_df, ctx = run_pipeline(raw)
				except ValueError as e:
					st.error(f"Upload error: {e}")
				else:
					st.session_state["report_df"] = report_df
					st.session_state["ctx"] = ctx
					st.success("Processed successfully. Switch to Results tab.")
		else:
			st.info("Upload a file to continue.")
	else:
		with st.container():
			rows = st.number_input("Number of rows", 1, 50, 5)
			data_entries = []
			for i in range(rows):
				cols = st.columns(3)
				sku = cols[0].text_input(f"SKU {i+1}", value=f"SKU{i+1}")
				date = cols[1].date_input(f"Date {i+1}")
				units = cols[2].number_input(f"Units Sold {i+1}", 0, 10000, 10)
				data_entries.append({"sku": sku, "date": date, "units_sold": units})
		st.markdown("<div class='ss-action-bar'>", unsafe_allow_html=True)
		go_form, clear_form = st.columns([1,1])
		run_ok = go_form.button("Run Forecast", type="primary")
		wipe_form = clear_form.button("Clear Form", help="Reset manual entries")
		st.markdown("</div>", unsafe_allow_html=True)
		if wipe_form:
			st.experimental_rerun()
		if run_ok:
			df = pd.DataFrame(data_entries)
			df["date"] = pd.to_datetime(df["date"])  # ensure dtype
			try:
				report_df, ctx = run_pipeline(df)
			except ValueError as e:
				st.error(f"Form data error: {e}")
			else:
				st.session_state["report_df"] = report_df
				st.session_state["ctx"] = ctx
				st.success("Completed. Switch to Results tab.")

with tabs[1]:
	st.markdown("#### Results & Insights")
	report_df = st.session_state.get("report_df")
	ctx = st.session_state.get("ctx", {})
	if report_df is None:
		st.info("No results yet. Use the Input tab first.")
	else:
		st.markdown("<div class='ss-section-title'>Key Metrics</div>", unsafe_allow_html=True)
		with st.container():
			c1, c2, c3, c4 = st.columns(4)
			stockouts = int((report_df["current_stock"] < report_df["reorder_point"]).sum())
			total_qty = int(report_df["recommended_order_qty"].sum())
			total_skus = int(report_df["sku"].nunique())
			total_forecast = int(report_df["forecast_30d"].sum())
			c1.metric("SKUs", f"{total_skus}")
			c2.metric("Potential Stockouts", f"{stockouts}")
			c3.metric("Total Forecast (30d)", f"{total_forecast}")
			c4.metric("Total Recommended Qty", f"{total_qty}")

		st.markdown("<div class='ss-section-title'>Reorder Recommendations</div>", unsafe_allow_html=True)
		st.dataframe(report_df, use_container_width=True)
		# Action bar for results
		st.markdown("<div class='ss-action-bar'>", unsafe_allow_html=True)
		d_col, c_col = st.columns([1,1])
		dl = d_col.download_button(
			"Download Report CSV",
			report_df.to_csv(index=False),
			file_name="reorder_report.csv",
		)
		clr = c_col.button("Clear Results", help="Remove current results from session")
		st.markdown("</div>", unsafe_allow_html=True)
		if clr:
			st.session_state.pop("report_df", None)
			st.session_state.pop("ctx", None)
			st.experimental_rerun()

		# Charts
		sales_df = ctx.get("sales")
		forecast_df = ctx.get("forecast")
		if isinstance(sales_df, pd.DataFrame) and isinstance(forecast_df, pd.DataFrame):
			sku_list = sorted(list(set(sales_df["sku"].unique())))
			st.markdown("<div class='ss-section-title'>Visualizations</div>", unsafe_allow_html=True)
			chosen = st.selectbox("Visualize SKU", options=sku_list)
			s_hist = sales_df[sales_df["sku"] == chosen]
			s_fc = forecast_df[forecast_df["sku"] == chosen]
			ch1, ch2 = st.columns(2)
			with ch1:
				st.plotly_chart(visualizer.plot_sales_history(s_hist), use_container_width=True)
			with ch2:
				st.plotly_chart(visualizer.plot_forecast(s_fc), use_container_width=True)

with tabs[2]:
	st.markdown(
		"""
		**About**

		SupplySense helps you forecast demand, compute safety stock, and
		generate reorder recommendations. Use the sidebar to configure the
		forecast horizon, service level, and default lead time.
		"""
	)
