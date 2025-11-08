"""Unified Streamlit app with both CSV Upload and Manual Form modes.

Redesigned UI with tabs, custom styling, KPIs, and charts.
"""

import sys
import os
from pathlib import Path
# Ensure project root is on path for package imports so imports like
# `from utils.config import ...` work when running the app from the
# `app/` folder (Streamlit or `python app/main.py`). We insert the
# project root at the front of sys.path before any local imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import requests

from utils.config import GEMINI_API_KEY  # noqa: E402

from agents.forecast_agent import ForecastAgent  # noqa: E402
from agents.safety_stock_agent import SafetyStockAgent  # noqa: E402
from agents.report_agent import ReportAgent  # noqa: E402
from agents.assistant_agent import AssistantAgent  # noqa: E402
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

	# Assistant & backend controls in a compact expander
	with st.expander("Assistant (optional)", expanded=False):
		have_key = bool(GEMINI_API_KEY)
		# Model selection only when key is present
		if have_key:
			default_model = os.getenv("GEMINI_MODEL", "gemini-pro")
			gemini_model = st.selectbox(
				"Gemini model",
				options=["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
				index=["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"].index(default_model) if default_model in ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"] else 0,
				help="Model used for Gemini generateContent API calls."
			)
			st.session_state["gemini_model"] = gemini_model
		else:
			st.caption("Set GEMINI_API_KEY in .env to enable LLM answers. The assistant will still provide rule-based tips.")

		# Backend route toggle
		use_backend = st.checkbox("Use backend", value=False, help="Send assistant questions to FastAPI /api/assistant")
		st.session_state["use_backend_assistant"] = use_backend

		# Backend health (lightweight and compact)
		backend_base = os.getenv("SUPPLYSENSE_API_URL", "http://localhost:8000").rstrip("/")
		health_url = backend_base + "/api/health"
		# Passive check first time
		alive = st.session_state.get("backend_alive")
		if alive is None:
			try:
				resp = requests.get(health_url, timeout=2)
				alive = (resp.status_code == 200 and resp.json().get("status") == "ok")
			except Exception:
				alive = False
			st.session_state["backend_alive"] = alive

		cols = st.columns([1,1])
		status_text = "Online" if st.session_state.get("backend_alive", False) else "Offline"
		status_color = "#16a34a" if st.session_state.get("backend_alive", False) else "#ef4444"
		cols[0].markdown(f"Backend: <span style='color:{status_color};font-weight:600'>{status_text}</span>", unsafe_allow_html=True)
		if cols[1].button("Refresh"):
			try:
				resp = requests.get(health_url, timeout=3)
				alive = (resp.status_code == 200 and resp.json().get("status") == "ok")
			except Exception:
				alive = False
			st.session_state["backend_alive"] = alive

		# Advanced controls hidden by default to reduce clutter
		with st.expander("Advanced LLM controls", expanded=False):
			gemini_url = st.text_input(
				"Override endpoint",
				value="",
				help="Full URL to Gemini endpoint. Leave empty to use the default API URL."
			)
			if gemini_url:
				st.session_state["gemini_url"] = gemini_url
			if st.button("Test endpoint"):
				model = st.session_state.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-pro")
				url = st.session_state.get("gemini_url") or os.getenv("GEMINI_API_URL") or f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
				st.info(f"Probing endpoint: {url}")
				prompt = "Provide one short suggestion to increase retail sales."
				payload = {
					"contents": [{"role": "user", "parts": [{"text": prompt}]}],
					"generationConfig": {"temperature": 0.2, "maxOutputTokens": 128}
				}
				params = None
				headers = {"Content-Type": "application/json"}
				if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
					params = {"key": GEMINI_API_KEY}
				elif GEMINI_API_KEY:
					headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
				try:
					resp = requests.post(url, json=payload, headers=headers, params=params, timeout=15)
					st.markdown(f"**Status:** {resp.status_code}")
					try:
						st.json(resp.json())
					except Exception:
						st.code(resp.text)
				except Exception as e:
					st.error(f"Request error: {e}")

forecast_agent = ForecastAgent()
safety_agent = SafetyStockAgent(default_lead_time_days=lead_time, service_level=service_level)
report_agent = ReportAgent()
assistant = AssistantAgent()

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
				st.dataframe(raw.head(20), width='stretch')
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
		st.dataframe(report_df, width='stretch')
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
				st.plotly_chart(visualizer.plot_sales_history(s_hist), width='stretch')
			with ch2:
				st.plotly_chart(visualizer.plot_forecast(s_fc), width='stretch')

		# Assistant suggestions and Q&A
		with st.expander("Assistant — actionable suggestions & Q&A", expanded=False):
			# Avoid using `or` with DataFrame (ambiguous truth value). Use an explicit check.
			sales_ctx = sales_df if isinstance(sales_df, pd.DataFrame) else pd.DataFrame()

			# Render suggestions only once: if backend is on and online, wait for backend response; otherwise show local suggestions immediately
			try:
				use_backend = st.session_state.get("use_backend_assistant", False)
				backend_alive = st.session_state.get("backend_alive", False)
				if not (use_backend and backend_alive):
					suggestions = assistant.suggest_actions(report_df, sales_ctx)
					st.markdown("**Suggestions to increase sales**")
					for s in suggestions:
						st.write("- ", s)
			except Exception as e:
				st.error(f"Assistant error: {e}")

			# simple question box
			q = st.text_input("Ask the assistant a question (e.g., 'promotion ideas')", key="assistant_q")
			if st.button("Ask Assistant", key="assistant_ask"):
				api_url_override = st.session_state.get("gemini_url")
				use_backend = st.session_state.get("use_backend_assistant")
				backend_alive = st.session_state.get("backend_alive", False)
				if use_backend:
					if not backend_alive:
						answer = "Backend is offline; using local assistant.\n" + assistant.answer_question(
							q,
							report_df,
							sales_ctx,
							api_url_override=api_url_override,
							model_name=st.session_state.get("gemini_model")
						)
						st.info(answer)
						st.stop()
					try:
						records = sales_ctx[['date','sku','units_sold']].copy()
						records['date'] = records['date'].astype(str)
						payload = {
							"records": records.to_dict(orient="records"),
							"horizon_days": int(horizon),
							"service_level": float(service_level),
							"lead_time_days": int(lead_time),
							"question": q,
						}
						base = os.getenv("SUPPLYSENSE_API_URL", "http://localhost:8000")
						url = base.rstrip("/") + "/api/assistant"
						resp = requests.post(url, json=payload, timeout=20)
						if resp.status_code == 200:
							data = resp.json()
							answer = data.get("answer") or "(No answer returned)"
							# If we did not render local suggestions earlier, show backend's suggestions now
							bk_suggestions = data.get("suggestions", [])
							if bk_suggestions and (use_backend and backend_alive):
								st.markdown("**Suggestions to increase sales**")
								for s in bk_suggestions:
									st.write("- ", s)
						else:
							answer = f"Backend error {resp.status_code}: {resp.text[:300]}"
					except Exception as e:
						answer = f"Backend request failed: {e}. Falling back to local assistant."\
						+ "\n" + assistant.answer_question(
							q,
							report_df,
							sales_ctx,
							api_url_override=api_url_override,
							model_name=st.session_state.get("gemini_model")
						)
				else:
					answer = assistant.answer_question(
						q,
						report_df,
						sales_ctx,
						api_url_override=api_url_override,
						model_name=st.session_state.get("gemini_model")
					)
				st.info(answer)

			# Show diagnostics if available
			try:
				from pathlib import Path
				df = Path("logs/assistant_diag.log")
				if df.exists():
					st.markdown("**Assistant diagnostics (last failure)**")
					st.code(df.read_text(), language="json")
			except Exception:
				pass

with tabs[2]:
	st.markdown(
		"""
		**About**

		SupplySense helps you forecast demand, compute safety stock, and
		generate reorder recommendations. Use the sidebar to configure the
		forecast horizon, service level, and default lead time.
		"""
	)
