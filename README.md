# SupplySense 🛍️📦

AI-driven inventory forecasting & reorder recommendations for small/medium retail.

## Features
- Dual input: CSV upload or manual form entry
- Demand forecasting (currently SMA stub; extensible to ARIMA/Prophet/Croston)
- Safety stock & reorder point calculation
- Consolidated SKU report with recommended order quantities
- Streamlit UI for interaction & CSV export

## Quick Start

### 1. Clone & install
```bash
python3 -m pip install -r requirements.txt
```

### 2. Configure environment (optional)
Copy `.env.example` to `.env` and adjust values:
```bash
cp .env.example .env
```
Available keys:
```
DEFAULT_SERVICE_LEVEL=0.95
FORECAST_HORIZON_DAYS=30
DEFAULT_LEAD_TIME_DAYS=7
PACK_SIZE_DEFAULT=1
GEMINI_API_KEY=your_key_here  # optional, not required yet
```

### 3. Run CLI pipeline
```bash
python3 main.py data/sample_sales.csv --out outputs/reorder_report.csv
```
Outputs written to `outputs/reorder_report.csv`.

### 4. Run Streamlit apps
CSV Upload mode:
```bash
streamlit run app/app_upload.py
```
Form Input mode:
```bash
streamlit run app/app_form.py
```
Both Upload & Input:
```bash
streamlit run app/main.py
```

## FastAPI backend (optional)

Install requirements if not already done, then run the server:

```bash
uvicorn api.server:app --reload
```

Open docs at http://127.0.0.1:8000/docs

Available endpoints (prefix /api):
- GET /api/health – health check
- POST /api/process – JSON body with records to process
- POST /api/upload – multipart file upload (.csv or .xlsx)

Example: upload CSV
```bash
curl -F "file=@data/sample_sales.csv" \
		 -F "horizon_days=30" -F "service_level=0.95" -F "lead_time_days=7" \
		 http://127.0.0.1:8000/api/upload
```

Example: JSON payload
```bash
curl -X POST http://127.0.0.1:8000/api/process \
	-H 'Content-Type: application/json' \
	-d '{
		"records": [
			{"date":"2025-08-01","sku":"SKU001","units_sold":10},
			{"date":"2025-08-02","sku":"SKU001","units_sold":12}
		],
		"horizon_days": 30, "service_level": 0.95, "lead_time_days": 7
	}'
```

## Project Structure
```
agents/      # ForecastAgent, SafetyStockAgent, ReportAgent
utils/       # Data loading, cleaning, calculations, config, visualization
templates/   # (Optional) HTML/CSS stubs
outputs/     # Generated reports & charts
app/         # Streamlit entry points
tests/       # Pytest unit tests
data/        # Sample input CSVs & form JSON
```

## Extending Forecasts
Replace SMA with ARIMA/Prophet by enhancing `ForecastAgent` while keeping public `forecast(...)` method consistent.

## Testing
```bash
python3 -m pytest -q
```

## Notes
- Prophet install can be heavier; current stub avoids heavy dependencies at runtime.
- Environment variables automatically loaded via `python-dotenv` in `utils/config.py`.

## Roadmap (Excerpt)
- Model auto-selection (ARIMA vs Prophet vs Croston)
- Performance: parallel per-SKU fitting, caching
- Rich explainability (variance drivers, confidence scores)
- Persistent historical runs (DuckDB / Parquet)

## License
See `LICENSE`.
