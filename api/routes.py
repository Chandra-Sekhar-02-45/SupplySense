from __future__ import annotations

import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from .schemas import ProcessRequest, ProcessResponse, ReportRow
from .schemas import ProcessRequest, ProcessResponse, ReportRow, AssistantRequest, AssistantResponse
from .deps import get_forecast_agent, get_safety_agent, get_report_agent, get_assistant_agent
from agents.forecast_agent import ForecastAgent
from agents.safety_stock_agent import SafetyStockAgent
from agents.report_agent import ReportAgent
from agents.assistant_agent import AssistantAgent
from utils.preprocess import clean_sales

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/process", response_model=ProcessResponse)
def process_json(
    payload: ProcessRequest,
    forecast_agent: ForecastAgent = Depends(get_forecast_agent),
    safety_agent: SafetyStockAgent = Depends(get_safety_agent),
    report_agent: ReportAgent = Depends(get_report_agent),
):
    df = pd.DataFrame([r.dict() for r in payload.records])
    try:
        df = clean_sales(df)
        forecast_df = forecast_agent.forecast(df, horizon_days=payload.horizon_days)
        safety_df = safety_agent.compute(df)
        last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
        current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
        report = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=payload.horizon_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = [ReportRow(**row._asdict()) if hasattr(row, "_asdict") else ReportRow(**row) for row in report.to_dict(orient="records")]
    return ProcessResponse(count=len(rows), rows=rows)


@router.post("/assistant", response_model=AssistantResponse)
def assistant_endpoint(
    payload: AssistantRequest,
    forecast_agent: ForecastAgent = Depends(get_forecast_agent),
    safety_agent: SafetyStockAgent = Depends(get_safety_agent),
    report_agent: ReportAgent = Depends(get_report_agent),
    assistant: AssistantAgent = Depends(get_assistant_agent),
):
    """Return assistant suggestions and optionally answer a single free-text question.

    The request body is the same as `/process` with an optional `question` field.
    """
    df = pd.DataFrame([r.dict() for r in payload.records])
    try:
        df = clean_sales(df)
        forecast_df = forecast_agent.forecast(df, horizon_days=payload.horizon_days)
        safety_df = safety_agent.compute(df)
        last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
        current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
        report = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=payload.horizon_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suggestions = assistant.suggest_actions(report, df)
    answer = None
    if payload.question:
        answer = assistant.answer_question(payload.question, report, df)

    return AssistantResponse(suggestions=suggestions, answer=answer)


@router.post("/upload", response_model=ProcessResponse)
async def upload_file(
    file: UploadFile = File(...),
    horizon_days: int = 30,
    service_level: float = 0.95,
    lead_time_days: int = 7,
    forecast_agent: ForecastAgent = Depends(get_forecast_agent),
    safety_agent: SafetyStockAgent = Depends(get_safety_agent),
    report_agent: ReportAgent = Depends(get_report_agent),
):
    if service_level < 0.5 or service_level > 0.999:
        raise HTTPException(status_code=400, detail="service_level out of range")

    content = await file.read()
    name = file.filename.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type; use .csv or .xlsx")
        df = clean_sales(df)
        forecast_df = forecast_agent.forecast(df, horizon_days=horizon_days)
        safety_agent.service_level = service_level  # adjust runtime
        safety_df = safety_agent.compute(df)
        last7 = df.sort_values(["sku", "date"]).groupby("sku").tail(7)
        current_stock = last7.groupby("sku")["units_sold"].mean().reset_index(name="current_stock")
        report = report_agent.build(current_stock, forecast_df, safety_df, horizon_days=horizon_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = [ReportRow(**row) for row in report.to_dict(orient="records")]
    return ProcessResponse(count=len(rows), rows=rows)
