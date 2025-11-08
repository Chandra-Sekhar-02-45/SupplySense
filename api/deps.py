from __future__ import annotations

from fastapi import Depends
from agents.forecast_agent import ForecastAgent
from agents.safety_stock_agent import SafetyStockAgent
from agents.report_agent import ReportAgent
from utils.config import (
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    PACK_SIZE_DEFAULT,
)
from agents.assistant_agent import AssistantAgent


def get_forecast_agent():
    return ForecastAgent()


def get_safety_agent():
    return SafetyStockAgent(default_lead_time_days=DEFAULT_LEAD_TIME_DAYS, service_level=DEFAULT_SERVICE_LEVEL)


def get_report_agent():
    return ReportAgent(pack_size_default=PACK_SIZE_DEFAULT)


def get_assistant_agent():
    return AssistantAgent()
