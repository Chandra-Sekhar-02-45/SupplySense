from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class SalesRecord(BaseModel):
    date: str
    sku: str
    units_sold: float = Field(ge=0)


class ProcessRequest(BaseModel):
    records: List[SalesRecord]
    horizon_days: int = 30
    service_level: float = 0.95
    lead_time_days: int = 7

    @validator("service_level")
    def _sl_range(cls, v):  # type: ignore
        if not (0.5 <= v <= 0.999):
            raise ValueError("service_level must be between 0.5 and 0.999")
        return v


class ReportRow(BaseModel):
    sku: str
    current_stock: float
    reorder_point: float
    safety_stock: float
    forecast_30d: float
    recommended_order_qty: int


class ProcessResponse(BaseModel):
    count: int
    rows: List[ReportRow]


class AssistantRequest(ProcessRequest):
    question: Optional[str] = None


class AssistantResponse(BaseModel):
    suggestions: List[str]
    answer: Optional[str] = None
