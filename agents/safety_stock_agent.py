from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from utils.calculations import safety_stock, reorder_point


@dataclass
class SafetyStockRow:
    sku: str
    lead_time_days: int
    demand_mean: float
    demand_std: float
    z: float
    safety_stock: float
    reorder_point: float


class SafetyStockAgent:
    """Computes safety stock and reorder points per SKU using basic assumptions."""

    def __init__(self, default_lead_time_days: int = 7, service_level: float = 0.95):
        self.default_lead_time_days = default_lead_time_days
        self.service_level = service_level

    def compute(self, sales_df: pd.DataFrame, lead_times: Dict[str, int] | None = None) -> pd.DataFrame:
        """
        Args:
            sales_df: columns [date, sku, units_sold]
            lead_times: optional per-sku lead time override
        Returns:
            DataFrame columns [sku, lead_time_days, demand_mean, demand_std, z, safety_stock, reorder_point]
        """
        lead_times = lead_times or {}
        records = []
        z = 1.645 if self.service_level >= 0.95 else 1.28

        grouped = sales_df.groupby("sku")["units_sold"]
        means = grouped.mean()
        stds = grouped.std().replace({np.nan: 0.0})

        for sku in grouped.groups.keys():
            lt = int(lead_times.get(sku, self.default_lead_time_days))
            m = float(means.get(sku, 0.0))
            s = float(stds.get(sku, 0.0))
            ss = safety_stock(s, lt, z)
            rop = reorder_point(m, lt, ss)
            records.append(
                SafetyStockRow(
                    sku=sku,
                    lead_time_days=lt,
                    demand_mean=m,
                    demand_std=s,
                    z=z,
                    safety_stock=max(ss, 0.0),
                    reorder_point=max(rop, 0.0),
                ).__dict__
            )

        return pd.DataFrame.from_records(records)
