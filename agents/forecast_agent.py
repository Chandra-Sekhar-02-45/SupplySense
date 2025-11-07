from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class ForecastResult:
    sku: str
    date: pd.Timestamp
    forecast_units: float
    model_used: str


class ForecastAgent:
    """
    Minimal forecasting agent.
    For MVP: per-SKU simple moving average (SMA) fallback to last-7-day mean.
    Later: plug ARIMA/Prophet/Croston auto-selection behind the same interface.
    """

    def __init__(self, window: int = 7):
        self.window = window

    def forecast(self, sales_df: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
        """
        Args:
            sales_df: columns [date, sku, units_sold]
            horizon_days: int forecast horizon
        Returns:
            DataFrame columns: [sku, date, forecast_units, model_used]
        """
        required = {"date", "sku", "units_sold"}
        if not required.issubset(sales_df.columns):
            raise ValueError(f"sales_df must contain columns {required}")

        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"])  # type: ignore
        # SMA per SKU
        sma = (
            sales_df.sort_values(["sku", "date"]).groupby("sku")["units_sold"].rolling(self.window).mean().reset_index()
        )
        sma.rename(columns={"units_sold": "sma"}, inplace=True)
        sales_df = sales_df.reset_index(drop=True)
        sales_df["sma"] = sma["sma"].values

        # For each SKU, take last SMA as constant forecast for horizon
        records = []
        last_date = sales_df["date"].max()
        for sku, group in sales_df.groupby("sku"):
            level = float(group["sma"].dropna().iloc[-1]) if group["sma"].notna().any() else float(group["units_sold"].tail(7).mean())
            for d in range(1, horizon_days + 1):
                records.append(
                    ForecastResult(
                        sku=sku,
                        date=last_date + pd.Timedelta(days=d),
                        forecast_units=max(level, 0.0),
                        model_used="SMA7",
                    ).__dict__
                )
        return pd.DataFrame.from_records(records)
