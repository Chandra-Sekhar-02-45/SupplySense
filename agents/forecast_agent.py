from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
    sku: str
    date: pd.Timestamp
    forecast_units: float
    model_used: str


class ForecastAgent:
    """
    Forecasting agent.

    Implements per-SKU ARIMA forecasting (non-seasonal). If ARIMA fitting fails
    for a SKU (e.g., too little data), falls back to a simple moving average.
    """

    def __init__(self, window: int = 7, arima_orders: Optional[List[Tuple[int,int,int]]] = None):
        # window used only for fallback SMA
        self.window = window
        # Small set of candidate (p,d,q) orders; we pick the best by AIC
        self.arima_orders = arima_orders or [(0,1,1), (1,1,0), (1,1,1), (2,1,1)]

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
        records = []

        for sku, grp in sales_df.groupby("sku"):
            g = grp.sort_values("date").copy()
            # Daily aggregation and fill missing days with 0s
            g = g.set_index("date").asfreq("D")
            if "units_sold" not in g.columns:
                # in case of asfreq dropping column; re-assign
                g["units_sold"] = grp.set_index("date")["units_sold"].resample("D").sum()
            g["units_sold"] = g["units_sold"].fillna(0.0)

            y = g["units_sold"].astype(float)
            last_date = y.index.max()

            model_used = "ARIMA"
            fcst = None
            # Try a few ARIMA orders, pick lowest AIC
            best_aic = np.inf
            best_res = None
            for order in self.arima_orders:
                try:
                    # enforce stationary differencing via d in order; suppress convergence warnings by try/except
                    res = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(method_kwargs={"warn_convergence": False})
                    if res.aic < best_aic and np.isfinite(res.aic):
                        best_aic = res.aic
                        best_res = res
                except Exception:
                    continue

            if best_res is not None:
                try:
                    fcst_vals = best_res.forecast(steps=horizon_days)
                    fcst = [max(float(v), 0.0) for v in fcst_vals]
                except Exception:
                    fcst = None

            if fcst is None:
                # fallback: constant SMA level
                model_used = "SMA7"
                sma = (
                    grp.sort_values(["sku", "date"]).groupby("sku")["units_sold"].rolling(self.window).mean().reset_index()
                )
                sma.rename(columns={"units_sold": "sma"}, inplace=True)
                grp2 = grp.reset_index(drop=True)
                grp2["sma"] = sma["sma"].values
                level = float(grp2["sma"].dropna().iloc[-1]) if grp2["sma"].notna().any() else float(grp2["units_sold"].tail(7).mean())
                fcst = [max(level, 0.0)] * horizon_days

            # build records
            for d in range(1, horizon_days + 1):
                records.append(
                    ForecastResult(
                        sku=sku,
                        date=last_date + pd.Timedelta(days=d),
                        forecast_units=fcst[d-1],
                        model_used=model_used,
                    ).__dict__
                )

        return pd.DataFrame.from_records(records)
