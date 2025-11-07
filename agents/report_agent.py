from __future__ import annotations

import math
import pandas as pd


class ReportAgent:
    """Combines forecast and safety-stock outputs to produce actionable reorder recommendations."""

    def __init__(self, pack_size_default: int = 1):
        self.pack = max(1, int(pack_size_default))

    def build(self,
              current_stock: pd.DataFrame,
              forecast_df: pd.DataFrame,
              safety_df: pd.DataFrame,
              horizon_days: int = 30) -> pd.DataFrame:
        """
        Args:
            current_stock: [sku, current_stock]
            forecast_df: [sku, date, forecast_units]
            safety_df: [sku, reorder_point, safety_stock]
        Returns:
            DataFrame: [sku, current_stock, reorder_point, safety_stock, forecast_30d, recommended_order_qty]
        """
        # Sum horizon demand
        horizon = forecast_df.copy()
        horizon = horizon.groupby("sku")["forecast_units"].sum().rename("forecast_30d").reset_index()

        df = (current_stock.merge(safety_df[["sku", "reorder_point", "safety_stock"]], on="sku", how="left")
                        .merge(horizon, on="sku", how="left"))
        df.fillna({"reorder_point": 0.0, "safety_stock": 0.0, "forecast_30d": 0.0}, inplace=True)

        need = (df[["reorder_point", "forecast_30d"]].max(axis=1) - df["current_stock"]).clip(lower=0)
        if self.pack > 1:
            need = need.apply(lambda x: int(math.ceil(x / self.pack)) * self.pack)
        df["recommended_order_qty"] = need.astype(int)
        return df[["sku", "current_stock", "reorder_point", "safety_stock", "forecast_30d", "recommended_order_qty"]]
