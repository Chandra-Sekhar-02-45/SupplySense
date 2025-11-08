from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_sales_history(df: pd.DataFrame):
    return px.line(df, x="date", y="units_sold", color="sku", title="Sales History")


def plot_forecast(forecast_df: pd.DataFrame, history_df: pd.DataFrame | None = None, history_days: int = 30):
    """
    Plot forecast and optionally recent history together for context.

    - forecast_df: DataFrame with columns [date, sku, forecast_units]
    - history_df: optional DataFrame with columns [date, sku, units_sold]
    - history_days: how many days of history to overlay
    """
    if history_df is None or history_df.empty:
        return px.line(forecast_df, x="date", y="forecast_units", color="sku", title="Forecast")

    # Ensure datetime types
    fdf = forecast_df.copy()
    fdf["date"] = pd.to_datetime(fdf["date"])  # type: ignore
    hdf = history_df.copy()
    hdf["date"] = pd.to_datetime(hdf["date"])  # type: ignore

    # Keep only recent history for clarity
    last_date = fdf["date"].min()
    cutoff = last_date - pd.Timedelta(days=history_days)
    hdf = hdf[hdf["date"] >= cutoff]

    fig = go.Figure()

    # History traces
    for sku, g in hdf.groupby("sku"):
        fig.add_trace(
            go.Scatter(
                x=g["date"], y=g["units_sold"], mode="lines",
                name=f"{sku} (history)", line=dict(color="#94a3b8")
            )
        )

    # Forecast traces
    for sku, g in fdf.groupby("sku"):
        fig.add_trace(
            go.Scatter(
                x=g["date"], y=g["forecast_units"], mode="lines",
                name=f"{sku} (forecast)", line=dict(dash="dash", width=3)
            )
        )

    # Vertical marker at forecast start
    try:
        x0 = fdf["date"].min()
        fig.add_vline(x=x0, line=dict(color="#475569", dash="dot"))
    except Exception:
        pass

    fig.update_layout(title="Forecast", xaxis_title="date", yaxis_title="forecast_units")
    return fig
