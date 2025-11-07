from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_sales_history(df: pd.DataFrame):
    return px.line(df, x="date", y="units_sold", color="sku", title="Sales History")


def plot_forecast(df: pd.DataFrame):
    return px.line(df, x="date", y="forecast_units", color="sku", title="Forecast")
