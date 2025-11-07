from __future__ import annotations

import pandas as pd


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["sku", "date"])  # remove broken rows
    df["units_sold"] = df["units_sold"].fillna(0).clip(lower=0)
    df = df.sort_values(["sku", "date"])  # ensure order for rolling calcs
    return df
