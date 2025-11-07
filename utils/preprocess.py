from __future__ import annotations

import pandas as pd

_ACCEPTABLE_DATE_COLUMNS = ["date", "dates", "order_date", "transaction_date"]
_ACCEPTABLE_UNITS_COLUMNS = ["units_sold", "units", "sales_qty", "quantity"]


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate raw sales data.

    Accepts variant column names for date and quantity, normalizes them to
    canonical: date, sku, units_sold. Raises ValueError with a clear message
    instead of triggering a KeyError when expected columns are absent.
    """
    if df is None or df.empty:
        raise ValueError("Provided sales data is empty.")

    original_cols = list(df.columns)
    df = df.copy()
    # Normalize column names to lowercase stripped for matching
    lower_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Resolve date column
    date_col = next((c for c in _ACCEPTABLE_DATE_COLUMNS if c in df.columns), None)
    if not date_col:
        raise ValueError(
            f"No date column found. Expected one of {_ACCEPTABLE_DATE_COLUMNS}. Got: {original_cols}"
        )
    if date_col != "date":
        df.rename(columns={date_col: "date"}, inplace=True)

    # Resolve units column
    units_col = next((c for c in _ACCEPTABLE_UNITS_COLUMNS if c in df.columns), None)
    if not units_col:
        raise ValueError(
            f"No units/quantity column found. Expected one of {_ACCEPTABLE_UNITS_COLUMNS}. Got: {original_cols}"
        )
    if units_col != "units_sold":
        df.rename(columns={units_col: "units_sold"}, inplace=True)

    if "sku" not in df.columns:
        raise ValueError("Missing 'sku' column in uploaded data.")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["sku", "date"])  # remove rows with invalid date/SKU

    # Clean numerical units
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).clip(lower=0)

    # Sort for downstream operations
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    return df
