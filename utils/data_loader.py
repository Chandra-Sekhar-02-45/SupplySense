from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"date", "sku", "units_sold"}


def load_sales_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])  # type: ignore
    return df


def load_form_json(path: str | Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of entries each with date, sku, units_sold
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if not isinstance(data, list):
        raise ValueError("Form JSON must contain a list or {'entries': [...]} structure")
    df = pd.DataFrame(data)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Form JSON missing required fields: {missing}")
    df["date"] = pd.to_datetime(df["date"])  # type: ignore
    return df
