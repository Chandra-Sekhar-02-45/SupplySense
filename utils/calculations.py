from __future__ import annotations

import math


def safety_stock(demand_std_per_day: float, lead_time_days: int, z: float) -> float:
    return z * demand_std_per_day * math.sqrt(max(lead_time_days, 0))


def reorder_point(demand_mean_per_day: float, lead_time_days: int, safety_stock_value: float) -> float:
    return demand_mean_per_day * lead_time_days + safety_stock_value
