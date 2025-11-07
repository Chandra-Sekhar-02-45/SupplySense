from __future__ import annotations
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def _get_float(name: str, default: float) -> float:
	try:
		return float(os.getenv(name, default))
	except Exception:
		return default

def _get_int(name: str, default: int) -> int:
	try:
		return int(float(os.getenv(name, default)))
	except Exception:
		return default

DEFAULT_SERVICE_LEVEL = _get_float("DEFAULT_SERVICE_LEVEL", 0.95)
DEFAULT_FORECAST_HORIZON_DAYS = _get_int("FORECAST_HORIZON_DAYS", 30)
DEFAULT_LEAD_TIME_DAYS = _get_int("DEFAULT_LEAD_TIME_DAYS", 7)
PACK_SIZE_DEFAULT = _get_int("PACK_SIZE_DEFAULT", 1)

# Optional AI key (not required for core pipeline)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
