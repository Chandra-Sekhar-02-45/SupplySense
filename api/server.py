from __future__ import annotations

import sys
from pathlib import Path
from fastapi import FastAPI

# Ensure root path for imports when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes import router  # noqa: E402

app = FastAPI(title="SupplySense API", version="0.1.0")
app.include_router(router, prefix="/api")


@app.get("/")
def index():
    return {"message": "SupplySense API", "docs": "/docs"}

# To run: uvicorn api.server:app --reload
