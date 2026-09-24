from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers.risk import router as risk_router


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SafeHer-AI",
    description=(
        "Geo-Fenced Emergency Alert and "
        "Safety Assistance System for Women"
    ),
    version="1.0.0"
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(risk_router)


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)