from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database.database import Base, engine
from database import models

from app.routers.auth import router as auth_router
from app.routers.risk import router as risk_router
from app.routers.sos import router as sos_router


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "app" / "frontend"


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# Creates the SQLite database and tables if they do not exist.
Base.metadata.create_all(bind=engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SafeHer-AI",
    description=(
        "Geo-Fenced Emergency Alert and Safety Assistance System "
        "for Women"
    ),
    version="1.0.0"
)


# ============================================================
# ROUTERS
# ============================================================

# Authentication
app.include_router(auth_router)

# Risk assessment and geofencing
app.include_router(risk_router)

# Emergency / SOS
app.include_router(sos_router)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "running",
        "project": "SafeHer-AI",
        "model": "MLP temporal risk prediction",
        "clustering": "K-Means",
        "gis": "active",
        "geofencing": "active",
        "sos": "active",
        "authentication": "active"
    }


# ============================================================
# FRONTEND
# ============================================================

# Serves:
# app/frontend/index.html
# app/frontend/style.css
# app/frontend/app.js

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)