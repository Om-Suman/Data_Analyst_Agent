"""
Data Analyst Agent — FastAPI Backend
Production-ready REST API for AI-powered data analytics.
"""
from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import (
    routes_anomalies,
    routes_cleaning,
    routes_config,
    routes_datasets,
    routes_document,
    routes_explorer,
    routes_forecasting,
    routes_insights,
    routes_query,
    routes_reports,
    routes_visualizations,
)

# Auto-load .env files
for env_file in [Path(".env"), Path("../.env")]:
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

app = FastAPI(
    title="Data Analyst Agent API",
    description="Production-grade AI-powered data analytics API powered by FastAPI and Hugging Face models.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "Data Analyst Agent API",
        "version": "2.0.0",
    }


# Register all API routers under /api
app.include_router(routes_config.router, prefix="/api")
app.include_router(routes_datasets.router, prefix="/api")
app.include_router(routes_cleaning.router, prefix="/api")
app.include_router(routes_explorer.router, prefix="/api")
app.include_router(routes_query.router, prefix="/api")
app.include_router(routes_document.router, prefix="/api")
app.include_router(routes_visualizations.router, prefix="/api")
app.include_router(routes_insights.router, prefix="/api")
app.include_router(routes_forecasting.router, prefix="/api")
app.include_router(routes_anomalies.router, prefix="/api")
app.include_router(routes_reports.router, prefix="/api")

# Serve frontend build if dist folder exists
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, reload=True)
