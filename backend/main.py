"""
Data Analyst Agent — FastAPI Backend
Production-ready REST API for AI-powered data analytics.
"""
from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
    description="Production-grade AI-powered data analytics API powered by FastAPI and Groq LLM models.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
env_origins = os.environ.get("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (StarletteHTTPException, RequestValidationError)):
        raise exc
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
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

# Serve frontend build if dist folder exists with SPA fallback
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith(("api", "docs", "redoc", "openapi.json")):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        target_file = frontend_dist / full_path
        if full_path and target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
