"""
FastAPI application entry point for the Problem Bank API.

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import check_db_connection, init_db
from . import models  # Import models to register with SQLAlchemy metadata
from .routers import problems
from .schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    logger.info("Starting Problem Bank API")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down Problem Bank API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for storing and querying math problems with concept vectors",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(problems.router, prefix=settings.api_prefix)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Check API and database health."""
    db_ok = check_db_connection()
    status = "healthy" if db_ok else "degraded"
    return HealthResponse(
        status=status,
        database=db_ok,
        version=settings.api_version,
    )


@app.get("/", tags=["root"])
def root() -> dict:
    """API root endpoint."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }
