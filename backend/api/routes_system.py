"""
[IMPLEMENTED] System health and diagnostics routes.
"""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings
from backend.dependencies import get_settings

router = APIRouter(prefix="/system", tags=["System"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    quantum_simulator: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def get_health(config: Settings = Depends(get_settings)):
    """
    Get system health status, runtime metadata, and simulator configuration.
    """
    return HealthResponse(
        status="healthy",
        app_name=config.APP_NAME,
        version=config.VERSION,
        environment=config.APP_ENV,
        quantum_simulator=config.QUANTUM_SIMULATOR_BACKEND,
        timestamp=datetime.now(UTC).isoformat()
    )
