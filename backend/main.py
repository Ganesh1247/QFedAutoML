"""
[IMPLEMENTED] QFedAutoML Backend Application Entrypoint.
Provides FastAPI routing, CORS middleware, and system endpoints.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_auth import router as auth_router
from backend.api.routes_automl import router as automl_router
from backend.api.routes_clients import router as clients_router
from backend.api.routes_datasets import router as datasets_router
from backend.api.routes_explainability import router as explain_router
from backend.api.routes_models import router as models_router
from backend.api.routes_predict import router as predict_router
from backend.api.routes_quantum import router as quantum_router
from backend.api.routes_system import router as system_router
from backend.api.routes_training import router as training_router
from backend.config import settings


def create_app() -> FastAPI:
    """Application factory for QFedAutoML."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Quantum-Enhanced Federated AutoML Platform for Privacy-Preserving Intelligent Systems",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers under API version prefix
    api_prefix = settings.API_V1_STR
    app.include_router(system_router, prefix=api_prefix)
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(clients_router, prefix=api_prefix)
    app.include_router(datasets_router, prefix=api_prefix)
    app.include_router(training_router, prefix=api_prefix)
    app.include_router(quantum_router, prefix=api_prefix)
    app.include_router(automl_router, prefix=api_prefix)
    app.include_router(explain_router, prefix=api_prefix)
    app.include_router(predict_router, prefix=api_prefix)
    app.include_router(models_router, prefix=api_prefix)

    # Root redirect / health route for root access
    @app.get("/")
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{api_prefix}/system/health"
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.APP_DEBUG)
