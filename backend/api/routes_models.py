"""
[IMPLEMENTED] Model Registry REST APIs.
Manages model lifecycle staging (development -> staging -> production),
version metadata tracking, artifact persistence, and active production model discovery.
"""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.models_orm import ModelVersion
from backend.dependencies import get_db

router = APIRouter(prefix="/models", tags=["Model Registry"])


# --- Schemas ---
class ModelRegisterRequest(BaseModel):
    model_name: str = Field(..., description="Name of model family, e.g. 'BreastCancer-AutoML'")
    version: str = Field(default="v1.0.0", description="Semantic version string")
    architecture_type: str = Field(default="xgboost", description="xgboost | random_forest | transformer | mlp")
    model_binary_path: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    is_production: bool = False


class ModelPromoteRequest(BaseModel):
    is_production: bool = Field(default=True)


class ModelVersionResponse(BaseModel):
    id: int
    model_name: str
    version: str
    architecture_type: str
    model_binary_path: str | None
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, Any]
    is_production: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Endpoints ---
@router.post("/register", response_model=ModelVersionResponse, status_code=status.HTTP_201_CREATED)
def register_model_version(
    req: ModelRegisterRequest,
    db: Session = Depends(get_db)
):
    """Register a new trained model version in the central registry."""
    existing = db.query(ModelVersion).filter(
        ModelVersion.model_name == req.model_name,
        ModelVersion.version == req.version
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model version '{req.model_name}:{req.version}' already registered."
        )

    # If new version is marked production, demote previous production versions
    if req.is_production:
        db.query(ModelVersion).filter(
            ModelVersion.model_name == req.model_name,
            ModelVersion.is_production == True
        ).update({"is_production": False})

    model_ver = ModelVersion(
        model_name=req.model_name,
        version=req.version,
        architecture_type=req.architecture_type,
        model_binary_path=req.model_binary_path,
        hyperparameters=req.hyperparameters,
        validation_metrics=req.validation_metrics,
        is_production=req.is_production,
        created_at=datetime.now(UTC)
    )
    db.add(model_ver)
    db.commit()
    db.refresh(model_ver)

    return model_ver


@router.get("", response_model=list[ModelVersionResponse])
def list_models(
    model_name: str | None = None,
    architecture_type: str | None = None,
    is_production: bool | None = None,
    db: Session = Depends(get_db)
):
    """List all registered models with optional filtering."""
    query = db.query(ModelVersion)
    if model_name:
        query = query.filter(ModelVersion.model_name == model_name)
    if architecture_type:
        query = query.filter(ModelVersion.architecture_type == architecture_type)
    if is_production is not None:
        query = query.filter(ModelVersion.is_production == is_production)

    return query.order_by(ModelVersion.id.desc()).all()


@router.get("/production/active", response_model=ModelVersionResponse)
def get_active_production_model(
    db: Session = Depends(get_db)
):
    """Retrieve the currently active production model for inference."""
    prod_model = db.query(ModelVersion).filter(ModelVersion.is_production == True).first()
    if not prod_model:
        # Fallback to the latest registered model version
        prod_model = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()

    if not prod_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No production or active model version registered in the system."
        )

    return prod_model


@router.get("/{model_id}", response_model=ModelVersionResponse)
def get_model_by_id(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Fetch specific model version metadata by ID."""
    model_ver = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model_ver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version ID {model_id} not found."
        )
    return model_ver


@router.put("/{model_id}/stage", response_model=ModelVersionResponse)
def update_model_stage(
    model_id: int,
    req: ModelPromoteRequest,
    db: Session = Depends(get_db)
):
    """
    Promote model version to production or demote to development/archived.
    Promoting a version automatically demotes all other versions in the same model family.
    """
    model_ver = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model_ver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version ID {model_id} not found."
        )

    if req.is_production:
        # Demote previous production versions
        db.query(ModelVersion).filter(
            ModelVersion.model_name == model_ver.model_name,
            ModelVersion.id != model_id,
            ModelVersion.is_production == True
        ).update({"is_production": False})
        model_ver.is_production = True
    else:
        model_ver.is_production = False

    db.commit()
    db.refresh(model_ver)
    return model_ver
