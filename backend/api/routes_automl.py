"""
[IMPLEMENTED] AutoML Engine REST APIs.
Endpoints to trigger Automated Machine Learning pipelines (Classical vs Quantum),
query dataset statistical profiles, and inspect the ranked candidate leaderboard.
"""
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.automl.dataset_profiler import profile_dataset
from backend.automl.leaderboard import leaderboard
from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.automl.quantum_bridge import quantum_bridge
from backend.database.models_orm import ModelVersion
from backend.dependencies import get_db

router = APIRouter(prefix="/automl", tags=["AutoML Engine"])


# --- Schemas ---
class AutoMLRunRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="xgboost | random_forest | logistic_regression")
    feature_optimizer: str = Field(default="quantum", description="quantum | classical")
    hpo_optimizer: str = Field(default="classical", description="classical | quantum")
    k_features: int = Field(default=6, ge=2, le=16)


def _clean_nans(data: Any) -> Any:
    """Recursively sanitize numpy scalars, arrays, NaNs, and Infs for JSON compliance."""
    if data is None:
        return None
    if isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    if isinstance(data, (int, np.integer)):
        return int(data)
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    if isinstance(data, np.ndarray):
        return _clean_nans(data.tolist())
    if isinstance(data, dict):
        return {str(k): _clean_nans(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [_clean_nans(v) for v in data]
    return data


# --- Endpoints ---
@router.post("/run", status_code=status.HTTP_200_OK)
def run_automl_job(
    req: AutoMLRunRequest,
    db: Session = Depends(get_db)
):
    """
    Trigger end-to-end Automated Machine Learning pipeline:
    Feature selection (Quantum QAOA vs Classical Mutual Info) -> Hyperparameter Search -> Model Training -> Leaderboard Registration.
    """
    splits = load_starter_tabular_dataset()

    res = quantum_bridge.run_full_automl_pipeline(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        model_type=req.model_type,
        k_features=req.k_features,
        feature_optimizer=req.feature_optimizer,
        hpo_optimizer=req.hpo_optimizer,
        feature_names=splits.feature_names,
        db=db,
        random_state=42
    )

    clean_res = _clean_nans(res)
    if "hpo_results" in clean_res and isinstance(clean_res["hpo_results"], dict):
        clean_res["hpo_results"].pop("model", None)

    # Register in Model Registry
    model_ver = ModelVersion(
        model_name=f"AutoML-{req.model_type.upper()}-{req.feature_optimizer.upper()}",
        version=f"v1.{len(leaderboard.candidates)}",
        architecture_type=req.model_type,
        hyperparameters=clean_res.get("hpo_results", {}).get("best_hyperparameters", {}) or {},
        validation_metrics=clean_res.get("validation_metrics", {}) or {},
        is_production=False,
        created_at=datetime.now(UTC)
    )
    db.add(model_ver)
    db.commit()
    db.refresh(model_ver)

    clean_res["registered_model_version_id"] = model_ver.id
    return clean_res


@router.get("/leaderboard")
def get_leaderboard():
    """Retrieve ranked candidate models evaluated by the AutoML engine."""
    candidates = leaderboard.get_leaderboard()
    best_candidate = leaderboard.get_best_candidate()
    return {
        "total_candidates": len(candidates),
        "best_candidate": best_candidate,
        "leaderboard": candidates
    }


@router.get("/profile")
def get_dataset_profile():
    """Compute statistical health, skewness, and multicollinearity profile of the tabular dataset."""
    splits = load_starter_tabular_dataset()
    profile = profile_dataset(
        X=splits.X_train,
        y=splits.y_train,
        feature_names=splits.feature_names
    )
    return profile
