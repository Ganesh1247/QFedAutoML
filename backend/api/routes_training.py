"""
[IMPLEMENTED] Federated Training routes: starting simulations, retrieving round progress and metrics.
"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.models_orm import Experiment, Metric
from backend.dependencies import get_db
from backend.federated.round_manager import round_manager

router = APIRouter(prefix="/training", tags=["Training"])


# --- Schemas ---
class StartTrainingRequest(BaseModel):
    name: str = Field(default="Federated Learning Run", min_length=3, max_length=128)
    num_clients: int = Field(default=5, ge=2, le=50)
    num_rounds: int = Field(default=5, ge=1, le=50)
    local_epochs: int = Field(default=2, ge=1, le=20)
    batch_size: int = Field(default=32, ge=4, le=512)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    partition_mode: str = Field(default="non_iid")  # iid, non_iid
    dirichlet_alpha: float = Field(default=0.5, gt=0.0)
    model_architecture: str = Field(default="tabular_nn")  # tabular_nn, transformer
    dataset_type: str = Field(default="tabular")  # tabular, sequence
    baseline_type: str = Field(default="federated_classical")


class TrainingRoundResponse(BaseModel):
    id: int
    round_number: int
    status: str
    selected_client_ids: list[str]
    loss: float | None
    accuracy: float | None
    round_metrics: dict[str, Any]

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    id: int
    name: str
    baseline_type: str
    status: str
    config: dict[str, Any]
    training_rounds: list[TrainingRoundResponse] = []

    class Config:
        from_attributes = True


class MetricResponse(BaseModel):
    id: int
    step: int
    metric_name: str
    metric_value: float
    round_id: int | None

    class Config:
        from_attributes = True


# --- Endpoints ---
@router.post("/start", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def start_training(
    req: StartTrainingRequest,
    db: Session = Depends(get_db)
):
    """
    Initiate and run a decentralized federated training simulation.
    """
    exp, _ = round_manager.start_training_run(
        db=db,
        name=req.name,
        num_clients=req.num_clients,
        num_rounds=req.num_rounds,
        local_epochs=req.local_epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        partition_mode=req.partition_mode,
        dirichlet_alpha=req.dirichlet_alpha,
        model_architecture=req.model_architecture,
        dataset_type=req.dataset_type,
        baseline_type=req.baseline_type
    )
    # Refresh to load rounds relationship
    db.refresh(exp)
    return exp


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_training_status(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve federated experiment status, configuration, and completed rounds.
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with ID {experiment_id} not found."
        )
    return exp


@router.get("/{experiment_id}/metrics", response_model=list[MetricResponse])
def get_training_metrics(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch all time-series telemetry metrics logged across rounds for an experiment.
    """
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with ID {experiment_id} not found."
        )

    metrics = db.query(Metric).filter(Metric.experiment_id == experiment_id).order_by(Metric.step.asc()).all()
    return metrics
