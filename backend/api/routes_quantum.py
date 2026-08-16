"""
[IMPLEMENTED] Quantum Optimization REST APIs.
Endpoints to trigger QAOA optimization jobs on-demand and retrieve dual-solver telemetry.
"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.automl.quantum_bridge import quantum_bridge
from backend.database.models_orm import QuantumJob
from backend.dependencies import get_db

router = APIRouter(prefix="/quantum", tags=["Quantum Optimization"])


# --- Schemas ---
class QuantumOptimizeRequest(BaseModel):
    problem_type: str = Field(default="feature_selection", description="feature_selection | client_selection | hpo")
    k: int = Field(default=5, ge=1, le=16)
    p_layers: int = Field(default=1, ge=1, le=4)
    shots: int = Field(default=1024, ge=128, le=8192)


class QuantumJobResponse(BaseModel):
    id: int
    job_type: str
    status: str
    backend_used: str
    num_qubits: int
    circuit_depth: int
    objective_value: float | None
    classical_objective_value: float | None
    execution_time_ms: float | None
    classical_time_ms: float | None
    result: dict[str, Any] = {}

    class Config:
        from_attributes = True


# --- Endpoints ---
@router.post("/optimize", status_code=status.HTTP_200_OK)
def trigger_quantum_optimization(
    req: QuantumOptimizeRequest,
    db: Session = Depends(get_db)
):
    """
    Trigger on-demand Quantum QAOA optimization with side-by-side classical benchmark.
    """
    problem = req.problem_type.lower().strip()

    if problem in ["feature_selection", "features"]:
        splits = load_starter_tabular_dataset()
        res = quantum_bridge.select_features(
            X=splits.X_train,
            y=splits.y_train,
            k=req.k,
            optimizer="quantum",
            feature_names=splits.feature_names,
            p_layers=req.p_layers,
            shots=req.shots,
            db=db
        )
        return res

    elif problem in ["client_selection", "clients"]:
        clients_data = [
            {"client_id": f"client_{i}", "data_quality_score": round(0.5 + 0.1 * i, 2), "comm_cost_score": round(0.2 + 0.15 * i, 2)}
            for i in range(5)
        ]
        res = quantum_bridge.select_clients(
            clients_data=clients_data,
            k=req.k,
            optimizer="quantum",
            p_layers=req.p_layers,
            shots=req.shots,
            db=db
        )
        return res

    elif problem in ["hpo", "hpo_selection"]:
        splits = load_starter_tabular_dataset()
        res = quantum_bridge.select_hyperparameters(
            model_type="xgboost",
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_val=splits.X_val,
            y_val=splits.y_val,
            optimizer="quantum",
            p_layers=req.p_layers,
            shots=req.shots,
            db=db
        )
        return {
            "optimizer": res["optimizer"],
            "best_hyperparameters": res["best_hyperparameters"],
            "validation_accuracy": res["validation_accuracy"],
            "quantum_job_id": res["quantum_job_id"],
            "telemetry": res["telemetry"]
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported quantum problem type: {req.problem_type}"
        )


@router.get("/jobs", response_model=list[QuantumJobResponse])
def list_quantum_jobs(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Retrieve historical quantum optimization jobs and comparative benchmark records.
    """
    jobs = db.query(QuantumJob).order_by(QuantumJob.id.desc()).limit(limit).all()
    return jobs


@router.get("/jobs/{job_id}", response_model=QuantumJobResponse)
def get_quantum_job(
    job_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch specific quantum job record by ID.
    """
    job = db.query(QuantumJob).filter(QuantumJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quantum job with ID {job_id} not found."
        )
    return job
