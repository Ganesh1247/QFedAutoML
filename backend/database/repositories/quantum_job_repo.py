"""
[IMPLEMENTED] Quantum job repository.
Logs quantum optimization jobs, circuit metrics, and side-by-side classical solver benchmarks.
"""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models_orm import QuantumJob


class QuantumJobRepository:
    @staticmethod
    def create(
        db: Session,
        job_type: str,
        backend_used: str = "qiskit_aer",
        num_qubits: int = 0,
        circuit_depth: int = 0,
        parameters: dict | None = None
    ) -> QuantumJob:
        """Create a new quantum job log."""
        job = QuantumJob(
            job_type=job_type,
            status="running",
            backend_used=backend_used,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            parameters=parameters or {},
            created_at=datetime.now(UTC)
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def finish(
        db: Session,
        job_id: int,
        objective_value: float,
        classical_objective_value: float,
        execution_time_ms: float,
        classical_time_ms: float,
        result: dict | None = None,
        status: str = "completed"
    ) -> QuantumJob | None:
        """Record completed results comparing quantum and classical solvers."""
        job = db.query(QuantumJob).filter(QuantumJob.id == job_id).first()
        if job:
            job.objective_value = objective_value
            job.classical_objective_value = classical_objective_value
            job.execution_time_ms = execution_time_ms
            job.classical_time_ms = classical_time_ms
            job.result = result or {}
            job.status = status
            db.commit()
            db.refresh(job)
        return job

    @staticmethod
    def get_by_id(db: Session, job_id: int) -> QuantumJob | None:
        """Retrieve job by primary key."""
        return db.query(QuantumJob).filter(QuantumJob.id == job_id).first()


quantum_job_repo = QuantumJobRepository()
