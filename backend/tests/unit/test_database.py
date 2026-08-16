"""
[IMPLEMENTED] Unit tests for database models and repositories.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base
from backend.database.models_orm import (
    Dataset,
    Prediction,
    SecurityEvent,
)
from backend.database.repositories.client_repo import client_repo
from backend.database.repositories.model_repo import model_repo
from backend.database.repositories.quantum_job_repo import quantum_job_repo
from backend.database.repositories.training_repo import training_repo
from backend.database.repositories.user_repo import user_repo


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def test_create_all_tables(test_db):
    """Verify all 10 tables are created and operational in SQLite."""
    user = user_repo.create(test_db, "test@example.com", "testuser", "securepass123", "Test User")
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.username == "testuser"

    client = client_repo.register_or_update(test_db, "client-node-01", "Edge Node Alpha", "192.168.1.100")
    assert client.id == "client-node-01"
    assert client.status == "online"

    dataset = Dataset(
        name="heart_disease_benchmark",
        dataset_type="tabular",
        num_samples=1000,
        num_features=14,
        feature_names=["age", "chol", "trestbps"],
        target_name="target"
    )
    test_db.add(dataset)
    test_db.commit()
    assert dataset.id is not None

    exp = training_repo.create_experiment(
        test_db,
        name="FL Baseline Exp 1",
        baseline_type="federated_classical",
        dataset_id=dataset.id
    )
    assert exp.id is not None

    t_round = training_repo.create_round(
        test_db,
        experiment_id=exp.id,
        round_number=1,
        selected_client_ids=["client-node-01"]
    )
    assert t_round.id is not None

    t_round_done = training_repo.finish_round(
        test_db,
        round_id=t_round.id,
        loss=0.342,
        accuracy=0.885,
        metrics={"f1": 0.87}
    )
    assert t_round_done.loss == 0.342

    mv = model_repo.register(
        test_db,
        model_name="GlobalTabularModel",
        version="v1.0.0",
        architecture_type="xgboost",
        validation_metrics={"accuracy": 0.885, "roc_auc": 0.92},
        is_production=True
    )
    assert mv.id is not None
    assert mv.is_production is True

    prod_model = model_repo.get_latest_production(test_db, "GlobalTabularModel")
    assert prod_model is not None
    assert prod_model.version == "v1.0.0"

    q_job = quantum_job_repo.create(
        test_db,
        job_type="feature_selection",
        backend_used="qiskit_aer",
        num_qubits=8,
        circuit_depth=2
    )
    assert q_job.id is not None
    assert q_job.num_qubits == 8

    q_job_done = quantum_job_repo.finish(
        test_db,
        job_id=q_job.id,
        objective_value=-4.52,
        classical_objective_value=-4.50,
        execution_time_ms=120.5,
        classical_time_ms=45.2,
        result={"selected_features": [0, 2, 5]}
    )
    assert q_job_done.objective_value == -4.52

    metric = training_repo.log_metric(
        test_db,
        metric_name="eval_accuracy",
        metric_value=0.885,
        step=1,
        experiment_id=exp.id,
        round_id=t_round.id
    )
    assert metric.id is not None

    pred = Prediction(
        model_version_id=mv.id,
        input_data={"age": 55, "chol": 240},
        prediction_output={"class": 1},
        confidence_score=0.94,
        latency_ms=12.4
    )
    test_db.add(pred)
    test_db.commit()
    assert pred.id is not None

    sec_event = SecurityEvent(
        event_type="poisoning_attempt",
        severity="warning",
        client_id=client.id,
        details={"gradient_norm": 45.2, "threshold": 5.0}
    )
    test_db.add(sec_event)
    test_db.commit()
    assert sec_event.id is not None
