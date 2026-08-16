"""
[IMPLEMENTED] Unit and integration tests for AutoML-Quantum Bridge and Quantum REST API routes.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.automl.quantum_bridge import quantum_bridge
from backend.database.connection import Base
from backend.dependencies import get_db
from backend.main import app

# Test in-memory database with StaticPool
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)


def test_quantum_vs_classical_feature_selection_bridge():
    """Verify AutoMLQuantumBridge feature selection toggle."""
    splits = load_starter_tabular_dataset()
    db = TestingSessionLocal()

    # 1. Quantum QAOA Feature Selection
    q_res = quantum_bridge.select_features(
        X=splits.X_train,
        y=splits.y_train,
        k=4,
        optimizer="quantum",
        feature_names=splits.feature_names,
        max_qubits=8,
        p_layers=1,
        shots=512,
        db=db,
        random_state=42
    )
    assert q_res["optimizer"] == "quantum_qaoa"
    assert q_res["k"] == 4
    assert len(q_res["selected_indices"]) == 4
    assert len(q_res["selected_features"]) == 4
    assert q_res["quantum_job_id"] is not None

    # 2. Classical Feature Selection
    c_res = quantum_bridge.select_features(
        X=splits.X_train,
        y=splits.y_train,
        k=4,
        optimizer="classical",
        feature_names=splits.feature_names,
        random_state=42
    )
    assert c_res["optimizer"] == "classical_mutual_info"
    assert c_res["k"] == 4
    assert len(c_res["selected_indices"]) == 4
    db.close()


def test_quantum_vs_classical_client_selection_bridge():
    """Verify AutoMLQuantumBridge client selection toggle."""
    clients_data = [
        {"client_id": "client_a", "data_quality_score": 0.85, "comm_cost_score": 0.2},
        {"client_id": "client_b", "data_quality_score": 0.65, "comm_cost_score": 0.5},
        {"client_id": "client_c", "data_quality_score": 0.90, "comm_cost_score": 0.1},
        {"client_id": "client_d", "data_quality_score": 0.40, "comm_cost_score": 0.8}
    ]
    db = TestingSessionLocal()

    # 1. Quantum Client Selection
    q_res = quantum_bridge.select_clients(clients_data, k=2, optimizer="quantum", db=db)
    assert q_res["optimizer"] == "quantum_qaoa"
    assert len(q_res["selected_client_ids"]) == 2

    # 2. Classical Client Selection
    c_res = quantum_bridge.select_clients(clients_data, k=2, optimizer="classical")
    assert c_res["optimizer"] == "classical_heuristic"
    assert len(c_res["selected_client_ids"]) == 2
    # Client C and A have the highest quality scores
    assert "client_c" in c_res["selected_client_ids"]
    db.close()


def test_full_automl_pipeline_execution():
    """Verify end-to-end pipeline execution with Quantum FS + Classical Training."""
    splits = load_starter_tabular_dataset()
    db = TestingSessionLocal()

    pipeline_res = quantum_bridge.run_full_automl_pipeline(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        model_type="xgboost",
        k_features=6,
        feature_optimizer="quantum",
        hpo_optimizer="classical",
        feature_names=splits.feature_names,
        db=db,
        random_state=42
    )

    assert pipeline_res["status"] == "completed"
    assert pipeline_res["model_type"] == "xgboost"
    assert pipeline_res["validation_metrics"]["accuracy"] > 0.85
    assert pipeline_res["validation_metrics"]["roc_auc"] > 0.90
    db.close()


def test_quantum_api_routes():
    """Verify POST /api/v1/quantum/optimize and GET /api/v1/quantum/jobs endpoints."""
    # 1. Trigger Quantum Feature Selection
    resp = client.post(
        "/api/v1/quantum/optimize",
        json={
            "problem_type": "feature_selection",
            "k": 4,
            "p_layers": 1,
            "shots": 512
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["optimizer"] == "quantum_qaoa"
    assert "selected_indices" in data
    assert "telemetry" in data
    job_id = data["quantum_job_id"]

    # 2. Query Quantum Jobs list
    jobs_resp = client.get("/api/v1/quantum/jobs")
    assert jobs_resp.status_code == 200
    jobs_list = jobs_resp.json()
    assert len(jobs_list) > 0

    # 3. Query specific job
    if job_id:
        job_detail = client.get(f"/api/v1/quantum/jobs/{job_id}")
        assert job_detail.status_code == 200
        detail_data = job_detail.json()
        assert detail_data["id"] == job_id
        assert detail_data["status"] == "completed"
