"""
[IMPLEMENTED] Unit and integration tests for Phase 11 REST API endpoints:
Clients management, Model Registry staging, Real-time Predict with latency auditing, AutoML triggers, and Explainability routes.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.database.connection import Base
from backend.dependencies import get_db
from backend.main import app

# Test database setup
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


def test_clients_api_flow():
    """Verify client registration, heartbeat polling, and list retrieval."""
    # 1. Register Client
    reg_resp = client.post(
        "/api/v1/clients/register",
        json={
            "id": "edge_node_42",
            "name": "Hospital Edge Server Alpha",
            "device_info": {"cpu": "Intel i7", "ram_gb": 32, "os": "Ubuntu 22.04"},
            "data_samples_count": 500,
            "data_quality_score": 0.95
        }
    )
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    assert reg_data["id"] == "edge_node_42"
    assert reg_data["status"] == "online"
    assert "privacy_status" in reg_data

    # 2. Heartbeat
    hb_resp = client.post(
        "/api/v1/clients/edge_node_42/heartbeat",
        json={"status": "online", "data_quality_score": 0.96}
    )
    assert hb_resp.status_code == 200
    assert hb_resp.json()["data_quality_score"] == 0.96

    # 3. List Clients
    list_resp = client.get("/api/v1/clients")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # 4. Get Client Detail
    det_resp = client.get("/api/v1/clients/edge_node_42")
    assert det_resp.status_code == 200
    assert det_resp.json()["name"] == "Hospital Edge Server Alpha"


def test_models_registry_lifecycle_and_staging():
    """Verify model version registration and production staging promotion."""
    # 1. Register Version 1
    m1_resp = client.post(
        "/api/v1/models/register",
        json={
            "model_name": "Wisconsin-Diagnostic-Model",
            "version": "v1.0.0",
            "architecture_type": "xgboost",
            "hyperparameters": {"n_estimators": 50, "max_depth": 3},
            "validation_metrics": {"accuracy": 0.945, "f1": 0.951},
            "is_production": False
        }
    )
    assert m1_resp.status_code == 201
    m1_id = m1_resp.json()["id"]

    # 2. Register Version 2 as Production
    m2_resp = client.post(
        "/api/v1/models/register",
        json={
            "model_name": "Wisconsin-Diagnostic-Model",
            "version": "v2.0.0",
            "architecture_type": "xgboost",
            "hyperparameters": {"n_estimators": 100, "max_depth": 4},
            "validation_metrics": {"accuracy": 0.972, "f1": 0.978},
            "is_production": True
        }
    )
    assert m2_resp.status_code == 201
    m2_id = m2_resp.json()["id"]

    # 3. Query Active Production Model
    prod_resp = client.get("/api/v1/models/production/active")
    assert prod_resp.status_code == 200
    assert prod_resp.json()["id"] == m2_id
    assert prod_resp.json()["version"] == "v2.0.0"

    # 4. Promote Version 1 to Production
    stage_resp = client.put(f"/api/v1/models/{m1_id}/stage", json={"is_production": True})
    assert stage_resp.status_code == 200
    assert stage_resp.json()["is_production"] is True

    # Check that Version 2 was automatically demoted
    m2_check = client.get(f"/api/v1/models/{m2_id}")
    assert m2_check.json()["is_production"] is False


def test_real_time_predict_endpoint():
    """Verify tabular and sequence real-time inference with latency measurement and DB logging."""
    splits = load_starter_tabular_dataset()
    sample_features = splits.X_val[0].tolist()

    # 1. Tabular Prediction
    pred_resp = client.post(
        "/api/v1/predict",
        json={"features": sample_features}
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["prediction"] in [0, 1]
    assert len(pred_data["probabilities"]) == 2
    assert pred_data["confidence_score"] > 0.5
    assert pred_data["latency_ms"] > 0.0

    # 2. Time-Series Sequence Prediction
    sample_sequence = [[0.1 * j for j in range(6)] for _ in range(10)]
    seq_resp = client.post(
        "/api/v1/predict",
        json={"sequence": sample_sequence}
    )
    assert seq_resp.status_code == 200
    seq_data = seq_resp.json()
    assert seq_data["prediction"] in [0, 1]
    assert seq_data["architecture"] == "TimeSeriesTransformer"


def test_automl_api_routes():
    """Verify AutoML dataset profile and candidate leaderboard endpoints."""
    # 1. Dataset Profile
    prof_resp = client.get("/api/v1/automl/profile")
    assert prof_resp.status_code == 200
    prof_data = prof_resp.json()
    assert "num_samples" in prof_data
    assert "high_correlation_pairs" in prof_data
    assert "has_multicollinearity" in prof_data

    # 2. Trigger Classical AutoML Job
    run_resp = client.post(
        "/api/v1/automl/run",
        json={
            "model_type": "xgboost",
            "feature_optimizer": "classical",
            "hpo_optimizer": "classical",
            "k_features": 4
        }
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["status"] == "completed"
    assert "registered_model_version_id" in run_data

    # 3. Query Leaderboard
    lb_resp = client.get("/api/v1/automl/leaderboard")
    assert lb_resp.status_code == 200
    assert lb_resp.json()["total_candidates"] >= 1


def test_explainability_api_routes():
    """Verify SHAP, LIME, and Trust report generation endpoints."""
    # Register a model first
    reg_resp = client.post(
        "/api/v1/models/register",
        json={
            "model_name": "Diagnostic-XGB-Explainer",
            "version": "v1.0.0",
            "architecture_type": "xgboost",
            "hyperparameters": {"n_estimators": 30, "max_depth": 3},
            "validation_metrics": {"accuracy": 0.95},
            "is_production": True
        }
    )
    model_id = reg_resp.json()["id"]
    splits = load_starter_tabular_dataset()
    sample_inst = splits.X_val[0].tolist()

    # 1. SHAP endpoint
    shap_resp = client.get(f"/api/v1/explain/shap/{model_id}")
    assert shap_resp.status_code == 200
    assert "global_shap" in shap_resp.json()

    # 2. LIME endpoint
    lime_resp = client.post(
        f"/api/v1/explain/lime/{model_id}",
        json={"instance": sample_inst, "num_samples": 80}
    )
    assert lime_resp.status_code == 200
    assert "lime_explanation" in lime_resp.json()

    # 3. JSON Report endpoint
    rep_resp = client.get(f"/api/v1/explain/report/{model_id}")
    assert rep_resp.status_code == 200
    assert rep_resp.json()["model_name"] == "Diagnostic-XGB-Explainer"

    # 4. HTML Report endpoint
    html_resp = client.get(f"/api/v1/explain/report/{model_id}/html")
    assert html_resp.status_code == 200
    assert "<!DOCTYPE html>" in html_resp.text
