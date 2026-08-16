"""
[IMPLEMENTED] Unit and integration tests for Flower Federated Learning core.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.database.connection import Base
from backend.dependencies import get_db
from backend.federated.client import (
    FlowerTabularClient,
)
from backend.federated.server import run_federated_simulation
from backend.main import app
from clients_simulation.data_partitioner import (
    partition_data_iid,
    partition_data_non_iid_dirichlet,
)

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


def test_data_partitioners():
    """Verify IID and Non-IID Dirichlet partitioners preserve sample counts and data locality."""
    splits = load_starter_tabular_dataset()
    num_clients = 4

    # 1. Test IID partitioning
    iid_parts = partition_data_iid(splits.X_train, splits.y_train, num_clients=num_clients)
    assert len(iid_parts) == num_clients
    total_iid_samples = sum(p.num_samples for p in iid_parts)
    assert total_iid_samples == len(splits.X_train)
    for p in iid_parts:
        assert len(p.X_train) > 0
        assert len(p.X_val) > 0
        assert p.X_train.shape[1] == 30

    # 2. Test Non-IID Dirichlet partitioning
    non_iid_parts = partition_data_non_iid_dirichlet(
        splits.X_train, splits.y_train, num_clients=num_clients, alpha=0.3
    )
    assert len(non_iid_parts) == num_clients
    for p in non_iid_parts:
        assert len(p.X_train) > 0
        assert len(p.X_val) > 0


def test_flower_tabular_client_fit_and_eval():
    """Verify client local PyTorch training step and evaluation."""
    splits = load_starter_tabular_dataset()
    parts = partition_data_iid(splits.X_train, splits.y_train, num_clients=2)
    fl_client = FlowerTabularClient(
        client_id="test_client_0",
        partition=parts[0],
        in_features=30,
        device="cpu"
    )

    init_params = fl_client.get_parameters(config={})
    assert len(init_params) > 0
    assert isinstance(init_params[0], np.ndarray)

    # Local training step
    updated_params, num_samples, fit_metrics = fl_client.fit(
        parameters=init_params,
        config={"local_epochs": 2, "batch_size": 16, "learning_rate": 0.01}
    )
    assert num_samples == len(parts[0].X_train)
    assert "train_loss" in fit_metrics
    assert "train_accuracy" in fit_metrics

    # Local evaluation step
    val_loss, eval_samples, eval_metrics = fl_client.evaluate(
        parameters=updated_params,
        config={}
    )
    assert eval_samples == len(parts[0].X_val)
    assert val_loss > 0.0
    assert "val_accuracy" in eval_metrics


def test_end_to_end_federated_simulation():
    """Verify end-to-end multi-round FedAvg simulation and DB logging."""
    db = TestingSessionLocal()
    splits = load_starter_tabular_dataset()
    parts = partition_data_iid(splits.X_train, splits.y_train, num_clients=3)

    # Run 2-round simulation
    sim_results = run_federated_simulation(
        partitions=parts,
        num_rounds=2,
        local_epochs=1,
        batch_size=32,
        db=db,
        experiment_id=None
    )

    assert sim_results["status"] == "completed"
    assert sim_results["num_rounds"] == 2
    assert sim_results["num_clients"] == 3
    assert len(sim_results["round_history"]) == 2
    assert sim_results["total_comm_mb"] > 0.0
    assert sim_results["final_train_accuracy"] > 0.70
    db.close()


def test_training_api_routes():
    """Verify POST /api/v1/training/start and GET /api/v1/training/{id} endpoints."""
    # 1. Start FL Training
    resp = client.post(
        "/api/v1/training/start",
        json={
            "name": "API FL Test Run",
            "num_clients": 3,
            "num_rounds": 2,
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
            "partition_mode": "iid"
        }
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "API FL Test Run"
    assert data["status"] == "completed"
    exp_id = data["id"]
    assert len(data["training_rounds"]) == 2

    # 2. Get status
    status_resp = client.get(f"/api/v1/training/{exp_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["id"] == exp_id
    assert status_data["status"] == "completed"

    # 3. Get metrics time-series
    metrics_resp = client.get(f"/api/v1/training/{exp_id}/metrics")
    assert metrics_resp.status_code == 200
    metrics_list = metrics_resp.json()
    assert len(metrics_list) > 0
    metric_names = {m["metric_name"] for m in metrics_list}
    assert "fl_train_loss" in metric_names
    assert "fl_total_comm_mb" in metric_names
