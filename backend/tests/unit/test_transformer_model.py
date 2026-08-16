"""
[IMPLEMENTED] Unit and integration tests for Time-Series Transformer Client model and sequence federated learning.
"""
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import load_sensor_timeseries_dataset
from backend.database.connection import Base
from backend.dependencies import get_db
from backend.federated.round_manager import round_manager
from backend.main import app
from backend.models.transformer_model import (
    PositionalEncoding,
    TimeSeriesTransformerNN,
    TransformerModelWrapper,
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


def test_positional_encoding_tensor_shapes():
    """Verify positional encoding preserves tensor dimensions."""
    pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
    x = torch.randn(4, 24, 32)  # (batch_size, seq_len, d_model)
    out = pe(x)
    assert out.shape == (4, 24, 32)


def test_transformer_nn_forward_pass():
    """Verify forward pass on 3D sequence and 2D tabular inputs."""
    model = TimeSeriesTransformerNN(in_features=6, d_model=32, nhead=4, num_layers=2, num_classes=2)

    # 3D Sequence Input
    seq_x = torch.randn(8, 24, 6)
    seq_out = model(seq_x)
    assert seq_out.shape == (8, 2)

    # 2D Tabular Input
    tab_x = torch.randn(8, 6)
    tab_out = model(tab_x)
    assert tab_out.shape == (8, 2)


def test_transformer_model_wrapper_training():
    """Verify TransformerModelWrapper fit, predict, and predict_proba."""
    splits = load_sensor_timeseries_dataset(num_samples=200, seq_len=16, num_features=4)
    model = TransformerModelWrapper(
        in_features=4,
        d_model=16,
        nhead=2,
        num_layers=1,
        epochs=3,
        batch_size=16,
        lr=0.01,
        device="cpu"
    )

    # Fit
    train_metrics = model.fit(splits.X_train, splits.y_train)
    assert "train_loss" in train_metrics
    assert "train_accuracy" in train_metrics

    # Predict
    preds = model.predict(splits.X_test)
    assert len(preds) == len(splits.y_test)
    assert set(preds).issubset({0, 1})

    # Predict Proba
    probs = model.predict_proba(splits.X_test)
    assert probs.shape == (len(splits.y_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_federated_transformer_simulation():
    """Verify end-to-end multi-client Federated Learning with Time-Series Transformer architecture."""
    db = TestingSessionLocal()
    exp, results = round_manager.start_training_run(
        db=db,
        name="FL Transformer Sequence Benchmark",
        num_clients=3,
        num_rounds=2,
        local_epochs=1,
        batch_size=16,
        learning_rate=0.01,
        model_architecture="transformer",
        dataset_type="sequence",
        baseline_type="federated_transformer"
    )

    assert exp.id is not None
    assert results["status"] == "completed"
    assert results["num_rounds"] == 2
    assert results["num_clients"] == 3
    assert len(results["round_history"]) == 2
    assert results["total_comm_mb"] > 0.0
    db.close()
