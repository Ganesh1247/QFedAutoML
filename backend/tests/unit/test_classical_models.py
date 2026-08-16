"""
[IMPLEMENTED] Unit tests for Classical ML models, preprocessing, and metrics logging.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import TabularPreprocessor, load_starter_tabular_dataset
from backend.database.connection import Base
from backend.database.models_orm import Dataset, Experiment, Metric
from backend.evaluation.metrics import (
    evaluate_and_log,
)
from backend.models.classical_models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    build_classical_model,
)


@pytest.fixture
def db_session():
    """Create fresh isolated in-memory DB session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_starter_dataset_loading():
    """Test loading and preprocessing the Wisconsin Diagnostic Breast Cancer dataset."""
    splits = load_starter_tabular_dataset(test_size=0.2, val_size=0.1, random_state=42)
    assert splits.num_samples == 569
    assert splits.num_features == 30
    assert len(splits.feature_names) == 30
    assert splits.target_name == "diagnosis_malignant"

    # Verify split sizes
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == 569
    assert splits.X_train.shape[1] == 30
    assert splits.X_test.shape[1] == 30

    # Test preprocessor independently
    pre = TabularPreprocessor()
    scaled = pre.fit_transform(splits.X_train)
    assert scaled.shape == splits.X_train.shape


@pytest.mark.parametrize("model_type,model_cls", [
    ("logistic_regression", LogisticRegressionModel),
    ("random_forest", RandomForestModel),
    ("xgboost", XGBoostModel)
])
def test_classical_models_fit_and_predict(model_type, model_cls):
    """Test training and inference for each classical model architecture."""
    splits = load_starter_tabular_dataset(test_size=0.2, val_size=0.1, random_state=42)
    model = build_classical_model(model_type=model_type, random_state=42)
    assert isinstance(model, model_cls)

    # Fit model
    metrics = model.fit(splits.X_train, splits.y_train)
    assert "train_accuracy" in metrics
    assert metrics["train_accuracy"] > 0.85

    # Predict hard labels
    preds = model.predict(splits.X_test)
    assert len(preds) == len(splits.y_test)
    assert set(preds).issubset({0, 1})

    # Predict probabilities
    probs = model.predict_proba(splits.X_test)
    assert probs.shape == (len(splits.y_test), 2)


def test_metrics_computation_and_database_persistence(db_session):
    """Test full evaluation pipeline and database persistence of metrics."""
    splits = load_starter_tabular_dataset(test_size=0.2, val_size=0.1, random_state=42)
    model = build_classical_model(model_type="xgboost", random_state=42)
    model.fit(splits.X_train, splits.y_train)

    # 1. Create dataset and experiment in DB
    dataset_rec = Dataset(
        name="breast_cancer_tabular",
        dataset_type="tabular",
        num_samples=splits.num_samples,
        num_features=splits.num_features,
        feature_names=splits.feature_names,
        target_name=splits.target_name
    )
    db_session.add(dataset_rec)
    db_session.commit()

    exp_rec = Experiment(
        name="Centralized Classical Baseline - XGBoost",
        baseline_type="centralized_classical",
        dataset_id=dataset_rec.id,
        status="completed",
        config={"model_type": "xgboost", "n_estimators": 100}
    )
    db_session.add(exp_rec)
    db_session.commit()

    # 2. Evaluate and persist metrics
    eval_metrics = evaluate_and_log(
        db=db_session,
        model=model,
        X_test=splits.X_test,
        y_test=splits.y_test,
        experiment_id=exp_rec.id,
        step=1
    )

    # Assert metric values
    assert eval_metrics["accuracy"] > 0.90
    assert eval_metrics["f1"] > 0.90
    assert eval_metrics["roc_auc"] > 0.95
    assert "confusion_matrix" in eval_metrics

    # 3. Query DB to verify all metrics were recorded
    db_metrics = db_session.query(Metric).filter(Metric.experiment_id == exp_rec.id).all()
    metric_names = {m.metric_name for m in db_metrics}
    assert "accuracy" in metric_names
    assert "f1" in metric_names
    assert "precision" in metric_names
    assert "recall" in metric_names
    assert "roc_auc" in metric_names
