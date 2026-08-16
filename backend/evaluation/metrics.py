from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy.orm import Session

from backend.database.models_orm import Metric
from backend.models.base_model import BaseModelWrapper


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """
    Calculate full suite of classification metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # Use positive class probabilities for binary classification
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            result["roc_auc"] = roc_auc
        except (ValueError, TypeError):
            result["roc_auc"] = None

    return result


def log_metrics_to_db(
    db: Session,
    metrics: dict[str, Any],
    experiment_id: int | None = None,
    round_id: int | None = None,
    job_id: int | None = None,
    step: int = 0
) -> list[Metric]:
    """
    Persist all scalar metrics into the database metrics table.
    """
    persisted_records = []
    for metric_name, val in metrics.items():
        if isinstance(val, (int, float)) and val is not None and not np.isnan(val):
            record = Metric(
                experiment_id=experiment_id,
                round_id=round_id,
                job_id=job_id,
                metric_name=metric_name,
                metric_value=float(val),
                step=step
            )
            db.add(record)
            persisted_records.append(record)

    db.commit()
    return persisted_records


def evaluate_and_log(
    db: Session,
    model: BaseModelWrapper,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_id: int | None = None,
    step: int = 0
) -> dict[str, Any]:
    """
    Run evaluation on test data, compute metrics, and log scalar metrics to the database.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except (AttributeError, ValueError, NotImplementedError):
        y_prob = None

    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    log_metrics_to_db(
        db=db,
        metrics=metrics,
        experiment_id=experiment_id,
        step=step
    )
    return metrics


def evaluate_model_performance(
    model: BaseModelWrapper,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> ClassificationMetrics:
    """
    Evaluate model and return structured ClassificationMetrics dataclass.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except (AttributeError, ValueError, NotImplementedError):
        y_prob = None

    metrics_dict = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    return ClassificationMetrics(
        accuracy=metrics_dict["accuracy"],
        precision=metrics_dict["precision"],
        recall=metrics_dict["recall"],
        f1=metrics_dict["f1"],
        roc_auc=metrics_dict.get("roc_auc"),
        confusion_matrix=metrics_dict["confusion_matrix"]
    )

