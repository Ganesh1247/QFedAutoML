"""
[IMPLEMENTED] Model Architecture Selector.
Screens candidate model families using Stratified K-Fold Cross-Validation.
"""
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from backend.models.classical_models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)


def screen_candidate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_models: list[str] | None = None,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42
) -> list[dict[str, Any]]:
    """
    Screen candidate model families on training data via stratified cross validation.
    """
    candidates = candidate_models or ["xgboost", "random_forest", "logistic_regression"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results = []
    for model_name in candidates:
        name_clean = model_name.lower().strip()
        if name_clean in ["xgboost", "xgb"]:
            estimator = XGBoostModel(n_estimators=50, max_depth=3, random_state=random_state).model
        elif name_clean in ["random_forest", "rf"]:
            estimator = RandomForestModel(n_estimators=50, max_depth=4, random_state=random_state).model
        elif name_clean in ["logistic_regression", "lr"]:
            estimator = LogisticRegressionModel(C=1.0, random_state=random_state).model
        else:
            continue

        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring)
        results.append({
            "model_type": name_clean,
            "mean_cv_score": round(float(np.mean(scores)), 4),
            "std_cv_score": round(float(np.std(scores)), 4),
            "scoring_metric": scoring,
            "cv_folds": cv_folds
        })

    # Sort descending by mean CV score
    results.sort(key=lambda x: x["mean_cv_score"], reverse=True)
    for rank, res in enumerate(results, start=1):
        res["rank"] = rank

    return results
