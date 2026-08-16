"""
[IMPLEMENTED] Classical Feature Selection Engine.
Implements Mutual Information, Recursive Feature Elimination (RFE), and L1-Lasso selection.
Serves as the classical benchmark counterpart to Quantum Feature Selection (QUBO).
"""
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression


def select_features_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    feature_names: list[str] | None = None,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Select top-k features maximizing mutual information with target y.
    """
    feat_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    scores = mutual_info_classif(X, y, random_state=random_state)

    # Top-k indices
    k = min(k, X.shape[1])
    top_indices = np.argsort(scores)[::-1][:k].tolist()
    selected_features = [feat_names[i] for i in top_indices]

    feature_scores = {feat_names[i]: round(float(scores[i]), 4) for i in top_indices}

    return {
        "method": "mutual_info",
        "k": k,
        "selected_indices": top_indices,
        "selected_features": selected_features,
        "feature_scores": feature_scores
    }


def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    feature_names: list[str] | None = None,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Select top-k features using Recursive Feature Elimination with Random Forest.
    """
    feat_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    k = min(k, X.shape[1])

    estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit(X, y)

    selected_indices = np.where(selector.support_)[0].tolist()
    # Sort selected indices by feature importance
    importances = selector.estimator_.feature_importances_
    sorted_order = np.argsort(importances)[::-1]
    sorted_selected_indices = [selected_indices[i] for i in sorted_order]
    selected_features = [feat_names[i] for i in sorted_selected_indices]

    feature_scores = {feat_names[i]: round(float(importances[idx]), 4) for idx, i in enumerate(sorted_selected_indices)}

    return {
        "method": "rfe",
        "k": k,
        "selected_indices": sorted_selected_indices,
        "selected_features": selected_features,
        "feature_scores": feature_scores
    }


def select_features_l1(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    feature_names: list[str] | None = None,
    C: float = 0.1,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Select top-k features based on L1-regularized sparse Logistic Regression coefficients.
    """
    feat_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    k = min(k, X.shape[1])

    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=random_state)
    clf.fit(X, y)

    coef_magnitudes = np.abs(clf.coef_).ravel()
    top_indices = np.argsort(coef_magnitudes)[::-1][:k].tolist()
    selected_features = [feat_names[i] for i in top_indices]
    feature_scores = {feat_names[i]: round(float(coef_magnitudes[i]), 4) for i in top_indices}

    return {
        "method": "l1_lasso",
        "k": k,
        "selected_indices": top_indices,
        "selected_features": selected_features,
        "feature_scores": feature_scores
    }


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "mutual_info",
    k: int = 10,
    feature_names: list[str] | None = None,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Unified feature selection dispatcher.
    """
    method_clean = method.lower().strip()
    if method_clean in ["mutual_info", "mi"]:
        return select_features_mutual_info(X, y, k=k, feature_names=feature_names, random_state=random_state)
    elif method_clean in ["rfe", "recursive_feature_elimination"]:
        return select_features_rfe(X, y, k=k, feature_names=feature_names, random_state=random_state)
    elif method_clean in ["l1", "lasso", "l1_lasso"]:
        return select_features_l1(X, y, k=k, feature_names=feature_names, random_state=random_state)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
