"""
[IMPLEMENTED] Classical Machine Learning model wrappers (XGBoost, Random Forest, Logistic Regression).
All model training strictly executes on classical CPU/GPU hardware.
"""
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from backend.models.base_model import BaseModelWrapper


class LogisticRegressionModel(BaseModelWrapper):
    """Classical Logistic Regression classifier wrapper."""

    def __init__(self, C: float = 1.0, max_iter: int = 200, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, float]:
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = float(self.model.score(X, y))
        return {"train_accuracy": train_acc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_weights(self) -> dict[str, Any]:
        """Extract coefficients and intercept for federated averaging."""
        if not self.is_fitted:
            return {}
        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy()
        }

    def set_weights(self, weights: dict[str, Any]) -> None:
        """Set coefficients and intercept."""
        if "coef" in weights and "intercept" in weights:
            self.model.coef_ = np.array(weights["coef"])
            self.model.intercept_ = np.array(weights["intercept"])
            self.is_fitted = True


class RandomForestModel(BaseModelWrapper):
    """Classical Random Forest classifier wrapper."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 6,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, float]:
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = float(self.model.score(X, y))
        return {"train_accuracy": train_acc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_weights(self) -> dict[str, Any]:
        return {"feature_importances": self.model.feature_importances_ if self.is_fitted else []}

    def set_weights(self, weights: Any) -> None:
        pass


class XGBoostModel(BaseModelWrapper):
    """Classical XGBoost Gradient Boosted Trees wrapper."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, float]:
        self.model.fit(X, y)
        self.is_fitted = True
        train_acc = float(self.model.score(X, y))
        return {"train_accuracy": train_acc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_weights(self) -> dict[str, Any]:
        return {"feature_importances": self.model.feature_importances_ if self.is_fitted else []}

    def set_weights(self, weights: Any) -> None:
        pass


def build_classical_model(
    model_type: str = "xgboost",
    **hyperparams
) -> BaseModelWrapper:
    """Factory function for classical tabular model instantiation."""
    model_type_clean = model_type.lower().strip()
    if model_type_clean in ["xgboost", "xgb"]:
        return XGBoostModel(**hyperparams)
    elif model_type_clean in ["random_forest", "rf"]:
        return RandomForestModel(**hyperparams)
    elif model_type_clean in ["logistic_regression", "lr"]:
        return LogisticRegressionModel(**hyperparams)
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")
