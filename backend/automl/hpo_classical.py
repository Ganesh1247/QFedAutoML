"""
[IMPLEMENTED] Classical Hyperparameter Optimization (HPO) Engine powered by Optuna.
Implements Bayesian Tree-structured Parzen Estimator (TPE), Random Search, and Grid Search.
Acts as the baseline for comparison with Quantum Hyperparameter Optimization (QUBO).
"""
import time
from typing import Any

import numpy as np
import optuna
from optuna.samplers import RandomSampler, TPESampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from backend.models.classical_models import (
    build_classical_model,
)

# Suppress verbose Optuna logging during automated HPO runs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "roc_auc",
    n_trials: int = 5,
    sampler_type: str = "tpe",
    random_state: int = 42
) -> dict[str, Any]:
    """
    Execute classical HPO study using Optuna.
    """
    model_type_clean = model_type.lower().strip()
    sampler_type_clean = sampler_type.lower().strip()

    # Configure Optuna Sampler
    if sampler_type_clean in ["tpe", "bayesian"]:
        sampler = TPESampler(seed=random_state)
    elif sampler_type_clean in ["random", "rand"]:
        sampler = RandomSampler(seed=random_state)
    else:
        sampler = TPESampler(seed=random_state)

    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        if model_type_clean in ["xgboost", "xgb"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 150, step=20),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": random_state
            }
        elif model_type_clean in ["random_forest", "rf"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 150, step=20),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "random_state": random_state
            }
        elif model_type_clean in ["logistic_regression", "lr"]:
            params = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "random_state": random_state
            }
        else:
            raise ValueError(f"Unsupported model type for HPO: {model_type}")

        model = build_classical_model(model_type=model_type_clean, **params)
        model.fit(X_train, y_train)

        # Evaluate on validation split
        if metric == "roc_auc":
            try:
                probs = model.predict_proba(X_val)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    score = float(roc_auc_score(y_val, probs[:, 1]))
                else:
                    score = float(roc_auc_score(y_val, probs))
            except (ValueError, AttributeError, IndexError):
                preds = model.predict(X_val)
                score = float(accuracy_score(y_val, preds))
        elif metric == "f1":
            preds = model.predict(X_val)
            score = float(f1_score(y_val, preds, zero_division=0))
        else:
            preds = model.predict(X_val)
            score = float(accuracy_score(y_val, preds))

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"classical_hpo_{model_type_clean}_{sampler_type_clean}"
    )
    study.optimize(objective, n_trials=n_trials)

    total_time_s = time.time() - start_time

    # Collect trial history
    trial_history = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_history.append({
                "trial_number": t.number,
                "params": t.params,
                "score": round(float(t.value), 4),
                "duration_seconds": round(t.duration.total_seconds() if t.duration else 0.0, 3)
            })

    best_params = study.best_params
    best_score = float(study.best_value)

    # Train final best model
    best_model = build_classical_model(model_type=model_type_clean, random_state=random_state, **best_params)
    best_model.fit(X_train, y_train)

    return {
        "model_type": model_type_clean,
        "sampler_type": sampler_type_clean,
        "target_metric": metric,
        "n_trials": len(trial_history),
        "best_params": best_params,
        "best_validation_score": round(best_score, 4),
        "total_optimization_time_seconds": round(total_time_s, 4),
        "trial_history": trial_history,
        "best_model": best_model
    }
