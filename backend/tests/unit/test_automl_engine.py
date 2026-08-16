"""
[IMPLEMENTED] Unit tests for Classical AutoML Engine:
Dataset Profiler, Feature Selectors, Model Selector, Optuna HPO, and Leaderboard.
"""
import pytest

from backend.automl.dataset_profiler import profile_dataset
from backend.automl.feature_selector import (
    select_features,
    select_features_l1,
    select_features_mutual_info,
    select_features_rfe,
)
from backend.automl.hpo_classical import optimize_hyperparameters
from backend.automl.leaderboard import AutoMLLeaderboard
from backend.automl.model_selector import screen_candidate_models
from backend.automl.preprocessing import load_starter_tabular_dataset


@pytest.fixture
def dataset_data():
    return load_starter_tabular_dataset(test_size=0.2, val_size=0.1, random_state=42)


def test_dataset_profiler(dataset_data):
    """Verify dataset profiler extracts correct dimensions, stats, and correlations."""
    profile = profile_dataset(
        dataset_data.X_train,
        dataset_data.y_train,
        feature_names=dataset_data.feature_names
    )

    assert profile["num_samples"] == len(dataset_data.X_train)
    assert profile["num_features"] == 30
    assert len(profile["feature_names"]) == 30
    assert 0 in profile["class_distribution"]
    assert 1 in profile["class_distribution"]
    assert profile["imbalance_ratio"] >= 1.0

    # Verify per-feature stats
    first_feat = dataset_data.feature_names[0]
    assert "mean" in profile["feature_statistics"][first_feat]
    assert "std" in profile["feature_statistics"][first_feat]
    assert "skewness" in profile["feature_statistics"][first_feat]
    assert isinstance(profile["high_correlation_pairs"], list)


@pytest.mark.parametrize("method_fn", [
    select_features_mutual_info,
    select_features_rfe,
    select_features_l1
])
def test_individual_feature_selectors(dataset_data, method_fn):
    """Verify each feature selection algorithm selects exactly k features."""
    k = 8
    res = method_fn(
        X=dataset_data.X_train,
        y=dataset_data.y_train,
        k=k,
        feature_names=dataset_data.feature_names,
        random_state=42
    )
    assert res["k"] == k
    assert len(res["selected_indices"]) == k
    assert len(res["selected_features"]) == k
    assert len(res["feature_scores"]) == k


def test_unified_feature_selector_dispatcher(dataset_data):
    """Verify unified feature selector interface."""
    res_mi = select_features(dataset_data.X_train, dataset_data.y_train, method="mutual_info", k=5)
    assert res_mi["k"] == 5
    assert len(res_mi["selected_indices"]) == 5

    res_rfe = select_features(dataset_data.X_train, dataset_data.y_train, method="rfe", k=5)
    assert res_rfe["k"] == 5

    with pytest.raises(ValueError, match="Unknown feature selection method"):
        select_features(dataset_data.X_train, dataset_data.y_train, method="invalid_method")


def test_model_selector_cross_validation(dataset_data):
    """Verify cross-validation screening across model families."""
    cv_results = screen_candidate_models(
        X_train=dataset_data.X_train,
        y_train=dataset_data.y_train,
        candidate_models=["xgboost", "random_forest", "logistic_regression"],
        cv_folds=3,
        scoring="roc_auc",
        random_state=42
    )

    assert len(cv_results) == 3
    assert cv_results[0]["rank"] == 1
    assert cv_results[0]["mean_cv_score"] > 0.90
    assert cv_results[0]["std_cv_score"] >= 0.0


def test_optuna_classical_hpo(dataset_data):
    """Verify Optuna HPO runs trials, optimizes hyperparameters, and returns best model."""
    hpo_results = optimize_hyperparameters(
        model_type="xgboost",
        X_train=dataset_data.X_train,
        y_train=dataset_data.y_train,
        X_val=dataset_data.X_val,
        y_val=dataset_data.y_val,
        metric="roc_auc",
        n_trials=5,
        sampler_type="tpe",
        random_state=42
    )

    assert hpo_results["model_type"] == "xgboost"
    assert hpo_results["n_trials"] == 5
    assert "n_estimators" in hpo_results["best_params"]
    assert "max_depth" in hpo_results["best_params"]
    assert hpo_results["best_validation_score"] > 0.90
    assert len(hpo_results["trial_history"]) == 5
    assert hpo_results["best_model"] is not None


def test_automl_leaderboard():
    """Verify leaderboard ranking and sorting."""
    lb = AutoMLLeaderboard()

    lb.add_candidate(
        model_name="XGBoost-Optuna-TPE",
        hyperparameters={"n_estimators": 100, "max_depth": 4},
        validation_metrics={"accuracy": 0.965, "f1": 0.960, "roc_auc": 0.992},
        search_method="optuna_tpe",
        execution_time_s=1.25
    )

    lb.add_candidate(
        model_name="RandomForest-Default",
        hyperparameters={"n_estimators": 50},
        validation_metrics={"accuracy": 0.940, "f1": 0.935, "roc_auc": 0.978},
        search_method="default",
        execution_time_s=0.45
    )

    lb.add_candidate(
        model_name="LogisticRegression-L1",
        hyperparameters={"C": 0.1},
        validation_metrics={"accuracy": 0.920, "f1": 0.915, "roc_auc": 0.965},
        search_method="l1_search",
        execution_time_s=0.10
    )

    ranked = lb.get_leaderboard(sort_by="roc_auc")
    assert len(ranked) == 3
    assert ranked[0]["model_name"] == "XGBoost-Optuna-TPE"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3

    best = lb.get_best_candidate(metric="roc_auc")
    assert best["model_name"] == "XGBoost-Optuna-TPE"

    df = lb.to_dataframe()
    assert len(df) == 3
    assert "roc_auc" in df.columns
