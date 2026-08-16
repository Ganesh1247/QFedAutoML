"""
[IMPLEMENTED] Unit tests for Comparative Benchmark Runner.
"""
import json
import os

import pytest

from backend.automl.preprocessing import (
    load_sensor_timeseries_dataset,
    load_starter_tabular_dataset,
)
from backend.evaluation.benchmark_runner import ComparativeBenchmarkRunner


@pytest.fixture
def benchmark_runner():
    return ComparativeBenchmarkRunner(random_state=42)


@pytest.fixture
def tabular_splits():
    return load_starter_tabular_dataset()


@pytest.fixture
def seq_splits():
    return load_sensor_timeseries_dataset(num_samples=40, seq_len=6, num_features=4)


def test_baseline_1_centralized(benchmark_runner, tabular_splits):
    res = benchmark_runner.run_baseline_1_centralized(tabular_splits)
    assert res["paradigm"] == "Centralized"
    assert "accuracy" in res
    assert res["accuracy"] >= 0.85
    assert res["roc_auc"] >= 0.85
    assert res["communication_overhead_mb"] == 0.0


def test_baseline_2_federated(benchmark_runner, tabular_splits):
    res = benchmark_runner.run_baseline_2_federated(tabular_splits, num_clients=2, num_rounds=2)
    assert res["paradigm"] == "Decentralized Federated"
    assert "accuracy" in res
    assert res["accuracy"] >= 0.80
    assert res["communication_overhead_mb"] > 0


def test_baseline_3_transformer(benchmark_runner, seq_splits):
    res = benchmark_runner.run_baseline_3_transformer(seq_splits, num_clients=2, num_rounds=1)
    assert res["paradigm"] == "Decentralized Sequence FL"
    assert "accuracy" in res
    assert res["accuracy"] >= 0.50
    assert res["avg_inference_latency_ms"] > 0


def test_baseline_4_qfedautoml(benchmark_runner, tabular_splits):
    res = benchmark_runner.run_baseline_4_qfedautoml(tabular_splits, k_features=4)
    assert res["paradigm"] == "Quantum-Enhanced Federated AutoML"
    assert res["feature_count"] == 4
    assert res["accuracy"] >= 0.85
    assert res["epsilon_spent"] == 1.42


def test_run_all_benchmarks(benchmark_runner, tmp_path):
    json_path = str(tmp_path / "benchmarks.json")
    suite = benchmark_runner.run_all_benchmarks(export_json_path=json_path)

    assert suite["total_baselines"] == 4
    assert len(suite["baselines"]) == 4
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_baselines"] == 4
    assert "statistical_summary" in loaded
