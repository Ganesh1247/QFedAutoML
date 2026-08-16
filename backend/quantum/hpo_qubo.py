"""
[IMPLEMENTED] Quantum Hyperparameter Optimization formulated as a QUBO problem.
Discretizes candidate hyperparameter sets and imposes a 1-hot constraint via quadratic penalty.
"""
from typing import Any

import numpy as np

from backend.quantum.qubo_builder import QUBOProblem


def build_hpo_qubo(
    candidate_configs: list[dict[str, Any]],
    estimated_scores: list[float] | None = None,
    gamma: float = 5.0
) -> tuple[QUBOProblem, list[dict[str, Any]]]:
    """
    Formulate discretized hyperparameter selection as a 1-in-M QUBO problem:
    E(x) = -sum(s_i * x_i) + gamma * (sum(x_i) - 1)^2

    Parameters:
        candidate_configs: list of hyperparameter configuration dicts.
        estimated_scores: prior surrogate performance estimates in [0, 1].
    """
    m = len(candidate_configs)
    if estimated_scores is None:
        # Default uniform prior
        scores = np.ones(m, dtype=np.float64) / m
    else:
        scores = np.asarray(estimated_scores, dtype=np.float64)
        if len(scores) != m:
            raise ValueError(f"Estimated scores length {len(scores)} must match candidates {m}")

    config_names = [f"config_{i}" for i in range(m)]
    Q = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        # Diagonal term: -s_i + gamma * (1 - 2*1) = -s_i - gamma
        Q[i, i] = -scores[i] - gamma

        for j in range(i + 1, m):
            # Off-diagonal penalty enforcing sum(x_i) == 1
            Q[i, j] = gamma
            Q[j, i] = gamma

    constant_offset = gamma * 1.0

    qubo = QUBOProblem(
        Q=Q,
        variable_names=config_names,
        problem_type="hpo_selection",
        constant_offset=constant_offset
    )

    return qubo, candidate_configs
