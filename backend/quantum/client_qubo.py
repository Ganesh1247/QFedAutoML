"""
[IMPLEMENTED] Quantum Client Selection formulated as a QUBO problem.
Selects an optimal subset of edge client nodes optimizing data quality, communication overhead, and diversity.
"""
from typing import Any

import numpy as np

from backend.quantum.qubo_builder import QUBOProblem


def build_client_selection_qubo(
    clients_data: list[dict[str, Any]],
    k: int = 3,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    delta: float = 0.2
) -> tuple[QUBOProblem, list[str]]:
    """
    Formulate Federated Client Selection as a QUBO problem:
    E(x) = -alpha * sum(q_i * x_i) + beta * sum(c_i * x_i) + gamma * (sum(x_i) - k)^2 + delta * sum(overlap_ij * x_i * x_j)

    Parameters:
        clients_data: list of dicts with keys:
            - client_id (str)
            - data_quality_score (float, 0..1, e.g. based on sample count or local loss)
            - comm_cost_score (float, 0..1, e.g. latency/bandwidth)
        k: target number of clients to participate in the round.
    """
    n = len(clients_data)
    client_ids = [str(c.get("client_id", f"client_{i}")) for i, c in enumerate(clients_data)]

    # Extract normalized quality and communication costs
    q = np.array([float(c.get("data_quality_score", 0.5)) for c in clients_data])
    c_cost = np.array([float(c.get("comm_cost_score", 0.5)) for c in clients_data])

    Q = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Diagonal term
        Q[i, i] = -alpha * q[i] + beta * c_cost[i] + gamma * (1.0 - 2.0 * k)

        for j in range(i + 1, n):
            # Synthetic overlap / redundancy penalty
            overlap_ij = 0.5 * (q[i] * q[j])
            term = gamma + 0.5 * delta * overlap_ij
            Q[i, j] = term
            Q[j, i] = term

    constant_offset = gamma * (k ** 2)

    qubo = QUBOProblem(
        Q=Q,
        variable_names=client_ids,
        problem_type="client_selection",
        constant_offset=constant_offset
    )

    return qubo, client_ids
