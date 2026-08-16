"""
[IMPLEMENTED] Quantum Feature Selection formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem.
Balances feature relevance, inter-feature redundancy, and exact cardinality budget k.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from backend.quantum.qubo_builder import QUBOProblem


def build_feature_selection_qubo(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    k: int = 5,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    max_qubits: int = 16,
    feature_names: list[str] | None = None,
    random_state: int = 42
) -> tuple[QUBOProblem, list[int], list[str]]:
    """
    Formulate feature selection as a QUBO minimization problem:
    E(x) = -alpha * sum(rel_i * x_i) + beta * sum(corr_ij * x_i * x_j) + gamma * (sum(x_i) - k)^2

    Returns:
        qubo: QUBOProblem instance ready for QAOA or classical solvers.
        candidate_indices: original column indices corresponding to the QUBO variables.
        candidate_feature_names: names of candidate features.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
        feat_names = list(X.columns)
    else:
        X_arr = np.asarray(X)
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y).ravel()
    total_features = X_arr.shape[1]

    # Calculate mutual information relevance
    relevance_scores = mutual_info_classif(X_arr, y_arr, random_state=random_state)
    # Normalize relevance scores to [0, 1]
    max_rel = np.max(relevance_scores) if np.max(relevance_scores) > 1e-9 else 1.0
    norm_relevance = relevance_scores / max_rel

    # Keep candidate pool within logical qubit budget (max_qubits <= 16)
    pool_size = min(total_features, max_qubits)
    top_candidate_indices = np.argsort(relevance_scores)[::-1][:pool_size].tolist()
    candidate_names = [feat_names[i] for i in top_candidate_indices]

    # Compute correlation sub-matrix for candidate features
    X_candidate = X_arr[:, top_candidate_indices]
    df_candidate = pd.DataFrame(X_candidate)
    corr_matrix = df_candidate.corr().abs().fillna(0.0).to_numpy()

    # Build Q matrix (pool_size x pool_size)
    Q = np.zeros((pool_size, pool_size), dtype=np.float64)

    for i in range(pool_size):
        # Diagonal term: -alpha * rel_i + gamma * (1 - 2k)
        rel_i = norm_relevance[top_candidate_indices[i]]
        Q[i, i] = -alpha * rel_i + gamma * (1.0 - 2.0 * k)

        for j in range(i + 1, pool_size):
            corr_ij = corr_matrix[i, j]
            # Off-diagonal symmetric term: 0.5 * beta * corr_ij + gamma
            term = 0.5 * beta * corr_ij + gamma
            Q[i, j] = term
            Q[j, i] = term

    constant_offset = gamma * (k ** 2)

    qubo = QUBOProblem(
        Q=Q,
        variable_names=candidate_names,
        problem_type="feature_selection",
        constant_offset=constant_offset
    )

    return qubo, top_candidate_indices, candidate_names
