"""
[IMPLEMENTED] QUBO (Quadratic Unconstrained Binary Optimization) and Ising Hamiltonian Builder.
Formulates combinatorial subproblems as mathematical minimization objectives: min x^T Q x.
"""
from typing import Any

import numpy as np


class QUBOProblem:
    """
    Represents a Quadratic Unconstrained Binary Optimization problem:
    Objective: E(x) = x^T Q x + offset, where x in {0, 1}^n.
    """

    def __init__(
        self,
        Q: np.ndarray,
        variable_names: list[str] | None = None,
        problem_type: str = "custom",
        constant_offset: float = 0.0
    ):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.num_variables = self.Q.shape[0]
        if self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"QUBO matrix Q must be square, got shape {self.Q.shape}")

        self.variable_names = variable_names or [f"x_{i}" for i in range(self.num_variables)]
        self.problem_type = problem_type
        self.constant_offset = float(constant_offset)

    def evaluate_energy(self, x: np.ndarray | list[int] | list[float]) -> float:
        """Compute the cost/energy for a binary configuration vector x."""
        x_vec = np.asarray(x, dtype=np.float64).ravel()
        if len(x_vec) != self.num_variables:
            raise ValueError(f"Vector length {len(x_vec)} does not match variable count {self.num_variables}")
        energy = float(x_vec @ self.Q @ x_vec + self.constant_offset)
        return energy

    def to_ising(self) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
        """
        Convert QUBO binary variables x_i in {0, 1} to Ising spin variables s_i in {-1, +1}
        Transformation: x_i = (1 - s_i) / 2.
        Returns:
            J: dictionary of 2-body coupling coefficients J[(i, j)] for s_i * s_j
            h: dictionary of 1-body local fields h[i] for s_i
            offset: scalar energy offset
        """
        n = self.num_variables
        # Symmetrize Q
        Q_sym = 0.5 * (self.Q + self.Q.T)

        J: dict[tuple[int, int], float] = {}
        h: dict[int, float] = {}
        offset: float = self.constant_offset

        # Linear and constant contributions from diagonal and off-diagonal
        for i in range(n):
            # Diagonal term Q_ii * x_i = Q_ii * (1 - s_i) / 2
            q_ii = Q_sym[i, i]
            offset += 0.5 * q_ii
            h[i] = h.get(i, 0.0) - 0.5 * q_ii

            for j in range(i + 1, n):
                q_ij = 2.0 * Q_sym[i, j]  # combined (i,j) and (j,i)
                # q_ij * x_i * x_j = q_ij * (1 - s_i - s_j + s_i * s_j) / 4
                offset += 0.25 * q_ij
                h[i] = h.get(i, 0.0) - 0.25 * q_ij
                h[j] = h.get(j, 0.0) - 0.25 * q_ij
                J[(i, j)] = 0.25 * q_ij

        return J, h, offset

    def to_dict(self) -> dict[str, Any]:
        """Serialize QUBO structure for logging and API telemetry."""
        return {
            "problem_type": self.problem_type,
            "num_variables": self.num_variables,
            "variable_names": self.variable_names,
            "constant_offset": self.constant_offset,
            "matrix_shape": list(self.Q.shape),
            "density": float(np.count_nonzero(self.Q) / (self.num_variables ** 2))
        }
