"""
[IMPLEMENTED] Classical QUBO Solvers: Simulated Annealing and Exact Brute-Force.
Serves as the mandatory, honest classical baseline to benchmark QAOA quantum execution.
"""
import itertools
import math
import random
import time
from typing import Any

import numpy as np

from backend.quantum.qubo_builder import QUBOProblem


def solve_qubo_exact(qubo: QUBOProblem) -> dict[str, Any]:
    """
    Exhaustively evaluate all 2^n binary configurations to find the guaranteed global minimum.
    Restricted to n <= 18 variables.
    """
    n = qubo.num_variables
    if n > 18:
        raise ValueError(f"Exact solver feasible for n <= 18, got {n}")

    start_time = time.time()
    best_energy = float("inf")
    best_bitstring = [0] * n

    for bit_tuple in itertools.product([0, 1], repeat=n):
        bit_vec = list(bit_tuple)
        energy = qubo.evaluate_energy(bit_vec)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bit_vec

    runtime = time.time() - start_time
    return {
        "solver": "exact_brute_force",
        "best_bitstring": best_bitstring,
        "best_energy": round(best_energy, 6),
        "execution_time_seconds": round(runtime, 6),
        "num_evaluations": 2 ** n
    }


def solve_qubo_simulated_annealing(
    qubo: QUBOProblem,
    num_reads: int = 30,
    max_steps: int = 400,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.96,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Simulated Annealing classical heuristic solver for QUBO minimization.
    Metropolis-Hastings acceptance: P(accept) = exp(-delta_E / T) for uphill moves.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    n = qubo.num_variables

    start_time = time.time()
    global_best_energy = float("inf")
    global_best_bitstring = [0] * n

    for _ in range(num_reads):
        # Random initial state
        current_state = [random.choice([0, 1]) for _ in range(n)]
        current_energy = qubo.evaluate_energy(current_state)

        if current_energy < global_best_energy:
            global_best_energy = current_energy
            global_best_bitstring = list(current_state)

        t = initial_temp
        for _ in range(max_steps):
            # Propose flipping a random bit
            flip_idx = random.randint(0, n - 1)
            candidate_state = list(current_state)
            candidate_state[flip_idx] = 1 - candidate_state[flip_idx]

            candidate_energy = qubo.evaluate_energy(candidate_state)
            delta_e = candidate_energy - current_energy

            # Metropolis acceptance criterion
            if delta_e < 0.0 or random.random() < math.exp(-delta_e / max(t, 1e-8)):
                current_state = candidate_state
                current_energy = candidate_energy

                if current_energy < global_best_energy:
                    global_best_energy = current_energy
                    global_best_bitstring = list(current_state)

            t *= cooling_rate

    runtime = time.time() - start_time
    return {
        "solver": "simulated_annealing",
        "best_bitstring": global_best_bitstring,
        "best_energy": round(global_best_energy, 6),
        "execution_time_seconds": round(runtime, 6),
        "num_reads": num_reads,
        "max_steps": max_steps
    }


def solve_qubo_classical(
    qubo: QUBOProblem,
    method: str = "simulated_annealing",
    num_reads: int = 30,
    max_steps: int = 400,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Unified classical solver dispatcher.
    """
    method_clean = method.lower().strip()
    if method_clean in ["exact", "brute_force", "exact_brute_force"] and qubo.num_variables <= 16:
        return solve_qubo_exact(qubo)
    else:
        return solve_qubo_simulated_annealing(
            qubo=qubo,
            num_reads=num_reads,
            max_steps=max_steps,
            random_state=random_state
        )
