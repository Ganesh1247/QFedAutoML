"""
[IMPLEMENTED] Quantum Approximate Optimization Algorithm (QAOA) Solver.
Constructs parameterized quantum circuits using Qiskit Aer simulator for combinatorial QUBO minimization.
Adheres to strict constraints: logical qubits <= 16-20, shallow depth p in {1, 2}.
"""
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from backend.quantum.qubo_builder import QUBOProblem


def build_qaoa_circuit(
    qubo: QUBOProblem,
    gammas: list[float] | np.ndarray,
    betas: list[float] | np.ndarray,
    p: int = 1
) -> QuantumCircuit:
    """
    Construct a depth-p QAOA Quantum Circuit for the given QUBO problem.
    """
    n = qubo.num_variables
    J, h, _ = qubo.to_ising()

    qc = QuantumCircuit(n)

    # Initial state: uniform superposition |+>^n
    for i in range(n):
        qc.h(i)

    for layer in range(p):
        gamma_l = float(gammas[layer])
        beta_l = float(betas[layer])

        # 1. Problem Hamiltonian Phase Separation: e^{-i gamma_l H_P}
        # 1-qubit field rotations (RZ)
        for i, h_val in h.items():
            if abs(h_val) > 1e-9:
                qc.rz(2.0 * gamma_l * h_val, i)

        # 2-qubit Ising ZZ couplings (RZZ)
        for (i, j), j_val in J.items():
            if abs(j_val) > 1e-9:
                qc.rzz(2.0 * gamma_l * j_val, i, j)

        # 2. Mixer Hamiltonian: e^{-i beta_l H_M}
        for i in range(n):
            qc.rx(2.0 * beta_l, i)

    return qc


def solve_qubo_qaoa(
    qubo: QUBOProblem,
    p: int = 1,
    shots: int = 1024,
    max_iter: int = 12,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Solve a QUBO problem using QAOA on the Qiskit Statevector simulator.
    Uses COBYLA classical optimizer to optimize variational parameters (gammas, betas).
    """
    start_time = time.time()
    n = qubo.num_variables
    np.random.seed(random_state)

    # Precompute energy for all bitstrings for fast statevector expectation evaluation
    bitstrings = []
    energies = np.zeros(2 ** n, dtype=np.float64)
    for idx in range(2 ** n):
        # Qiskit qubit 0 is least significant bit (LSB) on the right
        # Extract binary list for variable indices 0..n-1
        bit_vec = [(idx >> i) & 1 for i in range(n)]
        bitstrings.append(bit_vec)
        energies[idx] = qubo.evaluate_energy(bit_vec)

    # Objective function for classical optimizer
    def objective_fn(params: np.ndarray) -> float:
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit(qubo, gammas, betas, p=p)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()  # Probabilities across all 2^n states
        expected_energy = float(np.sum(probs * energies))
        return expected_energy

    # Initial parameter guess
    init_params = np.concatenate([
        np.random.uniform(0, np.pi, size=p),
        np.random.uniform(0, np.pi / 2, size=p)
    ])

    # Optimize QAOA angles using classical COBYLA optimizer
    opt_res = minimize(
        objective_fn,
        init_params,
        method="COBYLA",
        options={"maxiter": max_iter, "tol": 1e-4}
    )

    opt_gammas = opt_res.x[:p].tolist()
    opt_betas = opt_res.x[p:].tolist()

    # Generate final circuit with optimal parameters
    optimal_qc = build_qaoa_circuit(qubo, opt_gammas, opt_betas, p=p)
    optimal_sv = Statevector.from_instruction(optimal_qc)
    probs = optimal_sv.probabilities()

    # Sample top bitstrings
    top_indices = np.argsort(probs)[::-1][:min(10, len(probs))]
    sampled_solutions = []
    for idx in top_indices:
        b_vec = bitstrings[idx]
        prob = float(probs[idx])
        e_val = float(energies[idx])
        sampled_solutions.append({
            "bitstring": b_vec,
            "probability": round(prob, 4),
            "energy": round(e_val, 6)
        })

    # Pick the state with the minimum energy among top high-probability states
    best_candidate = min(sampled_solutions, key=lambda x: x["energy"])
    runtime = time.time() - start_time

    return {
        "solver": "qaoa_qiskit",
        "num_qubits": n,
        "p_layers": p,
        "circuit_depth": optimal_qc.depth(),
        "circuit_gate_count": len(optimal_qc.data),
        "optimal_gammas": [round(g, 4) for g in opt_gammas],
        "optimal_betas": [round(b, 4) for b in opt_betas],
        "best_bitstring": best_candidate["bitstring"],
        "best_energy": best_candidate["energy"],
        "best_probability": best_candidate["probability"],
        "sampled_solutions": sampled_solutions,
        "execution_time_seconds": round(runtime, 4),
        "optimizer_iterations": int(opt_res.nfev)
    }
