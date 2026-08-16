"""
[IMPLEMENTED] Unit and integration tests for Quantum Optimization Engine:
QUBO builder, Feature/Client/HPO QUBOs, QAOA circuit optimizer, Classical fallback, and Job Manager.
"""
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.automl.preprocessing import load_starter_tabular_dataset
from backend.database.connection import Base
from backend.database.models_orm import QuantumJob
from backend.quantum.classical_fallback import (
    solve_qubo_exact,
    solve_qubo_simulated_annealing,
)
from backend.quantum.client_qubo import build_client_selection_qubo
from backend.quantum.feature_qubo import build_feature_selection_qubo
from backend.quantum.hpo_qubo import build_hpo_qubo
from backend.quantum.qaoa_optimizer import build_qaoa_circuit, solve_qubo_qaoa
from backend.quantum.quantum_job_manager import quantum_job_manager
from backend.quantum.qubo_builder import QUBOProblem

# In-memory test database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_qubo_problem_and_ising_conversion():
    """Verify QUBO matrix energy evaluation and Ising Hamiltonian conversion equivalence."""
    Q = np.array([
        [-2.0, 1.5, 0.0],
        [1.5, -1.0, 2.0],
        [0.0, 2.0, -3.0]
    ])
    qubo = QUBOProblem(Q=Q, variable_names=["x0", "x1", "x2"], constant_offset=5.0)

    assert qubo.num_variables == 3
    # Evaluate energy for x = [1, 0, 1]
    # E(x) = Q_00*1 + Q_22*1 + 2*Q_02*(0) + offset = -2.0 + -3.0 + 5.0 = 0.0
    x_test = [1, 0, 1]
    e_qubo = qubo.evaluate_energy(x_test)
    assert np.isclose(e_qubo, 0.0)

    # Convert to Ising
    J, h, offset = qubo.to_ising()
    assert isinstance(J, dict)
    assert isinstance(h, dict)

    # Spin configuration for x = [1, 0, 1] is s = [-1, +1, -1] (s_i = 1 - 2*x_i)
    s_test = {0: -1.0, 1: 1.0, 2: -1.0}
    ising_energy = offset + sum(h[i] * s_test[i] for i in h) + sum(J[(i, j)] * s_test[i] * s_test[j] for (i, j) in J)

    # Verify Ising and QUBO energies match exactly
    assert np.isclose(e_qubo, ising_energy)


def test_feature_selection_qubo_generation():
    """Verify feature selection QUBO respects qubit budget and problem structure."""
    splits = load_starter_tabular_dataset()
    k = 4
    max_qubits = 8

    qubo, candidate_indices, candidate_names = build_feature_selection_qubo(
        X=splits.X_train,
        y=splits.y_train,
        k=k,
        max_qubits=max_qubits,
        feature_names=splits.feature_names,
        random_state=42
    )

    assert qubo.num_variables == max_qubits
    assert len(candidate_indices) == max_qubits
    assert len(candidate_names) == max_qubits
    assert qubo.problem_type == "feature_selection"
    assert qubo.constant_offset > 0.0


def test_client_selection_qubo():
    """Verify federated client selection QUBO formulation."""
    clients_data = [
        {"client_id": "client_1", "data_quality_score": 0.9, "comm_cost_score": 0.2},
        {"client_id": "client_2", "data_quality_score": 0.7, "comm_cost_score": 0.4},
        {"client_id": "client_3", "data_quality_score": 0.8, "comm_cost_score": 0.3},
        {"client_id": "client_4", "data_quality_score": 0.4, "comm_cost_score": 0.8},
    ]
    qubo, client_ids = build_client_selection_qubo(clients_data, k=2)

    assert qubo.num_variables == 4
    assert client_ids == ["client_1", "client_2", "client_3", "client_4"]
    assert qubo.problem_type == "client_selection"


def test_hpo_qubo():
    """Verify HPO 1-hot candidate selection QUBO."""
    candidates = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 150, "max_depth": 6}
    ]
    scores = [0.85, 0.93, 0.91]
    qubo, configs = build_hpo_qubo(candidates, estimated_scores=scores, gamma=4.0)

    assert qubo.num_variables == 3
    assert len(configs) == 3

    # Choosing exactly config_1 (the highest scoring) should produce lower energy than picking 2 configs
    e_single = qubo.evaluate_energy([0, 1, 0])
    e_double = qubo.evaluate_energy([0, 1, 1])  # Violates 1-hot penalty
    assert e_single < e_double


def test_classical_solvers():
    """Verify Exact and Simulated Annealing classical QUBO solvers."""
    Q = np.array([
        [-5.0, 3.0, 2.0],
        [3.0, -4.0, 1.0],
        [2.0, 1.0, -6.0]
    ])
    qubo = QUBOProblem(Q=Q, constant_offset=1.0)

    # 1. Exact solver
    exact_res = solve_qubo_exact(qubo)
    assert exact_res["solver"] == "exact_brute_force"
    assert len(exact_res["best_bitstring"]) == 3
    assert exact_res["num_evaluations"] == 8

    # 2. Simulated Annealing solver
    sa_res = solve_qubo_simulated_annealing(qubo, num_reads=20, max_steps=200)
    assert sa_res["solver"] == "simulated_annealing"
    assert len(sa_res["best_bitstring"]) == 3
    # Both should find the global optimum on a 3-variable problem
    assert exact_res["best_bitstring"] == sa_res["best_bitstring"]


def test_qaoa_qiskit_optimizer():
    """Verify QAOA parameterized quantum circuit generation and Qiskit simulator optimization."""
    Q = np.array([
        [-2.0, 1.0, 0.0],
        [1.0, -3.0, 1.5],
        [0.0, 1.5, -2.5]
    ])
    qubo = QUBOProblem(Q=Q, constant_offset=2.0)

    # Circuit construction test
    qc = build_qaoa_circuit(qubo, gammas=[0.5], betas=[0.3], p=1)
    assert qc.num_qubits == 3
    assert qc.depth() > 0

    # Optimization test
    qaoa_res = solve_qubo_qaoa(qubo, p=1, shots=512, max_iter=25)
    assert qaoa_res["solver"] == "qaoa_qiskit"
    assert qaoa_res["num_qubits"] == 3
    assert qaoa_res["circuit_depth"] > 0
    assert len(qaoa_res["best_bitstring"]) == 3
    assert len(qaoa_res["sampled_solutions"]) > 0
    assert qaoa_res["execution_time_seconds"] > 0.0


def test_quantum_job_manager_dual_execution():
    """Verify QuantumJobManager runs both solvers, persists to database, and records telemetry comparison."""
    db = TestingSessionLocal()
    Q = np.array([
        [-3.0, 2.0, 0.0, 1.0],
        [2.0, -4.0, 1.0, 0.0],
        [0.0, 1.0, -5.0, 2.0],
        [1.0, 0.0, 2.0, -2.0]
    ])
    qubo = QUBOProblem(Q=Q, problem_type="feature_selection", constant_offset=0.0)

    job_report = quantum_job_manager.execute_qubo_job(
        qubo=qubo,
        db=db,
        job_type="feature_selection",
        p_layers=1,
        classical_method="exact"
    )

    assert job_report["job_id"] is not None
    assert job_report["num_variables"] == 4
    assert "qaoa_result" in job_report
    assert "classical_result" in job_report
    assert "comparison" in job_report

    comp = job_report["comparison"]
    assert "energy_gap" in comp
    assert "qaoa_runtime_seconds" in comp
    assert "classical_runtime_seconds" in comp

    # Verify DB persistence
    db_job = db.query(QuantumJob).filter(QuantumJob.id == job_report["job_id"]).first()
    assert db_job is not None
    assert db_job.num_qubits == 4
    assert db_job.circuit_depth > 0
    assert db_job.status == "completed"
    db.close()
