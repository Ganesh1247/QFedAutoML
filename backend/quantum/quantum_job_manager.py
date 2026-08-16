"""
[IMPLEMENTED] Quantum Job Manager and Telemetry Logger.
Orchestrates dual execution of Quantum (QAOA) and Classical (Simulated Annealing/Exact) solvers
to provide honest, side-by-side performance benchmarking without hardcoded bias.
"""
from typing import Any

from sqlalchemy.orm import Session

from backend.database.repositories.quantum_job_repo import quantum_job_repo
from backend.quantum.classical_fallback import solve_qubo_classical
from backend.quantum.qaoa_optimizer import solve_qubo_qaoa
from backend.quantum.qubo_builder import QUBOProblem


class QuantumJobManager:
    """Manages execution and telemetry recording for quantum optimization jobs."""

    @staticmethod
    def execute_qubo_job(
        qubo: QUBOProblem,
        db: Session | None = None,
        job_type: str = "feature_selection",
        p_layers: int = 1,
        shots: int = 1024,
        classical_method: str = "simulated_annealing",
        experiment_id: int | None = None
    ) -> dict[str, Any]:
        """
        Execute QUBO on both Quantum QAOA simulator and Classical solver,
        compare results, and persist to database.
        """
        # 1. Run Quantum QAOA
        qaoa_result = solve_qubo_qaoa(qubo, p=p_layers, shots=shots)

        # 2. Run Classical Counterpart
        classical_result = solve_qubo_classical(qubo, method=classical_method)

        # 3. Compute Comparison Metrics
        q_energy = float(qaoa_result["best_energy"])
        c_energy = float(classical_result["best_energy"])
        energy_gap = round(abs(q_energy - c_energy), 6)

        q_runtime = float(qaoa_result["execution_time_seconds"])
        c_runtime = float(classical_result["execution_time_seconds"])

        comparison = {
            "qaoa_energy": q_energy,
            "classical_energy": c_energy,
            "energy_gap": energy_gap,
            "qaoa_runtime_seconds": q_runtime,
            "classical_runtime_seconds": c_runtime,
            "solutions_match": qaoa_result["best_bitstring"] == classical_result["best_bitstring"],
            "qaoa_qubit_count": qaoa_result["num_qubits"],
            "qaoa_circuit_depth": qaoa_result["circuit_depth"]
        }

        job_record = None
        if db is not None:
            job_record = quantum_job_repo.create(
                db=db,
                job_type=job_type,
                backend_used="qiskit_aer",
                num_qubits=qaoa_result["num_qubits"],
                circuit_depth=qaoa_result["circuit_depth"],
                parameters={"p_layers": p_layers, "shots": shots, "problem_type": qubo.problem_type}
            )
            quantum_job_repo.finish(
                db=db,
                job_id=job_record.id,
                objective_value=q_energy,
                classical_objective_value=c_energy,
                execution_time_ms=q_runtime * 1000.0,
                classical_time_ms=c_runtime * 1000.0,
                result={
                    "qaoa": qaoa_result,
                    "classical": classical_result,
                    "comparison": comparison
                },
                status="completed"
            )

        return {
            "job_id": job_record.id if job_record else None,
            "problem_type": qubo.problem_type,
            "num_variables": qubo.num_variables,
            "qaoa_result": qaoa_result,
            "classical_result": classical_result,
            "comparison": comparison
        }


quantum_job_manager = QuantumJobManager()
