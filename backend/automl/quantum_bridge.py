"""
[IMPLEMENTED] AutoML-Quantum Bridge.
Connects the Classical AutoML pipeline with the Quantum Optimization Engine.
Enables toggling optimizer: 'classical' vs 'quantum' across feature selection, client selection, and HPO.
All neural network and ML model training strictly executes on classical CPU/GPU hardware.
"""
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.automl.feature_selector import select_features as classical_select_features
from backend.automl.hpo_classical import optimize_hyperparameters as classical_optimize_hpo
from backend.automl.leaderboard import leaderboard
from backend.evaluation.metrics import evaluate_model_performance
from backend.models.classical_models import build_classical_model
from backend.quantum.client_qubo import build_client_selection_qubo
from backend.quantum.feature_qubo import build_feature_selection_qubo
from backend.quantum.hpo_qubo import build_hpo_qubo
from backend.quantum.quantum_job_manager import quantum_job_manager


class AutoMLQuantumBridge:
    """Orchestrates seamless switching between classical and quantum optimization."""

    @staticmethod
    def select_features(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        k: int = 5,
        optimizer: str = "quantum",
        feature_names: list[str] | None = None,
        max_qubits: int = 16,
        p_layers: int = 1,
        shots: int = 1024,
        db: Session | None = None,
        random_state: int = 42
    ) -> dict[str, Any]:
        """
        Perform feature selection using either Quantum QAOA QUBO or Classical (Mutual Info / RFE).
        """
        optimizer_clean = optimizer.lower().strip()
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

        if optimizer_clean in ["quantum", "qaoa", "qubo"]:
            # 1. Formulate QUBO
            qubo, candidate_indices, _candidate_names = build_feature_selection_qubo(
                X=X_arr,
                y=y_arr,
                k=k,
                max_qubits=max_qubits,
                feature_names=feat_names,
                random_state=random_state
            )

            # 2. Execute Dual Quantum QAOA + Classical Solvers
            job_report = quantum_job_manager.execute_qubo_job(
                qubo=qubo,
                db=db,
                job_type="feature_selection",
                p_layers=p_layers,
                shots=shots,
                classical_method="simulated_annealing"
            )

            # 3. Extract selected feature subset from QAOA bitstring
            best_bits = job_report["qaoa_result"]["best_bitstring"]
            selected_candidate_indices = [i for i, b in enumerate(best_bits) if b == 1]

            # Ensure exactly k features are selected
            if len(selected_candidate_indices) == 0:
                selected_candidate_indices = list(range(min(k, len(candidate_indices))))
            elif len(selected_candidate_indices) > k:
                selected_candidate_indices = selected_candidate_indices[:k]
            elif len(selected_candidate_indices) < k:
                remaining = [i for i in range(len(candidate_indices)) if i not in selected_candidate_indices]
                selected_candidate_indices.extend(remaining[:(k - len(selected_candidate_indices))])

            selected_orig_indices = [candidate_indices[i] for i in selected_candidate_indices]
            selected_feature_names = [feat_names[idx] for idx in selected_orig_indices]

            return {
                "optimizer": "quantum_qaoa",
                "k": len(selected_orig_indices),
                "selected_indices": selected_orig_indices,
                "selected_features": selected_feature_names,
                "quantum_job_id": job_report.get("job_id"),
                "telemetry": job_report
            }
        else:
            # Classical Feature Selection
            res = classical_select_features(
                X=X_arr,
                y=y_arr,
                method="mutual_info",
                k=k,
                feature_names=feat_names,
                random_state=random_state
            )
            return {
                "optimizer": "classical_mutual_info",
                "k": res["k"],
                "selected_indices": res["selected_indices"],
                "selected_features": res["selected_features"],
                "feature_scores": res.get("feature_scores", {}),
                "quantum_job_id": None
            }

    @staticmethod
    def select_clients(
        clients_data: list[dict[str, Any]],
        k: int = 3,
        optimizer: str = "quantum",
        p_layers: int = 1,
        shots: int = 1024,
        db: Session | None = None
    ) -> dict[str, Any]:
        """
        Select edge client participants for a federated round via Quantum QUBO or Classical heuristic.
        """
        optimizer_clean = optimizer.lower().strip()
        n = len(clients_data)
        k_eff = min(k, n)

        if optimizer_clean in ["quantum", "qaoa", "qubo"]:
            qubo, client_ids = build_client_selection_qubo(clients_data, k=k_eff)
            job_report = quantum_job_manager.execute_qubo_job(
                qubo=qubo,
                db=db,
                job_type="client_selection",
                p_layers=p_layers,
                shots=shots,
                classical_method="simulated_annealing"
            )
            best_bits = job_report["qaoa_result"]["best_bitstring"]
            selected_indices = [i for i, b in enumerate(best_bits) if b == 1]

            if not selected_indices:
                selected_indices = list(range(k_eff))
            elif len(selected_indices) > k_eff:
                selected_indices = selected_indices[:k_eff]

            selected_client_ids = [client_ids[i] for i in selected_indices]
            return {
                "optimizer": "quantum_qaoa",
                "k": len(selected_client_ids),
                "selected_client_ids": selected_client_ids,
                "quantum_job_id": job_report.get("job_id"),
                "telemetry": job_report
            }
        else:
            # Classical ranking by data quality score
            sorted_clients = sorted(clients_data, key=lambda c: float(c.get("data_quality_score", 0.5)), reverse=True)
            selected_client_ids = [str(c.get("client_id")) for c in sorted_clients[:k_eff]]
            return {
                "optimizer": "classical_heuristic",
                "k": len(selected_client_ids),
                "selected_client_ids": selected_client_ids,
                "quantum_job_id": None
            }

    @staticmethod
    def select_hyperparameters(
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimizer: str = "quantum",
        candidate_configs: list[dict[str, Any]] | None = None,
        p_layers: int = 1,
        shots: int = 1024,
        db: Session | None = None,
        random_state: int = 42
    ) -> dict[str, Any]:
        """
        Hyperparameter selection via Quantum QUBO 1-hot selection or Classical Optuna.
        """
        optimizer_clean = optimizer.lower().strip()

        if optimizer_clean in ["quantum", "qaoa", "qubo"]:
            configs = candidate_configs or [
                {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
                {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.1},
                {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.15},
                {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.2}
            ]

            # Fast proxy evaluation to create prior scores
            scores = []
            for cfg in configs:
                m = build_classical_model(model_type=model_type, random_state=random_state, **cfg)
                m.fit(X_train[:min(100, len(X_train))], y_train[:min(100, len(y_train))])
                perf = evaluate_model_performance(m, X_val, y_val)
                scores.append(float(perf.roc_auc if perf.roc_auc is not None else perf.accuracy))

            # Build and solve 1-hot HPO QUBO
            qubo, _ = build_hpo_qubo(configs, estimated_scores=scores)
            job_report = quantum_job_manager.execute_qubo_job(
                qubo=qubo,
                db=db,
                job_type="hpo_selection",
                p_layers=p_layers,
                shots=shots
            )

            best_bits = job_report["qaoa_result"]["best_bitstring"]
            chosen_idx = int(np.argmax(best_bits)) if any(best_bits) else int(np.argmax(scores))
            best_config = configs[chosen_idx]

            # Final classical training with selected hyperparameters
            final_model = build_classical_model(model_type=model_type, random_state=random_state, **best_config)
            final_model.fit(X_train, y_train)
            val_metrics = evaluate_model_performance(final_model, X_val, y_val)

            return {
                "optimizer": "quantum_qaoa_hpo",
                "best_hyperparameters": best_config,
                "validation_accuracy": val_metrics.accuracy,
                "validation_roc_auc": val_metrics.roc_auc,
                "validation_f1": val_metrics.f1,
                "quantum_job_id": job_report.get("job_id"),
                "telemetry": job_report,
                "model": final_model
            }
        else:
            # Classical Optuna HPO
            optuna_res = classical_optimize_hpo(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=10,
                sampler_type="tpe",
                random_state=random_state
            )
            return {
                "optimizer": "classical_optuna_tpe",
                "best_hyperparameters": optuna_res["best_params"],
                "best_validation_score": optuna_res["best_validation_score"],
                "quantum_job_id": None,
                "model": optuna_res["best_model"]
            }

    @classmethod
    def run_full_automl_pipeline(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = "xgboost",
        k_features: int = 8,
        feature_optimizer: str = "quantum",
        hpo_optimizer: str = "classical",
        feature_names: list[str] | None = None,
        db: Session | None = None,
        random_state: int = 42
    ) -> dict[str, Any]:
        """
        Execute an automated machine learning workflow combining feature selection,
        downstream model training, and leaderboard recording.
        """
        # 1. Feature Selection
        feat_res = cls.select_features(
            X=X_train,
            y=y_train,
            k=k_features,
            optimizer=feature_optimizer,
            feature_names=feature_names,
            db=db,
            random_state=random_state
        )
        selected_cols = feat_res["selected_indices"]
        X_train_sub = X_train[:, selected_cols]
        X_val_sub = X_val[:, selected_cols]

        # 2. Hyperparameter Optimization & Model Training
        hpo_res = cls.select_hyperparameters(
            model_type=model_type,
            X_train=X_train_sub,
            y_train=y_train,
            X_val=X_val_sub,
            y_val=y_val,
            optimizer=hpo_optimizer,
            db=db,
            random_state=random_state
        )

        model = hpo_res["model"]
        metrics = evaluate_model_performance(model, X_val_sub, y_val)

        # 3. Add to Leaderboard
        leaderboard.add_candidate(
            model_name=f"{model_type.upper()}-[{feature_optimizer.upper()}-FS]",
            hyperparameters=hpo_res.get("best_hyperparameters", {}),
            validation_metrics={
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc
            },
            search_method=f"FS:{feature_optimizer}_HPO:{hpo_optimizer}",
            feature_set=f"{len(selected_cols)}_features"
        )

        return {
            "status": "completed",
            "model_type": model_type,
            "feature_selection": feat_res,
            "hpo_results": hpo_res,
            "validation_metrics": {
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "precision": metrics.precision,
                "recall": metrics.recall
            }
        }


quantum_bridge = AutoMLQuantumBridge()
