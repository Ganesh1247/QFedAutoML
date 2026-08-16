"""
[IMPLEMENTED] Federated Threat and Model Poisoning Detector.
Inspects client model parameter submissions before server aggregation.
Detects Byzantine adversaries, sign-flipping attacks, and gradient explosion anomalies.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ThreatInspectionResult:
    accepted_client_ids: list[str]
    rejected_client_ids: list[str]
    anomaly_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_total: int = 0
    num_accepted: int = 0
    num_rejected: int = 0


def flatten_parameters(parameters: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1D vector."""
    return np.concatenate([p.ravel().astype(np.float64) for p in parameters])


class ThreatDetector:
    """Multi-layer Byzantine and anomaly filter for federated learning."""

    def __init__(
        self,
        min_cosine_similarity: float = -0.1,
        max_norm_multiplier: float = 3.5,
        absolute_max_norm: float = 50.0
    ):
        self.min_cosine_similarity = min_cosine_similarity
        self.max_norm_multiplier = max_norm_multiplier
        self.absolute_max_norm = absolute_max_norm

    def inspect_client_updates(
        self,
        client_updates: list[tuple[str, list[np.ndarray]]],
        baseline_parameters: list[np.ndarray] | None = None
    ) -> ThreatInspectionResult:
        """
        Inspect client submissions against norm and cosine similarity thresholds.

        Parameters:
            client_updates: list of tuples (client_id, parameter_arrays)
            baseline_parameters: optional server global model parameters before round
        """
        if not client_updates:
            return ThreatInspectionResult([], [])

        client_ids = [cid for cid, _ in client_updates]
        n_clients = len(client_ids)

        # 1. Flatten all client weight vectors or differentials
        flat_vectors = []
        norms = []
        for _, params in client_updates:
            if baseline_parameters is not None:
                # Differential update delta_w = w_client - w_global
                diff_params = [p - b for p, b in zip(params, baseline_parameters, strict=True)]
                vec = flatten_parameters(diff_params)
            else:
                vec = flatten_parameters(params)

            norm_val = float(np.linalg.norm(vec))
            flat_vectors.append(vec)
            norms.append(norm_val)

        # Compute median norm and reference median vector across all clients
        median_norm = float(np.median(norms)) if norms else 1.0
        stacked_vectors = np.stack(flat_vectors, axis=0)  # (N, D)
        median_reference_vector = np.median(stacked_vectors, axis=0)
        median_ref_norm = float(np.linalg.norm(median_reference_vector)) + 1e-9

        accepted_ids = []
        rejected_ids = []
        anomaly_reports = {}

        # 2. Evaluate each client
        for i, cid in enumerate(client_ids):
            vec = flat_vectors[i]
            norm_i = norms[i]
            reasons = []

            # Check 1: Norm Anomaly / Gradient Explosion
            norm_threshold = min(self.absolute_max_norm, max(1.0, median_norm * self.max_norm_multiplier))
            if norm_i > norm_threshold:
                reasons.append(f"Gradient explosion: norm {norm_i:.3f} exceeds threshold {norm_threshold:.3f}")

            # Check 2: Cosine Similarity Alignment with Consensus (if N >= 3)
            if n_clients >= 3 and norm_i > 1e-7:
                cos_sim = float(np.dot(vec, median_reference_vector) / (norm_i * median_ref_norm))
                if cos_sim < self.min_cosine_similarity:
                    reasons.append(f"Poisoning / Sign-flip: cosine similarity {cos_sim:.3f} < {self.min_cosine_similarity}")
            else:
                cos_sim = 1.0

            if reasons:
                rejected_ids.append(cid)
                anomaly_reports[cid] = {
                    "client_id": cid,
                    "reasons": reasons,
                    "l2_norm": round(norm_i, 4),
                    "cosine_similarity": round(cos_sim, 4),
                    "threat_level": "HIGH" if len(reasons) > 1 else "MEDIUM"
                }
            else:
                accepted_ids.append(cid)

        return ThreatInspectionResult(
            accepted_client_ids=accepted_ids,
            rejected_client_ids=rejected_ids,
            anomaly_reports=anomaly_reports,
            num_total=n_clients,
            num_accepted=len(accepted_ids),
            num_rejected=len(rejected_ids)
        )


threat_detector = ThreatDetector()
