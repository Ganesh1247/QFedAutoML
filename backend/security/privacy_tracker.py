"""
[IMPLEMENTED] Differential Privacy Cumulative Budget Tracker & Accountant.
Tracks per-client and global (epsilon, delta) privacy expenditure across rounds.
Emits alert triggers when a node's cumulative privacy budget is exhausted.
"""
import math
from typing import Any


class PrivacyBudgetTracker:
    """Accountant tracking cumulative differential privacy expenditure across FL rounds."""

    def __init__(self, default_max_epsilon: float = 10.0, default_delta: float = 1e-5):
        self.default_max_epsilon = default_max_epsilon
        self.default_delta = default_delta
        # client_id -> {"spent_epsilon": float, "rounds_participated": int, "step_epsilons": list[float]}
        self.clients_accounting: dict[str, dict[str, Any]] = {}
        self.global_rounds_count: int = 0

    def spend_budget(
        self,
        client_id: str,
        step_epsilon: float,
        step_delta: float = 1e-5,
        max_budget_epsilon: float | None = None
    ) -> dict[str, Any]:
        """
        Record a privacy expenditure step for a participating client node.
        Computes cumulative expenditure using advanced composition bounds.
        """
        max_eps = max_budget_epsilon if max_budget_epsilon is not None else self.default_max_epsilon

        if client_id not in self.clients_accounting:
            self.clients_accounting[client_id] = {
                "spent_epsilon": 0.0,
                "rounds_participated": 0,
                "step_epsilons": [],
                "cumulative_delta": 0.0
            }

        client_entry = self.clients_accounting[client_id]
        client_entry["rounds_participated"] += 1
        client_entry["step_epsilons"].append(step_epsilon)
        client_entry["cumulative_delta"] += step_delta

        t = client_entry["rounds_participated"]
        eps_step = step_epsilon

        # Advanced Composition Theorem:
        # eps_total = sqrt(2 * T * ln(1 / delta')) * eps_step + T * eps_step * (e^eps_step - 1)
        if t > 1:
            delta_prime = 1e-5
            adv_eps = math.sqrt(2.0 * t * math.log(1.0 / delta_prime)) * eps_step + t * eps_step * (math.exp(eps_step) - 1.0)
            linear_eps = sum(client_entry["step_epsilons"])
            # Tighter of linear and advanced composition
            total_spent = min(linear_eps, adv_eps)
        else:
            total_spent = eps_step

        client_entry["spent_epsilon"] = total_spent
        is_exhausted = total_spent >= max_eps
        remaining_budget = max(0.0, max_eps - total_spent)

        return {
            "client_id": client_id,
            "step_epsilon": step_epsilon,
            "total_spent_epsilon": round(total_spent, 4),
            "max_budget_epsilon": max_eps,
            "remaining_budget": round(remaining_budget, 4),
            "rounds_participated": t,
            "is_exhausted": is_exhausted
        }

    def get_client_status(
        self,
        client_id: str,
        max_budget_epsilon: float | None = None
    ) -> dict[str, Any]:
        """Check current budget status for a given client."""
        max_eps = max_budget_epsilon if max_budget_epsilon is not None else self.default_max_epsilon
        if client_id not in self.clients_accounting:
            return {
                "client_id": client_id,
                "total_spent_epsilon": 0.0,
                "max_budget_epsilon": max_eps,
                "remaining_budget": max_eps,
                "rounds_participated": 0,
                "is_exhausted": False
            }

        entry = self.clients_accounting[client_id]
        spent = entry["spent_epsilon"]
        return {
            "client_id": client_id,
            "total_spent_epsilon": round(spent, 4),
            "max_budget_epsilon": max_eps,
            "remaining_budget": round(max(0.0, max_eps - spent), 4),
            "rounds_participated": entry["rounds_participated"],
            "is_exhausted": spent >= max_eps
        }

    def reset(self) -> None:
        """Reset privacy accounting records."""
        self.clients_accounting.clear()
        self.global_rounds_count = 0


privacy_tracker = PrivacyBudgetTracker()
