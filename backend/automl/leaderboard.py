"""
[IMPLEMENTED] AutoML Leaderboard.
Ranks candidate model architectures, hyperparameter configurations, and feature sets.
"""
from typing import Any

import pandas as pd


class AutoMLLeaderboard:
    """Manages candidate evaluation ranking across AutoML optimization runs."""

    def __init__(self):
        self.candidates: list[dict[str, Any]] = []

    def add_candidate(
        self,
        model_name: str,
        hyperparameters: dict[str, Any],
        validation_metrics: dict[str, Any],
        search_method: str = "optuna_tpe",
        feature_set: str = "all_features",
        execution_time_s: float = 0.0
    ) -> dict[str, Any]:
        """Record an evaluated candidate configuration."""
        candidate_entry = {
            "model_name": model_name,
            "search_method": search_method,
            "feature_set": feature_set,
            "accuracy": round(float(validation_metrics.get("accuracy", 0.0)), 4),
            "f1": round(float(validation_metrics.get("f1", 0.0)), 4),
            "roc_auc": round(float(validation_metrics.get("roc_auc", 0.0)) if validation_metrics.get("roc_auc") is not None else 0.0, 4),
            "hyperparameters": hyperparameters,
            "execution_time_s": round(execution_time_s, 4)
        }
        self.candidates.append(candidate_entry)
        return candidate_entry

    def get_leaderboard(self, sort_by: str = "roc_auc", ascending: bool = False) -> list[dict[str, Any]]:
        """Return all candidates ranked by the specified validation metric."""
        if not self.candidates:
            return []

        sorted_candidates = sorted(
            self.candidates,
            key=lambda x: x.get(sort_by, 0.0),
            reverse=not ascending
        )

        ranked_list = []
        for rank, item in enumerate(sorted_candidates, start=1):
            entry = item.copy()
            entry["rank"] = rank
            ranked_list.append(entry)

        return ranked_list

    def get_best_candidate(self, metric: str = "roc_auc") -> dict[str, Any] | None:
        """Get the highest performing configuration."""
        leaderboard = self.get_leaderboard(sort_by=metric, ascending=False)
        return leaderboard[0] if leaderboard else None

    def to_dataframe(self, sort_by: str = "roc_auc") -> pd.DataFrame:
        """Export leaderboard as a pandas DataFrame for reporting."""
        leaderboard = self.get_leaderboard(sort_by=sort_by)
        return pd.DataFrame(leaderboard)

    def clear(self) -> None:
        """Reset the leaderboard."""
        self.candidates.clear()


leaderboard = AutoMLLeaderboard()
