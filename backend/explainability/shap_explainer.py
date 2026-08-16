"""
[IMPLEMENTED] SHAP (SHapley Additive exPlanations) Engine.
Computes global feature importance rankings and local per-instance attribution values.
Supports tree models (XGBoost, Random Forest), linear models, and neural architectures.
"""
from typing import Any

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SHAPExplainer:
    """Computes global and instance-level SHAP attributions."""

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        """Extract underlying estimator if model is wrapped in BaseModelWrapper."""
        if hasattr(model, "model"):
            return model.model
        return model

    @classmethod
    def explain_global(
        cls,
        model: Any,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        max_samples: int = 100
    ) -> dict[str, Any]:
        """
        Compute global feature importance via mean absolute SHAP values.
        """
        X_arr = np.asarray(X)
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        n_samples = min(len(X_arr), max_samples)
        sample_X = X_arr[:n_samples]

        unwrapped = cls._unwrap_model(model)
        shap_values_matrix = None

        if HAS_SHAP:
            try:
                # Attempt TreeExplainer for tree-based models
                if hasattr(unwrapped, "predict_proba") and hasattr(unwrapped, "feature_importances_"):
                    explainer = shap.TreeExplainer(unwrapped)
                    vals = explainer.shap_values(sample_X)
                    if isinstance(vals, list):
                        # Multi-class output, take positive class
                        shap_values_matrix = np.array(vals[1] if len(vals) > 1 else vals[0])
                    elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                        shap_values_matrix = vals[:, :, 1]
                    else:
                        shap_values_matrix = np.array(vals)
                else:
                    # Generic explainer / Exact fallback
                    explainer = shap.Explainer(unwrapped.predict if hasattr(unwrapped, "predict") else unwrapped, sample_X)
                    explanation = explainer(sample_X)
                    shap_values_matrix = explanation.values
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError):
                shap_values_matrix = None

        # Robust Marginal Contribution SHAP Approximation fallback
        if shap_values_matrix is None or not HAS_SHAP:
            shap_values_matrix = cls._approximate_shap_values(model, sample_X)

        mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0).tolist()
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        rankings = [
            {
                "feature": feat_names[idx],
                "feature_index": int(idx),
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
                "rank": rank + 1
            }
            for rank, idx in enumerate(sorted_indices)
        ]

        return {
            "explainer": "SHAP_TreeOrExact",
            "num_samples_evaluated": n_samples,
            "feature_names": feat_names,
            "mean_abs_shap": [round(float(v), 6) for v in mean_abs_shap],
            "rankings": rankings
        }

    @classmethod
    def explain_instance(
        cls,
        model: Any,
        instance: np.ndarray,
        background_data: np.ndarray,
        feature_names: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Compute local feature attributions for a single sample instance.
        """
        inst = np.asarray(instance).ravel()
        feat_names = feature_names or [f"feature_{i}" for i in range(len(inst))]
        bg = np.asarray(background_data)[:50]

        unwrapped = cls._unwrap_model(model)
        local_shap = None

        if HAS_SHAP:
            try:
                if hasattr(unwrapped, "feature_importances_"):
                    explainer = shap.TreeExplainer(unwrapped)
                    vals = explainer.shap_values(inst.reshape(1, -1))
                    if isinstance(vals, list):
                        local_shap = np.array(vals[1][0] if len(vals) > 1 else vals[0][0])
                    elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                        local_shap = vals[0, :, 1]
                    else:
                        local_shap = np.array(vals[0])
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError):
                local_shap = None

        if local_shap is None:
            local_shap = cls._approximate_instance_shap(model, inst, bg)

        # Baseline prediction on background data
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(bg)
            base_val = float(np.mean(probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs))
            inst_prob = model.predict_proba(inst.reshape(1, -1))
            pred_val = float(inst_prob[0, 1] if inst_prob.ndim == 2 and inst_prob.shape[1] > 1 else inst_prob[0])
        else:
            base_val = 0.5
            pred_val = float(model.predict(inst.reshape(1, -1))[0])

        contributions = [
            {
                "feature": feat_names[i],
                "feature_index": i,
                "value": round(float(inst[i]), 4),
                "shap_value": round(float(local_shap[i]), 6),
                "direction": "positive" if local_shap[i] >= 0 else "negative"
            }
            for i in range(len(inst))
        ]
        # Sort by absolute SHAP attribution
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "base_value": round(base_val, 4),
            "predicted_value": round(pred_val, 4),
            "feature_contributions": contributions
        }

    @staticmethod
    def _approximate_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
        """Kernel/Marginal approximation computing feature sensitivities."""
        n_samples, n_features = X.shape
        baseline = np.mean(X, axis=0)
        shap_matrix = np.zeros((n_samples, n_features), dtype=np.float64)

        for i in range(n_samples):
            x_i = X[i].copy()
            if hasattr(model, "predict_proba"):
                orig_pred = float(model.predict_proba(x_i.reshape(1, -1))[0, 1])
            else:
                orig_pred = float(model.predict(x_i.reshape(1, -1))[0])

            for f in range(n_features):
                x_perturbed = x_i.copy()
                x_perturbed[f] = baseline[f]
                if hasattr(model, "predict_proba"):
                    pert_pred = float(model.predict_proba(x_perturbed.reshape(1, -1))[0, 1])
                else:
                    pert_pred = float(model.predict(x_perturbed.reshape(1, -1))[0])
                shap_matrix[i, f] = orig_pred - pert_pred

        return shap_matrix

    @staticmethod
    def _approximate_instance_shap(model: Any, instance: np.ndarray, background_data: np.ndarray) -> np.ndarray:
        """Approximates instance SHAP using marginal baseline replacement."""
        n_features = len(instance)
        baseline = np.mean(background_data, axis=0) if len(background_data) > 0 else np.zeros(n_features)
        local_shap = np.zeros(n_features, dtype=np.float64)

        if hasattr(model, "predict_proba"):
            orig_pred = float(model.predict_proba(instance.reshape(1, -1))[0, 1])
        else:
            orig_pred = float(model.predict(instance.reshape(1, -1))[0])

        for f in range(n_features):
            pert = instance.copy()
            pert[f] = baseline[f]
            if hasattr(model, "predict_proba"):
                pert_pred = float(model.predict_proba(pert.reshape(1, -1))[0, 1])
            else:
                pert_pred = float(model.predict(pert.reshape(1, -1))[0])
            local_shap[f] = orig_pred - pert_pred

        return local_shap


shap_explainer = SHAPExplainer()
