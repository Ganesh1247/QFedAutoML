"""
[IMPLEMENTED] Local Interpretable Model-agnostic Explanations (LIME) Engine.
Generates local surrogate linear explanations with exponential kernel sample weighting.
Works seamlessly across tabular models and multi-sensor features.
"""
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


class LIMEExplainer:
    """Computes local surrogate linear models to explain single predictions."""

    @classmethod
    def explain_instance(
        cls,
        model: Any,
        instance: np.ndarray,
        training_data: np.ndarray,
        feature_names: list[str] | None = None,
        num_samples: int = 200,
        kernel_width: float | None = None,
        random_state: int = 42
    ) -> dict[str, Any]:
        """
        Fit a locally weighted Ridge regression surrogate around a given instance.
        """
        inst = np.asarray(instance).ravel()
        n_features = len(inst)
        feat_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        train_arr = np.asarray(training_data)

        # 1. Compute empirical feature standard deviations
        std_devs = np.std(train_arr, axis=0)
        std_devs[std_devs == 0] = 1.0

        # 2. Perturbation sampling around the instance
        rng = np.random.default_rng(random_state)
        # Sample in standardized space
        noise = rng.normal(loc=0.0, scale=1.0, size=(num_samples, n_features))
        perturbed_samples = inst + noise * std_devs
        # Ensure the instance itself is the first sample
        perturbed_samples[0] = inst

        # 3. Compute Euclidean distance in standardized space and exponential kernel weights
        scaled_diff = (perturbed_samples - inst) / std_devs
        distances = np.linalg.norm(scaled_diff, axis=1)

        width = kernel_width or np.sqrt(n_features) * 0.75
        weights = np.exp(-(distances ** 2) / (2.0 * (width ** 2)))

        # 4. Query Blackbox Model predictions on perturbed samples
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(perturbed_samples)
            y_target = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
            inst_pred_prob = float(probs[0, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs[0])
        else:
            preds = model.predict(perturbed_samples)
            y_target = preds.ravel().astype(np.float64)
            inst_pred_prob = float(preds[0])

        # 5. Fit Weighted Ridge Linear Surrogate
        surrogate = Ridge(alpha=1.0, fit_intercept=True, random_state=random_state)
        surrogate.fit(scaled_diff, y_target, sample_weight=weights)

        r2_fidelity = float(surrogate.score(scaled_diff, y_target, sample_weight=weights))
        coefs = surrogate.coef_

        contributions = [
            {
                "feature": feat_names[i],
                "feature_index": i,
                "feature_value": round(float(inst[i]), 4),
                "weight": round(float(coefs[i]), 6),
                "direction": "supports_positive_class" if coefs[i] >= 0 else "supports_negative_class"
            }
            for i in range(n_features)
        ]
        # Sort by magnitude of local weight
        contributions.sort(key=lambda c: abs(c["weight"]), reverse=True)

        return {
            "explainer": "LIME_LocalSurrogate",
            "prediction": round(inst_pred_prob, 4),
            "surrogate_intercept": round(float(surrogate.intercept_), 4),
            "surrogate_fidelity_r2": round(max(0.0, r2_fidelity), 4),
            "num_perturbations": num_samples,
            "feature_contributions": contributions
        }


lime_explainer = LIMEExplainer()
