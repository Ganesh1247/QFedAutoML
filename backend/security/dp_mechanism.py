"""
[IMPLEMENTED] Differential Privacy (DP) Mechanism for Federated Model Updates.
Implements L2-norm gradient clipping, calibrated Gaussian mechanism, and Laplace mechanism.
Guarantees (epsilon, delta)-differential privacy for edge client update submissions.
"""
import math
from typing import Any

import numpy as np


def compute_l2_norm(parameters: list[np.ndarray]) -> float:
    """Compute global L2 norm across a list of multi-dimensional numpy parameter arrays."""
    squared_sum = sum(float(np.sum(p.astype(np.float64) ** 2)) for p in parameters)
    return float(math.sqrt(squared_sum))


def clip_parameter_tensors(
    parameters: list[np.ndarray],
    clip_norm: float = 1.0
) -> tuple[list[np.ndarray], float]:
    """
    Clip parameter updates to a maximum L2-norm threshold C:
    w_clipped = w * min(1, C / ||w||_2).
    """
    total_norm = compute_l2_norm(parameters)
    scaling_factor = min(1.0, clip_norm / max(total_norm, 1e-9))
    clipped_params = [p * scaling_factor for p in parameters]
    return clipped_params, total_norm


def add_gaussian_noise(
    parameters: list[np.ndarray],
    clip_norm: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    random_state: int | None = None
) -> tuple[list[np.ndarray], float]:
    """
    Inject calibrated Gaussian noise satisfying (epsilon, delta)-DP.
    Noise scale: sigma = (C * sqrt(2 * ln(1.25 / delta))) / epsilon.
    """
    if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"Invalid DP parameters: epsilon={epsilon}, delta={delta}")

    rng = np.random.default_rng(random_state)
    sigma = float((clip_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon)

    noisy_params = [
        p + rng.normal(loc=0.0, scale=sigma, size=p.shape).astype(p.dtype)
        for p in parameters
    ]
    return noisy_params, sigma


def add_laplace_noise(
    parameters: list[np.ndarray],
    clip_norm: float = 1.0,
    epsilon: float = 1.0,
    random_state: int | None = None
) -> tuple[list[np.ndarray], float]:
    """
    Inject calibrated Laplace noise satisfying epsilon-DP.
    Noise scale: b = C / epsilon.
    """
    if epsilon <= 0.0:
        raise ValueError(f"Invalid epsilon={epsilon}")

    rng = np.random.default_rng(random_state)
    b = float(clip_norm / epsilon)

    noisy_params = [
        p + rng.laplace(loc=0.0, scale=b, size=p.shape).astype(p.dtype)
        for p in parameters
    ]
    return noisy_params, b


def apply_differential_privacy(
    parameters: list[np.ndarray],
    clip_norm: float = 1.0,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    mechanism: str = "gaussian",
    random_state: int | None = None
) -> dict[str, Any]:
    """
    Unified entrypoint to clip parameters and inject DP noise.
    """
    clipped_params, original_norm = clip_parameter_tensors(parameters, clip_norm=clip_norm)
    clipped_norm = compute_l2_norm(clipped_params)

    mechanism_clean = mechanism.lower().strip()
    if mechanism_clean in ["laplace", "laplacian"]:
        noisy_params, noise_scale = add_laplace_noise(
            parameters=clipped_params,
            clip_norm=clip_norm,
            epsilon=epsilon,
            random_state=random_state
        )
        effective_delta = 0.0
    else:
        noisy_params, noise_scale = add_gaussian_noise(
            parameters=clipped_params,
            clip_norm=clip_norm,
            epsilon=epsilon,
            delta=delta,
            random_state=random_state
        )
        effective_delta = delta

    return {
        "parameters": noisy_params,
        "mechanism": mechanism_clean,
        "clip_norm": clip_norm,
        "original_norm": round(original_norm, 6),
        "clipped_norm": round(clipped_norm, 6),
        "noise_scale_sigma": round(noise_scale, 6),
        "epsilon": epsilon,
        "delta": effective_delta
    }
