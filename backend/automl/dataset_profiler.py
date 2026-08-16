"""
[IMPLEMENTED] Tabular Dataset Profiler.
Analyzes dataset dimensions, feature distributions, skewness, class imbalance, and multicollinearity.
"""
from typing import Any

import numpy as np
import pandas as pd


def profile_dataset(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    feature_names: list[str] | None = None
) -> dict[str, Any]:
    """
    Profile a tabular dataset and return comprehensive summary statistics.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy()
        feat_names = list(X.columns)
    else:
        X_arr = np.asarray(X)
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y).ravel()

    num_samples, num_features = X_arr.shape

    # Class distribution & imbalance
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique_classes, counts, strict=True)}
    imbalance_ratio = float(max(counts) / max(1, min(counts)))

    # Feature statistical profiles
    feature_stats = {}
    for i, name in enumerate(feat_names):
        col = X_arr[:, i]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        min_val = float(np.min(col))
        max_val = float(np.max(col))

        # Skewness calculation (Fisher-Pearson coefficient)
        if std_val > 1e-9:
            skew_val = float(np.mean(((col - mean_val) / std_val) ** 3))
        else:
            skew_val = 0.0

        feature_stats[name] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "skewness": round(skew_val, 4)
        }

    # Correlation analysis
    df = pd.DataFrame(X_arr, columns=feat_names)
    corr_matrix = df.corr().abs()

    high_corr_pairs = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            r_val = corr_matrix.iloc[i, j]
            if not np.isnan(r_val) and r_val > 0.85:
                high_corr_pairs.append({
                    "feature_a": feat_names[i],
                    "feature_b": feat_names[j],
                    "correlation": round(float(r_val), 4)
                })

    return {
        "num_samples": num_samples,
        "num_features": num_features,
        "feature_names": feat_names,
        "class_distribution": class_dist,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "feature_statistics": feature_stats,
        "high_correlation_pairs": high_corr_pairs,
        "has_multicollinearity": len(high_corr_pairs) > 0
    }
