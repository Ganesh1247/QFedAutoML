"""
[IMPLEMENTED] Preprocessing and starter tabular dataset loader.
Provides standardized dataset partitioning, scaling, and feature normalization.
Supports transparent hot-swap to user-uploaded datasets via active_dataset.json registry.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Registry path — written by routes_datasets.py on every upload
_ACTIVE_REGISTRY = Path("backend/data/active_dataset.json")


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    num_samples: int
    num_features: int


class TabularPreprocessor:
    """Standard preprocessor for tabular data (scaling, splitting)."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler on training data and transform."""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform validation/test data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.scaler.transform(X)


def load_starter_tabular_dataset(
    dataset_name: str = "breast_cancer",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> DatasetSplits:
    """
    Load standardized healthcare tabular classification starter dataset.

    Priority order:
      1. User-uploaded dataset (if backend/data/active_dataset.json exists)
      2. Built-in Wisconsin Diagnostic Breast Cancer (default, 569 samples, 30 features)
    """
    # ── Priority 1: User-uploaded dataset ──────────────────────────────────
    if _ACTIVE_REGISTRY.exists():
        try:
            registry = json.loads(_ACTIVE_REGISTRY.read_text(encoding="utf-8"))
            csv_path = Path(registry["saved_path"])
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                target_col = registry["target_column"]
                feature_cols = registry["feature_columns"]
                X = df[feature_cols].values.astype(np.float64)
                y = df[target_col].values.astype(int)
                feature_names = feature_cols
                target_name = target_col
                # Fall through to split/scale below
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                adjusted_val_size = val_size / (1.0 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=adjusted_val_size,
                    random_state=random_state, stratify=y_temp
                )
                preprocessor = TabularPreprocessor()
                X_train_scaled = preprocessor.fit_transform(X_train)
                X_val_scaled = preprocessor.transform(X_val)
                X_test_scaled = preprocessor.transform(X_test)
                return DatasetSplits(
                    X_train=X_train_scaled,
                    X_val=X_val_scaled,
                    X_test=X_test_scaled,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    feature_names=feature_names,
                    target_name=target_name,
                    num_samples=len(X),
                    num_features=X.shape[1]
                )
        except (OSError, ValueError, KeyError):
            # If anything goes wrong loading user data, fall back silently
            pass

    # ── Priority 2: Built-in breast cancer dataset ──────────────────────────
    raw = load_breast_cancer()
    X = raw.data
    y = raw.target
    feature_names = [str(f) for f in raw.feature_names]
    target_name = "diagnosis_malignant"

    # First split off test set (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split temp into train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
    )

    # Normalize features using training statistics
    preprocessor = TabularPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    return DatasetSplits(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        target_name=target_name,
        num_samples=len(X),
        num_features=X.shape[1]
    )


@dataclass
class SequenceDatasetSplits:
    X_train: np.ndarray  # (N_train, seq_len, num_features)
    X_val: np.ndarray    # (N_val, seq_len, num_features)
    X_test: np.ndarray   # (N_test, seq_len, num_features)
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    num_samples: int
    seq_len: int
    num_features: int
    num_classes: int


def load_sensor_timeseries_dataset(
    num_samples: int = 500,
    seq_len: int = 24,
    num_features: int = 6,
    num_classes: int = 2,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> SequenceDatasetSplits:
    """
    Generate realistic multi-channel sensor / IoT time-series classification dataset.
    Features simulate periodic signals (sinusoids + noise + class-specific frequency drift).
    """
    np.random.seed(random_state)
    t = np.linspace(0, 4 * np.pi, seq_len)

    X = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)

    for i in range(num_samples):
        cls_shift = 1.5 if y[i] == 1 else 0.5
        for f in range(num_features):
            freq = (f + 1) * cls_shift
            noise = np.random.normal(0, 0.2, seq_len)
            X[i, :, f] = np.sin(freq * t) + 0.5 * np.cos(freq * 0.5 * t) + noise

    # Split test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
    )

    # Normalize per feature across time-steps using train statistics
    for f in range(num_features):
        mean = X_train[:, :, f].mean()
        std = X_train[:, :, f].std() + 1e-7
        X_train[:, :, f] = (X_train[:, :, f] - mean) / std
        X_val[:, :, f] = (X_val[:, :, f] - mean) / std
        X_test[:, :, f] = (X_test[:, :, f] - mean) / std

    return SequenceDatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        num_samples=num_samples,
        seq_len=seq_len,
        num_features=num_features,
        num_classes=num_classes
    )

