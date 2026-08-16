"""
[IMPLEMENTED] Abstract base class for all local, global, and centralized model wrappers.
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModelWrapper(ABC):
    """Abstract interface for all model wrappers in QFedAutoML."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, float]:
        """Train the model on input data and return training metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate hard class predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate prediction probability distributions."""

    @abstractmethod
    def get_weights(self) -> Any:
        """Extract serialized or numpy array representation of model weights/parameters."""

    @abstractmethod
    def set_weights(self, weights: Any) -> None:
        """Load weights/parameters into the model."""
