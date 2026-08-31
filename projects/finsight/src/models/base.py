"""Base model classes with uncertainty quantification"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.core import get_logger

logger = get_logger(__name__)


class PredictionResult(BaseModel):
    """Prediction with uncertainty quantification"""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    std_dev: float
    model_name: str


class BaseForecaster(ABC):
    """Abstract base class for forecasters"""

    def __init__(self, name: str):
        """Initialize forecaster

        Args:
            name: Model name for logging/tracking
        """
        self.name = name
        self.is_trained = False
        logger.info(f"{name} initialized")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model

        Args:
            X: Training features
            y: Training targets
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Make predictions with uncertainty

        Args:
            X: Features to predict

        Returns:
            Prediction result with confidence intervals
        """
        pass

    def validate_input(self, X: np.ndarray) -> None:
        """Validate input array

        Args:
            X: Input array to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be numpy array")

        if X.ndim < 1:
            raise ValueError("Input must be at least 1-dimensional")

        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
