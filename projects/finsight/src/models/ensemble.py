"""Ensemble forecasting combining multiple models"""

from typing import Any

import numpy as np

from projects.finsight.src.models.base import BaseForecaster, PredictionResult
from src.core import get_logger

logger = get_logger(__name__)


class EnsembleForecaster(BaseForecaster):
    """Ensemble combining multiple forecasters

    This class implements a weighted ensemble approach:
    1. Train multiple individual forecasters
    2. Combine predictions using optimal weights
    3. Quantify uncertainty across ensemble members
    """

    def __init__(self, forecasters: list[BaseForecaster], weights: list[float] | None = None):
        """Initialize ensemble

        Args:
            forecasters: List of individual forecasters
            weights: Weights for each forecaster (default: uniform)
        """
        super().__init__("EnsembleForecaster")
        self.forecasters = forecasters
        n_models = len(forecasters)

        if weights is None:
            weights = [1.0 / n_models] * n_models
        else:
            if len(weights) != n_models:
                raise ValueError(f"Expected {n_models} weights, got {len(weights)}")
            total = sum(weights)
            weights = [w / total for w in weights]

        self.weights = weights
        logger.info(
            "EnsembleForecaster created",
            n_models=n_models,
            weights=self.weights,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train all forecasters

        Args:
            X: Training features
            y: Training targets
        """
        self.validate_input(X)

        logger.info("Training ensemble", n_models=len(self.forecasters))

        for i, forecaster in enumerate(self.forecasters):
            logger.info(f"Training {forecaster.name} ({i+1}/{len(self.forecasters)})")
            forecaster.fit(X, y)

        self.is_trained = True
        logger.info("Ensemble training complete")

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Predict with ensemble

        Args:
            X: Features to predict

        Returns:
            Ensemble prediction with uncertainty
        """
        self.validate_input(X)

        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        logger.info("Making ensemble prediction", n_models=len(self.forecasters))

        # Get predictions from all models
        predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(X)
            predictions.append(pred.point_estimate)

        # Weighted combination
        predictions = np.array(predictions)
        point_estimate = np.average(predictions, weights=self.weights)

        # Uncertainty from ensemble variance
        ensemble_std = np.sqrt(np.average((predictions - point_estimate) ** 2, weights=self.weights))

        # 95% confidence interval
        z_score = 1.96  # 95% CI
        margin = z_score * ensemble_std

        return PredictionResult(
            point_estimate=float(point_estimate),
            lower_bound=float(point_estimate - margin),
            upper_bound=float(point_estimate + margin),
            confidence_level=0.95,
            std_dev=float(ensemble_std),
            model_name=self.name,
        )
