"""Tests for Finsight models"""

import numpy as np
import pytest

from projects.finsight.src.models.base import BaseForecaster, PredictionResult
from projects.finsight.src.models.ensemble import EnsembleForecaster


class DummyForecaster(BaseForecaster):
    """Dummy forecaster for testing"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model"""
        self.is_trained = True

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Return dummy prediction"""
        return PredictionResult(
            point_estimate=1.0,
            lower_bound=0.9,
            upper_bound=1.1,
            confidence_level=0.95,
            std_dev=0.1,
            model_name=self.name,
        )


class TestEnsembleForecaster:
    """Test ensemble forecaster"""

    def test_ensemble_creation(self) -> None:
        """Test ensemble creation"""
        forecasters = [DummyForecaster("model1"), DummyForecaster("model2")]
        ensemble = EnsembleForecaster(forecasters)
        assert len(ensemble.forecasters) == 2
        assert sum(ensemble.weights) == pytest.approx(1.0)

    def test_ensemble_training(self) -> None:
        """Test ensemble training"""
        forecasters = [DummyForecaster("model1"), DummyForecaster("model2")]
        ensemble = EnsembleForecaster(forecasters)

        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        ensemble.fit(X, y)
        assert ensemble.is_trained

    def test_ensemble_prediction(self) -> None:
        """Test ensemble prediction"""
        forecasters = [DummyForecaster("model1"), DummyForecaster("model2")]
        ensemble = EnsembleForecaster(forecasters)

        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        ensemble.fit(X, y)

        X_test = np.random.randn(1, 5)
        result = ensemble.predict(X_test)

        assert isinstance(result, PredictionResult)
        assert result.lower_bound < result.point_estimate < result.upper_bound
