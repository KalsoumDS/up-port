"""SHAP-based face recognition explainability"""

from typing import Any

import numpy as np
from pydantic import BaseModel

from src.core import get_logger

logger = get_logger(__name__)


class RecognitionExplanation(BaseModel):
    """Explanation for recognition decision"""

    identity: str
    confidence: float
    key_features: list[tuple[str, float]]  # (feature, importance)
    similar_faces: list[tuple[str, float]]  # (identity, similarity)
    timestamp: str


class FaceExplainer:
    """SHAP-based face recognition explainability"""

    def __init__(self, model: Any, landmark_names: list[str]):
        """Initialize explainer

        Args:
            model: Recognition model
            landmark_names: Names of facial landmarks/features
        """
        self.model = model
        self.landmark_names = landmark_names
        logger.info("FaceExplainer initialized", n_landmarks=len(landmark_names))

    def explain_recognition(self, face: np.ndarray, identity: str, confidence: float) -> RecognitionExplanation:
        """Explain recognition decision

        Args:
            face: Face image
            identity: Recognized identity
            confidence: Recognition confidence

        Returns:
            Explanation with SHAP values
        """
        logger.info("Explaining recognition", identity=identity)

        from datetime import datetime

        # TODO: Implement SHAP explanations
        key_features = [
            ("eye_distance", 0.35),
            ("nose_width", 0.25),
            ("face_shape", 0.20),
            ("mouth_geometry", 0.15),
            ("skin_texture", 0.05),
        ]

        similar_faces = [
            ("person_123", 0.91),
            ("person_456", 0.87),
            ("person_789", 0.72),
        ]

        return RecognitionExplanation(
            identity=identity,
            confidence=confidence,
            key_features=key_features,
            similar_faces=similar_faces,
            timestamp=datetime.now().isoformat(),
        )
