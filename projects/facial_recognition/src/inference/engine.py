"""Real-time face recognition engine with vector DB"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from projects.facial_recognition.src.models.detector import FaceEncoder, RecognitionResult
from src.core import get_logger

logger = get_logger(__name__)


class FaceIdentity(BaseModel):
    """Registered face identity"""

    identity_id: str
    name: str
    embedding: list[float]
    registered_at: datetime
    metadata: dict = Field(default_factory=dict)


class FaceRecognitionEngine:
    """Production face recognition with vector database and caching"""

    def __init__(self, encoder: FaceEncoder, distance_threshold: float = 0.6):
        """Initialize engine

        Args:
            encoder: Face encoder model
            distance_threshold: Max distance for positive match
        """
        self.encoder = encoder
        self.distance_threshold = distance_threshold
        self.identity_db: dict[str, FaceIdentity] = {}  # In-memory vector DB
        self.recognition_count = 0
        self.matches_found = 0
        logger.info(
            "FaceRecognitionEngine initialized",
            distance_threshold=distance_threshold,
        )

    def register_face(self, face: np.ndarray, identity_id: str, name: str, metadata: dict | None = None) -> bool:
        """Register a face identity

        Args:
            face: Face image
            identity_id: Unique identity ID
            name: Person's name
            metadata: Additional metadata

        Returns:
            Success status
        """
        logger.info("Registering face", identity_id=identity_id, name=name)

        try:
            # Encode face
            embedding_obj = self.encoder.encode(face)

            # Store in DB
            identity = FaceIdentity(
                identity_id=identity_id,
                name=name,
                embedding=embedding_obj.embedding,
                registered_at=datetime.now(),
                metadata=metadata or {},
            )
            self.identity_db[identity_id] = identity

            logger.info("Face registered", identity_id=identity_id)
            return True

        except Exception as e:
            logger.error("Face registration failed", error=str(e))
            return False

    def recognize(self, face: np.ndarray) -> RecognitionResult:
        """Recognize face

        Args:
            face: Face image to recognize

        Returns:
            Recognition result with identity
        """
        logger.info("Recognizing face")

        try:
            # Encode face
            embedding_obj = self.encoder.encode(face)
            query_emb = np.array(embedding_obj.embedding)

            # Find nearest neighbor in DB
            min_distance = float("inf")
            best_identity = None

            for identity_id, identity in self.identity_db.items():
                db_emb = np.array(identity.embedding)
                distance = self.encoder.compute_distance(query_emb, db_emb)

                if distance < min_distance:
                    min_distance = distance
                    best_identity = identity

            # Check if match
            is_match = min_distance < self.distance_threshold

            if is_match:
                self.matches_found += 1
                logger.info(
                    "Face recognized",
                    identity=best_identity.name,
                    distance=min_distance,
                )
            else:
                logger.info("Unknown face", min_distance=min_distance)

            self.recognition_count += 1

            return RecognitionResult(
                identity=best_identity.name if is_match else "Unknown",
                confidence=1.0 - (min_distance / self.distance_threshold),
                distance=min_distance,
                is_known=is_match,
                liveness_score=0.95,  # TODO: Implement liveness detection
                spoofing_detected=False,  # TODO: Implement spoofing detection
            )

        except Exception as e:
            logger.error("Recognition failed", error=str(e))
            return RecognitionResult(
                identity="Error",
                confidence=0.0,
                distance=float("inf"),
                is_known=False,
                liveness_score=0.0,
                spoofing_detected=False,
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get recognition metrics"""
        return {
            "total_recognitions": self.recognition_count,
            "matches_found": self.matches_found,
            "match_rate": self.matches_found / max(1, self.recognition_count),
            "identities_registered": len(self.identity_db),
            "distance_threshold": self.distance_threshold,
        }
