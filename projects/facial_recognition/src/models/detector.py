"""Face detection and embedding models"""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel

from src.core import get_logger

logger = get_logger(__name__)


class FaceEmbedding(BaseModel):
    """Face embedding vector"""

    embedding: list[float]  # 128-512 dimensional
    face_id: str
    confidence: float
    model_name: str


class RecognitionResult(BaseModel):
    """Face recognition result"""

    identity: str
    confidence: float
    distance: float  # Distance to nearest known face
    is_known: bool
    liveness_score: float  # 0-1, 1 = real face
    spoofing_detected: bool


class FaceDetector(ABC):
    """Abstract face detection model"""

    def __init__(self, name: str):
        """Initialize detector

        Args:
            name: Model name
        """
        self.name = name
        self.is_loaded = False
        logger.info(f"{name} initialized")

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in image

        Args:
            image: Input image (H, W, 3)

        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        pass


class FaceEncoder(ABC):
    """Abstract face encoding model"""

    def __init__(self, name: str, embedding_dim: int = 128):
        """Initialize encoder

        Args:
            name: Model name
            embedding_dim: Embedding dimension
        """
        self.name = name
        self.embedding_dim = embedding_dim
        self.is_loaded = False
        logger.info(f"{name} initialized", embedding_dim=embedding_dim)

    @abstractmethod
    def encode(self, face: np.ndarray) -> FaceEmbedding:
        """Encode face to embedding

        Args:
            face: Face image (H, W, 3)

        Returns:
            Face embedding
        """
        pass

    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute distance between embeddings

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(emb1 - emb2))
