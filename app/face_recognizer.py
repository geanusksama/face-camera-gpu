import numpy as np
from image_database import ImageDatabase


class FaceRecognizer:
    def __init__(self, db: ImageDatabase, threshold: float = 0.55):
        self.db = db
        self.threshold = threshold

    def recognize(self, face) -> tuple[str | None, float]:
        name, confidence = self.db.find_match(face.embedding, self.threshold)
        return name, confidence
