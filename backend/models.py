"""
Database models for storing prediction history and results.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Prediction(Base):
    """
    Stores metadata about each fault localization prediction run.
    """
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, index=True)  # UUID
    repository_path = Column(String, nullable=False)
    failing_test = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_candidates = Column(Integer, default=0)
    
    # Relationship to ranked results
    ranked_lines = relationship("RankedLine", back_populates="prediction", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert prediction to dictionary for API response."""
        return {
            "id": self.id,
            "repository_path": self.repository_path,
            "failing_test": self.failing_test,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "total_candidates": self.total_candidates
        }


class RankedLine(Base):
    """
    Stores individual ranked lines for each prediction.
    Each line represents a potential fault location with its suspiciousness score.
    """
    __tablename__ = "ranked_lines"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(String, ForeignKey("predictions.id"), nullable=False)
    file_path = Column(String, nullable=False)
    line_number = Column(Integer, nullable=False)
    code = Column(Text, nullable=False)
    probability = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    
    # Additional features stored as JSON
    features = Column(JSON, nullable=True)
    
    # Relationship back to prediction
    prediction = relationship("Prediction", back_populates="ranked_lines")
    
    def to_dict(self):
        """Convert ranked line to dictionary for API response."""
        return {
            "file": self.file_path,
            "line_number": self.line_number,
            "code": self.code,
            "probability": round(self.probability, 4),
            "rank": self.rank,
            "features": self.features
        }


class EmbeddingCache(Base):
    """
    Caches CodeBERT embeddings for code lines to speed up inference.
    Key is hash of (file_path + line_number + code content).
    """
    __tablename__ = "embedding_cache"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cache_key = Column(String, unique=True, index=True, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-serialized numpy array
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert cache entry to dictionary."""
        return {
            "cache_key": self.cache_key,
            "created_at": self.created_at.isoformat()
        }
