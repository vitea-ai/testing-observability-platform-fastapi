"""
Evaluation models for the testing observability platform.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.models.base import Base, TimestampMixin


class Evaluation(Base, TimestampMixin):
    """Evaluation model for storing evaluation results."""
    
    __tablename__ = "evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    
    # Evaluation configuration
    evaluator_id = Column(String(255), nullable=False, index=True)
    evaluator_name = Column(String(255), nullable=False)
    evaluator_config = Column(JSONB, default=dict)
    
    # Task tracking (for Celery)
    task_id = Column(String(255), nullable=True, index=True)
    
    # Results
    status = Column(String(50), default="pending", nullable=False, index=True)
    score = Column(Float, nullable=True)
    results = Column(JSONB, nullable=True)
    
    # Summary statistics
    total_tests = Column(Integer, default=0)
    passed_tests = Column(Integer, default=0)
    failed_tests = Column(Integer, default=0)
    error_tests = Column(Integer, default=0)
    
    # Execution details
    execution_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    meta_data = Column("metadata", JSONB, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    experiment = relationship("Experiment", backref="evaluations", lazy="joined")
    
    def __repr__(self):
        return f"<Evaluation(id={self.id}, experiment_id={self.experiment_id}, evaluator={self.evaluator_name})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "experiment_id": str(self.experiment_id),
            "evaluator_id": self.evaluator_id,
            "evaluator_name": self.evaluator_name,
            "evaluator_config": self.evaluator_config,
            "task_id": self.task_id,
            "status": self.status,
            "score": self.score,
            "results": self.results,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "error_tests": self.error_tests,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.meta_data,
        }


class EvaluationMetric(Base, TimestampMixin):
    """Evaluation metric model for defining available metrics."""
    
    __tablename__ = "evaluation_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False, index=True)
    implementation_type = Column(String(100), nullable=False)
    
    # Configuration schema
    config_schema = Column(JSONB, default=dict)
    default_config = Column(JSONB, default=dict)
    
    # Metadata
    version = Column(String(50), default="1.0.0")
    is_active = Column(Integer, default=1, nullable=False)
    meta_data = Column("metadata", JSONB, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    def __repr__(self):
        return f"<EvaluationMetric(id={self.id}, name='{self.name}', category={self.category})>"