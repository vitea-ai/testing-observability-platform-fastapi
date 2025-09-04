"""
Experiment models for the testing observability platform.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from app.models.base import Base, AuditMixin


class ExperimentStatus(str, enum.Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(str, enum.Enum):
    """Experiment execution mode."""
    BATCH = "batch"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class Experiment(Base, AuditMixin):
    """Experiment model for managing AI evaluation runs."""
    
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Reference to dataset
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True, index=True)
    
    # Experiment configuration
    agent_config = Column(JSONB, nullable=False, default=dict)
    execution_mode = Column(SQLEnum(ExecutionMode), default=ExecutionMode.BATCH, nullable=False)
    
    # Status tracking
    status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING, nullable=False, index=True)
    progress = Column(Float, default=0.0, nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional metadata
    tags = Column(JSONB, default=list)
    meta_data = Column("metadata", JSONB, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    # Relationships
    dataset = relationship("Dataset", backref="experiments", lazy="joined")
    
    def __repr__(self):
        return f"<Experiment(id={self.id}, name='{self.name}', status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "agent_config": self.agent_config,
            "execution_mode": self.execution_mode.value if self.execution_mode else None,
            "status": self.status.value if self.status else None,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "metadata": self.meta_data,
        }


class TestResult(Base):
    """Test result model for individual test case results within an experiment."""
    
    __tablename__ = "test_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    
    # Test case information
    test_id = Column(String(255), nullable=True)
    test_case_type = Column(String(50), nullable=True)
    
    # Input/Output
    input = Column(JSONB, nullable=False)
    expected_output = Column(JSONB, nullable=True)
    actual_output = Column(JSONB, nullable=True)
    
    # Context and metadata
    context = Column(JSONB, nullable=True)
    retrieval_context = Column(JSONB, nullable=True)
    tools_called = Column(JSONB, nullable=True)
    
    # Multi-turn specific
    scenario = Column(Text, nullable=True)
    expected_outcome = Column(Text, nullable=True)
    
    # Result status
    status = Column(String(50), default="pending", nullable=False, index=True)
    execution_time = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    
    # Metadata
    meta_data = Column("metadata", JSONB, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    experiment = relationship("Experiment", backref="test_results", lazy="joined")
    
    def __repr__(self):
        return f"<TestResult(id={self.id}, experiment_id={self.experiment_id}, status={self.status})>"