"""
Dataset models for the testing observability platform.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
import enum

from app.models.base import Base, AuditMixin


class DatasetStatus(str, enum.Enum):
    """Dataset status enumeration."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"


class DatasetType(str, enum.Enum):
    """Dataset type enumeration."""
    CUSTOM = "custom"
    HEALTHCARE = "healthcare"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    FUNCTIONAL = "functional"


class Dataset(Base, AuditMixin):
    """Dataset model for storing test datasets."""
    
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    type = Column(SQLEnum(DatasetType), default=DatasetType.CUSTOM, nullable=False, index=True)
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.ACTIVE, nullable=False, index=True)
    
    # Store the entire dataset as JSONB for flexibility
    data = Column(JSONB, nullable=False, default=list)
    record_count = Column(Integer, default=0, nullable=False)
    
    # Additional fields for versioning and tagging
    version = Column(String(50), default="1.0.0")
    tags = Column(JSONB, default=list)
    meta_data = Column("metadata", JSONB, default=dict)  # Renamed to avoid SQLAlchemy conflict
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', type={self.type}, records={self.record_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "type": self.type.value if self.type else None,
            "status": self.status.value if self.status else None,
            "data": self.data,
            "record_count": self.record_count,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.meta_data,
        }