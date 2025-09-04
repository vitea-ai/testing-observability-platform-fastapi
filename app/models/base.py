"""
Base SQLAlchemy model and common mixins.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import DeclarativeBase

# Base class for all models
class Base(DeclarativeBase):
    """Base class for all database models."""
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower()


class TimestampMixin:
    """Mixin for adding timestamp fields."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class AuditMixin(TimestampMixin):
    """Mixin for adding audit fields."""
    
    created_by = Column(String(255), nullable=True)
    updated_by = Column(String(255), nullable=True)
