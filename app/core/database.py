"""
Database configuration using psycopg3 with SQLAlchemy async.

This module provides database session management using psycopg3, which offers
better compatibility with SQLAlchemy's async ORM compared to asyncpg.
"""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.core.config import settings

# Create async engine with psycopg3
# Note: psycopg uses 'postgresql+psycopg' not 'postgresql+asyncpg'
engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+psycopg://")
    if settings.database_url
    else "sqlite+aiosqlite:///./test.db",  # Fallback for Tier 1
    echo=settings.debug,
    pool_pre_ping=True,  # Verify connections are alive
    pool_size=20,  # Conservative pool size to avoid exhausting PostgreSQL
    max_overflow=20,  # Total of 40 connections max for main API
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=10,  # Wait up to 10 seconds for a connection (faster failure)
    connect_args={
        "options": "-c statement_timeout=60000 -c jit=off"  # 60 second statement timeout and disable JIT
    } if settings.database_url and "postgresql" in settings.database_url else {}
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
)

# Create a separate engine for Celery workers with smaller pool
celery_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+psycopg://")
    if settings.database_url
    else "sqlite+aiosqlite:///./test.db",
    echo=False,  # Less logging for workers
    pool_pre_ping=True,
    pool_size=5,  # Small pool for workers to avoid connection exhaustion
    max_overflow=10,  # Total of 15 connections max for Celery
    pool_recycle=3600,
    pool_timeout=5,  # Shorter timeout for workers
    connect_args={
        "options": "-c statement_timeout=60000 -c jit=off"  # 60 second statement timeout and disable JIT
    } if settings.database_url and "postgresql" in settings.database_url else {}
)

# Celery worker session factory
CelerySessionLocal = async_sessionmaker(
    celery_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for SQLAlchemy models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Usage in FastAPI:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """
    Context manager for database session (for non-FastAPI usage).
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Initialize database - create tables if they don't exist."""
    async with engine.begin() as conn:
        # For development, create all tables
        # In production, use Alembic migrations instead
        if settings.deployment_tier == "development":
            await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    await engine.dispose()


# For high-performance scenarios where you need raw SQL
async def execute_raw_sql(query: str, *args):
    """
    Execute raw SQL query using psycopg3 directly.
    
    Note: This bypasses SQLAlchemy ORM. Use only when you need
    maximum performance and are comfortable with raw SQL.
    
    Example:
        users = await execute_raw_sql(
            "SELECT * FROM users WHERE age > %s AND city = %s",
            18, "New York"
        )
    """
    async with engine.raw_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, args)
            if cursor.description:
                # Query returns data
                columns = [desc[0] for desc in cursor.description]
                rows = await cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            else:
                # Query doesn't return data (INSERT, UPDATE, DELETE)
                await conn.commit()
                return cursor.rowcount


# Example model to demonstrate psycopg3 compatibility
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Usage with async SQLAlchemy:
from sqlalchemy import select

async def get_user_by_email(db: AsyncSession, email: str):
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()

# With named parameters (psycopg3 supports this!):
async def search_users(db: AsyncSession, name: str, age: int):
    result = await db.execute(
        text("SELECT * FROM users WHERE name = :name AND age > :age"),
        {"name": name, "age": age}  # Named parameters work!
    )
    return result.fetchall()
"""