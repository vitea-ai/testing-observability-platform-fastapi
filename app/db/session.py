"""
Database session management with tier-based configuration.
"""

from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import logger

# Database engines (initialized in init_db)
sync_engine = None
async_engine = None

# Session factories
SessionLocal = None
AsyncSessionLocal = None


async def init_db() -> None:
    """
    Initialize database connections based on deployment tier.
    """
    global sync_engine, async_engine, SessionLocal, AsyncSessionLocal
    
    db_url = settings.get_database_url()
    
    if not db_url:
        logger.warning("No database URL configured")
        return
    
    logger.info("Initializing database", url=db_url.split("@")[-1])
    
    # For SQLite, use synchronous engine
    if db_url.startswith("sqlite"):
        sync_engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=settings.database_echo,
            pool_pre_ping=True,
        )
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=sync_engine
        )
    else:
        # For PostgreSQL, use async engine with psycopg3
        # Convert postgresql:// to postgresql+psycopg://
        async_db_url = db_url.replace("postgresql://", "postgresql+psycopg://")
        
        async_engine = create_async_engine(
            async_db_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            poolclass=NullPool if settings.testing else None,
        )
        
        AsyncSessionLocal = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    # Create tables if in development
    if settings.is_development:
        from app.models.base import Base
        if sync_engine:
            Base.metadata.create_all(bind=sync_engine)
        # For async engine, tables should be created via Alembic
    
    logger.info("Database initialized successfully")


async def close_db() -> None:
    """
    Close database connections.
    """
    global sync_engine, async_engine
    
    if sync_engine:
        sync_engine.dispose()
        logger.info("Sync database engine disposed")
    
    if async_engine:
        await async_engine.dispose()
        logger.info("Async database engine disposed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        Database session
    """
    if AsyncSessionLocal:
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    elif SessionLocal:
        # Fallback to sync session for SQLite
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    else:
        # No database configured, yield None
        yield None


async def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        if async_engine:
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                return True
        elif sync_engine:
            with sync_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        return False
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False
