#!/usr/bin/env python3
"""
Initialize the database with tables and initial data.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import logger
from app.db.session import init_db, close_db, async_engine
from app.models import Base


async def create_database():
    """Create database if it doesn't exist (PostgreSQL only)."""
    if not settings.database_url or "sqlite" in settings.database_url:
        return
    
    # Extract database name from URL
    db_url_parts = settings.database_url.split("/")
    db_name = db_url_parts[-1].split("?")[0]
    base_url = "/".join(db_url_parts[:-1])
    
    # Connect to postgres database to create our database
    postgres_url = f"{base_url}/postgres"
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        temp_engine = create_async_engine(
            postgres_url.replace("postgresql://", "postgresql+asyncpg://")
        )
        
        async with temp_engine.connect() as conn:
            # Check if database exists
            result = await conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            )
            exists = result.scalar()
            
            if not exists:
                # Need to use isolation level for CREATE DATABASE
                await conn.execute(text("COMMIT"))
                await conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database already exists: {db_name}")
        
        await temp_engine.dispose()
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        # Database might already exist, continue anyway


async def init_tables():
    """Initialize database tables."""
    await init_db()
    
    if async_engine:
        async with async_engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Created all database tables")
    
    await close_db()


def run_migrations():
    """Run Alembic migrations."""
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Applied database migrations")
    except Exception as e:
        logger.warning(f"Could not run migrations: {e}")
        logger.info("Tables will be created directly if needed")


async def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")
    
    # Create database if needed
    await create_database()
    
    # Initialize tables
    await init_tables()
    
    # Run migrations (optional, will create initial migration if needed)
    # run_migrations()
    
    logger.info("Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())