#!/usr/bin/env python3
"""
Test database connection and basic operations.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.session import init_db, close_db, get_db
from app.models import Dataset, DatasetStatus, DatasetType


async def test_connection():
    """Test basic database connection."""
    logger.info("Testing database connection...")
    
    await init_db()
    
    # Get a database session
    async for db in get_db():
        if db:
            try:
                # Test basic query
                result = await db.execute(text("SELECT 1"))
                value = result.scalar()
                logger.info(f"✅ Database connection successful! Test query returned: {value}")
                
                # Test table access
                result = await db.execute(select(Dataset).limit(1))
                logger.info("✅ Can access datasets table")
                
                return True
            except Exception as e:
                logger.error(f"❌ Database error: {e}")
                return False
        else:
            logger.warning("⚠️  No database session available")
            return False
    
    await close_db()


async def test_crud_operations():
    """Test CRUD operations on Dataset model."""
    logger.info("Testing CRUD operations...")
    
    await init_db()
    
    async for db in get_db():
        if not db:
            logger.warning("No database session available")
            return False
        
        try:
            # CREATE
            test_dataset = Dataset(
                name=f"Test Dataset {uuid4().hex[:8]}",
                description="Test dataset for database connection",
                type=DatasetType.CUSTOM,
                status=DatasetStatus.ACTIVE,
                data=[
                    {"input": "test1", "expected_output": "output1"},
                    {"input": "test2", "expected_output": "output2"}
                ],
                record_count=2,
                created_by="test_script"
            )
            
            db.add(test_dataset)
            await db.commit()
            await db.refresh(test_dataset)
            logger.info(f"✅ Created dataset: {test_dataset.id}")
            
            # READ
            query = select(Dataset).where(Dataset.id == test_dataset.id)
            result = await db.execute(query)
            retrieved = result.scalar_one_or_none()
            
            if retrieved:
                logger.info(f"✅ Retrieved dataset: {retrieved.name}")
            else:
                logger.error("❌ Failed to retrieve dataset")
                return False
            
            # UPDATE
            retrieved.description = "Updated description"
            await db.commit()
            logger.info("✅ Updated dataset description")
            
            # LIST
            query = select(Dataset).where(Dataset.status == DatasetStatus.ACTIVE)
            result = await db.execute(query)
            datasets = result.scalars().all()
            logger.info(f"✅ Listed {len(datasets)} active datasets")
            
            # DELETE (soft delete)
            retrieved.status = DatasetStatus.DELETED
            await db.commit()
            logger.info("✅ Soft deleted dataset")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ CRUD operation failed: {e}")
            await db.rollback()
            return False
    
    await close_db()


async def main():
    """Run all database tests."""
    logger.info("=" * 60)
    logger.info("Database Connection Test")
    logger.info(f"Database URL: {settings.database_url}")
    logger.info(f"Deployment Tier: {settings.deployment_tier}")
    logger.info("=" * 60)
    
    # Test connection
    connection_ok = await test_connection()
    
    if connection_ok:
        # Test CRUD operations
        crud_ok = await test_crud_operations()
        
        if crud_ok:
            logger.info("\n✅ All database tests passed!")
        else:
            logger.error("\n❌ CRUD tests failed")
    else:
        logger.error("\n❌ Connection test failed")
    
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())