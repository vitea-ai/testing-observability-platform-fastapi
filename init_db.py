#!/usr/bin/env python3
"""Initialize database tables."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import engine, Base
from app.models import *  # Import all models to register them


async def init_db():
    """Create all database tables."""
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully!")


if __name__ == "__main__":
    asyncio.run(init_db())