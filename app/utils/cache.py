"""
Cache utilities for Redis integration (Tier 3+).
"""

from typing import Any, Optional
import json
import pickle
from functools import wraps

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import logger

# Global Redis client
redis_client: Optional[redis.Redis] = None


async def init_cache() -> None:
    """Initialize Redis connection."""
    global redis_client
    
    if not settings.redis_url:
        logger.warning("Redis URL not configured")
        return
    
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=False
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("Redis cache initialized", url=settings.redis_url.split("@")[-1])
    except Exception as e:
        logger.error("Failed to initialize Redis", error=str(e))
        redis_client = None


async def close_cache() -> None:
    """Close Redis connection."""
    global redis_client
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis cache connection closed")
        redis_client = None


async def check_cache_connection() -> bool:
    """Check if cache connection is working."""
    if not redis_client:
        return False
    
    try:
        await redis_client.ping()
        return True
    except Exception:
        return False


class CacheManager:
    """Cache manager for Redis operations."""
    
    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if not redis_client:
            return None
        
        try:
            value = await redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    @staticmethod
    async def set(key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        if not redis_client:
            return False
        
        try:
            serialized = pickle.dumps(value)
            await redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def delete(key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if successful
        """
        if not redis_client:
            return False
        
        try:
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def exists(key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists
        """
        if not redis_client:
            return False
        
        try:
            return await redis_client.exists(key) > 0
        except Exception as e:
            logger.error("Cache exists error", key=key, error=str(e))
            return False


def cache_key_wrapper(prefix: str, ttl: int = 300):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Only use cache if enabled
            if not settings.is_feature_enabled("caching"):
                return await func(*args, **kwargs)
            
            # Generate cache key
            key_parts = [prefix]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached = await CacheManager.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit", key=cache_key)
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await CacheManager.set(cache_key, result, ttl)
            logger.debug("Cache set", key=cache_key)
            
            return result
        
        return wrapper
    return decorator
