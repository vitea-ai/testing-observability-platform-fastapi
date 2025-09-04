"""
Common dependencies for FastAPI endpoints.
"""

from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db as get_database_session

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/users/token",
    auto_error=False
)


async def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """
    Get current user if token is provided, otherwise return None.
    
    This is useful for endpoints that work both authenticated and unauthenticated.
    """
    if not token:
        return None
    
    if not settings.is_feature_enabled("authentication"):
        return None
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        
        # In production, fetch user from database
        from app.api.v1.endpoints.users import users_db
        user = users_db.get(username)
        return user
    except JWTError:
        return None


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Get current authenticated user or raise exception.
    
    This is used for endpoints that require authentication.
    """
    if not settings.is_feature_enabled("authentication"):
        # Return mock user in development if auth is disabled
        return {"username": "dev_user", "is_superuser": True}
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # In production, fetch user from database
    from app.api.v1.endpoints.users import users_db
    user = users_db.get(username)
    
    if user is None:
        raise credentials_exception
    
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_superuser(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current user and verify they are a superuser.
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def apply_rate_limit() -> None:
    """
    Apply rate limiting if enabled.
    
    This is a placeholder for rate limiting logic.
    In production, use Redis or similar for distributed rate limiting.
    """
    if not settings.is_feature_enabled("rate_limiting"):
        return
    
    # Placeholder implementation
    # In production, implement proper rate limiting with Redis
    pass


async def audit_log(
    action: str,
    user: str = "anonymous",
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log audit events if audit logging is enabled.
    
    Args:
        action: The action being performed
        user: Username performing the action
        resource_id: ID of the resource being acted upon
        details: Additional details about the action
    """
    if not settings.is_feature_enabled("audit_logging"):
        return
    
    logger.info(
        "AUDIT",
        action=action,
        user=user,
        resource_id=resource_id,
        details=details,
        tier=settings.deployment_tier
    )
    
    # In production, write to audit log database or service
    # await write_audit_log_to_database(action, user, resource_id, details)


# Export database dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.
    
    This is a wrapper around the session manager's get_db function
    to provide a clean import for endpoints.
    """
    async for session in get_database_session():
        yield session
