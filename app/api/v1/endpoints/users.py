"""
User authentication and management endpoints (Tier 2+).
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt

from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# ==========================================
# Security Configuration
# ==========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_v1_prefix}/users/token")

# In-memory user storage (replace with database in production)
users_db: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow()
    },
    "user": {
        "username": "user",
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("user123"),
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow()
    }
}


# ==========================================
# Schemas
# ==========================================
class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    """Schema for creating users."""
    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    """Schema for user responses."""
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None


# ==========================================
# Utility Functions
# ==========================================
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user."""
    user = users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ==========================================
# Endpoints
# ==========================================
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user"
)
async def register(user: UserCreate) -> UserResponse:
    """
    Register a new user account.
    """
    # Check if user exists
    if user.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    for existing_user in users_db.values():
        if existing_user["email"] == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Create new user
    user_dict = user.model_dump()
    hashed_password = get_password_hash(user_dict.pop("password"))
    
    new_user = {
        **user_dict,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    users_db[user.username] = new_user
    
    logger.info("User registered", username=user.username, email=user.email)
    
    return new_user


@router.post(
    "/token",
    response_model=Token,
    summary="Login to get access token"
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """
    OAuth2 compatible token login.
    
    Get an access token for future requests.
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning("Failed login attempt", username=form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    logger.info("User logged in", username=user["username"])
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60
    }


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user"
)
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """
    Get the current authenticated user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    
    return user


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout user"
)
async def logout(token: str = Depends(oauth2_scheme)) -> None:
    """
    Logout the current user.
    
    Note: In a stateless JWT implementation, logout is typically handled client-side
    by removing the token. This endpoint is provided for audit logging purposes.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        logger.info("User logged out", username=username)
    except JWTError:
        pass
