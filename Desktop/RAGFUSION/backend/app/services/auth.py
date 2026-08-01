"""Authentication service with JWT, bcrypt, and role management."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.core.exceptions import (ConflictError, UnauthorizedError,
                                 ValidationError)
from app.models.entities import User, UserRole
from app.schemas.auth import (LoginRequest, RefreshRequest, SignupRequest,
                              TokenResponse, UserResponse)

# Use bcrypt for secure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Validates password requirements:
    - Minimum 8 characters
    - Maximum 72 characters (bcrypt limitation)
    """
    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters long")
    if len(password) > 72:
        raise ValidationError("Password too long (max 72 characters)")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: UUID, role: UserRole) -> str:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    to_encode = {
        "sub": str(user_id),
        "role": role.value,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_refresh_token(user_id: UUID) -> str:
    """Create a JWT refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    to_encode = {
        "sub": str(user_id),
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as e:
        raise UnauthorizedError("Invalid or expired token") from e


def create_token_response(user: User) -> TokenResponse:
    """Create a token response for a user."""
    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


async def signup(db: AsyncSession, data: SignupRequest) -> TokenResponse:
    """Register a new user."""
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise ConflictError("Email already registered")

    password_hash = hash_password(data.password)
    user = User(
        email=data.email,
        display_name=data.display_name,
        password_hash=password_hash,
        role=UserRole.user,
    )
    db.add(user)
    await db.flush()
    return create_token_response(user)


async def login(db: AsyncSession, data: LoginRequest) -> TokenResponse:
    """Authenticate a user and return tokens."""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    if not user or not user.password_hash:
        raise UnauthorizedError("Invalid credentials")

    if not verify_password(data.password, user.password_hash):
        raise UnauthorizedError("Invalid credentials")

    if not user.is_active:
        raise UnauthorizedError("Account is deactivated")

    return create_token_response(user)


async def refresh(db: AsyncSession, data: RefreshRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    payload = decode_token(data.refresh_token)

    if payload.get("type") != "refresh":
        raise UnauthorizedError("Invalid token type")

    user_id = UUID(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise UnauthorizedError("User not found or inactive")

    return create_token_response(user)


async def get_current_user(db: AsyncSession, token: str) -> User:
    """Get current user from access token."""
    payload = decode_token(token)

    if payload.get("type") != "access":
        raise UnauthorizedError("Invalid token type")

    user_id = UUID(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise UnauthorizedError("User not found or inactive")

    return user


def user_to_response(user: User) -> UserResponse:
    """Convert User entity to UserResponse schema."""
    return UserResponse.model_validate(user)
