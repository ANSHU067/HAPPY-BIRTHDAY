"""Authentication routes."""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.auth import (LoginRequest, LogoutResponse, RefreshRequest,
                              SignupRequest, TokenResponse, UserResponse)
from app.services.auth import (decode_token, get_current_user, login, refresh,
                               signup, user_to_response)

router = APIRouter(prefix="/auth", tags=["authentication"])


async def get_current_user_id(
    authorization: str = Header(..., alias="Authorization"),
    db: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    """Extract user from Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )

    token = authorization[7:]
    user = await get_current_user(db, token)
    return user_to_response(user)


@router.post(
    "/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
async def signup_endpoint(
    data: SignupRequest, db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """Register a new user and return access/refresh tokens."""
    return await signup(db, data)


@router.post("/login", response_model=TokenResponse)
async def login_endpoint(
    data: LoginRequest, db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """Authenticate user and return access/refresh tokens."""
    return await login(db, data)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_endpoint(
    data: RefreshRequest, db: AsyncSession = Depends(get_db_session)
) -> TokenResponse:
    """Refresh access token using refresh token."""
    return await refresh(db, data)


@router.post("/logout", response_model=LogoutResponse)
async def logout_endpoint(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> LogoutResponse:
    """Logout user (client-side token invalidation)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )

    token = authorization[7:]
    try:
        decode_token(token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    return LogoutResponse()


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: UserResponse = Depends(get_current_user_id),
) -> UserResponse:
    """Get current authenticated user."""
    return current_user
