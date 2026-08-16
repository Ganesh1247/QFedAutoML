"""
[IMPLEMENTED] Dependency injection providers for FastAPI routes (Database sessions, auth, settings).
"""
from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config import Settings, settings
from backend.database.connection import SessionLocal
from backend.database.models_orm import User
from backend.database.repositories.user_repo import user_repo
from backend.security.auth import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login", auto_error=False)


def get_settings() -> Settings:
    """Dependency to inject application settings."""
    return settings


def get_db() -> Generator[Session, None, None]:
    """Dependency to yield a managed database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    db: Session = Depends(get_db),
    token: str | None = Depends(oauth2_scheme)
) -> User:
    """Validate JWT bearer token and retrieve current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = user_repo.get_by_username(db, username=username)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is currently active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    return current_user
