"""
[IMPLEMENTED] Authentication routes: user registration, JWT login, and profile retrieval.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend.database.models_orm import User
from backend.database.repositories.user_repo import user_repo
from backend.dependencies import get_current_active_user, get_db
from backend.security.auth import create_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])


# --- Schemas ---
class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class JSONLoginRequest(BaseModel):
    username_or_email: str
    password: str


# --- Endpoints ---
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with unique email and username."""
    if user_repo.get_by_email(db, req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )
    if user_repo.get_by_username(db, req.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists."
        )

    user = user_repo.create(
        db=db,
        email=req.email,
        username=req.username,
        password=req.password,
        full_name=req.full_name
    )
    return user


@router.post("/login", response_model=TokenResponse)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 compatible token login, returning JWT access token."""
    user = user_repo.authenticate(
        db=db,
        username_or_email=form_data.username,
        password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account."
        )

    access_token = create_access_token(data={"sub": user.username, "email": user.email, "id": user.id})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.post("/login-json", response_model=TokenResponse)
def login_with_json(
    req: JSONLoginRequest,
    db: Session = Depends(get_db)
):
    """JSON payload token login for frontend REST clients."""
    user = user_repo.authenticate(
        db=db,
        username_or_email=req.username_or_email,
        password=req.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account."
        )

    access_token = create_access_token(data={"sub": user.username, "email": user.email, "id": user.id})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.get("/me", response_model=UserResponse)
def read_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Retrieve profile of currently authenticated user."""
    return current_user
