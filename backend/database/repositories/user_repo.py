"""
[IMPLEMENTED] User database repository.
Handles user creation, lookup, and authentication queries.
"""
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.database.models_orm import User
from backend.security.auth import hash_password, verify_password


class UserRepository:
    @staticmethod
    def create(
        db: Session,
        email: str,
        username: str,
        password: str,
        full_name: str | None = None,
        is_superuser: bool = False
    ) -> User:
        """Create and persist a new user."""
        hashed = hash_password(password)
        user = User(
            email=email.lower().strip(),
            username=username.strip(),
            hashed_password=hashed,
            full_name=full_name,
            is_superuser=is_superuser,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> User | None:
        """Fetch user by primary key."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> User | None:
        """Fetch user by email address."""
        return db.query(User).filter(User.email == email.lower().strip()).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> User | None:
        """Fetch user by username."""
        return db.query(User).filter(User.username == username.strip()).first()

    @staticmethod
    def authenticate(db: Session, username_or_email: str, password: str) -> User | None:
        """Authenticate user by matching either username or email and verifying password."""
        identifier = username_or_email.strip()
        user = db.query(User).filter(
            or_(User.email == identifier.lower(), User.username == identifier)
        ).first()

        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    def list_all(db: Session, skip: int = 0, limit: int = 100) -> list[User]:
        """List active users."""
        return db.query(User).offset(skip).limit(limit).all()


user_repo = UserRepository()
