"""
[IMPLEMENTED] Unit tests for authentication, password hashing, and JWT tokens.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import Base
from backend.dependencies import get_db
from backend.main import app
from backend.security.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

# Test in-memory database with StaticPool so all sessions share the memory DB
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)


def test_password_hashing_and_verification():
    """Verify bcrypt hashes passwords correctly and validates matching input."""
    raw = "MySecretPass123!"
    hashed = hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_encode_decode():
    """Verify JWT token encoding and decoding."""
    payload_data = {"sub": "alice", "email": "alice@example.com"}
    token = create_access_token(payload_data)
    assert isinstance(token, str)

    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "alice"
    assert decoded["email"] == "alice@example.com"
    assert "exp" in decoded


def test_auth_register_and_login_flow():
    """Verify full registration, login, and protected profile retrieval flow."""
    # 1. Register
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": "user1@qfed.org",
            "username": "user1",
            "password": "Password123!",
            "full_name": "User One"
        }
    )
    assert reg_resp.status_code == 201
    user_data = reg_resp.json()
    assert user_data["username"] == "user1"
    assert user_data["email"] == "user1@qfed.org"

    # 2. Duplicate registration fails
    dup_resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": "user1@qfed.org",
            "username": "user1_diff",
            "password": "Password123!"
        }
    )
    assert dup_resp.status_code == 400

    # 3. JSON Login
    login_resp = client.post(
        "/api/v1/auth/login-json",
        json={
            "username_or_email": "user1@qfed.org",
            "password": "Password123!"
        }
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 4. Form-data Login (OAuth2 standard)
    form_login_resp = client.post(
        "/api/v1/auth/login",
        data={
            "username": "user1",
            "password": "Password123!"
        }
    )
    assert form_login_resp.status_code == 200
    assert "access_token" in form_login_resp.json()

    # 5. Invalid password login fails
    fail_login = client.post(
        "/api/v1/auth/login-json",
        json={
            "username_or_email": "user1",
            "password": "IncorrectPassword"
        }
    )
    assert fail_login.status_code == 401

    # 6. Access protected route /auth/me with valid Bearer token
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_resp.status_code == 200
    profile = me_resp.json()
    assert profile["username"] == "user1"
    assert profile["email"] == "user1@qfed.org"

    # 7. Access protected route without token fails
    unauth_resp = client.get("/api/v1/auth/me")
    assert unauth_resp.status_code == 401
