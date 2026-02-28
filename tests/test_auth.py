"""Tests for reactor_twin.api.auth."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from reactor_twin.api.auth import (
    RateLimiter,
    create_token,
    verify_token,
)


class TestJWT:
    def test_create_and_verify(self):
        token = create_token("testuser")
        payload = verify_token(token)
        assert payload["sub"] == "testuser"
        assert "iat" in payload
        assert "exp" in payload

    def test_extra_claims(self):
        token = create_token("user1", extra_claims={"role": "admin"})
        payload = verify_token(token)
        assert payload["role"] == "admin"

    def test_invalid_token_raises(self):
        with pytest.raises(HTTPException):
            verify_token("invalid.token.here")

    def test_tampered_token_raises(self):
        token = create_token("user1")
        parts = token.split(".")
        # Tamper with the payload
        parts[1] = parts[1] + "x"
        tampered = ".".join(parts)
        with pytest.raises(HTTPException):
            verify_token(tampered)


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter(limit=5, window=60.0)

        # Create a mock request
        class MockClient:
            host = "127.0.0.1"

        class MockRequest:
            headers = {}
            client = MockClient()

        req = MockRequest()
        for _ in range(5):
            limiter.check(req)  # should not raise

    def test_blocks_over_limit(self):
        limiter = RateLimiter(limit=3, window=60.0)

        class MockClient:
            host = "10.0.0.1"

        class MockRequest:
            headers = {}
            client = MockClient()

        req = MockRequest()
        for _ in range(3):
            limiter.check(req)

        with pytest.raises(HTTPException):
            limiter.check(req)


class TestTokenExpiry:
    def test_expired_token_raises(self):
        import time
        from unittest.mock import patch

        # Create a token that expires immediately
        with patch("reactor_twin.api.auth._TOKEN_EXPIRY_SECONDS", 0):
            token = create_token("testuser")
        time.sleep(0.1)
        with pytest.raises(HTTPException, match="Token expired"):
            verify_token(token)

    def test_short_format_token_raises(self):
        with pytest.raises(HTTPException, match="Invalid token format"):
            verify_token("only.two")


class TestRequireAuth:
    def test_empty_token_raises(self):
        """verify_token with empty string raises HTTPException."""
        with pytest.raises(HTTPException):
            verify_token("")

    def test_malformed_token_raises(self):
        """verify_token with bad format raises."""
        with pytest.raises(HTTPException):
            verify_token("not-a-jwt")


class TestRateLimiterClientKey:
    def test_ip_from_x_forwarded_for(self):
        limiter = RateLimiter(limit=100, window=60.0)

        class MockClient:
            host = "127.0.0.1"

        class MockRequest:
            headers = {"X-Forwarded-For": "203.0.113.50, 70.41.3.18"}
            client = MockClient()

        req = MockRequest()
        key = limiter._client_key(req)
        assert key == "ip:203.0.113.50"

    def test_ip_from_client(self):
        limiter = RateLimiter(limit=100, window=60.0)

        class MockClient:
            host = "192.168.1.1"

        class MockRequest:
            headers = {}
            client = MockClient()

        req = MockRequest()
        key = limiter._client_key(req)
        assert key == "ip:192.168.1.1"

    def test_authenticated_user_key(self):
        limiter = RateLimiter(limit=100, window=60.0)
        token = create_token("api_user")

        class MockClient:
            host = "127.0.0.1"

        class MockRequest:
            headers = {"Authorization": f"Bearer {token}"}
            client = MockClient()

        req = MockRequest()
        key = limiter._client_key(req)
        assert key == "user:api_user"

    def test_no_client(self):
        limiter = RateLimiter(limit=100, window=60.0)

        class MockRequest:
            headers = {}
            client = None

        req = MockRequest()
        key = limiter._client_key(req)
        assert key == "ip:unknown"

    def test_invalid_bearer_token_falls_back_to_ip(self):
        """When Bearer token verification fails, _client_key falls back to
        IP-based key (auth.py lines 151-152)."""
        import base64

        limiter = RateLimiter(limit=100, window=60.0)

        # Build a structurally valid JWT (3 base64url segments) but with
        # a wrong signature so verify_token raises HTTPException.
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b'{"sub":"bad","exp":9999999999}').rstrip(b"=").decode()
        bad_sig = base64.urlsafe_b64encode(b"invalidsignaturebytes").rstrip(b"=").decode()
        bad_token = f"{header}.{payload}.{bad_sig}"

        class MockClient:
            host = "10.0.0.42"

        class MockRequest:
            headers = {"Authorization": f"Bearer {bad_token}"}
            client = MockClient()

        req = MockRequest()
        key = limiter._client_key(req)
        # Token is invalid, so should fall back to IP
        assert key == "ip:10.0.0.42"


# ── require_auth missing header (auth.py line 125) ───────────────


class TestRequireAuthMissingHeader:
    """Test require_auth dependency when no Authorization header is sent."""

    def test_missing_auth_header_returns_401(self):
        """Endpoints that call require_auth should return 401 when no
        Authorization header is provided (auth.py line 125)."""
        from starlette.testclient import TestClient

        from reactor_twin.api.server import app

        client = TestClient(app)
        # Hit an endpoint that requires auth, without any Authorization header
        resp = client.post(
            "/api/v2/models/upload",
            content=b"some data",
            # No headers — no Authorization
        )
        assert resp.status_code == 401
        assert "Missing authorization" in resp.json()["detail"]

    def test_list_models_no_auth_returns_401(self):
        """GET /api/v2/models requires auth; omitting header gives 401."""
        from starlette.testclient import TestClient

        from reactor_twin.api.server import app

        client = TestClient(app)
        resp = client.get("/api/v2/models")
        assert resp.status_code == 401
