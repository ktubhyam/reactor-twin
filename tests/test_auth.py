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
