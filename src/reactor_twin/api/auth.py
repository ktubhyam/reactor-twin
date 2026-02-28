"""JWT authentication and rate limiting middleware for the ReactorTwin API.

Requires the [api] optional dependencies:
    pip install reactor-twin[api]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any

try:
    from fastapi import HTTPException, Request
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the API server. Install with: pip install reactor-twin[api]"
    ) from exc

import base64

logger = logging.getLogger(__name__)

# ── JWT (HS256, no external dependency) ──────────────────────────────

_SECRET_KEY = os.environ.get("REACTOR_TWIN_JWT_SECRET", "reactor-twin-dev-secret")
_TOKEN_EXPIRY_SECONDS = int(os.environ.get("REACTOR_TWIN_TOKEN_EXPIRY", "3600"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def create_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """Create a signed JWT token.

    Args:
        subject: Token subject (e.g. username or API key ID).
        extra_claims: Additional claims to include.

    Returns:
        Encoded JWT string.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + _TOKEN_EXPIRY_SECONDS,
    }
    if extra_claims:
        payload.update(extra_claims)

    segments = [
        _b64url_encode(json.dumps(header).encode()),
        _b64url_encode(json.dumps(payload).encode()),
    ]
    signing_input = f"{segments[0]}.{segments[1]}"
    signature = hmac.new(_SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def verify_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token format")

    signing_input = f"{parts[0]}.{parts[1]}"
    expected_sig = hmac.new(
        _SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    actual_sig = _b64url_decode(parts[2])

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload = json.loads(_b64url_decode(parts[1]))

    if payload.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Token expired")

    return payload


# ── FastAPI dependency ───────────────────────────────────────────────

_security = HTTPBearer(auto_error=False)


async def require_auth(request: Request) -> dict[str, Any]:
    """FastAPI dependency that requires a valid JWT Bearer token.

    Usage::

        @app.get("/protected")
        async def protected(user: dict = Depends(require_auth)):
            return {"user": user["sub"]}
    """
    auth: HTTPAuthorizationCredentials | None = await _security(request)
    if auth is None:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    return verify_token(auth.credentials)


# ── Rate limiter (in-memory token bucket) ────────────────────────────

_RATE_LIMIT = int(os.environ.get("REACTOR_TWIN_RATE_LIMIT", "60"))  # requests per minute
_RATE_WINDOW = 60.0  # seconds


class RateLimiter:
    """Simple in-memory rate limiter using a sliding window counter."""

    def __init__(self, limit: int = _RATE_LIMIT, window: float = _RATE_WINDOW) -> None:
        self.limit = limit
        self.window = window
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use Authorization header subject if available, else IP
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                payload = verify_token(auth[7:])
                return f"user:{payload.get('sub', 'unknown')}"
            except HTTPException:
                pass
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def check(self, request: Request) -> None:
        """Check rate limit for a request. Raises HTTPException if exceeded."""
        key = self._client_key(request)
        now = time.time()
        cutoff = now - self.window

        # Clean old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({self.limit} requests per {self.window:.0f}s)",
            )

        self._requests[key].append(now)


# Global rate limiter instance
rate_limiter = RateLimiter()


__all__ = [
    "create_token",
    "verify_token",
    "require_auth",
    "RateLimiter",
    "rate_limiter",
]
