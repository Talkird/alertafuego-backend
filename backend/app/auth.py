"""Supabase JWT verification dependency (asymmetric JWT Signing Keys via JWKS)."""

import os
import uuid
from dataclasses import dataclass

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer()
_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_url:
            raise RuntimeError("SUPABASE_URL is not set in .env.")
        _jwks_client = jwt.PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def _decode(credentials: HTTPAuthorizationCredentials) -> dict:
    token = credentials.credentials
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=["ES256", "RS256"], audience="authenticated")
    except jwt.PyJWKClientConnectionError as exc:
        # Couldn't reach Supabase's JWKS: our problem, not the token's. A 401 here
        # would log every user out during a Supabase outage.
        raise HTTPException(status_code=503, detail="Auth key server unavailable") from exc
    except (jwt.InvalidTokenError, jwt.PyJWKClientError) as exc:
        # PyJWKClientError: no signing key matches the token's kid (forged, or
        # signed with a rotated-out key). Was an unhandled 500.
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> uuid.UUID:
    return uuid.UUID(_decode(credentials)["sub"])


@dataclass(frozen=True)
class UserWithEmail:
    id: uuid.UUID
    email: str


def get_current_user_with_email(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> UserWithEmail:
    """Like get_current_user, plus the `email` claim - for endpoints that need to
    email the user later (alert areas)."""
    payload = _decode(credentials)
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Your account has no email address")
    return UserWithEmail(id=uuid.UUID(payload["sub"]), email=email)
