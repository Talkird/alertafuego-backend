"""Request authentication: Supabase JWT verification (asymmetric JWT Signing Keys via
JWKS) for users, and a shared API key for the detection-run endpoints."""

import hmac
import os
import uuid
from dataclasses import dataclass

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer()
#: For public endpoints that do more for a signed-in user.
_optional_bearer = HTTPBearer(auto_error=False)
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


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_optional_bearer),
) -> uuid.UUID | None:
    """The user, or None without a token. A token that is sent but invalid is a 401."""
    return None if credentials is None else uuid.UUID(_decode(credentials)["sub"])


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


#: auto_error=False: a missing header gets the same 401 as a wrong one.
_detect_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_detect_key(key: str | None = Depends(_detect_key_header)) -> None:
    """Only callers holding DETECT_API_KEY (the operator, a scheduler) may run
    detections: each run costs Lambda time and Earth Engine quota, writes
    detections and can alert users. The frontend never calls these endpoints, so
    the key never reaches a browser. Fails closed when the key isn't configured."""
    load_dotenv()
    expected = os.getenv("DETECT_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="DETECT_API_KEY is not configured")
    if key is None or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
