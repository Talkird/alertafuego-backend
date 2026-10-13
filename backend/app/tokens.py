"""One-time tokens for emailed and Telegram links. Only a token's SHA-256 is stored,
so a database leak doesn't leak working links."""

import hashlib
import secrets
from datetime import datetime, timezone


def new_token() -> str:
    """32 URL-safe characters: fits Telegram's 64-character /start parameter."""
    return secrets.token_urlsafe(24)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def utcnow() -> datetime:
    """Naive UTC, as the database stores timestamps."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
