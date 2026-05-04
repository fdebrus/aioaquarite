"""Tests for AquariteAuth.user_id JWT decoding."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aioaquarite.auth import AquariteAuth
from aioaquarite.exceptions import AquariteError


def _make_jwt(payload: dict[str, Any]) -> str:
    """Build a header.payload.signature string with urlsafe-b64 segments."""

    def b64(raw: bytes) -> str:
        # Strip padding to mimic real JWTs (which omit `=` padding).
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    sig = b64(b"not-a-real-signature")
    return f"{header}.{body}.{sig}"


@pytest.fixture
def auth() -> AquariteAuth:
    session = MagicMock()
    return AquariteAuth(session, "user@example.com", "hunter2")


def test_user_id_none_before_authenticate(auth: AquariteAuth) -> None:
    assert auth.tokens is None
    assert auth.user_id is None


def test_user_id_returns_sub_after_authenticate(auth: AquariteAuth) -> None:
    token = _make_jwt(
        {"sub": "abc123", "email": "x@y.com", "iat": 1700000000, "exp": 1700003600}
    )

    async def fake_signin() -> None:
        auth.tokens = {
            "idToken": token,
            "refreshToken": "rt",
            "expiresIn": "3600",
        }

    with patch.object(auth, "_signin", side_effect=fake_signin):
        asyncio.run(auth.authenticate())

    assert auth.user_id == "abc123"


def test_user_id_updates_after_token_refresh(auth: AquariteAuth) -> None:
    first = _make_jwt({"sub": "user-old", "email": "x@y.com"})
    second = _make_jwt({"sub": "user-new", "email": "x@y.com"})

    auth.tokens = {"idToken": first, "refreshToken": "rt", "expiresIn": "3600"}
    assert auth.user_id == "user-old"
    # Sanity check: cached on repeat access for the same token.
    assert auth.user_id == "user-old"

    auth.tokens["idToken"] = second
    assert auth.user_id == "user-new"


def test_user_id_with_unpadded_segment(auth: AquariteAuth) -> None:
    # Construct a payload whose base64 encoding requires padding.
    # {"sub":"a"} -> 10 bytes -> base64 length 16 chars with padding "=" * 2 stripped.
    token = _make_jwt({"sub": "a"})
    auth.tokens = {"idToken": token, "refreshToken": "rt", "expiresIn": "3600"}
    assert auth.user_id == "a"


def test_user_id_malformed_token_raises(auth: AquariteAuth) -> None:
    auth.tokens = {
        "idToken": "not-a-jwt",
        "refreshToken": "rt",
        "expiresIn": "3600",
    }
    with pytest.raises(AquariteError):
        _ = auth.user_id


def test_user_id_missing_sub_raises(auth: AquariteAuth) -> None:
    token = _make_jwt({"email": "x@y.com"})
    auth.tokens = {"idToken": token, "refreshToken": "rt", "expiresIn": "3600"}
    with pytest.raises(AquariteError):
        _ = auth.user_id


def test_user_id_bad_base64_raises(auth: AquariteAuth) -> None:
    auth.tokens = {
        "idToken": "header.!!!notbase64!!!.sig",
        "refreshToken": "rt",
        "expiresIn": "3600",
    }
    with pytest.raises(AquariteError):
        _ = auth.user_id


def test_user_id_bad_json_raises(auth: AquariteAuth) -> None:
    # Valid base64, but the decoded bytes aren't JSON.
    bogus_payload = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode("ascii")
    auth.tokens = {
        "idToken": f"hdr.{bogus_payload}.sig",
        "refreshToken": "rt",
        "expiresIn": "3600",
    }
    with pytest.raises(AquariteError):
        _ = auth.user_id
