"""
firebase_manager.py
-------------------
Firebase Authentication (REST API) + Firestore (REST API) integration.

No extra packages required — uses the existing `requests` library.

Required env vars:
    FIREBASE_API_KEY    — Web API key (Firebase Console → Project Settings → General)
    FIREBASE_PROJECT_ID — Project ID  (Firebase Console → Project Settings → General)

If not configured, `is_configured()` returns False and all operations are no-ops.

Firestore security rules (set in Firebase Console) should allow:
    match /portfolios/{userId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
    }
"""

from __future__ import annotations

import json
import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_API_KEY    = os.getenv("FIREBASE_API_KEY", "")
_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

_AUTH_BASE      = "https://identitytoolkit.googleapis.com/v1/accounts"
_FIRESTORE_BASE = "https://firestore.googleapis.com/v1"


class FirebaseError(Exception):
    pass


class FirebaseNotConfiguredError(FirebaseError):
    pass


# ---------------------------------------------------------------------------
# Configuration check
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """Return True if Firebase env vars are set."""
    return bool(_API_KEY and _PROJECT_ID)


# ---------------------------------------------------------------------------
# Authentication (Firebase Auth REST API)
# ---------------------------------------------------------------------------

def register_user(email: str, password: str) -> dict:
    """Create a new user account with email + password.

    Returns {uid, email, idToken} on success.
    Raises FirebaseError on failure.
    """
    if not is_configured():
        raise FirebaseNotConfiguredError(
            "FIREBASE_API_KEY and FIREBASE_PROJECT_ID must be set to use authentication."
        )

    resp = requests.post(
        f"{_AUTH_BASE}:signUp?key={_API_KEY}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "Registration failed")
        if "EMAIL_EXISTS" in msg:
            raise FirebaseError("An account with this email already exists.")
        if "WEAK_PASSWORD" in msg:
            raise FirebaseError("Password must be at least 6 characters.")
        if "INVALID_EMAIL" in msg:
            raise FirebaseError("Invalid email address.")
        raise FirebaseError(msg)

    logger.info("firebase: new user registered — %s", data.get("email"))
    return {
        "uid":     data["localId"],
        "email":   data["email"],
        "idToken": data["idToken"],
    }


def login_user(email: str, password: str) -> dict:
    """Sign in with email + password.

    Returns {uid, email, idToken} on success.
    Raises FirebaseError on failure.
    """
    if not is_configured():
        raise FirebaseNotConfiguredError(
            "FIREBASE_API_KEY and FIREBASE_PROJECT_ID must be set to use authentication."
        )

    resp = requests.post(
        f"{_AUTH_BASE}:signInWithPassword?key={_API_KEY}",
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "Login failed")
        if any(x in msg for x in ("INVALID_LOGIN_CREDENTIALS", "EMAIL_NOT_FOUND",
                                    "WRONG_PASSWORD", "INVALID_PASSWORD")):
            raise FirebaseError("Invalid email or password.")
        if "USER_DISABLED" in msg:
            raise FirebaseError("This account has been disabled.")
        raise FirebaseError(msg)

    logger.info("firebase: user signed in — %s", data.get("email"))
    return {
        "uid":     data["localId"],
        "email":   data["email"],
        "idToken": data["idToken"],
    }


def sign_in_with_google(google_id_token: str) -> dict:
    """Exchange a Google ID token for a Firebase session.

    The ID token comes from Google Identity Services (client-side button).
    Firebase's signInWithIdp REST endpoint verifies it and returns a
    Firebase idToken that can be used for Firestore calls.

    Returns {uid, email, idToken} on success.
    Raises FirebaseError on failure.
    """
    if not is_configured():
        raise FirebaseNotConfiguredError(
            "FIREBASE_API_KEY and FIREBASE_PROJECT_ID must be set."
        )

    resp = requests.post(
        f"{_AUTH_BASE}:signInWithIdp?key={_API_KEY}",
        json={
            "postBody":            f"id_token={google_id_token}&providerId=google.com",
            "requestUri":          "http://localhost",
            "returnIdpCredential": True,
            "returnSecureToken":   True,
        },
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        msg = data["error"].get("message", "Google sign-in failed")
        raise FirebaseError(f"Google sign-in failed: {msg}")

    logger.info("firebase: Google sign-in — %s", data.get("email"))
    return {
        "uid":     data["localId"],
        "email":   data.get("email", ""),
        "idToken": data["idToken"],
    }


# ---------------------------------------------------------------------------
# Firestore portfolio storage (REST API)
# ---------------------------------------------------------------------------
# Portfolio document path: portfolios/{uid}
# Document format: { "fields": { "data": { "stringValue": "<JSON string>" } } }
# Storing as a JSON string in a single field avoids complex Firestore type mapping.

def _doc_url(uid: str) -> str:
    return (
        f"{_FIRESTORE_BASE}/projects/{_PROJECT_ID}"
        f"/databases/(default)/documents/portfolios/{uid}"
    )


def get_portfolio(uid: str, id_token: str) -> dict | None:
    """Fetch the user's portfolio from Firestore.

    Returns a portfolio dict (same schema as my_portfolio.json) or None if
    no portfolio has been saved yet (or on error).
    """
    if not is_configured():
        return None

    resp = requests.get(
        _doc_url(uid),
        headers={"Authorization": f"Bearer {id_token}"},
        timeout=10,
    )
    if resp.status_code == 404:
        return None  # first time — no portfolio yet
    if not resp.ok:
        logger.warning("firebase: get_portfolio failed %d — %s", resp.status_code, resp.text[:200])
        return None

    fields = resp.json().get("fields", {})
    raw = fields.get("data", {}).get("stringValue", "")
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("firebase: portfolio JSON parse error")
        return None


def save_portfolio(uid: str, id_token: str, portfolio: dict) -> bool:
    """Persist the user's portfolio to Firestore.

    Returns True on success, False on failure.
    """
    if not is_configured():
        return False

    body = {
        "fields": {
            "data": {"stringValue": json.dumps(portfolio, ensure_ascii=False)}
        }
    }
    resp = requests.patch(
        _doc_url(uid),
        headers={
            "Authorization": f"Bearer {id_token}",
            "Content-Type":  "application/json",
        },
        json=body,
        timeout=10,
    )
    if not resp.ok:
        logger.warning("firebase: save_portfolio failed %d — %s", resp.status_code, resp.text[:200])
    return resp.ok
