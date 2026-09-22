"""
backend/admin_auth.py — single source of truth for admin HTTP-Basic auth.

Previously duplicated: main.py hardcoded ADMIN_PASSWORD = "111111" directly
while admin_knowledge.py read it from os.getenv("ADMIN_PASSWORD", "111111").
Setting the env var only changed the knowledge-editing endpoints' password,
not the dashboard's — a real, silent inconsistency. Both modules now import
from here instead, so there is exactly ONE place credentials are read from.
"""
from __future__ import annotations

import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "111111")


def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={},
        )
    return credentials.username