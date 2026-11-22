"""Minimal stdlib HTTP client for the internal API."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ApiError(Exception):
    pass


class Client:
    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raise ApiError(f"{path}: HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ApiError(f"{path}: {e}") from e
