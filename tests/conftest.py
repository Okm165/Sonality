"""Pytest configuration for Sonality tests."""

from __future__ import annotations

import os

os.environ.setdefault("SONALITY_BASE_URL", "http://localhost:8080/v1")
os.environ.setdefault("SONALITY_MODEL", "test-model")
os.environ.setdefault("FATHOM_BASE_URL", "http://localhost:8080/v1")
os.environ.setdefault("FATHOM_MODEL", "test-model")
