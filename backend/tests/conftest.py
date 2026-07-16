"""Shared pytest configuration.

Integration tests requiring real PostgreSQL/MinIO/Qdrant validate a dedicated
test namespace through tests.support.resource_guard before any mutation.
"""

from __future__ import annotations
