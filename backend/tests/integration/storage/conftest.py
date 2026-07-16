"""Storage integration test configuration."""

from __future__ import annotations

import pytest

from tests.support.resource_guard import assert_safe_test_environment

assert_safe_test_environment()

pytestmark = pytest.mark.asyncio(scope="session")
