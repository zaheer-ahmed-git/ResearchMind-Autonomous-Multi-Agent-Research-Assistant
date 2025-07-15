"""Pytest configuration helpers."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
