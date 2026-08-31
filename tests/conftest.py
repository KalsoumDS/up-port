"""Pytest configuration and fixtures"""

import pytest


@pytest.fixture
def sample_data() -> dict:
    """Fixture: sample test data"""
    return {
        "value": 42.0,
        "timestamp": "2024-01-01T00:00:00",
    }
