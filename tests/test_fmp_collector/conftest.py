"""Common test fixtures and configuration."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["FMP_API_KEY"] = "test_api_key"
    os.environ["REDIS_PASSWORD"] = "test_redis_password"
    yield
    os.environ.pop("FMP_API_KEY", None)
    os.environ.pop("REDIS_PASSWORD", None)

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "1d").mkdir()
    (data_dir / "5min").mkdir()
    (data_dir / "calendars").mkdir()
    (data_dir / "instruments").mkdir()
    
    return data_dir

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch("redis.from_url") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    with patch("aiohttp.ClientSession") as mock:
        session = AsyncMock()
        mock.return_value = session
        yield session