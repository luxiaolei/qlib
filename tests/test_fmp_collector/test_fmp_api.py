"""Tests for FMP API client."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from scripts.data_collector.fmp.fmp_api import FMPClient, RedisRateLimiter

# Load test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Load and convert data to records format
DAILY_DATA = pd.read_json(FIXTURES_DIR / "daily_data.json")
FIVEMIN_DATA = pd.read_json(FIXTURES_DIR / "fivemin_data.json")
CONSTITUENTS = pd.read_json(FIXTURES_DIR / "constituents.json")
HISTORICAL_CONSTITUENTS = pd.read_json(FIXTURES_DIR / "historical_constituents.json")

# Convert to records format for API response mocking
DAILY_RECORDS = [
    {
        "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"],
        "adjClose": row["adjClose"]
    }
    for _, row in DAILY_DATA.iterrows()
]

FIVEMIN_RECORDS = [
    {
        "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d %H:%M:%S"),
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"]
    }
    for _, row in FIVEMIN_DATA.iterrows()
]

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    with patch("redis.from_url") as mock:
        client = MagicMock()
        pipeline = MagicMock()
        pipeline.execute.return_value = [1, 0, 1, 1]  # Mock pipeline results
        client.pipeline.return_value = pipeline
        mock.return_value = client
        yield client

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    with patch("aiohttp.ClientSession") as mock:
        session = AsyncMock()
        mock.return_value = session
        yield session

@pytest.fixture
async def fmp_client(mock_redis, mock_aiohttp_session):
    """Create a FMP client with mocked dependencies."""
    os.environ["FMP_API_KEY"] = "test_api_key"
    os.environ["REDIS_PASSWORD"] = "test_redis_password"
    
    client = FMPClient()
    await client.__aenter__()
    try:
        yield client
    finally:
        await client.__aexit__(None, None, None)

class TestRedisRateLimiter:
    """Tests for Redis-based rate limiter."""
    
    @pytest.mark.asyncio
    async def test_acquire_under_limit(self, mock_redis):
        """Test acquiring token when under rate limit."""
        pipeline = MagicMock()
        pipeline.execute.return_value = [1, 500, 1, 1]  # Mocked Redis pipeline results
        mock_redis.pipeline.return_value = pipeline
        
        limiter = RedisRateLimiter()
        result = await limiter.acquire()
        
        assert result is True
        
    @pytest.mark.asyncio
    async def test_acquire_over_limit(self, mock_redis):
        """Test acquiring token when over rate limit."""
        pipeline = MagicMock()
        pipeline.execute.return_value = [1, 750, 1, 1]  # Over the 740 limit
        mock_redis.pipeline.return_value = pipeline
        
        limiter = RedisRateLimiter()
        result = await limiter.acquire()
        
        assert result is False
        
    @pytest.mark.asyncio
    async def test_wait_if_needed(self, mock_redis):
        """Test waiting for rate limit token."""
        # First call over limit, second call under limit
        pipeline = MagicMock()
        pipeline.execute.side_effect = [
            [1, 750, 1, 1],  # First call - over limit
            [1, 500, 1, 1]   # Second call - under limit
        ]
        mock_redis.pipeline.return_value = pipeline
        
        limiter = RedisRateLimiter()
        await limiter.wait_if_needed()  # Should wait and then succeed
        
        assert mock_redis.pipeline.call_count == 2

class TestFMPClient:
    """Tests for FMP API client."""
    
    @pytest.mark.asyncio
    async def test_get_historical_daily_data(self, fmp_client, mock_aiohttp_session):
        """Test getting daily historical data."""
        async for client in fmp_client:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value={"historical": DAILY_RECORDS})
            mock_aiohttp_session.get = AsyncMock(return_value=response)
            
            df = await client.get_historical_data(
                symbol="AAPL",
                start_date="2023-12-01",
                end_date="2023-12-07",
                interval="1d"
            )
            
            # Compare only relevant columns
            expected_cols = ["date", "open", "high", "low", "close", "volume", "adjClose"]
            pd.testing.assert_frame_equal(
                df[expected_cols].sort_values("date").reset_index(drop=True),
                DAILY_DATA[expected_cols].sort_values("date").reset_index(drop=True)
            )
        
    @pytest.mark.asyncio
    async def test_get_historical_5min_data(self, fmp_client, mock_aiohttp_session):
        """Test getting 5-minute historical data."""
        async for client in fmp_client:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value={"historical": FIVEMIN_RECORDS})
            mock_aiohttp_session.get = AsyncMock(return_value=response)
            
            df = await client.get_historical_data(
                symbol="AAPL",
                start_date="2023-12-07",
                end_date="2023-12-08",
                interval="5min"
            )
            
            # Compare only relevant columns
            expected_cols = ["date", "open", "high", "low", "close", "volume"]
            pd.testing.assert_frame_equal(
                df[expected_cols].sort_values("date").reset_index(drop=True),
                FIVEMIN_DATA[expected_cols].sort_values("date").reset_index(drop=True)
            )
        
    @pytest.mark.asyncio
    async def test_get_index_constituents(self, fmp_client, mock_aiohttp_session):
        """Test getting index constituents."""
        async for client in fmp_client:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=CONSTITUENTS.to_dict(orient="records"))
            mock_aiohttp_session.get = AsyncMock(return_value=response)
            
            df = await client.get_index_constituents("sp500")
            
            pd.testing.assert_frame_equal(
                df.sort_values("symbol").reset_index(drop=True),
                CONSTITUENTS.sort_values("symbol").reset_index(drop=True)
            )
        
    @pytest.mark.asyncio
    async def test_get_historical_constituents(self, fmp_client, mock_aiohttp_session):
        """Test getting historical constituent changes."""
        async for client in fmp_client:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=HISTORICAL_CONSTITUENTS.to_dict(orient="records"))
            mock_aiohttp_session.get = AsyncMock(return_value=response)
            
            df = await client.get_historical_constituents("sp500")
            
            pd.testing.assert_frame_equal(
                df.sort_values("date").reset_index(drop=True),
                HISTORICAL_CONSTITUENTS.sort_values("date").reset_index(drop=True)
            )
        
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, fmp_client, mock_aiohttp_session):
        """Test handling of rate limit responses."""
        async for client in fmp_client:
            # First response is rate limited, second succeeds
            response1 = AsyncMock()
            response1.status = 429
            response1.text = AsyncMock(return_value="Rate limit exceeded")
            
            response2 = AsyncMock()
            response2.status = 200
            response2.json = AsyncMock(return_value={"historical": DAILY_RECORDS})
            
            mock_aiohttp_session.get = AsyncMock(side_effect=[response1, response2])
            
            df = await client.get_historical_data(
                symbol="AAPL",
                start_date="2023-12-01",
                end_date="2023-12-07"
            )
            
            # Compare only relevant columns
            expected_cols = ["date", "open", "high", "low", "close", "volume", "adjClose"]
            pd.testing.assert_frame_equal(
                df[expected_cols].sort_values("date").reset_index(drop=True),
                DAILY_DATA[expected_cols].sort_values("date").reset_index(drop=True)
            )
            assert mock_aiohttp_session.get.call_count == 2  # Should have retried
        
    @pytest.mark.asyncio
    async def test_error_handling(self, fmp_client, mock_aiohttp_session):
        """Test handling of API errors."""
        async for client in fmp_client:
            response = AsyncMock()
            response.status = 404
            response.text = AsyncMock(return_value="Not found")
            mock_aiohttp_session.get = AsyncMock(return_value=response)
            
            df = await client.get_historical_data(
                symbol="INVALID",
                start_date="2023-12-01",
                end_date="2023-12-07"
            )
            
            assert df.empty  # Should return empty DataFrame on error
        
    @pytest.mark.asyncio
    async def test_convert_to_constituent_df(self, fmp_client):
        """Test converting historical constituents to membership dataframe."""
        async for client in fmp_client:
            hist_df = pd.DataFrame({
                "symbol": ["AAPL", "MSFT"],
                "date_added": ["2024-01-01", "2024-01-02"],
                "removed_ticker": ["OLD1", "OLD2"],
                "date": ["2024-02-01", "2024-02-02"]
            })
            
            df = client.convert_to_constituent_df(hist_df)
            
            # Verify the output
            assert isinstance(df, pd.DataFrame)
            assert "symbol" in df.columns
            assert "date_added" in df.columns
            assert "date_removed" in df.columns