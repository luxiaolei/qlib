"""Script to fetch and save real API responses for testing."""

import asyncio
from pathlib import Path

from scripts.data_collector.fmp.fmp_api import FMPClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"

async def fetch_fixtures():
    """Fetch and save API responses."""
    async with FMPClient() as client:
        # Fetch daily data
        daily_data = await client.get_historical_data(
            symbol="AAPL",
            start_date="2023-12-01",
            end_date="2023-12-07",
            interval="1d"
        )
        daily_data.to_json(FIXTURES_DIR / "daily_data.json")

        # Fetch 5min data - use a full trading day
        fivemin_data = await client.get_historical_data(
            symbol="AAPL",
            start_date="2023-12-07",
            end_date="2023-12-08",
            interval="5min"
        )
        fivemin_data.to_json(FIXTURES_DIR / "fivemin_data.json")

        # Fetch index constituents
        constituents = await client.get_index_constituents("sp500")
        constituents.to_json(FIXTURES_DIR / "constituents.json")

        # Fetch historical constituents
        hist_constituents = await client.get_historical_constituents("sp500")
        hist_constituents.to_json(FIXTURES_DIR / "historical_constituents.json")

if __name__ == "__main__":
    FIXTURES_DIR.mkdir(exist_ok=True)
    asyncio.run(fetch_fixtures()) 