"""Financial Modeling Prep (FMP) API Client.

Enhanced async API client for interacting with Financial Modeling Prep.
Supports rate limiting, caching, and data retrieval for both daily and intraday data.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, TypedDict, Union

import aiohttp
import pandas as pd
import redis.asyncio as redis
from loguru import logger
from rich.console import Console

console = Console()

@dataclass
class IndexConstituent:
    """Data class for index constituent information."""
    symbol: str
    name: str
    sector: str
    sub_sector: str
    headquarters: str
    date_added: str
    cik: str
    founded: str

class IndexConstituentData(TypedDict):
    """Type hints for index constituent data from API."""
    symbol: str
    name: str
    sector: str
    subSector: str
    headQuarter: str
    dateFirstAdded: str
    cik: str
    founded: str

@dataclass
class HistoricalConstituent:
    """Data class for historical constituent information."""
    date_added: str  # YYYY-MM-DD
    added_security: str
    removed_ticker: str
    removed_security: str
    date: str  # YYYY-MM-DD
    symbol: str
    reason: str

class HistoricalConstituentData(TypedDict):
    """Type hints for historical constituent data from API."""
    dateAdded: str  # Month DD, YYYY
    addedSecurity: str
    removedTicker: str
    removedSecurity: str
    date: str  # YYYY-MM-DD
    symbol: str
    reason: str

class RateLimiter:
    """Rate limiter for API requests.
    
    Uses Redis to implement a sliding window rate limiter.
    
    Attributes
    ----------
    redis_client : redis.Redis
        Redis client for rate limiting
    max_requests : int
        Maximum number of requests allowed per window
    window_ms : int
        Window size in milliseconds
    key : str
        Redis key for storing rate limit data
    """
    
    def __init__(
        self,
        redis_url: str,
        max_requests: int = 250,
        window_ms: int = 60000,
        key_prefix: str = "fmp_rate_limit:"
    ):
        """Initialize rate limiter.
        
        Parameters
        ----------
        redis_url : str
            Redis connection URL
        max_requests : int, optional
            Maximum number of requests allowed per window, defaults to 250
        window_ms : int, optional
            Window size in milliseconds, defaults to 60000 (1 minute)
        key_prefix : str, optional
            Prefix for Redis keys, defaults to "fmp_rate_limit:"
        """
        self.redis_client = redis.from_url(redis_url)
        self.max_requests = max_requests
        self.window_ms = window_ms
        self.key = f"{key_prefix}{max_requests}:{window_ms}"

    async def acquire(self, wait: bool = True) -> bool:
        """Acquire a rate limit token.

        Parameters
        ----------
        wait : bool, optional
            Whether to wait for a token if rate limit is reached, by default True

        Returns
        -------
        bool
            True if a token was acquired, False otherwise
        """
        if not self.redis_client:
            return True

        try:
            # Get current timestamp
            timestamp = int(time.time() * 1000)

            # Add current request to the sliding window
            pipeline = self.redis_client.pipeline()
            pipeline.zadd(self.key, {timestamp: timestamp})

            # Remove timestamps outside the window
            window_start = timestamp - self.window_ms
            pipeline.zremrangebyscore(self.key, 0, window_start)

            # Count requests in the window
            pipeline.zcard(self.key)

            # Execute pipeline
            _, _, request_count = await pipeline.execute()

            if request_count <= self.max_requests:
                return True

            if not wait:
                return False

            # Calculate delay
            if request_count > self.max_requests:
                # Get oldest timestamp in window
                timestamps = await self.redis_client.zrange(
                    self.key, 0, 0, withscores=True
                )
                if timestamps:
                    oldest = int(timestamps[0][1])
                    delay = (oldest + self.window_ms - timestamp) / 1000
                    if delay > 0:
                        logger.warning(
                            f"Rate limit reached. Waiting {delay:.2f}s for next token."
                        )
                        await asyncio.sleep(delay)
                        return await self.acquire(wait=True)

            return False

        except Exception as e:
            logger.error(f"Redis error in rate limiter: {str(e)}")
            # Fallback to no rate limiting
            return True

class FMPClient:
    """Client for FMP API.
    
    This class handles communications with the FMP API, including rate limiting
    and request retries.
    
    Attributes
    ----------
    api_key : str
        FMP API key
    base_url : str
        Base URL for FMP API
    session : aiohttp.ClientSession
        HTTP session for making requests
    rate_limiter : Optional[RateLimiter]
        Rate limiter for API calls
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://financialmodelingprep.com/api",
        timeout: int = 30,
        retries: int = 3,
        redis_url: Optional[str] = None,
        redis_max_requests: int = 250,
        redis_window_seconds: int = 60,
    ):
        """Initialize FMP client.
        
        Parameters
        ----------
        api_key : Optional[str], optional
            FMP API key, defaults to None (will use environment variable)
        base_url : str, optional
            Base URL for FMP API, defaults to "https://financialmodelingprep.com/api"
        timeout : int, optional
            Timeout for API requests in seconds, defaults to 30
        retries : int, optional
            Number of retries for failed requests, defaults to 3
        redis_url : Optional[str], optional
            Redis URL for rate limiting, defaults to None
        redis_max_requests : int, optional
            Maximum number of requests per window, defaults to 250
        redis_window_seconds : int, optional
            Window size in seconds for rate limiting, defaults to 60
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key is required")
            
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.session = None
        
        # Set up rate limiter if Redis URL is provided
        if redis_url:
            window_ms = redis_window_seconds * 1000
            self.rate_limiter = RateLimiter(
                redis_url=redis_url,
                max_requests=redis_max_requests,
                window_ms=window_ms,
                key_prefix="fmp_rate_limit:"
            )
        else:
            self.rate_limiter = None
    
    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _get_data(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: int = 86400,  # 1 day in seconds
        retries: int = 3
    ) -> Any:
        """Get data from FMP API with caching and retries.
        
        Parameters
        ----------
        endpoint : str
            API endpoint path (without base URL)
        params : Optional[Dict[str, Any]]
            Query parameters
        use_cache : bool
            Whether to use cache
        cache_ttl : int
            Cache TTL in seconds
        retries : int
            Number of retries on failure
            
        Returns
        -------
        Any
            API response data
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context.")
        
        # Add API key to params
        params = params or {}
        params["apikey"] = self.api_key
        
        # Check cache if enabled
        cache_key = None
        if use_cache and self.cache_dir:
            import hashlib
            import json
            
            # Create unique cache key based on endpoint and params
            cache_id = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
            cache_key = hashlib.md5(cache_id.encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.json"
            
            if cache_path.exists():
                # Check if cache is still valid
                cache_time = cache_path.stat().st_mtime
                if time.time() - cache_time < cache_ttl:
                    try:
                        import json
                        with open(cache_path, "r") as f:
                            data = json.load(f)
                        return data
                    except Exception as e:
                        logger.warning(f"Error reading cache: {e}")
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Construct URL
        url = f"{self.base_url}/{endpoint}"
        
        # Make request with retries
        attempt = 0
        last_exception = None
        
        while attempt < retries:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 429:  # Too Many Requests
                        logger.warning("Rate limit exceeded, backing off...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        attempt += 1
                        continue
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Cache the result if caching is enabled
                    if use_cache and self.cache_dir and cache_key:
                        import json
                        with open(self.cache_dir / f"{cache_key}.json", "w") as f:
                            json.dump(data, f)
                    
                    return data
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt+1}/{retries}): {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                attempt += 1
        
        # All retries failed
        raise RuntimeError(f"Failed to get data after {retries} retries") from last_exception
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical price data for a symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : Union[str, datetime, pd.Timestamp]
            Start date
        end_date : Union[str, datetime, pd.Timestamp]
            End date
        interval : str, optional
            Data interval (1d, 1min, 5min, 15min, 30min, 1hour), defaults to "1d"

        Returns
        -------
        pd.DataFrame
            DataFrame with historical price data
        """
        # Convert dates to string format
        start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        
        try:
            # Acquire rate limiting token
            if self.rate_limiter:
                await self.rate_limiter.acquire()
                
            # Choose endpoint based on interval
            if interval == "1d":
                endpoint = f"/v3/historical-price-full/{symbol}"
                params = {
                    "from": start_str,
                    "to": end_str,
                    "apikey": self.api_key
                }
            else:
                # Map interval to FMP format
                interval_map = {
                    "1min": "1min",
                    "5min": "5min",
                    "15min": "15min",
                    "30min": "30min",
                    "1hour": "1hour"
                }
                fmp_interval = interval_map.get(interval, "15min")
                
                endpoint = f"/v3/historical-chart/{fmp_interval}/{symbol}"
                params = {"apikey": self.api_key}
                
            # Make API request
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(url, params=params, timeout=self.timeout) as response:
                if response.status != 200:
                    logger.error(f"Error fetching data for {symbol}: {response.status}")
                    return pd.DataFrame()
                    
                data = await response.json()
                
            # Process data based on interval
            if interval == "1d":
                if "historical" not in data or not data["historical"]:
                    logger.warning(f"No historical data for {symbol}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(data["historical"])
                df = df.sort_values("date")
                
                # Filter by date range
                df = df[(df["date"] >= start_str) & (df["date"] <= end_str)]
                
                # Rename columns to match Qlib format
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "adjClose": "adj_close",
                    "volume": "volume"
                })
                
                # Add additional adjusted columns
                if "adj_close" in df.columns and "close" in df.columns:
                    ratio = df["adj_close"] / df["close"]
                    df["adj_open"] = df["open"] * ratio
                    df["adj_high"] = df["high"] * ratio
                    df["adj_low"] = df["low"] * ratio
                    df["adj_volume"] = df["volume"]
                
            else:
                if not data or not isinstance(data, list):
                    logger.warning(f"No intraday data for {symbol}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(data)
                if df.empty:
                    return df
                    
                # Convert date to datetime
                df["date"] = pd.to_datetime(df["date"])
                
                # Filter by date range
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
                
                # Sort by date
                df = df.sort_values("date")
                
                # Convert date back to string
                df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Rename columns to match Qlib format
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })
                
                # Add adjusted columns (for intraday, typically same as unadjusted)
                df["adj_open"] = df["open"]
                df["adj_high"] = df["high"]
                df["adj_low"] = df["low"]
                df["adj_close"] = df["close"]
                df["adj_volume"] = df["volume"]
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_index_constituents(self, index: str = "sp500") -> pd.DataFrame:
        """Get current index constituents.
        
        Parameters
        ----------
        index : str
            Index name ('sp500', 'dowjones', 'nasdaq100')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with constituent information
        """
        try:
            endpoint = f"{index}_constituent"
            data = await self._get_data(endpoint)
            
            if not data:
                logger.warning(f"No constituents found for {index}")
                return pd.DataFrame()
                
            constituents = []
            for item in data:
                constituent = IndexConstituent(
                    symbol=item["symbol"],
                    name=item["name"],
                    sector=item["sector"],
                    sub_sector=item["subSector"],
                    headquarters=item["headQuarter"],
                    date_added=item["dateFirstAdded"],
                    cik=item.get("cik", ""),
                    founded=item.get("founded", "")
                )
                constituents.append(vars(constituent))
                
            df = pd.DataFrame(constituents)
            
            # Convert date_added to datetime
            if "date_added" in df.columns:
                df["date_added"] = pd.to_datetime(df["date_added"])
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting {index} constituents: {e}")
            return pd.DataFrame()
    
    async def get_historical_constituents(self, index: str = "sp500") -> pd.DataFrame:
        """Get historical index constituent changes.
        
        Parameters
        ----------
        index : str
            Index name ('sp500', 'dowjones', 'nasdaq100')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with historical constituent changes
        """
        try:
            endpoint = f"historical/{index}_constituent"
            data = await self._get_data(endpoint)
            
            if not data:
                logger.warning(f"No historical constituents found for {index}")
                return pd.DataFrame()
                
            constituents = []
            for item in data:
                # Convert date format from "Month DD, YYYY" to "YYYY-MM-DD"
                date_added = item.get("dateAdded", "")
                if date_added:
                    try:
                        date_obj = datetime.strptime(date_added, "%B %d, %Y")
                        date_added = date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                
                constituent = HistoricalConstituent(
                    date_added=date_added,
                    added_security=item.get("addedSecurity", ""),
                    removed_ticker=item.get("removedTicker", ""),
                    removed_security=item.get("removedSecurity", ""),
                    date=item.get("date", ""),
                    symbol=item.get("symbol", ""),
                    reason=item.get("reason", "")
                )
                constituents.append(vars(constituent))
                
            df = pd.DataFrame(constituents)
            
            # Convert dates to datetime
            for date_col in ["date", "date_added"]:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical {index} constituents: {e}")
            return pd.DataFrame()

    async def get_all_symbols(self) -> pd.DataFrame:
        """Get all tradable symbols.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all available symbols
        """
        try:
            endpoint = "stock/list"
            data = await self._get_data(endpoint)
            
            if not data:
                logger.warning("No symbols found")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # Filter out non-standard securities
            if "type" in df.columns:
                df = df[df["type"].isin(["stock", "etf"])]
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting symbol list: {e}")
            return pd.DataFrame()

async def test_api():
    """Test the FMP API client with a simple query."""
    import os
    from rich.console import Console
    
    console = Console()
    api_key = os.getenv("FMP_API_KEY")
    
    if not api_key:
        console.print("[bold red]Error: FMP_API_KEY environment variable not found[/bold red]")
        return
        
    console.print("[bold green]Testing FMP API client...[/bold green]")
    
    async with FMPClient(api_key=api_key) as client:
        # Test getting historical daily data for AAPL
        console.print("[bold]Getting AAPL daily data...[/bold]")
        daily_data = await client.get_historical_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-01-31",
            interval="1d"
        )
        console.print(f"Got {len(daily_data)} daily records")
        
        # Test getting 15-minute data for AAPL
        console.print("[bold]Getting AAPL 15-minute data...[/bold]")
        intraday_data = await client.get_historical_data(
            symbol="AAPL", 
            start_date="2023-01-30",
            end_date="2023-01-31",
            interval="15min"
        )
        console.print(f"Got {len(intraday_data)} 15-minute records")
        
        # Test getting S&P 500 constituents
        console.print("[bold]Getting S&P 500 constituents...[/bold]")
        constituents = await client.get_index_constituents(index="sp500")
        console.print(f"Got {len(constituents)} S&P 500 constituents")
        
    console.print("[bold green]API test completed successfully![/bold green]")

if __name__ == "__main__":
    asyncio.run(test_api()) 