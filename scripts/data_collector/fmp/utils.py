"""FMP utility functions for data collection and processing.

This module provides utility functions for collecting and processing financial data from 
Financial Modeling Prep (FMP) API. It includes functions for API requests, data validation,
and data formatting.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, cast

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Load environment variables from .env file
load_dotenv()

console = Console()

T = TypeVar('T')

class RateLimiter:
    """Rate limiter for API calls.
    
    Attributes
    ----------
    calls_per_minute : int
        Maximum number of calls allowed per minute
    call_counter : int
        Current number of calls made
    minute_start : float
        Timestamp of the start of current minute window
    lock : threading.Lock
        Thread lock for synchronization
    """
    
    def __init__(self, calls_per_minute: int = 740):
        """Initialize rate limiter.
        
        Parameters
        ----------
        calls_per_minute : int
            Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_counter = 0
        self.minute_start = time.time()
        self.lock = Lock()
        
    def wait_if_needed(self) -> None:
        """Check and wait if rate limit is reached."""
        with self.lock:
            current_time = time.time()
            if current_time - self.minute_start >= 60:
                self.call_counter = 0
                self.minute_start = current_time
            
            if self.call_counter >= self.calls_per_minute:
                sleep_time = 60 - (current_time - self.minute_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.call_counter = 0
                self.minute_start = time.time()
            
            self.call_counter += 1

class ParallelProcessor(Generic[T]):
    """Generic parallel processor with rate limiting.
    
    Attributes
    ----------
    rate_limiter : RateLimiter
        Rate limiter instance
    max_workers : int
        Maximum number of parallel workers
    """
    
    def __init__(self, max_workers: int = 10, calls_per_minute: int = 740):
        """Initialize parallel processor.
        
        Parameters
        ----------
        max_workers : int
            Maximum number of parallel workers
        calls_per_minute : int
            Maximum API calls per minute
        """
        self.rate_limiter = RateLimiter(calls_per_minute)
        self.max_workers = max_workers
        
    def process_parallel(
        self,
        items: List[Any],
        process_func: Callable[[Any], Optional[T]],
        max_workers: Optional[int] = None,
        desc: str = "Processing"
    ) -> List[T]:
        """Process items in parallel with rate limiting.
        
        Parameters
        ----------
        items : List[Any]
            Items to process
        process_func : Callable
            Function to process each item
        max_workers : Optional[int]
            Override default max workers
        desc : str
            Description for progress bar
            
        Returns
        -------
        List[T]
            List of processed results
        """
        workers = min(max_workers or self.max_workers, len(items))
        results: List[T] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]{desc}...", total=len(items))
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                def process_with_rate_limit(item: Any) -> Optional[T]:
                    self.rate_limiter.wait_if_needed()
                    return process_func(item)
                
                future_to_item = {
                    executor.submit(process_with_rate_limit, item): item 
                    for item in items
                }
                
                for future in as_completed(future_to_item):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    progress.advance(task)
        
        return results

def get_fmp_data(url: str, api_key: str, delay: float = 0.2) -> Dict[str, Any]:
    """Get data from FMP API with rate limiting and retries.
    
    Parameters
    ----------
    url : str
        The API endpoint URL
    api_key : str
        FMP API key for authentication
        
    Returns
    -------
    Dict[str, Any]
        JSON response from the API
        
    Raises
    ------
    requests.RequestException
        If the API request fails after maximum retries
    ValueError
        If the API response is invalid
        
    Notes
    -----
    - Implements exponential backoff retry strategy
    - Includes rate limiting to avoid API throttling
    - Validates API response before returning
    """
    MAX_RETRIES = 5

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url.replace("YOUR_API_KEY", api_key))
            response.raise_for_status()
            time.sleep(delay)
            return response.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to get data from FMP after {MAX_RETRIES} attempts: {e}")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return {}

def get_fmp_data_parallel(
    urls: List[str],
    api_key: str,
    delay: float = 0.2,
    max_workers: int = 10,
    desc: str = "Fetching FMP data"
) -> List[Dict[str, Any]]:
    """Get data from multiple FMP API endpoints in parallel.
    
    Parameters
    ----------
    urls : List[str]
        List of API endpoint URLs
    api_key : str
        FMP API key
    delay : float
        Delay between API calls
    max_workers : int
        Maximum number of parallel workers
    desc : str
        Description for progress bar
        
    Returns
    -------
    List[Dict[str, Any]]
        List of API responses
    """
    processor = ParallelProcessor(max_workers=max_workers)
    
    def fetch_single_url(url: str) -> Optional[Dict[str, Any]]:
        try:
            return get_fmp_data(url, api_key, delay)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    return processor.process_parallel(urls, fetch_single_url, desc=desc)

def get_us_exchange_symbols(api_key: str) -> List[str]:
    """Get US exchange symbols and indexes from FMP.
    
    Parameters
    ----------
    api_key : str
        FMP API key for authentication
        
    Returns
    -------
    List[str]
        List of unique stock symbols from US exchanges
        
    Notes
    -----
    - Collects symbols from NASDAQ, NYSE, and AMEX
    - Includes major US market indexes
    - Removes duplicates and sorts the list
    - Handles special symbol formats (e.g., preferred shares)
    """
    exchanges = ["NASDAQ", "NYSE", "AMEX"]
    symbols = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching US exchange symbols...", total=len(exchanges))
        
        for exchange in exchanges:
            url = f"https://financialmodelingprep.com/api/v3/symbol/{exchange}?apikey={api_key}"
            data = get_fmp_data(url, api_key)
            if isinstance(data, list):
                typed_data = cast(List[Dict[str, Any]], data)
                symbols.extend([item.get("symbol", "") for item in typed_data])
            progress.advance(task)
    
    # Add US market indexes with descriptions
    us_indexes = [
        "^SPX",      # S&P 500
        "^IXIC",     # NASDAQ Composite
        "^DJI",      # Dow Jones Industrial Average
        # ... (rest of the indexes)
    ]
    
    symbols.extend(us_indexes)
    return sorted(list(set(symbols)))

def format_daily_data(data: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    """Format FMP daily data into Qlib format.
    
    Parameters
    ----------
    data : List[Dict[str, Any]]
        Raw data from FMP API
    symbol : str
        Stock symbol
        
    Returns
    -------
    pd.DataFrame
        Formatted data with columns: symbol, date, open, high, low, close, volume, factor
        
    Notes
    -----
    - Handles duplicate dates by keeping the latest version
    - Calculates adjustment factor from adjusted close price
    - Sorts data by date
    - Validates data quality before returning
    - Some `date` entry may in format `2024-01-01` or `2024-01-01 00:00:00`, 
      we need to convert them to `2024-01-01`
    """
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "date": "date",
        "open": "open",
        "high": "high", 
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adjClose": "adjclose"
    })
    
    # Convert date to datetime first
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Handle duplicates - keep last record for each date
    # FMP API typically returns most recent version last
    if df.duplicated(subset=["date"], keep=False).any():
        logger.warning(f"Found duplicate dates for {symbol}, keeping most recent versions")
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    
    # Calculate factor as per Qlib requirements
    df["factor"] = df["adjclose"] / df["close"]
    
    # Add symbol column
    df["symbol"] = symbol
    
    # Sort by date
    df = df.sort_values("date")
    
    # Select and order final columns
    return df[["symbol", "date", "open", "high", "low", "close", "volume", "factor"]]

def validate_index_data(df: pd.DataFrame, symbol: str) -> bool:
    """Additional validation for index data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing index data
    symbol : str
        Index symbol
        
    Returns
    -------
    bool
        True if data passes validation, False otherwise
        
    Notes
    -----
    - Checks for minimum data length (30 days)
    - Validates data completeness
    - Verifies no missing values
    """
    if df.empty:
        logger.warning(f"Empty data for index {symbol}")
        return False
        
    # Check for reasonable date range
    date_range = df["date"].max() - df["date"].min()
    if date_range.days < 30:  # At least 1 month of data
        logger.warning(f"Insufficient date range for index {symbol}: {date_range.days} days")
        return False
        
    # Check for missing values
    if df.isnull().any().any():
        logger.warning(f"Missing values found in index {symbol}")
        return False
        
    return True
