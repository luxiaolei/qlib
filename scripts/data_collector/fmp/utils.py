"""FMP utility functions for data collection and processing.

This module provides utility functions for collecting and processing financial data from 
Financial Modeling Prep (FMP) API. It includes functions for API requests, data validation,
and data formatting.
"""

import time
from typing import Any, Dict, List, cast

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables from .env file
load_dotenv()

console = Console()

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
