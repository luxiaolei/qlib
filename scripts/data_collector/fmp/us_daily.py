"""US Daily Stock Data Collector for Financial Modeling Prep (FMP).

This module implements the data collection and normalization for US daily stock data
from FMP API. It includes classes for collecting, normalizing, and running the data
collection process.
"""

import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun
from scripts.data_collector.fmp.utils import (
    format_daily_data,
    get_fmp_data,
    get_us_exchange_symbols,
    validate_index_data,
)
from scripts.dump_bin import DumpDataAll, DumpDataUpdate

# Load environment variables
load_dotenv()
console = Console()

class FMPDailyCollector(BaseCollector):
    """Collector for US daily stock data from FMP API.
    
    This class handles the collection of daily stock data from Financial Modeling Prep API.
    It implements data fetching, validation, and storage mechanisms.
    
    Attributes
    ----------
    DEFAULT_START_DATETIME_1D : pd.Timestamp
        Default start date for data collection (2004-01-01)
    DEFAULT_END_DATETIME_1D : pd.Timestamp
        Default end date for data collection (current date)
    api_key : str
        FMP API key for authentication
    """
    
    DEFAULT_START_DATETIME_1D = pd.Timestamp("2004-01-01")
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.now())
    
    def __init__(
        self,
        save_dir: Union[str, Path] = "~/.qlib/qlib_data/us_fmp_d1",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_workers: int = 4,
        max_collector_count: int = 2,
        delay: int = 1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        instruments: Optional[List[str]] = None,  # Add this parameter
    ):
        """Initialize FMP daily data collector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save collected data
        start : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end : Optional[str]
            End date for data collection (YYYY-MM-DD)
        interval : str
            Data interval, default "1d" for daily data
        max_workers : int
            Maximum number of parallel workers
        max_collector_count : int
            Maximum number of collection attempts per symbol
        delay : int
            Delay between API calls in seconds
        check_data_length : Optional[int]
            Minimum required data length for validation
        limit_nums : Optional[int]
            Limit number of symbols to collect (for testing)
        instruments : Optional[List[str]]
            List of symbols to collect
            
        Raises
        ------
        ValueError
            If FMP_API_KEY is not found in environment variables
        """
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")
        self.api_key = api_key
        self._instruments = instruments
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,  # type: ignore
            limit_nums=limit_nums,  # type: ignore
        )

    def get_instrument_list(self) -> List[str]:
        """Get list of US stock symbols from FMP or use provided list.
        
        Returns
        -------
        List[str]
            List of stock symbols to collect data for
        """
        # First check if instruments were provided in initialization
        if hasattr(self, '_instruments') and self._instruments is not None:
            logger.info(f"Using provided instruments: {self._instruments}")
            return [str(symbol).strip().upper() for symbol in self._instruments]
        
        # If no instruments provided, fetch from FMP API
        with console.status("[bold green]Getting US stock symbols...") as status:
            try:
                symbols = get_us_exchange_symbols(self.api_key)
                if not symbols:  # If API call returns empty or None
                    logger.error("Failed to get symbols from FMP API")
                    return []  # Return empty list instead of None
                console.log(f"[green]Found {len(symbols)} symbols")
                return [str(symbol).strip().upper() for symbol in symbols]
            except Exception as e:
                logger.error(f"Error getting symbols from FMP API: {e}")
                return []  # Return empty list on error

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize stock symbol format.
        
        Parameters
        ----------
        symbol : str
            Raw stock symbol
            
        Returns
        -------
        str
            Normalized symbol in uppercase
        """
        return symbol.strip().upper()

    def _split_date_range(
        self, 
        start_datetime: pd.Timestamp, 
        end_datetime: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Split date range into chunks for efficient API calls.
        
        Parameters
        ----------
        start_datetime : pd.Timestamp
            Start date
        end_datetime : pd.Timestamp
            End date
            
        Returns
        -------
        List[Tuple[pd.Timestamp, pd.Timestamp]]
            List of (start, end) date pairs for each chunk
            
        Notes
        -----
        - Splits date range into 4-year chunks
        - For ranges > 4 years, creates multiple chunks
        - Handles partial periods at start and end
        - Ensures no gaps in date coverage
        """
        MAX_YEARS_PER_CHUNK = 4
        date_ranges = []
        current_date = start_datetime
        
        while current_date < end_datetime:
            # Calculate years difference from current to end
            years_remaining = (end_datetime - current_date).days / 365.25
            
            if years_remaining <= MAX_YEARS_PER_CHUNK:
                # If remaining period is less than max chunk size, use end_datetime
                chunk_end = end_datetime
            else:
                # Create a 4-year chunk
                chunk_end = current_date + pd.DateOffset(years=MAX_YEARS_PER_CHUNK)
                # Adjust to end of year if close to year boundary
                if (chunk_end - pd.Timestamp(f"{chunk_end.year}-12-31")).days < 7:
                    chunk_end = pd.Timestamp(f"{chunk_end.year}-12-31")
            
            date_ranges.append((current_date, chunk_end))
            current_date = chunk_end + pd.Timedelta(days=1)
            
            # Safety check to prevent infinite loops
            if len(date_ranges) > 100:  # Arbitrary large number
                logger.warning("Too many date chunks created, breaking loop")
                break
        
        return date_ranges

    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate quality of collected data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Collected data
        symbol : str
            Stock symbol
            
        Returns
        -------
        bool
            True if data passes validation, False otherwise
            
        Notes
        -----
        - Checks for empty dataframes
        - Validates required columns
        - Performs additional validation for index symbols
        """
        if df.empty:
            logger.warning(f"Empty data for {symbol}")
            return False
            
        # Additional validation for index symbols
        if symbol.startswith("^"):
            return validate_index_data(df, symbol)
            
        # Regular stock validation
        required_cols = ["date", "open", "high", "low", "close", "volume", "factor"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing columns for {symbol}")
            return False
            
        return True

    def get_data(
        self, 
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """Get daily stock data from FMP API.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        interval : str
            Data interval (should be "1d")
        start_datetime : pd.Timestamp
            Start date
        end_datetime : pd.Timestamp
            End date
            
        Returns
        -------
        pd.DataFrame
            Collected and formatted stock data
            
        Notes
        -----
        - Splits requests into yearly chunks
        - Handles rate limiting
        - Validates and formats data
        - Removes duplicates
        """
        all_data = []
        date_chunks = self._split_date_range(start_datetime, end_datetime)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Downloading {symbol}...", 
                total=len(date_chunks)
            )
            
            for chunk_start, chunk_end in date_chunks:
                url = (
                    f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
                    f"?from={chunk_start.strftime('%Y-%m-%d')}"
                    f"&to={chunk_end.strftime('%Y-%m-%d')}"
                    f"&apikey={self.api_key}"
                )
                
                data = get_fmp_data(url, self.api_key, delay=self.delay)
                if not data or "historical" not in data:
                    logger.warning(f"No data found for {symbol} in range {chunk_start} to {chunk_end}")
                    continue
                    
                df = format_daily_data(data["historical"], symbol)
                if self._validate_data(df, symbol):
                    all_data.append(df)
                    
                progress.advance(task)
            
        if not all_data:
            return pd.DataFrame()
            
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["date"]).sort_values("date")
        return final_df

class FMPNormalize(BaseNormalize):
    """Normalize FMP data to Qlib format.
    
    This class handles the normalization of raw FMP data into the format
    required by Qlib for further processing and analysis.
    """
    
    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get trading calendar.
        
        Returns
        -------
        List[pd.Timestamp]
            List of trading days
        """
        return sorted(self.kwargs.get("calendars", []))
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize FMP data to Qlib format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data from FMP
            
        Returns
        -------
        pd.DataFrame
            Normalized data
            
        Notes
        -----
        - Converts dates to datetime
        - Adjusts prices using factors
        - Handles volume and trading status
        - Sorts by date
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Convert to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort and reset index
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Calculate adjusted prices
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * df["factor"]
            
        # Handle volume
        df["volume"] = df["volume"].fillna(0).astype(float)
        
        # Add paused column (1 for suspended trading days)
        df["paused"] = 0
        df.loc[df["volume"] <= 0, "paused"] = 1
        
        return df
    
class FMPDailyRunner(BaseRun):
    """Runner for FMP daily data collection process."""
    
    def __init__(self):
        """Initialize FMP daily runner."""
        self._default_base_dir = Path("~/.qlib/qlib_data/us_fmp_d1")
        super().__init__()
        self._cur_module = importlib.import_module("us_daily")
        
    @property
    def default_base_dir(self) -> Path:
        """Get default base directory for data storage."""
        return self._default_base_dir

    @property
    def collector_class_name(self) -> str:
        """Get collector class name."""
        return "FMPDailyCollector"
        
    @property
    def normalize_class_name(self) -> str:
        """Get normalize class name."""
        return "FMPNormalize"

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
        **kwargs
    ):
        """Download data and convert to Qlib format."""
        # First download the data using parent's method
        super().download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            **kwargs
        )
        
        # Then normalize it using parent's method
        self.normalize_data()
        
        # Finally dump to bin format
        logger.info("Converting to Qlib binary format...")
        dumper = DumpDataAll(
            csv_path=str(self.normalize_dir),
            qlib_dir=str(self.default_base_dir),
            freq="day",
            max_workers=self.max_workers,
            date_field_name="date",
            symbol_field_name="symbol",
        )
        dumper.dump()
        logger.info("Finished converting to Qlib format.")
            
            
    def update_data(
        self,
        qlib_data_dir: Union[str, Path],
        check_data_length: Optional[int] = None,
        delay: int = 1,
    ) -> None:
        """Update data from the last available date.
        
        Parameters
        ----------
        qlib_data_dir : Union[str, Path]
            Qlib data directory containing calendars
        check_data_length : Optional[int]
            Minimum required data length
        delay : int
            Delay between API calls in seconds
        """
        qlib_dir = Path(qlib_data_dir).expanduser().resolve()
        calendar_path = qlib_dir.joinpath("calendars/day.txt")
        instruments_path = qlib_dir.joinpath("instruments/all.txt")

        # Get existing instruments
        existing_instruments = None
        if instruments_path.exists():
            try:
                instruments_df = pd.read_csv(
                    instruments_path, 
                    sep='\t', 
                    names=['symbol', 'start_time', 'end_time']
                )
                existing_instruments = instruments_df['symbol'].tolist()
                logger.info(f"Found {len(existing_instruments)} existing instruments")
            except Exception as e:
                logger.error(f"Error reading instruments file: {e}")

        # Get latest date from existing calendar
        if not calendar_path.exists():
            logger.warning(f"Calendar file not found: {calendar_path}")
            start = None
        else:
            try:
                with open(calendar_path, "r") as f:
                    dates = f.readlines()
                if dates:
                    latest_date = pd.Timestamp(dates[-1].strip())
                    start = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    start = None
            except Exception as e:
                logger.error(f"Error reading calendar file: {e}")
                start = None

        # Set end date to tomorrow to ensure we get today's data
        end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if start and pd.Timestamp(start) >= pd.Timestamp(end):
            logger.info("Already up to date, no new data to download")
            return

        # Download new data with existing instruments
        self.download_data(
            max_collector_count=2,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,  # type: ignore
            instruments=existing_instruments,  # Pass existing instruments
        )
        
        # Then normalize it
        self.normalize_data()
        
        # Finally dump to bin format with update mode
        logger.info("Updating Qlib binary format...")
        dumper = DumpDataUpdate(
            csv_path=str(self.normalize_dir),
            qlib_dir=str(qlib_data_dir),
            freq="day",
            max_workers=self.max_workers,
            date_field_name="date",
            symbol_field_name="symbol",
        )
        dumper.dump()
        logger.info("Finished updating Qlib format.")