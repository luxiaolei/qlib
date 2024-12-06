"""US 5-Minute Stock Data Collector for Financial Modeling Prep (FMP).

This module implements the data collection and normalization for US 5-minute stock data
from FMP API. It includes classes for collecting, normalizing, and running the data
collection process.
"""

import importlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console

from qlib.data import D
from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun
from scripts.data_collector.fmp.utils import (
    get_fmp_data_parallel,
    split_5min_date_range,
)
from scripts.data_collector.utils import (
    calc_adjusted_price,
    generate_minutes_calendar_from_daily,
)
from scripts.dump_bin import DumpDataAll, DumpDataUpdate

# Load environment variables
load_dotenv()
console = Console()

class FMP5minCollector(BaseCollector):
    """Collector for US 5-minute stock data from FMP API.
    
    This class handles the collection of 5-minute stock data from Financial Modeling Prep API.
    It implements data fetching, validation, and storage mechanisms.
    """
    
    DEFAULT_START_DATETIME_5MIN = pd.Timestamp(datetime.now() - timedelta(days=30))
    DEFAULT_END_DATETIME_5MIN = pd.Timestamp(datetime.now())
    
    def __init__(
        self,
        save_dir: Union[str, Path] = "~/.qlib/qlib_data/us_fmp_5min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "5min",
        max_workers: int = 4,
        max_collector_count: int = 2,
        delay: int = 1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        qlib_data_1d_dir: str = "~/.qlib/qlib_data/us_fmp_d1",
        instruments: Optional[List[str]] = None,
    ):
        """Initialize FMP 5-minute data collector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save collected data
        start : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end : Optional[str]
            End date for data collection (YYYY-MM-DD)
        interval : str
            Data interval, default "5min"
        max_workers : int
            Maximum number of parallel workers
        max_collector_count : int
            Maximum collection attempts
        delay : int
            Delay between API calls
        check_data_length : Optional[int]
            Minimum required data length
        limit_nums : Optional[int]
            Limit number of symbols to collect
        qlib_data_1d_dir : str
            Directory containing 1d data for factor calculation
        instruments : Optional[List[str]]
            List of symbols to collect. If None, will read from all.txt
        """
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")
        self.api_key = api_key
        
        # Check if 1d data exists
        qlib_data_1d_path = Path(qlib_data_1d_dir).expanduser()
        if not qlib_data_1d_path.exists():
            raise ValueError(f"1d data directory not found: {qlib_data_1d_path}")
        self.qlib_data_1d_path = qlib_data_1d_path
        
        self._instruments = instruments
        
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length or 0,
            limit_nums=limit_nums or 0,
        )

    def get_instrument_list(self) -> List[str]:
        """Get list of instruments from all.txt file or provided list.
        
        Returns
        -------
        List[str]
            List of stock symbols
        """
        # First check if instruments were provided in initialization
        if hasattr(self, '_instruments') and self._instruments is not None:
            console.print(f"[bold green]Using provided instruments: {self._instruments}[/]")
            # For provided instruments, we'll use the full date range
            self._instrument_dates = {
                symbol: (self.start_datetime, self.end_datetime)
                for symbol in self._instruments
            }
            return [str(symbol).strip().upper() for symbol in self._instruments]
        
        # If no instruments provided, read from all.txt
        instruments_file = self.qlib_data_1d_path / "instruments" / "all.txt"
        if not instruments_file.exists():
            raise ValueError(f"[bold red]Instruments file not found: {instruments_file}[/]")
            
        df = pd.read_csv(
            instruments_file,
            sep="\t",
            names=["symbol", "start_date", "end_date"],
        )
        
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        
        # Filter instruments based on date range
        mask = (
            (df["start_date"] <= self.end_datetime) &
            (df["end_date"] >= self.start_datetime)
        )
        filtered_df = df.loc[mask]
        
        # Store instrument date ranges in a dictionary
        self._instrument_dates = {
            row["symbol"]: (
                max(row["start_date"], self.start_datetime),
                min(row["end_date"], self.end_datetime)
            )
            for _, row in filtered_df.iterrows()
        }
        
        return filtered_df["symbol"].unique().tolist()

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

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """Get 5-minute stock data from FMP API.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        interval : str
            Data interval (should be "5min")
        start_datetime : pd.Timestamp
            Start date
        end_datetime : pd.Timestamp
            End date
            
        Returns
        -------
        pd.DataFrame
            Collected and formatted stock data
        """
        # Get instrument-specific date range from stored dictionary
        if hasattr(self, '_instrument_dates') and symbol in self._instrument_dates:
            symbol_start, symbol_end = self._instrument_dates[symbol]
            # Use the intersection of requested date range and instrument's available range
            start_datetime = max(start_datetime, symbol_start)
            end_datetime = min(end_datetime, symbol_end)
            
            if start_datetime >= end_datetime:
                console.print(f"[bold yellow]No valid date range for {symbol}[/]")
                return pd.DataFrame()
        
        # Split date range into 4-day chunks
        date_chunks = split_5min_date_range(start_datetime, end_datetime)
        
        # Prepare URLs for all chunks
        urls = [
            f"https://financialmodelingprep.com/api/v3/historical-chart/5min/{symbol}"
            f"?from={chunk_start.strftime('%Y-%m-%d')}"
            f"&to={chunk_end.strftime('%Y-%m-%d')}"
            f"&apikey={self.api_key}"
            for chunk_start, chunk_end in date_chunks
        ]
        
        # Fetch data in parallel
        chunk_data_list = get_fmp_data_parallel(
            urls,
            self.api_key,
            delay=self.delay,
            max_workers=min(10, len(urls)),
            desc=f"Downloading {symbol}"
        )
        
        # Combine all data into a single list
        all_data = []
        for data in chunk_data_list:
            if data:  # For 5min data, the response is directly a list
                all_data.extend(data)
        
        if not all_data:
            return pd.DataFrame()
        
        # Create DataFrame directly from the combined list
        df = pd.DataFrame(all_data)
        df = df.rename(columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        
        # Sort and remove duplicates
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    
class FMPNormalize5min(BaseNormalize):
    """Normalize FMP 5-minute data to Qlib format."""
    
    COLUMNS = ["open", "close", "high", "low", "volume"]
    
    AM_RANGE = ("09:30:00", "11:59:00")
    PM_RANGE = ("12:00:00", "15:59:00")
    
    def __init__(
        self,
        qlib_data_1d_dir: Union[str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs
    ):
        """Initialize normalizer.
        
        Parameters
        ----------
        qlib_data_1d_dir : Union[str, Path]
            Directory containing 1d Qlib data
        date_field_name : str
            Date field name
        symbol_field_name : str
            Symbol field name
        """
        qlib_data_1d_path = Path(qlib_data_1d_dir).expanduser()
        if not qlib_data_1d_path.exists():
            raise ValueError(f"1d data directory not found: {qlib_data_1d_path}")
            
        # Initialize Qlib with 1d data
        from qlib.utils import init_instance_by_config
        init_instance_by_config(
            {
                "provider_uri": str(qlib_data_1d_path),
                "region": "us"
            }
        )
        
        self.all_1d_data = D.features(
            D.instruments("all"), 
            ["$paused", "$volume", "$factor", "$close"],
            freq="day"
        )
        super().__init__(date_field_name, symbol_field_name)
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the 5-minute data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw 5-minute data
            
        Returns
        -------
        pd.DataFrame
            Normalized data with adjusted prices
        """
        if df.empty:
            return df
        
        # Normalize and adjust prices using 1d data
        df = calc_adjusted_price(
            df=df,
            _1d_data_all=self.all_1d_data,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="5min",
            am_range=self.AM_RANGE,
            pm_range=self.PM_RANGE
        )
        
        return df
    
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get calendar list for 5-minute data."""
        return generate_minutes_calendar_from_daily(
            self.calendar_list_1d,
            freq="5min",
            am_range=self.AM_RANGE,
            pm_range=self.PM_RANGE
        )
    
    @property
    def calendar_list_1d(self) -> List[pd.Timestamp]:
        """Get 1d calendar list."""
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = list(D.calendar(freq="day"))
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d
    
class FMP5minRunner(BaseRun):
    """Runner for FMP 5-minute data collection process."""
    
    def __init__(self):
        """Initialize FMP 5-minute runner."""
        self._default_base_dir = Path("~/.qlib/qlib_data/us_fmp_5min")
        super().__init__()
        self._cur_module = importlib.import_module("us_m5")
        
    @property
    def default_base_dir(self) -> Path:
        """Get default base directory for data storage."""
        return self._default_base_dir
        
    @property
    def collector_class_name(self) -> str:
        """Get collector class name."""
        return "FMP5minCollector"
        
    @property
    def normalize_class_name(self) -> str:
        """Get normalize class name."""
        return "FMPNormalize5min"
    
    def download_data(
        self,
        max_collector_count: int = 2,
        delay: int = 0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        qlib_data_1d_dir: str = "~/.qlib/qlib_data/us_fmp_d1",
        **kwargs
    ):
        """Download data and convert to Qlib format.
        
        Parameters
        ----------
        max_collector_count : int
            Maximum collection attempts per symbol
        delay : int
            Delay between API calls in seconds
        start : Optional[str]
            Start date (YYYY-MM-DD)
        end : Optional[str]
            End date (YYYY-MM-DD)
        check_data_length : Optional[int]
            Minimum required data length
        limit_nums : Optional[int]
            Limit number of symbols to collect
        qlib_data_1d_dir : str
            Directory containing 1d Qlib data
        """
        # Download data
        super().download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length or 0,
            limit_nums=limit_nums or 0,
            qlib_data_1d_dir=qlib_data_1d_dir,
            **kwargs
        )
        
        # Normalize data
        self.normalize_data(qlib_data_1d_dir=qlib_data_1d_dir)
        
        # Convert to Qlib binary format
        logger.info("Converting to Qlib binary format...")
        dumper = DumpDataAll(
            csv_path=str(self.normalize_dir),
            qlib_dir=str(self.default_base_dir),
            freq="5min",
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
        qlib_data_1d_dir: str = "~/.qlib/qlib_data/us_fmp_d1",
        instruments: Optional[List[str]] = None,
        **kwargs
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
        qlib_data_1d_dir : str
            Directory containing 1d Qlib data
        instruments : Optional[List[str]]
            List of specific instruments to update. If None, updates all instruments
        **kwargs : dict
            Additional keyword arguments
        """
        qlib_dir = Path(qlib_data_dir).expanduser().resolve()
        calendar_path = qlib_dir.joinpath("calendars/5min.txt")
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
                if instruments is not None:
                    # Filter existing instruments to only those requested
                    existing_instruments = [
                        symbol for symbol in instruments 
                        if symbol in instruments_df['symbol'].tolist()
                    ]
                    if len(existing_instruments) != len(instruments):
                        logger.warning(
                            f"Some requested instruments not found in {instruments_path}. "
                            f"Found {len(existing_instruments)} out of {len(instruments)}"
                        )
                else:
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
                    # For 5min data, start from the next trading day
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
            qlib_data_1d_dir=qlib_data_1d_dir,
            **kwargs
        )
        
        # Then normalize it
        self.normalize_data(qlib_data_1d_dir=qlib_data_1d_dir)
        
        # Finally dump to bin format with update mode
        logger.info("Updating Qlib binary format...")
        dumper = DumpDataUpdate(
            csv_path=str(self.normalize_dir),
            qlib_dir=str(qlib_data_dir),
            freq="5min",
            max_workers=self.max_workers,
            date_field_name="date",
            symbol_field_name="symbol",
        )
        dumper.dump()
        logger.info("Finished updating Qlib format.")

if __name__ == "__main__":
    import fire
    fire.Fire(FMP5minRunner)