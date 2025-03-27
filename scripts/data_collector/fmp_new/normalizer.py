"""Data normalizer for Financial Modeling Prep data.

This module provides classes for normalizing data from FMP to Qlib format.
It handles data quality checks, calendar alignment, and price adjustments.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskColumn, TextColumn

import qlib
from qlib.data import D
from scripts.data_collector.utils import (
    calc_adjusted_price,
    generate_minutes_calendar_from_daily,
)

console = Console()

class BaseFMPNormalizer:
    """Base normalizer for FMP data.
    
    This class provides common functionality for normalizing data from FMP to Qlib format.
    It handles data cleaning, calendar alignment, and price adjustments.
    
    Attributes
    ----------
    data_dir : Path
        Directory containing raw data files
    output_dir : Path
        Directory to save normalized data
    """
    
    # Data column names
    COLUMNS = ["open", "close", "high", "low", "volume"]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        max_workers: int = 8
    ):
        """Initialize normalizer.
        
        Parameters
        ----------
        data_dir : Union[str, Path]
            Directory containing raw data files
        output_dir : Union[str, Path]
            Directory to save normalized data
        date_field_name : str
            Name of date column
        symbol_field_name : str
            Name of symbol column
        max_workers : int
            Maximum number of parallel workers
        """
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.date_field_name = date_field_name
        self.symbol_field_name = symbol_field_name
        self.max_workers = max_workers
        
        # Cache for calendar data
        self._calendar_list = None
    
    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get trading calendar from data files.
        
        This method should be implemented by subclasses to provide
        the appropriate calendar for the data frequency.
        
        Returns
        -------
        List[pd.Timestamp]
            List of trading dates
        """
        raise NotImplementedError("Subclasses must implement _get_calendar_list")
    
    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """Get cached calendar list."""
        if self._calendar_list is None:
            self._calendar_list = self._get_calendar_list()
        return self._calendar_list
    
    def get_symbol_from_path(self, file_path: Path) -> str:
        """Extract symbol from file path.
        
        Parameters
        ----------
        file_path : Path
            Path to data file
            
        Returns
        -------
        str
            Symbol name
        """
        # Example: from /path/to/AAPL.csv, extract AAPL
        return file_path.stem
    
    def normalize_single_file(self, file_path: Path) -> pd.DataFrame:
        """Normalize a single data file.
        
        Parameters
        ----------
        file_path : Path
            Path to data file
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        # Extract symbol
        symbol = self.get_symbol_from_path(file_path)
        
        try:
            # Read data
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                return pd.DataFrame()
                
            # Add symbol column if not present
            if self.symbol_field_name not in df.columns:
                df[self.symbol_field_name] = symbol
                
            # Normalize
            return self.normalize(df)
            
        except Exception as e:
            logger.error(f"Error normalizing {file_path}: {e}")
            return pd.DataFrame()
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data from FMP
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        if df.empty:
            return df
            
        # Basic data cleaning
        df = df.copy()
        
        # Set date as index
        df.set_index(self.date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone info
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Clean invalid volume data
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), self.COLUMNS] = np.nan
        
        # Apply price adjustments
        df = self.adjusted_price(df)
        
        # Reset index to get date as a column again
        df = df.reset_index()
        
        return df
    
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply price adjustments using adjustment factors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw price data
            
        Returns
        -------
        pd.DataFrame
            Adjusted price data
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # Calculate adjustment factors if adjClose is available
        if "adj_close" in df.columns:
            # Calculate factors
            df["factor"] = df["adj_close"] / df["close"]
            df["factor"] = df["factor"].fillna(method="ffill")  # Fill missing factors
            df["factor"] = df["factor"].fillna(1.0)  # Default factor is 1.0
        else:
            # No adjustment data, use factor of 1.0
            df["factor"] = 1.0
            
        # Verify factors are valid
        df.loc[df["factor"] <= 0, "factor"] = 1.0
        
        # Apply adjustments to each price column
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col] * df["factor"]
                
        # Adjust volume inversely
        if "volume" in df.columns:
            df["volume"] = df["volume"] / df["factor"]
            
        return df
    
    def get_data_files(self) -> List[Path]:
        """Get list of data files to process.
        
        Returns
        -------
        List[Path]
            List of data file paths
        """
        return list(self.data_dir.glob("*.csv"))
    
    def normalize_all(self) -> Dict[str, bool]:
        """Normalize all data files.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping symbols to success status
        """
        # Get list of files
        files = self.get_data_files()
        
        if not files:
            logger.warning(f"No data files found in {self.data_dir}")
            return {}
            
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[bold]Normalizing {len(files)} files...",
                total=len(files)
            )
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for file_path in files:
                    symbol = self.get_symbol_from_path(file_path)
                    
                    try:
                        # Update progress
                        progress.update(task, description=f"Normalizing {symbol}")
                        
                        # Normalize file
                        df = self.normalize_single_file(file_path)
                        
                        if df.empty:
                            logger.warning(f"No data after normalization for {symbol}")
                            results[symbol] = False
                            continue
                            
                        # Save normalized data
                        output_file = self.output_dir / file_path.name
                        df.to_csv(output_file, index=False)
                        
                        results[symbol] = True
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        results[symbol] = False
                        
                    # Update progress
                    progress.advance(task)
        
        # Log results
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Normalization complete: {success_count}/{len(files)} files successful")
        
        return results

class DailyFMPNormalizer(BaseFMPNormalizer):
    """Normalizer for daily FMP data."""
    
    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get trading calendar from daily data files.
        
        Returns
        -------
        List[pd.Timestamp]
            List of trading dates
        """
        dates = set()
        
        # Collect all dates from data files
        for file_path in self.get_data_files():
            try:
                df = pd.read_csv(file_path)
                if not df.empty and self.date_field_name in df.columns:
                    df[self.date_field_name] = pd.to_datetime(df[self.date_field_name])
                    dates.update(df[self.date_field_name].tolist())
            except Exception as e:
                logger.warning(f"Error reading dates from {file_path}: {e}")
                
        # Sort dates
        return sorted(dates)
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize daily data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw daily data
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        df = super().normalize(df)
        
        if df.empty:
            return df
            
        # Calculate price changes
        df["change"] = self.calc_change(df)
        
        # Ensure the first close price is set to 1.0 if needed
        if "close" in df.columns and not df["close"].empty:
            first_valid_close = df["close"].iloc[0]
            if not pd.isna(first_valid_close) and first_valid_close != 1.0:
                # Record first close for verification
                df["first_close"] = first_valid_close
        
        return df
    
    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: Optional[float] = None) -> pd.Series:
        """Calculate price changes between consecutive days.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data
        last_close : Optional[float]
            Previous day's closing price for first day calculation
            
        Returns
        -------
        pd.Series
            Price changes
        """
        if "close" not in df.columns or df.empty:
            return pd.Series(index=df.index)
            
        # Calculate price changes
        close_series = df["close"].copy()
        close_series = close_series.fillna(method="ffill")
        
        # Shift to get previous close
        prev_close = close_series.shift(1)
        
        # Use provided last_close for first day if available
        if last_close is not None and not df.empty:
            prev_close.iloc[0] = last_close
            
        # Calculate change
        change = close_series / prev_close - 1.0
        
        return change

class IntradayFMPNormalizer(BaseFMPNormalizer):
    """Normalizer for intraday FMP data."""
    
    # US Market Hours (Eastern Time)
    AM_RANGE: Tuple[str, str] = ("09:30:00", "11:59:59")
    PM_RANGE: Tuple[str, str] = ("12:00:00", "15:59:59")
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        qlib_daily_dir: Union[str, Path],
        interval: str = "15min",
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        max_workers: int = 8
    ):
        """Initialize intraday normalizer.
        
        Parameters
        ----------
        data_dir : Union[str, Path]
            Directory containing raw intraday data
        output_dir : Union[str, Path]
            Directory to save normalized data
        qlib_daily_dir : Union[str, Path]
            Directory containing daily Qlib data (for adjustments)
        interval : str
            Intraday interval (1min, 5min, 15min, etc.)
        date_field_name : str
            Name of date column
        symbol_field_name : str
            Name of symbol column
        max_workers : int
            Maximum number of parallel workers
        """
        super().__init__(
            data_dir=data_dir,
            output_dir=output_dir,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            max_workers=max_workers
        )
        
        self.interval = interval
        self.qlib_daily_dir = Path(qlib_daily_dir).expanduser().resolve()
        
        # Initialize Qlib
        qlib.init(provider_uri=str(self.qlib_daily_dir))
        
        # Cache for daily data
        self._calendar_list_daily = None
        self._daily_data_cache = {}
    
    @property
    def calendar_list_daily(self) -> List[pd.Timestamp]:
        """Get daily calendar list.
        
        Returns
        -------
        List[pd.Timestamp]
            List of daily trading dates
        """
        if self._calendar_list_daily is None:
            from qlib.data import D
            
            # Get calendar from Qlib
            calendar = D.calendar()
            self._calendar_list_daily = calendar
            
        return self._calendar_list_daily
    
    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get intraday calendar list.
        
        Returns
        -------
        List[pd.Timestamp]
            List of intraday timestamps
        """
        # Generate intraday calendar from daily calendar
        calendar = generate_minutes_calendar_from_daily(
            self.calendar_list_daily,
            freq=self.interval,
            am_range=self.AM_RANGE,
            pm_range=self.PM_RANGE
        )
        
        return calendar.tolist()
    
    def _get_daily_data(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Get daily data for adjustments.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_time : pd.Timestamp
            Start date
        end_time : pd.Timestamp
            End date
            
        Returns
        -------
        pd.DataFrame
            Daily data with adjustment factors
        """
        # Check cache
        cache_key = f"{symbol}_{start_time.date()}_{end_time.date()}"
        if cache_key in self._daily_data_cache:
            return self._daily_data_cache[cache_key]
            
        try:
            # Extend the query range by one day to get proper factors
            query_start = start_time - pd.Timedelta(days=1)
            
            # Get data from Qlib
            daily_df = D.features(
                instruments=symbol,
                fields=["$volume", "$factor", "$close"],
                start_time=query_start,
                end_time=end_time,
                freq="day"
            )
            
            # Clean the data
            daily_df = daily_df.dropna(how="all")
            
            if not daily_df.empty:
                # Add paused column
                daily_df["paused"] = 0
                daily_df.loc[(daily_df["$volume"].isna()) | (daily_df["$volume"] <= 0), "paused"] = 1
                
                # Cache the result
                self._daily_data_cache[cache_key] = daily_df
                return daily_df
                
            # No data available, create default DataFrame
            logger.warning(f"No daily data for {symbol}, using default factors")
            dates = pd.date_range(start=query_start.date(), end=end_time.date(), freq="D")
            default_df = pd.DataFrame(
                {
                    "$factor": 1.0,
                    "$volume": np.nan,
                    "$close": np.nan,
                    "paused": 0
                },
                index=dates
            )
            
            # Cache the result
            self._daily_data_cache[cache_key] = default_df
            return default_df
            
        except Exception as e:
            logger.error(f"Error getting daily data for {symbol}: {e}")
            
            # Create empty DataFrame with default values
            dates = pd.date_range(start=start_time.date(), end=end_time.date(), freq="D")
            default_df = pd.DataFrame(
                {
                    "$factor": 1.0,
                    "$volume": np.nan,
                    "$close": np.nan,
                    "paused": 0
                },
                index=dates
            )
            
            # Cache the result
            self._daily_data_cache[cache_key] = default_df
            return default_df
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize intraday data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw intraday data
            
        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        if df.empty:
            return df
            
        # Extract symbol
        symbol = df[self.symbol_field_name].iloc[0]
        
        # Basic normalization
        df = df.copy()
        df.set_index(self.date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone info
        df = df[~df.index.duplicated(keep="first")]  # Remove duplicates
        df.sort_index(inplace=True)
        
        # Get data range
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Get daily data for adjustments
        daily_df = self._get_daily_data(symbol, start_time, end_time)
        
        # Check for today's live data
        today = pd.Timestamp.now().normalize()
        is_today = end_time.normalize() == today
        
        # Apply adjustments using daily factors
        if not daily_df.empty:
            # Get the calendar dates
            calendar = self.calendar_list
            
            # Reindex data to align with calendar
            df_aligned = df.reindex(pd.DatetimeIndex(calendar))
            
            # Apply adjustments
            adjusted_df = calc_adjusted_price(
                df_aligned, 
                daily_df,
                frequence=self.interval
            )
            
            # Adjust volume for zero/NaN
            adjusted_df.loc[
                (adjusted_df["volume"] <= 0) | np.isnan(adjusted_df["volume"]),
                self.COLUMNS
            ] = np.nan
            
            # Drop rows where all values are NaN
            adjusted_df = adjusted_df.dropna(how="all")
            
            # Reset index to get date as a column
            adjusted_df = adjusted_df.reset_index()
            
            return adjusted_df
            
        # No daily data available, return raw data with warning
        logger.warning(f"No daily data for {symbol}, returning raw data")
        return df.reset_index()

def run_normalization(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    interval: str = "1d",
    qlib_daily_dir: Optional[Union[str, Path]] = None,
    max_workers: int = 8
) -> None:
    """Run data normalization process.
    
    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing raw data
    output_dir : Union[str, Path]
        Directory to save normalized data
    interval : str
        Data interval (1d, 15min, etc.)
    qlib_daily_dir : Optional[Union[str, Path]]
        Directory containing daily Qlib data (for intraday adjustment)
    max_workers : int
        Maximum number of parallel workers
    """
    console.print(f"[bold]Normalizing {interval} data from {data_dir}...[/bold]")
    
    # Create normalizer based on interval
    if interval == "1d":
        normalizer = DailyFMPNormalizer(
            data_dir=data_dir,
            output_dir=output_dir,
            max_workers=max_workers
        )
    else:
        if not qlib_daily_dir:
            raise ValueError("qlib_daily_dir is required for intraday normalization")
            
        normalizer = IntradayFMPNormalizer(
            data_dir=data_dir,
            output_dir=output_dir,
            qlib_daily_dir=qlib_daily_dir,
            interval=interval,
            max_workers=max_workers
        )
    
    # Run normalization
    results = normalizer.normalize_all()
    
    # Print results
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    console.print(f"[bold green]Normalization complete: {success_count}/{total_count} files successful[/bold green]")

if __name__ == "__main__":
    # Example usage
    run_normalization(
        data_dir="~/.qlib/qlib_data/fmp_daily_raw",
        output_dir="~/.qlib/qlib_data/fmp_daily_norm",
        interval="1d",
        max_workers=8
    ) 