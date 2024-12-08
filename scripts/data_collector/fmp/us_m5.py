"""US 5-Minute Stock Data Collector for Financial Modeling Prep (FMP).

This module implements the data collection and normalization for US 5-minute stock data
from FMP API.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console

import qlib
from qlib.data import D
from scripts.data_collector.fmp.collector import (
    BaseFMPCollector,
    BaseFMPNormalize,
    BaseFMPNormalizeExtend,
    BaseFMPRunner,
)
from scripts.data_collector.utils import (
    calc_adjusted_price,
    generate_minutes_calendar_from_daily,
)

console = Console()

class FMP5minCollector(BaseFMPCollector):
    """Collector for US 5-minute stock data from FMP API."""
    
    DEFAULT_START_DATETIME_5MIN = pd.Timestamp(datetime.now() - timedelta(days=30))
    DEFAULT_END_DATETIME_5MIN = pd.Timestamp(datetime.now())
    
    def __init__(
        self,
        save_dir: Union[str, Path] = "~/.qlib/qlib_data/us_fmp_5min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        delay: float = 0.1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        instruments: Optional[List[str]] = None,
        skip_existing: bool = False,
        qlib_data_1d_dir: str = "~/.qlib/qlib_data/us_fmp_d1",
    ):
        """Initialize FMP 5-minute collector."""
        # Check if 1d data exists
        qlib_data_1d_path = Path(qlib_data_1d_dir).expanduser()
        if not qlib_data_1d_path.exists():
            raise ValueError(f"1d data directory not found: {qlib_data_1d_path}")
        self.qlib_data_1d_path = qlib_data_1d_path
        
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval="5min",
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            instruments=instruments,
            skip_existing=skip_existing,
        )

class FMP5minNormalize(BaseFMPNormalize):
    """Normalize FMP 5-minute data to Qlib format.
    
    This class handles the normalization of 5-minute stock data from FMP to Qlib format.
    The normalization process follows these exact steps:

    1. Basic Data Preparation:
       - Convert index to datetime: df.index = pd.to_datetime(df.index)
       - Remove timezone info: df.index = df.index.tz_localize(None)
       - Remove duplicates: df = df[~df.index.duplicated(keep="first")]
       - Sort by timestamp: df.sort_index(inplace=True)

    2. Daily Data Integration:
       - Get daily data for adjustment using D.features():
         * Fields: ["$volume", "$factor", "$close"]
         * Time range: [start_time - 1day, end_time]
       - For live trading (when end_time is today during market hours):
         * Query range: [start_time - 1day, last_trading_day]
         * Extend with today's entry using last known values
         * Set paused=0 for live data
       - If no daily data:
         * Create default DataFrame with factor=1.0
         * Mark as not paused (paused=0)

    3. Price Adjustment Process (via calc_adjusted_price):
       - For each price field (OHLC):
         adjusted_price = raw_price * factor
       - For volume:
         adjusted_volume = raw_volume / factor
       - Factors come from daily data's $factor field
       - Live trading uses previous day's factor

    4. Calendar Alignment:
       - Get market hours calendar:
         * AM: 9:30 AM - 11:59:59 AM ET
         * PM: 12:00 PM - 3:59:59 PM ET
       - Create 5-min intervals: pd.date_range(freq="5min")
       - Intersect with trading calendar
       - Reindex data to aligned timestamps

    5. Data Quality Handling:
       - Invalid volume check:
         df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), fields] = np.nan
       - Paused detection from daily data:
         daily_df.loc[(daily_df["$volume"].isna()) | (daily_df["$volume"] <= 0), "paused"] = 1

    6. Special Cases:
       a) Live Trading:
          - Detect using: end_time.normalize() == today && is_trading_hours
          - Use last_trading_day's data for adjustments
          - Extend daily data with:
            * last_factor from daily_df["$factor"].iloc[-1]
            * last_volume from daily_df["$volume"].iloc[-1]
            * last_close from daily_df["$close"].iloc[-1]
            * paused = 0 (assuming active trading)
       
       b) Missing Daily Data:
          - Create default data with:
            * $factor = 1.0
            * $volume = np.nan
            * $close = np.nan
            * paused = 0
          - Log warning about using raw data

    Key Features:
    - Maintains price continuity across corporate actions
    - Handles both historical and live data consistently
    - Caches daily data for efficiency
    - Properly aligns with market trading hours
    - Robust handling of missing or invalid data

    Dependencies:
    - Requires daily data with factors for proper adjustment
    - Uses qlib.data.D for data access
    - Relies on calc_adjusted_price for core adjustment logic
    """
    
    COLUMNS = ["open", "close", "high", "low", "volume"]
    
    # US Market Hours (Eastern Time)
    AM_RANGE: Tuple[str, str] = ("09:30:00", "11:59:59")
    PM_RANGE: Tuple[str, str] = ("12:00:00", "15:59:59")
    
    CALC_PAUSED_NUM = False  # US market doesn't need paused number calculation
    
    def __init__(
        self,
        qlib_data_1d_dir: Union[str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs
    ):
        """Initialize the normalizer."""
        super().__init__(date_field_name, symbol_field_name)
        qlib.init(provider_uri=str(qlib_data_1d_dir))
        
        # Store required fields from 1d data
        self._1d_fields = ["$volume", "$factor", "$close"]
        self._calendar_list_1d = None
        self._1d_data_cache: Dict[str, pd.DataFrame] = {}
        
    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get calendar list for 5-minute data.
        
        This method generates 5-minute intervals for US market hours:
        - Morning session: 9:30 AM - 12:00 PM ET
        - Afternoon session: 12:00 PM - 4:00 PM ET
        
        Note that US market trades continuously without a lunch break,
        but we split the ranges at noon for better data management.
        
        Returns
        -------
        List[pd.Timestamp]
            List of 5-minute timestamps during market hours
        """
        calendar = generate_minutes_calendar_from_daily(
            self.calendar_list_1d,
            freq="5min",
            am_range=self.AM_RANGE,
            pm_range=self.PM_RANGE
        )
        return calendar.tolist()
    
    @property
    def calendar_list_1d(self) -> List[pd.Timestamp]:
        """Get 1d calendar list.
        
        Returns
        -------
        List[pd.Timestamp]
            List of daily timestamps
        """
        if self._calendar_list_1d is None:
            self._calendar_list_1d = list(D.calendar(freq="day"))
        return self._calendar_list_1d
    
    def _get_1d_data(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """Get required 1d data for a symbol efficiently."""
        cache_key = f"{symbol}_{start_time}_{end_time}"
        if cache_key not in self._1d_data_cache:
            # Get the last available trading day from calendar
            last_trading_day = D.calendar(freq="day")[-1]
            
            try:
                # For any data request, always get one extra day before start_time
                # to ensure we have previous day's close and factor
                query_start = start_time.normalize() - pd.Timedelta(days=1)
                
                # For live trading during market hours:
                # Use data up to the last available trading day
                now = pd.Timestamp.now(tz='America/New_York')
                today = now.normalize()
                is_trading_hours = (
                    now.time() >= pd.Timestamp(self.AM_RANGE[0]).time() and 
                    now.time() <= pd.Timestamp(self.PM_RANGE[1]).time()
                )
                
                if is_trading_hours and end_time.normalize() == today:
                    query_end = last_trading_day
                    logger.warning(
                        f"Processing live data for {symbol} during market hours "
                        f"({start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}). "
                        "Using last available trading day's data."
                    )
                    
                    # Get historical data
                    daily_df = D.features(
                        [symbol], 
                        self._1d_fields,
                        start_time=query_start,
                        end_time=query_end,
                        freq="day"
                    )
                    
                    if daily_df.empty:
                        raise ValueError("No historical daily data available")
                    
                    # For live data, extend daily data with today using last known values
                    # and set paused=0 since we're getting live data
                    last_factor = daily_df["$factor"].iloc[-1]
                    last_volume = daily_df["$volume"].iloc[-1]
                    last_close = daily_df["$close"].iloc[-1]
                    
                    today_data = pd.DataFrame(
                        {
                            "$factor": [last_factor],
                            "$volume": [last_volume],
                            "$close": [last_close],
                            "paused": [0],  # Not paused since we're getting live data
                        },
                        index=pd.MultiIndex.from_tuples([(symbol, today)], names=["instrument", "datetime"])
                    )
                    daily_df = pd.concat([daily_df, today_data])
                    
                else:
                    # Normal case - get historical data
                    daily_df = D.features(
                        [symbol], 
                        self._1d_fields,
                        start_time=query_start,
                        end_time=end_time,
                        freq="day"
                    )
                    if daily_df.empty:
                        raise ValueError("No daily data available")
                    
                    # Add paused field based on volume
                    daily_df["paused"] = 0
                    daily_df.loc[(daily_df["$volume"].isna()) | (daily_df["$volume"] <= 0), "paused"] = 1
                    
            except (ValueError, KeyError) as e:
                # Handle case when no daily data is available
                logger.warning(
                    f"No daily data available for {symbol}. Using raw data with factor=1. "
                    f"Error: {str(e)}"
                )
                # Create a default DataFrame with factor=1 for the entire period
                dates = pd.date_range(
                    start=start_time.normalize() - pd.Timedelta(days=1),  # Include previous day
                    end=end_time.normalize(),
                    freq='D'
                )
                daily_df = pd.DataFrame(
                    {
                        "$factor": 1.0,
                        "$volume": np.nan,
                        "$close": np.nan,
                        "paused": 0,  # Default to not paused
                    },
                    index=pd.MultiIndex.from_product(
                        [[symbol], dates],
                        names=["instrument", "datetime"]
                    )
                )
            
            self._1d_data_cache[cache_key] = daily_df
            
        return self._1d_data_cache[cache_key]
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data to Qlib format."""
        if df.empty:
            return df
            
        # Basic normalization
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        
        # Get symbol and date range for 1d data
        symbol = df[self._symbol_field_name].iloc[0]
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Get 1d data for adjustments (will handle missing data case)
        daily_df = self._get_1d_data(symbol, start_time, end_time)
        
        # Calendar alignment
        if self._calendar_list is not None:
            calendar_index = pd.DatetimeIndex(list(self._calendar_list))
            date_range = pd.date_range(
                start=start_time,
                end=end_time,
                freq="5min"
            )
            df = df.reindex(date_range.intersection(calendar_index))
            
        df.sort_index(inplace=True)
        
        # Handle invalid volume data
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {self._symbol_field_name})] = np.nan
        
        # Adjust prices using 1d data
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="5min",
            _1d_data_all=daily_df,
            calc_paused=self.CALC_PAUSED_NUM,
            am_range=self.AM_RANGE,
            pm_range=self.PM_RANGE,
            consistent_1d=True,  # Ensure alignment with daily data
        )
        
        return df.reset_index()

class FMP5minNormalizeExtend(BaseFMPNormalizeExtend, FMP5minNormalize):
    """Normalize FMP 5-minute data while maintaining consistency with existing Qlib data.
    
    This class extends FMP5minNormalize to ensure data consistency when updating
    existing Qlib datasets. It does this by:
    1. Loading existing data from the old Qlib directory
    2. Finding the overlap point between old and new data
    3. Adjusting new data to maintain continuity with old data
    4. Ensuring price/volume ratios remain consistent
    
    This is particularly important when:
    - Updating existing datasets with new data
    - Ensuring price continuity across updates
    - Maintaining consistent adjustment factors
    """
    
    def __init__(
        self,
        old_qlib_data_dir: Union[str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs
    ):
        """Initialize the extended normalizer.
        
        Parameters
        ----------
        old_qlib_data_dir: Union[str, Path]
            Directory containing existing Qlib data to be updated
        date_field_name: str
            Name of the date field, default is "date"
        symbol_field_name: str
            Name of the symbol field, default is "symbol"
        kwargs: dict
            Additional arguments passed to parent class, including qlib_data_1d_dir
        """
        # Initialize the parent FMP5minNormalize first
        super().__init__(date_field_name=date_field_name,
                        symbol_field_name=symbol_field_name,
                        **kwargs)
        
        self.column_list = ["open", "high", "low", "close", "volume"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)
        
    def _get_old_data(self, qlib_data_dir: Union[str, Path]) -> pd.DataFrame:
        """Get existing data from old Qlib directory.
        
        Parameters
        ----------
        qlib_data_dir : Union[str, Path]
            Path to existing Qlib data directory
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing existing data with features
        """
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        # Initialize a new Qlib instance for old data
        qlib.init(provider_uri=qlib_data_dir, 
                 expression_cache=None,
                 dataset_cache=None)
        
        # Get features for all instruments
        df = D.features(
            D.instruments("all"), 
            ["$" + col for col in self.column_list],
            freq="5min"  # Important: specify 5min frequency
        )
        df.columns = self.column_list
        return df
        
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data while maintaining consistency with existing data.
        
        This method:
        1. Performs basic normalization using parent class
        2. Checks if symbol exists in old data
        3. If exists, adjusts new data to maintain continuity
        4. Handles special cases like missing data or gaps
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data to normalize
            
        Returns
        -------
        pd.DataFrame
            Normalized data aligned with existing data
        """
        # First apply standard normalization
        df = super().normalize(df)
        
        # Prepare data for alignment
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        
        # Check if symbol exists in old data
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            logger.warning(f"Symbol {symbol_name} not found in existing data, using standard normalization")
            return df.reset_index()
            
        # Get old data for this symbol
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        
        try:
            # Find the latest timestamp in old data
            latest_date = old_df.index[-1]
            
            # Get data from the overlap point onwards
            df = df.loc[latest_date:]
            
            if df.empty:
                logger.warning(f"No new data after {latest_date} for {symbol_name}")
                return df.reset_index()
                
            # Get the overlapping data point
            new_latest_data = df.iloc[0]
            old_latest_data = old_df.loc[latest_date]
            
            # Adjust each column to maintain continuity
            for col in self.column_list:
                if new_latest_data[col] == 0 or pd.isna(new_latest_data[col]):
                    logger.warning(f"Invalid overlap value for {col} in {symbol_name}, skipping adjustment")
                    continue
                    
                if col == "volume":
                    # Volume needs to be divided to maintain ratio
                    df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
                else:
                    # Prices need to be multiplied to maintain ratio
                    df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
            
            # Remove the overlapping point to avoid duplication
            df = df.drop(df.index[0])
            
        except Exception as e:
            logger.error(f"Error adjusting data for {symbol_name}: {str(e)}")
            # Return standard normalized data if adjustment fails
            return df.reset_index()
            
        return df.reset_index()

class FMP5minRunner(BaseFMPRunner):
    """Runner for FMP 5-minute data collection process."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the runner.
        
        Parameters
        ----------
        max_workers : int
            Maximum number of worker processes
        """
        super().__init__(max_workers=max_workers, interval="5min")
        
    @property
    def collector_class_name(self) -> str:
        return "FMP5minCollector"
        
    @property
    def normalize_class_name(self) -> str:
        return "FMP5minNormalize"
        
    def run(
        self,
        save_dir: Union[str, Path],
        qlib_dir: Union[str, Path],
        qlib_data_1d_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        delay: float = 0.1,
        limit_nums: Optional[int] = None,
        instruments: Optional[List[str]] = None,
        skip_existing: bool = False,
        dump_bin: bool = True,
        dump_all: bool = False,
        dump_update: bool = True,
    ):
        """Run the data collection and dumping process.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save raw data
        qlib_dir : Union[str, Path]
            Directory for Qlib format data
        qlib_data_1d_dir : Union[str, Path]
            Directory containing daily data for adjustments
        start : Optional[str]
            Start date for data collection
        end : Optional[str]
            End date for data collection
        delay : float
            Delay between API calls
        limit_nums : Optional[int]
            Limit number of symbols to process
        instruments : Optional[List[str]]
            List of instruments to process
        skip_existing : bool
            Whether to skip existing files
        dump_bin : bool
            Whether to dump to binary format
        dump_all : bool
            Whether to dump all data (overwrite existing)
        dump_update : bool
            Whether to update existing data
        """
        logger.info("Starting FMP 5-minute data collection process...")
        
        # Download and optionally dump data
        self.download_data(
            save_dir=save_dir,
            qlib_dir=qlib_dir,
            start=start,
            end=end,
            delay=delay,
            limit_nums=limit_nums,
            instruments=instruments,
            skip_existing=skip_existing,
            dump_bin=dump_bin,
            dump_all=dump_all,
            dump_update=dump_update,
            qlib_data_1d_dir=qlib_data_1d_dir,  # Pass 1d data dir for adjustments
        )
        
        logger.info("FMP 5-minute data collection process completed.")

if __name__ == "__main__":
    import fire
    fire.Fire(FMP5minRunner)