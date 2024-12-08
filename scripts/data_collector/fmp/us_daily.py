"""US Daily Stock Data Collector for Financial Modeling Prep (FMP).

This module implements the data collection and normalization for US daily stock data
from FMP API.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

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

console = Console()

class FMPDailyCollector(BaseFMPCollector):
    """Collector for US daily stock data from FMP API."""
    
    DEFAULT_START_DATETIME_1D = pd.Timestamp("2000-01-01")
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.now())
    
    def __init__(
        self,
        save_dir: Union[str, Path] = "~/.qlib/qlib_data/us_fmp_d1",
        start: Optional[str] = None,
        end: Optional[str] = None,
        delay: float = 0.1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        instruments: Optional[List[str]] = None,
        skip_existing: bool = False,
    ):
        """Initialize FMP daily collector."""
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval="1d",
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            instruments=instruments,
            skip_existing=skip_existing,
        )

class FMPDailyNormalize(BaseFMPNormalize):
    """Normalize FMP daily data to Qlib format.
    
    This class handles the normalization of daily stock data from FMP to Qlib format.
    It includes functionality for:
    1. Calculating price changes
    2. Adjusting prices using adjustment factors
    3. Manual data adjustments for consistency
    4. Calendar alignment
    """
    
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    def _get_calendar_list(self) -> List[pd.Timestamp]:
        """Get trading calendar from all instrument files.
        
        Returns
        -------
        List[pd.Timestamp]
            Sorted list of unique trading dates.
        """
        df = pd.DataFrame()
        for f in self.instruments_dir.glob("*.csv"):
            _df = pd.read_csv(f)
            _df["date"] = pd.to_datetime(_df["date"])
            df = pd.concat([df, _df[["date"]]], ignore_index=True)
        return sorted(df["date"].unique().tolist())

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: Optional[float] = None) -> pd.Series:
        """Calculate price changes between consecutive trading days.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data
        last_close : float, optional
            Previous day's closing price for first day calculation
            
        Returns
        -------
        pd.Series
            Series containing price changes
        """
        df = df.copy()
        _tmp_series = df["close"].copy()
        _tmp_series = _tmp_series.fillna(method="ffill") # type: ignore
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data to Qlib format.
        
        This method performs several steps:
        1. Basic normalization (calendar alignment, data cleaning)
        2. Price adjustments
        3. Manual adjustments for data consistency
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data from FMP
            
        Returns
        -------
        pd.DataFrame
            Normalized data in Qlib format
        """
        if df.empty:
            return df
            
        # Basic normalization
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        
        # Calendar alignment
        if self._calendar_list is not None:
            calendar_index = pd.DatetimeIndex(self._calendar_list) # type: ignore
            date_range = pd.date_range(
                start=pd.Timestamp(df.index.min()).date(),
                end=pd.Timestamp(df.index.max()).date() + pd.Timedelta(hours=23, minutes=59),
                freq='D'
            )
            df = df.reindex(date_range.intersection(calendar_index))
            
        df.sort_index(inplace=True)
        
        # Handle invalid volume data
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {self._symbol_field_name})] = np.nan
        
        # Calculate changes
        df["change"] = self.calc_change(df)
        
        # Price adjustments
        df = self.adjusted_price(df)
        
        # Manual adjustments
        df = self._manual_adj_data(df)
        
        return df.reset_index()

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust prices using adjustment factors.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw prices
            
        Returns
        -------
        pd.DataFrame
            DataFrame with adjusted prices
        """
        if df.empty:
            return df
            
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        
        # Calculate adjustment factor
        if "adjClose" in df:
            df["factor"] = df["adjClose"] / df["close"]
            df["factor"] = df["factor"].copy()
            df["factor"] = df["factor"].fillna(method="ffill") # type: ignore
        else:
            df["factor"] = 1
            
        # Apply adjustments
        for col in self.COLUMNS:
            if col not in df.columns:
                continue
            if col == "volume":
                df[col] = df[col] / df["factor"]
            else:
                df[col] = df[col] * df["factor"]
                
        return df.reset_index()

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """Get the first valid close price.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data
            
        Returns
        -------
        float
            First valid closing price
        """
        df = df.loc[df["close"].first_valid_index():]
        return df["close"].iloc[0]

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manually adjust data for consistency.
        
        All fields (except change) are standardized according to the close price
        of the first day.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to adjust
            
        Returns
        -------
        pd.DataFrame
            Manually adjusted data
        """
        if df.empty:
            return df
            
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        
        _close = self._get_first_close(df)
        
        for col in df.columns:
            if col in [self._symbol_field_name, "adjClose", "change"]:
                continue
            if col == "volume":
                df[col] = df[col] * _close
            else:
                df[col] = df[col] / _close
                
        return df.reset_index()

class FMPDailyNormalizeExtend(BaseFMPNormalizeExtend, FMPDailyNormalize):
    """Normalize FMP daily data while maintaining consistency with existing Qlib data.
    
    This class extends FMPDailyNormalize to ensure data consistency when updating
    existing Qlib datasets. It follows the same pattern as YahooNormalize1dExtend
    to maintain compatibility and consistency across data sources.
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
            Additional arguments passed to parent class
        """
        super().__init__(date_field_name=date_field_name,
                        symbol_field_name=symbol_field_name,
                        **kwargs)
        
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
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
        qlib.init(provider_uri=qlib_data_dir, 
                 expression_cache=None,
                 dataset_cache=None)
        
        # Get features for all instruments
        df = D.features(
            D.instruments("all"),
            ["$" + col for col in self.column_list]
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
            for col in self.column_list[:-1]:  # Skip 'change' as it's recalculated
                if new_latest_data[col] == 0 or pd.isna(new_latest_data[col]):
                    logger.warning(f"Invalid overlap value for {col} in {symbol_name}, skipping adjustment")
                    continue
                    
                if col == "volume":
                    # Volume needs to be divided to maintain ratio
                    df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
                else:
                    # Prices and factor need to be multiplied to maintain ratio
                    df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
            
            # Remove the overlapping point to avoid duplication
            df = df.drop(df.index[0])
            
            # Recalculate change after adjustments
            df["change"] = df["close"].pct_change()
            
        except Exception as e:
            logger.error(f"Error adjusting data for {symbol_name}: {str(e)}")
            # Return standard normalized data if adjustment fails
            return df.reset_index()
            
        return df.reset_index()

class FMPDailyRunner(BaseFMPRunner):
    """Runner for FMP daily data collection process."""
    
    def __init__(self, max_workers: int = 4):
        super().__init__(max_workers=max_workers, interval="1d")
        
    @property
    def collector_class_name(self) -> str:
        return "FMPDailyCollector"
        
    @property
    def normalize_class_name(self) -> str:
        return "FMPDailyNormalize"

if __name__ == "__main__":
    import fire
    fire.Fire(FMPDailyRunner)