# -*- coding: utf-8 -*-
import json
import os
from abc import ABC
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Annotated, List, Optional

import pandas as pd

from src import PACKAGE_ROOT
from src.fx_info import FxSymbol, TimeFrame
from src.logger import setup_logger

DATA_DIR = os.getenv("DATA_DIR", PACKAGE_ROOT / "db/market_data")


@lru_cache
def _get_data_by_years(
    data_dir: Path, symbol: str, start_year: int, end_year: int, _today: str
) -> pd.DataFrame:
    """
    Returns a DataFrame containing the bar data for the specified symbol and years.

    Args:
        data_dir (Path): The directory where the bar data is stored as:
            {data_dir}/{provider}/{timeframe}
        symbol (str): The symbol to retrieve data for.
        start_year (int): The start year of the data, inclusive.
        end_year (int): The end year of the data, inclusive.
        _today (str): The current date, this is used to makes the cache on different days.
            Otherwise when year is current year and on differnt days the cache will return different results.

    Returns:
        pd.DataFrame: A DataFrame containing the bar data for the specified symbol and years.
    """
    data_files = []
    for year in range(start_year, end_year + 1):
        data_files.extend(list(data_dir.glob(f"{year}/{symbol}.csv")))

    if not data_files:
        return pd.DataFrame()

    data = pd.concat(
        [pd.read_csv(file, index_col="time", parse_dates=True) for file in data_files]
    )
    return data.sort_index()


class BarDataProvider(ABC):
    """
    Abstract base class for bar data providers.

    Attributes:
        data_dir (str): The directory where the bar data is stored.
        provider (str): The name of the data provider, act as the db directory name.

    Methods:
        download_bars(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
            Abstract method to download bar data.
        _download_bars(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
            Child classes should implement this method to download bar data.
        get_bars(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
            Get bar data from either the provider or local CSV files.
        bulk_download(symbols: List[str|FxSymbol]|None = None, timeframes: List[str|TimeFrame]|None = None, start_date: str='2019-01-01', end_date: str|None=None) -> None:
            Download bar data for all symbols and timeframes.
    """

    @classmethod
    def check_availability(cls) -> bool:
        return True

    provider = "" 
    def __init__(self, data_dir: str | Path = DATA_DIR):
        """
        Initialize the BarDataProvider.

        Args:
            data_dir (str): The directory where the bar data is stored.
            provider (str): The name of the data provider.
        """
        if Path(data_dir).is_absolute():
            self.data_dir = data_dir
        else:
            self.data_dir = PACKAGE_ROOT / data_dir
        self.logger = setup_logger(self.provider)

        self._meta_info_file = Path(self.data_dir) / "meta_info.json"

        if self._meta_info_file.exists():
            self.meta_info = json.load(open(self._meta_info_file))
        else:
            self._meta_info_file.parent.mkdir(parents=True, exist_ok=True)
            self.meta_info = {}

        self._last_price = {}  # Store the last price for each symbol, to use as a cache when in error

    def _check_and_download_year(self, symbol, timeframe, start_date, end_date):
        """
        Check if the data for the year is already downloaded, if not download the data.
        """
        try:
            saved_start_date = self.meta_info[self.provider][timeframe][symbol][
                "start_date"
            ]
        except KeyError:
            saved_start_date = "2050-01-01"
        try:
            saved_end_date = self.meta_info[self.provider][timeframe][symbol][
                "end_date"
            ]
        except KeyError:
            saved_end_date = "1900-01-01"
        if start_date >= saved_start_date and end_date <= saved_end_date:
            return
        # Download data from the first day of the year of start_date to the last day of the year of end_date
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        for year in range(start_year, end_year + 1):
            year_start_date = f"{year}-01-01"
            year_end_date = f"{year}-12-31"
            self.get_bars(symbol, timeframe, year_start_date, year_end_date)

    @lru_cache(maxsize=None)
    def get_price(
        self,
        symbol: str,
        at_time: Annotated[
            date | datetime | str | None, "The time to get the price for"
        ] = None,
        use_bid: Annotated[
            bool, "Whether to use bid price instead of ask price"
        ] = True,
    ) -> float:
        """
        Get the price of a symbol at a specific time.

        Warning: This method is a fuzzy search and may not return the exact price at the specified time.

        Args:
            symbol (str): The symbol to retrieve data for.
            at_time (date|datetime|str|None): The time to get the price for.
                if None, the latest price stored in the data provider will be returned.

        Returns:
            float: The price of the symbol at the specified time.

        Raises:
            ValueError: If the field is not one of 'open', 'high', 'low', or 'close'.
            ValueError: If no data is loaded for the symbol and time frame requested at the time.
        """
        # Get the price field
        field = "open"

        # Make a start and end date that covers the at_time
        if at_time is None:
            at_time = pd.Timestamp.now()
            end_date = str(at_time.date())
            timeframe = TimeFrame("D1")
        else:
            _end_date = pd.to_datetime(at_time).date() + pd.Timedelta(days=1)
            if _end_date.year != pd.to_datetime(at_time).year:
                _end_date = pd.to_datetime(at_time).date()
            end_date = str(_end_date)
            timeframe = TimeFrame.get_mod_timeframe(at_time)

        end_date_dt = pd.to_datetime(end_date).date()
        
        # Get data for the entire year containing the end_date
        year_start = str(end_date_dt.replace(month=1, day=1))
        year_end = str(min(end_date_dt.replace(month=12, day=31), pd.Timestamp.now().date()))
        data = self.get_bars(symbol, timeframe, year_start, year_end)

        if data.empty:
            last_price = self._last_price.get(symbol)
            self.logger.debug(
                f"No data loaded for {symbol} {timeframe} requested at time {at_time}. Use the last price {last_price} stored in cache"
            )
            if last_price is not None:
                return last_price
            else:
                # If no last price stored in cache, raise error
                raise ValueError(
                    f"No data loaded for {symbol} {timeframe} requested at time {at_time}"
                )
        # Slice the data according to the original date range
        data = data[data.index <= pd.to_datetime(end_date)]

        # Find the price at time >= at_time
        if at_time is None:
            price = data[field].iloc[-1]
        else:
            if data.index.max() < pd.to_datetime(at_time):
                # Case when fetch during weekend or holiday, the `at_time` is current time
                price = data[field].iloc[-1]
            else:
                # Return the closest price at or after the `at_time`
                price = data[field].loc[data.index >= pd.to_datetime(at_time)].iloc[0]

        self._last_price[symbol] = price
        if use_bid:
            return price
        else:
            return price + FxSymbol(symbol).spread_price

    def _download_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Abstract method to download bar data.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (str): The timeframe of the data.
            start_date (str): The start date of the data (YYYY-MM-DD), inclusive.
            end_date (str): The end date of the data (YYYY-MM-DD), inclusive.

        Returns:
            pd.DataFrame: A DataFrame containing the downloaded bar data.
        """
        raise NotImplementedError("download_bars method is not implemented")

    @lru_cache
    def download_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Abstract method to download bar data.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (str): The timeframe of the data.
            start_date (str): The start date of the data (YYYY-MM-DD), inclusive.
            end_date (str): The end date of the data (YYYY-MM-DD), inclusive.
            kwargs: Additional keyword arguments for the download.

        Returns:
            pd.DataFrame: A DataFrame containing the downloaded bar data.
        """
        #TODO: correct to accomodate for historical data
        if start_date < "2000-01-01":
            self.logger.warning(
                f"Dowloading data before 2000-01-01 may not be accurate, setting start_date to 2000-01-01"
            )
            start_date = "2000-01-01"

        t0 = time()
        self.logger.info(
            f"Downloading {symbol} {timeframe} from {start_date} to {end_date}..."
        )
        data = self._download_bars(symbol, timeframe, start_date, end_date, **kwargs)
        t1 = time()
        self.logger.info(f"Downloaded {len(data)} bars in {t1 - t0:.2f} seconds")

        # Clear the cache, so that the new data is loaded
        if len(data) > 0:
            _get_data_by_years.cache_clear()
        return data

    def bulk_download(
        self,
        symbols: List[str | FxSymbol] | None = None,
        timeframes: List[str | TimeFrame] | None = None,
        start_date: str = "2019-01-01",
        end_date: str | None = None,
    ):
        """
        Download bar data for all symbols and timeframes.

        Args:
            symbols (List[str|FxSymbol]|None): The list of symbols to download data for.
                If None, all symbols (Exclude Stocks) will be downloaded.
            timeframes (List[str|TimeFrame]|None): The list of timeframes to download data for.
                If None, all timeframes will be downloaded.
            start_date (str): The start date of the data (YYYY-MM-DD). Default is '2019-01-01'.
            end_date (str|None): The end date of the data (YYYY-MM-DD). If None, the current date will be used.
                Default is None.
        """
        from tqdm import tqdm

        if symbols is None:
            symbols = FxSymbol.COMMON_SYMBOLS
        if timeframes is None:
            timeframes = TimeFrame.SUPPORTED_TIMEFRAMES
        if end_date is None:
            end_date = str(pd.Timestamp.now().date())

        failed_parms = []
        for symbol in tqdm(symbols, desc="Symbols"):
            for timeframe in timeframes:
                try:
                    # Download and save
                    self.get_bars(symbol, timeframe, start_date, end_date)

                except Exception as e:
                    self.logger.error(
                        f"Failed to download {symbol} {timeframe}: {str(e)}"
                    )
                    failed_parms.append((symbol, timeframe))
                    continue

        if failed_parms:
            self.logger.info(f"Retry downloading the failed symbols and timeframes")
            for symbol, timeframe in failed_parms:
                try:
                    # Download and save
                    self.get_bars(symbol, timeframe, start_date, end_date)

                except Exception as e:
                    self.logger.error(
                        f"Failed to download {symbol} {timeframe}: {str(e)}"
                    )
                    continue

    def _save_bars(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save the bar data to CSV files in the appropriate directory structure.

        Args:
            data (pd.DataFrame): The bar data to save.
            symbol (str): The symbol of the data.
            timeframe (str): The timeframe of the data.
        """
        data_dir = Path(self.data_dir) / self.provider / timeframe
        data_dir.mkdir(parents=True, exist_ok=True)

        for year, year_data in data.groupby(data.index.year):  # type: ignore #
            # Check if there exists a directory for the year
            # If there is, then read the data and append the new data
            # then drop duplicates index and save the data
            year_path = data_dir / str(year) / f"{symbol}.csv"
            if year_path.exists():
                exist_year_data = pd.read_csv(
                    year_path, index_col="time", parse_dates=True
                )
                year_data = pd.concat([exist_year_data, year_data])
                year_data = year_data[~year_data.index.duplicated(keep="first")]
            else:
                year_path.parent.mkdir(parents=True, exist_ok=True)
            year_data.sort_index().to_csv(year_path, index=True)

    def get_bars(
        self, symbol: str, timeframe: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get bar data from either the provider or local CSV files.

        The method follows these steps:
        1. Read the self.meta_info file to get the saved start and end dates for
        the given symbol, timeframe, and provider.
        2. Compare the saved dates with the requested dates to determine the
        actual dates for downloading missing data.
        3. If there is missing data to download:
        - Set the end date for downloading to the current date (today).
        - Call the `download_bars` method to fetch the missing data.
        - If the downloaded data is not empty:
            - Save the downloaded data using the `_save_bars` method.
            - Update the self.meta_info file with the actual start and end dates
            from the downloaded data.
        4. Load the data from local CSV files.
        5. If there is both downloaded data and loaded data:
        - Concatenate the downloaded data and loaded data.
        - Sort the resulting DataFrame by index.
        6. If only loaded data exists, use that as the final data.
        7. Filter the data based on the requested start and end dates.
        8. Return the filtered data.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (str): The timeframe of the data.
            start_date (str): The start date of the data (YYYY-MM-DD).
            end_date (str): The end date of the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A DataFrame containing the bar data.
        """

        if start_date > end_date:
            raise ValueError("start_date should be less than or equal to end_date")

        start_date_str = (
            start_date
            if isinstance(start_date, str)
            else start_date.strftime("%Y-%m-%d")
        )
        end_date_str = (
            end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
        )

        # If end_date is weekend set it to Friday
        end_day_wod = pd.to_datetime(end_date_str).dayofweek
        if end_day_wod == 5:
            days_shift = 1
        elif end_day_wod == 6:
            days_shift = 2
        else:
            days_shift = 0
        end_date_str = str(
            pd.to_datetime(end_date_str).date() - pd.Timedelta(days=days_shift)
        )

        # Get the saved start and end dates from self.meta_info
        default_start_date = "2050-01-10"
        default_end_date = "2000-01-10"
        saved_start_date = default_start_date
        saved_end_date = default_end_date

        missing_ranges = []
        
        if (
            self.provider in self.meta_info
            and timeframe in self.meta_info[self.provider]
            and symbol in self.meta_info[self.provider][timeframe]
        ):
            saved_start_date = self.meta_info[self.provider][timeframe][symbol][
                "start_date"
            ]
            saved_end_date = self.meta_info[self.provider][timeframe][symbol][
                "end_date"
            ]

            # Check for missing years in between saved_start_date and saved_end_date
            start_year = pd.to_datetime(saved_start_date).year
            end_year = pd.to_datetime(saved_end_date).year
            data_dir = Path(self.data_dir) / self.provider / timeframe

            for year in range(start_year, end_year + 1):
                year_file = data_dir / str(year) / f"{symbol}.csv"
                if not year_file.exists():
                    year_start = max(saved_start_date, f"{year}-01-01")
                    year_end = min(saved_end_date, f"{year}-12-31")
                    missing_ranges.append((year_start, year_end))

        if start_date_str < saved_start_date:
            missing_ranges.append(
                (start_date_str, min(end_date_str, saved_start_date))
            )
        if end_date_str > saved_end_date:
            missing_ranges.append(
                (max(start_date_str, saved_end_date), end_date_str)
            )
        
        if not self.meta_info.get(self.provider, {}).get(timeframe, {}).get(symbol):
            # If no valid metadata, assume the entire range is missing
            missing_ranges = [(start_date_str, end_date_str)]

        if len(missing_ranges) > 0:
            for drange in missing_ranges:
                d_start, d_end = drange
                data = self._download_bars(symbol, timeframe, d_start, d_end)
                if not data.empty:
                    self._save_bars(data, symbol, timeframe)
                    # Update meta info with the actual dates in the downloaded DataFrame
                    new_start_date = min(d_start, saved_start_date)
                    new_end_date = max(d_end, saved_end_date)

                    # Use setdefault to initialize nested dictionaries if they don't exist
                    self.meta_info.setdefault(self.provider, {}).setdefault(
                        timeframe, {}
                    )[symbol] = {
                        "start_date": new_start_date,
                        "end_date": new_end_date,
                    }

                    with open(self._meta_info_file, "w") as f:
                        json.dump(self.meta_info, f, indent=4)
                else:
                    self.logger.warning(
                        f"No data downloaded for {symbol} {timeframe} from {d_start} to {d_end}"
                    )

        # Load data from local files
        data_dir = Path(self.data_dir) / self.provider / timeframe
        data_yrs = _get_data_by_years(
            data_dir,
            symbol,
            pd.to_datetime(start_date).year,
            pd.to_datetime(end_date).year,
            _today=datetime.now().date(),
        )

        if data_yrs.empty:
            self.logger.warning(f"No data loaded for {symbol} {timeframe}")
            return pd.DataFrame()

        # Filter the data based on the requested start and end dates
        data = data_yrs[
            (data_yrs.index >= pd.to_datetime(start_date))
            & (data_yrs.index <= pd.to_datetime(end_date))
        ]

        return data.sort_index()
    
    def get_min_date(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Look up the earliest date for the given symbol and timeframe.

        This method checks both the meta_info file and the local data files to find the earliest date.
        If there's a mismatch, it updates the meta_info file with the new start date.

        Args:
            symbol (str): The symbol to check.
            timeframe (str): The timeframe to check.

        Returns:
            Optional[str]: The earliest date as a string in 'YYYY-MM-DD' format, or None if no data is found.
        """
        meta_start_date = None
        file_start_date = None

        # Check meta_info file
        try:
            meta_start_date = self.meta_info[self.provider][timeframe][symbol]["start_date"]
        except KeyError:
            self.logger.debug(f"No meta_info found for {symbol} {timeframe}")

        # Check local data files
        data_dir = Path(self.data_dir) / self.provider / timeframe
        data_files = list(data_dir.glob(f"*/{symbol}.csv"))
        file_start_date = None
        if data_files:
            earliest_file = min(data_files, key=lambda f: int(f.parent.name))
            try:
                df = pd.read_csv(earliest_file, index_col="time", parse_dates=True, nrows=1)
                file_start_date = df.index[0].strftime("%Y-%m-%d")
            except Exception as e:
                self.logger.error(f"Error reading file {earliest_file}: {e}")

        if file_start_date:
            earliest_date = file_start_date
        elif meta_start_date:
            # No files, but meta_start_date exists
            self.logger.warning(f"No data files found for {symbol} {timeframe}, but meta_start_date exists. Removing from meta info.")
            self.meta_info.get(self.provider, {}).get(timeframe, {}).pop(symbol, None)
            with open(self._meta_info_file, "w") as f:
                json.dump(self.meta_info, f, indent=4)
            return None
        else:
            self.logger.warning(f"No data found for {symbol} {timeframe}")
            return None

        # Update meta_info if necessary
        if earliest_date != meta_start_date:
            self.meta_info.setdefault(self.provider, {}).setdefault(timeframe, {})[symbol] = {
                "start_date": earliest_date,
                "end_date": self.meta_info.get(self.provider, {}).get(timeframe, {}).get(symbol, {}).get("end_date", "")
            }
            with open(self._meta_info_file, "w") as f:
                json.dump(self.meta_info, f, indent=4)
            self.logger.info(f"Updated meta_info for {symbol} {timeframe} with new start date: {earliest_date}")

        return earliest_date

    def get_max_date(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get the latest date for which data is available for a given symbol and timeframe.
        
        Args:
            symbol (str): The trading symbol.
            timeframe (str): The timeframe of the data.
        
        Returns:
            Optional[str]: The latest date as a string in 'YYYY-MM-DD' format, or None if no data is found.
        """
        meta_end_date = self.meta_info.get(self.provider, {}).get(timeframe, {}).get(symbol, {}).get("end_date")
        
        data_dir = Path(self.data_dir) / self.provider / timeframe
        data_files = list(data_dir.glob(f"*/{symbol}.csv"))
        file_end_date = None
        if data_files:
            latest_file = max(data_files, key=lambda f: int(f.parent.name))
            try:
                df = pd.read_csv(latest_file, index_col="time", parse_dates=True)
                file_end_date = df.index[-1].strftime("%Y-%m-%d")
            except Exception as e:
                self.logger.error(f"Error reading file {latest_file}: {e}")

        if file_end_date:
            latest_date = file_end_date
        elif meta_end_date:
            # No files, but meta_end_date exists
            self.logger.warning(f"No data files found for {symbol} {timeframe}, but meta_end_date exists. Removing from meta info.")
            self.meta_info.get(self.provider, {}).get(timeframe, {}).pop(symbol, None)
            with open(self._meta_info_file, "w") as f:
                json.dump(self.meta_info, f, indent=4)
            return None
        else:
            self.logger.warning(f"No data found for {symbol} {timeframe}")
            return None

        # Update meta_info if necessary
        if latest_date != meta_end_date:
            self.meta_info.setdefault(self.provider, {}).setdefault(timeframe, {})[symbol] = {
                "start_date": self.meta_info.get(self.provider, {}).get(timeframe, {}).get(symbol, {}).get("start_date", ""),
                "end_date": latest_date
            }
            with open(self._meta_info_file, "w") as f:
                json.dump(self.meta_info, f, indent=4)
            self.logger.info(f"Updated meta_info for {symbol} {timeframe} with new end date: {latest_date}")

        return latest_date



