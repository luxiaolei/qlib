# -*- coding: utf-8 -*-
import platform
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from src.barprovider._base import DATA_DIR, BarDataProvider
from src.config import BARPROVIDER_CONFIG

load_dotenv()

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import MetaTrader5 as mt5  # type: ignore

    __all__ = ["MT5BarDataProvider", "RemoteMT5BarDataProvider"]

else:
    __all__ = ["RemoteMT5BarDataProvider"]

LOCAL_MT5_PATH = BARPROVIDER_CONFIG.get("mt5_exe_path")
SERVER_URL = BARPROVIDER_CONFIG.get("server_url")
SERVER_SECRET = BARPROVIDER_CONFIG.get("server_secret")


class MT5BarDataProvider(BarDataProvider):
    """
    Local bar data provider for MetaTrader 5.
    Make sure to set the maximum bars in the MetaTrader 5 terminal settings to a large number.
    """
    _initialized = False
    provider = "LocalMT5"
    
    @classmethod
    def check_availability(cls, mt5_root: str | None = LOCAL_MT5_PATH) -> bool:
        """
        Check if the MetaTrader 5 terminal is available.

        Args:
            mt5_root (str|None): The root directory of MetaTrader 5.
                Default is None, which means the default installation directory.

        Returns:
            bool: True if MetaTrader 5 is available, False otherwise.
        """
        if mt5_root is not None:
            mt5_path = Path(mt5_root)
            if mt5_path.exists() and mt5_path.suffix.lower() == '.exe':
                return True
        return False

    
    def __init__(
        self, data_dir: str | Path = DATA_DIR, mt5_root: str | None = LOCAL_MT5_PATH
    ):
        """
        Initialize the MT5BarDataProvider.

        Args:
            data_dir (str): The directory where the bar data is stored.
            mt5_root (str|None): The root directory of MetaTrader 5.
                Default is None, which means the default installation directory.
        """
        super().__init__(data_dir)
        if not self.check_availability(mt5_root):
            raise RuntimeError("MetaTrader 5 is not available")
        self.mt5_root = mt5_root



    @lru_cache(maxsize=None)
    def _download_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        return_json: bool = False,
    ) -> pd.DataFrame | List[Dict]:
        """
        Download bar data from MetaTrader5.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (str): The timeframe of the data.
            start_date (str): The start date of the data (YYYY-MM-DD).
            end_date (str|None): The end date of the data (YYYY-MM-DD).
                Default is None, which means the current date.
            return_json (bool): Whether to return the data as a JSON object.

        Returns:
            pd.DataFrame||List[Dict]: A DataFrame containing the downloaded bar data.

        """
        if not self._initialized:
            mt5.initialize(self.mt5_root)  # type: ignore
            self._initialized = True

        if platform.system() != "Windows":
            raise RuntimeError("Downloading bars is only supported on Windows.")

        if not mt5.symbol_select(symbol):  # type: ignore
            # re-initialize mt5 and try again
            mt5.initialize(self.mt5_root)  # type: ignore
            if not mt5.symbol_select(symbol):  # type: ignore
                raise ValueError(f"Failed to select symbol: {symbol}")

        # create 'datetime' objects in UTC time zone
        utc_from = pd.to_datetime(start_date, utc=True)
        if end_date is None:
            utc_to = pd.Timestamp.now(tz="UTC")
        else:
            utc_to = pd.to_datetime(end_date, utc=True)

        # Check if the symbol is available in the Market Watch
        if not mt5.symbol_info(symbol).visible:  # type: ignore
            self.logger.warning(f"{symbol} is not visible in the Market Watch")
            return pd.DataFrame()

        rates = mt5.copy_rates_range(  # type: ignore
            symbol, self._get_mt5_timeframe(timeframe), utc_from, utc_to
        )
        rates_frame = pd.DataFrame(rates)
        if rates_frame.empty:
            self.logger.warning(f"{symbol} {timeframe} is empty")
            return rates_frame

        rates_frame = rates_frame[["time", "open", "high", "low", "close"]]
        if return_json:
            return rates_frame.to_dict(orient="records")

        rates_frame["time"] = pd.to_datetime(rates_frame["time"], unit="s")
        rates_frame.set_index("time", inplace=True)
        return rates_frame

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        return getattr(mt5, f"TIMEFRAME_{timeframe}")


class RemoteMT5BarDataProvider(BarDataProvider):
    """
    Remote bar data provider for MetaTrader 5 using a FastAPI endpoint.
    """

    @classmethod
    def check_availability(cls, api_url: str = SERVER_URL) -> bool:
        """
        Check if the FastAPI endpoint is available.

        Args:
            api_url (str): The URL of the FastAPI endpoint.

        Returns:
            bool: True if the FastAPI endpoint is available, False otherwise.
        """
        try:
            response = requests.get(
                f"{api_url}/login", auth=HTTPBasicAuth("admin", SERVER_SECRET)
            )
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.RequestException as e:
            return False

    provider = "RemoteMT5"
    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        api_url: str = SERVER_URL,
        username: str = "admin",
        password: str = SERVER_SECRET,
    ):
        """
        Initialize the RemoteMT5BarDataProvider.

        Args:
            data_dir (str): The directory where the bar data is stored.
            api_url (str): The URL of the FastAPI endpoint.
            username (str): The username for basic authentication.
            password (str): The password for basic authentication.
        """
        super().__init__(data_dir)
        self.api_url = api_url
        self.get_bars_endpoint = f"{api_url}/bars"
        self.auth = HTTPBasicAuth(username, password)

    @lru_cache(maxsize=None)
    def _download_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str | None = None,
        retried_times=0,
    ) -> pd.DataFrame | None:
        """
        Download bar data from the remote FastAPI endpoint.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (str): The timeframe of the data.
            start_date (str): The start date of the data (YYYY-MM-DD).
            end_date (str|None): The end date of the data (YYYY-MM-DD).
                Default is None, which means the current date.
            retried_times (int): The number of times the request has been retried.
                We set max retries to 3.

        Returns:
            pd.DataFrame|List[Dict]: A DataFrame containing the downloaded bar data.
        """
        end_date = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        MAX_RETRIES = 3

        def date_range_chunks(start_date, end_date):
            """Generate yearly chunks within a given date range."""
            current_start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            while current_start < end:
                current_end = min(
                    current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end
                )
                yield (
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                )
                current_start = current_end + pd.DateOffset(days=1)

        def fetch_data_chunk(start_date, end_date, retried_times):
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
            }
            response = requests.get(
                self.get_bars_endpoint, params=params, auth=self.auth
            )

            if response.status_code == 200:
                data = response.json()
                if not data:
                    self.logger.warning(
                        f"No data available for the specified parameters: {params}"
                    )
                    return pd.DataFrame()
                rates_frame = pd.DataFrame(data)
                rates_frame["time"] = pd.to_datetime(rates_frame["time"], unit="s")
                rates_frame.set_index("time", inplace=True)
                return rates_frame
            elif response.status_code == 404:
                self.logger.warning(
                    f"No data available for the specified parameters: {params}"
                )
                return pd.DataFrame()
            elif response.status_code == 502:
                # Bad Gateway
                # Sleep for 1 second and retry
                time.sleep(1)
                retried_times += 1
                if retried_times < MAX_RETRIES:
                    self.logger.warning(
                        f"Retrying request for {params} - Retry {retried_times}"
                    )
                    return fetch_data_chunk(start_date, end_date, retried_times)
                else:
                    self.logger.error(
                        f"Failed to download bars: {response.status_code} - {response.text}"
                    )
                    return pd.DataFrame()
            else:
                self.logger.error(
                    f"Failed to download bars: {response.status_code} - {response.text}"
                )
                return pd.DataFrame()

        if timeframe in ["M1", "M5"]:
            chunks = date_range_chunks(start_date, end_date)
            bars_list = [
                fetch_data_chunk(chunk_start, chunk_end, retried_times)
                for chunk_start, chunk_end in chunks
            ]
            bars_data = pd.concat(bars_list)
        else:
            bars_data = fetch_data_chunk(start_date, end_date, retried_times)

        return bars_data
