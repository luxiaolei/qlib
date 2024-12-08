# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import abc
import datetime
import importlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Type, Union

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from qlib.utils import code_to_fname


class BaseCollector(abc.ABC):
    CACHE_FLAG = "CACHED"
    NORMAL_FLAG = "NORMAL"

    DEFAULT_START_DATETIME_1D = pd.Timestamp("2000-01-01")
    DEFAULT_START_DATETIME_1MIN = pd.Timestamp(datetime.datetime.now() - pd.Timedelta(days=5 * 6 - 1)).date()
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1)).date()
    DEFAULT_END_DATETIME_1MIN = DEFAULT_END_DATETIME_1D

    INTERVAL_1min = "1min"
    INTERVAL_1d = "1d"

    def __init__(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_workers: int = 1,
        max_collector_count: int = 2,
        delay: float = 0,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
    ):
        """Initialize collector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Instrument save directory
        start : Optional[str]
            Start datetime
        end : Optional[str]
            End datetime
        interval : str
            Data interval ('1d' or '5min')
        max_workers : int
            Maximum number of parallel workers
        max_collector_count : int
            Maximum collection attempts per symbol
        delay : float
            Delay between API calls
        check_data_length : Optional[int]
            Minimum required data length
        limit_nums : Optional[int]
            Limit number of symbols to collect
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay
        self.max_workers = max_workers
        self.max_collector_count = max_collector_count
        self.mini_symbol_map = {}
        self.interval = interval
        self.check_data_length = max(int(check_data_length or 0), 0)

        self.start_datetime = self.normalize_start_datetime(start)
        self.end_datetime = self.normalize_end_datetime(end)

        self.instrument_list = sorted(set(self.get_instrument_list()))

        if limit_nums is not None:
            try:
                self.instrument_list = self.instrument_list[:int(limit_nums)]
            except Exception as e:
                logger.warning(f"Cannot use limit_nums={limit_nums}, the parameter will be ignored")

    def normalize_start_datetime(
        self, 
        start_datetime: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.Timestamp:
        """Normalize start datetime."""
        return (
            pd.Timestamp(str(start_datetime))
            if start_datetime
            else getattr(self, f"DEFAULT_START_DATETIME_{self.interval.upper()}")
        )

    def normalize_end_datetime(
        self, 
        end_datetime: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.Timestamp:
        """Normalize end datetime."""
        return (
            pd.Timestamp(str(end_datetime))
            if end_datetime
            else getattr(self, f"DEFAULT_END_DATETIME_{self.interval.upper()}")
        )

    @abc.abstractmethod
    def get_instrument_list(self):
        raise NotImplementedError("rewrite get_instrument_list")

    @abc.abstractmethod
    def normalize_symbol(self, symbol: str):
        """normalize symbol"""
        raise NotImplementedError("rewrite normalize_symbol")

    @abc.abstractmethod
    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """get data with symbol

        Parameters
        ----------
        symbol: str
        interval: str
            value from [1min, 1d]
        start_datetime: pd.Timestamp
        end_datetime: pd.Timestamp

        Returns
        ---------
            pd.DataFrame, "symbol" and "date"in pd.columns

        """
        raise NotImplementedError("rewrite get_timezone")

    def sleep(self):
        time.sleep(self.delay)

    def _simple_collector(self, symbol: str):
        """

        Parameters
        ----------
        symbol: str

        """
        self.sleep()
        df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
        _result = self.NORMAL_FLAG
        if self.check_data_length > 0:
            _result = self.cache_small_data(symbol, df)
        if _result == self.NORMAL_FLAG:
            self.save_instrument(symbol, df)
        return _result

    def save_instrument(self, symbol, df: pd.DataFrame):
        """save instrument data to file

        Parameters
        ----------
        symbol: str
            instrument code
        df : pd.DataFrame
            df.columns must contain "symbol" and "datetime"
        """
        if df is None or df.empty:
            logger.warning(f"{symbol} is empty")
            return

        symbol = self.normalize_symbol(symbol)
        symbol = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
        df["symbol"] = symbol
        if instrument_path.exists():
            _old_df = pd.read_csv(instrument_path)
            df = pd.concat([_old_df, df], sort=False)
        df.to_csv(instrument_path, index=False)

    def cache_small_data(self, symbol, df):
        if len(df) < self.check_data_length:
            logger.warning(f"the number of trading days of {symbol} is less than {self.check_data_length}!")
            _temp = self.mini_symbol_map.setdefault(symbol, [])
            _temp.append(df.copy())
            return self.CACHE_FLAG
        else:
            if symbol in self.mini_symbol_map:
                self.mini_symbol_map.pop(symbol)
            return self.NORMAL_FLAG

    def _collector(self, instrument_list):
        error_symbol = []
        res = Parallel(n_jobs=self.max_workers)(
            delayed(self._simple_collector)(_inst) for _inst in tqdm(instrument_list)
        )
        for _symbol, _result in zip(instrument_list, res):
            if _result != self.NORMAL_FLAG:
                error_symbol.append(_symbol)
        print(error_symbol)
        logger.info(f"error symbol nums: {len(error_symbol)}")
        logger.info(f"current get symbol nums: {len(instrument_list)}")
        error_symbol.extend(self.mini_symbol_map.keys())
        return sorted(set(error_symbol))

    def collector_data(self):
        """collector data"""
        logger.info("start collector data......")
        instrument_list = self.instrument_list
        for i in range(self.max_collector_count):
            if not instrument_list:
                break
            logger.info(f"getting data: {i+1}")
            instrument_list = self._collector(instrument_list)
            logger.info(f"{i+1} finish.")
        for _symbol, _df_list in self.mini_symbol_map.items():
            _df = pd.concat(_df_list, sort=False)
            if not _df.empty:
                self.save_instrument(_symbol, _df.drop_duplicates(["date"]).sort_values(["date"]))
        if self.mini_symbol_map:
            logger.warning(f"less than {self.check_data_length} instrument list: {list(self.mini_symbol_map.keys())}")
        logger.info(f"total {len(self.instrument_list)}, error: {len(set(instrument_list))}")


class BaseNormalize(abc.ABC):
    def __init__(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        """

        Parameters
        ----------
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        self._date_field_name = date_field_name
        self._symbol_field_name = symbol_field_name
        self.kwargs = kwargs
        self._calendar_list = self._get_calendar_list()

    @abc.abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        raise NotImplementedError("")

    @abc.abstractmethod
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get benchmark calendar"""
        raise NotImplementedError("")


class Normalize:
    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        normalize_class: Type[BaseNormalize],
        max_workers: int = 16,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs,
    ):
        """Initialize normalizer."""
        if not (source_dir and target_dir):
            raise ValueError("source_dir and target_dir cannot be None")
        self._source_dir = Path(source_dir).expanduser()
        self._target_dir = Path(target_dir).expanduser()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._date_field_name = date_field_name
        self._symbol_field_name = symbol_field_name
        self._end_date = kwargs.get("end_date", None)
        self._max_workers = max_workers

        self._normalize_obj = normalize_class(
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            **kwargs
        )

    def _executor(self, file_path: Path):
        """Execute normalization for a file."""
        file_path = Path(file_path)

        # Get pandas NA values safely
        import pandas._libs.parsers as parsers
        default_na = list(parsers.STR_NA_VALUES)
        symbol_na = list(default_na)
        symbol_na.remove("NA")
        
        columns = pd.read_csv(file_path, nrows=0).columns
        df = pd.read_csv(
            file_path,
            dtype={self._symbol_field_name: str},
            keep_default_na=False,
            na_values={
                col: symbol_na if col == self._symbol_field_name else default_na 
                for col in columns
            },
        )

        df = self._normalize_obj.normalize(df)
        if df is not None and not df.empty:
            if self._end_date is not None:
                _mask = pd.to_datetime(df[self._date_field_name]) <= pd.Timestamp(self._end_date)
                df = df[_mask]
            df.to_csv(self._target_dir.joinpath(file_path.name), index=False)

    def normalize(self):
        logger.info("normalize data......")

        with ProcessPoolExecutor(max_workers=self._max_workers) as worker:
            file_list = list(self._source_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                for _ in worker.map(self._executor, file_list):
                    p_bar.update()


class BaseRun(abc.ABC):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        """
        if source_dir is None:
            source_dir = Path(self.default_base_dir).joinpath("source")
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)

        if normalize_dir is None:
            normalize_dir = Path(self.default_base_dir).joinpath("normalize")
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.normalize_dir.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_cur_module"):
            self._cur_module = importlib.import_module("collector")
        self.max_workers = max_workers
        self.interval = interval

    @property
    @abc.abstractmethod
    def collector_class_name(self):
        raise NotImplementedError("rewrite collector_class_name")

    @property
    @abc.abstractmethod
    def normalize_class_name(self):
        raise NotImplementedError("rewrite normalize_class_name")

    @property
    @abc.abstractmethod
    def default_base_dir(self) -> Union[Path, str]:
        """Get default base directory."""
        raise NotImplementedError("rewrite default_base_dir")

    def download_data(
        self,
        max_collector_count: int = 2,
        delay: float = 0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        **kwargs,
    ):
        """Download data."""
        _class = getattr(self._cur_module, self.collector_class_name)
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            **kwargs,
        ).collector_data()

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/instrument_data/source --normalize_dir ~/.qlib/instrument_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, self.normalize_class_name)
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            **kwargs,
        )
        yc.normalize()
