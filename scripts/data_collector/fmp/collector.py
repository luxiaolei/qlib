"""Base classes for FMP data collection."""

import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from loguru import logger

from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun
from scripts.dump_bin import DumpDataAll, DumpDataFix, DumpDataUpdate

"""Base FMP data collector.

This module provides a base collector class for FMP data collection.
It handles common functionality for both daily and 5-minute data collection.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from qlib.utils import code_to_fname
from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun
from scripts.data_collector.fmp.fmp_api import FMPClient
from scripts.dump_bin import DumpDataUpdate

console = Console()

class BaseFMPCollector(BaseCollector):
    """Base collector for FMP data.
    
    This class provides common functionality for collecting data from FMP API.
    It should be subclassed for specific intervals (daily, 5min).
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        max_workers: int = 4,
        max_collector_count: int = 2,
        delay: float = 0.1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        redis_url: str = "redis://localhost:6379",
        redis_password: Optional[str] = None,
        instruments: Optional[List[str]] = None,
        skip_existing: bool = False,
    ):
        """Initialize FMP collector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save collected data
        start : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end : Optional[str]
            End date for data collection (YYYY-MM-DD)
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
        redis_url : str
            Redis connection URL for rate limiting
        redis_password : Optional[str]
            Redis password for authentication
        instruments : Optional[List[str]]
            List of specific instruments to collect. If None, gets from API
        skip_existing : bool
            If True, skip downloading data for date ranges that already exist
        """
        self.skip_existing = skip_existing
        self._instruments = instruments
        self.fmp_client = FMPClient(
            redis_url=redis_url,
            redis_password=redis_password
        )
        
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_existing_data_range(self, symbol: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        """Get the date range of existing data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol to check
            
        Returns
        -------
        Optional[tuple[pd.Timestamp, pd.Timestamp]]
            (start_date, end_date) if data exists, None otherwise
        """
        symbol = self.normalize_symbol(symbol)
        symbol = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol}.csv")
        
        if not instrument_path.exists():
            return None
            
        try:
            df = pd.read_csv(instrument_path)
            if df.empty:
                return None
                
            df["date"] = pd.to_datetime(df["date"])
            return df["date"].min(), df["date"].max()
        except Exception as e:
            logger.warning(f"Error reading existing data for {symbol}: {e}")
            return None

    def get_missing_ranges(
        self, 
        symbol: str, 
        target_start: pd.Timestamp,
        target_end: pd.Timestamp
    ) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
        """Get missing date ranges for a symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol to check
        target_start : pd.Timestamp
            Desired start date
        target_end : pd.Timestamp
            Desired end date
            
        Returns
        -------
        List[tuple[pd.Timestamp, pd.Timestamp]]
            List of (start, end) ranges that need to be downloaded
        """
        existing_range = self.get_existing_data_range(symbol)
        if not existing_range:
            return [(target_start, target_end)]
            
        existing_start, existing_end = existing_range
        missing_ranges = []
        
        # Check if we need data before existing data
        if target_start < existing_start:
            missing_ranges.append((target_start, existing_start - pd.Timedelta(days=1)))
            
        # Check if we need data after existing data
        if target_end > existing_end:
            missing_ranges.append((existing_end + pd.Timedelta(days=1), target_end))
            
        return missing_ranges

    async def _async_get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """Get data from FMP API asynchronously."""
        if self.skip_existing:
            # Get only missing date ranges
            missing_ranges = self.get_missing_ranges(symbol, start_datetime, end_datetime)
            if not missing_ranges:
                logger.info(f"Skipping {symbol}: data already exists for requested range")
                return pd.DataFrame()
                
            # Collect data for missing ranges
            all_data = []
            async with self.fmp_client as client:
                for range_start, range_end in missing_ranges:
                    logger.info(
                        f"Collecting {symbol} from {range_start.date()} to {range_end.date()}"
                    )
                    df = await client.get_historical_data(
                        symbol=symbol,
                        start_date=range_start,
                        end_date=range_end,
                        interval=interval,
                        delay=self.delay
                    )
                    if not df.empty:
                        all_data.append(df)
                        
            if not all_data:
                return pd.DataFrame()
            return pd.concat(all_data, ignore_index=True)
        else:
            # Original behavior
            async with self.fmp_client as client:
                return await client.get_historical_data(
                    symbol=symbol,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    interval=interval,
                    delay=self.delay
                )

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """Get data from FMP API.
        
        This method runs the async get_data in an event loop.
        """
        return asyncio.run(
            self._async_get_data(
                symbol=symbol,
                interval=interval,
                start_datetime=start_datetime,
                end_datetime=end_datetime
            )
        )

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize stock symbol format."""
        return symbol.strip().upper()

    async def _async_get_instrument_list(self) -> List[str]:
        """Get list of instruments from FMP API."""
        if hasattr(self, '_instruments') and self._instruments is not None:
            logger.info(f"Using provided instruments: {len(self._instruments)} symbols")
            return [self.normalize_symbol(s) for s in self._instruments]
            
        async with self.fmp_client as client:
            symbols = await client.get_exchange_symbols()
            logger.info(f"Got {len(symbols)} symbols from FMP API")
            return symbols

    def get_instrument_list(self) -> List[str]:
        """Get list of instruments.
        
        Returns either provided instruments or fetches from FMP API.
        """
        return asyncio.run(self._async_get_instrument_list())

    async def _async_collector(self, instrument_list: List[str], batch_size: int = 50) -> List[str]:
        """Collect data for instruments in parallel batches.
        
        Parameters
        ----------
        instrument_list : List[str]
            List of instruments to collect
        batch_size : int
            Size of parallel batches
            
        Returns
        -------
        List[str]
            List of instruments that failed to collect
        """
        error_symbols = []
        
        async def process_symbol(symbol: str) -> str:
            """Process a single symbol."""
            try:
                df = await self._async_get_data(
                    symbol=symbol,
                    interval=self.interval,
                    start_datetime=self.start_datetime,
                    end_datetime=self.end_datetime
                )
                
                result = self.NORMAL_FLAG
                if self.check_data_length > 0:
                    result = self.cache_small_data(symbol, df)
                if result == self.NORMAL_FLAG:
                    self.save_instrument(symbol, df)
                return result
            except Exception as e:
                logger.error(f"Error collecting {symbol}: {e}")
                return self.CACHE_FLAG

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Collecting {self.interval} data (batch size: {batch_size})...", 
                total=len(instrument_list)
            )
            
            async with self.fmp_client as client:  # Reuse client across batches
                for i in range(0, len(instrument_list), batch_size):
                    batch = instrument_list[i:i + batch_size]
                    
                    # Process batch in parallel
                    results = await asyncio.gather(
                        *[process_symbol(symbol) for symbol in batch],
                        return_exceptions=True
                    )
                    
                    # Process results
                    for symbol, result in zip(batch, results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to collect {symbol}: {result}")
                            error_symbols.append(symbol)
                        elif result != self.NORMAL_FLAG:
                            error_symbols.append(symbol)
                        progress.advance(task)

        # Add any cached symbols to error list
        error_symbols.extend(self.mini_symbol_map.keys())
        return sorted(set(error_symbols))

    def _collector(self, instrument_list: List[str]) -> List[str]:
        """Override to use async collection."""
        return asyncio.run(self._async_collector(instrument_list))

    def collector_data(self):
        """Override to use async collection with retries."""
        logger.info(f"Start collecting {self.interval} data from FMP...")
        instrument_list = self.instrument_list
        
        for attempt in range(self.max_collector_count):
            if not instrument_list:
                break
            logger.info(f"Collection attempt {attempt + 1}/{self.max_collector_count}")
            instrument_list = self._collector(instrument_list)
            
        # Handle any remaining cached data
        for symbol, df_list in self.mini_symbol_map.items():
            if df_list:
                df = pd.concat(df_list, sort=False)
                if not df.empty:
                    self.save_instrument(
                        symbol, 
                        df.drop_duplicates(["date"]).sort_values(["date"])
                    )
                    
        if self.mini_symbol_map:
            logger.warning(
                f"Symbols with insufficient data (< {self.check_data_length}): "
                f"{list(self.mini_symbol_map.keys())}"
            )
            
        total = len(self.instrument_list)
        failed = len(set(instrument_list))
        logger.info(
            f"Collection finished. Total: {total}, "
            f"Succeeded: {total - failed}, Failed: {failed}"
        )

class BaseFMPNormalize(BaseNormalize):
    """Base class for normalizing FMP data to Qlib format."""
    
    def __init__(
        self,
        instruments_dir: Union[str, Path],
        qlib_dir: Union[str, Path],
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
    ):
        """Initialize FMP normalizer.
        
        Parameters
        ----------
        instruments_dir : Union[str, Path]
            Directory containing raw instrument data
        qlib_dir : Union[str, Path]
            Directory to store normalized Qlib data
        freq : str
            Data frequency (day/min5)
        max_workers : int
            Maximum number of parallel workers
        date_field_name : str
            Name of date field in raw data
        symbol_field_name : str
            Name of symbol field in raw data
        """
        super().__init__(
            instruments_dir=instruments_dir,
            qlib_dir=qlib_dir,
            freq=freq,
            max_workers=max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
        )
        self.instruments_dir = Path(instruments_dir).expanduser().resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.freq = freq
        self.max_workers = max_workers
        self.date_field_name = date_field_name
        self.symbol_field_name = symbol_field_name
        
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        raise NotImplementedError("")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get benchmark calendar"""
        raise NotImplementedError("")
    

    def _get_symbol(self, file_path: Path) -> str:
        """Extract symbol from file path."""
        return file_path.stem.upper()

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into Qlib format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data from FMP
            
        Returns
        -------
        pd.DataFrame
            Processed data in Qlib format
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Map column names to Qlib format
        rename_dict = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adjClose": "Adj_Close",
        }
        df = df.rename(columns=rename_dict)
        
        # Calculate adjusted prices
        for field in ["Open", "High", "Low"]:
            df[f"Adj_{field}"] = df[field] * df["Adj_Close"] / df["Close"]
            
        return df



class BaseFMPRunner(BaseRun):
    """Base class for running FMP data collection and normalization."""
    
    def __init__(self, max_workers: int = 4, interval: str = "1d"):
        """Initialize FMP runner.
        
        Parameters
        ----------
        max_workers : int
            Maximum number of parallel workers
        interval : str
            Data interval (1d/5min)
        """
        super().__init__()
        self.max_workers = max_workers
        self.interval = interval
        self.fmp_client = FMPClient()
        
    async def _get_constituent_data(self, index_name: str) -> pd.DataFrame:
        """Get constituent data for an index from FMP API.
        
        Parameters
        ----------
        index_name : str
            Name of the index (e.g., 'SP500', 'NASDAQ100')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: symbol, start_date, end_date
        """
        async with self.fmp_client as client:
            constituents = await client.get_index_constituents(index_name)
            if constituents is None or constituents.empty:
                logger.warning(f"No constituent data found for {index_name}")
                return pd.DataFrame()
                
            # Get historical constituents to determine entry dates
            historical = await client.get_historical_constituents(index_name)
            if historical is None or historical.empty:
                logger.warning(f"No historical constituent data found for {index_name}")
                historical = pd.DataFrame()
                
            # Process current constituents
            current_date = pd.Timestamp(datetime.now().date())
            result = []
            
            for symbol in constituents["symbol"]:
                # Find earliest date from historical data
                if not historical.empty:
                    symbol_hist = historical[historical["symbol"] == symbol]
                    if not symbol_hist.empty:
                        start_date = symbol_hist["date"].min()
                    else:
                        start_date = current_date
                else:
                    start_date = current_date
                    
                result.append({
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": pd.Timestamp("2099-12-31")
                })
                
            return pd.DataFrame(result)
            
    def dump_constituent_data(
        self,
        qlib_dir: Union[str, Path],
        index_name: str = "SP500",
    ):
        """Dump constituent data to Qlib format.
        
        Parameters
        ----------
        qlib_dir : Union[str, Path]
            Qlib data directory
        index_name : str
            Name of the index (e.g., 'SP500', 'NASDAQ100')
        """
        qlib_dir = Path(qlib_dir).expanduser().resolve()
        instruments_dir = qlib_dir.joinpath("instruments")
        instruments_dir.mkdir(parents=True, exist_ok=True)
        
        # Get constituent data
        df = asyncio.run(self._get_constituent_data(index_name))
        if df.empty:
            logger.error(f"Failed to get constituent data for {index_name}")
            return
            
        # Format data for writing
        output_lines = []
        for _, row in df.sort_values("symbol").iterrows():
            start_date = row["start_date"].strftime("%Y-%m-%d")
            end_date = row["end_date"].strftime("%Y-%m-%d")
            output_lines.append(f"{row['symbol']}\t{start_date}\t{end_date}")
            
        # Write to file
        output_file = instruments_dir.joinpath(f"{index_name.lower()}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))
            
        logger.info(f"Constituent data written to {output_file}")
        
    def download_and_normalize(
        self,
        qlib_dir: Union[str, Path],
        index_name: str = "SP500",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Download and normalize data for index constituents.
        
        Parameters
        ----------
        qlib_dir : Union[str, Path]
            Qlib data directory
        index_name : str
            Name of the index (e.g., 'SP500', 'NASDAQ100')
        start_date : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end_date : Optional[str]
            End date for data collection (YYYY-MM-DD)
        """
        qlib_dir = Path(qlib_dir).expanduser().resolve()
        cache_dir = qlib_dir.joinpath("stock_data", self.interval, "raw")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # First dump constituent data
        self.dump_constituent_data(qlib_dir, index_name)
        
        # Then download data for constituents
        self.download_data(
            save_dir=cache_dir,
            start=start_date,
            end=end_date,
        )
        
        # Finally normalize the data
        self.normalize_data(
            source_dir=cache_dir,
            target_dir=qlib_dir,
        )
        
    def download_data(
        self,
        save_dir: Union[str, Path],
        qlib_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        delay: float = 0.1,
        check_data_length: Optional[int] = None,
        limit_nums: Optional[int] = None,
        instruments: Optional[List[str]] = None,
        skip_existing: bool = False,
        dump_bin: bool = True,
        dump_all: bool = False,
        dump_update: bool = True,
        **kwargs
    ):
        """Download data and optionally dump to binary format.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save raw data
        qlib_dir : Union[str, Path]
            Directory for Qlib format data
        start : Optional[str]
            Start date for data collection
        end : Optional[str]
            End date for data collection
        delay : float
            Delay between API calls
        check_data_length : Optional[int]
            Check if data length matches this value
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
        kwargs : dict
            Additional arguments passed to collector
        """
        # Get collector class
        collector_class = getattr(sys.modules[__name__], self.collector_class_name)
        
        # Initialize collector
        collector = collector_class(
            save_dir=save_dir,
            start=start,
            end=end,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            instruments=instruments,
            skip_existing=skip_existing,
            **kwargs
        )
        
        # Run collection
        collector.run()
        
        if dump_bin:
            logger.info("Dumping data to binary format...")
            # Get frequency from interval
            freq = self.interval.replace("min", "t")  # Convert "5min" to "5t" for Qlib format
            
            # Determine fields based on collector class
            if hasattr(collector, "normalize_class"):
                normalize_class = collector.normalize_class
                include_fields = ",".join(normalize_class.COLUMNS)
            else:
                include_fields = "open,close,high,low,volume,factor"
            
            self._dump_bin(
                csv_path=save_dir,
                qlib_dir=qlib_dir,
                freq=freq,
                date_field_name="date",
                symbol_field_name="symbol",
                include_fields=include_fields,
                limit_nums=limit_nums,
                dump_all=dump_all,
                dump_update=dump_update,
            )
        
    def normalize_data(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        max_workers: int = 16,
        date_field_name: str = "date",
    ):
        """Normalize data using the appropriate normalizer."""
        normalize_cls = self._get_normalize_class()
        normalizer = normalize_cls(
            instruments_dir=source_dir,
            qlib_dir=target_dir,
            freq="day" if self.interval == "1d" else "min5",
            max_workers=max_workers,
            date_field_name=date_field_name,
        )
        normalizer.normalize()
        
    def _get_collector_class(self):
        """Get collector class based on interval."""
        import importlib
        module = importlib.import_module("scripts.data_collector.fmp.us_daily" if self.interval == "1d" else "scripts.data_collector.fmp.us_m5")
        return getattr(module, self.collector_class_name)
        
    def _get_normalize_class(self):
        """Get normalizer class based on interval."""
        import importlib
        module = importlib.import_module("scripts.data_collector.fmp.us_daily" if self.interval == "1d" else "scripts.data_collector.fmp.us_m5")
        return getattr(module, self.normalize_class_name)
        
    @property
    def collector_class_name(self) -> str:
        """Name of collector class to use."""
        raise NotImplementedError
        
    @property
    def normalize_class_name(self) -> str:
        """Name of normalizer class to use."""
        raise NotImplementedError
        
    def _dump_bin(
        self,
        csv_path: Union[str, Path],
        qlib_dir: Union[str, Path],
        freq: str,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "open,close,high,low,volume,factor",
        limit_nums: Optional[int] = None,
        dump_all: bool = False,
        dump_update: bool = True,
    ):
        """Dump data to binary format.
        
        Parameters
        ----------
        csv_path : Union[str, Path]
            Path to CSV files directory
        qlib_dir : Union[str, Path]
            Path to Qlib data directory
        freq : str
            Data frequency (e.g., day, 5min)
        date_field_name : str
            Name of date field
        symbol_field_name : str
            Name of symbol field
        exclude_fields : str
            Fields to exclude, comma-separated
        include_fields : str
            Fields to include, comma-separated
        limit_nums : Optional[int]
            Limit number of symbols to process
        dump_all : bool
            Whether to dump all data (overwrite existing)
        dump_update : bool
            Whether to update existing data
        """
        if dump_all:
            dumper = DumpDataAll
        elif dump_update:
            dumper = DumpDataUpdate
        else:
            dumper = DumpDataFix
            
        dump_obj = dumper(
            csv_path=str(csv_path),
            qlib_dir=str(qlib_dir),
            freq=freq,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            exclude_fields=exclude_fields,
            include_fields=include_fields,
            limit_nums=limit_nums,
        )
        dump_obj.dump()

    def run_index(
        self,
        index_name: str,
        qlib_dir: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dump_bin: bool = True,
        dump_all: bool = False,
        dump_update: bool = True,
        **kwargs
    ):
        """Run collection for an index.
        
        Parameters
        ----------
        index_name : str
            Name of the index to process
        qlib_dir : Union[str, Path]
            Directory for Qlib format data
        start_date : Optional[str]
            Start date for data collection
        end_date : Optional[str]
            End date for data collection
        dump_bin : bool
            Whether to dump to binary format
        dump_all : bool
            Whether to dump all data (overwrite existing)
        dump_update : bool
            Whether to update existing data
        kwargs : dict
            Additional arguments passed to collector
        """
        # Create cache directory
        cache_dir = Path(qlib_dir).expanduser() / f"cache_{index_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download data for constituents
        self.download_data(
            save_dir=cache_dir,
            qlib_dir=qlib_dir,
            start=start_date,
            end=end_date,
            dump_bin=dump_bin,
            dump_all=dump_all,
            dump_update=dump_update,
            **kwargs
        )
        
        # Finally normalize the data
        self.normalize_data(
            qlib_dir=qlib_dir,
            instruments=None,  # Process all instruments
            **kwargs
        )
