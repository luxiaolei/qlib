"""Base collector for Financial Modeling Prep data.

This module provides base classes for collecting stock data from FMP API.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import contextlib
from datetime import datetime

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskID, TaskProgressColumn, TimeElapsedColumn

from qlib.utils import code_to_fname
from scripts.data_collector.fmp_new.api import FMPClient

console = Console()

class BaseFMPCollector:
    """Base collector for FMP data.
    
    Provides common functionality for all FMP data collectors.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        start: str,
        end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        max_workers: int = 5,
        redis_url: Optional[str] = None,
        incremental: bool = False,
        overwrite: bool = False,
        batch_size: int = 5,
        **kwargs
    ):
        """Initialize the FMP collector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save data to
        start : str
            Start date (YYYY-MM-DD)
        end : Optional[str], optional
            End date (YYYY-MM-DD), defaults to today
        symbols : Optional[List[str]], optional
            List of symbols to collect, defaults to None
        api_key : Optional[str], optional
            FMP API key, defaults to None
        max_workers : int, optional
            Maximum number of workers, defaults to 5
        redis_url : Optional[str], optional
            Redis URL for rate limiting, defaults to None
        incremental : bool, optional
            Whether to collect only new data, defaults to False
        overwrite : bool, optional
            Whether to overwrite existing files, defaults to False
        batch_size : int, optional
            Number of symbols to process in parallel, defaults to 5
        **kwargs : dict
            Additional keyword arguments
        """
        # Save directory setup
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Date handling
        self.start_date = start
        self.end_date = end or datetime.now().strftime("%Y-%m-%d")
        
        # Convert to timestamps for consistent comparison
        self.start_time = pd.Timestamp(self.start_date)
        self.end_time = pd.Timestamp(self.end_date)
        
        # API client setup
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Either provide it as a parameter "
                "or set the FMP_API_KEY environment variable."
            )
            
        # Create API client
        self.api_client = FMPClient(
            api_key=self.api_key,
            redis_url=redis_url
        )
        
        # Collection settings
        self.incremental = incremental
        self.overwrite = overwrite
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Default interval (subclasses may override)
        self.interval = "1d"
        
        # Symbol handling
        if symbols:
            if isinstance(symbols, str):
                # Handle comma-separated symbols
                self.symbols = [s.strip() for s in symbols.split(",")]
            else:
                self.symbols = list(symbols)
        else:
            self.symbols = None
        
        # Statistics
        self.success_count = 0
        self.error_count = 0
        self.skip_count = 0
        
    async def get_symbols(self) -> List[str]:
        """Get symbols to collect.
        
        If symbols were provided in the constructor, they will be returned.
        Otherwise, all symbols from the API will be fetched.
        
        Returns
        -------
        List[str]
            List of symbols
        """
        if self.symbols:
            return self.symbols
        
        async with self.api_client as client:
            logger.info("Fetching symbols from API...")
            symbols = await client.get_all_symbols()
            
            # Filter out invalid symbols
            valid_symbols = [s for s in symbols if not any(c in s for c in "^*/\\")]
            
            self.symbols = valid_symbols
            return valid_symbols
    
    async def get_existing_data_range(
        self, 
        symbol: str
    ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get date range of existing data for a symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
            
        Returns
        -------
        Optional[Tuple[pd.Timestamp, pd.Timestamp]]
            (start_date, end_date) of existing data, or None if no data
        """
        symbol_file = self.get_symbol_file_path(symbol)
        
        if not symbol_file.exists():
            return None
            
        try:
            df = pd.read_csv(symbol_file)
            if df.empty:
                return None
                
            # Ensure date column is in datetime format
            df["date"] = pd.to_datetime(df["date"])
            
            # Return min and max dates
            return df["date"].min(), df["date"].max()
            
        except Exception as e:
            logger.warning(f"Error reading existing data for {symbol}: {e}")
            return None
    
    def get_symbol_file_path(self, symbol: str) -> Path:
        """Get file path for a symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
            
        Returns
        -------
        Path
            Path to the symbol's data file
        """
        # Normalize symbol name for filesystem
        symbol_fname = code_to_fname(symbol)
        return self.save_dir / f"{symbol_fname}.csv"
    
    async def get_missing_ranges(
        self,
        symbol: str,
        target_start: pd.Timestamp,
        target_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get missing date ranges that need to be downloaded.
        
        This is useful for incremental updates to avoid redownloading existing data.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        target_start : pd.Timestamp
            Desired start date
        target_end : pd.Timestamp
            Desired end date
            
        Returns
        -------
        List[Tuple[pd.Timestamp, pd.Timestamp]]
            List of (start, end) date ranges to download
        """
        existing_range = await self.get_existing_data_range(symbol)
        
        if not existing_range:
            # No existing data, download everything
            return [(target_start, target_end)]
            
        existing_start, existing_end = existing_range
        missing_ranges = []
        
        # Check for data before existing range
        if target_start < existing_start:
            missing_ranges.append((target_start, existing_start - pd.Timedelta(days=1)))
            
        # Check for data after existing range
        if target_end > existing_end:
            missing_ranges.append((existing_end + pd.Timedelta(days=1), target_end))
            
        return missing_ranges
    
    async def download_symbol_data(
        self,
        symbol: str,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None
    ) -> Optional[pd.DataFrame]:
        """Download data for a single symbol.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        progress : Optional[Progress]
            Rich progress instance for tracking
        task_id : Optional[TaskID]
            Task ID for the progress bar
            
        Returns
        -------
        Optional[pd.DataFrame]
            Downloaded data or None on error
        """
        try:
            # Update progress status
            if progress and task_id is not None:
                progress.update(task_id, description=f"Downloading {symbol}")
            
            # Check for existing data if skip_existing is enabled
            if self.incremental:
                missing_ranges = await self.get_missing_ranges(
                    symbol, self.start_time, self.end_time
                )
                
                if not missing_ranges:
                    self.skip_count += 1
                    if progress and task_id is not None:
                        progress.update(task_id, description=f"{symbol} - Already complete")
                        progress.advance(task_id)
                    return None
                    
                # Only download missing ranges
                all_data = []
                
                for range_start, range_end in missing_ranges:
                    data = await self.api_client.get_historical_data(
                        symbol=symbol,
                        start_date=range_start,
                        end_date=range_end,
                        interval=self.interval
                    )
                    
                    if not data.empty:
                        all_data.append(data)
                        
                if not all_data:
                    return None
                    
                combined_data = pd.concat(all_data).drop_duplicates(subset=["date"])
                
                # Merge with existing data if available
                symbol_file = self.get_symbol_file_path(symbol)
                if symbol_file.exists():
                    existing_data = pd.read_csv(symbol_file)
                    existing_data["date"] = pd.to_datetime(existing_data["date"])
                    
                    # Combine existing and new data
                    final_data = pd.concat([existing_data, combined_data])
                    final_data = final_data.drop_duplicates(subset=["date"])
                    final_data = final_data.sort_values("date")
                    
                    return final_data
                    
                return combined_data.sort_values("date")
                
            else:
                # Download full date range
                data = await self.api_client.get_historical_data(
                    symbol=symbol,
                    start_date=self.start_time,
                    end_date=self.end_time,
                    interval=self.interval
                )
                
                return data
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error downloading {symbol}: {e}")
            
            if progress and task_id is not None:
                progress.update(task_id, description=f"{symbol} - Error: {str(e)[:30]}...")
                
            return None
    
    async def save_symbol_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None
    ) -> bool:
        """Save symbol data to file.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        data : pd.DataFrame
            Data to save
        progress : Optional[Progress]
            Rich progress instance
        task_id : Optional[TaskID]
            Task ID for progress tracking
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            if data.empty:
                if progress and task_id is not None:
                    progress.update(task_id, description=f"{symbol} - No data")
                return False
                
            # Ensure data is sorted by date
            data = data.sort_values("date")
            
            # Save to CSV
            symbol_file = self.get_symbol_file_path(symbol)
            data.to_csv(symbol_file, index=False)
            
            self.success_count += 1
            
            if progress and task_id is not None:
                progress.update(
                    task_id, 
                    description=f"{symbol} - Saved {len(data)} records",
                    completed=100
                )
                progress.advance(task_id)
                
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error saving {symbol}: {e}")
            
            if progress and task_id is not None:
                progress.update(task_id, description=f"{symbol} - Save error: {str(e)[:30]}...")
                
            return False
    
    async def process_symbol(
        self,
        symbol: str,
        progress: Progress,
        task_id: TaskID
    ) -> bool:
        """Process a single symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol to process
        progress : Progress
            Progress bar instance
        task_id : TaskID
            Task ID for progress tracking
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Update task description
            progress.update(task_id, description=f"Downloading {symbol}")
            
            # Check for missing data ranges
            if self.incremental:
                missing_ranges = await self.get_missing_ranges(
                    symbol, self.start_time, self.end_time
                )
                
                if not missing_ranges:
                    progress.update(
                        task_id,
                        description=f"{symbol} - Already up to date",
                        completed=1
                    )
                    return True
                
                # Get data for each missing range
                all_data = []
                for range_start, range_end in missing_ranges:
                    data = await self.api_client.get_historical_data(
                        symbol=symbol,
                        start_date=range_start,
                        end_date=range_end,
                        interval=self.interval
                    )
                    
                    if not data.empty:
                        all_data.append(data)
                    
                # Combine all data
                if all_data:
                    df = pd.concat(all_data).drop_duplicates().sort_values("date")
                else:
                    df = pd.DataFrame()
            else:
                # Download full date range
                df = await self.api_client.get_historical_data(
                    symbol=symbol,
                    start_date=self.start_time,
                    end_date=self.end_time,
                    interval=self.interval
                )
            
            if df.empty:
                progress.update(
                    task_id,
                    description=f"{symbol} - No data available",
                    completed=1
                )
                return False
                
            # Save data
            success = await self.save_data(symbol, df)
            
            if success:
                progress.update(
                    task_id,
                    description=f"{symbol} - Saved {len(df)} records",
                    completed=1
                )
                return True
            else:
                progress.update(
                    task_id,
                    description=f"{symbol} - Failed to save data",
                    completed=1
                )
                return False
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            progress.update(
                task_id,
                description=f"{symbol} - Error: {str(e)}",
                completed=1
            )
            return False
    
    async def collect_data(
        self, 
        symbols: Optional[List[str]] = None,
        max_retries: int = 3,
        current_retry: int = 0,
        existing_progress: Optional[Progress] = None,
        existing_task: Optional[TaskID] = None
    ) -> Dict[str, bool]:
        """
        Collect data for all symbols in parallel.

        Args:
            symbols: Optional list of symbols to collect. If None, uses self.symbols.
            max_retries: Maximum number of retries for failed symbols.
            current_retry: Current retry count (for internal use).
            existing_progress: Existing Progress instance for retries.
            existing_task: Existing task ID for tracking overall progress.

        Returns:
            Dict mapping symbol to success status (True/False).
        """
        # Get symbols list if not provided
        if symbols is None:
            if not hasattr(self, 'symbols') or not self.symbols:
                self.symbols = await self.get_symbols()
            symbols = self.symbols
        
        # Initialize results dictionary
        results = {}
        failed_symbols = set()

        # Create progress display
        if existing_progress is None:
            # Only create a new progress instance if one wasn't provided
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            )
            
            start_progress = True
        else:
            # Use the existing progress instance
            progress = existing_progress
            start_progress = False
            
        # Define a context manager that only starts the progress if needed
        @contextlib.contextmanager
        def progress_context():
            if start_progress:
                with progress:
                    yield progress
            else:
                yield progress
        
        # Use API client context if not in a retry
        async with (contextlib.AsyncExitStack() as stack):
            # Only create a new client session if this is the first call
            if current_retry == 0:
                await stack.enter_async_context(self.api_client)
                
            with progress_context() as prog:
                # Create main task if not provided
                if existing_task is None:
                    overall_task = prog.add_task(
                        f"Collecting {self.interval} data for {len(symbols)} symbols...",
                        total=len(symbols)
                    )
                else:
                    overall_task = existing_task
                    # Update the description for the retry case
                    if current_retry > 0:
                        prog.update(overall_task, description=f"Retry #{current_retry}: Collecting {self.interval} data for {len(symbols)} symbols...")
                
                # Process symbols in batches
                symbol_batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
                
                for batch_idx, batch in enumerate(symbol_batches):
                    batch_tasks = []
                    tasks = {}
                    
                    for symbol in batch:
                        # Initialize task for this symbol
                        task_id = prog.add_task(f"{symbol} - Downloading...", total=1)
                        tasks[symbol] = task_id
                        
                        # Create task
                        task = asyncio.create_task(
                            self.process_symbol(symbol, prog, task_id)
                        )
                        batch_tasks.append((symbol, task))
                    
                    # Wait for all tasks in this batch to complete
                    for symbol, task in batch_tasks:
                        try:
                            success = await task
                            results[symbol] = success
                            if not success:
                                failed_symbols.add(symbol)
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                            results[symbol] = False
                            failed_symbols.add(symbol)
                        
                        # Update overall progress
                        prog.update(overall_task, advance=1)
                
                # Handle retries for failed symbols if needed
                if failed_symbols and current_retry < max_retries:
                    if len(failed_symbols) > 0:
                        prog.print(f"Retrying {len(failed_symbols)} failed symbols...")
                        
                        retry_symbols = list(failed_symbols)
                        retry_results = await self.collect_data(
                            symbols=retry_symbols,
                            max_retries=max_retries,
                            current_retry=current_retry + 1,
                            existing_progress=prog,
                            existing_task=overall_task
                        )
                        
                        # Update results with retry results
                        results.update(retry_results)
        
        # Output summary of results
        success_count = sum(1 for success in results.values() if success)
        if start_progress:  # Only print summary for the original call
            logger.info(f"Collected data for {success_count}/{len(symbols)} symbols successfully")
            logger.info(f"Success rate: {success_count/len(symbols)*100:.1f}%")
        
        return results
    
    @staticmethod
    def filter_symbols_by_index(symbols: List[str], index_name: str) -> List[str]:
        """Filter symbols to include only those in the specified index.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to filter
        index_name : str
            Index name (sp500, dowjones, nasdaq100)
            
        Returns
        -------
        List[str]
            Filtered list of symbols
        """
        # This is a static method so it can be used without instantiating the class
        async def get_index_symbols():
            async with FMPClient() as client:
                df = await client.get_index_constituents(index_name)
                if df.empty:
                    return []
                return df["symbol"].tolist()
                
        # Run async function in an event loop
        index_symbols = asyncio.run(get_index_symbols())
        
        # Filter input symbols to those in the index
        filtered = [s for s in symbols if s in index_symbols]
        
        return filtered

    async def save_data(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> bool:
        """Save data to file.
        
        Parameters
        ----------
        symbol : str
            Symbol to save
        data : pd.DataFrame
            Data to save
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            if data.empty:
                logger.warning(f"No data to save for {symbol}")
                return False
                
            # Create file path
            file_path = self.save_dir / f"{symbol}.csv"
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            return False

class DailyFMPCollector(BaseFMPCollector):
    """Collector for daily FMP data.
    
    Collects daily price data from FMP API.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        start: str,
        end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        max_workers: int = 5,
        redis_url: Optional[str] = None,
        incremental: bool = False,
        overwrite: bool = False,
        **kwargs
    ):
        """Initialize the DailyFMPCollector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save data to
        start : str
            Start date (YYYY-MM-DD)
        end : Optional[str], optional
            End date (YYYY-MM-DD), defaults to today
        symbols : Optional[List[str]], optional
            List of symbols to collect, defaults to None
        api_key : Optional[str], optional
            FMP API key, defaults to None
        max_workers : int, optional
            Maximum number of workers, defaults to 5
        redis_url : Optional[str], optional
            Redis URL for rate limiting, defaults to None
        incremental : bool, optional
            Whether to collect only new data, defaults to False
        overwrite : bool, optional
            Whether to overwrite existing files, defaults to False
        **kwargs : dict
            Additional keyword arguments
        """
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            symbols=symbols,
            api_key=api_key,
            max_workers=max_workers,
            redis_url=redis_url,
            incremental=incremental,
            overwrite=overwrite,
            **kwargs
        )
        
        # Set interval for daily data
        self.interval = "1d"
    
    async def process_symbol(self, symbol: str, progress: Progress, task_id: TaskID) -> bool:
        """Process data for a single symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol to process
        progress : Progress
            Progress bar instance for updating progress
        task_id : TaskID
            Task ID in the progress bar
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        progress.update(task_id, description=f"{symbol} - Downloading...")
        
        try:
            # Get data from API
            df = await self.api_client.get_historical_data(
                symbol=symbol,
                start_date=self.start_time,
                end_date=self.end_time,
                interval="1d"
            )
            
            if df.empty:
                progress.update(task_id, description=f"{symbol} - No data available")
                progress.update(task_id, completed=1)
                return False
                
            # Count data points
            progress.update(
                task_id, 
                description=f"{symbol} - Downloaded {len(df)} records"
            )
            
            # Save data
            success = await self.save_data(symbol, df)
            
            if success:
                progress.update(
                    task_id, 
                    description=f"{symbol} - Saved {len(df)} records", 
                    completed=1
                )
            else:
                progress.update(
                    task_id, 
                    description=f"{symbol} - Failed to save", 
                    completed=1
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            progress.update(task_id, description=f"{symbol} - Error: {str(e)}", completed=1)
            return False

class IntradayFMPCollector(BaseFMPCollector):
    """Collector for intraday FMP data.
    
    Collects intraday price data from FMP API.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        start: str,
        end: Optional[str] = None,
        interval: str = "15min",
        symbols: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        max_workers: int = 5,
        redis_url: Optional[str] = None,
        incremental: bool = False,
        overwrite: bool = False,
        lookback_days: int = 7,
        **kwargs
    ):
        """Initialize the IntradayFMPCollector.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save data to
        start : str
            Start date (YYYY-MM-DD)
        end : Optional[str], optional
            End date (YYYY-MM-DD), defaults to today
        interval : str, optional
            Data interval (1min, 5min, 15min, 30min, 1hour), defaults to "15min"
        symbols : Optional[List[str]], optional
            List of symbols to collect, defaults to None
        api_key : Optional[str], optional
            FMP API key, defaults to None
        max_workers : int, optional
            Maximum number of workers, defaults to 5
        redis_url : Optional[str], optional
            Redis URL for rate limiting, defaults to None
        incremental : bool, optional
            Whether to do incremental collection, defaults to False
        overwrite : bool, optional
            Whether to overwrite existing files, defaults to False
        lookback_days : int, optional
            Number of days to look back when checking missing data, defaults to 7
        **kwargs : dict
            Additional keyword arguments
        """
        # Create specific directory for this interval
        interval_dir = Path(save_dir) / interval
        
        super().__init__(
            save_dir=interval_dir,
            start=start,
            end=end,
            symbols=symbols,
            api_key=api_key,
            max_workers=max_workers,
            redis_url=redis_url,
            incremental=incremental,
            overwrite=overwrite,
            **kwargs
        )
        
        # Save specific properties
        self.interval = interval
        self.lookback_days = lookback_days
        
        # Validate interval
        valid_intervals = ["1min", "5min", "15min", "30min", "1hour"]
        if self.interval not in valid_intervals:
            raise ValueError(
                f"Invalid interval: {self.interval}. "
                f"Must be one of {valid_intervals}"
            )
            
    async def process_symbol(self, symbol: str, progress: Progress, task_id: TaskID) -> bool:
        """Process data for a single symbol.
        
        Parameters
        ----------
        symbol : str
            Symbol to process
        progress : Progress
            Progress bar instance for updating progress
        task_id : TaskID
            Task ID in the progress bar
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        progress.update(task_id, description=f"{symbol} - Downloading...")
        
        try:
            # Get data from API
            df = await self.api_client.get_historical_data(
                symbol=symbol,
                start_date=self.start_time,
                end_date=self.end_time,
                interval=self.interval
            )
            
            if df.empty:
                progress.update(task_id, description=f"{symbol} - No data available")
                progress.update(task_id, completed=1)
                return False
                
            # Count data points
            progress.update(
                task_id, 
                description=f"{symbol} - Downloaded {len(df)} records"
            )
            
            # Save data
            success = await self.save_data(symbol, df)
            
            if success:
                progress.update(
                    task_id, 
                    description=f"{symbol} - Saved {len(df)} records", 
                    completed=1
                )
            else:
                progress.update(
                    task_id, 
                    description=f"{symbol} - Failed to save", 
                    completed=1
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            progress.update(task_id, description=f"{symbol} - Error: {str(e)}", completed=1)
            return False

async def main():
    """Run a sample data collection."""
    # For testing the collector
    from rich.console import Console
    console = Console()
    
    # Check for API key
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        console.print("[bold red]Error: FMP_API_KEY not found in environment[/bold red]")
        return
        
    # Create collector for S&P 500 daily data
    console.print("[bold]Collecting S&P 500 daily data (sample)...[/bold]")
    collector = DailyFMPCollector(
        save_dir="~/.qlib/qlib_data/fmp_daily_test",
        start="2023-01-01",
        end="2023-01-31",
        api_key=api_key,
        cache_dir="~/.qlib/cache/fmp"
    )
    
    # Get S&P 500 symbols
    async with collector.api_client as client:
        constituents_df = await client.get_index_constituents("sp500")
        symbols = constituents_df["symbol"].tolist()[:5]  # Just test 5 symbols
    
    # Collect data
    results = await collector.collect_data(symbols)
    
    # Print results
    success_count = sum(1 for v in results.values() if v)
    console.print(f"[green]Successfully collected {success_count}/{len(symbols)} symbols[/green]")
    
if __name__ == "__main__":
    asyncio.run(main()) 