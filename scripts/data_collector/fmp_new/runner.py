"""Runner for Financial Modeling Prep data collection.

This module provides a runner class that coordinates the entire process of
collecting, normalizing, and dumping data from FMP to Qlib format.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from scripts.data_collector.fmp_new.api import FMPClient
from scripts.data_collector.fmp_new.collector import (
    DailyFMPCollector,
    IntradayFMPCollector,
)
from scripts.data_collector.fmp_new.normalizer import (
    run_normalization,
)
from scripts.dump_bin import DumpDataAll, DumpDataUpdate

console = Console()

class FMPRunner:
    """Runner for FMP data collection and processing.
    
    This class coordinates the entire process of:
    1. Collecting data from FMP API
    2. Normalizing the data to Qlib format
    3. Dumping the data to Qlib binary format
    
    It supports both daily and intraday data collection.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        max_collector_count: int = 2,
        delay: float = 0.1,
        api_key: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        redis_password: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize runner.
        
        Parameters
        ----------
        max_workers : int
            Maximum number of parallel workers
        max_collector_count : int
            Maximum collection attempts per symbol
        delay : float
            Delay between API calls
        api_key : Optional[str]
            FMP API key
        redis_url : str
            Redis URL for rate limiting
        redis_password : Optional[str]
            Redis password
        cache_dir : Optional[Union[str, Path]]
            Directory for API response cache
        """
        self.max_workers = max_workers
        self.max_collector_count = max_collector_count
        self.delay = delay
        
        # API credentials
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key not provided and not found in environment variables")
            
        self.redis_url = redis_url
        self.redis_password = redis_password
        
        # Cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    async def get_index_constituents(self, index_name: str = "sp500") -> pd.DataFrame:
        """Get current index constituents.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dowjones, nasdaq100)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with constituents information
        """
        async with FMPClient(
            api_key=self.api_key,
            redis_url=self.redis_url,
            redis_password=self.redis_password,
            cache_dir=self.cache_dir
        ) as client:
            df = await client.get_index_constituents(index_name)
            
        if df.empty:
            logger.warning(f"No constituents found for {index_name}")
            
        return df
    
    async def get_historical_constituents(self, index_name: str = "sp500") -> pd.DataFrame:
        """Get historical index constituent changes.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dowjones, nasdaq100)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with historical constituent changes
        """
        async with FMPClient(
            api_key=self.api_key,
            redis_url=self.redis_url,
            redis_password=self.redis_password,
            cache_dir=self.cache_dir
        ) as client:
            df = await client.get_historical_constituents(index_name)
            
        if df.empty:
            logger.warning(f"No historical constituents found for {index_name}")
            
        return df
    
    def save_index_constituents(
        self,
        output_dir: Union[str, Path],
        index_name: str = "sp500"
    ) -> bool:
        """Save current and historical index constituents.
        
        Parameters
        ----------
        output_dir : Union[str, Path]
            Directory to save constituent data
        index_name : str
            Index name (sp500, dowjones, nasdaq100)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get current constituents
            current_df = asyncio.run(self.get_index_constituents(index_name))
            if not current_df.empty:
                current_file = output_path / f"{index_name}_constituents.csv"
                current_df.to_csv(current_file, index=False)
                logger.info(f"Saved {len(current_df)} current constituents to {current_file}")
                
            # Get historical changes
            historical_df = asyncio.run(self.get_historical_constituents(index_name))
            if not historical_df.empty:
                historical_file = output_path / f"{index_name}_historical_constituents.csv"
                historical_df.to_csv(historical_file, index=False)
                logger.info(f"Saved {len(historical_df)} historical constituent changes to {historical_file}")
                
            return not (current_df.empty and historical_df.empty)
                
        except Exception as e:
            logger.error(f"Error saving {index_name} constituents: {e}")
            return False
    
    def collect_daily_data(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        skip_existing: bool = False
    ) -> bool:
        """Collect daily data.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save collected data
        start : Optional[str]
            Start date (YYYY-MM-DD)
        end : Optional[str]
            End date (YYYY-MM-DD)
        symbols : Optional[List[str]]
            List of symbols to collect
        index_name : Optional[str]
            Index name to filter symbols (sp500, dowjones, nasdaq100)
        skip_existing : bool
            Skip symbols with existing data
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # If index_name is provided but not symbols, get symbols from index
            if index_name and not symbols:
                index_df = asyncio.run(self.get_index_constituents(index_name))
                if not index_df.empty:
                    symbols = index_df["symbol"].tolist()
                    logger.info(f"Using {len(symbols)} symbols from {index_name}")
                else:
                    logger.error(f"Failed to get symbols for {index_name}")
                    return False
                    
            # Create collector
            collector = DailyFMPCollector(
                save_dir=save_dir,
                start=start,
                end=end,
                symbols=symbols,
                max_workers=self.max_workers,
                delay=self.delay,
                skip_existing=skip_existing,
                api_key=self.api_key,
                redis_url=self.redis_url,
                redis_password=self.redis_password,
                cache_dir=self.cache_dir
            )
            
            # Collect data
            results = asyncio.run(collector.collect_data())
            
            # Check results
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            
            logger.info(f"Daily data collection complete: {success_count}/{total_count} symbols successful")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error collecting daily data: {e}")
            return False
    
    def collect_intraday_data(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "15min",
        symbols: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        skip_existing: bool = False
    ) -> bool:
        """Collect intraday data.
        
        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save collected data
        start : Optional[str]
            Start date (YYYY-MM-DD)
        end : Optional[str]
            End date (YYYY-MM-DD)
        interval : str
            Intraday interval (1min, 5min, 15min, 30min, 1hour, 4hour)
        symbols : Optional[List[str]]
            List of symbols to collect
        index_name : Optional[str]
            Index name to filter symbols (sp500, dowjones, nasdaq100)
        skip_existing : bool
            Skip symbols with existing data
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # If index_name is provided but not symbols, get symbols from index
            if index_name and not symbols:
                index_df = asyncio.run(self.get_index_constituents(index_name))
                if not index_df.empty:
                    symbols = index_df["symbol"].tolist()
                    logger.info(f"Using {len(symbols)} symbols from {index_name}")
                else:
                    logger.error(f"Failed to get symbols for {index_name}")
                    return False
                    
            # Create collector
            collector = IntradayFMPCollector(
                save_dir=save_dir,
                start=start,
                end=end,
                interval=interval,
                symbols=symbols,
                max_workers=self.max_workers,
                delay=self.delay,
                skip_existing=skip_existing,
                api_key=self.api_key,
                redis_url=self.redis_url,
                redis_password=self.redis_password,
                cache_dir=self.cache_dir
            )
            
            # Collect data
            results = asyncio.run(collector.collect_data())
            
            # Check results
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            
            logger.info(f"Intraday {interval} data collection complete: {success_count}/{total_count} symbols successful")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error collecting intraday data: {e}")
            return False
    
    def normalize_data(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        interval: str = "1d",
        qlib_daily_dir: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Normalize raw data to Qlib format.
        
        Parameters
        ----------
        data_dir : Union[str, Path]
            Directory containing raw data
        output_dir : Union[str, Path]
            Directory to save normalized data
        interval : str
            Data interval (1d, 15min, etc.)
        qlib_daily_dir : Optional[Union[str, Path]]
            Directory containing daily Qlib data (for intraday normalization)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_dir).expanduser().resolve()
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Run normalization
            run_normalization(
                data_dir=data_dir,
                output_dir=output_dir,
                interval=interval,
                qlib_daily_dir=qlib_daily_dir,
                max_workers=self.max_workers
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return False
    
    def dump_data(
        self,
        csv_path: Union[str, Path],
        qlib_dir: Union[str, Path],
        freq: str = "day",
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "open,close,high,low,volume,factor",
        limit_symbols: Optional[List[str]] = None,
        update_mode: bool = True
    ) -> bool:
        """Dump normalized data to Qlib binary format.
        
        Parameters
        ----------
        csv_path : Union[str, Path]
            Directory containing normalized CSV files
        qlib_dir : Union[str, Path]
            Directory to save Qlib data
        freq : str
            Data frequency (day, 15min, etc.)
        date_field_name : str
            Name of date column
        symbol_field_name : str
            Name of symbol column
        exclude_fields : str
            Comma-separated list of fields to exclude
        include_fields : str
            Comma-separated list of fields to include
        limit_symbols : Optional[List[str]]
            Limit dumping to these symbols only
        update_mode : bool
            If True, use update mode instead of full dump
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Convert paths
            csv_path = Path(csv_path).expanduser().resolve()
            qlib_dir = Path(qlib_dir).expanduser().resolve()
            qlib_dir.mkdir(parents=True, exist_ok=True)
            
            # Dump parameters
            kwargs = {
                "csv_path": str(csv_path),
                "qlib_dir": str(qlib_dir),
                "freq": freq,
                "date_field_name": date_field_name,
                "symbol_field_name": symbol_field_name,
                "exclude_fields": exclude_fields,
                "include_fields": include_fields,
            }
            
            if limit_symbols:
                kwargs["symbol"] = ",".join(limit_symbols)
                
            # Use appropriate dumper based on mode
            if update_mode:
                dumper = DumpDataUpdate(**kwargs)
            else:
                dumper = DumpDataAll(**kwargs)
                
            dumper.dump()
            
            return True
            
        except Exception as e:
            logger.error(f"Error dumping data: {e}")
            return False
    
    def run_daily_pipeline(
        self,
        raw_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_daily_raw",
        normalized_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_daily_normalized",
        qlib_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_daily",
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        index_name: Optional[str] = "sp500",
        skip_existing: bool = True,
        save_constituents: bool = True,
        update_mode: bool = True
    ) -> bool:
        """Run complete daily data pipeline.
        
        Parameters
        ----------
        raw_dir : Union[str, Path]
            Directory to save raw data
        normalized_dir : Union[str, Path]
            Directory to save normalized data
        qlib_dir : Union[str, Path]
            Directory to save Qlib data
        start : Optional[str]
            Start date (YYYY-MM-DD)
        end : Optional[str]
            End date (YYYY-MM-DD)
        symbols : Optional[List[str]]
            List of symbols to collect
        index_name : Optional[str]
            Index name to filter symbols (sp500, dowjones, nasdaq100)
        skip_existing : bool
            Skip symbols with existing data
        save_constituents : bool
            Save index constituents data
        update_mode : bool
            Use update mode for dumping
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            console.print(Panel.fit(
                "[bold yellow]Running Daily Data Pipeline[/bold yellow]",
                title="FMP Data Collector"
            ))
            
            # Save index constituents if requested
            if save_constituents and index_name:
                console.print("[bold]Step 1/4: Saving index constituents[/bold]")
                self.save_index_constituents(
                    output_dir=Path(qlib_dir).expanduser().resolve() / "instruments",
                    index_name=index_name
                )
                
            # Collect raw data
            console.print("[bold]Step 2/4: Collecting raw daily data[/bold]")
            collect_success = self.collect_daily_data(
                save_dir=raw_dir,
                start=start,
                end=end,
                symbols=symbols,
                index_name=index_name,
                skip_existing=skip_existing
            )
            
            if not collect_success:
                logger.warning("Data collection had some issues, but continuing with available data")
                
            # Normalize data
            console.print("[bold]Step 3/4: Normalizing daily data[/bold]")
            normalize_success = self.normalize_data(
                data_dir=raw_dir,
                output_dir=normalized_dir,
                interval="1d"
            )
            
            if not normalize_success:
                logger.error("Data normalization failed")
                return False
                
            # Dump to Qlib format
            console.print("[bold]Step 4/4: Dumping data to Qlib format[/bold]")
            dump_success = self.dump_data(
                csv_path=normalized_dir,
                qlib_dir=qlib_dir,
                freq="day",
                update_mode=update_mode
            )
            
            if dump_success:
                console.print("[bold green]Daily data pipeline completed successfully![/bold green]")
                return True
            else:
                console.print("[bold red]Failed to dump data to Qlib format[/bold red]")
                return False
                
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")
            console.print(f"[bold red]Pipeline error: {e}[/bold red]")
            return False
    
    def run_intraday_pipeline(
        self,
        raw_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_15min_raw",
        normalized_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_15min_normalized",
        qlib_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_15min",
        qlib_daily_dir: Union[str, Path] = "~/.qlib/qlib_data/fmp_daily",
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "15min",
        symbols: Optional[List[str]] = None,
        index_name: Optional[str] = "sp500",
        skip_existing: bool = True,
        update_mode: bool = True
    ) -> bool:
        """Run complete intraday data pipeline.
        
        Parameters
        ----------
        raw_dir : Union[str, Path]
            Directory to save raw data
        normalized_dir : Union[str, Path]
            Directory to save normalized data
        qlib_dir : Union[str, Path]
            Directory to save Qlib data
        qlib_daily_dir : Union[str, Path]
            Directory containing daily Qlib data
        start : Optional[str]
            Start date (YYYY-MM-DD)
        end : Optional[str]
            End date (YYYY-MM-DD)
        interval : str
            Intraday interval (1min, 5min, 15min, 30min, 1hour, 4hour)
        symbols : Optional[List[str]]
            List of symbols to collect
        index_name : Optional[str]
            Index name to filter symbols (sp500, dowjones, nasdaq100)
        skip_existing : bool
            Skip symbols with existing data
        update_mode : bool
            Use update mode for dumping
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            console.print(Panel.fit(
                f"[bold yellow]Running {interval} Intraday Data Pipeline[/bold yellow]",
                title="FMP Data Collector"
            ))
            
            # Map FMP interval to Qlib frequency
            freq_map = {
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
                "1hour": "60min",
                "4hour": "240min"
            }
            
            qlib_freq = freq_map.get(interval, interval)
            
            # Collect raw data
            console.print("[bold]Step 1/3: Collecting raw intraday data[/bold]")
            collect_success = self.collect_intraday_data(
                save_dir=raw_dir,
                start=start,
                end=end,
                interval=interval,
                symbols=symbols,
                index_name=index_name,
                skip_existing=skip_existing
            )
            
            if not collect_success:
                logger.warning("Data collection had some issues, but continuing with available data")
                
            # Normalize data
            console.print("[bold]Step 2/3: Normalizing intraday data[/bold]")
            normalize_success = self.normalize_data(
                data_dir=raw_dir,
                output_dir=normalized_dir,
                interval=interval,
                qlib_daily_dir=qlib_daily_dir
            )
            
            if not normalize_success:
                logger.error("Data normalization failed")
                return False
                
            # Dump to Qlib format
            console.print("[bold]Step 3/3: Dumping data to Qlib format[/bold]")
            dump_success = self.dump_data(
                csv_path=normalized_dir,
                qlib_dir=qlib_dir,
                freq=qlib_freq,
                update_mode=update_mode
            )
            
            if dump_success:
                console.print(f"[bold green]{interval} intraday data pipeline completed successfully![/bold green]")
                return True
            else:
                console.print("[bold red]Failed to dump data to Qlib format[/bold red]")
                return False
                
        except Exception as e:
            logger.error(f"Error in intraday pipeline: {e}")
            console.print(f"[bold red]Pipeline error: {e}[/bold red]")
            return False

def main():
    """Run the FMP data collection pipeline from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FMP Data Collector")
    
    # Common parameters
    parser.add_argument("--mode", type=str, choices=["daily", "intraday", "both"], default="both",
                      help="Data collection mode (daily, intraday, or both)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--index", type=str, default="sp500", 
                      help="Index name (sp500, dowjones, nasdaq100)")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    parser.add_argument("--no-skip", action="store_true", 
                      help="Do not skip existing data")
    
    # Daily data parameters
    parser.add_argument("--daily-raw-dir", type=str, default="~/.qlib/qlib_data/fmp_daily_raw",
                      help="Directory for raw daily data")
    parser.add_argument("--daily-norm-dir", type=str, default="~/.qlib/qlib_data/fmp_daily_normalized",
                      help="Directory for normalized daily data")
    parser.add_argument("--daily-qlib-dir", type=str, default="~/.qlib/qlib_data/fmp_daily",
                      help="Directory for Qlib daily data")
    
    # Intraday data parameters
    parser.add_argument("--interval", type=str, default="15min",
                      help="Intraday interval (1min, 5min, 15min, 30min, 1hour, 4hour)")
    parser.add_argument("--intraday-raw-dir", type=str, default="~/.qlib/qlib_data/fmp_15min_raw",
                      help="Directory for raw intraday data")
    parser.add_argument("--intraday-norm-dir", type=str, default="~/.qlib/qlib_data/fmp_15min_normalized",
                      help="Directory for normalized intraday data")
    parser.add_argument("--intraday-qlib-dir", type=str, default="~/.qlib/qlib_data/fmp_15min",
                      help="Directory for Qlib intraday data")
    
    # Other parameters
    parser.add_argument("--cache-dir", type=str, default="~/.qlib/cache/fmp",
                      help="Directory for API response cache")
    parser.add_argument("--max-workers", type=int, default=10,
                      help="Maximum number of parallel workers")
    parser.add_argument("--full-dump", action="store_true",
                      help="Use full dump instead of update mode")
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Create runner
    runner = FMPRunner(
        max_workers=args.max_workers,
        cache_dir=args.cache_dir
    )
    
    # Run pipelines based on mode
    if args.mode in ["daily", "both"]:
        runner.run_daily_pipeline(
            raw_dir=args.daily_raw_dir,
            normalized_dir=args.daily_norm_dir,
            qlib_dir=args.daily_qlib_dir,
            start=args.start,
            end=args.end,
            symbols=symbols,
            index_name=args.index,
            skip_existing=not args.no_skip,
            update_mode=not args.full_dump
        )
    
    if args.mode in ["intraday", "both"]:
        # Update intraday dirs based on interval if using default
        if args.interval != "15min" and args.intraday_raw_dir == "~/.qlib/qlib_data/fmp_15min_raw":
            args.intraday_raw_dir = f"~/.qlib/qlib_data/fmp_{args.interval}_raw"
            args.intraday_norm_dir = f"~/.qlib/qlib_data/fmp_{args.interval}_normalized"
            args.intraday_qlib_dir = f"~/.qlib/qlib_data/fmp_{args.interval}"
        
        runner.run_intraday_pipeline(
            raw_dir=args.intraday_raw_dir,
            normalized_dir=args.intraday_norm_dir,
            qlib_dir=args.intraday_qlib_dir,
            qlib_daily_dir=args.daily_qlib_dir,
            start=args.start,
            end=args.end,
            interval=args.interval,
            symbols=symbols,
            index_name=args.index,
            skip_existing=not args.no_skip,
            update_mode=not args.full_dump
        )

if __name__ == "__main__":
    main() 