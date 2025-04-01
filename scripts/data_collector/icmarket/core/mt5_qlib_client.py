#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MT5 Data Fetching Client for QLib

This client fetches data from a MetaTrader 5 server (via WebSocket) and saves it in QLib format.
Features:
1. Checks qlib for last available datetime for each symbol/timeframe
2. Downloads missing data to fill gaps
3. Periodically updates data for each timeframe
4. Robust error handling with retries

Usage:
    python mt5_qlib_client.py --qlib_dir ~/.qlib/qlib_data/forex_data --server_url 192.168.160.1:8765
"""

import sys
import json
import asyncio
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import websockets
from rich.console import Console
from rich.logging import RichHandler

# Add parent directory to path so we can import modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.dump_bin import DumpDataUpdate
from scripts.data_collector.icmarket.fx_info import FxSymbol, TimeFrame
from scripts.data_collector.icmarket.migrate_eaframework import create_calendar, create_instruments_files

# Initialize Rich console for nice output
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("mt5_qlib_client")

# Mapping of MT5 timeframes to QLib frequencies
TIMEFRAME_TO_FREQ = {
    "M1": "1t",
    "M5": "5t",
    "M15": "15t",
    "M30": "30t",
    "H1": "1h",
    "H4": "4h",
    "D1": "day"
}

# Default list of symbols to monitor
DEFAULT_SYMBOLS = [
    # Major pairs
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
    # Commodities
    "XAUUSD", "XAGUSD",
    # Indices
    "US30", "US500", "USTEC"
]

# Default list of timeframes to monitor
DEFAULT_TIMEFRAMES = ["M1", "M5", "H1", "D1"]


class MT5QlibClient:
    """
    Client that fetches data from MT5 server and saves it to QLib format.
    
    Features:
    - Checks last available data in QLib
    - Downloads missing data to close gaps
    - Periodically updates data for each timeframe
    - Converts data to QLib format
    - Handles errors with retry logic
    """
    
    def __init__(
        self,
        qlib_dir: str,
        server_url: str,
        symbols: List[str] = DEFAULT_SYMBOLS,
        timeframes: List[str] = DEFAULT_TIMEFRAMES,
        download_days: int = 365,
        update_interval: int = 60,
        retry_wait: int = 10,
        retry_max: int = 3,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize the MT5 QLib Client
        
        Args:
            qlib_dir: Directory where QLib data is stored
            server_url: URL of the MT5 WebSocket server (format: host:port)
            symbols: List of symbols to download
            timeframes: List of timeframes to download
            download_days: Number of days of history to download if no data exists
            update_interval: Seconds between update checks
            retry_wait: Seconds to wait between retries on failure
            retry_max: Maximum number of retries before giving up
            temp_dir: Directory for temporary CSV files (defaults to qlib_dir/temp)
        """
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.server_url = f"ws://{server_url}" if not server_url.startswith("ws://") else server_url
        self.symbols = [str(FxSymbol(s)) for s in symbols]  # Normalize symbols
        self.timeframes = [str(TimeFrame(tf)) for tf in timeframes]  # Normalize timeframes
        self.download_days = download_days
        self.update_interval = update_interval
        self.retry_wait = retry_wait
        self.retry_max = retry_max
        
        # Set up directory structure
        self.temp_dir = Path(temp_dir) if temp_dir else self.qlib_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active connections and tasks
        self.active_tasks = {}
        self.all_connections = set()
        
        # Create a queue for dumping data to QLib format
        self.dump_queue = asyncio.Queue()
    
    async def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Get the last available timestamp for a symbol/timeframe in QLib
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe (e.g., "M1", "H1", "D1")
            
        Returns:
            Last timestamp as datetime or None if no data
        """
        # Map MT5 timeframe to QLib frequency
        freq = TIMEFRAME_TO_FREQ.get(timeframe, "day")
        
        # Check if the features directory exists
        features_dir = self.qlib_dir / "features" / symbol.lower()
        if not features_dir.exists():
            return None
            
        # Look for bin files matching the frequency
        bin_files = list(features_dir.glob(f"*.{freq}.bin"))
        if not bin_files:
            return None
            
        # For simplicity, we'll use the close price file to get the last timestamp
        close_file = features_dir / f"close.{freq}.bin"
        if not close_file.exists():
            # Try another file if close doesn't exist
            close_file = bin_files[0]
        
        try:
            # Binary files are stored as float32 values
            # First entry is date index, second is value
            import numpy as np
            data = np.fromfile(close_file, dtype="<f")
            
            if len(data) < 2:
                return None
                
            # Last timestamp is at the end of the file, taking two values at a time
            last_timestamp_float = data[-2]  # The date index
            
            # Convert from float to datetime
            last_timestamp = pd.Timestamp(datetime.fromtimestamp(last_timestamp_float))
            return last_timestamp
        except Exception as e:
            logger.warning(f"Error reading bin file for {symbol} {timeframe}: {e}")
            return None
    
    async def download_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[datetime] = None,
        history: int = 1000
    ) -> pd.DataFrame:
        """
        Download data for a symbol and timeframe from the MT5 server
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe (e.g., "M1", "H1", "D1")
            start_time: Start time for downloading data (None for now - history)
            history: Number of bars to download if start_time is None
            
        Returns:
            DataFrame with the downloaded data
        """
        # Create a websocket connection
        logger.info(f"Connecting to {self.server_url} for {symbol} {timeframe}")
        
        request = {
            "symbol": symbol,
            "timeframe": timeframe,
            "history": history
        }
        
        retry_count = 0
        while retry_count < self.retry_max:
            try:
                async with websockets.connect(self.server_url, ping_timeout=None) as websocket:
                    self.all_connections.add(websocket)
                    
                    # Send initial request
                    await websocket.send(json.dumps(request))
                    
                    # Get response with historical data
                    data = await websocket.recv()
                    bars = json.loads(data)
                    
                    # Remove websocket from tracking
                    self.all_connections.remove(websocket)
                    
                    if not bars:
                        logger.warning(f"No data received for {symbol} {timeframe}")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(bars)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    df.sort_index(inplace=True)
                    
                    logger.info(f"Downloaded {len(df)} bars for {symbol} {timeframe}")
                    return df
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Error downloading data for {symbol} {timeframe}: {e}")
                logger.info(f"Retrying in {self.retry_wait} seconds... ({retry_count}/{self.retry_max})")
                await asyncio.sleep(self.retry_wait)
        
        logger.error(f"Failed to download data for {symbol} {timeframe} after {self.retry_max} retries")
        return pd.DataFrame()
    
    async def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Save DataFrame to CSV in the temp directory
        
        Args:
            df: DataFrame with the data
            symbol: The trading symbol
            timeframe: The timeframe (e.g., "M1", "H1", "D1")
            
        Returns:
            Path to the saved CSV file or None if failed
        """
        if df.empty:
            return None
            
        # Add required columns for QLib
        df.reset_index(inplace=True)
        df.rename(columns={
            "time": "date",
            "open": "open",
            "high": "high", 
            "low": "low",
            "close": "close",
            "tick_volume": "volume"  # Use tick_volume as volume
        }, inplace=True)
        
        # Add factor column (always 1.0 for forex)
        df["factor"] = 1.0
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Add VWAP column (approximated)
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        
        # Save to CSV
        output_file = self.temp_dir / f"{symbol}_{timeframe}.csv"
        try:
            df.to_csv(output_file, index=False)
            return output_file
        except Exception as e:
            logger.error(f"Error saving CSV for {symbol} {timeframe}: {e}")
            return None
    
    async def dump_to_qlib(self, csv_file: Path, timeframe: str):
        """
        Dump CSV file to QLib format
        
        Args:
            csv_file: Path to the CSV file
            timeframe: The timeframe to determine frequency
        """
        freq = TIMEFRAME_TO_FREQ.get(timeframe, "day")
        
        try:
            # Use DumpDataUpdate to update existing data
            dumper = DumpDataUpdate(
                csv_path=str(self.temp_dir),
                qlib_dir=str(self.qlib_dir),
                include_fields="open,high,low,close,volume,factor,vwap",
                date_field_name="date",
                symbol_field_name="symbol",
                freq=freq
            )
            dumper.dump()
            logger.info(f"Successfully dumped {csv_file.name} to QLib with frequency {freq}")
        except Exception as e:
            logger.error(f"Error dumping to QLib: {e}")
            logger.debug(traceback.format_exc())
    
    async def process_dump_queue(self):
        """Process the queue of CSV files to dump to QLib format"""
        while True:
            try:
                csv_file, timeframe = await self.dump_queue.get()
                await self.dump_to_qlib(csv_file, timeframe)
                self.dump_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing dump queue: {e}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(5)  # Wait a bit before retrying
    
    async def update_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Update data for a specific symbol and timeframe
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe (e.g., "M1", "H1", "D1")
        """
        # Get last timestamp from QLib
        last_timestamp = await self.get_last_timestamp(symbol, timeframe)
        
        # If no data exists, download initial history
        if last_timestamp is None:
            logger.info(f"No data found for {symbol} {timeframe}, downloading {self.download_days} days of history")
            days_multiplier = {
                "M1": 2,    # For M1, just get 2 days to avoid too much data
                "M5": 5,    # For M5, get 5 days
                "M15": 10,  # For M15, get 10 days
                "M30": 15,  # For M30, get 15 days
                "H1": 30,   # For H1, get 30 days
                "H4": 120,  # For H4, get 120 days
                "D1": 365,  # For D1, get a full year
            }
            
            # Calculate history based on timeframe
            multiplier = days_multiplier.get(timeframe, 1)
            history_days = min(self.download_days, multiplier)
            history_bars = history_days * 24 * 60 // TimeFrame(timeframe).minutes
            
            df = await self.download_data(symbol, timeframe, history=history_bars)
        else:
            # Calculate how many bars to download based on the timeframe
            minutes_per_bar = TimeFrame(timeframe).minutes
            now = datetime.now()
            missing_minutes = int((now - last_timestamp).total_seconds() / 60)
            bars_to_download = max(10, missing_minutes // minutes_per_bar + 10)  # Add some buffer
            
            logger.info(f"Last data for {symbol} {timeframe} is from {last_timestamp}, downloading {bars_to_download} bars")
            df = await self.download_data(symbol, timeframe, history=bars_to_download)
        
        # Save to CSV and queue for dumping to QLib
        if not df.empty:
            csv_file = await self.save_to_csv(df, symbol, timeframe)
            if csv_file:
                await self.dump_queue.put((csv_file, timeframe))
    
    async def monitor_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Continuously monitor and update data for a symbol and timeframe
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe (e.g., "M1", "H1", "D1")
        """
        update_intervals = {
            "M1": 60,      # Update M1 data every minute
            "M5": 300,     # Update M5 data every 5 minutes
            "M15": 900,    # Update M15 data every 15 minutes
            "M30": 1800,   # Update M30 data every 30 minutes
            "H1": 3600,    # Update H1 data every hour
            "H4": 14400,   # Update H4 data every 4 hours
            "D1": 86400,   # Update D1 data every day
        }
        
        interval = update_intervals.get(timeframe, self.update_interval)
        
        while True:
            try:
                # FIXME: the update can cost time, but the sleep is fixed, then 
                # this may introduce cumulative delay, makes the update not accurately on
                # the correct time, it should be, e.g for M5, it should update at :00, :05, :10 etc..
                await self.update_symbol_timeframe(symbol, timeframe)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info(f"Task for {symbol} {timeframe} was cancelled")
                break
            except Exception as e:
                logger.error(f"Error updating {symbol} {timeframe}: {e}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(self.retry_wait)
    
    async def setup_instruments(self):
        """Set up instrument files in QLib directory"""
        try:
            # Group symbols by category
            symbols_by_category = {}
            for symbol in self.symbols:
                fx_symbol = FxSymbol(symbol)
                category = fx_symbol.category.lower()
                if category not in symbols_by_category:
                    symbols_by_category[category] = []
                symbols_by_category[category].append(symbol)
            
            # Create instrument files
            create_instruments_files(str(self.qlib_dir), symbols_by_category)
            
            # Create calendar files
            create_calendar(str(self.temp_dir), str(self.qlib_dir))
            
            logger.info("Created instrument and calendar files")
        except Exception as e:
            logger.error(f"Error setting up instruments: {e}")
            logger.debug(traceback.format_exc())
    
    async def start(self):
        """Start the client and monitor all symbols and timeframes"""
        logger.info(f"Starting MT5 QLib Client for {len(self.symbols)} symbols and {len(self.timeframes)} timeframes")
        logger.info(f"QLib directory: {self.qlib_dir}")
        logger.info(f"MT5 server: {self.server_url}")
        
        # Set up QLib directory structure
        await self.setup_instruments()
        
        # Start dump queue processor
        asyncio.create_task(self.process_dump_queue())
        
        # Create tasks for each symbol and timeframe
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task_key = f"{symbol}_{timeframe}"
                task = asyncio.create_task(self.monitor_symbol_timeframe(symbol, timeframe))
                self.active_tasks[task_key] = task
        
        # Wait for all tasks
        try:
            while self.active_tasks:
                # Check for completed tasks
                done_tasks = []
                for key, task in self.active_tasks.items():
                    if task.done():
                        if task.exception():
                            logger.error(f"Task {key} failed with exception: {task.exception()}")
                            # Restart the task
                            symbol, timeframe = key.split("_")
                            self.active_tasks[key] = asyncio.create_task(
                                self.monitor_symbol_timeframe(symbol, timeframe)
                            )
                            logger.info(f"Restarted task for {key}")
                        else:
                            done_tasks.append(key)
                
                # Remove completed tasks
                for key in done_tasks:
                    del self.active_tasks[key]
                
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            # Cancel all tasks
            for key, task in self.active_tasks.items():
                task.cancel()
            
            # Close all connections
            for websocket in self.all_connections:
                if not websocket.closed:
                    await websocket.close()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MT5 Data Fetching Client for QLib")
    parser.add_argument("--qlib_dir", type=str, default="~/.qlib/qlib_data/forex_data",
                        help="Directory where QLib data is stored")
    parser.add_argument("--server_url", type=str, default="192.168.160.1:8765",
                        help="URL of the MT5 WebSocket server (format: host:port)")
    parser.add_argument("--symbols", type=str, nargs="+", default=DEFAULT_SYMBOLS,
                        help="List of symbols to download")
    parser.add_argument("--timeframes", type=str, nargs="+", default=DEFAULT_TIMEFRAMES,
                        help="List of timeframes to download")
    parser.add_argument("--download_days", type=int, default=365,
                        help="Number of days of history to download if no data exists")
    parser.add_argument("--update_interval", type=int, default=60,
                        help="Seconds between update checks")
    parser.add_argument("--retry_wait", type=int, default=10,
                        help="Seconds to wait between retries on failure")
    parser.add_argument("--retry_max", type=int, default=3,
                        help="Maximum number of retries before giving up")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Directory for temporary CSV files (defaults to qlib_dir/temp)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create and start client
    client = MT5QlibClient(
        qlib_dir=args.qlib_dir,
        server_url=args.server_url,
        symbols=args.symbols,
        timeframes=args.timeframes,
        download_days=args.download_days,
        update_interval=args.update_interval,
        retry_wait=args.retry_wait,
        retry_max=args.retry_max,
        temp_dir=args.temp_dir
    )
    
    await client.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1) 