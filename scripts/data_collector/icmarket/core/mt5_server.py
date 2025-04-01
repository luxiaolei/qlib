#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MT5 WebSocket Server

This script creates a WebSocket server that connects to a MetaTrader 5 terminal
and serves data to clients. It's designed to work with the mt5_qlib_client.py script.

Features:
1. Connects to a local MT5 terminal
2. Serves WebSocket connections from clients
3. Handles requests for historical bar data
4. Exposes MT5 functionality without requiring direct MT5 access from clients

Usage:
    python mt5_server.py --host 0.0.0.0 --port 8765 --log_level INFO
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import MetaTrader5 as mt5  # This requires the MetaTrader5 package: pip install MetaTrader5
import websockets
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Install Rich traceback handler
install_rich_traceback()

# Initialize Rich console for nice output
console = Console()

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("mt5_server")

# Import TimeFrame from local module
# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Mapping of timeframe strings to MT5 timeframe constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Cache to store recent data requests to minimize MT5 API calls
class DataCache:
    """Cache for storing recent data requests to minimize MT5 API calls"""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """
        Initialize the data cache
        
        Args:
            max_size: Maximum number of items to store in cache
            ttl: Time to live in seconds for cache items
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache if it exists and is not expired"""
        if key in self.cache:
            item = self.cache[key]
            if datetime.now().timestamp() - item["timestamp"] <= self.ttl:
                return item["data"]
            # Item expired, remove it
            del self.cache[key]
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Set an item in the cache with current timestamp"""
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now().timestamp()
        }
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()

"""
FIXME:
It should have configs of
  mt5_exe_path: "C:/Program Files/MetaTrader 5/terminal64.exe"
  server_url: "http://185.150.189.132:8000"
  server_secret: "EAFramework2024@"
as in `/Users/xlmini/MyRepos/EAFramework/src/barprovider/_mt5barprovider.py` `RemoteMT5BarDataProvider`
"""
class MT5Server:
    """
    WebSocket server that connects to a MetaTrader 5 terminal and serves data to clients
    
    Features:
    - Connects to local MT5 terminal
    - Handles requests for historical bar data
    - Caches data to reduce MT5 API calls
    - Gracefully handles disconnections
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        cache_ttl: int = 300,
        cache_size: int = 100
    ):
        """
        Initialize the MT5 Server
        
        Args:
            host: Host to bind the WebSocket server to
            port: Port to bind the WebSocket server to
            cache_ttl: Time to live in seconds for cache items
            cache_size: Maximum number of items to store in cache
        """
        self.host = host
        self.port = port
        self.cache = DataCache(max_size=cache_size, ttl=cache_ttl)
        self.server = None
        self.connected_clients = set()
        self._shutdown_requested = False
        
        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown"""
        if self._shutdown_requested:
            logger.warning("Forced shutdown")
            sys.exit(1)
        
        logger.info("Shutdown requested, closing connections...")
        self._shutdown_requested = True
        if self.server:
            asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Gracefully shut down the server"""
        logger.info("Shutting down MT5 server...")
        
        # Close all client connections
        if self.connected_clients:
            close_tasks = [client.close() for client in self.connected_clients 
                          if not client.closed]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Close MT5 connection
        if mt5.terminal_info() is not None:
            logger.info("Shutting down MT5 connection...")
            mt5.shutdown()
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Server shutdown complete")
    
    async def connect_mt5(self) -> bool:
        """
        Connect to the MetaTrader 5 terminal
        
        Returns:
            True if connection was successful, False otherwise
        """
        try:
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                return False
            
            # Check connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("MT5 terminal not connected")
                return False
            
            # Log connection info
            account_info = mt5.account_info()
            if account_info is not None:
                logger.info(f"Connected to MT5 - Account: {account_info.login} ({account_info.server})")
                logger.info(f"MT5 terminal build: {terminal_info.build}")
                logger.info(f"MT5 terminal path: {terminal_info.path}")
                return True
            else:
                logger.error("Failed to get account info")
                return False
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    async def get_bar_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        history: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get bar data from MT5
        
        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe (e.g., "M1", "H1", "D1")
            start_time: Start time for data (None for current time - history)
            history: Number of bars to get if start_time is None
            
        Returns:
            List of bar data dictionaries
        """
        # Check if MT5 is connected
        if mt5.terminal_info() is None:
            logger.warning("MT5 not connected, attempting to reconnect...")
            if not await self.connect_mt5():
                logger.error("Failed to reconnect to MT5")
                return []
        
        # Convert timeframe string to MT5 timeframe constant
        mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return []
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol not found: {symbol}")
            return []
        
        # Ensure symbol is selected in Market Watch
        if not symbol_info.visible:
            logger.info(f"Symbol {symbol} not visible, selecting...")
            mt5.symbol_select(symbol, True)
        
        # Generate cache key
        cache_key = f"{symbol}_{timeframe}_{start_time}_{history}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return cached_data
        
        # Set up time range
        if start_time is None:
            # Get data for the last 'history' bars
            bars = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, history)
        else:
            # Get data from start_time until now
            bars = mt5.copy_rates_from(symbol, mt5_timeframe, start_time, history)
        
        if bars is None or len(bars) == 0:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return []
        
        # Convert bars to list of dictionaries
        result = []
        for bar in bars:
            result.append({
                "time": int(bar["time"]),  # Unix timestamp
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "tick_volume": int(bar["tick_volume"]),
                "spread": int(bar["spread"]),
                "real_volume": int(bar["real_volume"])
            })
        
        # Cache the result
        self.cache.set(cache_key, result)
        
        logger.info(f"Retrieved {len(result)} bars for {symbol} {timeframe}")
        return result
    
    async def handle_client(self, websocket, path):
        """
        Handle a client connection
        
        Args:
            websocket: WebSocket connection object
            path: Connection path
        """
        client_id = id(websocket)
        self.connected_clients.add(websocket)
        
        logger.info(f"Client connected: {client_id} from {websocket.remote_address[0]}")
        
        try:
            while True:
                try:
                    # Wait for messages from client
                    message = await websocket.recv()
                    request = json.loads(message)
                    
                    logger.info(f"Received request from client {client_id}: {request}")
                    
                    # Process request
                    if "symbol" in request and "timeframe" in request:
                        symbol = request["symbol"]
                        timeframe = request["timeframe"]
                        start_time = datetime.fromtimestamp(request["start_time"]) if "start_time" in request else None
                        history = request.get("history", 1000)
                        
                        # Get data
                        bars = await self.get_bar_data(symbol, timeframe, start_time, history)
                        
                        # Send response
                        await websocket.send(json.dumps(bars))
                    else:
                        logger.warning(f"Invalid request from client {client_id}: {request}")
                        await websocket.send(json.dumps({"error": "Invalid request"}))
                
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Client {client_id} disconnected")
                    break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error handling client {client_id}: {e}")
                    logger.debug(traceback.format_exc())
                    try:
                        await websocket.send(json.dumps({"error": str(e)}))
                    except:
                        pass
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client {client_id} connection closed")
    
    async def start(self):
        """Start the WebSocket server"""
        # Connect to MT5
        connected = await self.connect_mt5()
        if not connected:
            logger.error("Failed to connect to MT5, server will not start")
            return False
        
        # Start WebSocket server
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=None
            )
            
            logger.info(f"MT5 WebSocket server started on {self.host}:{self.port}")
            
            # Keep the server running
            while not self._shutdown_requested:
                await asyncio.sleep(1)
            
            logger.info("Shutdown flag detected, closing server...")
            return True
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            logger.debug(traceback.format_exc())
            return False
        finally:
            await self.shutdown()


async def main():
    # FIXME: use console !
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="MT5 WebSocket Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind the WebSocket server to")
    parser.add_argument("--port", type=int, default=8765,
                      help="Port to bind the WebSocket server to")
    parser.add_argument("--cache_ttl", type=int, default=300,
                      help="Time to live in seconds for cache items")
    parser.add_argument("--cache_size", type=int, default=100,
                      help="Maximum number of items to store in cache")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create and start server
    server = MT5Server(
        host=args.host,
        port=args.port,
        cache_ttl=args.cache_ttl,
        cache_size=args.cache_size
    )
    
    # Start the server and run until interrupted
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1) 