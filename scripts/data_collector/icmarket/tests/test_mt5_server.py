#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for MT5 server integration.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import websockets
from rich.console import Console

# Add parent directory to path to import qlib modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add icmarket directory to path
icmarket_dir = Path(__file__).resolve().parent.parent
if str(icmarket_dir) not in sys.path:
    sys.path.append(str(icmarket_dir))

# Import the module to test with mocked dependencies
with patch.dict('sys.modules', {'MetaTrader5': MagicMock()}):
    import MetaTrader5 as mt5
    from core.mt5_server import DataCache, MT5Server, Request, get_symbol_info

console = Console()

# Mock data for tests
MOCK_TERMINAL_INFO = MagicMock()
MOCK_TERMINAL_INFO.build = "1234"
MOCK_TERMINAL_INFO.path = "C:\\Program Files\\MetaTrader 5"

MOCK_ACCOUNT_INFO = MagicMock()
MOCK_ACCOUNT_INFO.login = "12345"
MOCK_ACCOUNT_INFO.server = "ICMarkets-Demo"

MOCK_SYMBOL_INFO = MagicMock()
MOCK_SYMBOL_INFO.visible = True

MOCK_BAR_DATA = [
    {
        "time": int(datetime.now().timestamp()) - 3600,
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
        "tick_volume": 100,
        "spread": 2,
        "real_volume": 0
    },
    {
        "time": int(datetime.now().timestamp()) - 3540,
        "open": 1.1005,
        "high": 1.1015,
        "low": 1.0995,
        "close": 1.1010,
        "tick_volume": 120,
        "spread": 2,
        "real_volume": 0
    }
]


class TestDataCache(unittest.TestCase):
    """Test case for DataCache"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = DataCache(max_size=2, ttl=5)
    
    def test_set_get(self):
        """Test setting and getting items from cache"""
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertIsNone(self.cache.get("key2"))
    
    def test_expiration(self):
        """Test cache item expiration"""
        self.cache.set("key1", "value1")
        
        # Override the timestamp to make the item expired
        self.cache.cache["key1"]["timestamp"] = datetime.now().timestamp() - 10
        
        self.assertIsNone(self.cache.get("key1"))
        self.assertNotIn("key1", self.cache.cache)
    
    def test_max_size(self):
        """Test cache max size limitation"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Setting a third item should remove the oldest one
        self.cache.set("key3", "value3")
        
        self.assertNotIn("key1", self.cache.cache)
        self.assertIn("key2", self.cache.cache)
        self.assertIn("key3", self.cache.cache)
    
    def test_clear(self):
        """Test clearing the cache"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.clear()
        
        self.assertNotIn("key1", self.cache.cache)
        self.assertNotIn("key2", self.cache.cache)
        self.assertEqual(len(self.cache.cache), 0)


class TestMT5Server(unittest.TestCase):
    """Test case for MT5Server"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a server instance
        self.server = MT5Server(host="localhost", port=8765, cache_ttl=5, cache_size=10)
        
        # Mock MT5 API functions
        mt5.initialize.return_value = True
        mt5.terminal_info.return_value = MOCK_TERMINAL_INFO
        mt5.account_info.return_value = MOCK_ACCOUNT_INFO
        mt5.symbol_info.return_value = MOCK_SYMBOL_INFO
        mt5.symbol_select.return_value = True
        
        # Create mock bar data that matches the MT5 API return format
        import numpy as np
        self.mock_bars = np.array(
            [(int(datetime.now().timestamp()) - 3600, 1.1000, 1.1010, 1.0990, 1.1005, 100, 2, 0),
             (int(datetime.now().timestamp()) - 3540, 1.1005, 1.1015, 1.0995, 1.1010, 120, 2, 0)],
            dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), 
                   ('close', '<f8'), ('tick_volume', '<i8'), ('spread', '<i8'), ('real_volume', '<i8')]
        )
        
        mt5.copy_rates_from_pos.return_value = self.mock_bars
        mt5.copy_rates_from.return_value = self.mock_bars
    
    async def test_connect_mt5(self):
        """Test connecting to MT5"""
        result = await self.server.connect_mt5()
        
        # Verify the result and function calls
        self.assertTrue(result)
        mt5.initialize.assert_called_once()
        mt5.terminal_info.assert_called()
        mt5.account_info.assert_called_once()
    
    async def test_connect_mt5_failure(self):
        """Test connecting to MT5 with failure"""
        # Configure mock to simulate failure
        mt5.initialize.return_value = False
        
        result = await self.server.connect_mt5()
        
        # Verify the result
        self.assertFalse(result)
        mt5.initialize.assert_called_once()
    
    async def test_get_bar_data(self):
        """Test getting bar data from MT5"""
        result = await self.server.get_bar_data("EURUSD", "M1", history=10)
        
        # Verify the result
        self.assertEqual(len(result), len(self.mock_bars))
        self.assertEqual(result[0]["open"], float(self.mock_bars[0]["open"]))
        self.assertEqual(result[0]["close"], float(self.mock_bars[0]["close"]))
        
        # Verify function calls
        mt5.symbol_info.assert_called_once_with("EURUSD")
        mt5.copy_rates_from_pos.assert_called_once()
    
    async def test_get_bar_data_with_start_time(self):
        """Test getting bar data with start time"""
        start_time = datetime.now() - timedelta(days=1)
        result = await self.server.get_bar_data("EURUSD", "M1", start_time=start_time, history=10)
        
        # Verify the result
        self.assertEqual(len(result), len(self.mock_bars))
        
        # Verify function calls
        mt5.copy_rates_from.assert_called_once_with("EURUSD", mt5.TIMEFRAME_M1, start_time, 10)
    
    async def test_get_bar_data_invalid_timeframe(self):
        """Test getting bar data with invalid timeframe"""
        result = await self.server.get_bar_data("EURUSD", "INVALID", history=10)
        
        # Verify empty result due to invalid timeframe
        self.assertEqual(result, [])
    
    async def test_get_bar_data_symbol_not_found(self):
        """Test getting bar data with symbol not found"""
        # Configure mock to simulate symbol not found
        mt5.symbol_info.return_value = None
        
        result = await self.server.get_bar_data("INVALID", "M1", history=10)
        
        # Verify empty result due to symbol not found
        self.assertEqual(result, [])
    
    async def test_get_bar_data_cache(self):
        """Test bar data caching"""
        # First call should hit MT5 API
        result1 = await self.server.get_bar_data("EURUSD", "M1", history=10)
        
        # Reset the mock to verify it's not called again
        mt5.copy_rates_from_pos.reset_mock()
        
        # Second call should use cache
        result2 = await self.server.get_bar_data("EURUSD", "M1", history=10)
        
        # Verify results are the same
        self.assertEqual(result1, result2)
        
        # Verify MT5 API not called for second request
        mt5.copy_rates_from_pos.assert_not_called()
    
    @patch('websockets.serve')
    async def test_handle_client(self, mock_serve):
        """Test handling a client connection"""
        # Create mock client
        mock_client = AsyncMock()
        
        # Set up mock behavior
        request = {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "history": 10
        }
        mock_client.recv.return_value = json.dumps(request)
        
        # Monkeypatch the server's get_bar_data method
        original_get_bar_data = self.server.get_bar_data
        self.server.get_bar_data = AsyncMock(return_value=MOCK_BAR_DATA)
        
        try:
            # Call handle_client directly
            task = asyncio.create_task(self.server.handle_client(mock_client, "/"))
            
            # Allow some time for the handler to process the request
            await asyncio.sleep(0.1)
            
            # Simulate client disconnect by raising an exception on the second recv
            mock_client.recv.side_effect = websockets.exceptions.ConnectionClosed(1000, "Test disconnect")
            
            # Wait for handler to exit
            await task
            
            # Verify interactions
            mock_client.recv.assert_called_once()
            mock_client.send.assert_called_once()
            
            # Verify response data
            sent_data = json.loads(mock_client.send.call_args[0][0])
            self.assertEqual(sent_data, MOCK_BAR_DATA)
            
            # Verify get_bar_data was called with correct parameters
            self.server.get_bar_data.assert_called_once_with("EURUSD", "M1", None, 10)
        finally:
            # Restore original method
            self.server.get_bar_data = original_get_bar_data


# Helper function to run async tests
def run_async_test(coroutine):
    """Run an async test coroutine"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Main test runner
if __name__ == "__main__":
    console.print("[bold blue]Running MT5 Server Tests[/bold blue]")
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataCache)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMT5Server))
    
    # Patch asyncio.run to use our run_async_test function for async methods
    original_run = asyncio.run
    asyncio.run = run_async_test
    
    try:
        # Run the tests
        result = unittest.TextTestRunner().run(suite)
        
        if result.wasSuccessful():
            console.print("[bold green]All tests passed![/bold green]")
            sys.exit(0)
        else:
            console.print("[bold red]Some tests failed![/bold red]")
            sys.exit(1)
    finally:
        # Restore original asyncio.run
        asyncio.run = original_run 