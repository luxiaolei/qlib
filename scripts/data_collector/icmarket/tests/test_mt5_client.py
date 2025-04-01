#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for MT5 QLib Client
This test mocks the websocket server to verify client functionality.
"""

import sys
import json
import asyncio
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console

# Add parent directory to path to import qlib modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the module to test
from scripts.data_collector.icmarket.mt5_qlib_client import MT5QlibClient

console = Console()

# Mock data for tests
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

class AsyncContextManagerMock(AsyncMock):
    """Mock for async context manager (for websockets.connect)"""
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, *args):
        pass

class TestMT5QlibClient(unittest.TestCase):
    """Test case for MT5QlibClient"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for qlib data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.qlib_dir = Path(self.temp_dir.name) / "qlib_data"
        self.qlib_dir.mkdir(parents=True)
        
        # Create features directory for test symbol
        self.features_dir = self.qlib_dir / "features" / "eurusd"
        self.features_dir.mkdir(parents=True)
        
        # Create a test bin file (close.1t.bin)
        self.create_test_bin_file(self.features_dir / "close.1t.bin")
        
        # Create instance of client with mocked server
        self.client = MT5QlibClient(
            qlib_dir=str(self.qlib_dir),
            server_url="ws://localhost:8765",
            symbols=["EURUSD"],
            timeframes=["M1"],
            download_days=1,
            update_interval=1,
            retry_wait=1,
            retry_max=1
        )
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.temp_dir.cleanup()
    
    def create_test_bin_file(self, file_path, timestamp=None):
        """Create a test binary file for QLib format"""
        if timestamp is None:
            timestamp = datetime.now() - timedelta(minutes=10)
        
        # Create a simple array with timestamp and value pairs
        # QLib binary files are stored as float32 values
        # First entry is date index (timestamp), second is value
        timestamp_float = timestamp.timestamp()
        data = np.array([timestamp_float, 1.1000, timestamp_float + 60, 1.1010], dtype='<f')
        data.tofile(file_path)
        
        return timestamp
    
    @patch('websockets.connect')
    async def test_get_last_timestamp(self, mock_connect):
        """Test getting the last timestamp from QLib data"""
        # The timestamp we saved in the bin file
        timestamp = datetime.now() - timedelta(minutes=10)
        self.create_test_bin_file(self.features_dir / "close.1t.bin", timestamp)
        
        # Call the method
        result = await self.client.get_last_timestamp("EURUSD", "M1")
        
        # Verify the result (needs to be close due to float32 conversion)
        self.assertIsNotNone(result)
        self.assertTrue(abs((result - pd.Timestamp(timestamp)).total_seconds()) < 1)
    
    @patch('websockets.connect')
    async def test_download_data(self, mock_connect):
        """Test downloading data from the server"""
        # Create mock websocket
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps(MOCK_BAR_DATA)
        
        # Set up the mock connect to return our mock websocket
        mock_connect.return_value = AsyncContextManagerMock()
        mock_connect.return_value.return_value = mock_ws
        
        # Call the method
        result = await self.client.download_data("EURUSD", "M1")
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(MOCK_BAR_DATA))
        self.assertTrue("open" in result.columns)
        self.assertTrue("close" in result.columns)
        
        # Verify the mock was called correctly
        mock_connect.assert_called_once()
        mock_ws.send.assert_called_once()
        data_sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(data_sent["symbol"], "EURUSD")
        self.assertEqual(data_sent["timeframe"], "M1")
    
    @patch('websockets.connect')
    async def test_save_to_csv(self, mock_connect):
        """Test saving data to CSV"""
        # Create a test DataFrame
        df = pd.DataFrame(MOCK_BAR_DATA)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        # Call the method
        csv_file = await self.client.save_to_csv(df, "EURUSD", "M1")
        
        # Verify the result
        self.assertIsNotNone(csv_file)
        self.assertTrue(csv_file.exists())
        
        # Check CSV content
        csv_df = pd.read_csv(csv_file)
        self.assertEqual(len(csv_df), len(MOCK_BAR_DATA))
        self.assertTrue("date" in csv_df.columns)
        self.assertTrue("open" in csv_df.columns)
        self.assertTrue("close" in csv_df.columns)
        self.assertTrue("factor" in csv_df.columns)
        self.assertTrue("symbol" in csv_df.columns)
        self.assertTrue("vwap" in csv_df.columns)
    
    @patch('scripts.dump_bin.DumpDataUpdate')
    async def test_dump_to_qlib(self, mock_dumper):
        """Test dumping data to QLib format"""
        # Create a test CSV file
        df = pd.DataFrame(MOCK_BAR_DATA)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        csv_file = await self.client.save_to_csv(df, "EURUSD", "M1")
        
        # Call the method
        await self.client.dump_to_qlib(csv_file, "M1")
        
        # Verify the dumper was called correctly
        mock_dumper.assert_called_once()
        args, kwargs = mock_dumper.call_args
        self.assertEqual(kwargs["csv_path"], str(self.client.temp_dir))
        self.assertEqual(kwargs["qlib_dir"], str(self.client.qlib_dir))
        self.assertEqual(kwargs["freq"], "1t")
    
    @patch.object(MT5QlibClient, 'get_last_timestamp')
    @patch.object(MT5QlibClient, 'download_data')
    @patch.object(MT5QlibClient, 'save_to_csv')
    @patch.object(MT5QlibClient, 'dump_to_qlib')
    async def test_update_symbol_timeframe(self, mock_dump, mock_save, mock_download, mock_timestamp):
        """Test updating symbol timeframe data"""
        # Mock return values
        mock_timestamp.return_value = datetime.now() - timedelta(minutes=10)
        
        # Create mock dataframe
        df = pd.DataFrame(MOCK_BAR_DATA)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        mock_download.return_value = df
        
        # Mock CSV file
        mock_save.return_value = Path("/tmp/test.csv")
        
        # Call the method
        await self.client.update_symbol_timeframe("EURUSD", "M1")
        
        # Verify the methods were called correctly
        mock_timestamp.assert_called_once_with("EURUSD", "M1")
        mock_download.assert_called_once()
        mock_save.assert_called_once_with(df, "EURUSD", "M1")
        # Note: dump_to_qlib is called via queue, so we can't easily verify it here
    
    @patch('asyncio.create_task')
    @patch.object(MT5QlibClient, 'setup_instruments')
    @patch.object(MT5QlibClient, 'process_dump_queue')
    @patch.object(MT5QlibClient, 'monitor_symbol_timeframe')
    async def test_start(self, mock_monitor, mock_process, mock_setup, mock_create_task):
        """Test starting the client"""
        # Set up task mocks
        task_mock = MagicMock()
        task_mock.done.return_value = False
        mock_create_task.return_value = task_mock
        
        # Start the client but run only briefly
        client_task = asyncio.create_task(self.client.start())
        await asyncio.sleep(0.1)  # Allow some time for startup
        client_task.cancel()  # Cancel to exit the infinite loop
        
        try:
            await client_task
        except asyncio.CancelledError:
            pass
        
        # Verify setup was called
        mock_setup.assert_called_once()
        
        # Verify dump queue processor was started
        mock_create_task.assert_any_call(mock_process.return_value)
        
        # Verify monitor tasks were created for each symbol/timeframe
        mock_create_task.assert_any_call(mock_monitor.return_value)


# Helper function to run async tests
def run_async_test(coroutine):
    """Run an async test coroutine"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Main test runner
if __name__ == "__main__":
    console.print("[bold blue]Running MT5 QLib Client Tests[/bold blue]")
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMT5QlibClient)
    
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