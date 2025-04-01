#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner script to execute all MT5 tests with proper async handling.
"""

import sys
import unittest
import asyncio
from pathlib import Path
from rich.console import Console

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

console = Console()

# Helper function to run async tests
def async_test(test_case):
    """Decorator to run async test methods"""
    def wrapper(*args, **kwargs):
        coro = test_case(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    return wrapper

def run_tests():
    """Run all MT5 tests"""
    console.print("[bold blue]Running MT5 Integration Tests[/bold blue]")
    
    # Import test modules
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from test_mt5_client import TestMT5QlibClient
    from test_mt5_server import TestMT5Server, TestDataCache
    
    # Patch async test methods
    for cls in [TestMT5QlibClient, TestMT5Server]:
        for name in dir(cls):
            if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(cls, name)):
                setattr(cls, name, async_test(getattr(cls, name)))
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDataCache))
    suite.addTest(unittest.makeSuite(TestMT5Server))
    suite.addTest(unittest.makeSuite(TestMT5QlibClient))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        console.print("[bold green]All tests passed![/bold green]")
        return 0
    else:
        console.print("[bold red]Some tests failed![/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 