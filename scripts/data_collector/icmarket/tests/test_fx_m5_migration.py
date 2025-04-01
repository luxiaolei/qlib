#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for FX M5 data migration.
This script verifies that the data migration process works correctly.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Add qlib parent directory to system path
repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import COMMODITY_PAIRS, EXOTIC_PAIRS, MAJOR_PAIRS, MINOR_PAIRS
from migrate_fx_m5 import create_calendar, create_instruments_files, process_csv_file


class TestFXM5Migration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data for migration tests"""
        # Create temporary directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.source_dir = Path(cls.temp_dir) / "source"
        cls.qlib_dir = Path(cls.temp_dir) / "qlib_data"
        
        os.makedirs(cls.source_dir, exist_ok=True)
        os.makedirs(cls.qlib_dir, exist_ok=True)
        
        # Create sample FX M5 data for testing
        cls._create_sample_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data"""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample FX data for testing"""
        # Sample data for major, minor, exotic, and commodity pairs
        test_pairs = {
            "EURUSD": MAJOR_PAIRS,  # Major
            "EURGBP": MINOR_PAIRS,  # Minor
            "USDSGD": EXOTIC_PAIRS,  # Exotic
            "XAUUSD": COMMODITY_PAIRS  # Commodity
        }
        
        # Create sample data for each test pair
        for pair, category in test_pairs.items():
            # Simple sample data with 100 rows of M5 data
            dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
            data = {
                "time": dates,
                "open": [1.0 + i * 0.001 for i in range(100)],
                "high": [1.0 + i * 0.001 + 0.0005 for i in range(100)],
                "low": [1.0 + i * 0.001 - 0.0005 for i in range(100)],
                "close": [1.0 + i * 0.001 + 0.0002 for i in range(100)]
            }
            
            df = pd.DataFrame(data)
            
            # Save to source directory
            csv_path = cls.source_dir / f"{pair}.csv"
            df.to_csv(csv_path, index=False)
    
    def test_process_csv_file(self):
        """Test processing of a single CSV file"""
        # Process the EURUSD sample file
        output_dir = Path(self.temp_dir) / "output"
        os.makedirs(output_dir, exist_ok=True)
        
        source_file = self.source_dir / "EURUSD.csv"
        symbol, num_rows = process_csv_file(
            file_path=str(source_file),
            output_dir=str(output_dir)
        )
        
        # Check results
        self.assertEqual(symbol, "EURUSD")
        self.assertEqual(num_rows, 100)
        
        # Check that the output file exists
        output_file = output_dir / "EURUSD.csv"
        self.assertTrue(output_file.exists())
        
        # Read the processed file and check its structure
        df = pd.read_csv(output_file)
        self.assertIn("volume", df.columns)
        self.assertIn("factor", df.columns)
        self.assertIn("change", df.columns)
        
        # Check that volume is 1
        self.assertTrue((df["volume"] == 1).all())
        # Check that factor is 1.0
        self.assertTrue((df["factor"] == 1.0).all())
    
    def test_create_instruments_files(self):
        """Test the creation of instrument files"""
        # Define the symbols by category
        symbols_by_category = {
            "major": ["EURUSD", "USDJPY"],
            "minor": ["EURGBP", "GBPJPY"],
            "exotic": ["USDSGD", "EURPLN"],
            "commodity": ["XAUUSD", "XAGUSD"]
        }
        
        # Create instrument files
        create_instruments_files(self.qlib_dir, symbols_by_category)
        
        # Check that the instrument files exist
        instruments_dir = self.qlib_dir / "instruments"
        self.assertTrue(instruments_dir.exists())
        
        # Check all category files
        for category in ["all", "major", "minor", "exotic", "commodity"]:
            file_path = instruments_dir / f"{category}.txt"
            self.assertTrue(file_path.exists())
            
            # Check file content
            with open(file_path, "r") as f:
                lines = f.readlines()
                if category == "all":
                    # Should contain all symbols
                    all_symbols = []
                    for symbols in symbols_by_category.values():
                        all_symbols.extend(symbols)
                    self.assertEqual(len(lines), len(set(all_symbols)))
                else:
                    # Should contain only symbols for this category
                    self.assertEqual(len(lines), len(symbols_by_category[category]))
    
    def test_create_calendar(self):
        """Test the creation of calendar files"""
        # First create some normalized CSV data
        csv_dir = Path(self.temp_dir) / "csv_data"
        os.makedirs(csv_dir, exist_ok=True)
        
        # Create a sample CSV file with data spanning multiple days
        dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="5min")
        data = {
            "date": dates,
            "open": [1.0] * len(dates),
            "high": [1.1] * len(dates),
            "low": [0.9] * len(dates),
            "close": [1.05] * len(dates),
            "volume": [1] * len(dates),
            "symbol": ["EURUSD"] * len(dates)
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_dir / "EURUSD.csv", index=False)
        
        # Create calendar files
        create_calendar(csv_dir, self.qlib_dir, 2023, 2023)
        
        # Check that the calendar files exist
        calendars_dir = self.qlib_dir / "calendars"
        self.assertTrue(calendars_dir.exists())
        
        # Check the day and 5t calendar files
        day_file = calendars_dir / "day.txt"
        m5_file = calendars_dir / "5t.txt"
        
        self.assertTrue(day_file.exists())
        self.assertTrue(m5_file.exists())
        
        # Check calendar content
        with open(day_file, "r") as f:
            day_lines = f.readlines()
            # Should have 10 days (Jan 1-10, 2023)
            self.assertGreaterEqual(len(day_lines), 10)
        
        with open(m5_file, "r") as f:
            m5_lines = f.readlines()
            # Each day should have 288 5-minute intervals (24*60/5)
            # So 10 days should have at least 2880 lines
            self.assertGreaterEqual(len(m5_lines), 2880)


if __name__ == "__main__":
    unittest.main() 