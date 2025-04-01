#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock module for migrate_eaframework to support testing without the actual module.
"""

from pathlib import Path
import pandas as pd
import logging

# Set up logger
logger = logging.getLogger("mock_migrate")


def create_calendar(temp_dir, qlib_dir):
    """
    Create a mock forex calendar file in the QLib directory.
    
    Args:
        temp_dir: Directory for temporary files
        qlib_dir: QLib data directory
    """
    logger.info(f"Creating mock calendar in {qlib_dir}")
    
    # Ensure calendars directory exists
    calendar_dir = Path(qlib_dir) / "calendars"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple forex calendar (24/5)
    # Start from 2020-01-01 to current date + 1 year
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp.now() + pd.DateOffset(years=1)
    
    # Create date range with business days (Monday to Friday)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    
    # Save to CSV
    calendar_file = calendar_dir / "forex_calendar.txt"
    with open(calendar_file, "w") as f:
        for date in dates:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")
    
    logger.info(f"Created mock calendar at {calendar_file}")


def create_instruments_files(qlib_dir, symbols_by_category):
    """
    Create mock instrument files in the QLib directory.
    
    Args:
        qlib_dir: QLib data directory
        symbols_by_category: Dictionary mapping categories to symbol lists
    """
    logger.info(f"Creating mock instrument files in {qlib_dir}")
    
    # Ensure instruments directory exists
    instruments_dir = Path(qlib_dir) / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    
    # Create instrument files
    for category, symbols in symbols_by_category.items():
        # All symbols file
        all_symbols_file = instruments_dir / f"{category}.txt"
        with open(all_symbols_file, "w") as f:
            for symbol in symbols:
                # Format: symbol,start_date,end_date
                f.write(f"{symbol.lower()},2020-01-01,2099-12-31\n")
        
        logger.info(f"Created instrument file for {category} with {len(symbols)} symbols")
    
    # Create a combined all.txt file
    all_symbols = []
    for symbols in symbols_by_category.values():
        all_symbols.extend(symbols)
    
    all_file = instruments_dir / "all.txt"
    with open(all_file, "w") as f:
        for symbol in all_symbols:
            f.write(f"{symbol.lower()},2020-01-01,2099-12-31\n")
    
    logger.info(f"Created all.txt with {len(all_symbols)} symbols")


def regenerate_symbols_info(symbols):
    """
    Mock function for regenerating symbols info.
    
    Args:
        symbols: List of symbols
    """
    logger.info(f"Mock regenerating symbol info for {len(symbols)} symbols")
    return 