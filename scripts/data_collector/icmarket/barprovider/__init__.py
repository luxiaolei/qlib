#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bar provider implementations for ICMarket data.

This module provides base classes and implementations for accessing
time series data (bars) from various sources, including MT5.
"""

from pathlib import Path

# Make the directory path available
BARPROVIDER_DIR = Path(__file__).resolve().parent
ICMARKET_DIR = BARPROVIDER_DIR.parent

# Import the main classes if available
try:
    from ._base import Bar, BarProvider
    from ._mt5barprovider import MT5BarProvider
    
    __all__ = ["BarProvider", "Bar", "MT5BarProvider"]
except ImportError:
    # MT5 dependencies might not be installed
    __all__ = [] 