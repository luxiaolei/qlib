#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICMarket data collector for QLib.

This module provides tools for:
1. Migrating FX data to QLib format
2. Integrating with MetaTrader 5 for data collection
3. Creating calendar and instrument files

Directory structure:
- core/: MT5 integration and server modules
- examples/: Example scripts for data usage
- tests/: Unit tests
- barprovider/: Bar provider implementations
"""

from pathlib import Path

# Make the directory paths available
MODULE_DIR = Path(__file__).resolve().parent
CORE_DIR = MODULE_DIR / "core"
EXAMPLES_DIR = MODULE_DIR / "examples"
BARPROVIDER_DIR = MODULE_DIR / "barprovider"
TEST_DIR = MODULE_DIR / "tests"

# Import commonly used modules
from .config import (
    COMMODITY_PAIRS,
    EXOTIC_PAIRS,
    MAJOR_PAIRS,
    MINOR_PAIRS,
)

# Import the main migration function
from .migrate_fx_m5 import migrate_fx_m5_data

# For backwards compatibility
try:
    from .core.mt5_qlib_client import MT5QlibClient
    from .core.mt5_server import MT5Server
except ImportError:
    pass  # MT5 dependencies may not be installed

__all__ = ["migrate_fx_m5_data", "MAJOR_PAIRS", "MINOR_PAIRS", "EXOTIC_PAIRS", "COMMODITY_PAIRS"]

