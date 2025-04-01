#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example scripts for using ICMarket data with QLib.

This directory contains examples demonstrating:
- FX data migration to QLib format
- Data health checks and fixes
- Using the data for backtesting
- Standalone usage of the data
"""

from pathlib import Path

# Make the directory path available
EXAMPLES_DIR = Path(__file__).resolve().parent
ICMARKET_DIR = EXAMPLES_DIR.parent 