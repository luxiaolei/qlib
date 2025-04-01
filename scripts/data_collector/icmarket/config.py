#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration file for FX data migration.
"""

import os

# FX Data paths
DEFAULT_FX_DATA_PATH = os.path.expanduser("~/.qlib/qlib_data/icm_fx_m5")

# Source data paths (TDS - Tick Data Suite)
TDS_FX_M5_DATA_PATH = "/Volumes/extdisk/MyRepos/EAFramework/db/market_data/TDS/M5"

# Timezone info for the data
# The TDS data is in GMT+2 with US DST adjustments
TDS_TIMEZONE = "GMT+2"
TDS_TIMEZONE_INFO = "GMT+2 with US DST adjustments"

# FX pair categories
MAJOR_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"
]

MINOR_PAIRS = [
    "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD", 
    "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY"
]

EXOTIC_PAIRS = [
    "USDSGD", "USDNOK", "USDPLN", "EURSGD", "EURPLN", "EURNOK"
]

COMMODITY_PAIRS = [
    "XAUUSD", "XAGUSD"
]

# Indices and crypto that might be in the dataset
INDICES_AND_CRYPTO = [
    "US500", "USTEC", "DE40", "BTCUSD", "ETHUSD"
]

# Group all categories
ALL_CATEGORIES = {
    "major": MAJOR_PAIRS,
    "minor": MINOR_PAIRS,
    "exotic": EXOTIC_PAIRS,
    "commodity": COMMODITY_PAIRS,
}

# Data formats
COLUMNS = ["open", "close", "high", "low", "volume"]
DATE_FIELD_NAME = "date"
SYMBOL_FIELD_NAME = "symbol" 