#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forex Symbol and Timeframe utilities for MT5 data collection.

This module provides classes for handling forex symbols and timeframes
consistently across the MT5 data collection system.
"""

from enum import Enum
from typing import Dict, List


class SymbolCategory(Enum):
    """Categories of trading symbols"""
    FOREX = "forex"
    COMMODITIES = "commodities"
    INDICES = "indices"
    STOCKS = "stocks"
    CRYPTO = "crypto"
    UNKNOWN = "unknown"


class FxSymbol:
    """
    Forex Symbol class for handling various symbol formats and categories.
    
    Features:
    - Normalizes symbol names (e.g., eur/usd -> EURUSD)
    - Categorizes symbols based on patterns
    - Provides consistent string representation
    """
    
    # Mapping of symbol patterns to categories
    CATEGORY_PATTERNS: Dict[SymbolCategory, List[str]] = {
        SymbolCategory.FOREX: [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "AUDCAD", "AUDNZD"
        ],
        SymbolCategory.COMMODITIES: [
            "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "OIL", "USOIL", "UKOIL", "CORNF",
            "NATGAS"
        ],
        SymbolCategory.INDICES: [
            "US30", "US500", "USTEC", "GER40", "UK100", "FRA40", "AUS200", "JP225",
            "HK50", "CN50", "EU50", "ESP35", "ITA40"
        ],
        SymbolCategory.CRYPTO: [
            "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BCHUSD", "ADAUSD"
        ]
    }
    
    def __init__(self, symbol: str):
        """
        Initialize a forex symbol
        
        Args:
            symbol: Symbol name in any format (e.g., "EUR/USD", "eurusd", "EURUSD")
        """
        self.original = symbol
        
        # Normalize symbol (remove /, spaces, and convert to uppercase)
        self.symbol = symbol.replace("/", "").replace(" ", "").upper()
        
        # Determine category
        self.category = self._categorize()
    
    def _categorize(self) -> SymbolCategory:
        """
        Categorize the symbol based on patterns
        
        Returns:
            SymbolCategory enum value
        """
        for category, patterns in self.CATEGORY_PATTERNS.items():
            if any(pattern in self.symbol for pattern in patterns):
                return category
        
        # Default checks for common patterns
        if len(self.symbol) == 6 and self.symbol[:3] != self.symbol[3:]:
            # Likely a forex pair (6 chars, two different 3-letter codes)
            return SymbolCategory.FOREX
        
        return SymbolCategory.UNKNOWN
    
    def __str__(self) -> str:
        """String representation of the symbol (normalized format)"""
        return self.symbol
    
    def __repr__(self) -> str:
        """Developer representation with category"""
        return f"FxSymbol({self.symbol}, {self.category.value})"
    
    def __eq__(self, other) -> bool:
        """Equal comparison"""
        if isinstance(other, FxSymbol):
            return self.symbol == other.symbol
        elif isinstance(other, str):
            return self.symbol == other.replace("/", "").replace(" ", "").upper()
        return False


class TimeFrame:
    """
    Timeframe class for handling MT5 timeframes consistently.
    
    Features:
    - Normalizes timeframe strings (e.g., "1m" -> "M1")
    - Provides properties like minutes, hours, days
    - Consistent string representation
    """
    
    # Valid timeframes
    VALID_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
    
    # Mapping from various formats to standard format
    FORMAT_MAPPING = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4",
        "1d": "D1", "d1": "D1", "daily": "D1",
        "1w": "W1", "w1": "W1", "weekly": "W1",
        "1mn": "MN1", "mn1": "MN1", "monthly": "MN1",
        "m1": "M1", "m5": "M5", "m15": "M15", "m30": "M30",
        "h1": "H1", "h4": "H4", "d1": "D1", "w1": "W1", "mn1": "MN1"
    }
    
    def __init__(self, timeframe: str):
        """
        Initialize a timeframe
        
        Args:
            timeframe: Timeframe string in any format (e.g., "M1", "1m", "1min")
        """
        self.original = timeframe
        
        # Normalize timeframe
        self.timeframe = self._normalize(timeframe)
        
        if self.timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def _normalize(self, timeframe: str) -> str:
        """
        Normalize timeframe to standard format
        
        Args:
            timeframe: Timeframe string in any format
            
        Returns:
            Normalized timeframe string
        """
        # First check if it's already in standard format
        if timeframe in self.VALID_TIMEFRAMES:
            return timeframe
        
        # Check mapping
        norm_tf = self.FORMAT_MAPPING.get(timeframe.lower())
        if norm_tf:
            return norm_tf
        
        # Try to parse based on pattern
        tf = timeframe.lower().replace(" ", "")
        
        # Check minute patterns
        if tf.endswith("min") or tf.endswith("m"):
            value = tf.rstrip("min").rstrip("m")
            if value.isdigit():
                if value == "1":
                    return "M1"
                elif value == "5":
                    return "M5"
                elif value == "15":
                    return "M15"
                elif value == "30":
                    return "M30"
        
        # Check hour patterns
        elif tf.endswith("hour") or tf.endswith("h"):
            value = tf.rstrip("hour").rstrip("h")
            if value.isdigit():
                if value == "1":
                    return "H1"
                elif value == "4":
                    return "H4"
        
        # Check day patterns
        elif tf.endswith("day") or tf.endswith("d"):
            value = tf.rstrip("day").rstrip("d")
            if value.isdigit() and value == "1":
                return "D1"
        
        # Check week patterns
        elif tf.endswith("week") or tf.endswith("w"):
            value = tf.rstrip("week").rstrip("w")
            if value.isdigit() and value == "1":
                return "W1"
        
        # Check month patterns
        elif tf.endswith("month") or tf.endswith("mn"):
            value = tf.rstrip("month").rstrip("mn")
            if value.isdigit() and value == "1":
                return "MN1"
        
        # Default to the original if we can't normalize
        return timeframe
    
    @property
    def minutes(self) -> int:
        """Get minutes represented by the timeframe"""
        if self.timeframe == "M1":
            return 1
        elif self.timeframe == "M5":
            return 5
        elif self.timeframe == "M15":
            return 15
        elif self.timeframe == "M30":
            return 30
        elif self.timeframe == "H1":
            return 60
        elif self.timeframe == "H4":
            return 240
        elif self.timeframe == "D1":
            return 1440  # 24 * 60
        elif self.timeframe == "W1":
            return 10080  # 7 * 24 * 60
        elif self.timeframe == "MN1":
            return 43200  # 30 * 24 * 60 (approximation)
        return 0
    
    @property
    def hours(self) -> float:
        """Get hours represented by the timeframe"""
        return self.minutes / 60
    
    @property
    def days(self) -> float:
        """Get days represented by the timeframe"""
        return self.minutes / (24 * 60)
    
    def __str__(self) -> str:
        """String representation of the timeframe (normalized format)"""
        return self.timeframe
    
    def __repr__(self) -> str:
        """Developer representation with minutes"""
        return f"TimeFrame({self.timeframe}, {self.minutes}min)"
    
    def __eq__(self, other) -> bool:
        """Equal comparison"""
        if isinstance(other, TimeFrame):
            return self.timeframe == other.timeframe
        elif isinstance(other, str):
            return self.timeframe == self._normalize(other)
        return False


if __name__ == "__main__":
    # Simple tests
    for symbol in ["EURUSD", "EUR/USD", "eurusd", "XAU/USD", "US30"]:
        fx = FxSymbol(symbol)
        print(f"{symbol} -> {fx} (Category: {fx.category.value})")
    
    print()
    
    for tf in ["M1", "1m", "5min", "H1", "1hour", "D1", "daily"]:
        timeframe = TimeFrame(tf)
        print(f"{tf} -> {timeframe} ({timeframe.minutes}min, {timeframe.hours}h)")
