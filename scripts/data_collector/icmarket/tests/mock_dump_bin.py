#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mock implementation of dump_bin.py for testing."""

class DumpDataUpdate:
    """Mock implementation of DumpDataUpdate"""
    
    def __init__(self, csv_path, qlib_dir, include_fields="open,high,low,close", 
                 date_field_name="date", symbol_field_name="symbol", freq="1d"):
        """Initialize mock dumper"""
        self.csv_path = csv_path
        self.qlib_dir = qlib_dir
        self.include_fields = include_fields
        self.date_field_name = date_field_name
        self.symbol_field_name = symbol_field_name
        self.freq = freq
    
    def dump(self):
        """Mock dump method that does nothing"""
        print(f"Mock dumping CSV from {self.csv_path} to {self.qlib_dir}")
        return True 
 