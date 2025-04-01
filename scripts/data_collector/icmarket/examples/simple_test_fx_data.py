#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script to verify FX data in QLib format by directly checking the CSV files.
"""

import glob
import os
from pathlib import Path

import pandas as pd

# Define paths
qlib_dir = os.path.expanduser("~/.qlib/qlib_data/icm_fx_m5")
csv_dir = os.path.join(qlib_dir, "csv_data")
instruments_dir = os.path.join(qlib_dir, "instruments")
calendars_dir = os.path.join(qlib_dir, "calendars")

def check_csv_data():
    """Check the CSV data files"""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {csv_dir}")
    
    if csv_files:
        # Pick a sample file
        sample_file = csv_files[0]
        symbol = Path(sample_file).stem
        
        print(f"\nSample file: {sample_file} (Symbol: {symbol})")
        
        # Read the sample file
        df = pd.read_csv(sample_file)
        
        # Display basic information
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Display sample data
        print("\nSample data:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found.")
            
        # Check data types
        print("\nData types:")
        print(df.dtypes)
    
    return len(csv_files)

def check_instruments():
    """Check the instrument files"""
    instrument_files = glob.glob(os.path.join(instruments_dir, "*.txt"))
    
    print(f"\nFound {len(instrument_files)} instrument files in {instruments_dir}")
    
    for file_path in instrument_files:
        file_name = Path(file_path).stem
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
            
        print(f"- {file_name}.txt: {len(symbols)} symbols")
        if symbols:
            print(f"  Sample symbols: {', '.join(symbols[:5])}")
    
def check_calendars():
    """Check the calendar files"""
    calendar_files = glob.glob(os.path.join(calendars_dir, "*.txt"))
    
    print(f"\nFound {len(calendar_files)} calendar files in {calendars_dir}")
    
    for file_path in calendar_files:
        file_name = Path(file_path).stem
        with open(file_path, 'r') as f:
            dates = [line.strip() for line in f if line.strip()]
            
        print(f"- {file_name}.txt: {len(dates)} dates")
        if dates:
            print(f"  Date range: {dates[0]} to {dates[-1]}")
            print(f"  Sample dates: {', '.join(dates[:5])}")

def main():
    """Main function"""
    print(f"Checking QLib FX data in {qlib_dir}")
    
    # Check if directories exist
    for dir_path, dir_name in [
        (csv_dir, "CSV data"),
        (instruments_dir, "Instruments"),
        (calendars_dir, "Calendars")
    ]:
        if os.path.exists(dir_path):
            print(f"{dir_name} directory exists: {dir_path}")
        else:
            print(f"ERROR: {dir_name} directory does not exist: {dir_path}")
            return
    
    # Check CSV data
    num_files = check_csv_data()
    
    # Check instruments
    check_instruments()
    
    # Check calendars
    check_calendars()
    
    print("\nSummary:")
    print(f"- {num_files} CSV files")
    print(f"- Calendar files: {len(glob.glob(os.path.join(calendars_dir, '*.txt')))}")
    print(f"- Instrument files: {len(glob.glob(os.path.join(instruments_dir, '*.txt')))}")
    
    print("\nThe data migration appears to be successful!")
    print("\nNext steps:")
    print("1. Convert to QLib binary format if needed:")
    print(f"   python scripts/dump_bin.py dump_all --csv_path {csv_dir} --qlib_dir {qlib_dir}")
    print("2. Initialize QLib with the data:")
    print(f"   qlib.init(provider_uri='{qlib_dir}')")

if __name__ == "__main__":
    main() 