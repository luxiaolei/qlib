#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified FX data health check
"""

import glob
import os
from pathlib import Path

import pandas as pd


def check_data_health(qlib_dir):
    """Check the health of FX data"""
    print(f"Checking data health in {qlib_dir}...")
    
    csv_dir = os.path.join(qlib_dir, "csv_data")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    results = {
        "missing_data": {},
        "large_price_changes": {},
        "negative_values": {},
        "volume_issues": {},
    }
    
    for csv_file in csv_files:
        symbol = Path(csv_file).stem
        print(f"Checking {symbol}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check for missing data
            missing = df.isnull().sum()
            if missing.sum() > 0:
                results["missing_data"][symbol] = missing.to_dict()
            
            # Check for abnormal price changes (>10%)
            price_changes = df['change'].abs()
            large_changes = price_changes[price_changes > 0.1]
            if not large_changes.empty:
                results["large_price_changes"][symbol] = len(large_changes)
                
            # Check for negative prices
            for col in ['open', 'high', 'low', 'close']:
                negatives = df[df[col] < 0]
                if not negatives.empty:
                    if symbol not in results["negative_values"]:
                        results["negative_values"][symbol] = {}
                    results["negative_values"][symbol][col] = len(negatives)
            
            # Check for zero or negative volume
            volume_issues = df[df['volume'] <= 0]
            if not volume_issues.empty:
                results["volume_issues"][symbol] = len(volume_issues)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Print summary report
    print("\n=== DATA HEALTH REPORT ===")
    
    if not results["missing_data"]:
        print("✅ No missing data found")
    else:
        print("❌ Missing data found in the following symbols:")
        for symbol, missing in results["missing_data"].items():
            print(f"  - {symbol}: {missing}")
    
    if not results["large_price_changes"]:
        print("✅ No abnormal price changes found")
    else:
        print("⚠️ Large price changes (>10%) found:")
        for symbol, count in results["large_price_changes"].items():
            print(f"  - {symbol}: {count} instances")
    
    if not results["negative_values"]:
        print("✅ No negative prices found")
    else:
        print("❌ Negative prices found:")
        for symbol, cols in results["negative_values"].items():
            print(f"  - {symbol}: {cols}")
    
    if not results["volume_issues"]:
        print("✅ No volume issues found")
    else:
        print("⚠️ Zero or negative volume found:")
        for symbol, count in results["volume_issues"].items():
            print(f"  - {symbol}: {count} instances")
    
    # Overall assessment
    issues_count = sum(len(r) for r in results.values())
    if issues_count == 0:
        print("\n✅ OVERALL: Data appears to be healthy")
    else:
        print(f"\n⚠️ OVERALL: Found {issues_count} types of issues")

    return results

if __name__ == "__main__":
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/icm_fx_m5")
    check_data_health(qlib_dir) 