#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to fix common issues in FX M5 data after migration to QLib format.
This script addresses:
1. Missing volumes (sets them to 1)
2. Missing factors (sets them to 1.0)
3. Linear interpolation for missing data points
4. Outlier detection and fixing for large price jumps
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress

# Add qlib parent directory to system path
repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

# Add parent directory to system path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATE_FIELD_NAME, DEFAULT_FX_DATA_PATH, SYMBOL_FIELD_NAME

console = Console()

def fix_csv_file(file_path, output_dir=None, outlier_threshold=0.03):
    """
    Fix common issues in a FX M5 data CSV file
    
    Args:
        file_path: Path to the CSV file to fix
        output_dir: Directory to save the fixed file (if None, overwrite original)
        outlier_threshold: Threshold for detecting abnormal price jumps
        
    Returns:
        dict: Summary of fixes applied
    """
    try:
        # Extract symbol from filename
        symbol = Path(file_path).stem
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Track fixes
        fixes = {
            "added_volume": False,
            "added_factor": False,
            "fixed_missing_data": 0,
            "fixed_outliers": 0
        }
        
        # Fix 1: Ensure volume is present and set to 1
        if 'volume' not in df.columns:
            df['volume'] = 1
            fixes["added_volume"] = True
        else:
            # Set all volume values to 1
            missing_volume = (df['volume'] <= 0).sum() + df['volume'].isna().sum()
            if missing_volume > 0:
                df['volume'] = 1
                fixes["added_volume"] = True
        
        # Fix 2: Ensure factor column is present and set to 1.0
        if 'factor' not in df.columns:
            df['factor'] = 1.0
            fixes["added_factor"] = True
        else:
            # Set all factor values to 1.0
            missing_factor = (df['factor'] != 1.0).sum() + df['factor'].isna().sum()
            if missing_factor > 0:
                df['factor'] = 1.0
                fixes["added_factor"] = True
        
        # Fix 3: Ensure date is in datetime format and sorted
        df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])
        df = df.sort_values(DATE_FIELD_NAME)
        
        # Fix 4: Linear interpolation for missing data
        # First, set index to date for easier detection of gaps
        df = df.set_index(DATE_FIELD_NAME)
        
        # Check for missing data by creating a complete range of 5-minute intervals
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
        if len(date_range) > len(df):
            # There are missing timestamps
            full_df = pd.DataFrame(index=date_range)
            df = df.join(full_df, how='right')
            
            # Interpolate missing values for price columns
            for col in ['open', 'high', 'low', 'close']:
                missing_before = df[col].isna().sum()
                df[col] = df[col].interpolate(method='linear')
                fixes["fixed_missing_data"] += missing_before - df[col].isna().sum()
            
            # Fill remaining NaNs with forward/backward fill
            df = df.fillna(value=None, method='ffill')
            df = df.fillna(value=None, method='bfill')
            
            # Ensure volume and factor are set to 1 for interpolated points
            df['volume'] = 1
            df['factor'] = 1.0
        
        # Fix 5: Detect and fix outliers (extreme price jumps)
        for col in ['open', 'high', 'low', 'close']:
            # Calculate returns
            returns = df[col].pct_change()
            
            # Find outliers
            outliers = (returns.abs() > outlier_threshold)
            
            if outliers.sum() > 0:
                # Replace outlier values with interpolated values
                outlier_indices = df.index[outliers]
                
                for idx in outlier_indices:
                    # Get neighboring values (excluding other outliers)
                    prev_values = df[col][:idx].iloc[-5:]  # Last 5 valid values
                    next_values = df[col][idx:].iloc[1:6]  # Next 5 valid values
                    
                    # Calculate median of neighboring values
                    neighbors = pd.concat([prev_values, next_values])
                    median_value = neighbors.median()
                    
                    # Replace outlier with median value
                    df.loc[idx, col] = median_value
                    fixes["fixed_outliers"] += 1
        
        # Reset index to make DATE_FIELD_NAME a column again
        df = df.reset_index()
        
        # Ensure symbol column is present
        df[SYMBOL_FIELD_NAME] = symbol
        
        # Save the fixed file
        if output_dir is None:
            df.to_csv(file_path, index=False)
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{symbol}.csv")
            df.to_csv(output_file, index=False)
        
        return {
            "symbol": symbol,
            "fixes": fixes,
            "success": True
        }
    
    except Exception as e:
        console.print(f"[bold red]Error fixing {file_path}: {str(e)}[/bold red]")
        return {
            "symbol": Path(file_path).stem,
            "success": False,
            "error": str(e)
        }

def fix_all_fx_data(data_dir, max_workers=8, outlier_threshold=0.03):
    """
    Fix all FX M5 data files in the given directory
    
    Args:
        data_dir: Directory containing QLib-formatted CSV files
        max_workers: Number of parallel workers
        outlier_threshold: Threshold for detecting abnormal price jumps
        
    Returns:
        dict: Summary of all fixes applied
    """
    # Expand and resolve path
    data_dir = Path(data_dir).expanduser().resolve()
    
    # Path to CSV data
    csv_dir = data_dir / "csv_data"
    
    if not csv_dir.exists():
        console.print(f"[bold red]CSV data directory not found: {csv_dir}[/bold red]")
        return
    
    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        console.print(f"[bold red]No CSV files found in {csv_dir}[/bold red]")
        return
    
    console.print(f"[bold green]Found {len(csv_files)} CSV files to fix[/bold green]")
    
    # Summary of fixes
    summary = {
        "total_files": len(csv_files),
        "success_count": 0,
        "error_count": 0,
        "added_volume_count": 0,
        "added_factor_count": 0,
        "fixed_missing_data_count": 0,
        "fixed_outliers_count": 0
    }
    
    # Process files in parallel
    with Progress() as progress:
        task = progress.add_task("[cyan]Fixing FX M5 data files...", total=len(csv_files))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for file_path in csv_files:
                futures.append(
                    executor.submit(
                        fix_csv_file,
                        file_path=str(file_path),
                        outlier_threshold=outlier_threshold
                    )
                )
            
            for future in futures:
                result = future.result()
                
                if result["success"]:
                    summary["success_count"] += 1
                    if result["fixes"]["added_volume"]:
                        summary["added_volume_count"] += 1
                    if result["fixes"]["added_factor"]:
                        summary["added_factor_count"] += 1
                    summary["fixed_missing_data_count"] += result["fixes"]["fixed_missing_data"]
                    summary["fixed_outliers_count"] += result["fixes"]["fixed_outliers"]
                else:
                    summary["error_count"] += 1
                
                progress.update(task, advance=1)
    
    # Print summary
    console.print("\n[bold green]===== FX M5 DATA FIX SUMMARY =====[/bold green]")
    console.print(f"Total files processed: {summary['total_files']}")
    console.print(f"Successfully fixed: {summary['success_count']}")
    console.print(f"Errors: {summary['error_count']}")
    console.print(f"Files with volume fixed: {summary['added_volume_count']}")
    console.print(f"Files with factor fixed: {summary['added_factor_count']}")
    console.print(f"Missing data points interpolated: {summary['fixed_missing_data_count']}")
    console.print(f"Outliers fixed: {summary['fixed_outliers_count']}")
    
    return summary

def main():
    """Main function to run the FX M5 data fixer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix common issues in FX M5 data after migration to QLib format")
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=DEFAULT_FX_DATA_PATH,
        help=f"QLib data directory (default: {DEFAULT_FX_DATA_PATH})"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    parser.add_argument(
        "--outlier_threshold", 
        type=float, 
        default=0.03,
        help="Threshold for detecting abnormal price jumps (default: 0.03 or 3%)"
    )
    
    args = parser.parse_args()
    
    console.print(f"[bold]Running FX M5 data fixer on {args.data_dir}[/bold]")
    
    fix_all_fx_data(
        data_dir=args.data_dir,
        max_workers=args.max_workers,
        outlier_threshold=args.outlier_threshold
    )
    
    console.print("[bold green]Data fix completed![/bold green]")
    console.print("[bold yellow]Next steps:[/bold yellow]")
    console.print("1. Run data health check: python scripts/data_collector/icmarket/examples/check_fx_data_health.py")
    console.print("2. If issues persist, adjust outlier_threshold and run this script again")

if __name__ == "__main__":
    main() 