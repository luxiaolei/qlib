#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to migrate FX M5 data to QLib format.
"""

import argparse
import glob
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (
    ALL_CATEGORIES,
    COMMODITY_PAIRS,
    DATE_FIELD_NAME,
    DEFAULT_FX_DATA_PATH,
    EXOTIC_PAIRS,
    MAJOR_PAIRS,
    MINOR_PAIRS,
    SYMBOL_FIELD_NAME,
)

console = Console()

def process_csv_file(
    file_path: str, 
    output_dir: str,
    symbol: Optional[str] = None
) -> Tuple[Optional[str], int]:
    """
    Process a single CSV file from source format to QLib format
    
    Args:
        file_path: Path to the source CSV file
        output_dir: Directory to save the processed file
        symbol: Symbol name (if different from filename)
        
    Returns:
        Tuple of (symbol, number of rows processed)
    """
    try:
        # Extract symbol from filename if not provided
        if symbol is None:
            symbol = Path(file_path).stem
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Rename columns to match QLib format
        df.rename(columns={
            'time': DATE_FIELD_NAME,
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        }, inplace=True)
        
        # Ensure datetime format is correct
        df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])
        
        # Sort by date (QLib requirement)
        df = df.sort_values(DATE_FIELD_NAME)
        
        # Add missing columns required by QLib
        # Use volume=1 instead of 0 as specified in the requirements
        df['volume'] = 1  
        df['factor'] = 1.0  # No adjustment factor for FX
        
        # Calculate the change (percentage change from previous close)
        df['change'] = df['close'].pct_change().fillna(0)
        
        # Add symbol column
        df[SYMBOL_FIELD_NAME] = symbol
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to the output directory
        output_file = os.path.join(output_dir, f"{symbol}.csv")
        df.to_csv(output_file, index=False)
        
        return symbol, len(df)
    except Exception as e:
        console.print(f"[bold red]Error processing {file_path}: {str(e)}[/bold red]")
        return None, 0

def create_instruments_files(
    qlib_dir: Union[str, Path],
    symbols_by_category: Optional[Dict[str, List[str]]] = None
) -> None:
    """
    Create instrument files for each category in the QLib format
    
    Args:
        qlib_dir: QLib data directory
        symbols_by_category: Dict mapping category names to lists of symbols
                            If None, use ALL_CATEGORIES from config
    """
    if symbols_by_category is None:
        symbols_by_category = ALL_CATEGORIES
    
    instruments_dir = os.path.join(qlib_dir, "instruments")
    os.makedirs(instruments_dir, exist_ok=True)
    
    # Create all.txt with all symbols
    all_symbols = []
    for symbols in symbols_by_category.values():
        all_symbols.extend(symbols)
    
    # Remove duplicates and sort
    all_symbols = sorted(list(set(all_symbols)))
    
    # Write all.txt
    with open(os.path.join(instruments_dir, "all.txt"), 'w') as f:
        for symbol in all_symbols:
            f.write(f"{symbol}\n")
    
    # Write category-specific files
    for category, symbols in symbols_by_category.items():
        with open(os.path.join(instruments_dir, f"{category}.txt"), 'w') as f:
            for symbol in sorted(symbols):
                f.write(f"{symbol}\n")
    
    console.print(f"[green]Created instrument files in {instruments_dir}[/green]")

def create_calendar(
    csv_dir: Union[str, Path],
    qlib_dir: Union[str, Path],
    start_year: int = 2000,
    end_year: int = 2030
) -> None:
    """
    Create calendar files for FX data
    
    Args:
        csv_dir: Directory with normalized CSV files
        qlib_dir: QLib data directory
        start_year: Start year for calendar
        end_year: End year for calendar
    """
    calendars_dir = os.path.join(qlib_dir, "calendars")
    os.makedirs(calendars_dir, exist_ok=True)
    
    # Get all available dates from the CSV files
    all_dates = set()
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        console.print("[yellow]Warning: No CSV files found for calendar creation[/yellow]")
        return
    
    # Process files to get trading dates
    for file_path in csv_files[:5]:  # Use up to 5 files to get trading dates
        try:
            df = pd.read_csv(file_path)
            df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])
            dates = pd.DatetimeIndex(df[DATE_FIELD_NAME].dt.date).unique()
            all_dates.update(dates)
        except Exception as e:
            console.print(f"[yellow]Warning: Error reading {file_path} for calendar: {str(e)}[/yellow]")
    
    # Convert to sorted list
    all_dates = sorted(list(all_dates))
    
    if not all_dates:
        console.print("[yellow]Warning: No dates extracted for calendar creation[/yellow]")
        # Generate a default 24/5 calendar
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter weekends
        all_dates = [d for d in all_dates if d.weekday() < 5]  # Monday to Friday
    
    # Create the M5 calendar
    # First create daily calendar
    with open(os.path.join(calendars_dir, "day.txt"), 'w') as f:
        for date in all_dates:
            f.write(f"{pd.Timestamp(date).strftime('%Y-%m-%d')} 00:00:00\n")
    
    # Generate M5 calendar
    intraday_dates = []
    for date in all_dates:
        day_start = pd.Timestamp(date)
        # Generate timestamps for 24-hour FX trading (except weekends)
        timestamps = pd.date_range(
            start=day_start, 
            periods=24*60//5,  # 288 5-minute bars per day 
            freq='5min'
        )
        intraday_dates.extend(timestamps)
    
    # Write to file for 5min
    with open(os.path.join(calendars_dir, "5t.txt"), 'w') as f:
        for timestamp in intraday_dates:
            f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    console.print(f"[green]Created calendar files in {calendars_dir}[/green]")

def migrate_fx_m5_data(
    source_dir: Union[str, Path],
    qlib_dir: Optional[Union[str, Path]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_workers: int = 8
) -> None:
    """
    Migrate FX M5 data to QLib format
    
    Args:
        source_dir: Source directory with raw CSV files
                   Either a single directory with all CSV files or a directory with year subdirectories
        qlib_dir: QLib data directory, if None, use DEFAULT_FX_DATA_PATH
        start_year: Start year for data migration, if None, infer from data
        end_year: End year for data migration, if None, infer from data
        max_workers: Number of parallel workers
    """
    # Configure paths
    source_dir = os.path.expanduser(source_dir)
    if qlib_dir is None:
        qlib_dir = DEFAULT_FX_DATA_PATH
    qlib_dir = os.path.expanduser(qlib_dir)
    
    # Create QLib directory if it doesn't exist
    os.makedirs(qlib_dir, exist_ok=True)
    
    # Output directory for normalized CSV files
    csv_data_dir = os.path.join(qlib_dir, "csv_data")
    os.makedirs(csv_data_dir, exist_ok=True)
    
    console.print(f"[bold]Migrating FX M5 data from {source_dir} to {qlib_dir}[/bold]")
    
    # Check if source_dir contains year directories
    year_dirs = sorted([d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d)) and d.isdigit()])
    
    # Handle TDS directory structure with year subdirectories
    if year_dirs:
        console.print(f"[bold]TDS directory structure detected with {len(year_dirs)} year folders[/bold]")
        
        # Filter years if specified
        if start_year is not None:
            year_dirs = [y for y in year_dirs if int(y) >= start_year]
        if end_year is not None:
            year_dirs = [y for y in year_dirs if int(y) <= end_year]
        
        console.print(f"Processing years: {', '.join(year_dirs)}")
        
        # First, create a list of all symbols across all years
        all_symbols = set()
        for year_dir in year_dirs:
            year_path = os.path.join(source_dir, year_dir)
            csv_files = glob.glob(os.path.join(year_path, "*.csv"))
            for file_path in csv_files:
                symbol = Path(file_path).stem
                all_symbols.add(symbol)
        
        # Process each symbol by combining data from all available years
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing symbols...", total=len(all_symbols))
            
            # Categorize symbols
            symbols_by_category = {
                "major": [],
                "minor": [],
                "exotic": [],
                "commodity": []
            }
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for symbol in all_symbols:
                    # Create a combined dataframe for this symbol from all years
                    combined_df = pd.DataFrame()
                    
                    # Track if any files were processed for this symbol
                    symbol_processed = False
                    
                    for year_dir in year_dirs:
                        file_path = os.path.join(source_dir, year_dir, f"{symbol}.csv")
                        if os.path.exists(file_path):
                            try:
                                # Read the CSV file for this year
                                df = pd.read_csv(file_path)
                                
                                # Skip empty dataframes
                                if df.empty:
                                    continue
                                
                                # Rename columns to match QLib format
                                df.rename(columns={
                                    'time': DATE_FIELD_NAME,
                                }, inplace=True)
                                
                                # Ensure datetime format is correct
                                df[DATE_FIELD_NAME] = pd.to_datetime(df[DATE_FIELD_NAME])
                                
                                # Append to combined dataframe
                                combined_df = pd.concat([combined_df, df], ignore_index=True)
                                symbol_processed = True
                                
                            except Exception as e:
                                console.print(f"[yellow]Error reading {file_path}: {str(e)}[/yellow]")
                    
                    if symbol_processed and not combined_df.empty:
                        # Sort by date
                        combined_df = combined_df.sort_values(DATE_FIELD_NAME)
                        
                        # Remove duplicates (if any)
                        combined_df = combined_df.drop_duplicates(subset=[DATE_FIELD_NAME])
                        
                        # Save the combined dataframe to a temporary file
                        temp_dir = os.path.join(qlib_dir, "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_file = os.path.join(temp_dir, f"{symbol}.csv")
                        combined_df.to_csv(temp_file, index=False)
                        
                        # Process the combined file
                        futures.append(
                            executor.submit(
                                process_csv_file, 
                                file_path=temp_file, 
                                output_dir=csv_data_dir,
                                symbol=symbol
                            )
                        )
                        
                        # Categorize the symbol
                        if symbol in MAJOR_PAIRS:
                            symbols_by_category["major"].append(symbol)
                        elif symbol in MINOR_PAIRS:
                            symbols_by_category["minor"].append(symbol)
                        elif symbol in EXOTIC_PAIRS:
                            symbols_by_category["exotic"].append(symbol)
                        elif symbol in COMMODITY_PAIRS:
                            symbols_by_category["commodity"].append(symbol)
                
                # Wait for all futures to complete
                for future in as_completed(futures):
                    symbol, num_rows = future.result()
                    if symbol:
                        console.print(f"Processed {symbol}: {num_rows} rows")
                    progress.update(task, advance=1)
            
            # Clean up temporary directory
            if os.path.exists(os.path.join(qlib_dir, "temp")):
                shutil.rmtree(os.path.join(qlib_dir, "temp"))
                
    else:
        # Single directory with all CSV files
        console.print(f"[bold]Standard directory structure detected[/bold]")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
        
        if not csv_files:
            console.print(f"[red]No CSV files found in {source_dir}[/red]")
            return
        
        console.print(f"Found {len(csv_files)} CSV files")
        
        # Process each file in parallel
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]Processing files...", total=len(csv_files))
            
            # Categorize symbols
            symbols_by_category = {
                "major": [],
                "minor": [],
                "exotic": [],
                "commodity": []
            }
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for file_path in csv_files:
                    symbol = Path(file_path).stem
                    futures.append(executor.submit(process_csv_file, file_path, csv_data_dir))
                    
                    # Categorize the symbol
                    if symbol in MAJOR_PAIRS:
                        symbols_by_category["major"].append(symbol)
                    elif symbol in MINOR_PAIRS:
                        symbols_by_category["minor"].append(symbol)
                    elif symbol in EXOTIC_PAIRS:
                        symbols_by_category["exotic"].append(symbol)
                    elif symbol in COMMODITY_PAIRS:
                        symbols_by_category["commodity"].append(symbol)
                
                # Wait for all futures to complete
                for future in as_completed(futures):
                    symbol, num_rows = future.result()
                    if symbol:
                        console.print(f"Processed {symbol}: {num_rows} rows")
                    progress.update(task, advance=1)
    
    # Create instrument files
    create_instruments_files(qlib_dir, symbols_by_category)
    
    # Set default years if not specified
    start_year_val = start_year if start_year is not None else 2000
    end_year_val = end_year if end_year is not None else 2030
    
    # Create calendar files
    create_calendar(csv_data_dir, qlib_dir, start_year_val, end_year_val)
    
    console.print(f"[bold green]✓ Completed migration to {qlib_dir}[/bold green]")
    
    # Print next steps
    console.print(
        "\n[bold]Next steps:[/bold]"
        "\n1. Verify the data in the csv_data directory"
        "\n2. Check for data quality issues with scripts/data_collector/icmarket/examples/check_fx_data_health.py"
        "\n3. Run the dump_bin.py script to convert to binary format:"
        f"\n   python scripts/dump_bin.py dump_all --csv_path {csv_data_dir} --qlib_dir {qlib_dir}"
        "\n4. Initialize QLib with the FX data"
        f"\n   qlib.init(provider_uri='{qlib_dir}')"
    )

def main():
    """Main function that parses command line arguments and runs the migration"""
    parser = argparse.ArgumentParser(description="Migrate FX M5 data to QLib format")
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True,
        help="Source directory with FX M5 CSV files"
    )
    
    parser.add_argument(
        "--qlib_dir", 
        type=str, 
        default=DEFAULT_FX_DATA_PATH,
        help=f"Target QLib data directory (default: {DEFAULT_FX_DATA_PATH})"
    )
    
    parser.add_argument(
        "--start_year", 
        type=int, 
        default=None,
        help="Start year for filtering data (optional)"
    )
    
    parser.add_argument(
        "--end_year", 
        type=int, 
        default=None,
        help="End year for filtering data (optional)"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    migrate_fx_m5_data(
        source_dir=args.source_dir,
        qlib_dir=args.qlib_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main() 