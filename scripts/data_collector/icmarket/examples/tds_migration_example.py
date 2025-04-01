#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for migrating TDS M5 data to QLib format.
This shows how to use the migrate_fx_m5.py script with the TDS data structure.
"""

import os
import sys
from pathlib import Path

from rich.console import Console

# Add the parent directory to sys.path to import QLib modules
script_dir = Path(__file__).resolve().parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Import the migration function and config
from config import TDS_FX_M5_DATA_PATH, TDS_TIMEZONE_INFO
from migrate_fx_m5 import migrate_fx_m5_data

console = Console()

def run_migration_example():
    """Run a complete example of migrating TDS data to QLib"""
    # Define paths and parameters
    source_dir = TDS_FX_M5_DATA_PATH
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/icm_fx_m5")
    
    # Print information about the data source
    console.print(f"[bold]TDS Data Migration Example[/bold]")
    console.print(f"Source directory: {source_dir}")
    console.print(f"Target QLib directory: {qlib_dir}")
    console.print(f"Timezone info: {TDS_TIMEZONE_INFO}\n")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        console.print(f"[bold red]Error: Source directory {source_dir} does not exist![/bold red]")
        console.print("Please update the TDS_FX_M5_DATA_PATH in config.py to point to your TDS data.")
        return
    
    # Display available years
    year_dirs = sorted([d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d)) and d.isdigit()])
    
    console.print(f"[bold]Available years:[/bold] {', '.join(year_dirs)}")
    console.print(f"[bold]Total years:[/bold] {len(year_dirs)}")
    
    # Get a sample of available symbols from the most recent year
    most_recent_year = year_dirs[-1]
    most_recent_dir = os.path.join(source_dir, most_recent_year)
    csv_files = [f for f in os.listdir(most_recent_dir) if f.endswith('.csv')]
    symbols = [Path(f).stem for f in csv_files]
    
    console.print(f"\n[bold]Available symbols in {most_recent_year}:[/bold] {len(symbols)}")
    console.print(f"Sample symbols: {', '.join(sorted(symbols)[:10])}...")
    
    # Ask for confirmation before proceeding
    console.print("\n[bold yellow]This process will migrate data from all years and can take a long time.[/bold yellow]")
    console.print("[bold]Would you like to proceed with the migration? (y/n)[/bold]")
    
    choice = input().lower()
    if choice != 'y':
        console.print("Migration cancelled.")
        return
    
    # Set parameters for the migration
    start_year = 2010  # You can adjust this as needed
    end_year = int(most_recent_year)
    max_workers = 8  # Adjust based on your system's capabilities
    
    console.print(f"\n[bold]Starting migration with the following parameters:[/bold]")
    console.print(f"- Start year: {start_year}")
    console.print(f"- End year: {end_year}")
    console.print(f"- Max workers: {max_workers}\n")
    
    # Run the migration
    migrate_fx_m5_data(
        source_dir=source_dir,
        qlib_dir=qlib_dir,
        start_year=start_year,
        end_year=end_year,
        max_workers=max_workers
    )
    
    # Print post-migration instructions
    console.print("\n[bold]Migration Complete![/bold]")
    console.print("\n[bold]Post-Migration Steps:[/bold]")
    console.print("1. Check data health:")
    console.print(f"   python scripts/data_collector/icmarket/examples/check_fx_data_health.py --qlib_dir {qlib_dir}")
    console.print("\n2. Convert to QLib binary format:")
    console.print(f"   python scripts/dump_bin.py dump_all --csv_path {qlib_dir}/csv_data --qlib_dir {qlib_dir}")
    console.print("\n3. Use the data in Python:")
    console.print("""
    import qlib
    from qlib.constant import REG_CN
    
    # Initialize QLib with forex data
    qlib.init(
        provider_uri="~/.qlib/qlib_data/icm_fx_m5",
        region=REG_CN,
    )
    
    # Use the data
    from qlib.data import D
    instruments = D.instruments("major.txt")
    fields = ["$close", "$open", "$high", "$low"]
    df = D.features(instruments, fields, start_time="2020-01-01", end_time="2023-01-01", freq="5t")
    print(df.head())
    """)

if __name__ == "__main__":
    run_migration_example() 