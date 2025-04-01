#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script to verify that the forex data files have been properly created.
This script does not use QLib APIs, it just examines the files directly.
"""

import os
import glob
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def verify_directory_structure(base_dir: str):
    """Verify that the directory structure is correct"""
    base_dir = os.path.expanduser(base_dir)
    console.print(f"[bold]Verifying directory structure in {base_dir}...[/bold]")
    
    expected_dirs = ["calendars", "instruments", "features", "csv_data"]
    expected_instrument_files = ["all.txt", "major.txt", "minor.txt", "exotic.txt", 
                               "commodity.txt", "crypto.txt", "index.txt"]
    expected_calendar_files = ["day.txt", "1h.txt", "5t.txt", "1t.txt"]
    
    table = Table(title="Directory Structure")
    table.add_column("Item", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes", style="yellow")
    
    # Check main directories
    for dir_name in expected_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        status = "✅ Found" if os.path.isdir(dir_path) else "❌ Missing"
        table.add_row(f"Directory: {dir_name}", status, "")
    
    # Check instrument files
    for file_name in expected_instrument_files:
        file_path = os.path.join(base_dir, "instruments", file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            status = "✅ Found"
            notes = f"{len(lines)} instruments"
        else:
            status = "❌ Missing"
            notes = ""
        table.add_row(f"Instrument file: {file_name}", status, notes)
    
    # Check calendar files
    for file_name in expected_calendar_files:
        file_path = os.path.join(base_dir, "calendars", file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            status = "✅ Found"
            notes = f"{len(lines)} trading days"
        else:
            status = "❌ Missing"
            notes = ""
        table.add_row(f"Calendar file: {file_name}", status, notes)
    
    console.print(table)

def count_feature_files(base_dir: str):
    """Count how many feature files exist for each symbol"""
    base_dir = os.path.expanduser(base_dir)
    features_dir = os.path.join(base_dir, "features")
    console.print(f"[bold]Counting feature files in {features_dir}...[/bold]")
    
    table = Table(title="Feature Files")
    table.add_column("Symbol Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Example Symbols", style="yellow")
    
    # Get all symbol directories
    symbols = [d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))]
    symbols.sort()
    
    # Load instrument categorization
    symbol_categories = {}
    category_symbols = {}
    category_dirs = {
        "major": os.path.join(base_dir, "instruments", "major.txt"),
        "minor": os.path.join(base_dir, "instruments", "minor.txt"),
        "exotic": os.path.join(base_dir, "instruments", "exotic.txt"),
        "commodity": os.path.join(base_dir, "instruments", "commodity.txt"),
        "crypto": os.path.join(base_dir, "instruments", "crypto.txt"),
        "index": os.path.join(base_dir, "instruments", "index.txt")
    }
    
    for category, file_path in category_dirs.items():
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            cat_symbols = []
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) > 0:
                    symbol = parts[0].lower()  # Feature directories are lowercase
                    cat_symbols.append(symbol)
                    symbol_categories[symbol] = category
            
            category_symbols[category] = cat_symbols
    
    # Count symbols by category
    for category in ["major", "minor", "exotic", "commodity", "crypto", "index"]:
        if category in category_symbols:
            cat_symbols = category_symbols[category]
            existing_symbols = [s for s in cat_symbols if s in symbols]
            count = len(existing_symbols)
            examples = ", ".join(existing_symbols[:3]) if count > 0 else "None"
            table.add_row(category, str(count), examples)
        else:
            table.add_row(category, "0", "None")
    
    # Also show uncategorized symbols
    uncategorized = [s for s in symbols if s not in symbol_categories]
    if uncategorized:
        examples = ", ".join(uncategorized[:3])
        table.add_row("uncategorized", str(len(uncategorized)), examples)
    
    console.print(table)

def check_data_sample(base_dir: str):
    """Check a sample of the data to see if it looks reasonable"""
    base_dir = os.path.expanduser(base_dir)
    csv_dir = os.path.join(base_dir, "csv_data", "fx_normalized")
    console.print(f"[bold]Checking sample data in {csv_dir}...[/bold]")
    
    # Find a sample file
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        console.print("[bold red]No CSV files found![/bold red]")
        return
    
    sample_file = csv_files[0]
    console.print(f"[bold]Sample file: {os.path.basename(sample_file)}[/bold]")
    
    # Load the data
    try:
        df = pd.read_csv(sample_file)
        
        table = Table(title=f"Sample Data: {os.path.basename(sample_file)}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        # Show basic info
        table.add_row("Row count", str(len(df)))
        table.add_row("Columns", ", ".join(df.columns))
        
        # Show date range
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min()
            max_date = df['date'].max()
            table.add_row("Date range", f"{min_date} to {max_date}")
        
        # Show some statistics
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                table.add_row(f"Mean {col}", f"{df[col].mean():.4f}")
                table.add_row(f"Min {col}", f"{df[col].min():.4f}")
                table.add_row(f"Max {col}", f"{df[col].max():.4f}")
        
        console.print(table)
        
        # Show a few rows
        console.print("[bold]Sample rows:[/bold]")
        console.print(df.head(5))
        
    except Exception as e:
        console.print(f"[bold red]Error reading file: {str(e)}[/bold red]")

def main():
    """Main verification function"""
    qlib_dir = "~/.qlib/qlib_data/forex_data"
    console.print(f"[bold green]Verifying data in {qlib_dir}[/bold green]")
    
    # Verify directory structure
    verify_directory_structure(qlib_dir)
    
    # Count feature files
    count_feature_files(qlib_dir)
    
    # Check data sample
    check_data_sample(qlib_dir)
    
    console.print("[bold green]Verification complete![/bold green]")

if __name__ == "__main__":
    main() 