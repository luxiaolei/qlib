#!/usr/bin/env python
"""Command-line interface for the FMP data collector.

This module provides a command-line interface for collecting data from FMP.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from scripts.data_collector.fmp_new.collector import DailyFMPCollector, IntradayFMPCollector
from scripts.data_collector.fmp_new.index import IndexManager

# Create console and app
console = Console()
app = typer.Typer(help="FMP Data Collector CLI")


def validate_date(date_str: str) -> str:
    """Validate a date string (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        console.print(f"[bold red]Invalid date format: {date_str}. Use YYYY-MM-DD.[/bold red]")
        sys.exit(1)


def validate_api_key() -> str:
    """Get and validate the FMP API key."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        console.print("[bold red]FMP_API_KEY environment variable not found.[/bold red]")
        console.print("Please set the FMP_API_KEY environment variable or provide it as an argument.")
        sys.exit(1)
    return api_key

@app.command()
def daily(
    start: str = typer.Argument(..., help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(None, help="End date in YYYY-MM-DD format"),
    symbols: str = typer.Option(None, help="Comma-separated list of symbols to collect"),
    index: str = typer.Option("sp500", help="Index name to collect (sp500, dowjones, nasdaq100)"),
    save_dir: str = typer.Option("./data/fmp/daily", help="Directory to save data"),
    incremental: bool = typer.Option(False, help="Collect only new data"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
    max_workers: int = typer.Option(5, help="Maximum number of workers"),
    redis_url: str = typer.Option(None, help="Redis URL for rate limiting"),
):
    """
    Collect daily price data from FMP API.
    """
    # Validate inputs
    start = validate_date(start)
    if end:
        end = validate_date(end)
    
    api_key = validate_api_key()
    
    # Process symbols from command line or index
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        console.print(f"✓ Using {len(symbol_list)} custom symbols")
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Print collection plan
    console.print(Panel.fit(
        "\n".join([
            "[bold]Data Collection Plan[/bold]",
            "",
            f"Interval: Daily",
            f"Date Range: {start} to {end or 'today'}",
            f"Symbols: {len(symbol_list) if symbol_list else index}",
            f"Save Directory: {save_dir}",
            f"Workers: {max_workers}",
            f"Redis: {'Enabled' if redis_url else 'Disabled'}",
            f"Mode: {'Incremental' if incremental else 'Full'}"
        ]),
        title="FMP Data Collector"
    ))
    
    # Ask for confirmation
    proceed = input("Proceed with data collection? [y/n]: ")
    if proceed.lower() != "y":
        console.print("[yellow]Collection cancelled by user[/yellow]")
        return
    
    # Create collector
    collector = DailyFMPCollector(
        save_dir=save_path,
        start=start,
        end=end,
        symbols=symbol_list,
        api_key=api_key,
        max_workers=max_workers,
        redis_url=redis_url,
        incremental=incremental,
        overwrite=overwrite
    )
    
    # Run collection
    console.print("\n[bold]Starting data collection...[/bold]\n")
    results = asyncio.run(collector.collect_data())
    
    # Report results
    success_count = sum(1 for success in results.values() if success)
    error_count = sum(1 for success in results.values() if not success)
    
    console.print(f"\n[bold]Collection Results:[/bold]")
    console.print(f"Processed: {len(results)} symbols")
    console.print(f"Success: {success_count} symbols")
    console.print(f"Errors: {error_count} symbols")
    console.print(f"Skipped: 0 symbols")
    console.print(f"Success Rate: {success_count/len(results)*100 if len(results) > 0 else 0:.1f}%")

@app.command()
def intraday(
    start: str = typer.Argument(..., help="Start date in YYYY-MM-DD format"),
    end: str = typer.Option(None, help="End date in YYYY-MM-DD format"),
    interval: str = typer.Option("15min", help="Data interval (1min, 5min, 15min, 30min, 1hour)"),
    symbols: str = typer.Option(None, help="Comma-separated list of symbols to collect"),
    index: str = typer.Option("sp500", help="Index name to collect (sp500, dowjones, nasdaq100)"),
    save_dir: str = typer.Option("./data/fmp/intraday", help="Directory to save data"),
    incremental: bool = typer.Option(False, help="Collect only new data"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
    max_workers: int = typer.Option(5, help="Maximum number of workers"),
    redis_url: str = typer.Option(None, help="Redis URL for rate limiting"),
    lookback_days: int = typer.Option(7, help="Number of days to look back for intraday data"),
):
    """
    Collect intraday price data from FMP API.
    """
    # Validate inputs
    start = validate_date(start)
    if end:
        end = validate_date(end)
    
    api_key = validate_api_key()
    
    # Validate interval
    valid_intervals = ["1min", "5min", "15min", "30min", "1hour"]
    if interval not in valid_intervals:
        console.print(f"[bold red]Error: Invalid interval '{interval}'[/bold red]")
        console.print(f"Valid intervals: {', '.join(valid_intervals)}")
        return
    
    # Process symbols from command line or index
    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        console.print(f"✓ Using {len(symbol_list)} custom symbols")
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Print collection plan
    console.print(Panel.fit(
        "\n".join([
            "[bold]Data Collection Plan[/bold]",
            "",
            f"Interval: {interval}",
            f"Date Range: {start} to {end or 'today'}",
            f"Symbols: {len(symbol_list) if symbol_list else index}",
            f"Save Directory: {save_dir}/{interval}",
            f"Workers: {max_workers}",
            f"Redis: {'Enabled' if redis_url else 'Disabled'}",
            f"Mode: {'Incremental' if incremental else 'Full'}"
        ]),
        title="FMP Data Collector"
    ))
    
    # Ask for confirmation
    proceed = input("Proceed with data collection? [y/n]: ")
    if proceed.lower() != "y":
        console.print("[yellow]Collection cancelled by user[/yellow]")
        return
    
    # Create collector
    collector = IntradayFMPCollector(
        save_dir=save_path,
        start=start,
        end=end,
        interval=interval,
        symbols=symbol_list,
        api_key=api_key,
        max_workers=max_workers,
        redis_url=redis_url,
        incremental=incremental,
        overwrite=overwrite,
        lookback_days=lookback_days
    )
    
    # Run collection
    console.print("\n[bold]Starting data collection...[/bold]\n")
    results = asyncio.run(collector.collect_data())
    
    # Report results
    success_count = sum(1 for success in results.values() if success)
    error_count = len(results) - success_count
    
    console.print(f"\n[bold]Collection Results:[/bold]")
    console.print(f"Processed: {len(results)} symbols")
    console.print(f"Success: {success_count} symbols")
    console.print(f"Errors: {error_count} symbols")
    console.print(f"Skipped: 0 symbols")
    console.print(f"Success Rate: {success_count/len(results)*100 if len(results) > 0 else 0:.1f}%")

@app.command()
def index(
    action: str = typer.Argument(..., help="Action to perform (list, update)"),
    index_name: str = typer.Option("sp500", help="Index name (sp500, dow, nasdaq)"),
    save_dir: str = typer.Option("./data/fmp/index", help="Directory to save index data"),
):
    """Manage index constituents."""
    api_key = validate_api_key()
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize index manager
    index_manager = IndexManager(api_key=api_key, save_dir=save_path)
    
    if action == "list":
        # List index constituents
        console.print(f"[bold]Fetching current {index_name} constituents...[/bold]")
        symbols = asyncio.run(index_manager.get_index_symbols(index_name))
        
        if not symbols:
            console.print(f"[bold red]Failed to fetch symbols for index: {index_name}[/bold red]")
            sys.exit(1)
        
        console.print(Panel.fit(
            "\n".join(f"[cyan]{symbol}[/cyan]" for symbol in sorted(symbols)),
            title=f"{index_name.upper()} Constituents ({len(symbols)})",
            border_style="blue"
        ))
        
    elif action == "update":
        # Update index constituents file
        console.print(f"[bold]Updating {index_name} constituents file...[/bold]")
        result = asyncio.run(index_manager.update_index_file(index_name))
        
        if result:
            console.print(f"[bold green]Successfully updated {index_name} constituents file.[/bold green]")
            console.print(f"[bold]File saved to:[/bold] {save_path / f'{index_name}_constituents.csv'}")
        else:
            console.print(f"[bold red]Failed to update {index_name} constituents file.[/bold red]")
            sys.exit(1)
            
    else:
        console.print(f"[bold red]Invalid action: {action}. Use 'list' or 'update'.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    app() 