#!/usr/bin/env python
"""Financial Modeling Prep (FMP) Data Collector

This script provides a command-line interface for collecting and processing
stock data from Financial Modeling Prep API.

Features:
- Daily and intraday data collection
- Support for multiple indexes (S&P 500, Dow Jones, Nasdaq 100)
- Multiple time intervals (1min, 5min, 15min, 30min, 1hour, 4hour, 1day)
- Data quality checks
- Incremental updates
- Live feed
- Rich console UI

Usage:
    python main.py daily               # Collect daily data
    python main.py intraday --interval 15min  # Collect 15-minute data
    python main.py both                # Collect both daily and intraday data
    python main.py constituents        # Download index constituents only
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from scripts.data_collector.fmp_new.runner import FMPRunner

console = Console()

# Default paths
DEFAULT_CACHE_DIR = "~/.qlib/cache/fmp"
DEFAULT_DAILY_RAW_DIR = "~/.qlib/qlib_data/fmp_daily_raw"
DEFAULT_DAILY_NORM_DIR = "~/.qlib/qlib_data/fmp_daily_normalized"
DEFAULT_DAILY_QLIB_DIR = "~/.qlib/qlib_data/fmp_daily"
DEFAULT_INTRADAY_RAW_DIR_TEMPLATE = "~/.qlib/qlib_data/fmp_{interval}_raw"
DEFAULT_INTRADAY_NORM_DIR_TEMPLATE = "~/.qlib/qlib_data/fmp_{interval}_normalized"
DEFAULT_INTRADAY_QLIB_DIR_TEMPLATE = "~/.qlib/qlib_data/fmp_{interval}"

def check_api_key() -> bool:
    """Check if FMP API key is available.
    
    Returns
    -------
    bool
        True if API key is available, False otherwise
    """
    api_key = os.environ.get("FMP_API_KEY")
    
    if not api_key:
        console.print("[bold red]Error: FMP_API_KEY environment variable not found.[/bold red]")
        console.print("Please set the environment variable with your API key:")
        console.print("[yellow]export FMP_API_KEY='your_api_key'[/yellow]")
        return False
        
    return True

def check_redis() -> bool:
    """Check if Redis is available.
    
    Returns
    -------
    bool
        True if Redis is available, False otherwise
    """
    try:
        import redis
        
        client = redis.Redis(host="localhost", port=6379)
        client.ping()
        return True
        
    except Exception as e:
        console.print("[bold yellow]Warning: Redis not available.[/bold yellow]")
        console.print(f"Error: {e}")
        console.print("Rate limiting will be less effective. Consider installing Redis:")
        console.print("[yellow]1. Install Redis server[/yellow]")
        console.print("[yellow]2. Start Redis service[/yellow]")
        return False

def show_welcome():
    """Show welcome message."""
    console.print(Panel.fit(
        "[bold blue]Financial Modeling Prep (FMP) Data Collector[/bold blue]\n\n"
        "This tool collects stock data from Financial Modeling Prep API and "
        "processes it for use with Qlib.",
        title="FMP Data Collector",
        border_style="blue"
    ))

def collect_daily_data(args):
    """Collect daily data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Set up paths
    raw_dir = args.daily_raw_dir or DEFAULT_DAILY_RAW_DIR
    norm_dir = args.daily_norm_dir or DEFAULT_DAILY_NORM_DIR
    qlib_dir = args.daily_qlib_dir or DEFAULT_DAILY_QLIB_DIR
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    
    # Create runner
    runner = FMPRunner(
        max_workers=args.max_workers,
        cache_dir=cache_dir
    )
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Run pipeline
    runner.run_daily_pipeline(
        raw_dir=raw_dir,
        normalized_dir=norm_dir,
        qlib_dir=qlib_dir,
        start=args.start,
        end=args.end,
        symbols=symbols,
        index_name=args.index,
        skip_existing=not args.no_skip,
        save_constituents=True,
        update_mode=not args.full_dump
    )

def collect_intraday_data(args):
    """Collect intraday data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Set up paths
    interval = args.interval
    
    # Use templates for default paths
    raw_dir = args.intraday_raw_dir or DEFAULT_INTRADAY_RAW_DIR_TEMPLATE.format(interval=interval)
    norm_dir = args.intraday_norm_dir or DEFAULT_INTRADAY_NORM_DIR_TEMPLATE.format(interval=interval)
    qlib_dir = args.intraday_qlib_dir or DEFAULT_INTRADAY_QLIB_DIR_TEMPLATE.format(interval=interval)
    daily_qlib_dir = args.daily_qlib_dir or DEFAULT_DAILY_QLIB_DIR
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    
    # Create runner
    runner = FMPRunner(
        max_workers=args.max_workers,
        cache_dir=cache_dir
    )
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Run pipeline
    runner.run_intraday_pipeline(
        raw_dir=raw_dir,
        normalized_dir=norm_dir,
        qlib_dir=qlib_dir,
        qlib_daily_dir=daily_qlib_dir,
        start=args.start,
        end=args.end,
        interval=interval,
        symbols=symbols,
        index_name=args.index,
        skip_existing=not args.no_skip,
        update_mode=not args.full_dump
    )

def collect_constituents(args):
    """Collect index constituents.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Set up paths
    qlib_dir = args.daily_qlib_dir or DEFAULT_DAILY_QLIB_DIR
    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    
    # Create runner
    runner = FMPRunner(
        max_workers=args.max_workers,
        cache_dir=cache_dir
    )
    
    # Run collection
    instruments_dir = Path(qlib_dir).expanduser().resolve() / "instruments"
    instruments_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold]Collecting {args.index} constituents...[/bold]")
    
    success = runner.save_index_constituents(
        output_dir=instruments_dir,
        index_name=args.index
    )
    
    if success:
        console.print(f"[bold green]Successfully saved {args.index} constituents to {instruments_dir}[/bold green]")
    else:
        console.print(f"[bold red]Failed to save {args.index} constituents[/bold red]")

def show_info(args):
    """Show information about available indexes and intervals.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create tables
    index_table = Table(title="Available Indexes")
    index_table.add_column("Index", style="cyan")
    index_table.add_column("Description", style="green")
    index_table.add_column("Approximate Size", style="yellow")
    
    index_table.add_row("sp500", "S&P 500", "500 companies")
    index_table.add_row("dowjones", "Dow Jones Industrial Average", "30 companies")
    index_table.add_row("nasdaq100", "Nasdaq 100", "100 companies")
    
    interval_table = Table(title="Available Intervals")
    interval_table.add_column("Interval", style="cyan")
    interval_table.add_column("Description", style="green")
    interval_table.add_column("Data Availability", style="yellow")
    
    interval_table.add_row("1min", "1-minute intraday data", "Last 1-3 months")
    interval_table.add_row("5min", "5-minute intraday data", "Last 1-3 months")
    interval_table.add_row("15min", "15-minute intraday data", "Last 1-3 months")
    interval_table.add_row("30min", "30-minute intraday data", "Last 1-3 months")
    interval_table.add_row("1hour", "1-hour intraday data", "Last 1-3 months")
    interval_table.add_row("4hour", "4-hour intraday data", "Last 1-3 months")
    interval_table.add_row("1d", "Daily data", "20+ years")
    
    # Show tables
    console.print(index_table)
    console.print()
    console.print(interval_table)
    
    # Show API usage
    console.print("\n[bold]API Usage Information[/bold]")
    console.print("- Free tier: 250 API calls per minute")
    console.print("- Starter tier: 600 API calls per minute")
    console.print("- Professional tier: 1200 API calls per minute")
    console.print("\nThis collector is optimized to work within these rate limits.")

def run_interactive():
    """Run in interactive mode."""
    show_welcome()
    
    # Check API key
    if not check_api_key():
        return
        
    # Check Redis
    check_redis()
    
    # Collect options
    mode = Prompt.ask(
        "Choose collection mode",
        choices=["daily", "intraday", "both", "constituents", "info", "exit"],
        default="info"
    )
    
    if mode == "exit":
        return
        
    if mode == "info":
        show_info(None)
        return
        
    # Common options
    index = Prompt.ask(
        "Select index",
        choices=["sp500", "dowjones", "nasdaq100", "all"],
        default="sp500"
    )
    
    # Date range
    default_end = datetime.now().strftime("%Y-%m-%d")
    
    if mode in ["daily", "both"]:
        default_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    else:
        default_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
    start = Prompt.ask("Start date (YYYY-MM-DD)", default=default_start)
    end = Prompt.ask("End date (YYYY-MM-DD)", default=default_end)
    
    # Create args object
    class Args:
        pass
        
    args = Args()
    args.index = index
    args.start = start
    args.end = end
    args.symbols = None
    args.no_skip = False
    args.full_dump = False
    args.max_workers = 10
    args.daily_raw_dir = None
    args.daily_norm_dir = None
    args.daily_qlib_dir = None
    args.intraday_raw_dir = None
    args.intraday_norm_dir = None
    args.intraday_qlib_dir = None
    args.cache_dir = None
    
    # Intraday specific options
    if mode in ["intraday", "both"]:
        args.interval = Prompt.ask(
            "Select interval",
            choices=["1min", "5min", "15min", "30min", "1hour", "4hour"],
            default="15min"
        )
    
    # Run collection
    if mode in ["daily", "both"]:
        collect_daily_data(args)
        
    if mode in ["intraday", "both"]:
        collect_intraday_data(args)
        
    if mode == "constituents":
        collect_constituents(args)

def main():
    """Main entry point."""
    import argparse
    
    show_welcome()
    
    # Check API key
    if not check_api_key():
        return 1
        
    # Check Redis
    redis_available = check_redis()
    
    # Create parser
    parser = argparse.ArgumentParser(description="FMP Data Collector")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Daily command
    daily_parser = subparsers.add_parser("daily", help="Collect daily data")
    daily_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    daily_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    daily_parser.add_argument("--index", type=str, default="sp500", 
                           help="Index name (sp500, dowjones, nasdaq100)")
    daily_parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    daily_parser.add_argument("--no-skip", action="store_true", 
                           help="Do not skip existing data")
    daily_parser.add_argument("--daily-raw-dir", type=str, 
                           help="Directory for raw daily data")
    daily_parser.add_argument("--daily-norm-dir", type=str, 
                           help="Directory for normalized daily data")
    daily_parser.add_argument("--daily-qlib-dir", type=str, 
                           help="Directory for Qlib daily data")
    daily_parser.add_argument("--cache-dir", type=str, 
                           help="Directory for API response cache")
    daily_parser.add_argument("--max-workers", type=int, default=10,
                           help="Maximum number of parallel workers")
    daily_parser.add_argument("--full-dump", action="store_true",
                           help="Use full dump instead of update mode")
    
    # Intraday command
    intraday_parser = subparsers.add_parser("intraday", help="Collect intraday data")
    intraday_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    intraday_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    intraday_parser.add_argument("--interval", type=str, default="15min",
                             help="Intraday interval (1min, 5min, 15min, 30min, 1hour, 4hour)")
    intraday_parser.add_argument("--index", type=str, default="sp500", 
                              help="Index name (sp500, dowjones, nasdaq100)")
    intraday_parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    intraday_parser.add_argument("--no-skip", action="store_true", 
                              help="Do not skip existing data")
    intraday_parser.add_argument("--intraday-raw-dir", type=str, 
                              help="Directory for raw intraday data")
    intraday_parser.add_argument("--intraday-norm-dir", type=str, 
                              help="Directory for normalized intraday data")
    intraday_parser.add_argument("--intraday-qlib-dir", type=str, 
                              help="Directory for Qlib intraday data")
    intraday_parser.add_argument("--daily-qlib-dir", type=str, 
                              help="Directory for Qlib daily data")
    intraday_parser.add_argument("--cache-dir", type=str, 
                              help="Directory for API response cache")
    intraday_parser.add_argument("--max-workers", type=int, default=10,
                              help="Maximum number of parallel workers")
    intraday_parser.add_argument("--full-dump", action="store_true",
                              help="Use full dump instead of update mode")
    
    # Both command
    both_parser = subparsers.add_parser("both", help="Collect both daily and intraday data")
    both_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    both_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    both_parser.add_argument("--interval", type=str, default="15min",
                         help="Intraday interval (1min, 5min, 15min, 30min, 1hour, 4hour)")
    both_parser.add_argument("--index", type=str, default="sp500", 
                          help="Index name (sp500, dowjones, nasdaq100)")
    both_parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    both_parser.add_argument("--no-skip", action="store_true", 
                          help="Do not skip existing data")
    both_parser.add_argument("--daily-raw-dir", type=str, 
                          help="Directory for raw daily data")
    both_parser.add_argument("--daily-norm-dir", type=str, 
                          help="Directory for normalized daily data")
    both_parser.add_argument("--daily-qlib-dir", type=str, 
                          help="Directory for Qlib daily data")
    both_parser.add_argument("--intraday-raw-dir", type=str, 
                          help="Directory for raw intraday data")
    both_parser.add_argument("--intraday-norm-dir", type=str, 
                          help="Directory for normalized intraday data")
    both_parser.add_argument("--intraday-qlib-dir", type=str, 
                          help="Directory for Qlib intraday data")
    both_parser.add_argument("--cache-dir", type=str, 
                          help="Directory for API response cache")
    both_parser.add_argument("--max-workers", type=int, default=10,
                          help="Maximum number of parallel workers")
    both_parser.add_argument("--full-dump", action="store_true",
                          help="Use full dump instead of update mode")
    
    # Constituents command
    constituents_parser = subparsers.add_parser("constituents", help="Download index constituents only")
    constituents_parser.add_argument("--index", type=str, default="sp500", 
                                  help="Index name (sp500, dowjones, nasdaq100)")
    constituents_parser.add_argument("--daily-qlib-dir", type=str, 
                                  help="Directory for Qlib daily data")
    constituents_parser.add_argument("--cache-dir", type=str, 
                                  help="Directory for API response cache")
    constituents_parser.add_argument("--max-workers", type=int, default=10,
                                  help="Maximum number of parallel workers")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about available indexes and intervals")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default values
    if not args.command:
        # No command provided, show help
        parser.print_help()
        return 1
        
    # Run appropriate command
    if args.command == "daily":
        collect_daily_data(args)
    elif args.command == "intraday":
        collect_intraday_data(args)
    elif args.command == "both":
        collect_daily_data(args)
        collect_intraday_data(args)
    elif args.command == "constituents":
        collect_constituents(args)
    elif args.command == "info":
        show_info(args)
    elif args.command == "interactive":
        run_interactive()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 