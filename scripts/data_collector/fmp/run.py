"""
Command line interface for FMP data collection.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import fire
import pandas_market_calendars as mcal
import schedule
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from scripts.data_collector.fmp.us_daily import FMPDailyRunner
from scripts.data_collector.fmp.us_m5 import FMP5minRunner

# Load environment variables
load_dotenv()
console = Console()

# Debug mode settings
DEBUG = False 
if DEBUG:
    console.print("[bold yellow]Running in DEBUG mode - limited data collection[/]")
    DEBUG_START_DATE = "2022-01-01"  # Last 2 years
    DEBUG_END_DATE = "2024-12-01"
    DEBUG_SYMBOLS_LIMIT = 10


class FMPDataManager:
    """Manager for FMP data collection operations."""
    
    DEFAULT_INDEXES = [
        "^SPX",      # S&P 500
        "^IXIC",     # NASDAQ Composite
        "^DJI",      # Dow Jones Industrial Average
        "^RUT",      # Russell 2000
        "^VIX",      # VIX Volatility Index
    ]
    
    def __init__(self):
        """Initialize FMP data manager."""
        self.daily_runner = FMPDailyRunner()
        self.m5_runner = FMP5minRunner()
        self.nyse = mcal.get_calendar('NYSE')
        self.ny_tz = ZoneInfo('America/New_York')
        self.favourite_instruments = self._load_favourite_instruments()


    def show_menu(self) -> None:
        """Display interactive menu for data collection options."""
        table = Table(title="FMP Data Collection Options")
        table.add_column("Option", justify="right", style="cyan")
        table.add_column("Description", style="magenta")
        
        table.add_row("1", "Manual Download (Full Historical Data)")
        table.add_row("2", "Manual Update (Incremental Update)")
        table.add_row("3", "Start Routine Update Service")
        table.add_row("4", "Exit")
        
        while True:
            console.clear()
            console.print(table)
            choice = IntPrompt.ask("Select an option", choices=["1", "2", "3", "4"])
            
            if choice == 1:
                self.manual_download()
            elif choice == 2:
                self.manual_update()
            elif choice == 3:
                self.start_routine_update()
            else:
                console.print("[yellow]Exiting...[/]")
                break
            
    def _load_favourite_instruments(self) -> List[str]:
        """Load or create favourite instruments list.
        
        Returns
        -------
        List[str]
            List of favourite instruments including indexes
        """
        favourites_file = Path("~/.qlib/qlib_data/us_fmp_5min/instruments/favourites.txt").expanduser()
        favourites_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not favourites_file.exists():
            # Create default favourites file with indexes
            with favourites_file.open("w") as f:
                for index in self.DEFAULT_INDEXES:
                    f.write(f"{index}\n")
            logger.info(f"Created default favourites file with indexes at {favourites_file}")
            return self.DEFAULT_INDEXES
        
        with favourites_file.open("r") as f:
            instruments = [line.strip() for line in f if line.strip()]
            
        if len(instruments) > 740:
            logger.warning("Too many favourite instruments (>740), performance may be affected")
            
        return instruments

    def _get_non_favourite_instruments(self) -> List[str]:
        """Get list of instruments not in favourites list.
        
        Returns
        -------
        List[str]
            List of non-favourite instruments
        """
        all_instruments_file = Path("~/.qlib/qlib_data/us_fmp_5min/instruments/all.txt").expanduser()
        if not all_instruments_file.exists():
            logger.warning("all.txt not found, running initial download first")
            return []
            
        with all_instruments_file.open("r") as f:
            all_instruments = [line.split("\t")[0].strip() for line in f if line.strip()]
            
        return [symbol for symbol in all_instruments if symbol not in self.favourite_instruments]

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        now = datetime.now(self.ny_tz)
        schedule = self.nyse.schedule(start_date=now.date(), end_date=now.date())
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]['market_open'].tz_convert(self.ny_tz)
        market_close = schedule.iloc[0]['market_close'].tz_convert(self.ny_tz)
        return market_open <= now <= market_close

    def manual_update(self) -> None:
        """Handle manual update of data."""
        console.print("\n[bold cyan]Manual Update Configuration[/]")
        
        # Common parameters
        delay = IntPrompt.ask("Enter delay between API calls (seconds)", default=1)
        
        # Ask which data types to update
        update_daily = Confirm.ask("Update daily data?", default=True)
        update_m5 = Confirm.ask("Update 5-minute data?", default=True)
        
        try:
            if update_daily:
                console.print("\n[bold green]Updating daily data...[/]")
                self.daily_runner.update_data(
                    qlib_data_dir="~/.qlib/qlib_data/us_fmp_d1",
                    delay=delay,
                )
                
            if update_m5:
                console.print("\n[bold green]Updating 5-minute data...[/]")
                self.m5_runner.update_data(
                    qlib_data_dir="~/.qlib/qlib_data/us_fmp_5min",
                    delay=delay,
                )
                
            console.print("[bold green]Update completed successfully![/]")
            
        except Exception as e:
            console.print(f"[bold red]Error during update: {str(e)}[/]")
            logger.exception("Update failed")

    def manual_download(self) -> None:
        """Handle manual download of historical data."""
        console.print("\n[bold cyan]Manual Download Configuration[/]")
        
        # Ask which data types to download
        download_daily = Confirm.ask("Download daily data?", default=True)
        download_m5 = Confirm.ask("Download 5-minute data?", default=True)
        
        try:
            if download_daily:
                console.print("\n[bold green]Downloading daily data...[/]")
                self.daily_runner.download_data(
                    delay=0.2, # type: ignore
                    max_workers=4, 
                )
                
            if download_m5:
                console.print("\n[bold green]Downloading 5-minute data...[/]")
                self.m5_runner.download_data(
                    delay=0,
                    max_workers=4,
                )
                
            console.print("[bold green]Download completed successfully![/]")
            
        except Exception as e:
            console.print(f"[bold red]Error during download: {str(e)}[/]")
            logger.exception("Download failed")

    def update_m5_data_favourites(self) -> None:
        """Update 5-minute data for favourite instruments."""
        if not self.is_market_open():
            logger.info("Market is closed, skipping favourite instruments update")
            return
            
        try:
            logger.info("Updating 5-minute data for favourite instruments...")
            self.m5_runner.update_data(
                qlib_data_dir="~/.qlib/qlib_data/us_fmp_5min",
                delay=0,
                instruments=self.favourite_instruments
            )
            logger.info("Favourite instruments update completed")
        except Exception as e:
            logger.error(f"Error updating favourite instruments: {e}")

    def update_m5_data_others(self) -> None:
        """Update 5-minute data for non-favourite instruments."""
        if self.is_market_open():
            logger.info("Market is open, skipping other instruments update")
            return
            
        try:
            non_favourites = self._get_non_favourite_instruments()
            if non_favourites:
                logger.info("Updating 5-minute data for other instruments...")
                self.m5_runner.update_data(
                    qlib_data_dir="~/.qlib/qlib_data/us_fmp_5min",
                    delay=0,
                    instruments=non_favourites
                )
                logger.info("Other instruments update completed")
        except Exception as e:
            logger.error(f"Error updating other instruments: {e}")

    def start_routine_update(self) -> None:
        """Start routine update service with scheduling."""
        console.print("[bold green]Starting routine update service...[/]")
        
        # Schedule daily updates at 16:30 ET
        schedule.every().day.at("16:30").do(
            self.daily_runner.update_data,
            qlib_data_dir="~/.qlib/qlib_data/us_fmp_d1",
            delay=0.2
        )
        
        # Schedule favourite instruments updates during market hours
        def schedule_m5_favourite_updates():
            """Schedule 5-minute updates for favourite instruments."""
            now = datetime.now(self.ny_tz)
            schedule_date = self.nyse.schedule(start_date=now.date(), end_date=now.date())
            
            if not schedule_date.empty:
                market_open = schedule_date.iloc[0]['market_open'].tz_convert(self.ny_tz)
                market_close = schedule_date.iloc[0]['market_close'].tz_convert(self.ny_tz)
                
                current_time = market_open
                while current_time <= market_close:
                    schedule.every().day.at(current_time.strftime("%H:%M")).do(
                        self.update_m5_data_favourites
                    )
                    current_time += timedelta(minutes=5)

        # Schedule non-favourite instruments updates during off-market hours
        schedule.every().day.at("20:00").do(self.update_m5_data_others)  # 8 PM ET
        schedule.every().day.at("04:00").do(self.update_m5_data_others)  # 4 AM ET
        
        # Schedule the initial m5 updates and reschedule daily
        schedule.every().day.at("00:01").do(schedule_m5_favourite_updates)
        
        console.print("[green]Update service started. Press Ctrl+C to stop.[/]")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]Stopping update service...[/]")
            schedule.clear()



def main(
    method: str = "menu",  # menu, download, update, or service
) -> None:
    """
    FMP data collection interface.
    
    Parameters
    ----------
    api_key : Optional[str]
        FMP API key. If not provided, will try to get from FMP_API_KEY env variable
    qlib_dir : str
        Qlib data directory
    method : str
        'menu' for interactive menu
        'download' for full download
        'update' for incremental update
        'service' for routine update service
    start_date : Optional[str]
        Start date for download (YYYY-MM-DD)
    end_date : Optional[str]
        End date for download (YYYY-MM-DD)
    delay : float
        Delay between API calls in seconds
    check_data_length : Optional[int]
        Minimum required data length
    max_workers : int
        Number of parallel workers
    max_collector_count : int
        Maximum number of collection attempts per symbol
    """
    manager = FMPDataManager()
    
    if method == "menu":
        manager.show_menu()
    elif method == "download":
        manager.manual_download()
    elif method == "update":
        manager.manual_update()
    elif method == "service":
        manager.start_routine_update()
    else:
        console.print(f"[bold red]Invalid method: {method}[/]")

if __name__ == "__main__":
    fire.Fire(main)