"""
Command line interface for FMP data collection.
"""
import os
from typing import Optional, cast

import fire
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from scripts.data_collector.fmp.us_daily import FMPDailyRunner

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

def main(
    api_key: Optional[str] = None,
    qlib_dir: str = "~/.qlib/qlib_data/us_fmp_d1",
    method: str = "download",  # download or update
    start_date: Optional[str] = None, # default: 2004-01-01
    end_date: Optional[str] = None, # default: today
    delay: float = 0.2,
    check_data_length: Optional[int] = None,
    max_workers: int = 4,
    max_collector_count: int = 2,
) -> None:
    """
    Download or update US stock data from FMP.
    
    Parameters
    ----------
    api_key : Optional[str]
        FMP API key. If not provided, will try to get from FMP_API_KEY env variable
    qlib_dir : str
        Qlib data directory, default "~/.qlib/qlib_data/us_fmp_d1"
    method : str
        'download' for full download, 'update' for incremental update
    start_date : Optional[str]
        Start date for download (YYYY-MM-DD)
    end_date : Optional[str]
        End date for download (YYYY-MM-DD)
    delay : int
        Delay between API calls in seconds
    check_data_length : Optional[int]
        Minimum required data length
    max_workers : int
        Number of parallel workers
    max_collector_count : int
        Maximum number of collection attempts per symbol
    """
    # Get API key from parameter or environment
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError(
            "FMP API key not provided. Either pass as parameter or set FMP_API_KEY environment variable"
        )
    os.environ["FMP_API_KEY"] = api_key
    
    # Initialize runner with progress tracking
    runner = FMPDailyRunner()
    runner.max_workers = max_workers
    
    # Apply DEBUG mode settings if enabled
    if DEBUG:
        console.print("[yellow]DEBUG MODE: Limited to 10 symbols and 2 years of data[/]")
        start_date = DEBUG_START_DATE
        end_date = DEBUG_END_DATE
        limit_nums = DEBUG_SYMBOLS_LIMIT
    else:
        limit_nums = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        try:
            if method == "download":
                task_desc = (
                    f"[bold green]Downloading data from "
                    f"{start_date or '2004-01-01'} to {end_date or 'today'}"
                )
                task_id = progress.add_task(task_desc, total=1)
                
                runner.download_data(
                    start=start_date,
                    end=end_date,
                    delay=delay, # type: ignore
                    check_data_length=cast(int, check_data_length),
                    max_collector_count=max_collector_count,
                    limit_nums=limit_nums,
                )
                progress.update(task_id, advance=1)
                
            elif method == "update":
                task_id = progress.add_task("[bold green]Updating data", total=1)
                runner.update_data(
                    qlib_data_dir=qlib_dir,
                    check_data_length=check_data_length,
                    delay=delay, # type: ignore
                )
                progress.update(task_id, advance=1)
                
            else:
                raise ValueError(
                    f"Invalid method: {method}. Must be either 'download' or 'update'"
                )
            
            # Normalize data after successful download/update
            task_id = progress.add_task("[bold yellow]Normalizing data", total=1)
            runner.normalize_data()
            progress.update(task_id, advance=1)
            
            console.print("[bold green]Data collection completed successfully![/]")
            
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/]")
            logger.exception("Data collection failed")
            raise

if __name__ == "__main__":
    fire.Fire(main)