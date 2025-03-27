"""Index constituent management for FMP data collector.

This module provides functionality for managing index constituents, including
fetching current index components and historical changes.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from scripts.data_collector.fmp_new.api import FMPClient

console = Console()

class IndexManager:
    """Manager for index constituents.
    
    This class handles fetching and saving index constituent data from FMP.
    It supports both current and historical constituent data.
    
    Attributes
    ----------
    api_key : str
        FMP API key
    save_dir : Path
        Directory to save constituent data
    """
    
    # Map of index names to FMP API names
    INDEX_MAP = {
        "sp500": "sp500",
        "dow": "dowjones",
        "nasdaq": "nasdaq100"
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the index manager.
        
        Parameters
        ----------
        api_key : Optional[str]
            FMP API key
        save_dir : Optional[Union[str, Path]]
            Directory to save constituent data
        """
        self.api_key = api_key
        
        if save_dir:
            self.save_dir = Path(save_dir).expanduser().resolve()
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
    
    async def get_index_constituents(self, index_name: str) -> pd.DataFrame:
        """Get the current constituents of an index.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with constituent information
        """
        if index_name not in self.INDEX_MAP:
            raise ValueError(f"Invalid index name: {index_name}. Must be one of {list(self.INDEX_MAP.keys())}")
            
        fmp_index = self.INDEX_MAP[index_name]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Fetching {index_name.upper()} constituents...", total=1)
            
            async with FMPClient(api_key=self.api_key) as client:
                df = await client.get_index_constituents(index=fmp_index)
                progress.update(task, advance=1)
                
            return df
    
    async def get_historical_constituents(self, index_name: str) -> pd.DataFrame:
        """Get historical constituent changes for an index.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with historical constituent changes
        """
        if index_name not in self.INDEX_MAP:
            raise ValueError(f"Invalid index name: {index_name}. Must be one of {list(self.INDEX_MAP.keys())}")
            
        fmp_index = self.INDEX_MAP[index_name]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Fetching historical {index_name.upper()} constituents...", total=1)
            
            async with FMPClient(api_key=self.api_key) as client:
                df = await client.get_historical_constituents(index=fmp_index)
                progress.update(task, advance=1)
                
            return df
    
    async def get_index_symbols(self, index_name: str) -> List[str]:
        """Get the current symbols in an index.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
            
        Returns
        -------
        List[str]
            List of symbols in the index
        """
        constituents = await self.get_index_constituents(index_name)
        
        if constituents.empty:
            return []
            
        return list(constituents["symbol"].unique())
    
    async def get_symbols_with_dates(
        self,
        index_name: str
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Get symbols with start and end dates for an index.
        
        This combines current constituents with historical changes to create
        a complete history of index membership.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
            
        Returns
        -------
        Dict[str, Dict[str, Optional[str]]]
            Dictionary mapping symbols to their start and end dates in the index
        """
        # Get current constituents
        current = await self.get_index_constituents(index_name)
        
        # Get historical changes
        historical = await self.get_historical_constituents(index_name)
        
        # Current symbols are all active
        symbols = {
            row["symbol"]: {"start_date": row.get("date_added"), "end_date": None}
            for _, row in current.iterrows()
        }
        
        # Process historical removals
        if not historical.empty:
            for _, row in historical.iterrows():
                if pd.notna(row["removed_ticker"]) and row["removed_ticker"]:
                    symbol = row["removed_ticker"]
                    
                    # If symbol already in our list, update end date
                    if symbol in symbols and symbols[symbol]["end_date"] is None:
                        symbols[symbol]["end_date"] = row["date"]
                        
                # If it's an addition, check if we need to add or update
                if pd.notna(row["symbol"]) and row["symbol"]:
                    symbol = row["symbol"]
                    date = row["date"]
                    
                    if symbol not in symbols:
                        symbols[symbol] = {"start_date": date, "end_date": None}
                    elif symbols[symbol]["start_date"] is None or date < symbols[symbol]["start_date"]:
                        symbols[symbol]["start_date"] = date
        
        return symbols
    
    async def update_index_file(self, index_name: str) -> bool:
        """Update the index constituent file.
        
        Creates or updates a CSV file with index constituents and their
        membership dates.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not self.save_dir:
            raise ValueError("save_dir must be provided to update index file")
            
        try:
            # Get symbols with dates
            symbols_with_dates = await self.get_symbols_with_dates(index_name)
            
            if not symbols_with_dates:
                console.print(f"[bold red]No constituents found for {index_name}[/bold red]")
                return False
                
            # Create file path
            file_path = self.save_dir / f"{index_name}_constituents.csv"
            
            # Write to CSV
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["symbol", "start_date", "end_date"])
                
                for symbol, dates in symbols_with_dates.items():
                    writer.writerow([
                        symbol,
                        dates["start_date"] or "",
                        dates["end_date"] or ""
                    ])
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error updating index file: {e}[/bold red]")
            return False
    
    async def get_constituents_as_of_date(
        self,
        index_name: str,
        date: Union[str, datetime, pd.Timestamp]
    ) -> Set[str]:
        """Get the constituents of an index as of a specific date.
        
        Parameters
        ----------
        index_name : str
            Index name (sp500, dow, nasdaq)
        date : Union[str, datetime, pd.Timestamp]
            Reference date
            
        Returns
        -------
        Set[str]
            Set of symbols in the index as of the specified date
        """
        # Convert date to string format
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        
        # Get symbols with dates
        symbols_with_dates = await self.get_symbols_with_dates(index_name)
        
        # Filter symbols that were in the index on the specified date
        constituents = set()
        
        for symbol, dates in symbols_with_dates.items():
            start_date = dates["start_date"]
            end_date = dates["end_date"]
            
            # Skip if no start date
            if not start_date:
                continue
                
            # Check if symbol was in index on date
            if start_date <= date_str and (end_date is None or date_str <= end_date):
                constituents.add(symbol)
                
        return constituents 