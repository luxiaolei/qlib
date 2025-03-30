#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to verify the fixed forex data by loading and displaying historical data
spanning multiple years.
"""

import qlib
from qlib.data import D
from qlib.config import REG_CN
from rich.console import Console
from rich.table import Table

console = Console()

def verify_forex_data():
    """Verify the fixed forex data by loading and displaying samples across years"""
    # Initialize QLib with the forex data
    qlib.init(provider_uri="/Users/xlmini/.qlib/qlib_data/forex_data", 
              region=REG_CN)
    
    # List of forex pairs to check
    forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # Define time periods to sample
    time_periods = [
        ("2003", "2003-05-05", "2003-05-10"),
        ("2010", "2010-01-04", "2010-01-08"),
        ("2015", "2015-01-05", "2015-01-09"),
        ("2020", "2020-01-06", "2020-01-10"),
        ("2024", "2024-01-02", "2024-01-05")
    ]
    
    # Check each forex pair
    for forex_pair in forex_pairs:
        console.print(f"\n[bold green]Checking {forex_pair} data across years[/bold green]")
        
        # Check if the instrument exists
        try:
            # For each time period
            for period_name, start_date, end_date in time_periods:
                table = Table(title=f"{forex_pair} - {period_name}")
                table.add_column("Date")
                table.add_column("Open", justify="right")
                table.add_column("High", justify="right")
                table.add_column("Low", justify="right")
                table.add_column("Close", justify="right")
                
                try:
                    # Get data
                    data = D.features(
                        [forex_pair],
                        ["$open", "$high", "$low", "$close"],
                        start_time=start_date,
                        end_time=end_date,
                        freq="day"
                    )
                    
                    # Check if we got data
                    if data is None or data.empty:
                        console.print(f"[yellow]No data found for {forex_pair} in {period_name}[/yellow]")
                        continue
                    
                    # Print results
                    for idx, row in data.iterrows():
                        date_str = idx[1].strftime("%Y-%m-%d")
                        table.add_row(
                            date_str,
                            f"{row['$open']:.5f}",
                            f"{row['$high']:.5f}",
                            f"{row['$low']:.5f}",
                            f"{row['$close']:.5f}"
                        )
                    
                    console.print(table)
                    
                except Exception as e:
                    console.print(f"[red]Error loading {forex_pair} data for {period_name}: {str(e)}[/red]")
                
        except Exception as e:
            console.print(f"[bold red]Error with {forex_pair}: {str(e)}[/bold red]")

    # Also verify intraday data for 2024
    console.print("\n[bold green]Checking intraday data for EURUSD (2024-01-02)[/bold green]")
    
    for freq in ["1min", "5min", "1h"]:
        try:
            table = Table(title=f"EURUSD - {freq} data (2024-01-02)")
            table.add_column("Timestamp")
            table.add_column("Open", justify="right")
            table.add_column("Close", justify="right")
            
            # Get intraday data
            data = D.features(
                ["EURUSD"],
                ["$open", "$close"],
                start_time="2024-01-02 00:00:00",
                end_time="2024-01-02 23:59:59",
                freq=freq
            )
            
            # Sample some rows (first 5)
            sample_data = data.head(5)
            
            for idx, row in sample_data.iterrows():
                timestamp = idx[1].strftime("%Y-%m-%d %H:%M:%S")
                table.add_row(
                    timestamp,
                    f"{row['$open']:.5f}",
                    f"{row['$close']:.5f}"
                )
            
            console.print(table)
            console.print(f"Total {freq} datapoints for 2024-01-02: {len(data)}")
            
        except Exception as e:
            console.print(f"[red]Error loading intraday data ({freq}): {str(e)}[/red]")
    
    console.print("\n[bold green]Data verification completed[/bold green]")

if __name__ == "__main__":
    verify_forex_data() 