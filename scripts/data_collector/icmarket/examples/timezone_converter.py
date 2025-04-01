#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility script to convert timestamps between the TDS data timezone (GMT+2 with US DST) 
and other timezones. This can help users ensure proper timestamp alignment.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from rich.console import Console
from rich.table import Table

# Add the parent directory to sys.path
script_dir = Path(__file__).resolve().parent.parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# Import the TDS timezone info
from config import TDS_TIMEZONE, TDS_TIMEZONE_INFO

console = Console()

def convert_timestamp(timestamp, source_tz, target_tz):
    """
    Convert a timestamp from source timezone to target timezone
    
    Args:
        timestamp: The timestamp to convert (string or datetime)
        source_tz: Source timezone (pytz timezone object)
        target_tz: Target timezone (pytz timezone object)
        
    Returns:
        Converted timestamp as datetime
    """
    if isinstance(timestamp, str):
        # Parse the timestamp if it's a string
        timestamp = pd.to_datetime(timestamp)
    
    # For naive timestamps, assume they're in the source timezone
    if timestamp.tzinfo is None:
        timestamp = source_tz.localize(timestamp)
    
    # Convert to the target timezone
    return timestamp.astimezone(target_tz)

def display_timezone_info():
    """Display information about timezones used in the TDS data"""
    table = Table(title="TDS Data Timezone Information")
    
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("TDS Timezone", TDS_TIMEZONE)
    table.add_row("Description", TDS_TIMEZONE_INFO)
    
    # Add some notes about the timezone
    table.add_row("Notes", "GMT+2 with US DST means GMT+2 during standard time \n"
                          "and GMT+3 during daylight saving time periods")
    
    console.print(table)
    console.print()

def display_timezone_comparison():
    """Display a comparison of important timezones"""
    now = datetime.now(pytz.utc)
    
    table = Table(title=f"Current Time in Different Timezones (UTC: {now.strftime('%Y-%m-%d %H:%M:%S')})")
    
    table.add_column("Timezone", style="cyan")
    table.add_column("Abbreviation", style="yellow")
    table.add_column("Current Time", style="green")
    table.add_column("UTC Offset", style="magenta")
    
    # List of important timezones to display
    timezones = [
        ("UTC", "UTC"),
        ("Europe/London", "GMT/BST"),
        ("Europe/Zurich", "CET/CEST"),
        ("America/New_York", "EST/EDT"),
        ("Asia/Tokyo", "JST"),
        ("Australia/Sydney", "AEST/AEDT")
    ]
    
    for tz_name, abbr in timezones:
        tz = pytz.timezone(tz_name)
        current_time = now.astimezone(tz)
        offset = current_time.strftime("%z")
        table.add_row(tz_name, abbr, current_time.strftime("%Y-%m-%d %H:%M:%S"), f"UTC{offset}")
    
    console.print(table)
    console.print()

def convert_and_display_timestamp(timestamp_str):
    """Convert and display a timestamp in various timezones"""
    try:
        # Parse the timestamp
        timestamp = pd.to_datetime(timestamp_str)
        
        # Create a table for the results
        table = Table(title=f"Timestamp Conversion: {timestamp_str}")
        
        table.add_column("Timezone", style="cyan")
        table.add_column("Converted Time", style="green")
        table.add_column("UTC Offset", style="magenta")
        
        # List of timezones to convert to
        timezones = [
            "UTC",
            "Europe/London",
            "Europe/Zurich",
            "America/New_York",
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        
        # Use Europe/Athens as an approximation of GMT+2 with DST
        source_tz = pytz.timezone("Europe/Athens")
        
        for tz_name in timezones:
            target_tz = pytz.timezone(tz_name)
            converted = convert_timestamp(timestamp, source_tz, target_tz)
            offset = converted.strftime("%z")
            table.add_row(tz_name, converted.strftime("%Y-%m-%d %H:%M:%S"), f"UTC{offset}")
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]Error converting timestamp: {str(e)}[/bold red]")

def main():
    """Main function to run the timezone converter tool"""
    console.print("[bold]TDS Data Timezone Converter[/bold]")
    console.print("This tool helps you convert timestamps between the TDS data timezone and other timezones.\n")
    
    # Display timezone information
    display_timezone_info()
    
    # Display timezone comparison
    display_timezone_comparison()
    
    # Interactive mode
    console.print("[bold]Timestamp Converter[/bold]")
    console.print("Enter a timestamp to convert (format: YYYY-MM-DD HH:MM:SS) or 'q' to quit:")
    
    while True:
        timestamp_str = input("> ")
        if timestamp_str.lower() == 'q':
            break
        
        convert_and_display_timestamp(timestamp_str)

if __name__ == "__main__":
    main() 