#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script that demonstrates how to use the migrated FX data with QLib.
This script is designed to be run outside the QLib repository to avoid import issues.

Instructions for use:
1. Install QLib: pip install pyqlib
2. Run this script from any directory
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()

def run_qlib_example():
    """
    Run a basic QLib example with the FX data
    
    This example will:
    1. Initialize QLib with the forex data
    2. Fetch price data for major pairs
    3. Calculate simple moving averages and generate signals
    4. Print out sample results
    """
    # Check if QLib is installed
    try:
        import qlib
        from qlib.data import D
    except ImportError:
        console.print("[bold red]QLib is not installed. Please install it with: pip install pyqlib[/bold red]")
        return
    
    # Initialize QLib with the forex data
    console.print("[bold]Initializing QLib with forex data...[/bold]")
    qlib.init(
        provider_uri="~/.qlib/qlib_data/forex_data",
        region="custom",
    )
    
    # Define data loading parameters
    instruments = "major.txt"  # Use the major currency pairs
    start_time = "2020-01-01"
    end_time = "2023-12-31"
    fields = ["$close", "$open", "$high", "$low"]
    freq = "day"
    
    # Step 1: List available instruments
    console.print("[bold]Listing available instruments in major.txt...[/bold]")
    try:
        instrument_list = D.list_instruments(instruments=instruments)
        console.print(f"Available instruments: {', '.join(instrument_list)}")
        
        # Step 2: Load data for all instruments
        console.print(f"\n[bold]Loading price data ({freq}) for instruments from {start_time} to {end_time}...[/bold]")
        data = D.features(instrument_list, fields, start_time=start_time, end_time=end_time, freq=freq)
        
        # Reshape data to have instruments as columns
        close_data = data["$close"].unstack(level='instrument')
        
        # Display data shape
        console.print(f"Data shape: {close_data.shape}")
        
        # Step 3: Calculate simple moving averages
        console.print("\n[bold]Calculating simple moving averages...[/bold]")
        
        # Create a table for results
        table = Table(title="Simple Moving Average Crossover Signals")
        table.add_column("Instrument", style="cyan")
        table.add_column("Last Price", style="green")
        table.add_column("SMA(10)", style="yellow")
        table.add_column("SMA(50)", style="red")
        table.add_column("Signal", style="magenta")
        
        for column in close_data.columns:
            # Calculate moving averages
            sma_10 = close_data[column].rolling(window=10).mean()
            sma_50 = close_data[column].rolling(window=50).mean()
            
            # Generate signals
            # 1: Buy (fast crosses above slow)
            # -1: Sell (fast crosses below slow)
            # 0: Hold (no crossover)
            last_price = close_data[column].iloc[-1]
            last_sma_10 = sma_10.iloc[-1]
            last_sma_50 = sma_50.iloc[-1]
            
            if last_sma_10 > last_sma_50:
                signal = "BUY" if sma_10.iloc[-2] <= sma_50.iloc[-2] else "HOLD LONG"
                signal_style = "bold green"
            elif last_sma_10 < last_sma_50:
                signal = "SELL" if sma_10.iloc[-2] >= sma_50.iloc[-2] else "HOLD SHORT"
                signal_style = "bold red"
            else:
                signal = "NEUTRAL"
                signal_style = "bold yellow"
            
            # Add to table
            table.add_row(
                column,
                f"{last_price:.4f}",
                f"{last_sma_10:.4f}",
                f"{last_sma_50:.4f}",
                f"[{signal_style}]{signal}[/{signal_style}]"
            )
        
        console.print(table)
        
        # Step 4: Plot example chart for first instrument
        console.print("\n[bold]Generating chart for demonstration (if running in an environment with display)...[/bold]")
        try:
            plt.figure(figsize=(12, 6))
            
            first_instrument = close_data.columns[0]
            plt.plot(close_data[first_instrument].index, close_data[first_instrument], label=f"{first_instrument} Close")
            plt.plot(sma_10.index, sma_10, label="SMA(10)")
            plt.plot(sma_50.index, sma_50, label="SMA(50)")
            
            plt.title(f"{first_instrument} Price and Moving Averages")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            
            # Save the chart to a file
            plt.savefig("fx_chart_example.png")
            console.print("[green]Chart saved to fx_chart_example.png[/green]")
            
            # Try to display the chart if in interactive mode
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            
        except Exception as e:
            console.print(f"[yellow]Could not generate chart: {str(e)}[/yellow]")
    
    except Exception as e:
        console.print(f"[bold red]Error in QLib example: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()

def run_basic_data_analysis():
    """
    Run a basic data analysis without QLib
    
    This is a fallback if QLib is not available
    """
    console.print("[bold yellow]Falling back to basic data analysis without QLib[/bold yellow]")
    
    # Define paths
    forex_dir = os.path.expanduser("~/.qlib/qlib_data/forex_data")
    csv_dir = os.path.join(forex_dir, "csv_data", "fx_normalized")
    
    # Find a sample file
    import glob
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        console.print("[bold red]No CSV files found![/bold red]")
        return
    
    # Focus on major pairs
    major_pairs = ["EURUSD.csv", "GBPUSD.csv", "USDJPY.csv", "AUDUSD.csv"]
    sample_files = [f for f in csv_files if os.path.basename(f) in major_pairs]
    
    if not sample_files:
        sample_files = csv_files[:4]  # Just take first 4 files if no major pairs found
    
    # Analyze each file
    console.print("[bold]Analyzing sample FX data files...[/bold]")
    
    for file_path in sample_files:
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Ensure date column is a datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # Calculate moving averages
            df['SMA10'] = df['close'].rolling(window=10).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # Calculate crossover signals
            df['signal'] = 0
            df.loc[(df['SMA10'] > df['SMA50']) & (df['SMA10'].shift(1) <= df['SMA50'].shift(1)), 'signal'] = 1  # Buy
            df.loc[(df['SMA10'] < df['SMA50']) & (df['SMA10'].shift(1) >= df['SMA50'].shift(1)), 'signal'] = -1  # Sell
            
            # Print summary
            pair_name = os.path.basename(file_path).replace(".csv", "")
            console.print(f"\n[bold cyan]{pair_name} Analysis[/bold cyan]")
            
            # Create a results table
            table = Table(title=f"{pair_name} Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            # Add data to table
            table.add_row("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
            table.add_row("Data Points", str(len(df)))
            table.add_row("Average Price", f"{df['close'].mean():.4f}")
            table.add_row("Min Price", f"{df['close'].min():.4f}")
            table.add_row("Max Price", f"{df['close'].max():.4f}")
            table.add_row("Buy Signals", str(sum(df['signal'] == 1)))
            table.add_row("Sell Signals", str(sum(df['signal'] == -1)))
            
            console.print(table)
            
            # Try to plot
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(df['date'], df['close'], label=f"{pair_name} Close")
                plt.plot(df['date'], df['SMA10'], label="SMA(10)")
                plt.plot(df['date'], df['SMA50'], label="SMA(50)")
                
                # Mark buy and sell signals
                buy_signals = df[df['signal'] == 1]
                sell_signals = df[df['signal'] == -1]
                
                plt.scatter(buy_signals['date'], buy_signals['close'], color='green', marker='^', s=100, label='Buy')
                plt.scatter(sell_signals['date'], sell_signals['close'], color='red', marker='v', s=100, label='Sell')
                
                plt.title(f"{pair_name} Price and Moving Averages")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                
                # Save the chart to a file
                plt.savefig(f"{pair_name}_chart.png")
                console.print(f"[green]Chart saved to {pair_name}_chart.png[/green]")
                
                # Try to display the chart if in interactive mode
                plt.show(block=False)
                plt.pause(1)
                plt.close()
                
            except Exception as e:
                console.print(f"[yellow]Could not generate chart: {str(e)}[/yellow]")
        
        except Exception as e:
            console.print(f"[bold red]Error analyzing {os.path.basename(file_path)}: {str(e)}[/bold red]")

def main():
    """Main function"""
    console.print("[bold green]===== FX Data Usage Example =====[/bold green]")
    
    try:
        # Try to run the QLib example
        run_qlib_example()
    except Exception as e:
        console.print(f"[bold red]Error running QLib example: {str(e)}[/bold red]")
        # Fallback to basic data analysis
        run_basic_data_analysis()
    
    console.print("[bold green]Example completed![/bold green]")

if __name__ == "__main__":
    main() 