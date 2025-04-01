#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use migrated forex data for backtesting in QLib.
This example creates a simple moving average crossover strategy for forex pairs.
"""

import qlib
import pandas as pd
import numpy as np
from qlib.utils import init_instance_by_config
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from rich.console import Console

console = Console()

def init_qlib_forex():
    """Initialize QLib with forex data"""
    qlib.init(
        provider_uri='~/.qlib/qlib_data/forex_data',
        region='custom',
    )

def create_ma_crossover_signal(instruments, start_time, end_time, fast_window=5, slow_window=20):
    """
    Create moving average crossover signals
    
    Args:
        instruments: List of instruments to get data for
        start_time: Start time for data
        end_time: End time for data
        fast_window: Fast moving average window size
        slow_window: Slow moving average window size
        
    Returns:
        DataFrame with crossover signals
    """
    # Get price data
    fields = ['$close']
    freq = '1h'  # Use hourly data
    
    data = D.features(instruments, fields, start_time=start_time, end_time=end_time, freq=freq)
    
    # Reshape data to have instruments as columns
    console.print(f"Data shape: {data.shape}")
    data = data.unstack(level='instrument')
    data.columns = data.columns.droplevel(0)  # Drop 'close' level from MultiIndex
    
    # Generate signals for each instrument
    signals = pd.DataFrame(index=data.index)
    
    for instrument in data.columns:
        # Calculate moving averages
        price = data[instrument]
        ma_fast = price.rolling(window=fast_window).mean()
        ma_slow = price.rolling(window=slow_window).mean()
        
        # Generate crossover signals
        # 1: Buy (fast crosses above slow)
        # 0: Hold (no crossover)
        # -1: Sell (fast crosses below slow)
        signal = np.zeros_like(price)
        signal[(ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))] = 1  # Buy signal
        signal[(ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))] = -1  # Sell signal
        
        signals[instrument] = signal
    
    # Drop NaN values due to moving average calculation
    signals = signals.dropna()
    
    return signals

def run_forex_backtest():
    """Run a moving average crossover backtest on forex data"""
    # Parameters
    instruments = "major.txt"  # Major forex pairs
    start_time = "2021-01-01"
    end_time = "2022-12-31"
    
    # Get the list of instruments
    instrument_list = D.list_instruments(instruments=instruments, 
                                         start_time=start_time, 
                                         end_time=end_time)
    
    console.print(f"[bold]Running backtest on instruments:[/bold] {instrument_list}")
    
    # Create trading signals
    signals = create_ma_crossover_signal(
        instruments=instrument_list,
        start_time=start_time,
        end_time=end_time,
        fast_window=5,  # 5-hour MA
        slow_window=20  # 20-hour MA
    )
    
    # Configure backtest
    STRATEGY_CONFIG = {
        "class": "SignalStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": signals,
            "topk": 2,  # Trade top 2 pairs based on signal strength
            "n_drop": 1,  # Drop 1 pair when re-balancing
        },
    }
    
    EXECUTOR_CONFIG = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "1h",
            "generate_portfolio_metrics": True,
        },
    }
    
    # Special config for forex - tighter spreads
    backtest_config = {
        "start_time": signals.index[0],
        "end_time": signals.index[-1],
        "account": 10000,
        "benchmark": None,  # No benchmark for forex
        "exchange_kwargs": {
            "limit_threshold": 0.0001,  # Tighter spreads for forex
            "deal_price": "close",
            "open_cost": 0.0001,  # 1 pip spread (approximately)
            "close_cost": 0.0001,
            "min_cost": 0,
        },
    }
    
    # Initialize strategy and executor
    console.print("[bold]Initializing backtest strategy and executor...[/bold]")
    strategy = init_instance_by_config(STRATEGY_CONFIG)
    executor_config = EXECUTOR_CONFIG.copy()
    executor_instance = init_instance_by_config(executor_config)
    
    # Run backtest
    console.print("[bold]Running backtest...[/bold]")
    portfolio_metrics = backtest(
        executor=executor_instance,
        strategy=strategy,
        **backtest_config
    )
    
    # Print results
    console.print("\n[bold green]Backtest Results:[/bold green]")
    
    # Portfolio analysis
    analysis = risk_analysis(
        portfolio_metrics["portfolio"].iloc[:, 0],
        freq="1h"
    )
    
    # Print key metrics
    console.print(f"[bold]Annual Return:[/bold] {analysis['annual_return']:.2%}")
    console.print(f"[bold]Maximum Drawdown:[/bold] {analysis['max_drawdown']:.2%}")
    console.print(f"[bold]Sharpe Ratio:[/bold] {analysis['sharpe']:.2f}")
    console.print(f"[bold]Information Ratio:[/bold] {analysis['information_ratio']:.2f}")
    
    # Plot portfolio value
    console.print("\n[bold]To visualize the results, add this code:[/bold]")
    console.print("""
    import matplotlib.pyplot as plt
    
    # Plot portfolio value
    portfolio_metrics["portfolio"].iloc[:, 0].plot(figsize=(12, 6))
    plt.title("Forex Trading Strategy - Portfolio Value")
    plt.tight_layout()
    plt.show()
    """)

if __name__ == "__main__":
    # Initialize QLib with forex data
    init_qlib_forex()
    
    # Run the backtest
    run_forex_backtest() 