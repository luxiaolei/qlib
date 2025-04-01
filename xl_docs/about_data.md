# Custom Data Implementation in QLib

This document provides a comprehensive guide on implementing custom data in QLib, specifically for stock market data using Financial Modeling Prep (FMP) from M5 to Daily timeframes, and for forex data using ICMarket from M1 to Daily timeframes.

## Table of Contents
- [Data Structure in ~/.qlib](#data-structure-in-qlib)
- [Timezone Handling](#timezone-handling)
- [Stock Index Constitution Changes](#stock-index-constitution-changes)
- [Bid/Ask Handling](#bidask-handling)
- [Adjusted Price in Intraday Data](#adjusted-price-in-intraday-data)
- [Live Trading Data Feeding](#live-trading-data-feeding)
- [Point-in-Time (PIT) Data](#point-in-time-pit-data)
- [Implementation Steps](#implementation-steps)
  - [Stock Data Implementation](#stock-data-implementation)
  - [Forex Data Implementation](#forex-data-implementation)

## <a id="data-structure-in-qlib"></a>Data Structure in ~/.qlib

QLib expects data to be stored in a specific directory structure under `~/.qlib`. The standard structure is:

```
~/.qlib/qlib_data/[region]_data/
├── calendars/           # Trading calendar information
├── features/            # Feature data in binary format
│   ├── [symbol1]/
│   │   ├── [feature1].day.bin
│   │   ├── [feature2].day.bin
│   │   └── ...
│   ├── [symbol2]/
│   └── ...
├── instruments/         # Instrument definition files
│   ├── all.txt          # All instruments
│   ├── csi300.txt       # CSI300 index constituents
│   ├── sp500.txt        # S&P 500 index constituents
│   └── ...
└── financial/           # Point-in-Time (PIT) data
    ├── [symbol1]/
    │   ├── [feature1]_a.data    # Annual data
    │   ├── [feature1]_a.index   # Annual index
    │   ├── [feature1]_q.data    # Quarterly data
    │   ├── [feature1]_q.index   # Quarterly index
    │   └── ...
    ├── [symbol2]/
    └── ...
```

The data is stored in binary format for efficiency. Each symbol has its directory under `features/`, containing binary files for various features (open, high, low, close, volume, etc.).

For your specific implementation:
- Stock data from FMP will go into `~/.qlib/qlib_data/us_data/` or `~/.qlib/qlib_data/custom_stock_data/`
- Forex data from ICMarket will go into `~/.qlib/qlib_data/forex_data/`

## <a id="timezone-handling"></a>Timezone Handling

QLib handles timezones based on region settings. The key regions defined are:
- CN_TIME (China time)
- US_TIME (US Eastern time)
- TW_TIME (Taiwan time)

For your specific case:
- FMP stock data (America/New York timezone): Use US_TIME region setting when initializing QLib
- ICMarket forex data (GMT+2 with NY DST): You'll need to normalize this data

When mixing data from different timezones:
1. Decide on a standard timezone for your system (preferably UTC)
2. When collecting data, record the original timezone information
3. During preprocessing, convert all timestamps to your standard timezone
4. When dumping data to QLib format, ensure all timestamps are consistent

For forex data in GMT+2 with NY DST:
```python
# Sample code for timezone conversion
import pandas as pd
from datetime import datetime, timezone

# Convert GMT+2 to UTC
def convert_gmt2_to_utc(df):
    # Make sure the date column is timezone aware
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize('Etc/GMT-2')
    # Convert to UTC
    df['date'] = df['date'].dt.tz_convert('UTC')
    # Remove timezone info as QLib expects naive datetime
    df['date'] = df['date'].dt.tz_localize(None)
    return df
```

When initializing QLib, you can specify the timezone:
```python
import qlib
from qlib.constant import REG_US

# For stock data
qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_US)

# For custom timezone handling
qlib.init(
    provider_uri='~/.qlib/qlib_data/forex_data',
    calendar_provider="LocalCalendarProvider",
    expression_provider="LocalExpressionProvider",
    # Other custom settings as needed
)
```

## <a id="stock-index-constitution-changes"></a>Stock Index Constitution Changes

QLib handles stock index constitution changes through instrument files in the `instruments/` directory. Each instrument file (e.g., `sp500.txt`) specifies the start and end dates for each constituent:

```
# Format: symbol    start_date    end_date
AAPL    1980-12-12    2099-12-31
MSFT    1986-03-13    2099-12-31
GE      1962-01-02    2018-06-26
```

When stocks are added or removed from an index, the file is updated with appropriate dates. This allows QLib to use the correct constituents for any given historical date during backtesting.

Implementation steps:
1. Create or obtain the historical changes in index constituents
2. Format the data as tab-separated text with symbol, start_date, and end_date
3. Store in the `instruments/` directory
4. For new indices not provided by QLib, create custom index files

For FMP data, you can use the `scripts/data_collector/fmp/collector.py` module which has a `dump_constituent_data` method to create these files automatically.

## <a id="bidask-handling"></a>Bid/Ask Handling

QLib doesn't require explicit bid/ask data in the basic data format. Instead, it uses configuration-based approaches to simulate bid/ask spreads:

1. For backtesting, slippage models approximate the effect of bid/ask spreads
2. You can configure parameters like:
   - `limit_threshold`: Price threshold for order execution 
   - `deal_price`: How to determine execution price
   - `open_cost`: Trading cost for opening positions
   - `close_cost`: Trading cost for closing positions
   - `min_cost`: Minimum transaction cost

Example configuration:
```python
exchange_kwargs = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}
```

For more precise simulation in forex trading:
```python
forex_exchange_kwargs = {
    "limit_threshold": 0.0001,  # Tighter spreads for forex
    "deal_price": ("If($ask == 0, $bid, $ask)", "If($bid == 0, $ask, $bid)"),  # Conditional logic
    "open_cost": 0.0001,
    "close_cost": 0.0001,
    "min_cost": 0,
}
```

If you have actual bid/ask data, you can include it as additional features with names like "bid" and "ask", but standard QLib models and backtesting can work with approximations.

## <a id="adjusted-price-in-intraday-data"></a>Adjusted Price in Intraday Data

QLib uses adjustment factors to handle corporate actions (splits, dividends) for price continuity. For intraday data, this presents challenges since prices during the day may differ from adjusted closing prices.

The solution in QLib:
1. Daily data includes adjustment factors
2. Intraday data is adjusted using the same daily factors
3. The `calc_adjusted_price` function:
   - Multiplies price fields (open, high, low, close) by the factor
   - Divides volume by the factor
   - Aligns intraday data with daily data to ensure consistency

For your FMP intraday data:
1. First, prepare properly adjusted daily data with factors
2. When processing intraday data, align it with the daily data
3. Use the factor from the corresponding day to adjust intraday prices

Example implementation (based on FMP5minNormalize class):
```python
from qlib.data import D
import pandas as pd

def get_daily_factors(symbol, start_time, end_time):
    # Get daily data with factors
    fields = ["$factor", "$volume", "$close"]
    daily_df = D.features([symbol], fields, start_time=start_time, end_time=end_time, freq="day")
    return daily_df

def adjust_intraday_prices(intraday_df, daily_factors_df):
    # Map each intraday row to its corresponding daily factor
    intraday_df['date_key'] = intraday_df.index.date
    daily_factors_df['date_key'] = daily_factors_df.index.get_level_values('datetime').date
    
    # Merge the factor data
    merged_df = pd.merge(intraday_df, daily_factors_df[['$factor']], 
                         left_on='date_key', right_on='date_key')
    
    # Apply adjustments
    for price_col in ['open', 'high', 'low', 'close']:
        merged_df[f'adj_{price_col}'] = merged_df[price_col] * merged_df['$factor']
    
    merged_df['adj_volume'] = merged_df['volume'] / merged_df['$factor']
    
    return merged_df
```

For handling gaps between intraday raw prices and adjusted daily prices:
1. Always use the same adjustment methodology consistently
2. Consider adding a "data quality" flag feature to mark questionable data points
3. When live trading, be aware of potential adjustment changes after market close

## <a id="live-trading-data-feeding"></a>Live Trading Data Feeding

QLib has special handling for live trading data to ensure a seamless transition between historical and real-time data:

1. Live trading detection:
   - Check if the end_time is during market hours of the current day
   - Use different adjustment logic in this case

2. Implementation approach:
   - For historical data: Use stored adjustment factors
   - For live data: Use the last known factor from the previous day
   - Set `paused=0` to indicate the stock is actively trading

3. Data feeding options:
   - File-based: Continuously update CSV/data files and let QLib reload
   - API-based: Implement a custom data provider in QLib

For implementing a real-time data feed:
```python
from rich.console import Console
import asyncio
import pandas as pd
from qlib.utils import get_latest_record_date

console = Console()

class RealtimeDataFeed:
    def __init__(self, data_source, qlib_dir):
        self.data_source = data_source
        self.qlib_dir = qlib_dir
        self.console = Console()
        
    async def fetch_latest_data(self):
        """Fetch the latest data from your source"""
        # Implementation depends on your data source (FMP or ICMarket)
        pass
        
    async def process_and_store(self, new_data):
        """Process new data and store in QLib format"""
        # 1. Normalize data
        # 2. Apply adjustments
        # 3. Store in QLib format
        self.console.log(f"Processed {len(new_data)} new data points")
        
    async def run_feed(self, interval_seconds=60):
        """Run the data feed continuously"""
        while True:
            try:
                new_data = await self.fetch_latest_data()
                if not new_data.empty:
                    await self.process_and_store(new_data)
                else:
                    self.console.log("No new data received")
            except Exception as e:
                self.console.log(f"Error in data feed: {e}", style="bold red")
            
            await asyncio.sleep(interval_seconds)
```

When implementing live trading, consider:
1. Data latency and update frequency
2. Error handling and connection stability
3. Synchronization with trading decisions
4. Logging for debugging and monitoring

## <a id="point-in-time-pit-data"></a>Point-in-Time (PIT) Data

Point-in-Time (PIT) data refers to maintaining multiple versions of the same data point as they were known at different points in time. This is crucial for financial statement data that gets revised over time.

For example, a company might report earnings that are later restated. To avoid look-ahead bias, you need to use the data that was available at the time of the backtest, not the final revised data.

QLib's PIT data structure:
- Stored in the `financial/` directory
- Each feature contains 4 columns: date, period, value, _next
  - `date`: When the statement was published
  - `period`: The period the statement covers (e.g., quarterly or annual)
  - `value`: The reported value
  - `_next`: Pointer to the next revision (if any)

The data is sorted by publication date to ensure proper historical simulation.

For your implementation:
- For fundamental data that gets revised (like earnings reports), use the PIT format
- For market data that doesn't change historically (like prices), use the standard format

Example for collecting and formatting PIT data:
```python
from scripts.dump_pit import DumpPitData

# After preparing your data in CSV format
dump_pit = DumpPitData(
    csv_path="/path/to/your/financial/data/csv",
    qlib_dir="~/.qlib/qlib_data/us_data",
    date_column_name="announcement_date",
    period_column_name="fiscal_period",
    value_column_name="value",
    field_column_name="indicator"
)

# Dump the data in PIT format
dump_pit.dump(interval="quarterly")  # or "annual"
```

## <a id="implementation-steps"></a>Implementation Steps

### <a id="stock-data-implementation"></a>Stock Data Implementation

1. **Data Collection**:
   ```bash
   # Create directories
   mkdir -p ~/.qlib/csv_data/fmp_stock_data

   # Install required packages (ensure you're using uv)
   uv install pandas requests rich asyncio

   # Create a data collection script
   python scripts/data_collector/fmp/run.py download_data \
       --save_dir ~/.qlib/csv_data/fmp_stock_data \
       --qlib_dir ~/.qlib/qlib_data/us_data \
       --start 2018-01-01 \
       --end 2023-12-31 \
       --interval 5min
   ```

2. **Data Normalization**:
   ```bash
   python scripts/data_collector/fmp/run.py normalize_data \
       --source_dir ~/.qlib/csv_data/fmp_stock_data \
       --target_dir ~/.qlib/csv_data/fmp_stock_normalized
   ```

3. **Data Dumping**:
   ```bash
   python scripts/dump_bin.py dump_all \
       --csv_path ~/.qlib/csv_data/fmp_stock_normalized \
       --qlib_dir ~/.qlib/qlib_data/us_data \
       --include_fields open,high,low,close,volume,factor,vwap \
       --freq 5t  # 5-minute data
   ```

4. **Index Constituent Data**:
   ```bash
   python scripts/data_collector/fmp/utils.py
   ```

### <a id="forex-data-implementation"></a>Forex Data Implementation

1. **Create Custom Data Collector**:
   First, create a custom data collector for ICMarket data in `scripts/data_collector/icmarket/`:

   ```python
   # scripts/data_collector/icmarket/collector.py
   import pandas as pd
   from datetime import datetime, timedelta
   from pathlib import Path
   from rich.console import Console
   import asyncio
   
   console = Console()
   
   class ICMarketCollector:
       def __init__(self, save_dir, instruments, start_date, end_date):
           self.save_dir = Path(save_dir).expanduser()
           self.save_dir.mkdir(parents=True, exist_ok=True)
           self.instruments = instruments
           self.start_date = pd.Timestamp(start_date)
           self.end_date = pd.Timestamp(end_date)
           
       async def download_instrument(self, instrument):
           """Download data for a specific instrument"""
           # Implement your ICMarket API connection here
           console.log(f"Downloading data for {instrument}")
           
           # Create sample data for demonstration
           dates = pd.date_range(self.start_date, self.end_date, freq="1min")
           df = pd.DataFrame({
               "date": dates,
               "open": [100 + i * 0.01 for i in range(len(dates))],
               "high": [101 + i * 0.01 for i in range(len(dates))],
               "low": [99 + i * 0.01 for i in range(len(dates))],
               "close": [100.5 + i * 0.01 for i in range(len(dates))],
               "volume": [1000 + i for i in range(len(dates))],
               "symbol": instrument
           })
           
           # Save to CSV
           output_file = self.save_dir / f"{instrument}.csv"
           df.to_csv(output_file, index=False)
           console.log(f"Saved {len(df)} records to {output_file}")
           
       async def download_all(self):
           """Download data for all instruments"""
           tasks = [self.download_instrument(inst) for inst in self.instruments]
           await asyncio.gather(*tasks)
           
       def run(self):
           """Run the data collection"""
           asyncio.run(self.download_all())
   
   def main():
       forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
       collector = ICMarketCollector(
           save_dir="~/.qlib/csv_data/icmarket_fx",
           instruments=forex_pairs,
           start_date="2018-01-01",
           end_date="2023-12-31"
       )
       collector.run()
       
   if __name__ == "__main__":
       main()
   ```

2. **Create a Normalizer for Forex Data**:
   ```python
   # scripts/data_collector/icmarket/normalizer.py
   import pandas as pd
   from pathlib import Path
   from concurrent.futures import ProcessPoolExecutor
   from rich.console import Console
   
   console = Console()
   
   class ICMarketNormalizer:
       def __init__(self, source_dir, target_dir, max_workers=16):
           self.source_dir = Path(source_dir).expanduser()
           self.target_dir = Path(target_dir).expanduser()
           self.target_dir.mkdir(parents=True, exist_ok=True)
           self.max_workers = max_workers
           
       def normalize_file(self, file_path):
           """Normalize a single file"""
           symbol = file_path.stem
           console.log(f"Normalizing {symbol}")
           
           # Read data
           df = pd.read_csv(file_path)
           
           # Convert timestamps and handle timezone
           df["date"] = pd.to_datetime(df["date"])
           # Convert from GMT+2 to UTC
           df["date"] = df["date"] - pd.Timedelta(hours=2)
           
           # For forex, typically no adjustment factors needed
           df["factor"] = 1.0
           
           # Calculate derived fields
           df["vwap"] = df["close"]  # Simple approximation
           
           # Save normalized data
           output_file = self.target_dir / f"{symbol}.csv"
           df.to_csv(output_file, index=False)
           console.log(f"Saved normalized data to {output_file}")
           
       def normalize(self):
           """Normalize all files"""
           files = list(self.source_dir.glob("*.csv"))
           console.log(f"Found {len(files)} files to normalize")
           
           with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
               list(executor.map(self.normalize_file, files))
   
   def main():
       normalizer = ICMarketNormalizer(
           source_dir="~/.qlib/csv_data/icmarket_fx",
           target_dir="~/.qlib/csv_data/icmarket_fx_normalized"
       )
       normalizer.normalize()
       
   if __name__ == "__main__":
       main()
   ```

3. **Dump the Data to QLib Format**:
   ```bash
   python scripts/dump_bin.py dump_all \
       --csv_path ~/.qlib/csv_data/icmarket_fx_normalized \
       --qlib_dir ~/.qlib/qlib_data/forex_data \
       --include_fields open,high,low,close,volume,factor,vwap \
       --freq 1t  # 1-minute data
   ```

4. **Create Custom Calendar for Forex**:
   Since forex markets have different trading hours than stock markets:
   ```python
   # scripts/create_forex_calendar.py
   import pandas as pd
   from pathlib import Path
   from rich.console import Console
   
   console = Console()
   
   def create_forex_calendar(start_date, end_date, output_dir):
       """Create a forex trading calendar
       
       Forex markets typically operate 24/5 (Sunday evening to Friday evening)
       """
       output_dir = Path(output_dir).expanduser()
       output_dir.mkdir(parents=True, exist_ok=True)
       
       # Generate raw date range
       all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
       
       # Filter weekends (forex is 24/5)
       # In forex, the week typically starts on Sunday evening and ends Friday evening
       trading_dates = [d for d in all_dates if d.dayofweek < 5]  # Monday to Friday
       
       # Create calendar files for different frequencies
       calendars = {
           "day": pd.DatetimeIndex([d for d in trading_dates]),
           "1min": pd.DatetimeIndex(pd.date_range(start=start_date, end=end_date, freq='1min'))
       }
       
       # Filter minute-level calendar to remove weekends
       calendars["1min"] = calendars["1min"][calendars["1min"].dayofweek < 5]
       
       # Write calendar files
       for freq, cal in calendars.items():
           cal_file = output_dir / f"{freq}.txt"
           with open(cal_file, "w") as f:
               for d in cal:
                   f.write(f"{d.strftime('%Y-%m-%d %H:%M:%S')}\n")
           
           console.log(f"Created {freq} calendar with {len(cal)} entries at {cal_file}")
   
   if __name__ == "__main__":
       create_forex_calendar(
           start_date="2018-01-01", 
           end_date="2023-12-31",
           output_dir="~/.qlib/qlib_data/forex_data/calendars"
       )
   ```

5. **Initialize QLib with Custom Settings**:
   ```python
   import qlib
   
   # For forex data
   qlib.init(
       provider_uri="~/.qlib/qlib_data/forex_data",
       region="custom",
       custom_ops={
           "custom_time": lambda x: x  # Add any custom operations needed
       }
   )
   ```

By following these steps, you'll have a comprehensive custom data implementation for both stock market data from FMP and forex data from ICMarket in QLib. 