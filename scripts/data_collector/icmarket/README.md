# Forex Data Migration Tool for QLib

This module provides tools to migrate forex (FX) data to QLib format. It supports multiple timeframes with an optimized workflow for M5 (5-minute) data which is the focus of the newest implementation.

## Features

- Migrates forex data from raw CSV format to QLib format
- Special focus on M5 (5-minute) data for optimal backtesting performance
- Categorizes forex pairs into appropriate instrument groups
- Creates calendar files for forex trading days
- Processes data in parallel for faster migration
- Sets volume to 1 to avoid potential errors in calculations
- Comprehensive data health checks and fixing utilities

## Data Source

The primary data source supported is TDS (Tick Data Suite) data from ICMarkets. This data has the following characteristics:

- **Timeframe**: M5 (5-minute) data
- **Timezone**: GMT+2 with US DST adjustments
- **Currency Pairs**: Major, minor, exotic, and commodity pairs
- **Format**: CSV files with time, open, high, low, close columns
- **Structure**: Organized by year directories (2003-2024)

### TDS Data Structure

The TDS data is organized in the following structure:

```
TDS/M5/
├── 2003/        # Each year has its own directory
│   ├── EURUSD.csv
│   ├── GBPUSD.csv
│   └── ...
├── 2004/
│   ├── EURUSD.csv
│   ├── GBPUSD.csv
│   └── ...
...
├── 2024/
    ├── EURUSD.csv
    ├── GBPUSD.csv
    └── ...
```

Each CSV file has the following format:

```
time,open,high,low,close
2024-01-02 00:00:00,1.10427,1.10431,1.10424,1.10425
2024-01-02 00:05:00,1.10424,1.1043,1.10424,1.10427
...
```

The migration tool combines data from all year directories for each symbol to create comprehensive historical datasets.

## Directory Structure

### Module Structure

This module is organized as follows:

```
icmarket/
├── core/               # Core MT5 integration code
│   ├── mt5_server.py   # MT5 WebSocket server
│   ├── mt5_qlib_client.py  # QLib client for MT5
│   └── ...
├── examples/           # Usage examples
│   ├── migrate_example.py
│   ├── check_fx_data_health.py
│   └── ...
├── tests/              # Unit tests
│   ├── test_fx_m5_migration.py
│   └── ...
├── barprovider/        # Bar data providers
│   ├── _base.py
│   └── _mt5barprovider.py
├── migrate_fx_m5.py    # Main migration script
├── config.py           # Configuration
├── fx_info.py          # FX pair information
└── README.md           # This file
```

### QLib Data Structure

The migrated data will be structured according to QLib's requirements:
```
~/.qlib/qlib_data/icm_fx_m5/
├── calendars/           # Trading calendar information
├── instruments/         # Instrument definition files
│   ├── all.txt          # All instruments
│   ├── major.txt        # Major currency pairs
│   ├── minor.txt        # Minor currency pairs
│   ├── exotic.txt       # Exotic currency pairs
│   └── commodity.txt    # Commodity currencies (XAUUSD, XAGUSD)
├── features/            # Binary data will be here after running dump_bin.py
└── csv_data/            # Normalized CSV files
```

## Usage

### Migrating M5 (5-minute) Data

The preferred workflow is to use the optimized M5 data migration:

```bash
# Navigate to the qlib root directory
cd /path/to/qlib

# Run the M5 migration script
python scripts/data_collector/icmarket/migrate_fx_m5.py \
    --source_dir /Volumes/extdisk/MyRepos/EAFramework/db/market_data/TDS/M5 \
    --qlib_dir ~/.qlib/qlib_data/icm_fx_m5 \
    --start_year 2010 \
    --end_year 2024 \
    --max_workers 8
```

The migration process:
1. Detects the TDS directory structure with year folders
2. Identifies all available symbols across all years
3. For each symbol, combines data from all years into a single dataset
4. Processes each symbol to QLib format
5. Creates instrument category files
6. Generates calendar files for all trading days

### Checking Data Health

After migration, it's recommended to check the data health:

```bash
python scripts/data_collector/icmarket/examples/check_fx_data_health.py
```

This will check for:
- Missing data points
- Large price jumps
- Missing required columns
- Missing factor values

### Fixing Common Data Issues

If issues are found during the health check, you can fix them with:

```bash
python scripts/data_collector/icmarket/examples/fix_fx_data_issues.py
```

This will automatically:
- Set volume to 1 for all data points (as required)
- Set factor to 1.0
- Interpolate missing data points
- Fix outliers (abnormal price jumps)

### Dumping to QLib Binary Format

After migration and health checks, dump the data to QLib binary format:

```bash
python scripts/dump_bin.py dump_all \
    --csv_path ~/.qlib/qlib_data/icm_fx_m5/csv_data \
    --qlib_dir ~/.qlib/qlib_data/icm_fx_m5 \
    --include_fields open,high,low,close,volume,factor
```

## Using the Data in QLib

Initialize QLib with the forex data:

```python
import qlib
from qlib.constant import REG_CN

# Initialize QLib with forex data
qlib.init(
    provider_uri='~/.qlib/qlib_data/icm_fx_m5',
    region=REG_CN,  # You can use CN region for forex
)

# Use specific instrument categories
from qlib.data import D

# Major pairs only
instruments = D.instruments("major.txt")  # or "minor.txt", "exotic.txt", "commodity.txt", "all.txt"
fields = ["$close", "$open", "$high", "$low"]
df = D.features(instruments, fields, start_time='2020-01-01', end_time='2023-01-01', freq='5t')
```

## Instrument Categories

The script categorizes forex pairs into the following groups:

1. **Major Pairs**: Major currency pairs involving USD (EURUSD, GBPUSD, etc.)
2. **Minor Pairs**: Currency pairs between non-USD major currencies (EURGBP, EURJPY, etc.)
3. **Exotic Pairs**: Pairs involving emerging or smaller economies (USDSGD, USDPLN, etc.)
4. **Commodity**: Commodity-related pairs (XAUUSD, XAGUSD)

Additional symbols that may be present in the TDS data include:
- **Indices**: US500 (S&P 500), USTEC (NASDAQ), DE40 (DAX)
- **Cryptocurrencies**: BTCUSD, ETHUSD

## Notes

- Forex data volume is set to 1 to avoid potential calculation errors
- No adjustment factors are needed for forex data, so `factor` is set to 1.0
- The data is in GMT+2 timezone with US DST adjustments
- For more accurate forex analysis, consider implementing custom factors that account for bid/ask spreads
- The M5 data is optimized for backtesting and strategy development

## MT5 Integration

For users with MetaTrader 5 installed, this module also provides integration for direct data access:

```bash
# Start the MT5 data server (Windows only)
python scripts/data_collector/icmarket/core/start_mt5_data_server.py

# Install the MT5 service (Windows only)
python scripts/data_collector/icmarket/core/install_mt5_service.py
```

See examples directory for more usage details.

## Examples

Check the `examples/` directory for usage examples and the `tests/` directory for verification tests. 