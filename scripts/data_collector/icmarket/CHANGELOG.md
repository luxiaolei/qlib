# Changelog for ICMarket Data Collector

## [1.0.0] - 2023-03-31

### Code Reorganization
- Improved directory structure for better maintainability and organization
- Created dedicated subdirectories with appropriate __init__.py files:
  - `core/`: Contains MT5 integration code
  - `tests/`: Centralized location for all unit tests
  - `examples/`: Sample code and utilities
  - `barprovider/`: Bar data providers for various data sources

### Refactoring
- Moved test files from the main directory to the tests/ directory
- Moved MT5-related modules to the core/ directory
- Created proper import paths in __init__.py files
- Added verification script for import validation

### Cleanup
- Removed obsolete migration scripts:
  - Removed fix_calendar.py (integrated into migrate_fx_m5.py)
  - Removed fix_migration.py (integrated into migrate_fx_m5.py)
  - Removed migrate_eaframework.py (replaced by migrate_fx_m5.py)

### Documentation
- Updated README.md with the new directory structure
- Added module docstrings to __init__.py files
- Created detailed comments in the code 