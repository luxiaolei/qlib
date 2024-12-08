# Financial Modeling Prep (FMP) Data Collector

This module provides functionality to collect stock market data from Financial Modeling Prep API for US stocks and major indices.

## Features

- Daily and 5-minute data collection
- Support for US stocks and major indices (S&P 500, NASDAQ 100, Dow Jones)
- Parallel data collection with configurable workers
- Rate limiting and retry logic
- Data normalization to Qlib format
- Calendar-based data alignment

## Prerequisites

1. FMP API Key
   - Sign up at [Financial Modeling Prep](https://financialmodelingprep.com/developer)
   - Get your API key from the dashboard

2. Environment Setup
   ```bash
   export FMP_API_KEY="your_api_key_here"
   ```

## TODOs:

- [x] Implement async download data.
- [ ] Move download and update routines to another project and save data to a database, and expose methods to get historical data also a push mechanism for new data for update.
- [ ] Record also the first close price in the database. 
- [ ] Dont use the base class of normalizer and dump class as they are not suitable. Rewrite them.


## How to update M5 data ? Durning market hours and after market hours.

Problem:
- Durning market hours, the adjusted close price is not available, the daily factor is not available, which means we cannot normalize the M5 data, we can only estimate the factor using last day's data.
- After market hours, the adjusted close price is available. We also need to fix the previouse
  estimated factor using the real factor.
- Since we only have 740 calls per minutes, so its not possible to get all M5 data within few secs, hence we can only update a collection of instruments M5 data in real time. 

## How to dump M5 data efficiently durning market hours and after market hours.

## How to normalize data

You're absolutely right, and your analysis is spot-on. The discrepancy arises because the factor calculated in the current code does not account for the normalization step, and therefore, the relationship `factor * close = original close price` does not hold.

To ensure that:

1. **The first close is 1, and all other prices are adjusted based on it.**
2. **`factor * close = original close price`.**

We need to adjust both the calculation of the price adjustments and the way the `factor` is computed in the code.

Below, I'll guide you through the modifications needed in the code to achieve these requirements.

---

**Step-by-Step Code Modification:**

1. **Compute the Adjustment Factor (`adj_factor`).**

   Before any adjustments, calculate the adjustment factor using the original `adjclose` and `close` prices from Yahoo Finance data:

   ```python
   df['adj_factor'] = df['adjclose'] / df['close']
   ```

2. **Adjust the Price Fields to Reflect Corporate Actions.**

   Multiply all price fields by the `adj_factor` to adjust for splits, dividends, etc.:

   ```python
   df['open'] = df['open'] * df['adj_factor']
   df['high'] = df['high'] * df['adj_factor']
   df['low'] = df['low'] * df['adj_factor']
   df['close'] = df['close'] * df['adj_factor']  # Now, df['close'] equals df['adjclose']
   df['volume'] = df['volume'] / df['adj_factor']
   ```

3. **Normalize Prices so that the First Close is 1.**

   - **Get the First Adjusted Close Price:**

     ```python
     first_close = df['close'].iloc[0]
     ```

   - **Normalize All Price Fields:**

     ```python
     df['open'] = df['open'] / first_close
     df['high'] = df['high'] / first_close
     df['low'] = df['low'] / first_close
     df['close'] = df['close'] / first_close
     df['volume'] = df['volume'] * first_close
     ```

     Now, the first `close` in the dataset will be 1, satisfying your first requirement.

4. **Adjust the `factor` to Account for Normalization.**

   Since we've normalized the prices, we need to adjust the `factor` accordingly so that it can correctly revert the adjusted prices back to the original prices.

   - **Compute the New Factor:**

     ```python
     df['factor'] = first_close / df['adj_factor']
     ```

     **Explanation:**

     - **Original Adjustment Factor (`adj_factor`):**

       This factor adjusts prices for corporate actions:

       ```python
       adj_factor = adjclose / close
       ```

     - **New Factor (`factor`):**

       By computing `first_close / adj_factor`, we adjust the factor to account for normalization, ensuring that the product of the adjusted `close` and `factor` equals the original `close` price:

       ```python
       original_close = factor * adjusted_close
       ```

5. **Verify the Relationship Between Adjusted Data and Original Data.**

   - **Adjusted Close Price:**

     ```python
     adjusted_close = df['close']
     ```

   - **Original Close Price:**

     ```python
     original_close = df['factor'] * adjusted_close
     ```

   - **Verification:**

     ```python
     # Should be True for all rows
     all(df['original_close'] == df['close'] * df['adj_factor'])
     ```

6. **Clean Up the DataFrame (Optional).**

   If you no longer need the `adj_factor`, you can drop it:

   ```python
   df = df.drop(columns=['adj_factor'])
   ```

7. **Final Adjusted DataFrame.**

   Your `df` now contains:

   - Adjusted prices (`open`, `high`, `low`, `close`) where the first `close` is 1.
   - Adjusted `volume`.
   - A `factor` column that can be used to revert the adjusted prices back to the original prices using `original_price = factor * adjusted_price`.

---

**Updated Code Snippet:**

Here's the modified code incorporating the above steps:

```python
def adjust_prices(df):
    # Step 1: Compute adjustment factor from original data
    df['adj_factor'] = df['adjclose'] / df['close']

    # Step 2: Adjust prices using adj_factor
    df['open'] = df['open'] * df['adj_factor']
    df['high'] = df['high'] * df['adj_factor']
    df['low'] = df['low'] * df['adj_factor']
    df['close'] = df['close'] * df['adj_factor']
    df['volume'] = df['volume'] / df['adj_factor']

    # Step 3: Get first adjusted close (after step 2)
    first_close = df['close'].iloc[0]

    # Step 4: Normalize prices by first_close
    df['open'] = df['open'] / first_close
    df['high'] = df['high'] / first_close
    df['low'] = df['low'] / first_close
    df['close'] = df['close'] / first_close
    df['volume'] = df['volume'] * first_close

    # Step 5: Compute factor
    df['factor'] = first_close / df['adj_factor']

    # Optional: Drop adj_factor if no longer needed
    df = df.drop(columns=['adj_factor'])

    return df
```

**Usage Example:**

```python
# Assuming df is your DataFrame with columns: ['open', 'high', 'low', 'close', 'adjclose', 'volume']
df = adjust_prices(df)

# Now, df['close'].iloc[0] will be 1
# And to get the original close price back:
df['original_close'] = df['factor'] * df['close']
```

---

**Explanation of Adjustments:**

- **Normalization:**

  By dividing all adjusted prices by `first_close`, we ensure that the first `close` is 1, and all other prices are scaled accordingly.

- **Factor Adjustment:**

  The original `adj_factor` only accounts for corporate actions. By adjusting it to `factor = first_close / adj_factor`, we incorporate the effect of normalization into the factor. This ensures that:

  ```python
  original_price = factor * adjusted_price
  ```

  This satisfies your second requirement.

---

**Verification:**

Let's verify that `factor * close = original close price`.

- **From the Adjusted Data:**

  ```python
  adjusted_close = (original_close * adj_factor) / first_close
  ```

- **Compute Factor:**

  ```python
  factor = first_close / adj_factor
  ```

- **Recover Original Close Price:**

  ```python
  original_close = factor * adjusted_close
                = (first_close / adj_factor) * (original_close * adj_factor) / first_close
                = original_close
  ```

  The computation confirms that the relationship holds true.

---

**Adjusting Volume (Optional):**

The volume adjustment in the current code inverts the price adjustments. After adjusting for corporate actions and normalization, the volume is adjusted as:

```python
df['volume'] = df['volume'] / df['adj_factor']  # Adjust for corporate actions
df['volume'] = df['volume'] * first_close       # Adjust for normalization
```

Ensure that your volume adjustments align with how you interpret the effect of corporate actions on volume.

---

**Conclusion:**

By modifying the code as outlined above, we align the data processing with the documentation and ensure that:

1. **The first `close` is 1, and all other prices are adjusted based on it.**

2. **Users can retrieve the original trading price using `original_price = factor * adjusted_price`.**

---

**Note:** It's crucial to ensure that data integrity is maintained throughout the adjustment process. Always verify the adjusted data against known values when possible.
