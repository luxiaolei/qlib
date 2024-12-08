import numpy as np
import pandas as pd

from qlib.data.ops import ElemOperator


class LastNToString(ElemOperator):
    """Convert last N elements of a feature into a string representation.
    
    Parameters
    ----------
    feature : Expression
        The input feature instance
    n : int, optional
        Number of last elements to convert to string. Defaults to 3.
    separator : str, optional
        String separator between elements. Defaults to ','.
        
    Returns
    -------
    Expression
        A feature instance with the last N elements converted to string
        
    Examples
    --------
    >>> # Convert last 3 close prices to string
    >>> LastNToString("$close", n=3)
    >>> # Result example: "10.5,11.2,10.8"
    """
    
    def __init__(self, feature, n: int = 3, separator: str = ','):
        super().__init__(feature)
        self.n = n
        self.separator = separator
        
    def _load_internal(self, instrument, start_index, end_index, freq):
        # Load the original series
        series = self.feature.load(instrument, start_index, end_index, freq)
        
        # Create a rolling window view of the last N elements
        def to_string(x):
            # Get last N valid (non-NaN) values
            valid_values = x[~np.isnan(x)][-self.n:]
            # Convert to string with the specified separator
            return self.separator.join(map(str, valid_values))
            
        # Apply rolling window conversion
        result = series.rolling(window=self.n, min_periods=1).apply(
            lambda x: to_string(x)
        )
        
        return result
        
    def get_extended_window_size(self):
        """Get the extended window size required for this operator.
        
        Returns
        -------
        tuple
            (left_extension, right_extension)
        """
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        # We need N-1 additional previous values
        return lft_etd + (self.n - 1), rght_etd 
    
class UpperBarrierTouch(ElemOperator):
    """Calculate the first timestep where price touches an upper barrier within next N steps.
    
    The upper barrier is calculated using volatility: price + (volatility * multiplier)
    
    Parameters
    ----------
    base_feature : Expression
        The input feature instance (typically high prices)
    n : int
        Number of forward steps to check for barrier touch
    volatility_multiplier : float, optional
        Multiplier for volatility to set barrier height. Defaults to 1.0
        
    Returns
    -------
    Expression
        Index of first touch (0 to n-1) or -1 if no touch occurs
        
    Examples
    --------
    >>> # Check when price first touches barrier in next 5 steps
    >>> UpperBarrierTouch("$high", n=5, volatility_multiplier=1.0)
    >>> # Result example: 2 (touched on 3rd step) or -1 (never touched)
    """
    
    def __init__(self, base_feature="$high", n: int = 5, volatility_multiplier: float = 1.0):
        super().__init__(base_feature)
        self.n = n
        self.volatility_multiplier = volatility_multiplier
        
    def _load_internal(self, instrument, start_index, end_index, freq):
        # Load the price series
        series = self.feature.load(instrument, start_index, end_index, freq)
        
        # Calculate rolling volatility (using 20-day standard deviation)
        volatility = series.rolling(window=20, min_periods=1).std()
        
        def find_first_touch(prices, start_price, barrier):
            # Convert to numpy for faster processing
            price_array = prices.values
            
            # Check each step for barrier touch
            for i in range(len(price_array)):
                if np.isnan(price_array[i]):
                    continue
                if price_array[i] >= barrier:
                    return i
            return -1
        
        result = pd.Series(index=series.index, dtype=float)
        
        # For each time point
        for i in range(len(series) - self.n):
            current_price = series.iloc[i]
            current_vol = volatility.iloc[i]
            
            if np.isnan(current_price) or np.isnan(current_vol):
                result.iloc[i] = np.nan
                continue
                
            # Calculate barrier
            barrier = current_price + (current_vol * self.volatility_multiplier)
            
            # Get next n prices
            forward_prices = series.iloc[i+1:i+1+self.n]
            
            # Find first touch
            touch_idx = find_first_touch(forward_prices, current_price, barrier)
            result.iloc[i] = touch_idx
            
        # Fill remaining positions with NaN
        result.iloc[-self.n:] = np.nan
        
        return result
        
    def get_extended_window_size(self):
        """Get the extended window size required for this operator.
        
        Returns
        -------
        tuple
            (left_extension, right_extension)
        """
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        # Need 20 days for volatility calculation
        return lft_etd + 20, rght_etd + self.n
    
if __name__ == "__main__":
    import sys
    import traceback

    import pandas as pd
    from loguru import logger

    import qlib
    from qlib.data import D

    # Configure loguru for better error tracing
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="TRACE",  # Capture all levels
        diagnose=True,  # Enable exception diagnostics
        backtrace=True  # Show full traceback
    )
    
    # Add file logging
    logger.add(
        "debug.log",
        rotation="500 MB",
        level="DEBUG",
        diagnose=True,
        backtrace=True
    )

    try:
        # Initialize qlib
        provider_uri = "~/.qlib/qlib_data/cn_data"
        logger.debug(f"Initializing qlib with provider_uri: {provider_uri}")
        qlib.init(provider_uri=provider_uri, region="cn", custom_ops=[UpperBarrierTouch])
        
        # Example usage
        instruments = ["SH600000", "SH600004"]
        fields = [
            "UpperBarrierTouch($high, 5, 1.0)",
        ]
        
        start_time = "2022-01-01"
        end_time = "2022-01-10"
        
        
        # Debug the data loading process
        logger.debug("Loading raw close prices for debugging...")
        raw_close = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=start_time,
            end_time=end_time,
            freq="day"
        )
        
        test_data = D.features(
            instruments=instruments,
            fields=[
                "$close",
                "Lt(Ref(Max($close, 5), -5), Max($close, 2))",
                "Max(Ref($close, -3), 3)",
                ],
            start_time=start_time,
            end_time=end_time,
            freq="day"
        )

        
        feature_data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq="day"
        )
        
        logger.info("Feature data successfully retrieved")
        logger.info("\n" + str(feature_data))
        
    except Exception as e:
        logger.opt(exception=True).error("An error occurred:")
        
        # Get the full stack trace
        tb = traceback.format_exc()
        logger.error(f"Full traceback:\n{tb}")
        
        # Log additional debug information
        logger.debug("Debug information:")
        logger.debug(f"Exception type: {type(e).__name__}")
        logger.debug(f"Exception args: {e.args}")
        
        # If we have feature data available, log its state
        if 'raw_close' in locals():
            logger.debug(f"Last known raw_close data types:\n{raw_close.dtypes}")
            logger.debug(f"Raw close data sample:\n{raw_close.head()}")
            
    finally:
        logger.info("Script execution completed")
        