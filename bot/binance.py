import json
import logging
from datetime import datetime

import pandas as pd
import pandas_ta as ta
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

from bot.data_cache import get_cache

# Set up logging
logger = logging.getLogger("trading_alerts.binance")

# Valid intervals that Binance accepts
VALID_INTERVALS = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]

# Map intervals to their frequency tier
INTERVAL_TO_FREQUENCY = {
    "1m": "high",
    "3m": "high",
    "5m": "high",
    "15m": "medium",
    "30m": "medium",
    "1h": "medium",
    "2h": "low",
    "4h": "low",
    "6h": "low",
    "8h": "low",
    "12h": "low",
    "1d": "low",
    "3d": "low",
    "1w": "low",
    "1M": "low",
}

# Default batch size for fetching multiple symbols
DEFAULT_BATCH_SIZE = 5  # Process 5 symbols at a time

# How often to force refresh data by interval (in seconds)
FORCE_REFRESH_SECONDS = {
    "high": 60,  # High frequency: force refresh every minute
    "medium": 300,  # Medium frequency: force refresh every 5 minutes
    "low": 900,  # Low frequency: force refresh every 15 minutes
}


def get_binance_klines(symbol="BTCUSDT", interval="15m", limit=100):
    """
    Fetch kline/candlestick data from Binance API

    Parameters:
    -----------
    symbol : str
        Trading pair (e.g., 'BTCUSDT')
    interval : str
        Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
    limit : int
        Number of candles to retrieve (max 1000)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data and properly formatted datetime index
    """
    base_url = "https://api.binance.com/api/v3/klines"

    # Parameter validation
    if not symbol:
        logger.error("Symbol cannot be empty")
        return None

    # Validate interval
    if interval not in VALID_INTERVALS:
        logger.warning(f"Invalid interval: {interval}. Using default '15m'")
        interval = "15m"

    # Validate limit
    try:
        limit = int(limit)
        if limit <= 0:
            logger.warning(f"Invalid limit: {limit}. Using default 100")
            limit = 100
        elif limit > 1000:
            logger.warning(f"Limit {limit} exceeds maximum 1000. Using 1000")
            limit = 1000
    except (ValueError, TypeError):
        logger.warning(f"Invalid limit: {limit}. Using default 100")
        limit = 100

    params = {
        "symbol": symbol.upper(),  # Binance requires uppercase symbols
        "interval": interval,
        "limit": limit,
    }

    try:
        logger.debug(f"Requesting data from Binance: {params}")
        response = requests.get(base_url, params=params, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise exception for HTTP errors

        # Binance kline data format:
        # [
        #   [
        #     1499040000000,      // Open time
        #     "8.01000000",       // Open
        #     "8.08000000",       // High
        #     "7.99000000",       // Low
        #     "8.05000000",       // Close
        #     "1111.00000000",    // Volume
        #     1499644799999,      // Close time
        #     "8760.27510000",    // Quote asset volume
        #     100,                // Number of trades
        #     "1105.00000000",    // Taker buy base asset volume
        #     "8751.87510000",    // Taker buy quote asset volume
        #     "0"                 // Ignore
        #   ]
        # ]

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Binance: {e}")
            return None

        # Handle empty response
        if not data:
            logger.warning(f"Empty response from Binance for {symbol} ({interval})")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        try:
            # Create DataFrame with proper column names
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Check if we have enough columns in the data
            if (
                df.shape[1] < 6
            ):  # Need at least timestamp, open, high, low, close, volume
                logger.error(
                    f"Malformed data from Binance API for {symbol}: not enough columns"
                )
                return None

            # Convert numeric columns to float
            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ]:
                try:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting column {col} to float: {e}")
                    # Try to handle it gracefully - replace invalid values with NaN
                    if col in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]:  # Critical columns
                        df[col] = pd.to_numeric(df[col], errors="coerce")

            # Convert timestamp to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            # Keep only the basic OHLCV columns for simplicity
            df = df[["open", "high", "low", "close", "volume"]].copy()

            return df

        except Exception as e:
            logger.error(f"Error processing Binance data for {symbol}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    except Timeout:
        logger.error(f"Request to Binance API timed out for {symbol}")
        return None
    except ConnectionError:
        logger.error(f"Connection error accessing Binance API for {symbol}")
        return None
    except RequestException as e:
        logger.error(f"Error fetching data from Binance API for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching data from Binance for {symbol}: {e}")
        return None


def fetch_market_data(symbol="BTCUSDT", interval="15m", limit=100, force_refresh=False):
    """
    Fetch and format market data ready for technical analysis, using cache when possible

    Parameters:
    -----------
    symbol : str
        Trading pair (e.g., 'BTCUSDT')
    interval : str
        Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', etc.)
    limit : int
        Number of candles to retrieve
    force_refresh : bool
        If True, always fetch fresh data from API instead of using cache

    Returns:
    --------
    pandas.DataFrame or None
        Clean DataFrame with OHLCV data or None if data couldn't be fetched
    """
    # Check cache first unless forced to refresh
    if not force_refresh:
        cache = get_cache()
        cached_df = cache.get(symbol, interval)
        if cached_df is not None:
            logger.debug(f"Using cached data for {symbol} ({interval})")
            return cached_df

    # Cache miss or forced refresh, fetch from API
    df = get_binance_klines(symbol, interval, limit)

    # Always return None when API returns None (error cases)
    if df is None:
        logger.error(f"Failed to fetch data for {symbol} ({interval})")
        return None

    # If we got data back
    if not df.empty and len(df) >= 2:  # Need at least 2 candles for most indicators
        # Store in cache
        cache = get_cache()
        cache.put(symbol, interval, df)
        return df
    elif not df.empty:
        logger.warning(
            f"Not enough data points for {symbol} ({interval}), got {len(df)}"
        )
        return df  # Return what we have even if it's not enough
    else:
        logger.warning(f"Empty dataframe returned for {symbol} ({interval})")
        return df  # Return empty DataFrame instead of None


def fetch_market_data_batch(
    symbols,
    interval="15m",
    limit=100,
    force_refresh=False,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Fetch market data for multiple symbols at once

    Parameters:
    -----------
    symbols : list
        List of symbols to fetch data for
    interval : str
        Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', etc.)
    limit : int
        Number of candles to retrieve
    force_refresh : bool
        If True, always fetch fresh data from API
    batch_size : int
        Number of symbols to process in each batch

    Returns:
    --------
    dict
        Dictionary mapping symbols to their dataframes
    """
    results = {}

    # Process symbols in batches to avoid overwhelming the API
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        logger.debug(
            f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}: {batch}"
        )

        for symbol in batch:
            try:
                df = fetch_market_data(symbol, interval, limit, force_refresh)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = None

    return results


def should_force_refresh(interval, last_check_time=None):
    """
    Determine if we should force refresh data based on interval and last check time

    Parameters:
    -----------
    interval : str
        The timeframe interval
    last_check_time : datetime, optional
        Last time this interval was checked

    Returns:
    --------
    bool
        True if data should be force refreshed
    """
    if last_check_time is None:
        return True

    # Get the frequency tier for this interval
    frequency = INTERVAL_TO_FREQUENCY.get(interval, "medium")

    # Get how long we should wait before forcing refresh
    refresh_seconds = FORCE_REFRESH_SECONDS.get(frequency, 300)

    # Calculate time elapsed since last check
    elapsed = (datetime.now() - last_check_time).total_seconds()

    # Force refresh if enough time has passed
    return elapsed >= refresh_seconds


if __name__ == "__main__":
    # Configure logging for interactive mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    df = fetch_market_data()
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print(df.head())
    else:
        print("Failed to fetch market data")
