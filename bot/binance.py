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
    pandas.DataFrame or None
        DataFrame with OHLCV data and properly formatted datetime index, or None on error
    """
    base_url = "https://api.binance.com/api/v3/klines"

    # Create empty DataFrame with correct columns for error cases
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Binance: {e}")
            return None

        # Handle empty response with empty DataFrame with correct columns
        if not data:
            logger.warning(f"Empty response from Binance for {symbol} ({interval})")
            return empty_df

        # Respect the requested limit parameter
        if len(data) > limit:
            logger.debug(f"Trimming response to requested limit of {limit} candles")
            data = data[:limit]

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
            if df.shape[1] < 6:  # Need timestamp, open, high, low, close, volume
                logger.error(
                    f"Malformed data from Binance API for {symbol}: not enough columns"
                )
                return None

            # Convert numeric columns to float
            for col in ["open", "high", "low", "close", "volume"]:
                try:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting column {col} to float: {e}")
                    # Try to handle it gracefully - replace invalid values with NaN
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        logger.error(
                            f"Failed to convert {col} to numeric. Returning None."
                        )
                        return None

            # Check if we have NaN values in critical columns
            if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
                logger.error(f"NaN values found in critical columns for {symbol}")
                return None

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

    except Timeout as e:
        logger.error(f"Request to Binance API timed out for {symbol}: {e}")
        return None
    except ConnectionError as e:
        logger.error(f"Connection error accessing Binance API for {symbol}: {e}")
        return None
    except RequestException as e:
        logger.error(f"Error fetching data from Binance API for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching data from Binance for {symbol}: {e}")
        return None


def fetch_market_data(symbol="BTCUSDT", interval="15m", limit=100, force_refresh=False):
    """
    Fetch and format market data ready for technical analysis

    Parameters:
    -----------
    symbol : str
        Trading pair (e.g., 'BTCUSDT')
    interval : str
        Kline interval ('1m', '3m', '5m', '15m', '30m', '1h', etc.)
    limit : int
        Number of candles to retrieve
    force_refresh : bool
        Ignored parameter - for compatibility only

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with OHLCV data (empty DataFrame with correct columns on empty results),
        or None on errors (Timeout, ConnectionError, RequestException, JSONDecodeError)
    """
    # Always fetch fresh data from API for testing
    try:
        df = get_binance_klines(symbol, interval, limit)

        # Return None on error conditions
        if df is None:
            logger.error(f"Failed to fetch data for {symbol} ({interval})")
            return None

        # If we got data back, store in cache if it has enough rows
        if not df.empty and len(df) >= 2:
            cache = get_cache()
            cache.put(symbol, interval, df)
        elif not df.empty and len(df) < 2:
            logger.warning(
                f"Not enough data points for {symbol} ({interval}), got {len(df)}"
            )

        # Return the fetched data directly
        return df

    except Timeout as e:
        logger.error(f"Timeout error in fetch_market_data for {symbol}: {e}")
        return None
    except ConnectionError as e:
        logger.error(f"Connection error in fetch_market_data for {symbol}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in fetch_market_data for {symbol}: {e}")
        return None
    except RequestException as e:
        logger.error(f"Request error in fetch_market_data for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in fetch_market_data for {symbol}: {e}")
        return None


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
        Ignored parameter - for compatibility only
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
                df = fetch_market_data(symbol, interval, limit)
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
    # Configure logging for interactive mode (only when running this file directly)
    # Check if the root logger already has handlers before configuring
    if not logging.root.handlers:
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
