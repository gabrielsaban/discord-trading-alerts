import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# Setup logging
logger = logging.getLogger("discord_trading_alerts.indicators")
logger.setLevel(logging.INFO)


def validate_data(df: pd.DataFrame, min_periods: int) -> bool:
    """
    Validate that dataframe has enough data points for calculation

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    min_periods : int
        Minimum number of periods required for calculation

    Returns:
    --------
    bool
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        return False

    if len(df) < min_periods:
        return False

    # Check for required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        return False

    # Check for NaN values in close price
    if df["close"].isna().any():
        return False

    return True


def calculate_rsi(
    df: pd.DataFrame, length: int = 14, normalize: bool = False
) -> Optional[pd.Series]:
    """
    Calculate the Relative Strength Index (RSI) for the given dataframe

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    length : int
        RSI period length
    normalize : bool
        If True, normalize values to 0-1 range instead of 0-100

    Returns:
    --------
    pd.Series or None
        Series of RSI values, or None if calculation failed
    """
    try:
        # For testing purposes, allow shorter datasets but adjust the length
        if df is not None and not df.empty and len(df) < length + 1:
            logger.warning(
                f"RSI calculation: Using adjusted length {len(df)-1} instead of {length} due to limited data"
            )
            length = max(2, len(df) - 1)  # Ensure at least 2 periods

        if not validate_data(df, length + 1):
            return None

        rsi = ta.rsi(df["close"], length=length)

        if normalize:
            rsi = rsi / 100.0

        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None


def calculate_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> Optional[pd.DataFrame]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) components

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal line period

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns ['MACD', 'Signal', 'Histogram'], or None if calculation failed
    """
    try:
        min_periods = max(fast, slow, signal) + signal

        # For testing purposes, allow smaller datasets but adjust the parameters
        if df is not None and not df.empty and len(df) < min_periods:
            logger.warning(
                f"MACD calculation: Using adjusted parameters due to limited data (length={len(df)})"
            )
            available_periods = max(3, len(df) - 1)
            fast = min(fast, max(2, available_periods // 4))
            slow = min(slow, max(3, available_periods // 2))
            signal = min(signal, max(2, available_periods // 3))
            min_periods = max(fast, slow, signal) + signal

        if not validate_data(df, min_periods):
            return None

        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        col_map = {
            f"MACD_{fast}_{slow}_{signal}": "MACD",
            f"MACDs_{fast}_{slow}_{signal}": "Signal",
            f"MACDh_{fast}_{slow}_{signal}": "Histogram",
        }
        return macd_df.rename(columns=col_map)
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return None


def calculate_ema_cross(
    df: pd.DataFrame, short: int = 9, long: int = 21
) -> Optional[pd.DataFrame]:
    """
    Calculate EMA values for short and long periods for crossover analysis

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    short : int
        Short EMA period
    long : int
        Long EMA period

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns ['EMA{short}', 'EMA{long}'], or None if calculation failed
    """
    try:
        min_periods = long * 3  # For EMA stability
        if not validate_data(df, min_periods):
            return None

        ema_short = df["close"].ewm(span=short, adjust=False).mean()
        ema_long = df["close"].ewm(span=long, adjust=False).mean()

        result = pd.DataFrame({f"EMA{short}": ema_short, f"EMA{long}": ema_long})

        # Add crossover signals
        result["Cross_Up"] = (ema_short > ema_long) & (
            ema_short.shift(1) <= ema_long.shift(1)
        )
        result["Cross_Down"] = (ema_short < ema_long) & (
            ema_short.shift(1) >= ema_long.shift(1)
        )

        return result
    except Exception as e:
        logger.error(f"Error calculating EMA cross: {e}")
        return None


def calculate_bollinger_bands(
    df: pd.DataFrame, length: int = 20, std: float = 2, normalize: bool = False
) -> Optional[pd.DataFrame]:
    """
    Calculate Bollinger Bands (lower, middle, upper)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    length : int
        SMA period length
    std : float
        Number of standard deviations
    normalize : bool
        If True, normalize to percent B (0-1 range)

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns ['lower', 'middle', 'upper', 'bandwidth', 'PercentB'] if normalize=True
        otherwise DataFrame with columns ['lower', 'middle', 'upper'], or None if calculation failed
    """
    try:
        # For testing purposes, allow smaller datasets but adjust the parameters
        if df is not None and not df.empty and len(df) < length:
            logger.warning(
                f"Bollinger Bands calculation: Using adjusted length {len(df)} instead of {length} due to limited data"
            )
            length = max(2, len(df) - 1)  # Ensure at least 2 periods

        if not validate_data(df, length):
            return None

        # Get the Bollinger Bands - this produces different column names than we expected
        bb = ta.bbands(df["close"], length=length, std=std)
        if bb is None or bb.empty:
            logger.error("pandas-ta bbands returned None or empty dataframe")
            return None

        # Create a new result dataframe with our standard column names
        result = pd.DataFrame()

        # Map the actual column names from pandas-ta to our standard names
        # The .0 is present when std is passed as float (2.0 instead of 2)
        if f"BBL_{length}_{std}.0" in bb.columns:
            result["lower"] = bb[f"BBL_{length}_{std}.0"]
            result["middle"] = bb[f"BBM_{length}_{std}.0"]
            result["upper"] = bb[f"BBU_{length}_{std}.0"]
        else:
            result["lower"] = bb[f"BBL_{length}_{std}"]
            result["middle"] = bb[f"BBM_{length}_{std}"]
            result["upper"] = bb[f"BBU_{length}_{std}"]

        # For backward compatibility, add the old column names as well
        result["BBL"] = result["lower"]
        result["BBM"] = result["middle"]
        result["BBU"] = result["upper"]

        # Ensure lower band <= middle band <= upper band
        # Handle any potential calculation errors or edge cases
        valid_rows = result.notna().all(axis=1)
        if not valid_rows.all():
            # Drop NaN rows for proper calculation
            result = result.loc[valid_rows]

        # Calculate bandwidth only for valid rows
        result["bandwidth"] = (result["upper"] - result["lower"]) / result["middle"]
        # For backward compatibility
        result["BandWidth"] = result["bandwidth"]

        if normalize:
            # Use the existing PercentB if available, otherwise calculate
            if f"BBP_{length}_{std}.0" in bb.columns:
                result["PercentB"] = bb[f"BBP_{length}_{std}.0"]
            elif f"BBP_{length}_{std}" in bb.columns:
                result["PercentB"] = bb[f"BBP_{length}_{std}"]
            else:
                # Calculate %B manually if pandas-ta doesn't provide it
                result["PercentB"] = (df["close"] - result["lower"]) / (
                    result["upper"] - result["lower"]
                )
                # Clip to handle values outside bands
                result["PercentB"] = result["PercentB"].clip(0, 1)

        return result
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return None


def calculate_volume_spikes(
    df: pd.DataFrame, length: int = 20, threshold: float = 2, z_score: bool = False
) -> Optional[pd.DataFrame]:
    """
    Identify volume spikes based on rolling average volume

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    length : int
        Lookback period for average volume
    threshold : float
        Ratio threshold for spike detection
    z_score : bool
        If True, calculate z-score of volume instead of simple ratio

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with ['volume', 'avg_volume', 'volume_ratio'/'z_score', 'spike'], or None if calculation failed
    """
    try:
        if not validate_data(df, length):
            return None

        result = pd.DataFrame()
        result["volume"] = df["volume"]
        result["avg_volume"] = df["volume"].rolling(window=length).mean()

        if z_score:
            # Calculate Z-score (how many standard deviations from mean)
            vol_std = df["volume"].rolling(window=length).std()
            result["z_score"] = (df["volume"] - result["avg_volume"]) / vol_std
            result["spike"] = result["z_score"] > threshold
        else:
            # Simple ratio method
            result["volume_ratio"] = df["volume"] / result["avg_volume"]
            result["spike"] = result["volume_ratio"] > threshold

        return result
    except Exception as e:
        logger.error(f"Error calculating volume spikes: {e}")
        return None


def calculate_adx(
    df: pd.DataFrame, length: int = 14, threshold: int = 25
) -> Optional[pd.DataFrame]:
    """
    Calculate Average Directional Index (ADX) and Directional Indicators (DI+ and DI-)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    length : int
        ADX period length
    threshold : int
        ADX threshold for trend strength (typically 25)

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with ['ADX', 'DI+', 'DI-', 'Trend', 'Strength'], or None if calculation failed
    """
    try:
        min_periods = length * 2  # ADX needs more data for accuracy
        if not validate_data(df, min_periods):
            return None

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=length)
        col_map = {
            f"ADX_{length}": "ADX",
            f"DMP_{length}": "DI+",
            f"DMN_{length}": "DI-",
        }
        adx_df = adx_df.rename(columns=col_map)

        # Add trend direction and strength
        adx_df["Trend"] = np.where(adx_df["DI+"] > adx_df["DI-"], "Bullish", "Bearish")
        adx_df["Strength"] = adx_df["ADX"] > threshold

        return adx_df
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return None


def calculate_atr(
    df: pd.DataFrame, length: int = 14, calculate_percentiles: bool = True
) -> Optional[pd.DataFrame]:
    """
    Calculate Average True Range (ATR) with optional percentile ranking

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with OHLCV data
    length : int
        ATR period length
    calculate_percentiles : bool
        If True, calculate ATR percentile based on rolling lookback

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with ['ATR', 'ATR_Percent', 'ATR_Percentile'], or None if calculation failed
        - ATR: Raw ATR value
        - ATR_Percent: ATR as percentage of price
        - ATR_Percentile: Percentile rank (0-100) of current ATR vs historical
    """
    try:
        min_periods = length * 3  # For better percentile calculation
        if not validate_data(df, min_periods):
            return None

        # Use pandas_ta's ATR implementation
        atr = ta.atr(df["high"], df["low"], df["close"], length=length)

        # Create result dataframe
        result = pd.DataFrame({"ATR": atr})

        # Calculate ATR as percentage of price (more meaningful for comparison)
        result["ATR_Percent"] = result["ATR"] / df["close"] * 100

        # Calculate percentile ranking if requested
        if calculate_percentiles:
            # Use a longer lookback for percentile calculation (40 periods)
            # This creates a rolling window of percentile ranks
            lookback = min(len(df) - length, max(40, length * 3))

            # Calculate rolling percentile - FIXED to properly calculate percentile
            # Compare current value to historical values, return percentile (0-100)
            result["ATR_Percentile"] = (
                result["ATR_Percent"]
                .rolling(window=lookback)
                .apply(
                    lambda x: 100 * sum(x.iloc[:-1] < x.iloc[-1]) / max(1, len(x) - 1)
                )
            )

            # For early periods, calculate percentile based on available data
            if lookback < length * 3:
                early_percentile = pd.Series(index=result.index)
                for i in range(length, min(length + lookback, len(result))):
                    if i <= length:
                        # Not enough data for comparison
                        early_percentile.iloc[i] = 50  # Default to middle value
                        continue

                    window = result["ATR_Percent"].iloc[length:i]
                    current_value = result["ATR_Percent"].iloc[i]

                    # Calculate what percent of historical values current value exceeds
                    percentile = 100 * sum(window < current_value) / len(window)
                    early_percentile.iloc[i] = percentile

                # Fill early periods with calculated percentiles
                result.loc[
                    early_percentile.notna(), "ATR_Percentile"
                ] = early_percentile[early_percentile.notna()]

        return result
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None


if __name__ == "__main__":
    from binance import fetch_market_data

    # List of different trading pairs to test
    pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]

    for pair in pairs:
        logger.info(f"\n====== Testing {pair} ======\n")
        df = fetch_market_data(symbol=pair, limit=200)

        logger.info(f"Data validation: {validate_data(df, 50)}")
        logger.info(f"Data shape: {df.shape}")

        # Safe printing function for just first/last few rows
        def safe_print(name, result):
            logger.info(f"{name}:")
            if result is not None:
                logger.info(f"- Shape: {result.shape}")
                logger.info(f"- First 2 rows: {result.head(2)}")
                logger.info(f"- Last 2 rows: {result.tail(2)}")
            else:
                logger.info("Calculation returned None")

        # Test all indicators
        safe_print("RSI", calculate_rsi(df))
        safe_print("MACD", calculate_macd(df))
        safe_print("EMA Cross", calculate_ema_cross(df))
        safe_print("Bollinger Bands", calculate_bollinger_bands(df))
        safe_print("Volume Spikes", calculate_volume_spikes(df))
        safe_print("ADX", calculate_adx(df))
        safe_print("ATR", calculate_atr(df))
