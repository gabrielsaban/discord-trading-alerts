import requests
import pandas as pd
import pandas_ta as ta
import json
import logging
from datetime import datetime
from requests.exceptions import RequestException, Timeout, ConnectionError

# Set up logging
logger = logging.getLogger('trading_alerts.binance')

# Valid intervals that Binance accepts
VALID_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

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
        'symbol': symbol.upper(),  # Binance requires uppercase symbols
        'interval': interval,
        'limit': limit
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
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        try:
            # Create DataFrame with proper column names
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Check if we have enough columns in the data
            if df.shape[1] < 6:  # Need at least timestamp, open, high, low, close, volume
                logger.error(f"Malformed data from Binance API for {symbol}: not enough columns")
                return None
            
            # Convert numeric columns to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                try:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting column {col} to float: {e}")
                    # Try to handle it gracefully - replace invalid values with NaN
                    if col in ['open', 'high', 'low', 'close', 'volume']:  # Critical columns
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp to datetime and set as index
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting timestamp: {e}")
                # If timestamp conversion fails, use a range index
                df = df.reset_index(drop=True)
            
            # Keep only OHLCV columns and handle missing columns
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in response. Adding with NaN values.")
                    df[col] = float('nan')
            
            ohlcv_df = df[ohlcv_columns]
            
            # Drop rows with NaN in critical columns
            ohlcv_df = ohlcv_df.dropna(subset=['open', 'high', 'low', 'close'])
            
            return ohlcv_df
            
        except Exception as e:
            logger.error(f"Error processing Binance data for {symbol}: {e}")
            return None
            
    except Timeout:
        logger.error(f"Timeout error fetching data from Binance for {symbol}")
        return None
    except ConnectionError:
        logger.error(f"Connection error fetching data from Binance for {symbol}")
        return None
    except RequestException as e:
        logger.error(f"Request error fetching data from Binance for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching data from Binance for {symbol}: {e}")
        return None

def fetch_market_data(symbol="BTCUSDT", interval="15m", limit=100):
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
        
    Returns:
    --------
    pandas.DataFrame or None
        Clean DataFrame with OHLCV data or None if data couldn't be fetched
    """
    df = get_binance_klines(symbol, interval, limit)
    
    if df is not None:
        # Check if we have enough data for analysis
        if not df.empty and len(df) >= 2:  # Need at least 2 candles for most indicators
            return df
        elif not df.empty:
            logger.warning(f"Not enough data points for {symbol} ({interval}), got {len(df)}")
            return df  # Return what we have even if it's not enough
        else:
            logger.warning(f"Empty dataframe returned for {symbol} ({interval})")
            return df  # Return empty DataFrame instead of None
    else:
        logger.error(f"Failed to fetch data for {symbol} ({interval})")
        return None

if __name__ == "__main__":
    # Configure logging for interactive mode
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    df = fetch_market_data()
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print(df.head())
    else:
        print("Failed to fetch market data")
