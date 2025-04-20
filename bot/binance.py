import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime

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
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(base_url, params=params)
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
        
        data = response.json()
        
        # Create DataFrame with proper column names
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns to float
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
            df[col] = df[col].astype(float)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Keep only OHLCV columns
        ohlcv_df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return ohlcv_df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance: {e}")
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
    pandas.DataFrame
        Clean DataFrame with OHLCV data
    """
    df = get_binance_klines(symbol, interval, limit)
    
    if df is not None:
        return df
    else:
        # Return empty dataframe with correct columns if fetch fails
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

if __name__ == "__main__":
    # Example usage
    df = fetch_market_data()
    print(df.head())
