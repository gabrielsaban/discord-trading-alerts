import os
import sys
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import sqlite3
import logging

# Ensure bot module is in path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import bot modules after path setup
from bot.db import DatabaseManager
from bot.alerts import AlertManager, AlertCondition
from bot.indicators import validate_data


@pytest.fixture
def sample_ohlcv_data():
    """Return a sample DataFrame with OHLCV data for testing indicators"""
    # Create a fixed set of test data
    np.random.seed(42)  # For reproducible random data
    
    # Generate 100 candles of test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    
    # Generate test data with predictable patterns for testing indicators
    base_price = 100.0
    trend = np.concatenate([
        np.linspace(0, 15, 30),        # Uptrend
        np.linspace(15, 5, 20),        # Downtrend
        np.linspace(5, 10, 30),        # Uptrend
        np.linspace(10, 0, 20)         # Downtrend
    ])
    
    # Add some volatility/noise
    noise = np.random.normal(0, 1, 100)
    
    # Generate OHLC data
    close = base_price + trend + noise
    # Convert NumPy array to pandas Series before using shift
    close_series = pd.Series(close)
    open_prices = close_series.shift(1).fillna(close[0] - 1).values
    high = np.maximum(open_prices, close) + np.random.uniform(0.1, 1.0, 100)
    low = np.minimum(open_prices, close) - np.random.uniform(0.1, 1.0, 100)
    
    # Generate volume with occasional spikes
    volume = np.random.normal(1000, 200, 100)
    # Add some volume spikes at key points
    spike_indices = [10, 25, 50, 75]
    for idx in spike_indices:
        volume[idx] = volume[idx] * 3
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def in_memory_db():
    """Create an in-memory database for testing"""
    # Create test database in memory
    db = DatabaseManager(":memory:")
    
    # Create the schema
    db.create_tables()
    
    # Return the db instance
    yield db
    
    # Clean up
    db.close()


@pytest.fixture
def alert_manager():
    """Create an alert manager for testing"""
    return AlertManager()


class MockAlert(AlertCondition):
    """Mock alert class for testing alert manager"""
    def __init__(self, symbol: str, should_trigger: bool = False):
        super().__init__(symbol)
        self.should_trigger = should_trigger
        self.was_checked = False
    
    def check(self, df: pd.DataFrame):
        self.was_checked = True
        if self.should_trigger:
            return f"MOCK ALERT: {self.symbol}"
        return None


@pytest.fixture
def mock_alert():
    """Create a mock alert for testing"""
    return MockAlert 