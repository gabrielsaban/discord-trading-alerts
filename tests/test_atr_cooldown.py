import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bot.alerts import AlertManager, RsiAlert
from bot.indicators import calculate_atr

# Configure logging
logger = logging.getLogger("discord_trading_alerts.tests.atr_cooldown")


def create_sample_data(length=200, volatility=0.02):
    """Create sample OHLCV data with adjustable volatility"""
    # Create timestamps
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(length)]
    timestamps.reverse()

    # Generate price data with random walk
    np.random.seed(42)  # For reproducibility
    price = 100.0
    prices = [price]

    for i in range(1, length):
        # Random walk with drift
        change = np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, volatility)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, volatility)) for p in prices],
            "close": [p * (1 + np.random.normal(0, volatility / 2)) for p in prices],
            "volume": [np.random.uniform(50, 200) for _ in range(length)],
        },
        index=timestamps,
    )

    return df


def test_atr_calculation():
    """Test the ATR calculation functionality"""
    logger.info("Testing ATR calculation...")

    # Create sample data with different volatility levels
    df_low_vol = create_sample_data(length=200, volatility=0.01)
    df_med_vol = create_sample_data(length=200, volatility=0.02)
    df_high_vol = create_sample_data(length=200, volatility=0.04)

    # Calculate ATR for each dataset
    atr_low = calculate_atr(df_low_vol, length=14, calculate_percentiles=True)
    atr_med = calculate_atr(df_med_vol, length=14, calculate_percentiles=True)
    atr_high = calculate_atr(df_high_vol, length=14, calculate_percentiles=True)

    # Verify ATR values reflect volatility
    logger.info(
        f"Low volatility ATR: {atr_low['ATR'].iloc[-1]:.4f}, "
        f"Percent: {atr_low['ATR_Percent'].iloc[-1]:.2f}%"
    )
    logger.info(
        f"Medium volatility ATR: {atr_med['ATR'].iloc[-1]:.4f}, "
        f"Percent: {atr_med['ATR_Percent'].iloc[-1]:.2f}%"
    )
    logger.info(
        f"High volatility ATR: {atr_high['ATR'].iloc[-1]:.4f}, "
        f"Percent: {atr_high['ATR_Percent'].iloc[-1]:.2f}%"
    )

    # Verify percentiles are calculated
    logger.info(
        f"ATR Percentiles - Low: {atr_low['ATR_Percentile'].iloc[-1]:.1f}, "
        f"Med: {atr_med['ATR_Percentile'].iloc[-1]:.1f}, "
        f"High: {atr_high['ATR_Percentile'].iloc[-1]:.1f}"
    )


def test_dynamic_cooldown():
    """Test the dynamic cooldown adjustment based on ATR"""
    logger.info("Testing dynamic cooldown adjustment...")

    # Create sample data with different volatility levels
    df_low_vol = create_sample_data(length=200, volatility=0.01)
    df_high_vol = create_sample_data(length=200, volatility=0.04)

    # Calculate ATR
    atr_low = calculate_atr(df_low_vol, length=14, calculate_percentiles=True)
    atr_high = calculate_atr(df_high_vol, length=14, calculate_percentiles=True)

    # Create market data with ATR
    market_low = df_low_vol.copy()
    market_low["ATR"] = atr_low["ATR"].iloc[-1]
    market_low["ATR_Percent"] = atr_low["ATR_Percent"].iloc[-1]
    # Force low percentile to ensure shorter cooldown
    market_low["ATR_Percentile"] = 10.0  # Bottom 25% -> -25% adjustment

    market_high = df_high_vol.copy()
    market_high["ATR"] = atr_high["ATR"].iloc[-1]
    market_high["ATR_Percent"] = atr_high["ATR_Percent"].iloc[-1]
    # Force high percentile to ensure longer cooldown
    market_high["ATR_Percentile"] = 90.0  # Top 25% -> +25% adjustment

    # Create alert managers
    mgr_low = AlertManager()
    mgr_high = AlertManager()

    # Test cooldown adjustment for different intervals
    test_intervals = ["5m", "15m", "1h", "4h"]

    for interval in test_intervals:
        # Get base cooldown
        base_cooldown = mgr_low.timeframe_cooldowns.get(interval, 60)

        # Calculate adjusted cooldowns
        cooldown_low = mgr_low._get_atr_adjusted_cooldown(
            base_cooldown, interval, "BTCUSDT", market_low
        )

        cooldown_high = mgr_high._get_atr_adjusted_cooldown(
            base_cooldown, interval, "BTCUSDT", market_high
        )

        logger.info(f"Interval: {interval}")
        logger.info(f"  Base cooldown: {base_cooldown} minutes")
        logger.info(f"  Low volatility adjusted cooldown: {cooldown_low} minutes")
        logger.info(f"  High volatility adjusted cooldown: {cooldown_high} minutes")

        # Verify high volatility should have longer cooldown
        if interval != "4h":  # 4h has fixed cooldown
            assert (
                cooldown_high > cooldown_low
            ), f"High volatility should have longer cooldown for {interval}"


def test_override_logic():
    """Test the enhanced override logic for cooldowns"""
    logger.info("Testing enhanced override logic...")

    # Create alert manager
    manager = AlertManager()

    # Create test data
    df = create_sample_data(length=100)

    # Setup test conditions
    symbol = "BTCUSDT"
    alert_type = "RsiAlert"
    alert_subtype = "OVERSOLD"

    # Test extreme signal override
    logger.info("Testing extreme signal override...")

    # First trigger with a moderate signal
    now = datetime.now() - timedelta(minutes=5)  # 5 minutes ago
    manager.global_cooldowns[f"{symbol}_{alert_subtype}"] = {
        "timestamp": now,
        "interval": "5m",
        "strength": 5.0,
    }

    # Create extreme signal (RSI very low)
    extreme_message = f"🔴 **OVERSOLD** 🔴\nRSI at 18.5\nPrice: $45,000"

    # Check if extreme signal overrides cooldown
    can_trigger = manager._is_globally_cooled_down(
        symbol, alert_type, alert_subtype, "5m", extreme_message
    )

    logger.info(f"Extreme signal override result: {can_trigger}")

    # Test higher timeframe override
    logger.info("Testing higher timeframe override...")

    # Set cooldown from a lower timeframe
    now = datetime.now() - timedelta(minutes=10)  # 10 minutes ago
    manager.global_cooldowns[f"{symbol}_{alert_subtype}"] = {
        "timestamp": now,
        "interval": "5m",
        "strength": 6.0,
    }

    # Check if higher timeframe overrides lower timeframe cooldown
    higher_tf_message = f"🔴 **OVERSOLD** 🔴\nRSI at 28.5\nPrice: $44,500"
    can_trigger = manager._is_globally_cooled_down(
        symbol, alert_type, alert_subtype, "1h", higher_tf_message
    )

    logger.info(f"Higher timeframe override result: {can_trigger}")


def test_alert_batching():
    """Test the alert batching functionality"""
    logger.info("Testing alert batching...")

    # Create alert manager
    manager = AlertManager()

    # Add some alerts to the batch
    symbol = "BTCUSDT"
    alert_type = "RsiAlert"
    alert_subtype = "OVERSOLD"

    # Queue several messages
    messages = [
        f"🔴 **OVERSOLD** 🔴\nRSI at 29.5\nPrice: $45,100",
        f"🔴 **OVERSOLD** 🔴\nRSI at 28.2\nPrice: $44,900",
        f"🔴 **OVERSOLD** 🔴\nRSI at 27.8\nPrice: $44,750",
    ]

    intervals = ["5m", "15m", "1h"]

    # Queue messages
    for i, msg in enumerate(messages):
        manager._queue_for_batching(
            symbol, alert_type, alert_subtype, msg, intervals[i]
        )

    # Get batched alerts
    batched = manager.get_batched_alerts(symbol)

    if symbol in batched and batched[symbol]:
        logger.info(f"Batched alerts: {len(batched[symbol])}")
        for i, alert in enumerate(batched[symbol]):
            logger.info(f"Alert {i+1}: {alert['message'][:50]}...")
            logger.info(f"  Interval: {alert['interval']}")
            logger.info(f"  Is batched: {alert.get('is_batched', False)}")
    else:
        logger.error("No batched alerts found!")


if __name__ == "__main__":
    print("=== Testing ATR-Based Cooling System ===")

    # Run tests
    test_atr_calculation()
    print("\n")

    test_dynamic_cooldown()
    print("\n")

    test_override_logic()
    print("\n")

    test_alert_batching()

    print("\n=== Testing Complete ===")
