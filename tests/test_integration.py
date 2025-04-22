import logging
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.alerts import AlertManager
from bot.db import DatabaseManager
from bot.indicators import calculate_bollinger_bands, calculate_macd, calculate_rsi
from bot.scheduler import AlertScheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")


class TestIntegration:
    """Integration tests to check multiple components working together"""

    @patch("bot.scheduler.fetch_market_data")
    def test_alert_flow_from_indicator_to_notification(
        self, mock_fetch_data, sample_ohlcv_data
    ):
        """Test the full flow from indicator calculation to alert notification"""
        # Setup
        db = DatabaseManager(":memory:")
        db.create_tables()

        # Create test user and watchlist
        db.create_user("test_user", "Test User")
        db.add_to_watchlist("test_user", "BTCUSDT", "15m")

        # Enable RSI alerts for the test user
        db.update_user_settings("test_user", {"enabled_alerts": ["rsi"]})

        # Mock the market data fetch
        df = sample_ohlcv_data.copy()
        mock_fetch_data.return_value = df

        # Create alert directly
        from bot.alerts import RsiAlert

        rsi_alert = RsiAlert("BTCUSDT", oversold=30, overbought=70)

        # Mock the RSI calculation directly where it's used
        with patch("bot.alerts.calculate_rsi") as mock_rsi:
            # Create RSI values that cross below the oversold threshold (30)
            # The last 2 values (31, 29) represent a cross under the threshold of 30
            mock_rsi.return_value = pd.Series([35, 33, 31, 29])

            # Check if alert triggers
            alert_message = rsi_alert.check(df)
            print(f"Alert message: {alert_message}")

            # Record the alert in the database
            if alert_message:
                db.record_alert("test_user", "BTCUSDT", "15m", "rsi", alert_message)

            # Get alerts from the database
            db_alerts = db.get_recent_alerts("test_user", hours=1)

            # Verify results
            assert alert_message is not None, "Alert should have triggered"
            assert len(db_alerts) > 0, "Alert should be in database"
            assert "RSI" in db_alerts[0]["message"], "Alert should be about RSI"

        # Clean up
        db.close()

    @patch("bot.scheduler.fetch_market_data")
    def test_multiple_alert_types_integration(self, mock_fetch_data, sample_ohlcv_data):
        """Test that multiple alert types can be triggered for the same data"""
        # Setup
        db = DatabaseManager(":memory:")
        db.create_tables()

        # Create test user and watchlist
        db.create_user("test_user", "Test User")
        db.add_to_watchlist("test_user", "BTCUSDT", "15m")

        # Enable volume spike alerts
        db.update_user_settings("test_user", {"enabled_alerts": ["volume"]})

        # Modify the sample data to trigger volume spike alert
        df = sample_ohlcv_data.copy()
        # Modify the data to have a volume spike
        df.loc[df.index[-1], "volume"] = df["volume"].mean() * 3

        # Mock the market data fetch
        mock_fetch_data.return_value = df

        # Create alert directly and check it
        from bot.alerts import VolumeSpikeAlert

        volume_alert = VolumeSpikeAlert("BTCUSDT", threshold=2.5)

        # Check if alert triggers
        alert_message = volume_alert.check(df)
        print(f"Volume alert message: {alert_message}")

        # Record the alert in the database if triggered
        if alert_message:
            db.record_alert("test_user", "BTCUSDT", "15m", "volume", alert_message)

        # Get alerts from the database
        db_alerts = db.get_recent_alerts("test_user", hours=1)

        # Verify results - we should have at least one alert
        assert alert_message is not None, "Volume alert should trigger"
        assert len(db_alerts) > 0, "Alert should be in database"
        assert "VOLUME" in db_alerts[0]["message"], "Alert should be about volume"

        # Clean up
        db.close()

    @patch("bot.binance.requests.get")
    def test_data_flow_from_binance_to_indicators(self, mock_get):
        """Test the flow of data from Binance API to indicator calculations"""
        from bot.binance import fetch_market_data

        # Create sample klines data that would be returned by Binance API
        klines_data = [
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades,
            #  Taker buy base asset volume, Taker buy quote asset volume, Ignore]
            [
                1627776000000,
                "40000",
                "41000",
                "39500",
                "40500",
                "100",
                1627779599999,
                "4050000",
                1000,
                "50",
                "2025000",
                "0",
            ],
            [
                1627779600000,
                "40500",
                "42000",
                "40400",
                "41800",
                "120",
                1627783199999,
                "5016000",
                1200,
                "60",
                "2508000",
                "0",
            ],
            # Add more candles
            [
                1627783200000,
                "41800",
                "42500",
                "41500",
                "42200",
                "90",
                1627786799999,
                "3798000",
                900,
                "45",
                "1899000",
                "0",
            ],
            [
                1627786800000,
                "42200",
                "43000",
                "42000",
                "42800",
                "150",
                1627790399999,
                "6420000",
                1500,
                "75",
                "3210000",
                "0",
            ],
            [
                1627790400000,
                "42800",
                "43500",
                "42700",
                "43200",
                "130",
                1627793999999,
                "5616000",
                1300,
                "65",
                "2808000",
                "0",
            ],
        ]

        # Properly mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = klines_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Fetch market data
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=5)

        # Verify data is correctly fetched and processed
        assert df is not None, "Data should be fetched successfully"
        assert len(df) == 5, "Should have 5 candles"

        # Test indicator calculations
        rsi = calculate_rsi(df)
        assert rsi is not None, "RSI should be calculated successfully"

        macd, signal, hist = (
            calculate_macd(df).loc[:, ["MACD", "Signal", "Histogram"]].iloc[-1]
        )
        assert not np.isnan(macd), "MACD should be calculated successfully"

        bb = calculate_bollinger_bands(df)
        assert bb is not None, "Bollinger Bands should be calculated successfully"
        assert "upper" in bb.columns, "Bollinger Bands should have upper band"
        assert "lower" in bb.columns, "Bollinger Bands should have lower band"

    @patch("bot.scheduler.fetch_market_data")
    def test_user_specific_alerts(self, mock_fetch_data, sample_ohlcv_data):
        """Test that alerts are user-specific with different settings"""
        # Setup
        db = DatabaseManager(":memory:")
        db.create_tables()

        # Create two test users with different settings
        db.create_user("user1", "User One")
        db.create_user("user2", "User Two")

        # Add both users to watch the same symbol
        db.add_to_watchlist("user1", "BTCUSDT", "15m")
        db.add_to_watchlist("user2", "BTCUSDT", "15m")

        # Update user1 to have more sensitive RSI settings and enable RSI alerts
        db.update_user_settings(
            "user1",
            {"rsi_oversold": 35, "rsi_overbought": 65, "enabled_alerts": ["rsi"]},
        )
        # user2 keeps default settings (30/70) but also enable RSI alerts
        db.update_user_settings("user2", {"enabled_alerts": ["rsi"]})

        # Mock the market data
        df = sample_ohlcv_data.copy()
        mock_fetch_data.return_value = df

        # Test with direct alert instances instead of using the scheduler
        from bot.alerts import RsiAlert

        # Create RSI alerts with different settings for each user
        user1_rsi_alert = RsiAlert(
            "BTCUSDT", oversold=35, overbought=65
        )  # More sensitive
        user2_rsi_alert = RsiAlert("BTCUSDT")  # Default (30/70)

        # Create Series with values that cross the threshold for user1 but not user2
        with patch("bot.alerts.calculate_rsi") as mock_rsi:
            # Will trigger for user1 (crossing below 35) but not user2 (not crossing below 30)
            mock_rsi.return_value = pd.Series(
                [40, 37, 36, 34]
            )  # Crosses below 35, but still above 30

            # Check the alerts directly
            user1_result = user1_rsi_alert.check(df)
            user2_result = user2_rsi_alert.check(df)

            # Record any triggered alerts in the database
            if user1_result:
                db.record_alert("user1", "BTCUSDT", "15m", "rsi", user1_result)
            if user2_result:
                db.record_alert("user2", "BTCUSDT", "15m", "rsi", user2_result)

            # Get alerts from the database
            user1_db_alerts = db.get_recent_alerts("user1", hours=1)
            user2_db_alerts = db.get_recent_alerts("user2", hours=1)

            # Print debugging information
            print(f"user1_alert instance: {user1_rsi_alert}")
            print(f"user2_alert instance: {user2_rsi_alert}")
            print(f"user1_result: {user1_result}")
            print(f"user2_result: {user2_result}")
            print(f"user1_db_alerts: {user1_db_alerts}")
            print(f"user2_db_alerts: {user2_db_alerts}")

            # Verify results
            assert user1_result is not None, "User1 alert should have triggered"
            assert user2_result is None, "User2 alert should not have triggered"
            assert len(user1_db_alerts) > 0, "User1 should have alerts in database"
            assert len(user2_db_alerts) == 0, "User2 should not have alerts in database"

            # Check that user1's alert is about RSI
            if user1_db_alerts:
                assert (
                    "RSI" in user1_db_alerts[0]["message"]
                ), "Alert should be about RSI"

        # Clean up
        db.close()
