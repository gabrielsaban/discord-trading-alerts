import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from bot.scheduler import AlertScheduler
from bot.alerts import AlertManager


class TestAlertScheduler:
    """Tests for the AlertScheduler class"""

    def test_init(self):
        """Test scheduler initialization"""
        mock_callback = MagicMock()
        scheduler = AlertScheduler(mock_callback)

        assert scheduler.alert_callback == mock_callback
        assert scheduler.running is False
        assert len(scheduler.user_alert_managers) == 0
        assert len(scheduler.scheduled_symbols) == 0

    @patch("bot.scheduler.get_db")
    def test_get_user_alert_manager(self, mock_get_db):
        """Test getting user-specific alert managers"""
        scheduler = AlertScheduler()

        # Get alert manager for a new user/symbol
        manager1 = scheduler.get_user_alert_manager("user1", "BTCUSDT", "15m")

        assert isinstance(manager1, AlertManager)
        assert "user1" in scheduler.user_alert_managers
        assert "BTCUSDT_15m" in scheduler.user_alert_managers["user1"]

        # Get again (should return same instance)
        manager2 = scheduler.get_user_alert_manager("user1", "BTCUSDT", "15m")
        assert manager1 is manager2  # Same instance

        # Get for different symbol
        manager3 = scheduler.get_user_alert_manager("user1", "ETHUSDT", "15m")
        assert manager3 is not manager1  # Different instance
        assert "ETHUSDT_15m" in scheduler.user_alert_managers["user1"]

        # Get for different user
        manager4 = scheduler.get_user_alert_manager("user2", "BTCUSDT", "15m")
        assert manager4 is not manager1  # Different instance
        assert "user2" in scheduler.user_alert_managers

    @patch("bot.scheduler.get_db")
    @patch("bot.scheduler.fetch_market_data")
    def test_check_symbol_alerts(self, mock_fetch_data, mock_get_db):
        """Test checking symbol alerts"""
        import pandas as pd

        # Mock the database
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Mock users watching
        mock_db.get_users_watching_symbol.return_value = ["user1", "user2"]

        # Mock user data
        mock_db.get_user.side_effect = lambda user_id: {
            "user1": {"is_active": True, "settings": {"enabled_alerts": ["rsi"]}},
            "user2": {"is_active": True, "settings": {"enabled_alerts": ["macd"]}},
        }.get(user_id)

        # Mock market data
        mock_data = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [1000, 2000, 3000],
            }
        )
        mock_fetch_data.return_value = mock_data

        # Create scheduler with mock callback
        mock_callback = MagicMock()
        scheduler = AlertScheduler(mock_callback)

        # Create mock alert managers that will return some alerts
        class MockManager(AlertManager):
            def check_alerts(self, symbol, df):
                return [f"TEST ALERT: {symbol}"]

        # Patch the get_user_alert_manager method
        original_method = scheduler.get_user_alert_manager
        scheduler.get_user_alert_manager = MagicMock(return_value=MockManager())

        # Run the check
        scheduler.check_symbol_alerts("BTCUSDT", "15m")

        # Verify interactions
        mock_db.get_users_watching_symbol.assert_called_once_with("BTCUSDT", "15m")
        assert mock_db.get_user.call_count == 2  # Called for each user
        assert scheduler.get_user_alert_manager.call_count == 2  # Called for each user

        # Verify alert recording
        assert mock_db.record_alert.call_count == 2  # One alert per user

        # Verify callback was called
        assert mock_callback.call_count == 2  # Called for each user with alerts

        # Restore original method
        scheduler.get_user_alert_manager = original_method

    @patch("bot.scheduler.get_db")
    def test_add_and_remove_symbol(self, mock_get_db):
        """Test adding and removing symbols"""
        scheduler = AlertScheduler()

        # Patch the schedule_symbol_check method
        scheduler.schedule_symbol_check = MagicMock()

        # Add a symbol
        scheduler.running = True  # Set as running to enable add_symbol
        scheduler.add_symbol("BTCUSDT", "15m")

        # Verify it called schedule_symbol_check
        scheduler.schedule_symbol_check.assert_called_once_with("BTCUSDT", "15m")

        # Reset the mock
        scheduler.schedule_symbol_check.reset_mock()

        # Set up for testing remove_symbol
        scheduler.scheduled_symbols.add("BTCUSDT_15m")

        # Create a mock user alert manager
        scheduler.user_alert_managers = {
            "user1": {"BTCUSDT_15m": AlertManager()},
            "user2": {"BTCUSDT_15m": AlertManager(), "ETHUSDT_15m": AlertManager()},
        }

        # Mock the scheduler.remove_job method
        scheduler.scheduler.remove_job = MagicMock()

        # Remove the symbol
        scheduler.remove_symbol("BTCUSDT", "15m")

        # Verify the symbol was removed from scheduled_symbols
        assert "BTCUSDT_15m" not in scheduler.scheduled_symbols

        # Verify remove_job was called
        scheduler.scheduler.remove_job.assert_called_once_with("check_BTCUSDT_15m")

        # Verify the alert managers were removed for all users
        assert "BTCUSDT_15m" not in scheduler.user_alert_managers["user1"]
        assert "BTCUSDT_15m" not in scheduler.user_alert_managers["user2"]
        assert "ETHUSDT_15m" in scheduler.user_alert_managers["user2"]  # Still there

    def test_thread_safety(self):
        """Test thread safety of the scheduler"""
        scheduler = AlertScheduler()

        # Patch methods to avoid actual scheduling
        scheduler.schedule_symbol_check = MagicMock()
        scheduler.check_symbol_alerts = MagicMock()

        # Test data
        test_symbols = [("BTCUSDT", "15m"), ("ETHUSDT", "15m"), ("BNBUSDT", "15m")]

        # Function for thread to run
        def worker_thread(symbols):
            for symbol, interval in symbols:
                # Add the symbol
                scheduler.add_symbol(symbol, interval)

        # Create and run threads
        threads = []
        for i in range(3):  # 3 threads
            symbols = [
                test_symbols[i % len(test_symbols)]
            ]  # Each thread gets one symbol
            t = threading.Thread(target=worker_thread, args=(symbols,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify that schedule_symbol_check was called for each symbol
        expected_calls = sum(1 for _ in range(3))  # 3 threads, 1 symbol each
        assert scheduler.schedule_symbol_check.call_count == expected_calls
