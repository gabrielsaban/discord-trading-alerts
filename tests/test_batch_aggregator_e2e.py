import logging
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_batch_aggregator_e2e")

# Add the parent directory to path for imports
sys.path.append("..")

# Import the necessary modules
from bot.alerts import AlertManager
from bot.scheduler import AlertScheduler, get_scheduler
from bot.services.batch_aggregator import BatchAggregator, get_batch_aggregator
from bot.services.feature_flags import get_flag, set_flag
from bot.services.reset_service_state import reset_all_service_singletons


class TestBatchAggregatorE2E(unittest.TestCase):
    """End-to-end test for BatchAggregator integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset all singletons for a clean test environment
        reset_all_service_singletons()

        # Enable the batch aggregator feature flag
        set_flag("ENABLE_BATCH_AGGREGATOR", True)

        # Create a mock callback function
        self.mock_callback = MagicMock()

        # Create an AlertScheduler instance with the mock callback
        self.scheduler = AlertScheduler(alert_callback=self.mock_callback)

        # Initialize the scheduler
        self.scheduler.initialize()

        # Get the batch aggregator singleton
        self.batch_aggregator = get_batch_aggregator()

        # Make sure we have a clean state
        self.batch_aggregator.clear_all()

        # Create test data
        self.user_id = "test_user"
        self.symbol = "BTCUSDT"
        self.interval = "15m"
        self.alert_type = "RSI"

    def tearDown(self):
        """Clean up after test"""
        # Stop the scheduler
        self.scheduler.stop()

        # Disable the batch aggregator feature flag
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Reset all singletons to clean state
        reset_all_service_singletons()

    def test_alert_batching_e2e(self):
        """Test that alerts are batched and sent as summaries"""
        # Create a test alert manager
        alert_manager = self.scheduler.get_user_alert_manager(
            self.user_id, self.symbol, self.interval
        )

        # Directly use the batch aggregator for this test

        # Generate several alerts for the same symbol
        alert_manager.current_user_id = self.user_id

        # Enqueue several alerts
        self.batch_aggregator.enqueue(
            user_id=self.user_id,
            symbol=self.symbol,
            interval=self.interval,
            alert_type="RSI",
            alert_subtype="OVERSOLD",
            alert_msg="🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            strength=6.0,
        )

        self.batch_aggregator.enqueue(
            user_id=self.user_id,
            symbol=self.symbol,
            interval=self.interval,
            alert_type="RSI",
            alert_subtype="OVERSOLD",
            alert_msg="🔴 **RSI OVERSOLD**: BTCUSDT RSI at 25.0\nPrice: 49000.00\nThreshold: 30, Latest RSI: 25.0",
            strength=7.5,  # Stronger signal
        )

        self.batch_aggregator.enqueue(
            user_id=self.user_id,
            symbol=self.symbol,
            interval=self.interval,
            alert_type="MACD",
            alert_subtype="BEARISH CROSS",
            alert_msg="🔴 **MACD BEARISH CROSS**: BTCUSDT\nPrice: 48000.00\nMACD: -50.0000, Signal: -40.0000, Histogram: -10.0000",
            strength=5.5,
        )

        # Process the batched alerts
        self.batch_aggregator._process_all_batches()

        # Check that the callback was called with a batch summary
        self.mock_callback.assert_called_once()

        # Get the arguments from the callback
        args = self.mock_callback.call_args[0]
        user_id = args[0]
        alerts = args[1]

        # Verify correct user ID
        self.assertEqual(user_id, self.user_id)

        # Verify alerts format
        self.assertEqual(len(alerts), 1)  # Should be one summary message
        summary = alerts[0]

        # The summary should contain info about both alert types
        self.assertIn("ALERT SUMMARY", summary)
        self.assertIn("RSI", summary)
        self.assertIn("MACD", summary)

        # Reset the callback for next test
        self.mock_callback.reset_mock()

        # Test feature flag control - disable the feature
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Enqueue an alert (should not be processed)
        self.batch_aggregator.enqueue(
            user_id=self.user_id,
            symbol=self.symbol,
            interval=self.interval,
            alert_type="RSI",
            alert_subtype="OVERSOLD",
            alert_msg="🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            strength=6.0,
        )

        # Process the batched alerts
        self.batch_aggregator._process_all_batches()

        # The callback should not be called again because feature flag is off
        self.mock_callback.assert_not_called()

    def test_alert_manager_integration(self):
        """Test that AlertManager correctly uses BatchAggregator when in cooldown"""
        # Reset all singletons again for this specific test
        reset_all_service_singletons()

        # Create patches before creating any objects
        with patch("bot.alerts.get_flag") as mock_get_flag, patch(
            "bot.alerts.get_batch_aggregator"
        ) as mock_get_aggregator:
            # Configure the get_flag mock to enable batch aggregator but disable cooldown service
            def mock_flag_function(flag_name, default=False):
                if flag_name == "ENABLE_BATCH_AGGREGATOR":
                    return True
                if flag_name == "ENABLE_COOLDOWN_SERVICE":
                    return False  # Ensure we use the legacy implementation
                return default

            mock_get_flag.side_effect = mock_flag_function

            # Create a mock batch aggregator with an enqueue method
            mock_aggregator = MagicMock(spec=BatchAggregator)
            mock_enqueue = MagicMock()
            mock_aggregator.enqueue = mock_enqueue
            mock_get_aggregator.return_value = mock_aggregator

            # Create a test AlertManager
            alert_manager = AlertManager()
            alert_manager.current_user_id = self.user_id

            # The alert subtype that will be used in the test
            alert_subtype = "OVERSOLD"

            # Set up a global cooldown to trigger batching
            # The cooldown key should match how it's used in the _is_globally_cooled_down method
            cooldown_key = f"{self.symbol}_{alert_subtype}"
            AlertManager.global_cooldowns[cooldown_key] = {
                "timestamp": datetime.utcnow(),  # Very recent to ensure it's in cooldown
                "interval": self.interval,
                "strength": 5.0,
            }

            # Check if alert is in cooldown - it should be
            result = alert_manager._is_globally_cooled_down(
                self.symbol,
                self.alert_type,
                alert_subtype,
                self.interval,
                "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            )

            # Verify the alert was detected as being in cooldown
            self.assertFalse(result)

            # Verify the BatchAggregator.enqueue method was called
            mock_enqueue.assert_called_once()


if __name__ == "__main__":
    unittest.main()
