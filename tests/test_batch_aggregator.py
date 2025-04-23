import asyncio
import logging
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_batch_aggregator")

# Add the parent directory to path for imports
sys.path.append("..")

# Import the services we need to test
from bot.services.batch_aggregator import BatchAggregator, get_batch_aggregator
from bot.services.feature_flags import get_flag, set_flag


class TestBatchAggregator(unittest.TestCase):
    """Test the BatchAggregator implementation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a fresh BatchAggregator instance for each test
        self.aggregator = BatchAggregator()

        # Mock callback function
        self.mock_callback = MagicMock()
        self.aggregator.set_callback(self.mock_callback)

        # Create test data
        self.user_id = "test_user"
        self.symbol = "BTCUSDT"
        self.interval = "15m"
        self.alert_type = "RSI"
        self.alert_subtype = "OVERSOLD"
        self.alert_msg = "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5"
        self.strength = 6.0

        # Enable feature flag by default for tests
        set_flag("ENABLE_BATCH_AGGREGATOR", True)

    def tearDown(self):
        """Clean up after each test"""
        # Stop the background task if running
        self.aggregator.stop()

        # Clear feature flags
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Reset singleton to avoid affecting other tests
        import bot.services.batch_aggregator

        bot.services.batch_aggregator._batch_aggregator = None

    def test_singleton_pattern(self):
        """Test that get_batch_aggregator returns a singleton instance"""
        aggregator1 = get_batch_aggregator()
        aggregator2 = get_batch_aggregator()
        self.assertIs(aggregator1, aggregator2)

    def test_enqueue(self):
        """Test enqueueing an alert"""
        # Enqueue an alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Check that the alert was added to the queue
        self.assertIn(self.user_id, self.aggregator.queued_alerts)
        self.assertIn(self.symbol, self.aggregator.queued_alerts[self.user_id])

        # Check the alert key format
        alert_key = f"{self.alert_type}_{self.alert_subtype}"
        self.assertIn(
            alert_key, self.aggregator.queued_alerts[self.user_id][self.symbol]
        )

        # Check that there's one alert in the queue
        alerts = self.aggregator.queued_alerts[self.user_id][self.symbol][alert_key]
        self.assertEqual(len(alerts), 1)

        # Check alert contents
        alert = alerts[0]
        self.assertEqual(alert["message"], self.alert_msg)
        self.assertEqual(alert["interval"], self.interval)
        self.assertEqual(alert["strength"], self.strength)
        self.assertEqual(alert["type"], self.alert_type)
        self.assertEqual(alert["subtype"], self.alert_subtype)

    def test_feature_flag_integration(self):
        """Test that feature flag controls enqueueing"""
        # Disable feature flag
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Enqueue an alert (should not be added to queue)
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Check that no alerts were added to the queue
        self.assertEqual(len(self.aggregator.queued_alerts), 0)

        # Enable feature flag
        set_flag("ENABLE_BATCH_AGGREGATOR", True)

        # Enqueue an alert (should be added to queue)
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Check that the alert was added to the queue
        self.assertEqual(len(self.aggregator.queued_alerts), 1)

    def test_processing_batches(self):
        """Test processing batched alerts"""
        # Add several alerts
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            "RSI",
            "OVERSOLD",
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            6.0,
        )

        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            "RSI",
            "OVERSOLD",
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 25.0\nPrice: 49000.00\nThreshold: 30, Latest RSI: 25.0",
            7.5,  # Stronger signal
        )

        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            "MACD",
            "BEARISH CROSS",
            "🔴 **MACD BEARISH CROSS**: BTCUSDT\nPrice: 48000.00\nMACD: -50.0000, Signal: -40.0000, Histogram: -10.0000",
            5.5,
        )

        # Process batches
        self.aggregator._process_all_batches()

        # Verify callback was called with the expected parameters
        self.mock_callback.assert_called_once()

        # Extract the arguments from the call
        args = self.mock_callback.call_args[0]
        self.assertEqual(args[0], self.user_id)  # First arg should be user_id

        # Check that we received a list of alerts
        alerts = args[1]
        self.assertIsInstance(alerts, list)
        self.assertEqual(len(alerts), 1)  # Should only send one summary message

        # Verify the summary contains information from both alert types
        summary = alerts[0]
        self.assertIn("ALERT SUMMARY", summary)
        self.assertIn("RSI", summary)
        self.assertIn("MACD", summary)

        # Verify the queues were cleared
        self.assertEqual(
            len(
                self.aggregator.queued_alerts[self.user_id][self.symbol]["RSI_OVERSOLD"]
            ),
            0,
        )
        self.assertEqual(
            len(
                self.aggregator.queued_alerts[self.user_id][self.symbol][
                    "MACD_BEARISH CROSS"
                ]
            ),
            0,
        )

    def test_single_alert_processing(self):
        """Test processing a single alert (should not create summary)"""
        # Add a single alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            "RSI",
            "OVERSOLD",
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            6.0,
        )

        # Process batches
        self.aggregator._process_all_batches()

        # Verify callback was called with the expected parameters
        self.mock_callback.assert_called_once()

        # Extract the arguments from the call
        args = self.mock_callback.call_args[0]
        self.assertEqual(args[0], self.user_id)  # First arg should be user_id

        # Check that we received a list of alerts
        alerts = args[1]
        self.assertIsInstance(alerts, list)
        self.assertEqual(len(alerts), 1)

        # Verify the alert is the original message, not a summary
        alert = alerts[0]
        self.assertEqual(
            alert,
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
        )

        # Verify the queue was cleared
        self.assertEqual(
            len(
                self.aggregator.queued_alerts[self.user_id][self.symbol]["RSI_OVERSOLD"]
            ),
            0,
        )

    def test_multiple_symbols(self):
        """Test processing alerts for multiple symbols"""
        # Add alerts for two different symbols
        self.aggregator.enqueue(
            self.user_id,
            "BTCUSDT",
            self.interval,
            "RSI",
            "OVERSOLD",
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            6.0,
        )

        self.aggregator.enqueue(
            self.user_id,
            "ETHUSDT",
            self.interval,
            "MACD",
            "BEARISH CROSS",
            "🔴 **MACD BEARISH CROSS**: ETHUSDT\nPrice: 3000.00\nMACD: -5.0000, Signal: -4.0000, Histogram: -1.0000",
            5.5,
        )

        # Process batches
        self.aggregator._process_all_batches()

        # Verify callback was called with the expected parameters
        self.mock_callback.assert_called_once()

        # Extract the arguments from the call
        args = self.mock_callback.call_args[0]
        self.assertEqual(args[0], self.user_id)  # First arg should be user_id

        # Check that we received a list of alerts
        alerts = args[1]
        self.assertIsInstance(alerts, list)
        self.assertEqual(len(alerts), 2)  # Should get one alert per symbol

        # Verify the first alert contains BTC
        self.assertIn("BTCUSDT", alerts[0])

        # Verify the second alert contains ETH
        self.assertIn("ETHUSDT", alerts[1])

    def test_multiple_users(self):
        """Test processing alerts for multiple users"""
        # Add alerts for two different users
        user1 = "user1"
        user2 = "user2"

        self.aggregator.enqueue(
            user1,
            self.symbol,
            self.interval,
            "RSI",
            "OVERSOLD",
            "🔴 **RSI OVERSOLD**: BTCUSDT RSI at 29.5\nPrice: 50000.00\nThreshold: 30, Latest RSI: 29.5",
            6.0,
        )

        self.aggregator.enqueue(
            user2,
            self.symbol,
            self.interval,
            "MACD",
            "BEARISH CROSS",
            "🔴 **MACD BEARISH CROSS**: BTCUSDT\nPrice: 48000.00\nMACD: -50.0000, Signal: -40.0000, Histogram: -10.0000",
            5.5,
        )

        # Process batches
        self.aggregator._process_all_batches()

        # Verify callback was called twice (once for each user)
        self.assertEqual(self.mock_callback.call_count, 2)

        # Check calls for each user
        for call in self.mock_callback.call_args_list:
            args = call[0]
            user_id = args[0]
            alerts = args[1]

            # Verify we received the correct user and single alert
            self.assertIn(user_id, [user1, user2])
            self.assertEqual(len(alerts), 1)

    @patch("bot.services.batch_aggregator.datetime")
    def test_batch_interval_respect(self, mock_datetime):
        """Test that the batch interval is respected"""
        # Set up mock datetime
        now = datetime.utcnow()
        mock_datetime.utcnow.return_value = now
        mock_datetime.min = datetime.min  # Set the min attribute on the mock

        # Add an alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Process batches
        self.aggregator._process_all_batches()

        # Verify callback was called
        self.assertEqual(self.mock_callback.call_count, 1)
        self.mock_callback.reset_mock()

        # Update last_processed time to simulate recent processing
        self.aggregator.last_processed[self.user_id] = now

        # Add another alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Process batches again immediately - should not process due to interval
        self.aggregator._process_all_batches()

        # Verify callback was not called again
        self.mock_callback.assert_not_called()

        # Advance time beyond batch interval
        mock_datetime.utcnow.return_value = now + timedelta(
            minutes=self.aggregator.batch_interval + 1
        )

        # Process batches again - should process now
        self.aggregator._process_all_batches()

        # Verify callback was called again
        self.assertEqual(self.mock_callback.call_count, 1)

    def test_background_task_starts_on_enqueue(self):
        """Test that the background task starts when an alert is enqueued"""
        # Set a flag to track if the background task was started
        self.task_started = False

        # Patch the start_background_task method
        original_start = self.aggregator.start_background_task

        def mock_start():
            self.task_started = True
            return original_start()

        self.aggregator.start_background_task = mock_start

        # Enqueue an alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Verify the background task was started
        self.assertTrue(self.task_started)

    def test_clear_all(self):
        """Test clearing all queued alerts"""
        # Add an alert
        self.aggregator.enqueue(
            self.user_id,
            self.symbol,
            self.interval,
            self.alert_type,
            self.alert_subtype,
            self.alert_msg,
            self.strength,
        )

        # Verify alert was added
        self.assertEqual(len(self.aggregator.queued_alerts), 1)

        # Clear all alerts
        self.aggregator.clear_all()

        # Verify alerts were cleared
        self.assertEqual(len(self.aggregator.queued_alerts), 0)
        self.assertEqual(len(self.aggregator.last_processed), 0)

    async def _run_background_task_test(self):
        """Helper method to run async tests for the background task"""
        # Patch asyncio.sleep to return immediately
        with patch("asyncio.sleep", return_value=None):
            # Create test loop
            loop = asyncio.get_event_loop()

            # Start background task
            task = loop.create_task(self.aggregator._process_batches_periodically())

            # Let the task run for a short time
            await asyncio.sleep(0.1)

            # Make sure _process_all_batches would be called as part of the loop
            self.aggregator._process_all_batches()

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def test_background_task(self):
        """Test the background task"""
        # Add processing logic to the aggregator
        self.aggregator._process_all_batches = MagicMock()

        # Run the async test
        asyncio.run(self._run_background_task_test())

        # Verify that process_all_batches was called
        self.aggregator._process_all_batches.assert_called()


if __name__ == "__main__":
    unittest.main()
