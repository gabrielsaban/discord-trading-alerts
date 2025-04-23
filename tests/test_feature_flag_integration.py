import logging
from datetime import datetime, timedelta
from unittest.mock import DEFAULT, MagicMock, patch

import pandas as pd
import pytest

import bot.services.feature_flags as feature_flags
from bot.alerts import AlertManager, MacdAlert, RsiAlert
from bot.scheduler import AlertScheduler
from bot.services.batch_aggregator import BatchAggregator, get_batch_aggregator
from bot.services.cooldown_service import CooldownService, get_cooldown_service
from bot.services.feature_flags import get_flag, set_flag
from bot.services.override_engine import OverrideEngine, get_override_engine
from bot.services.reset_service_state import reset_all_service_singletons

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_feature_flag_integration")


class TestFeatureFlagIntegration:
    """Tests to verify behavior parity when feature flags are toggled"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down for each test"""
        # Reset all singleton instances to ensure clean state
        reset_all_service_singletons()

        # Reset all feature flags before each test
        set_flag("ENABLE_COOLDOWN_SERVICE", False)
        set_flag("ENABLE_OVERRIDE_ENGINE", False)
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Set up test data
        self.user_id = "test_user"
        self.symbol = "BTCUSDT"
        self.interval = "15m"

        # Log the current state
        logger.info(f"Test setup: Global cooldowns: {AlertManager.global_cooldowns}")

        yield

        # Reset all feature flags after each test
        set_flag("ENABLE_COOLDOWN_SERVICE", False)
        set_flag("ENABLE_OVERRIDE_ENGINE", False)
        set_flag("ENABLE_BATCH_AGGREGATOR", False)

        # Clean up singletons again
        reset_all_service_singletons()

    @patch("bot.services.cooldown_repository.CooldownRepository")
    @patch("bot.alerts.calculate_rsi")
    def test_cooldown_service_parity(self, mock_rsi, MockCooldownRepo):
        """Test that alerts with ENABLE_COOLDOWN_SERVICE produce same results as legacy"""
        symbol = "BTCUSDT"
        alert_type = "RSI"
        interval = "1h"

        # Create test manager
        manager = AlertManager()

        # Set initial state - no cooldowns for either implementation
        AlertManager.global_cooldowns = {}
        cooldown_service = get_cooldown_service()
        cooldown_service.clear_cooldowns()

        # We need to create a cooldown manually in the legacy system
        cooldown_key = f"{symbol}_{alert_type}"
        AlertManager.global_cooldowns[cooldown_key] = {
            "timestamp": datetime.utcnow(),
            "interval": interval,
            "strength": 5.0,
        }

        with patch("bot.alerts.get_flag") as mock_get_flag:
            # First check with legacy implementation (feature flag off)
            mock_get_flag.return_value = False
            legacy_result = manager._is_globally_cooled_down(
                symbol, alert_type, interval=interval
            )

            # Then with cooldown service (feature flag on)
            # Only enable cooldown service
            mock_get_flag.side_effect = (
                lambda key, default=False: key == "ENABLE_COOLDOWN_SERVICE"
            )

            # Set the cooldown in the service
            cooldown_service.update_cooldown(symbol, alert_type, None, interval)

            new_result = manager._is_globally_cooled_down(
                symbol, alert_type, interval=interval
            )

            # Both should now be in cooldown
            assert legacy_result == new_result, "Cooldown state should be the same"
            assert not legacy_result, "No alerts should be produced due to cooldown"
            assert not new_result, "No alerts should be produced due to cooldown"

    @patch("bot.alerts.calculate_rsi")
    def test_override_engine_parity(self, mock_rsi):
        """Test that alerts with ENABLE_OVERRIDE_ENGINE produce same results as legacy"""
        symbol = "ETHUSDT"
        alert_type = "EMA"
        interval = "15m"
        message = "STRONG SIGNAL - EXTREMELY OVERSOLD"

        # Reset state to ensure clean test
        reset_all_service_singletons()

        # Create test manager
        manager = AlertManager()

        # Set up a patched get_flag first
        with patch("bot.alerts.get_flag") as mock_get_flag:
            # Set up flag to simulate override engine enabled
            mock_get_flag.side_effect = (
                lambda key, default=False: key == "ENABLE_OVERRIDE_ENGINE"
            )

            # Add a cooldown
            cooldown_key = f"{symbol}_{alert_type}"
            AlertManager.global_cooldowns[cooldown_key] = {
                "timestamp": datetime.utcnow(),
                "interval": interval,
                "strength": 5.0,
            }

            # First, confirm we're actually in cooldown
            result = manager._is_globally_cooled_down(
                symbol, alert_type, interval=interval
            )
            assert not result, "Should be in cooldown initially"

            # Get the override engine and patch the can_override method
            with patch("bot.alerts.get_override_engine") as mock_get_override_engine:
                # Create a mock override engine
                mock_override_engine = MagicMock(spec=OverrideEngine)
                mock_override_engine.can_override.return_value = (
                    True,
                    "Mocked override",
                )
                mock_get_override_engine.return_value = mock_override_engine

                # Try with a message that should trigger the override
                override_result = manager._is_globally_cooled_down(
                    symbol, alert_type, interval=interval, message=message
                )

                # Should override the cooldown
                assert override_result, "Strong signal should override cooldown"

                # Verify the override engine was called
                mock_override_engine.can_override.assert_called_once()

    @patch("bot.alerts.calculate_rsi")
    def test_batch_aggregator_parity(self, mock_rsi):
        """Test that alerts with ENABLE_BATCH_AGGREGATOR enqueue properly"""
        symbol = "ADAUSDT"
        alert_type = "MACD"
        interval = "1h"
        message = "MACD Crossover"

        # Reset state to ensure clean test
        reset_all_service_singletons()

        # Setup mocks first before creating manager
        with patch("bot.alerts.get_batch_aggregator") as mock_get_aggregator, patch(
            "bot.alerts.get_cooldown_service"
        ) as mock_get_cooldown_service, patch("bot.alerts.get_flag") as mock_get_flag:
            # Configure mock cooldown service
            mock_cooldown_service = MagicMock(spec=CooldownService)
            mock_cooldown_service.is_in_cooldown.return_value = (
                True  # Signal is in cooldown
            )
            mock_get_cooldown_service.return_value = mock_cooldown_service

            # Configure mocks
            mock_batch_aggregator = MagicMock(spec=BatchAggregator)
            mock_enqueue = MagicMock()
            mock_batch_aggregator.enqueue = mock_enqueue
            mock_get_aggregator.return_value = mock_batch_aggregator

            # Configure feature flags
            mock_get_flag.side_effect = lambda key, default=False: key in [
                "ENABLE_COOLDOWN_SERVICE",
                "ENABLE_BATCH_AGGREGATOR",
            ]

            # Create test manager after mocks are set up
            manager = AlertManager()

            # Set the user ID
            manager.current_user_id = self.user_id

            # Create a cooldown
            manager._update_global_cooldown(symbol, alert_type, interval=interval)

            # Test alert in cooldown
            result = manager._is_globally_cooled_down(
                symbol, alert_type, interval=interval, message=message
            )

            # Should be in cooldown
            assert not result, "Alert should be in cooldown"

            # BatchAggregator.enqueue should have been called
            mock_enqueue.assert_called_once()
            args = mock_enqueue.call_args[1]  # Get keyword arguments
            assert args["symbol"] == symbol
            assert args["alert_type"] == alert_type
            assert args["interval"] == interval
            assert args["alert_msg"] == message

    @patch("bot.services.cooldown_repository.CooldownRepository")
    @patch("bot.alerts.calculate_rsi")
    def test_all_flags_together(self, mock_rsi, MockCooldownRepo):
        """Test that all feature flags work together properly"""
        symbol = "LTCUSDT"
        alert_type = "BB"
        interval = "4h"
        message = "Price above upper band"

        # Reset state to ensure clean test
        reset_all_service_singletons()

        # Setup the mocks first
        with patch("bot.alerts.get_batch_aggregator") as mock_get_aggregator, patch(
            "bot.alerts.get_flag"
        ) as mock_get_flag, patch(
            "bot.alerts.get_cooldown_service"
        ) as mock_get_cooldown_service:
            # Configure mocks
            mock_batch_aggregator = MagicMock(spec=BatchAggregator)
            mock_enqueue = MagicMock()
            mock_batch_aggregator.enqueue = mock_enqueue
            mock_get_aggregator.return_value = mock_batch_aggregator

            # Configure cooldown service
            mock_cooldown_service = MagicMock(spec=CooldownService)
            mock_cooldown_service.is_in_cooldown.return_value = (
                True  # Signal is in cooldown
            )
            mock_get_cooldown_service.return_value = mock_cooldown_service

            # Configure feature flags - all enabled
            mock_get_flag.return_value = True

            # Create test manager after mocks are set up
            manager = AlertManager()

            # Set the user ID
            manager.current_user_id = self.user_id

            # Create a cooldown
            manager._update_global_cooldown(symbol, alert_type, interval=interval)

            # Check if cooldown would prevent alert
            result = manager._is_globally_cooled_down(
                symbol, alert_type, interval=interval, message=message
            )

            # Should be in cooldown because we're using the cooldown service (feature flag enabled)
            assert not result, "Alert should be in cooldown with all flags enabled"

            # Verify that the batch aggregator enqueue was called
            mock_enqueue.assert_called_once()
            args = mock_enqueue.call_args[1]  # Get keyword arguments
            assert args["symbol"] == symbol
            assert args["alert_type"] == alert_type


if __name__ == "__main__":
    pytest.main(["-xvs", "test_feature_flag_integration.py"])
