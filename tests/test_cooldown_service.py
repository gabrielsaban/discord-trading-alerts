import logging
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_cooldown_service")

# Add the parent directory to path for imports
sys.path.append("..")

# Import the services we need to test
from bot.alerts import AlertManager
from bot.services.cooldown_repository import CooldownRepository, get_cooldown_repository
from bot.services.cooldown_service import CooldownService, get_cooldown_service
from bot.services.feature_flags import get_flag, set_flag


class TestCooldownService(unittest.TestCase):
    """Test the CooldownService implementation"""

    def setUp(self):
        """Set up test fixtures"""
        # Reset repository singleton for clean testing
        self._reset_repository_singleton()

        # Mock the repository to avoid file system access
        self.mock_repository = MagicMock()
        self.mock_repository.get_all_cooldowns.return_value = {}
        self.mock_repository.get_cooldown.return_value = (
            None  # Ensure get_cooldown returns None
        )

        # Patch the get_cooldown_repository function to return our mock
        self.repository_patcher = patch(
            "bot.services.cooldown_service.get_cooldown_repository",
            return_value=self.mock_repository,
        )
        self.repository_patcher.start()

        # Reset service singleton for clean testing
        self._reset_service_singleton()

        # Create a fresh CooldownService instance for each test
        self.service = CooldownService()

        # Also create a standard AlertManager for comparison
        self.alert_manager = AlertManager()

        # Create test data
        self.symbol = "BTCUSDT"
        self.alert_type = "RsiAlert"
        self.alert_subtype = "OVERSOLD"
        self.interval = "15m"

        # Mock the feature flag to enable the service
        set_flag("ENABLE_COOLDOWN_SERVICE", True)

        # Clear cooldowns before each test - do this AFTER creating service and manager
        self.service.clear_cooldowns()
        AlertManager.global_cooldowns = {}

    def tearDown(self):
        """Clean up after each test"""
        # Stop all patches
        self.repository_patcher.stop()

        # Clear singletons
        self._reset_service_singleton()
        self._reset_repository_singleton()

    def _reset_service_singleton(self):
        """Reset the CooldownService singleton for testing"""
        import bot.services.cooldown_service

        bot.services.cooldown_service._cooldown_service = None

    def _reset_repository_singleton(self):
        """Reset the CooldownRepository singleton for testing"""
        import bot.services.cooldown_repository

        bot.services.cooldown_repository.CooldownRepository._instance = None

    def test_singleton_pattern(self):
        """Test that get_cooldown_service returns a singleton instance"""
        with patch(
            "bot.services.cooldown_service.get_cooldown_repository",
            return_value=self.mock_repository,
        ):
            service1 = get_cooldown_service()
            service2 = get_cooldown_service()
            self.assertIs(service1, service2)

    def test_cooldown_check_parity(self):
        """Test that CooldownService._check_cooldown behaves the same as AlertManager._is_globally_cooled_down"""
        # Ensure both implementations are using clean state
        self.service.clear_cooldowns()
        AlertManager.global_cooldowns = {}

        # Disable feature flag to ensure we're testing the legacy implementation in AlertManager
        with patch("bot.alerts.get_flag", return_value=False):
            # No previous cooldown should allow alert
            self.assertTrue(
                self.service._check_cooldown(
                    self.symbol, self.alert_type, self.alert_subtype, self.interval
                )
            )
            self.assertTrue(
                self.alert_manager._is_globally_cooled_down(
                    self.symbol, self.alert_type, self.alert_subtype, self.interval
                )
            )

            # Update cooldown in both implementations
            self.service.update_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
            self.alert_manager._update_global_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )

            # Now both should return False (in cooldown)
            self.assertFalse(
                self.service._check_cooldown(
                    self.symbol, self.alert_type, self.alert_subtype, self.interval
                )
            )
            self.assertFalse(
                self.alert_manager._is_globally_cooled_down(
                    self.symbol, self.alert_type, self.alert_subtype, self.interval
                )
            )

    def test_is_in_cooldown_inverse_logic(self):
        """Test that is_in_cooldown returns the inverse of _check_cooldown"""
        # No previous cooldown
        self.assertTrue(
            self.service._check_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
        )
        self.assertFalse(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
        )

        # Update cooldown
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, self.interval
        )

        # Now in cooldown
        self.assertFalse(
            self.service._check_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
        )
        self.assertTrue(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
        )

    def test_higher_timeframe_bypass(self):
        """Test that higher timeframes bypass lower timeframe cooldowns"""
        # Update cooldown for 15m interval
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, "15m"
        )

        # 15m should be in cooldown
        self.assertTrue(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, "15m"
            )
        )

        # 4h should bypass the cooldown
        self.assertFalse(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, "4h"
            )
        )

        # Update cooldown for 4h interval
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, "4h"
        )

        # Now 4h should be in cooldown
        self.assertTrue(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, "4h"
            )
        )

    def test_override_with_extreme_signal(self):
        """Test that extreme signals can override cooldowns"""
        # Update cooldown with normal strength
        self.service.update_cooldown(
            self.symbol,
            self.alert_type,
            self.alert_subtype,
            self.interval,
            signal_strength=5.0,
        )

        # Should be in cooldown normally
        self.assertTrue(
            self.service.is_in_cooldown(
                self.symbol, self.alert_type, self.alert_subtype, self.interval
            )
        )

        # Extreme signal should override cooldown
        self.assertFalse(
            self.service.is_in_cooldown(
                self.symbol,
                self.alert_type,
                self.alert_subtype,
                self.interval,
                signal_strength=9.5,  # Extreme value
            )
        )

    def test_strength_based_override(self):
        """Test that a significantly stronger signal can override cooldown after some time"""
        # Update cooldown with moderate strength
        self.service.update_cooldown(
            self.symbol,
            self.alert_type,
            self.alert_subtype,
            self.interval,
            signal_strength=6.0,
        )

        # Modify the timestamp to simulate partial cooldown elapsed
        cooldown_key = f"{self.symbol}_{self.alert_subtype}"
        with self.service.lock:
            self.service.cooldowns[cooldown_key][
                "timestamp"
            ] = datetime.utcnow() - timedelta(minutes=30)

        # Even stronger signal should override once 30% of cooldown has passed
        self.assertFalse(
            self.service.is_in_cooldown(
                self.symbol,
                self.alert_type,
                self.alert_subtype,
                self.interval,
                signal_strength=8.5,  # Much stronger than original 6.0
            )
        )

        # Slightly stronger signal should not override
        self.assertTrue(
            self.service.is_in_cooldown(
                self.symbol,
                self.alert_type,
                self.alert_subtype,
                self.interval,
                signal_strength=6.8,  # Not strong enough difference
            )
        )

    def test_atr_adjusted_cooldown(self):
        """Test that ATR data adjusts cooldown periods properly"""
        # Default cooldown for 15m should be 60 minutes
        base_cooldown = self.service.timeframe_cooldowns["15m"]
        self.assertEqual(base_cooldown, 60)

        # Mock market data with low ATR percentile (bottom 25%)
        market_data_low = {"ATR_Percentile": 10.0}

        # Calculate adjusted cooldown
        low_vol_cooldown = self.service._get_atr_adjusted_cooldown(
            base_cooldown, self.interval, self.symbol, market_data_low
        )

        # Verify increase by 25%
        self.assertEqual(low_vol_cooldown, int(base_cooldown * 1.25))

        # Mock market data with high ATR percentile (top 25%)
        market_data_high = {"ATR_Percentile": 90.0}

        # Calculate adjusted cooldown
        high_vol_cooldown = self.service._get_atr_adjusted_cooldown(
            base_cooldown, self.interval, self.symbol, market_data_high
        )

        # Verify decrease by 25%
        self.assertEqual(high_vol_cooldown, int(base_cooldown * 0.75))

        # Test middle range (50%)
        market_data_mid = {"ATR_Percentile": 50.0}
        mid_vol_cooldown = self.service._get_atr_adjusted_cooldown(
            base_cooldown, self.interval, self.symbol, market_data_mid
        )

        # At 50% percentile, adjustment should be 1.0 (no change)
        self.assertEqual(mid_vol_cooldown, base_cooldown)

    def test_high_volatility_session(self):
        """Test that high volatility session time adds 10% to cooldown for short timeframes"""
        base_cooldown = 20  # 5m interval

        # Mock current hour to be market opening hour
        with patch("bot.services.cooldown_service.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(
                2023, 1, 1, 14, 0, 0
            )  # US market open hour
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # High volatility session should add 10% for short timeframes
            adjusted_cooldown = self.service._get_atr_adjusted_cooldown(
                base_cooldown, "5m", self.symbol
            )

            # Verify 10% increase
            self.assertEqual(adjusted_cooldown, int(base_cooldown * 1.1))

    def test_pruning_expired_cooldowns(self):
        """Test that expired cooldowns are properly pruned"""
        # Add some cooldowns with different timestamps
        self.service.update_cooldown(
            "BTC1", self.alert_type, self.alert_subtype, self.interval
        )
        self.service.update_cooldown(
            "BTC2", self.alert_type, self.alert_subtype, self.interval
        )
        self.service.update_cooldown(
            "BTC3", self.alert_type, self.alert_subtype, self.interval
        )

        # Modify timestamps to simulate different ages
        now = datetime.utcnow()
        with self.service.lock:
            self.service.cooldowns["BTC1_OVERSOLD"]["timestamp"] = now - timedelta(
                hours=25
            )  # Expired
            self.service.cooldowns["BTC2_OVERSOLD"]["timestamp"] = now - timedelta(
                hours=23
            )  # Not expired
            self.service.cooldowns["BTC3_OVERSOLD"]["timestamp"] = now - timedelta(
                hours=48
            )  # Expired

        # Should have 3 cooldowns before pruning
        self.assertEqual(len(self.service.get_all_cooldowns()), 3)

        # Prune with 24-hour max age
        count = self.service.prune_expired_cooldowns(max_age_hours=24)

        # Should have pruned 2 cooldowns
        self.assertEqual(count, 2)

        # Should have 1 cooldown left
        self.assertEqual(len(self.service.get_all_cooldowns()), 1)

        # Verify repository pruning was called
        self.mock_repository.prune_expired_cooldowns.assert_called_once_with(24)

    def test_feature_flag_integration(self):
        """Test integration with feature flag system"""
        # Set feature flag to disable service
        with patch("bot.alerts.get_flag", return_value=False):
            # This setup would happen in AlertManager's code
            alert_manager = AlertManager()

            # Check if the feature flag logic in AlertManager would use the cooldown service
            with patch.object(
                get_cooldown_service(), "is_in_cooldown"
            ) as mock_is_in_cooldown:
                # Run a cooldown check (would use legacy implementation)
                should_trigger = alert_manager._is_globally_cooled_down(
                    self.symbol, self.alert_type, self.alert_subtype, self.interval
                )

                # Service should not be called when feature flag is disabled
                mock_is_in_cooldown.assert_not_called()

    def test_repository_integration(self):
        """Test integration with the CooldownRepository"""
        # Update cooldown data
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, self.interval
        )

        # Verify repository set_cooldown was called
        self.mock_repository.set_cooldown.assert_called()

        # Verify repository save_to_file was called
        self.mock_repository.save_to_file.assert_called()

        # Check repository was used in initial load
        self.mock_repository.get_all_cooldowns.assert_called()

    def test_auto_save_enabled(self):
        """Test that auto-save can be enabled and disabled"""
        # Default is enabled
        self.assertTrue(self.service.auto_save_enabled)

        # Update with auto-save enabled (default)
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, self.interval
        )

        # Repository save should be called
        save_calls = self.mock_repository.save_to_file.call_count
        self.assertGreater(save_calls, 0)

        # Reset mock
        self.mock_repository.save_to_file.reset_mock()

        # Disable auto-save
        self.service.set_auto_save(False)

        # Update cooldown
        self.service.update_cooldown(
            "ETH", self.alert_type, self.alert_subtype, self.interval
        )

        # Repository save should not be called
        self.mock_repository.save_to_file.assert_not_called()

        # Manually save
        self.service.save_cooldowns()

        # Now repository save should be called
        self.mock_repository.save_to_file.assert_called_once()

    def test_manual_save(self):
        """Test manual saving of cooldowns"""
        # Set up some cooldowns
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, self.interval
        )

        # Reset mock to clear call history
        self.mock_repository.save_to_file.reset_mock()

        # Perform manual save
        result = self.service.save_cooldowns(force=True)

        # Verify save was called with force=True
        self.mock_repository.save_to_file.assert_called_once_with(force=True)

    def test_higher_timeframe_cooldown_key(self):
        """Test that higher timeframes use a specific cooldown key format"""
        # Update with normal timeframe
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, "15m"
        )

        # Normal cooldown key should be used for repository
        self.mock_repository.set_cooldown.assert_called_with(
            f"{self.symbol}_{self.alert_subtype}",
            {"timestamp": unittest.mock.ANY, "interval": "15m", "strength": 5.0},
        )

        # Reset mock
        self.mock_repository.set_cooldown.reset_mock()

        # Update with higher timeframe
        self.service.update_cooldown(
            self.symbol, self.alert_type, self.alert_subtype, "4h"
        )

        # Higher timeframe should include interval in key
        self.mock_repository.set_cooldown.assert_called_with(
            f"{self.symbol}_{self.alert_subtype}_4h",
            {"timestamp": unittest.mock.ANY, "interval": "4h", "strength": 5.0},
        )

    def test_repository_fallback(self):
        """Test that repository is used as fallback for cooldown info"""
        # Set up mocked repository response
        cooldown_key = f"{self.symbol}_{self.alert_subtype}"
        mock_cooldown_data = {
            "timestamp": datetime.utcnow(),
            "interval": self.interval,
            "strength": 7.0,
        }
        self.mock_repository.get_cooldown.return_value = mock_cooldown_data

        # Memory cooldown is empty
        self.assertEqual(len(self.service.cooldowns), 0)

        # Get cooldown info should check repository
        result = self.service.get_cooldown_info(
            self.symbol, self.alert_type, self.alert_subtype
        )

        # Verify repository was queried
        self.mock_repository.get_cooldown.assert_called_once_with(cooldown_key)

        # Result should match repository data
        self.assertEqual(result, mock_cooldown_data)


if __name__ == "__main__":
    unittest.main()
