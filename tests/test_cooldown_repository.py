import json
import logging
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_cooldown_repository")

# Add the parent directory to path for imports
sys.path.append("..")

# Import the repository we need to test
from bot.services.cooldown_repository import CooldownRepository, get_cooldown_repository


class TestCooldownRepository(unittest.TestCase):
    """Test the CooldownRepository implementation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock the file path to use the temporary directory
        with patch(
            "bot.services.cooldown_repository.os.path.join",
            return_value=os.path.join(self.temp_dir.name, "cooldowns.json"),
        ):
            # Create a fresh repository instance for each test
            self._reset_singleton()
            self.repo = CooldownRepository()

        # Clear cooldowns before each test
        self.repo.clear_all_cooldowns()

        # Create test data
        self.symbol = "BTCUSDT"
        self.alert_type = "RsiAlert"
        self.alert_subtype = "OVERSOLD"
        self.interval = "15m"
        self.cooldown_key = f"{self.symbol}_{self.alert_subtype}"
        self.cooldown_data = {
            "timestamp": datetime.utcnow(),
            "interval": self.interval,
            "strength": 6.5,
            "message": "RSI Oversold Alert",
        }

    def tearDown(self):
        """Clean up after tests"""
        # Reset the singleton for the next test
        self._reset_singleton()
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def _reset_singleton(self):
        """Reset the CooldownRepository singleton for testing"""
        CooldownRepository._instance = None

    def test_singleton_pattern(self):
        """Test that get_cooldown_repository returns a singleton instance"""
        repo1 = get_cooldown_repository()
        repo2 = get_cooldown_repository()
        self.assertIs(repo1, repo2)

    def test_get_and_set_cooldown(self):
        """Test setting and getting a cooldown"""
        # Initially should be None
        self.assertIsNone(self.repo.get_cooldown(self.cooldown_key))

        # Set cooldown
        self.repo.set_cooldown(self.cooldown_key, self.cooldown_data)

        # Now should return data
        result = self.repo.get_cooldown(self.cooldown_key)
        self.assertIsNotNone(result)
        self.assertEqual(result["interval"], self.interval)
        self.assertEqual(result["strength"], 6.5)

    def test_remove_cooldown(self):
        """Test removing a cooldown"""
        # Set cooldown
        self.repo.set_cooldown(self.cooldown_key, self.cooldown_data)

        # Should exist
        self.assertIsNotNone(self.repo.get_cooldown(self.cooldown_key))

        # Remove it
        result = self.repo.remove_cooldown(self.cooldown_key)
        self.assertTrue(result)

        # Should be gone
        self.assertIsNone(self.repo.get_cooldown(self.cooldown_key))

        # Removing non-existent key should return False
        result = self.repo.remove_cooldown("NONEXISTENT_KEY")
        self.assertFalse(result)

    def test_get_all_cooldowns(self):
        """Test getting all cooldowns"""
        # Add multiple cooldowns
        self.repo.set_cooldown(
            "BTC_RSI", {"timestamp": datetime.utcnow(), "interval": "5m"}
        )
        self.repo.set_cooldown(
            "ETH_MACD", {"timestamp": datetime.utcnow(), "interval": "1h"}
        )

        # Get all
        all_cooldowns = self.repo.get_all_cooldowns()
        self.assertEqual(len(all_cooldowns), 2)
        self.assertIn("BTC_RSI", all_cooldowns)
        self.assertIn("ETH_MACD", all_cooldowns)

    def test_clear_all_cooldowns(self):
        """Test clearing all cooldowns"""
        # Add multiple cooldowns
        self.repo.set_cooldown("BTC_RSI", {"timestamp": datetime.utcnow()})
        self.repo.set_cooldown("ETH_MACD", {"timestamp": datetime.utcnow()})

        # Verify they exist
        self.assertEqual(len(self.repo.get_all_cooldowns()), 2)

        # Clear all
        self.repo.clear_all_cooldowns()

        # Should be empty
        self.assertEqual(len(self.repo.get_all_cooldowns()), 0)

    def test_save_and_load_file(self):
        """Test saving to and loading from file"""
        # Add cooldowns
        now = datetime.utcnow()
        self.repo.set_cooldown(
            "BTC_RSI", {"timestamp": now, "interval": "5m", "strength": 7.0}
        )
        self.repo.set_cooldown(
            "ETH_MACD", {"timestamp": now, "interval": "1h", "strength": 8.5}
        )

        # Save to file
        save_result = self.repo.save_to_file()
        self.assertTrue(save_result)

        # Create a new repository instance to test loading
        with patch(
            "bot.services.cooldown_repository.os.path.join",
            return_value=os.path.join(self.temp_dir.name, "cooldowns.json"),
        ):
            self._reset_singleton()
            new_repo = CooldownRepository()

        # Data should be loaded in the new instance
        loaded_cooldowns = new_repo.get_all_cooldowns()
        self.assertEqual(len(loaded_cooldowns), 2)
        self.assertIn("BTC_RSI", loaded_cooldowns)
        self.assertIn("ETH_MACD", loaded_cooldowns)

        # Check if data was properly deserialized
        btc_data = new_repo.get_cooldown("BTC_RSI")
        self.assertEqual(btc_data["interval"], "5m")
        self.assertEqual(btc_data["strength"], 7.0)
        self.assertIsInstance(btc_data["timestamp"], datetime)

    def test_prune_expired_cooldowns(self):
        """Test pruning expired cooldowns"""
        now = datetime.utcnow()

        # Add cooldowns with different ages
        self.repo.set_cooldown("BTC_RSI", {"timestamp": now - timedelta(hours=25)})
        self.repo.set_cooldown("ETH_MACD", {"timestamp": now - timedelta(hours=12)})
        self.repo.set_cooldown("LTC_BB", {"timestamp": now - timedelta(hours=36)})

        # Verify initial count
        self.assertEqual(len(self.repo.get_all_cooldowns()), 3)

        # Prune with 24 hour max age
        pruned_count = self.repo.prune_expired_cooldowns(max_age_hours=24)

        # Should have removed two entries
        self.assertEqual(pruned_count, 2)
        self.assertEqual(len(self.repo.get_all_cooldowns()), 1)

        # Only the middle one should remain
        self.assertIsNotNone(self.repo.get_cooldown("ETH_MACD"))
        self.assertIsNone(self.repo.get_cooldown("BTC_RSI"))
        self.assertIsNone(self.repo.get_cooldown("LTC_BB"))

    def test_get_cooldowns_by_symbol(self):
        """Test getting cooldowns by symbol"""
        now = datetime.utcnow()

        # Add cooldowns for different symbols
        self.repo.set_cooldown("BTC_RSI", {"timestamp": now})
        self.repo.set_cooldown("BTC_MACD", {"timestamp": now})
        self.repo.set_cooldown("ETH_RSI", {"timestamp": now})

        # Get by symbol
        btc_cooldowns = self.repo.get_cooldowns_by_symbol("BTC")

        # Should have two entries
        self.assertEqual(len(btc_cooldowns), 2)
        self.assertIn("BTC_RSI", btc_cooldowns)
        self.assertIn("BTC_MACD", btc_cooldowns)

        # Get another symbol
        eth_cooldowns = self.repo.get_cooldowns_by_symbol("ETH")

        # Should have one entry
        self.assertEqual(len(eth_cooldowns), 1)
        self.assertIn("ETH_RSI", eth_cooldowns)

        # Non-existent symbol should return empty dict
        xrp_cooldowns = self.repo.get_cooldowns_by_symbol("XRP")
        self.assertEqual(len(xrp_cooldowns), 0)

    def test_get_symbols_with_cooldowns(self):
        """Test getting all symbols with active cooldowns"""
        now = datetime.utcnow()

        # Add cooldowns for different symbols
        self.repo.set_cooldown("BTC_RSI", {"timestamp": now})
        self.repo.set_cooldown("BTC_MACD", {"timestamp": now})
        self.repo.set_cooldown("ETH_RSI", {"timestamp": now})
        self.repo.set_cooldown("LTC_BB", {"timestamp": now})

        # Get all symbols
        symbols = self.repo.get_symbols_with_cooldowns()

        # Should have three unique symbols
        self.assertEqual(len(symbols), 3)
        self.assertIn("BTC", symbols)
        self.assertIn("ETH", symbols)
        self.assertIn("LTC", symbols)

    def test_save_with_force(self):
        """Test force saving with no changes"""
        # Add cooldowns
        self.repo.set_cooldown("BTC_RSI", {"timestamp": datetime.utcnow()})

        # Save to file
        self.repo.save_to_file()

        # Reset modified flag
        self.repo.modified_since_save = False

        # Normal save should return False (no changes)
        self.assertFalse(self.repo.save_to_file())

        # Force save should return True
        self.assertTrue(self.repo.save_to_file(force=True))

    def test_timestamp_conversion(self):
        """Test automatic timestamp conversion between string and datetime"""
        # Set with string timestamp
        timestamp_str = "2023-04-15T12:30:45.123456"
        self.repo.set_cooldown("TEST_KEY", {"timestamp": timestamp_str, "value": 42})

        # Get should return datetime
        result = self.repo.get_cooldown("TEST_KEY")
        self.assertIsInstance(result["timestamp"], datetime)

        # Save and load to test serialization
        self.repo.save_to_file()

        # Recreate repository
        with patch(
            "bot.services.cooldown_repository.os.path.join",
            return_value=os.path.join(self.temp_dir.name, "cooldowns.json"),
        ):
            self._reset_singleton()
            new_repo = CooldownRepository()

        # Loaded data should have datetime
        loaded = new_repo.get_cooldown("TEST_KEY")
        self.assertIsInstance(loaded["timestamp"], datetime)
        self.assertEqual(loaded["value"], 42)

    def test_invalid_timestamp_handling(self):
        """Test handling of invalid timestamp formats"""
        # Set with invalid timestamp
        with self.assertLogs(
            logger="discord_trading_alerts.cooldown_repository", level=logging.ERROR
        ):
            self.repo.set_cooldown(
                "TEST_KEY", {"timestamp": "invalid-timestamp", "value": 42}
            )

        # Should still set with current time fallback
        result = self.repo.get_cooldown("TEST_KEY")
        self.assertIsNotNone(result)
        self.assertIsInstance(result["timestamp"], datetime)
        self.assertEqual(result["value"], 42)

    def test_deep_copy_on_get(self):
        """Test that get methods return deep copies to prevent modification of internal state"""
        # Set cooldown
        self.repo.set_cooldown(self.cooldown_key, self.cooldown_data)

        # Get a reference
        cooldown = self.repo.get_cooldown(self.cooldown_key)

        # Modify the returned data
        cooldown["strength"] = 999.9

        # Internal state should not be affected
        internal_cooldown = self.repo.get_cooldown(self.cooldown_key)
        self.assertEqual(internal_cooldown["strength"], 6.5)

        # Similarly for get_all_cooldowns
        all_cooldowns = self.repo.get_all_cooldowns()
        all_cooldowns[self.cooldown_key]["strength"] = 888.8

        # Internal state should not be affected
        internal_cooldown = self.repo.get_cooldown(self.cooldown_key)
        self.assertEqual(internal_cooldown["strength"], 6.5)

    def test_file_error_handling(self):
        """Test error handling when file operations fail"""
        # Mock open to raise an exception
        with patch("builtins.open", side_effect=IOError("Mocked file error")):
            # Save should handle the error and return False
            with self.assertLogs(
                logger="discord_trading_alerts.cooldown_repository", level=logging.ERROR
            ):
                result = self.repo.save_to_file()
                self.assertFalse(result)

    def test_thread_safety(self):
        """Basic test of thread safety with lock usage"""
        # Mock RLock to verify it's being used
        original_lock = self.repo.lock
        mock_lock = MagicMock()
        self.repo.lock = mock_lock

        # Test operations
        self.repo.get_cooldown("test")
        self.repo.set_cooldown("test", {"timestamp": datetime.utcnow()})
        self.repo.remove_cooldown("test")
        self.repo.get_all_cooldowns()

        # Verify lock was used
        self.assertGreater(mock_lock.__enter__.call_count, 0)
        self.assertGreater(mock_lock.__exit__.call_count, 0)

        # Restore original lock
        self.repo.lock = original_lock


if __name__ == "__main__":
    unittest.main()
