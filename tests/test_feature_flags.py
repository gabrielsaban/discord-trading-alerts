import os
import threading
import unittest
from unittest.mock import patch

from bot.services.feature_flags import (
    DEFAULT_FLAGS,
    _feature_flags,
    get_flag,
    reload_flags,
    set_flag,
)


class TestFeatureFlags(unittest.TestCase):
    """Test the feature flags module"""

    def setUp(self):
        """Reset feature flags before each test"""
        # Clear any existing flags
        _feature_flags.clear()
        # Reload with defaults
        reload_flags()

    def test_default_flags(self):
        """Test that default flags are set correctly"""
        # Check each default flag
        for flag_name, expected_value in DEFAULT_FLAGS.items():
            self.assertEqual(get_flag(flag_name), expected_value)

    def test_set_flag(self):
        """Test setting a flag value"""
        # Set a flag to a new value
        set_flag("ENABLE_COOLDOWN_SERVICE", True)

        # Check that the flag was set
        self.assertTrue(get_flag("ENABLE_COOLDOWN_SERVICE"))

        # Set another flag
        set_flag("ENABLE_OVERRIDE_ENGINE", True)
        self.assertTrue(get_flag("ENABLE_OVERRIDE_ENGINE"))

        # Set a non-default flag
        set_flag("CUSTOM_FLAG", "custom_value")
        self.assertEqual(get_flag("CUSTOM_FLAG"), "custom_value")

    @patch.dict(os.environ, {"ENABLE_COOLDOWN_SERVICE": "true"})
    def test_environment_variables(self):
        """Test that environment variables override defaults"""
        # Reload flags to pick up environment variables
        reload_flags()

        # Check that the environment variable was applied
        self.assertTrue(get_flag("ENABLE_COOLDOWN_SERVICE"))

        # Other flags should still have default values
        self.assertFalse(get_flag("ENABLE_OVERRIDE_ENGINE"))

    @patch.dict(
        os.environ,
        {
            "ENABLE_COOLDOWN_SERVICE": "true",
            "ENABLE_OVERRIDE_ENGINE": "1",
            "ENABLE_BATCH_AGGREGATOR": "yes",
            "DEBUG_LOG_OVERRIDES": "on",
        },
    )
    def test_boolean_conversion(self):
        """Test conversion of string values to boolean"""
        # Reload flags to pick up environment variables
        reload_flags()

        # Check that different string values are converted to booleans
        self.assertTrue(get_flag("ENABLE_COOLDOWN_SERVICE"))
        self.assertTrue(get_flag("ENABLE_OVERRIDE_ENGINE"))
        self.assertTrue(get_flag("ENABLE_BATCH_AGGREGATOR"))
        self.assertTrue(get_flag("DEBUG_LOG_OVERRIDES"))

    @patch.dict(
        os.environ,
        {
            "ENABLE_COOLDOWN_SERVICE": "false",
            "ENABLE_OVERRIDE_ENGINE": "0",
            "ENABLE_BATCH_AGGREGATOR": "no",
            "DEBUG_LOG_OVERRIDES": "off",
        },
    )
    def test_boolean_conversion_negative(self):
        """Test conversion of negative string values to boolean"""
        # Reload flags to pick up environment variables
        reload_flags()

        # Check that different string values are converted to booleans
        self.assertFalse(get_flag("ENABLE_COOLDOWN_SERVICE"))
        self.assertFalse(get_flag("ENABLE_OVERRIDE_ENGINE"))
        self.assertFalse(get_flag("ENABLE_BATCH_AGGREGATOR"))
        self.assertFalse(get_flag("DEBUG_LOG_OVERRIDES"))

    def test_default_value(self):
        """Test providing a default value for non-existent flags"""
        # Get a non-existent flag with a default value
        self.assertEqual(
            get_flag("NON_EXISTENT_FLAG", "default_value"), "default_value"
        )

        # Get a non-existent flag with no default (should return None)
        self.assertIsNone(get_flag("ANOTHER_NON_EXISTENT_FLAG"))

    def test_thread_safety(self):
        """Test that feature flags are thread-safe"""
        # Set initial values
        set_flag("TEST_FLAG_1", False)
        set_flag("TEST_FLAG_2", False)

        # Create an error flag to track any issues
        errors = []

        def worker_1():
            """Worker that sets and gets flags repeatedly"""
            try:
                for i in range(100):
                    set_flag("TEST_FLAG_1", True)
                    self.assertTrue(get_flag("TEST_FLAG_1"))
                    set_flag("TEST_FLAG_1", False)
                    self.assertFalse(get_flag("TEST_FLAG_1"))
            except Exception as e:
                errors.append(f"Worker 1 error: {e}")

        def worker_2():
            """Another worker that sets and gets different flags"""
            try:
                for i in range(100):
                    set_flag("TEST_FLAG_2", True)
                    self.assertTrue(get_flag("TEST_FLAG_2"))
                    set_flag("TEST_FLAG_2", False)
                    self.assertFalse(get_flag("TEST_FLAG_2"))
            except Exception as e:
                errors.append(f"Worker 2 error: {e}")

        # Create and start threads
        threads = [threading.Thread(target=worker_1), threading.Thread(target=worker_2)]

        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Check for errors
        self.assertEqual(0, len(errors), f"Thread errors: {errors}")


if __name__ == "__main__":
    unittest.main()
