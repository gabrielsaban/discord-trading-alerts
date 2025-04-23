import threading
import time
import unittest
from datetime import datetime, timedelta

from bot.services.override_engine import (
    ExtremeReadingRule,
    HigherTimeframeRule,
    OverrideEngine,
    OverrideRule,
    StrengthThresholdRule,
    TimeframeHierarchyRule,
    get_override_engine,
)


class MockRule(OverrideRule):
    """Mock rule for testing"""

    def __init__(self, return_value=True, reason="Mock rule triggered"):
        self.return_value = return_value
        self.reason = reason
        self.called = False

    def can_override(self, **kwargs):
        self.called = True
        self.kwargs = kwargs
        return self.return_value, self.reason


class TestOverrideRules(unittest.TestCase):
    """Test individual override rules"""

    def test_higher_timeframe_rule(self):
        """Test HigherTimeframeRule identifies higher timeframes correctly"""
        rule = HigherTimeframeRule(higher_timeframes=["4h", "1d"])

        # Test with higher timeframe
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=5.0,
            last_strength=5.0,
            interval="4h",
            last_interval="1h",
            cooldown_progress=0.5,
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 4h", reason)

        # Test with lower timeframe
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=5.0,
            last_strength=5.0,
            interval="15m",
            last_interval="1h",
            cooldown_progress=0.5,
        )
        self.assertFalse(override)
        self.assertIn("Not a higher timeframe", reason)

        # Test with custom higher timeframes
        rule = HigherTimeframeRule(higher_timeframes=["1h", "4h", "1d"])
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=5.0,
            last_strength=5.0,
            interval="1h",
            last_interval="15m",
            cooldown_progress=0.5,
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 1h", reason)

    def test_extreme_reading_rule(self):
        """Test ExtremeReadingRule identifies extreme values correctly"""
        rule = ExtremeReadingRule(threshold=9.0)

        # Test with extreme value
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=9.5,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertTrue(override)
        self.assertIn("Extreme reading detected", reason)

        # Test with non-extreme value
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.5,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("Reading not extreme enough", reason)

        # Test with custom threshold
        rule = ExtremeReadingRule(threshold=8.0)
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.5,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertTrue(override)
        self.assertIn("Extreme reading detected", reason)

    def test_strength_threshold_rule(self):
        """Test StrengthThresholdRule identifies stronger signals correctly"""
        rule = StrengthThresholdRule(
            strength_diff=2.0, min_strength=7.0, min_progress=0.3
        )

        # Test with stronger signal and sufficient progress
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.0,
            last_strength=5.5,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.4,
        )
        self.assertTrue(override)
        self.assertIn("Stronger signal", reason)

        # Test with stronger signal but insufficient progress
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.0,
            last_strength=5.5,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("Strength threshold not met", reason)

        # Test with insufficient strength difference
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=7.0,
            last_strength=6.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.4,
        )
        self.assertFalse(override)
        self.assertIn("Strength threshold not met", reason)

        # Test with lower than minimum strength
        override, reason = rule.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=6.5,
            last_strength=4.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.4,
        )
        self.assertFalse(override)
        self.assertIn("Strength threshold not met", reason)

    def test_timeframe_hierarchy_rule(self):
        """Test TimeframeHierarchyRule identifies higher timeframes with sufficient strength"""
        rule = TimeframeHierarchyRule(
            timeframe_order=["1m", "5m", "15m", "1h", "4h", "1d"], min_strength=6.5
        )

        # Test with higher timeframe and sufficient strength
        override, reason = rule.can_override(
            alert_type="MacdAlert",
            alert_subtype="BULLISH CROSS",
            current_strength=7.0,
            last_strength=5.0,
            interval="1h",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 1h > 15m", reason)

        # Test with higher timeframe but insufficient strength
        override, reason = rule.can_override(
            alert_type="MacdAlert",
            alert_subtype="BULLISH CROSS",
            current_strength=6.0,
            last_strength=5.0,
            interval="1h",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("Not a higher timeframe or insufficient strength", reason)

        # Test with same timeframe
        override, reason = rule.can_override(
            alert_type="MacdAlert",
            alert_subtype="BULLISH CROSS",
            current_strength=7.0,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("Same timeframe", reason)

        # Test with lower timeframe
        override, reason = rule.can_override(
            alert_type="MacdAlert",
            alert_subtype="BULLISH CROSS",
            current_strength=7.0,
            last_strength=5.0,
            interval="5m",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("Not a higher timeframe", reason)

        # Test with unknown timeframe
        override, reason = rule.can_override(
            alert_type="MacdAlert",
            alert_subtype="BULLISH CROSS",
            current_strength=7.0,
            last_strength=5.0,
            interval="unknown",
            last_interval="15m",
            cooldown_progress=0.2,
        )
        self.assertFalse(override)
        self.assertIn("not in recognized hierarchy", reason)


class TestOverrideEngine(unittest.TestCase):
    """Test the OverrideEngine class"""

    def setUp(self):
        # Create a fresh engine for each test
        self.engine = OverrideEngine()

    def test_singleton_pattern(self):
        """Test that get_override_engine returns a singleton instance"""
        engine1 = get_override_engine()
        engine2 = get_override_engine()
        self.assertIs(engine1, engine2)

    def test_thread_safety(self):
        """Test thread safety of the OverrideEngine"""
        engine = get_override_engine()
        results = []
        errors = []

        def worker(idx):
            try:
                # Test with different data to ensure no data corruption
                override, reason = engine.can_override(
                    alert_type=f"TestAlert{idx}",
                    alert_subtype=f"TEST{idx}",
                    current_strength=5.0 + (idx % 5),
                    last_strength=5.0,
                    interval="15m",
                    last_interval="15m",
                    time_elapsed=timedelta(minutes=30),
                    cooldown_period=timedelta(minutes=60),
                    message=f"Test message {idx}",
                )
                results.append((idx, override, reason))
            except Exception as e:
                errors.append((idx, str(e)))

        # Create multiple threads to test concurrency
        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check for errors
        self.assertEqual(0, len(errors), f"Thread errors: {errors}")

        # Check results - should have 20 results
        self.assertEqual(20, len(results))

    def test_with_custom_rules(self):
        """Test engine with custom rules"""
        engine = OverrideEngine()

        # Add a custom rule that always returns True
        mock_rule = MockRule(return_value=True, reason="Custom rule activated")
        engine.add_rule(mock_rule)

        # Test the engine
        override, reason = engine.can_override(
            alert_type="TestAlert",
            alert_subtype="TEST",
            current_strength=5.0,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=30),
            cooldown_period=timedelta(minutes=60),
        )

        # The custom rule should have been called
        self.assertTrue(mock_rule.called)

        # The engine should return the custom rule's result
        self.assertTrue(override)
        self.assertEqual("Custom rule activated", reason)

    def test_rule_precedence(self):
        """Test that rules are checked in order"""
        engine = OverrideEngine()
        engine.clear_rules()

        # Add multiple rules with known return values
        rule1 = MockRule(return_value=False, reason="Rule 1")
        rule2 = MockRule(return_value=True, reason="Rule 2")
        rule3 = MockRule(return_value=True, reason="Rule 3")

        engine.add_rule(rule1)
        engine.add_rule(rule2)
        engine.add_rule(rule3)

        # Test the engine
        override, reason = engine.can_override(
            alert_type="TestAlert",
            alert_subtype="TEST",
            current_strength=5.0,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=30),
            cooldown_period=timedelta(minutes=60),
        )

        # All rules should have been called until a true result
        self.assertTrue(rule1.called)
        self.assertTrue(rule2.called)
        self.assertFalse(
            rule3.called
        )  # Rule 3 should not be called since Rule 2 returned True

        # The engine should return Rule 2's result
        self.assertTrue(override)
        self.assertEqual("Rule 2", reason)

    def test_engine_configure(self):
        """Test the engine.configure method"""
        engine = OverrideEngine()

        # Configure with custom settings
        engine.configure(
            extreme_threshold=8.5,
            strength_diff=1.5,
            min_strength=6.0,
            min_progress=0.4,
            timeframe_min_strength=7.0,
            timeframe_order=["5m", "15m", "1h", "4h"],
            higher_timeframes=["1h", "4h"],
        )

        # Test with extreme reading that meets the new threshold
        override, reason = engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=9.0,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=30),
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Extreme reading detected", reason)

        # Test higher timeframe with new settings
        override, reason = engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=5.0,
            last_strength=5.0,
            interval="1h",
            last_interval="15m",
            time_elapsed=timedelta(minutes=30),
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 1h", reason)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios from requirements"""

    def setUp(self):
        self.engine = OverrideEngine()

    def test_higher_tf_bypass(self):
        """Test that 4h/1d only check their own 24h cooldown"""
        # 4h signal should bypass lower timeframe cooldowns
        override, reason = self.engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=6.0,  # Even with moderate strength
            last_strength=5.0,
            interval="4h",
            last_interval="15m",
            time_elapsed=timedelta(minutes=10),  # Very recent alert
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 4h", reason)

        # 1d signal should bypass lower timeframe cooldowns
        override, reason = self.engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=6.0,  # Even with moderate strength
            last_strength=5.0,
            interval="1d",
            last_interval="1h",
            time_elapsed=timedelta(minutes=30),  # Very recent alert
            cooldown_period=timedelta(hours=2),
        )
        self.assertTrue(override)
        self.assertIn("Higher timeframe 1d", reason)

    def test_extreme_reading_overrides(self):
        """Test extreme reading overrides (RSI<20/>80, ADX>40)"""
        # Extreme RSI reading
        override, reason = self.engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=9.5,  # Represents RSI < 20
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=5),  # Very recent alert
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Extreme reading detected", reason)

        # Extreme ADX reading
        override, reason = self.engine.can_override(
            alert_type="AdxAlert",
            alert_subtype="STRONG TREND",
            current_strength=9.3,  # Represents ADX > 40
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=5),  # Very recent alert
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Extreme reading detected", reason)

    def test_strength_threshold_overrides(self):
        """Test tuneable strength threshold overrides"""
        # Stronger signal with sufficient cooldown progress
        override, reason = self.engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.0,  # Much stronger than previous
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=20),  # 1/3 of cooldown elapsed
            cooldown_period=timedelta(minutes=60),
        )
        self.assertTrue(override)
        self.assertIn("Stronger signal", reason)

        # Reconfigure with stricter settings
        self.engine.configure(
            strength_diff=3.5,  # Require larger difference
            min_strength=8.0,  # Require higher minimum
            min_progress=0.5,  # Require more cooldown progress
        )

        # Same scenario with stricter settings should now fail
        override, reason = self.engine.can_override(
            alert_type="RsiAlert",
            alert_subtype="OVERSOLD",
            current_strength=8.0,
            last_strength=5.0,
            interval="15m",
            last_interval="15m",
            time_elapsed=timedelta(minutes=20),
            cooldown_period=timedelta(minutes=60),
        )
        self.assertFalse(override)
        self.assertIn("No override rules matched", reason)


if __name__ == "__main__":
    unittest.main()
