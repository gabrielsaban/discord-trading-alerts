import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Setup logging
logger = logging.getLogger("discord_trading_alerts.services.override_engine")
logger.setLevel(logging.DEBUG)

# Singleton instance
_override_engine_instance = None
_instance_lock = threading.Lock()


class OverrideRule:
    """Base class for override rules"""

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        cooldown_progress: float,
        message: str = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        """
        Check if a rule can override the cooldown

        Parameters:
        -----------
        alert_type : str
            Type of alert (e.g., 'RsiAlert')
        alert_subtype : str
            Subtype of alert (e.g., 'OVERSOLD')
        current_strength : float
            Current signal strength (1-10)
        last_strength : float
            Previous signal strength (1-10)
        interval : str
            Current timeframe (e.g., '5m', '1h')
        last_interval : str
            Previous timeframe that set the cooldown
        cooldown_progress : float
            Progress through the cooldown period (0.0-1.0)
        message : str, optional
            Alert message text
        kwargs : dict
            Additional parameters specific to each rule

        Returns:
        --------
        (bool, str)
            Whether the cooldown should be overridden and reason
        """
        return False, "Base rule not implemented"


class HigherTimeframeRule(OverrideRule):
    """Rule for higher timeframe overrides (4h/1d)"""

    def __init__(self, higher_timeframes=None):
        self.higher_timeframes = higher_timeframes or ["4h", "1d"]

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        cooldown_progress: float,
        message: str = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        # Higher timeframes (4h/1d) bypass lower timeframe cooldowns completely
        if interval in self.higher_timeframes:
            return (
                True,
                f"Higher timeframe {interval} bypasses lower timeframe cooldowns",
            )
        return False, "Not a higher timeframe alert"


class ExtremeReadingRule(OverrideRule):
    """Rule for extreme indicator readings"""

    def __init__(self, threshold: float = 9.0):
        self.threshold = threshold

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        cooldown_progress: float,
        message: str = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        # Extreme readings always override (RSI<20, RSI>80, ADX>40)
        if current_strength >= self.threshold:
            return (
                True,
                f"Extreme reading detected (strength={current_strength:.1f} >= {self.threshold:.1f})",
            )
        return (
            False,
            f"Reading not extreme enough ({current_strength:.1f} < {self.threshold:.1f})",
        )


class StrengthThresholdRule(OverrideRule):
    """Rule for strength-based overrides"""

    def __init__(
        self,
        strength_diff: float = 2.0,
        min_strength: float = 7.0,
        min_progress: float = 0.3,
    ):
        self.strength_diff = strength_diff
        self.min_strength = min_strength
        self.min_progress = min_progress

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        cooldown_progress: float,
        message: str = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        # Signal significantly stronger than previous + cooldown partially elapsed
        if (
            current_strength >= last_strength + self.strength_diff
            and cooldown_progress >= self.min_progress
            and current_strength >= self.min_strength
        ):
            return True, (
                f"Stronger signal ({current_strength:.1f} >= {last_strength:.1f} + {self.strength_diff}) "
                f"with sufficient cooldown progress ({cooldown_progress:.1%} >= {self.min_progress:.1%})"
            )

        return False, (
            f"Strength threshold not met ({current_strength:.1f} vs {last_strength:.1f}, "
            f"progress: {cooldown_progress:.1%})"
        )


class TimeframeHierarchyRule(OverrideRule):
    """Rule for higher vs lower timeframe comparison"""

    def __init__(self, timeframe_order=None, min_strength: float = 6.5):
        # Timeline order for comparison operations (lowest to highest)
        self.timeframe_order = timeframe_order or [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
        ]
        self.min_strength = min_strength

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        cooldown_progress: float,
        message: str = None,
        **kwargs,
    ) -> Tuple[bool, str]:
        # If intervals are the same, this rule doesn't apply
        if interval == last_interval:
            return False, "Same timeframe - hierarchy rule not applicable"

        # Try to get indices for comparison from timeframe_order
        try:
            current_idx = self.timeframe_order.index(interval)
            last_idx = self.timeframe_order.index(last_interval)

            # If current interval is higher than last interval and signal is decent strength
            if current_idx > last_idx and current_strength >= self.min_strength:
                return True, (
                    f"Higher timeframe {interval} > {last_interval} with "
                    f"sufficient strength ({current_strength:.1f} >= {self.min_strength})"
                )

        except ValueError:
            # If interval not found in list, don't apply this override
            return (
                False,
                f"Timeframe {interval} or {last_interval} not in recognized hierarchy",
            )

        return (
            False,
            f"Not a higher timeframe or insufficient strength ({current_strength:.1f})",
        )


class OverrideEngine:
    """Engine for determining when cooldown periods can be overridden"""

    def __init__(self):
        self._lock = threading.RLock()
        self.rules: List[OverrideRule] = []

        # Add default rules
        self.rules.append(HigherTimeframeRule())
        self.rules.append(ExtremeReadingRule())
        self.rules.append(StrengthThresholdRule())
        self.rules.append(TimeframeHierarchyRule())

        # Configuration
        self.timeframe_order = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
        ]

    def can_override(
        self,
        alert_type: str,
        alert_subtype: str,
        current_strength: float,
        last_strength: float,
        interval: str,
        last_interval: str,
        time_elapsed: timedelta,
        cooldown_period: timedelta,
        message: str = None,
        additional_data: Dict = None,
    ) -> Tuple[bool, str]:
        """
        Check if a cooldown should be overridden based on various rules

        Parameters:
        -----------
        alert_type : str
            Type of alert (e.g., 'RsiAlert')
        alert_subtype : str
            Subtype of alert (e.g., 'OVERSOLD')
        current_strength : float
            Current signal strength (1-10)
        last_strength : float
            Previous signal strength (1-10)
        interval : str
            Current timeframe (e.g., '5m', '1h')
        last_interval : str
            Previous timeframe that set the cooldown
        time_elapsed : timedelta
            Time elapsed since last alert
        cooldown_period : timedelta
            Total cooldown period
        message : str, optional
            Alert message text
        additional_data : dict, optional
            Additional data for rule evaluation

        Returns:
        --------
        (bool, str)
            Whether the cooldown should be overridden and reason
        """
        with self._lock:
            # Calculate cooldown progress
            cooldown_progress = 0.0
            if cooldown_period.total_seconds() > 0:
                cooldown_progress = (
                    time_elapsed.total_seconds() / cooldown_period.total_seconds()
                )

            # Prepare additional data for rules
            kwargs = {
                "additional_data": additional_data or {},
                "cooldown_progress": cooldown_progress,
                "time_elapsed": time_elapsed,
                "cooldown_period": cooldown_period,
            }

            # Check each rule
            for rule in self.rules:
                can_override, reason = rule.can_override(
                    alert_type=alert_type,
                    alert_subtype=alert_subtype,
                    current_strength=current_strength,
                    last_strength=last_strength,
                    interval=interval,
                    last_interval=last_interval,
                    message=message,
                    **kwargs,
                )

                if can_override:
                    logger.debug(
                        f"Override approved by {rule.__class__.__name__}: {reason} "
                        f"({interval} {alert_type}/{alert_subtype}, strength={current_strength:.1f})"
                    )
                    return True, reason

            # If no rules triggered an override
            return False, "No override rules matched"

    def add_rule(self, rule: OverrideRule) -> None:
        """Add a custom override rule"""
        with self._lock:
            self.rules.append(rule)

    def clear_rules(self) -> None:
        """Clear all override rules"""
        with self._lock:
            self.rules.clear()

    def configure(self, **kwargs) -> None:
        """Configure the override engine

        Parameters:
        -----------
        extreme_threshold : float
            Threshold for extreme reading override
        strength_diff : float
            Required difference in strength for override
        min_strength : float
            Minimum strength required for override
        min_progress : float
            Minimum cooldown progress for override
        timeframe_order : list
            Order of timeframes from lowest to highest
        """
        with self._lock:
            # Clear existing rules
            self.rules.clear()

            # Extract configuration parameters with defaults
            extreme_threshold = kwargs.get("extreme_threshold", 9.0)
            strength_diff = kwargs.get("strength_diff", 2.0)
            min_strength = kwargs.get("min_strength", 7.0)
            min_progress = kwargs.get("min_progress", 0.3)
            timeframe_min_strength = kwargs.get("timeframe_min_strength", 6.5)
            timeframe_order = kwargs.get("timeframe_order", self.timeframe_order)
            higher_timeframes = kwargs.get("higher_timeframes", ["4h", "1d"])

            # Update timeframe order
            self.timeframe_order = timeframe_order

            # Create rules with new configuration
            self.rules.append(HigherTimeframeRule(higher_timeframes=higher_timeframes))
            self.rules.append(ExtremeReadingRule(threshold=extreme_threshold))
            self.rules.append(
                StrengthThresholdRule(
                    strength_diff=strength_diff,
                    min_strength=min_strength,
                    min_progress=min_progress,
                )
            )
            self.rules.append(
                TimeframeHierarchyRule(
                    timeframe_order=timeframe_order, min_strength=timeframe_min_strength
                )
            )


def get_override_engine() -> OverrideEngine:
    """Get or create the singleton instance of OverrideEngine"""
    global _override_engine_instance

    if _override_engine_instance is None:
        with _instance_lock:
            if _override_engine_instance is None:
                _override_engine_instance = OverrideEngine()

    return _override_engine_instance


# For testing
if __name__ == "__main__":
    import time

    # Setup console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    engine = get_override_engine()

    # Test higher timeframe rule
    override, reason = engine.can_override(
        alert_type="RsiAlert",
        alert_subtype="OVERSOLD",
        current_strength=6.0,
        last_strength=5.0,
        interval="4h",
        last_interval="15m",
        time_elapsed=timedelta(minutes=30),
        cooldown_period=timedelta(minutes=60),
    )
    print(f"Higher timeframe override: {override}, reason: {reason}")

    # Test extreme reading rule
    override, reason = engine.can_override(
        alert_type="RsiAlert",
        alert_subtype="OVERSOLD",
        current_strength=9.5,
        last_strength=5.0,
        interval="15m",
        last_interval="15m",
        time_elapsed=timedelta(minutes=30),
        cooldown_period=timedelta(minutes=60),
    )
    print(f"Extreme reading override: {override}, reason: {reason}")

    # Test strength threshold rule
    override, reason = engine.can_override(
        alert_type="RsiAlert",
        alert_subtype="OVERSOLD",
        current_strength=8.0,
        last_strength=5.0,
        interval="15m",
        last_interval="15m",
        time_elapsed=timedelta(minutes=40),
        cooldown_period=timedelta(minutes=60),
    )
    print(f"Strength threshold override: {override}, reason: {reason}")

    # Test timeframe hierarchy rule
    override, reason = engine.can_override(
        alert_type="MacdAlert",
        alert_subtype="BULLISH CROSS",
        current_strength=7.0,
        last_strength=6.0,
        interval="1h",
        last_interval="15m",
        time_elapsed=timedelta(minutes=10),
        cooldown_period=timedelta(minutes=60),
    )
    print(f"Timeframe hierarchy override: {override}, reason: {reason}")
