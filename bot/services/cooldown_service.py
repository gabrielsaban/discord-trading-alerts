import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("discord_trading_alerts.cooldown_service")

# Import the repository
from bot.services.cooldown_repository import get_cooldown_repository

# Import the feature flags
from bot.services.feature_flags import get_flag

"""
IMPORTANT: Interval-specific cooldowns are now implemented
The cooldown key now includes the interval information, which means that:
1. Alerts of the same type but on different timeframes no longer block each other
2. Each timeframe (1m, 5m, 15m, 1h, 4h, etc.) maintains its own separate cooldown tracking
3. This enables monitoring the same pair across multiple timeframes without interference

Example: RSI oversold signals on BTCUSDT 5m don't prevent RSI oversold signals on BTCUSDT 1h
"""

class CooldownService:
    """
    Service for managing alert cooldowns across different timeframes and alert types.
    Extracted from the original AlertManager implementation to provide better separation
    of concerns and testability.

    Enhanced with persistent storage via CooldownRepository.
    """

    def __init__(self):
        """Initialize the cooldown service with default settings"""
        # Thread-safe dict for in-memory cooldowns
        self.cooldowns = {}

        # Lock for thread safety
        self.lock = threading.RLock()

        # Get the cooldown repository for persistence
        self.repository = get_cooldown_repository()

        # Flag to control auto-save behavior
        self.auto_save_enabled = True

        # Fixed cooldown for high timeframes to avoid duplicates
        self.HIGH_TF_COOLDOWN = 1440  # 24 hours in minutes

        # Default global cooldown minutes - precisely calibrated based on timeframe
        self.timeframe_cooldowns = {
            # Primary intervals with tuned cooldowns based on README spec
            "5m": 20,  # 20 min base cooldown (±5 min based on 30-min ATR)
            "15m": 60,  # 1h base cooldown (±15 min based on 1h ATR)
            "1h": 120,  # 2h base cooldown (±30 min based on 4h ATR)
            "4h": self.HIGH_TF_COOLDOWN,  # 24h fixed cooldown (just to prevent duplicates)
            # Secondary/legacy intervals - maintained for compatibility
            "1m": 15,
            "3m": 20,
            "30m": 90,
            "2h": 360,
            "6h": 1080,
            "8h": 1440,
            "12h": 2160,
            "1d": 2880,
        }

        # Timeline order for comparison operations (lowest to highest)
        self.TIMEFRAME_ORDER = [
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

        # Maps intervals to their ATR reference intervals for volatility adjustment
        self.atr_reference_intervals = {
            "5m": "30m",  # 5m uses 30m ATR for volatility
            "15m": "1h",  # 15m uses 1h ATR
            "1h": "4h",  # 1h uses 4h ATR
            "4h": "1d",  # 4h uses 1d for reference but has fixed cooldown
            "1d": None,  # 1d has no higher reference
            # Legacy intervals
            "30m": "1h",
            "2h": "4h",
            "6h": "1d",
            "8h": "1d",
            "12h": "1d",
            "3m": "15m",  # Add missing legacy intervals
            "1m": "5m",
        }

        # Default cooldown if interval not found
        self.default_cooldown_minutes = 60

        # Signal strength threshold for cooldown override
        self.override_strength_threshold = 7.0

        # Load cooldowns from the repository on initialization
        self._load_from_repository()

    def _create_cooldown_key(
        self, symbol: str, alert_type: str, alert_subtype: str = None, interval: str = None
    ) -> str:
        """
        Create a consistent cooldown key following the format:
        f"{symbol}_{alert_type}_{alert_subtype or 'None'}_{interval}"
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name (e.g., 'RsiAlert')
        alert_subtype : str, optional
            Specific alert condition (e.g., 'OVERSOLD', 'OVERBOUGHT')
        interval : str, optional
            Timeframe of the alert (e.g., '5m', '1h', '4h')
            
        Returns:
        --------
        str
            Formatted cooldown key
        """
        # Alert type must always be provided
        alert_subtype_str = str(alert_subtype) if alert_subtype is not None else "None"
        interval_str = str(interval) if interval is not None else "None"
        
        return f"{symbol}_{alert_type}_{alert_subtype_str}_{interval_str}"

    def _load_from_repository(self):
        """Load cooldowns from the repository into memory"""
        with self.lock:
            # Get all cooldowns from the repository
            repo_cooldowns = self.repository.get_all_cooldowns()

            # Update the in-memory cooldowns
            self.cooldowns.update(repo_cooldowns)

            logger.info(f"Loaded {len(repo_cooldowns)} cooldowns from repository")

    def _save_to_repository(self, force=False):
        """Save current cooldowns to the repository if auto-save is enabled"""
        if not self.auto_save_enabled and not force:
            return

        # No need to acquire lock here as individual methods will handle it
        # Transfer all in-memory cooldowns to the repository
        for key, data in self.cooldowns.items():
            self.repository.set_cooldown(key, data.copy())

        # Save to disk
        saved = self.repository.save_to_file(force=force)
        if saved:
            logger.debug(f"Saved {len(self.cooldowns)} cooldowns to repository")

    def is_in_cooldown(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str = None,
        interval: str = None,
        message: str = None,
        signal_strength: float = 5.0,
        market_data: Any = None,
    ) -> bool:
        """
        Check if an alert type for a symbol is in cooldown

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name (e.g., 'RsiAlert')
        alert_subtype : str, optional
            Specific alert condition (e.g., 'OVERSOLD', 'OVERBOUGHT')
        interval : str, optional
            Timeframe of the current alert (e.g., '5m', '1h', '4h')
        message : str, optional
            Alert message text, used to calculate signal strength
        signal_strength : float, optional
            Pre-calculated signal strength (1.0-10.0)
        market_data : Any, optional
            Market data with ATR calculated (if available)

        Returns:
        --------
        bool
            True if the alert is in cooldown (should NOT trigger)
            False if the alert is NOT in cooldown (can trigger)
        """
        # Return inverse of legacy method to maintain same logic but with clearer naming
        # Legacy method returned True if alert CAN trigger, but this method returns True if alert IS IN cooldown
        return not self._check_cooldown(
            symbol,
            alert_type,
            alert_subtype,
            interval,
            message,
            signal_strength,
            market_data,
        )

    def _check_cooldown(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str = None,
        interval: str = None,
        message: str = None,
        signal_strength: float = 5.0,
        market_data: Any = None,
    ) -> bool:
        """
        Legacy-compatible method for cooldown checking

        Parameters match is_in_cooldown but return value is inverted:
        True = can trigger (NOT in cooldown)
        False = cannot trigger (IS in cooldown)
        """
        now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator

        with self.lock:
            # Create a consistent cooldown key using the dedicated method
            cooldown_key = self._create_cooldown_key(symbol, alert_type, alert_subtype, interval)

            # Log cooldown check attempt
            logger.debug(
                f"Checking cooldown for {cooldown_key} (Interval: {interval or 'None'})"
            )

            # Special case for 4h/1d signals - they bypass lower timeframe cooldowns
            # and only need to respect their own timeframe's duplicate-block cooldown
            is_higher_timeframe = interval in ["4h", "1d"]

            if is_higher_timeframe:
                # For 4h/1d, we already have the correct key via _create_cooldown_key
                if cooldown_key in self.cooldowns:
                    # Get cooldown info for this specific higher timeframe alert
                    cooldown_info = self.cooldowns[cooldown_key]
                    last_triggered = cooldown_info.get("timestamp")

                    # Fixed 24h cooldown for 4h/1d to prevent exact duplicates
                    cooldown_period = timedelta(minutes=self.HIGH_TF_COOLDOWN)

                    # Check if cooldown period has passed
                    time_since_last = now - last_triggered
                    if time_since_last < cooldown_period:
                        # Still in cooldown for this specific higher timeframe
                        minutes_remaining = int(
                            (cooldown_period - time_since_last).total_seconds() / 60
                        )
                        logger.debug(
                            f"Higher timeframe {interval} for {cooldown_key} in cooldown: {minutes_remaining} minutes remaining."
                        )
                        # The Override Engine will be used to determine if this can be overridden
                        return False

                # For 4h/1d signals that pass the duplicate check, they can always trigger
                # regardless of lower timeframe cooldowns
                return True

            # If this alert type has never been triggered, it's not in cooldown
            if cooldown_key not in self.cooldowns:
                logger.debug(f"No previous cooldown found for {cooldown_key}")
                return True

            # Get cooldown info for this alert
            cooldown_info = self.cooldowns[cooldown_key]

            # Log the cooldown info to debug dictionary structure
            logger.debug(f"Cooldown info for {cooldown_key}: {cooldown_info}")

            # Extract data from cooldown_info dictionary
            last_triggered = cooldown_info.get("timestamp")
            last_interval = cooldown_info.get("interval", "unknown")
            last_strength = cooldown_info.get("strength", 5.0)

            # Determine base cooldown period based on last triggered interval
            base_cooldown_minutes = self.timeframe_cooldowns.get(
                last_interval, self.default_cooldown_minutes
            )

            # Calculate dynamic cooldown period based on ATR and market conditions
            adjusted_cooldown_minutes = self._get_atr_adjusted_cooldown(
                base_cooldown_minutes, last_interval, symbol, market_data
            )

            logger.debug(
                f"Using {adjusted_cooldown_minutes} minute cooldown (base: {base_cooldown_minutes}) for {last_interval} timeframe"
            )

            cooldown_period = timedelta(minutes=adjusted_cooldown_minutes)

            # Check if cooldown period has passed
            time_since_last = now - last_triggered
            if time_since_last < cooldown_period:
                # Still in cooldown
                minutes_remaining = int(
                    (cooldown_period - time_since_last).total_seconds() / 60
                )

                # Calculate how much of the cooldown has passed (as a ratio)
                cooldown_progress = (
                    time_since_last.total_seconds() / cooldown_period.total_seconds()
                )

                logger.debug(
                    f"{cooldown_key} is in cooldown: {minutes_remaining} minutes remaining. "
                    f"Current strength: {signal_strength:.1f}, Previous: {last_strength:.1f}, "
                    f"Progress: {cooldown_progress:.1%}"
                )

                # No override logic here - the OverrideEngine will handle this
                return False

            # Cooldown period has passed
            logger.debug(
                f"Cooldown expired for {cooldown_key}, last triggered {time_since_last.total_seconds()/60:.1f} minutes ago"
            )
            return True

    def update_cooldown(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str = None,
        interval: str = None,
        signal_strength: float = 5.0,
    ) -> None:
        """
        Update cooldown timestamp for a symbol and alert type

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name (e.g., 'RsiAlert')
        alert_subtype : str, optional
            Specific alert condition (e.g., 'OVERSOLD', 'OVERBOUGHT')
        interval : str, optional
            Timeframe of the current alert (e.g., '5m', '1h', '4h')
        signal_strength : float, optional
            Signal strength from 1.0 to 10.0
        """
        now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator

        with self.lock:
            # Create a consistent cooldown key using the dedicated method
            cooldown_key = self._create_cooldown_key(symbol, alert_type, alert_subtype, interval)

            # Update or create the cooldown entry
            cooldown_data = {
                "timestamp": now,
                "interval": interval,
                "strength": signal_strength,
            }

            # Update in-memory cooldown
            self.cooldowns[cooldown_key] = cooldown_data

            # Update repository cooldown
            self.repository.set_cooldown(cooldown_key, cooldown_data.copy())

            logger.debug(
                f"Updated cooldown for {cooldown_key} with interval {interval or 'None'} and strength {signal_strength:.1f}"
            )

            # Auto-save the repository if enabled
            if self.auto_save_enabled:
                self.repository.save_to_file()

    def _get_atr_adjusted_cooldown(
        self, base_cooldown: int, interval: str, symbol: str, market_data=None
    ) -> int:
        """
        Adjust cooldown based on ATR volatility and time of day

        Parameters:
        -----------
        base_cooldown : int
            Base cooldown in minutes
        interval : str
            Timeframe interval
        symbol : str
            Symbol for the alert
        market_data : DataFrame, optional
            Market data with ATR calculated

        Returns:
        --------
        int
            Adjusted cooldown in minutes
        """
        # Start with the base cooldown
        adjusted_cooldown = base_cooldown

        # Apply market session volatility adjustment if enabled
        if self._is_high_volatility_session() and interval in ["1m", "3m", "5m", "15m"]:
            # Add 10% to cooldown during market open hours for short timeframes
            adjusted_cooldown = int(adjusted_cooldown * 1.1)
            logger.debug(
                f"High volatility session adjustment: +10% = {adjusted_cooldown} min"
            )

        # Apply ATR-based volatility adjustment when market data is available
        if market_data is not None and isinstance(market_data, dict):
            # ATR percentile is between 0-100, where higher means more volatile
            atr_percentile = market_data.get("ATR_Percentile")

            if atr_percentile is not None:
                # Adjusted to provide fewer alerts in volatile markets and more in steady markets:
                # - Bottom 25% (0-25): Reduce cooldown by 25% (low volatility = more alerts)
                # - Top 25% (75-100): Extend cooldown by 25% (high volatility = fewer alerts)
                # - Middle 50% (25-75): Linear interpolation between 0.75x and 1.25x

                # Calculate the adjustment factor with the inverted logic
                if atr_percentile <= 25:
                    adjustment_factor = (
                        0.75  # Low volatility = shorter cooldown = more alerts
                    )
                elif atr_percentile >= 75:
                    adjustment_factor = (
                        1.25  # High volatility = longer cooldown = fewer alerts
                    )
                else:
                    # Linear interpolation between 0.75 and 1.25 for values between 25 and 75
                    # At 50%, should be exactly 1.0
                    adjustment_factor = 0.75 + ((atr_percentile - 25) / 50) * 0.5

                # Apply adjustment and ensure it's an integer
                adjusted_cooldown = int(base_cooldown * adjustment_factor)

                logger.debug(
                    f"ATR-based volatility adjustment: ATR percentile {atr_percentile:.1f}, "
                    f"Factor {adjustment_factor:.2f}, New cooldown {adjusted_cooldown} min"
                )

        return adjusted_cooldown

    def _is_high_volatility_session(self) -> bool:
        """
        Check if the current time is during a typically high-volatility market session.
        Used to potentially extend cooldowns during volatile market hours.

        Returns:
        --------
        bool
            True if current time is in a high volatility session, False otherwise
        """
        # Get current UTC hour
        current_hour = (datetime.utcnow() + timedelta(hours=1)).hour  # Add 1 hour to match batch_aggregator

        # Market opens are typically high volatility:
        # - US market open: 13:30-15:30 UTC
        # - Asian market open: 0:00-02:00 UTC
        high_volatility_hours = [0, 1, 13, 14]

        return current_hour in high_volatility_hours

    def clear_cooldowns(self) -> None:
        """Clear all cooldowns from memory and repository"""
        with self.lock:
            self.cooldowns.clear()
            self.repository.clear_all_cooldowns()

            # Save the cleared state
            if self.auto_save_enabled:
                self.repository.save_to_file(force=True)

            logger.info("Cleared all cooldowns")

    def get_cooldown_info(
        self, symbol: str, alert_type: str, alert_subtype: str = None, interval: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cooldown information for a specific alert

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name (e.g., 'RsiAlert')
        alert_subtype : str, optional
            Specific alert condition (e.g., 'OVERSOLD', 'OVERBOUGHT')
        interval : str, optional
            Timeframe of the alert (e.g., '5m', '1h', '4h')

        Returns:
        --------
        dict or None
            Cooldown information if found, None otherwise
        """
        with self.lock:
            # Create a consistent cooldown key using the dedicated method
            cooldown_key = self._create_cooldown_key(symbol, alert_type, alert_subtype, interval)

            # Get cooldown data
            cooldown_data = self.cooldowns.get(cooldown_key)

            if cooldown_data:
                # Return a copy to prevent modification
                return cooldown_data.copy()

            # Check repository as a fallback
            return self.repository.get_cooldown(cooldown_key)

    def get_all_cooldowns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cooldowns

        Returns:
        --------
        dict
            Dictionary of all cooldowns
        """
        with self.lock:
            # Return a copy to prevent modification
            return {k: v.copy() for k, v in self.cooldowns.items()}

    def prune_expired_cooldowns(self, max_age_hours: int = 24) -> int:
        """
        Remove cooldowns older than the specified time

        Parameters:
        -----------
        max_age_hours : int
            Maximum age in hours for cooldown records

        Returns:
        --------
        int
            Number of cooldowns removed
        """
        with self.lock:
            now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
            cutoff_time = now - timedelta(hours=max_age_hours)
            keys_to_remove = []

            # Find expired cooldowns
            for key, data in self.cooldowns.items():
                timestamp = data.get("timestamp")
                if timestamp and timestamp < cutoff_time:
                    keys_to_remove.append(key)

            # Remove them from memory
            for key in keys_to_remove:
                del self.cooldowns[key]

            # Prune from repository
            repo_pruned = self.repository.prune_expired_cooldowns(max_age_hours)

            # Save changes if any were made
            if keys_to_remove and self.auto_save_enabled:
                self.repository.save_to_file()

            total_pruned = len(keys_to_remove)
            logger.info(
                f"Pruned {total_pruned} expired cooldowns older than {max_age_hours} hours"
            )
            return total_pruned

    def set_auto_save(self, enabled: bool) -> None:
        """
        Enable or disable automatic saving to repository

        Parameters:
        -----------
        enabled : bool
            Whether to enable auto-save
        """
        with self.lock:
            self.auto_save_enabled = enabled
            logger.debug(f"Auto-save {'enabled' if enabled else 'disabled'}")

    def save_cooldowns(self, force: bool = True) -> bool:
        """
        Manually save cooldowns to the repository

        Parameters:
        -----------
        force : bool
            Force save even if no changes detected

        Returns:
        --------
        bool
            True if save was successful, False otherwise
        """
        return self._save_to_repository(force=force)


# Singleton instance
_cooldown_service = None


def get_cooldown_service() -> CooldownService:
    """Get the singleton CooldownService instance"""
    global _cooldown_service
    if _cooldown_service is None:
        _cooldown_service = CooldownService()
    return _cooldown_service
