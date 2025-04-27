import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logger first before any usage
logger = logging.getLogger("discord_trading_alerts.alerts")

# Import indicator functions
from bot.indicators import (
    calculate_adx,
    calculate_bollinger_bands,
    calculate_ema_cross,
    calculate_macd,
    calculate_rsi,
    calculate_volume_spikes,
)

# Using try/except allows tests to run without feature flag dependency
# and avoids circular imports
try:
    from bot.services.batch_aggregator import get_batch_aggregator
    from bot.services.cooldown_service import get_cooldown_service
    from bot.services.feature_flags import get_flag
    from bot.services.override_engine import get_override_engine

    FEATURE_FLAGS_AVAILABLE = True

    # Log feature flag status on module load
    logger.info("FEATURE FLAGS STATUS:")
    logger.info(
        f"  ENABLE_COOLDOWN_SERVICE: {get_flag('ENABLE_COOLDOWN_SERVICE', False)}"
    )
    logger.info(
        f"  ENABLE_OVERRIDE_ENGINE: {get_flag('ENABLE_OVERRIDE_ENGINE', False)}"
    )
    logger.info(
        f"  ENABLE_BATCH_AGGREGATOR: {get_flag('ENABLE_BATCH_AGGREGATOR', False)}"
    )
except ImportError:
    logger.warning("Feature flags module not available, using legacy implementation")
    FEATURE_FLAGS_AVAILABLE = False

    def get_flag(flag_name: str, default_value: Any = None) -> Any:
        """Fallback implementation when feature flags are not available"""
        return default_value


class AlertCondition(ABC):
    """Base class for alert conditions"""

    def __init__(self, symbol: str, cooldown_minutes: int = 10):
        """
        Initialize alert condition

        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        cooldown_minutes : int
            Minimum minutes between repeated alerts (legacy parameter, unused)
        """
        self.symbol = symbol
        # Store interval for reference by AlertManager
        self.interval = None
        # We keep these fields for backward compatibility
        self.last_triggered = None
        self.cooldown = timedelta(minutes=cooldown_minutes)

    def check(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check if alert condition is met

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV dataframe

        Returns:
        --------
        str or None
            Alert message if condition is met, None otherwise
        """
        raise NotImplementedError("Subclasses must implement check()")

    def can_trigger(self) -> bool:
        """
        DEPRECATED: Use AlertManager._is_globally_cooled_down instead.
        Legacy cooldown check - not used in current system.
        """
        logger.debug(f"DEPRECATED can_trigger() called on {type(self).__name__}")
        return True  # Always allow AlertManager to handle cooldowns

    def mark_triggered(self, strength=None):
        """Mark alert as triggered now, with optional strength value."""
        self.triggered = True
        self.last_triggered = datetime.utcnow() + timedelta(hours=1)
        self.strength = strength if strength is not None else self.strength

    def format_price(self, price: float) -> str:
        """Format price based on magnitude"""
        if price < 0.001:
            return f"{price:.8f}"
        elif price < 0.01:
            return f"{price:.6f}"
        elif price < 1:
            return f"{price:.4f}"
        elif price < 1000:
            return f"{price:.2f}"
        else:
            return f"{price:.1f}"

    # These methods are deprecated and should not be used
    def calculate_dynamic_cooldown(
        self,
        pair,
        atr_percentile=None,
        is_high_volatility=False,
        is_strong_signal=False,
    ):
        """
        DEPRECATED: Use AlertManager._get_atr_adjusted_cooldown instead.
        Legacy method for dynamic cooldowns - not used in current system.
        """
        raise NotImplementedError(
            "Legacy cooldown method: calculate_dynamic_cooldown is no longer supported. Use CooldownService instead."
        )

    def is_in_cooldown(
        self,
        pair,
        timestamp,
        atr_percentile=None,
        is_high_volatility=False,
        is_strong_signal=False,
    ):
        """
        DEPRECATED: Use AlertManager._is_globally_cooled_down instead.
        Legacy method for cooldown checking - not used in current system.
        """
        raise NotImplementedError(
            "Legacy cooldown method: is_in_cooldown is no longer supported. Use CooldownService instead."
        )

    def add_cooldown(self, pair, timestamp):
        """
        DEPRECATED: Use AlertManager._update_global_cooldown instead.
        Legacy method for adding cooldowns - not used in current system.
        """
        raise NotImplementedError(
            "Legacy cooldown method: add_cooldown is no longer supported. Use CooldownService instead."
        )


class RsiAlert(AlertCondition):
    """Alert for RSI crossing above/below thresholds"""

    def __init__(
        self,
        symbol: str,
        oversold: float = 30,
        overbought: float = 70,
        cooldown_minutes: int = 10,
    ):
        super().__init__(symbol, cooldown_minutes)
        self.oversold = oversold
        self.overbought = overbought

    def check(self, df: pd.DataFrame) -> Optional[str]:
        rsi = calculate_rsi(df)
        if rsi is None or len(rsi) < 2:
            logger.debug(f"RSI calculation failed for {self.symbol}")
            return None

        # Get last two values to detect crosses
        prev_rsi, latest_rsi = rsi.iloc[-2], rsi.iloc[-1]
        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        # Debug log
        logger.debug(
            f"RSI check for {self.symbol}: prev_rsi={prev_rsi:.1f}, latest_rsi={latest_rsi:.1f}, oversold={self.oversold}, overbought={self.overbought}"
        )

        message = None
        # Check for oversold condition (crossing below threshold)
        if latest_rsi < self.oversold and prev_rsi >= self.oversold:
            logger.info(
                f"RSI oversold alert triggered for {self.symbol}: prev_rsi={prev_rsi:.1f}, latest_rsi={latest_rsi:.1f}, threshold={self.oversold}"
            )
            message = f"🔴 **RSI OVERSOLD**: {self.symbol} RSI at {latest_rsi:.1f}\nPrice: {price_str}\nThreshold: {self.oversold}, Latest RSI: {latest_rsi:.1f}"
        # Check for overbought condition (crossing above threshold)
        elif latest_rsi > self.overbought and prev_rsi <= self.overbought:
            logger.info(
                f"RSI overbought alert triggered for {self.symbol}: prev_rsi={prev_rsi:.1f}, latest_rsi={latest_rsi:.1f}, threshold={self.overbought}"
            )
            message = f"🟢 **RSI OVERBOUGHT**: {self.symbol} RSI at {latest_rsi:.1f}\nPrice: {price_str}\nThreshold: {self.overbought}, Latest RSI: {latest_rsi:.1f}"

        return message


class MacdAlert(AlertCondition):
    """Alert for MACD signal line crossovers"""

    def __init__(self, symbol: str, cooldown_minutes: int = 10):
        super().__init__(symbol, cooldown_minutes)

    def check(self, df: pd.DataFrame) -> Optional[str]:
        macd_df = calculate_macd(df)
        if macd_df is None or len(macd_df) < 2:
            return None

        # Get last two rows to detect crosses
        prev, latest = macd_df.iloc[-2], macd_df.iloc[-1]
        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        message = None
        # Check for bullish crossover (MACD crosses above Signal)
        if latest["MACD"] > latest["Signal"] and prev["MACD"] <= prev["Signal"]:
            message = f"🟢 **MACD BULLISH CROSS**: {self.symbol}\nPrice: {price_str}\nMACD: {latest['MACD']:.4f}, Signal: {latest['Signal']:.4f}, Histogram: {latest['Histogram']:.4f}"
        # Check for bearish crossover (MACD crosses below Signal)
        elif latest["MACD"] < latest["Signal"] and prev["MACD"] >= prev["Signal"]:
            message = f"🔴 **MACD BEARISH CROSS**: {self.symbol}\nPrice: {price_str}\nMACD: {latest['MACD']:.4f}, Signal: {latest['Signal']:.4f}, Histogram: {latest['Histogram']:.4f}"

        return message


class EmaCrossAlert(AlertCondition):
    """Alert for EMA crossovers"""

    def __init__(
        self, symbol: str, short: int = 9, long: int = 21, cooldown_minutes: int = 10
    ):
        super().__init__(symbol, cooldown_minutes)
        self.short = short
        self.long = long

    def check(self, df: pd.DataFrame) -> Optional[str]:
        ema_df = calculate_ema_cross(df, short=self.short, long=self.long)
        if ema_df is None or len(ema_df) < 2:
            return None

        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        # Get the latest EMA values - using correct column names
        latest_short_ema = ema_df[f"EMA{self.short}"].iloc[-1]
        latest_long_ema = ema_df[f"EMA{self.long}"].iloc[-1]

        message = None
        # Check for bullish cross (short EMA crosses above long EMA)
        if ema_df["Cross_Up"].iloc[-1]:
            message = f"🟢 **EMA BULLISH CROSS**: {self.symbol} EMA{self.short} crossed above EMA{self.long}\nPrice: {price_str}\nEMA{self.short}: {latest_short_ema:.4f}, EMA{self.long}: {latest_long_ema:.4f}"
        # Check for bearish cross (short EMA crosses below long EMA)
        elif ema_df["Cross_Down"].iloc[-1]:
            message = f"🔴 **EMA BEARISH CROSS**: {self.symbol} EMA{self.short} crossed below EMA{self.long}\nPrice: {price_str}\nEMA{self.short}: {latest_short_ema:.4f}, EMA{self.long}: {latest_long_ema:.4f}"

        return message


class BollingerBandAlert(AlertCondition):
    """Alert for price breaking out of Bollinger Bands"""

    def __init__(
        self, symbol: str, cooldown_minutes: int = 10, squeeze_threshold: float = 0.05
    ):
        super().__init__(symbol, cooldown_minutes)
        self.squeeze_threshold = squeeze_threshold  # For band squeeze detection

    def check(self, df: pd.DataFrame) -> Optional[str]:
        bb = calculate_bollinger_bands(df, length=20, std=2, normalize=True)
        if bb is None or len(bb) < 2:
            return None

        latest_price = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        price_str = self.format_price(latest_price)

        # Get current and previous datapoints
        prev, current = bb.iloc[-2], bb.iloc[-1]

        message = None

        # Upper band breakout (price crosses above upper band)
        if latest_price > current["BBU"] and prev_close <= prev["BBU"]:
            message = f"🟢 **UPPER BB BREAKOUT**: {self.symbol} Price broke above upper band\nPrice: {price_str}\nUpper Band: {current['BBU']:.4f}, Middle Band: {current['BBM']:.4f}, Lower Band: {current['BBL']:.4f}"

        # Lower band breakout (price crosses below lower band)
        elif latest_price < current["BBL"] and prev_close >= prev["BBL"]:
            message = f"🔴 **LOWER BB BREAKOUT**: {self.symbol} Price broke below lower band\nPrice: {price_str}\nUpper Band: {current['BBU']:.4f}, Middle Band: {current['BBM']:.4f}, Lower Band: {current['BBL']:.4f}"

        # Band squeeze (bandwidth narrowing significantly)
        elif (
            current["BandWidth"] < self.squeeze_threshold
            and current["BandWidth"] < prev["BandWidth"]
        ):
            message = f"🟡 **BOLLINGER SQUEEZE**: {self.symbol} Bands narrowing, potential breakout\nPrice: {price_str}\nBandwidth: {current['BandWidth']:.4f}, Threshold: {self.squeeze_threshold}"

        return message


class VolumeSpikeAlert(AlertCondition):
    """Alert for unusual volume spikes"""

    def __init__(
        self,
        symbol: str,
        threshold: float = 2.5,
        cooldown_minutes: int = 10,
        z_score: bool = False,
    ):
        super().__init__(symbol, cooldown_minutes)
        self.threshold = threshold
        self.z_score = z_score

    def check(self, df: pd.DataFrame) -> Optional[str]:
        vol_df = calculate_volume_spikes(
            df, length=20, threshold=self.threshold, z_score=self.z_score
        )
        if vol_df is None or len(vol_df) < 1:
            return None

        latest = vol_df.iloc[-1]
        if not latest["spike"]:
            return None

        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        # Determine if price moved up or down with the volume spike
        price_change = df["close"].iloc[-1] - df["open"].iloc[-1]
        direction = "UP 📈" if price_change > 0 else "DOWN 📉"

        if self.z_score:
            message = f"📊 **VOLUME Z-SCORE SPIKE**: {self.symbol} {direction}\nZ-score: {latest['z_score']:.1f}\nPrice: {price_str}\nThreshold: {self.threshold}, Z-score: {latest['z_score']:.2f}"
        else:
            message = f"📊 **VOLUME SPIKE**: {self.symbol} {direction}\n{latest['volume_ratio']:.1f}x average\nPrice: {price_str}\nThreshold: {self.threshold}x, Current: {latest['volume_ratio']:.2f}x"

        return message


class AdxAlert(AlertCondition):
    """Alert for strong trends detected by ADX"""

    def __init__(self, symbol: str, threshold: int = 25, cooldown_minutes: int = 10):
        super().__init__(symbol, cooldown_minutes)
        self.threshold = threshold

    def check(self, df: pd.DataFrame) -> Optional[str]:
        adx_df = calculate_adx(df, length=14, threshold=self.threshold)
        if adx_df is None or len(adx_df) < 2:
            return None

        prev, latest = adx_df.iloc[-2], adx_df.iloc[-1]
        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        message = None

        # New strong trend starting (ADX crosses above threshold)
        if latest["Strength"] and not prev["Strength"]:
            direction = "BULLISH 📈" if latest["Trend"] == "Bullish" else "BEARISH 📉"
            message = f"📏 **STRONG {direction} TREND**: {self.symbol} ADX: {latest['ADX']:.1f}\nPrice: {price_str}\nThreshold: {self.threshold}, Current ADX: {latest['ADX']:.1f}, Direction: {latest['Trend']}"

        # Trend direction change during strong trend
        elif (
            latest["Strength"] and prev["Strength"] and latest["Trend"] != prev["Trend"]
        ):
            new_direction = "BULLISH 📈" if latest["Trend"] == "Bullish" else "BEARISH 📉"
            message = f"🔄 **TREND REVERSAL to {new_direction}**: {self.symbol} ADX: {latest['ADX']:.1f}\nPrice: {price_str}\nThreshold: {self.threshold}, Current ADX: {latest['ADX']:.1f}, Previous Direction: {prev['Trend']}"

        return message


class PatternAlert(AlertCondition):
    """Alert for common price patterns"""

    def __init__(self, symbol: str, cooldown_minutes: int = 10):
        super().__init__(symbol, cooldown_minutes)

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < 5:  # Need at least 5 candles for pattern detection
            return None

        # Get recent candles
        candles = df.iloc[-5:]
        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        message = None

        # Simple hammer detection (long lower wick, small body, small/no upper wick)
        if self._is_hammer(candles.iloc[-1]):
            body = abs(candles.iloc[-1]["close"] - candles.iloc[-1]["open"])
            lower_wick = (
                min(candles.iloc[-1]["open"], candles.iloc[-1]["close"])
                - candles.iloc[-1]["low"]
            )
            upper_wick = candles.iloc[-1]["high"] - max(
                candles.iloc[-1]["open"], candles.iloc[-1]["close"]
            )
            message = f"🔨 **HAMMER PATTERN**: {self.symbol} Potential reversal\nPrice: {price_str}\nBody: {body:.4f}, Lower Wick: {lower_wick:.4f}, Upper Wick: {upper_wick:.4f}"

        # Simple evening star (bullish-neutral-bearish sequence at top)
        elif self._is_evening_star(candles.iloc[-3:]):
            message = f"⭐ **EVENING STAR**: {self.symbol} Potential bearish reversal\nPrice: {price_str}\nPattern: Bullish → Doji → Bearish, Confidence: High"

        # Simple morning star (bearish-neutral-bullish sequence at bottom)
        elif self._is_morning_star(candles.iloc[-3:]):
            message = f"⭐ **MORNING STAR**: {self.symbol} Potential bullish reversal\nPrice: {price_str}\nPattern: Bearish → Doji → Bullish, Confidence: High"

        # Engulfing pattern
        elif self._is_engulfing(candles.iloc[-2:]):
            pattern_type = (
                "BULLISH"
                if candles.iloc[-1]["close"] > candles.iloc[-1]["open"]
                else "BEARISH"
            )
            c1_body = abs(candles.iloc[-2]["close"] - candles.iloc[-2]["open"])
            c2_body = abs(candles.iloc[-1]["close"] - candles.iloc[-1]["open"])
            body_ratio = c2_body / c1_body if c1_body > 0 else 0
            message = f"🔄 **{pattern_type} ENGULFING**: {self.symbol}\nPrice: {price_str}\nBody Ratio: {body_ratio:.2f}x, Confidence: {'High' if body_ratio > 1.5 else 'Medium'}"

        return message

    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer"""
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])

        # Lower wick should be at least 2x body, upper wick should be small
        return lower_wick > (body * 2) and upper_wick < (body * 0.5)

    def _is_evening_star(self, candles: pd.DataFrame) -> bool:
        """Check for evening star pattern (bullish-neutral-bearish)"""
        if len(candles) != 3:
            return False

        c1_bullish = candles.iloc[0]["close"] > candles.iloc[0]["open"]
        c2_small = (
            abs(candles.iloc[1]["close"] - candles.iloc[1]["open"])
            < abs(candles.iloc[0]["close"] - candles.iloc[0]["open"]) * 0.5
        )
        c3_bearish = candles.iloc[2]["close"] < candles.iloc[2]["open"]

        return c1_bullish and c2_small and c3_bearish

    def _is_morning_star(self, candles: pd.DataFrame) -> bool:
        """Check for morning star pattern (bearish-neutral-bullish)"""
        if len(candles) != 3:
            return False

        c1_bearish = candles.iloc[0]["close"] < candles.iloc[0]["open"]
        c2_small = (
            abs(candles.iloc[1]["close"] - candles.iloc[1]["open"])
            < abs(candles.iloc[0]["close"] - candles.iloc[0]["open"]) * 0.5
        )
        c3_bullish = candles.iloc[2]["close"] > candles.iloc[2]["open"]

        return c1_bearish and c2_small and c3_bullish

    def _is_engulfing(self, candles: pd.DataFrame) -> bool:
        """Check for engulfing pattern"""
        if len(candles) != 2:
            return False

        c1_body = abs(candles.iloc[0]["close"] - candles.iloc[0]["open"])
        c2_body = abs(candles.iloc[1]["close"] - candles.iloc[1]["open"])

        c1_bullish = candles.iloc[0]["close"] > candles.iloc[0]["open"]
        c2_bullish = candles.iloc[1]["close"] > candles.iloc[1]["open"]

        # Different directions and second candle's body engulfs first candle's body
        if c1_bullish != c2_bullish and c2_body > c1_body:
            if c2_bullish:  # Bullish engulfing
                return (
                    candles.iloc[1]["open"] < candles.iloc[0]["close"]
                    and candles.iloc[1]["close"] > candles.iloc[0]["open"]
                )
            else:  # Bearish engulfing
                return (
                    candles.iloc[1]["open"] > candles.iloc[0]["close"]
                    and candles.iloc[1]["close"] < candles.iloc[0]["open"]
                )

        return False


class AlertManager:
    """Manages multiple alert conditions for different symbols"""

    # Class variable to control verbose logging
    _verbose_logging = False

    def __init__(self):
        self.alerts: Dict[str, List[AlertCondition]] = {}

        # Maps intervals to their ATR reference intervals for volatility adjustment
        # This is kept for reference but actual cooldown logic is in CooldownService
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

        # Batched alerts storage: {symbol: {alert_type: [messages]}}
        self.batched_alerts = {}
        # Last batch send time
        self.last_batch_send = {}
        # Current interval for this manager instance
        self.current_interval = None
        # Current user ID for alert identification
        self.current_user_id = "unknown"

    def add_alert(self, alert: AlertCondition):
        """Add alert condition for a symbol"""
        if alert.symbol not in self.alerts:
            self.alerts[alert.symbol] = []

        # Set the interval on the alert condition
        alert.interval = self.current_interval

        self.alerts[alert.symbol].append(alert)

        # Use more concise logging to prevent spam
        if AlertManager._verbose_logging:
            logger.debug(
                f"Added {type(alert).__name__} for {alert.symbol} with interval {alert.interval}"
            )

    def remove_alert(self, symbol: str, alert_type: type):
        """Remove all alerts of a specific type for a symbol"""
        if symbol not in self.alerts:
            return

        self.alerts[symbol] = [
            alert for alert in self.alerts[symbol] if not isinstance(alert, alert_type)
        ]

    def clear_alerts(self, symbol: str = None):
        """Clear all alerts for a symbol or all symbols"""
        if symbol:
            if symbol in self.alerts:
                del self.alerts[symbol]
        else:
            self.alerts.clear()

    def _calculate_signal_strength(
        self, alert_type: str, alert_subtype: str, message: str
    ) -> float:
        """Calculate signal strength based on indicator values and timeframe

        Higher values indicate stronger signals that may override cooldowns.
        Base scale: 1-10 where 5 is average strength, 10 is very strong
        """
        strength = 5.0  # Default/average strength

        # Extract values from the message based on alert type
        if alert_type == "RsiAlert":
            if "OVERSOLD" in alert_subtype:
                # Extract RSI value - format: "RSI at XX.X"
                try:
                    rsi_value = float(message.split("RSI at ")[1].split("\n")[0])
                    # Lower RSI = stronger signal for oversold
                    if rsi_value <= 20:
                        strength = 9.0
                    elif rsi_value <= 25:
                        strength = 7.0
                    elif rsi_value <= 30:
                        strength = 5.0
                except:
                    pass
            elif "OVERBOUGHT" in alert_subtype:
                try:
                    rsi_value = float(message.split("RSI at ")[1].split("\n")[0])
                    # Higher RSI = stronger signal for overbought
                    if rsi_value >= 80:
                        strength = 9.0
                    elif rsi_value >= 75:
                        strength = 7.0
                    elif rsi_value >= 70:
                        strength = 5.0
                except:
                    pass

        elif alert_type == "BollingerBandAlert" and "SQUEEZE" in alert_subtype:
            # Extract bandwidth - tighter squeeze = stronger signal
            try:
                bandwidth = float(message.split("Bandwidth: ")[1].split(",")[0])
                if bandwidth < 0.02:
                    strength = 8.5
                elif bandwidth < 0.05:
                    strength = 7.0
                elif bandwidth < 0.1:
                    strength = 5.0
            except:
                pass

        elif alert_type == "AdxAlert":
            # Extract ADX value - higher ADX = stronger trend signal
            try:
                adx_value = float(message.split("ADX: ")[1].split("\n")[0])
                if adx_value >= 40:
                    strength = 9.0
                elif adx_value >= 30:
                    strength = 7.0
                elif adx_value >= 25:
                    strength = 5.0
            except:
                pass

        elif alert_type == "VolumeSpikeAlert":
            # Extract volume ratio - higher ratio = stronger signal
            try:
                if "x average" in message:
                    volume_ratio = float(message.split("x average")[0].split("\n")[-1])
                    if volume_ratio >= 5.0:
                        strength = 9.0
                    elif volume_ratio >= 3.0:
                        strength = 7.0
                    elif volume_ratio >= 2.0:
                        strength = 5.0
            except:
                pass

        # Pattern signals tend to be more reliable on longer timeframes
        elif alert_type == "PatternAlert":
            # Base strength for pattern signals is slightly higher
            strength = 6.0

        # Extract interval from message to adjust signal strength
        interval = None
        if "(" in message and ")" in message:
            interval_part = message.split("(")[1].split(")")[0]
            if interval_part in self.atr_reference_intervals:
                interval = interval_part

        # Adjust strength based on timeframe - higher timeframes get priority
        if interval:
            # Higher timeframes get a strength boost
            if interval == "4h" or interval == "1d":
                strength += 3.0  # Significant boost for 4h/1d signals
            elif interval == "1h" or interval == "2h":
                strength += 1.5  # Medium boost for 1h/2h
            elif interval == "15m" or interval == "30m":
                strength += 0.5  # Small boost for 15m/30m

        return min(10.0, strength)  # Cap at 10.0

    def _is_high_volatility_session(self) -> bool:
        """Check if current time is during a high-volatility session (London/NY overlap)

        Returns:
        --------
        bool
            True if current time is during London/NY overlap (13:00-16:00 UTC)
        """
        now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
        hour = now.hour

        # London/NY overlap typically 13:00-16:00 UTC
        return 13 <= hour < 16

    def _get_atr_adjusted_cooldown(
        self, base_cooldown: int, interval: str, symbol: str, market_data=None
    ) -> int:
        """
        DEPRECATED: This method is now handled by CooldownService

        Calculate cooldown adjusted for market volatility based on ATR

        Redirects to CooldownService for the actual implementation
        """
        try:
            from bot.services.cooldown_service import get_cooldown_service

            # Get the CooldownService instance
            service = get_cooldown_service()

            # Use the service's implementation
            return service._get_atr_adjusted_cooldown(
                base_cooldown, interval, symbol, market_data
            )

        except ImportError:
            # If we can't import the service, throw an error
            raise ImportError(
                "CooldownService is required but not available. "
                "Please ensure bot/services/cooldown_service.py exists and is properly set up."
            )

    def _is_globally_cooled_down(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str = None,
        interval: str = None,
        message: str = None,
        market_data=None,
    ) -> bool:
        """Check if an alert type for a symbol is in global cooldown

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
        market_data : pd.DataFrame, optional
            Market data with ATR calculated (if available)

        Returns:
        --------
        bool
            True if the alert can trigger (NOT in cooldown)
            False if the alert should not trigger (IS in cooldown)
        """
        # Calculate signal strength - used by multiple code paths
        signal_strength = 5.0  # Default strength
        if message:
            signal_strength = self._calculate_signal_strength(
                alert_type, alert_subtype or "", message
            )

        # Get the CooldownService instance
        try:
            from bot.services.cooldown_service import get_cooldown_service
            from bot.services.feature_flags import get_flag
            from bot.services.batch_aggregator import get_batch_aggregator

            service = get_cooldown_service()

            # Check cooldown using the service
            is_in_cooldown = service.is_in_cooldown(
                symbol,
                alert_type,
                alert_subtype,
                interval,
                message,
                signal_strength,
                market_data,
            )

            # Check for overrides using the override engine if needed
            if is_in_cooldown and get_flag("ENABLE_OVERRIDE_ENGINE", False):
                logger.debug(f"Using OverrideEngine for {symbol} {alert_type}")
                try:
                    from bot.services.override_engine import get_override_engine

                    engine = get_override_engine()

                    # Get cooldown info for potential override checks
                    cooldown_info = service.get_cooldown_info(
                        symbol, alert_type, alert_subtype
                    )
                    if cooldown_info:
                        last_triggered = cooldown_info.get(
                            "timestamp", datetime.utcnow() + timedelta(hours=1)
                        )
                        last_interval = cooldown_info.get("interval", interval)
                        last_strength = cooldown_info.get("strength", 5.0)

                        # Determine base cooldown period from cooldown service
                        base_cooldown_minutes = service.timeframe_cooldowns.get(
                            last_interval, service.default_cooldown_minutes
                        )

                        cooldown_period = timedelta(minutes=base_cooldown_minutes)
                        time_elapsed = (datetime.utcnow() + timedelta(hours=1)) - last_triggered

                        # Check if override engine allows override
                        can_override, reason = engine.can_override(
                            alert_type=alert_type,
                            alert_subtype=alert_subtype or "",
                            current_strength=signal_strength,
                            last_strength=last_strength or 5.0,
                            interval=interval or "unknown",
                            last_interval=last_interval,
                            time_elapsed=time_elapsed,
                            cooldown_period=cooldown_period,
                            message=message,
                        )

                        if can_override:
                            logger.info(f"Override allowed: {reason}")

                            # Check if we should send batched alerts or just bypass the cooldown
                            if get_flag("ENABLE_BATCH_AGGREGATOR", False):
                                # Queue for batch aggregator instead of sending immediately
                                self._queue_for_batching(
                                    symbol,
                                    alert_type,
                                    alert_subtype or "",
                                    message,
                                    interval or "unknown",
                                )
                                # Return False to indicate we're not sending immediately
                                return False
                            else:
                                # No batch aggregator, send immediately
                                return True
                        else:
                            logger.debug(f"Override denied: {reason}")

                except (ImportError, Exception) as e:
                    logger.error(f"Error checking override: {e}")

            # Queue for batch aggregator if in cooldown but batch aggregator is enabled
            if is_in_cooldown and get_flag("ENABLE_BATCH_AGGREGATOR", False):
                # Check if this exact alert was already sent to the batch aggregator recently
                # This prevents duplicates when the same alert is triggered multiple times in quick succession
                batch_aggregator = get_batch_aggregator()
                
                # Create a unique key for this specific alert
                alert_key = f"{symbol}_{alert_type}_{alert_subtype or 'general'}_{interval or 'unknown'}"
                
                # Only queue if this isn't a duplicate of something recently queued
                if not self._is_recently_queued_for_batch(alert_key):
                    self._queue_for_batching(
                        symbol,
                        alert_type,
                        alert_subtype or "",
                        message,
                        interval or "unknown",
                    )
                    # Mark that we just queued this alert
                    self._mark_queued_for_batch(alert_key)
                else:
                    logger.debug(f"Skipping batch queue for {alert_key} - already queued recently")

            # Return opposite of is_in_cooldown (True = can trigger)
            return not is_in_cooldown

        except ImportError:
            # If we can't import the service, throw an error - we no longer support legacy mode
            raise ImportError(
                "CooldownService is required but not available. "
                "Please ensure bot/services/cooldown_service.py exists and is properly set up."
            )

    # Dictionary to track recently queued batch alerts
    _batch_queue_track = {}
    
    def _is_recently_queued_for_batch(self, alert_key, max_age_seconds=300):
        """Check if an alert was recently queued for batch processing
        
        Parameters:
        -----------
        alert_key : str
            Unique key for the alert
        max_age_seconds : int
            Maximum age in seconds to consider "recent"
            
        Returns:
        --------
        bool
            True if the alert was queued recently, False otherwise
        """
        now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
        if alert_key in self._batch_queue_track:
            queued_time = self._batch_queue_track[alert_key]
            age = (now - queued_time).total_seconds()
            return age <= max_age_seconds
        return False
    
    def _mark_queued_for_batch(self, alert_key):
        """Mark an alert as queued for batch processing
        
        Parameters:
        -----------
        alert_key : str
            Unique key for the alert
        """
        self._batch_queue_track[alert_key] = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
        
        # Clean up old entries to prevent memory growth
        self._cleanup_batch_queue_track()
    
    def _cleanup_batch_queue_track(self, max_age_seconds=600):
        """Clean up old entries in the batch queue tracking dictionary
        
        Parameters:
        -----------
        max_age_seconds : int
            Maximum age in seconds to keep entries
        """
        now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
        keys_to_remove = []
        
        for key, timestamp in self._batch_queue_track.items():
            age = (now - timestamp).total_seconds()
            if age > max_age_seconds:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self._batch_queue_track[key]

    def _queue_for_batching(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str,
        message: str,
        interval: str,
    ):
        """Queue alert for batched sending when cooldown expires

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name
        alert_subtype : str
            Alert subtype
        message : str
            Alert message
        interval : str
            Timeframe of the alert
        """
        if not message:
            return

        try:
            # Import and use the BatchAggregator service
            from bot.services.batch_aggregator import get_batch_aggregator
            from bot.services.feature_flags import get_flag

            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                alert_type, alert_subtype or "", message
            )

            # Get user ID from instance if available
            user_id = getattr(self, "current_user_id", "unknown")

            # Only use batch aggregator if enabled by feature flag
            if get_flag("ENABLE_BATCH_AGGREGATOR", False):
                batch_aggregator = get_batch_aggregator()
                batch_aggregator.enqueue(
                    user_id=user_id,
                    symbol=symbol,
                    interval=interval,
                    alert_type=alert_type,
                    alert_subtype=alert_subtype,
                    alert_msg=message,
                    strength=signal_strength,
                )
                logger.info(
                    f"Queued alert for batch aggregator: {symbol} - {alert_type} ({interval})"
                )
            else:
                # BatchAggregator service is disabled, log warning
                logger.warning(
                    f"BatchAggregator service is disabled. Enable with ENABLE_BATCH_AGGREGATOR flag."
                )

        except ImportError:
            # If we can't import the service, throw an error - we no longer support legacy mode
            raise ImportError(
                "BatchAggregator is required but not available. "
                "Please ensure bot/services/batch_aggregator.py exists and is properly set up."
            )

    def get_batched_alerts(self, symbol: str = None) -> Dict[str, List[Dict]]:
        """Get batched alerts for a symbol or all symbols

        Parameters:
        -----------
        symbol : str, optional
            Symbol to get batched alerts for. If None, get all batched alerts.

        Returns:
        --------
        Dict[str, List[Dict]]
            Dictionary of batched alerts by symbol
        """
        try:
            # Import and use the BatchAggregator service
            from bot.services.batch_aggregator import get_batch_aggregator
            from bot.services.feature_flags import get_flag

            # Only use batch aggregator if enabled by feature flag
            if get_flag("ENABLE_BATCH_AGGREGATOR", False):
                batch_aggregator = get_batch_aggregator()
                # Process the batched alerts in the service
                # The batch aggregator's processing should happen automatically
                # based on its own timers, but we can trigger it manually if needed
                return (
                    {}
                )  # Return empty dict since actual processing happens via callbacks
            else:
                # BatchAggregator service is disabled, log warning
                logger.warning(
                    f"BatchAggregator service is disabled. Enable with ENABLE_BATCH_AGGREGATOR flag."
                )
                return {}

        except ImportError:
            # If we can't import the service, throw an error - we no longer support legacy mode
            raise ImportError(
                "BatchAggregator is required but not available. "
                "Please ensure bot/services/batch_aggregator.py exists and is properly set up."
            )

    def _update_global_cooldown(
        self,
        symbol: str,
        alert_type: str,
        alert_subtype: str = None,
        interval: str = None,
        message: str = None,
    ):
        """Mark an alert type as triggered for global cooldown tracking

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        alert_type : str
            Alert class name (e.g., 'RsiAlert')
        alert_subtype : str, optional
            Specific alert condition (e.g., 'OVERSOLD', 'OVERBOUGHT')
        interval : str, optional
            Timeframe of the triggered alert
        message : str, optional
            Alert message text, used to calculate signal strength
        """
        try:
            from bot.services.cooldown_service import get_cooldown_service

            # Get the CooldownService instance
            cooldown_service_instance = get_cooldown_service()

            # Calculate signal strength
            signal_strength = 5.0  # Default strength
            if message:
                signal_strength = self._calculate_signal_strength(
                    alert_type, alert_subtype or "", message
                )

            # Update cooldown using the service
            cooldown_service_instance.update_cooldown(
                symbol, alert_type, alert_subtype, interval, signal_strength
            )

        except ImportError:
            # If we can't import the service, throw an error - we no longer support legacy mode
            raise ImportError(
                "CooldownService is required but not available. "
                "Please ensure bot/services/cooldown_service.py exists and is properly set up."
            )

    def check_alerts(
        self, symbol: str, df: pd.DataFrame, interval: str = None, market_data=None
    ) -> List[str]:
        """Check all alerts for a symbol and return triggered messages

        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        df : pd.DataFrame
            OHLCV data for the symbol
        interval : str, optional
            Timeframe of the data (e.g., '5m', '1h', '4h')
        market_data : pd.DataFrame, optional
            Additional market data with ATR information
        """
        import time
        check_start = time.time()
        
        if symbol not in self.alerts:
            logger.debug(f"No alerts registered for {symbol}")
            return []

        alert_count = len(self.alerts[symbol])
        # Log only once to avoid excessive logging
        logger.debug(f"Checking {alert_count} alerts for {symbol} (interval: {interval})")

        triggered = []
        for idx, alert in enumerate(self.alerts[symbol]):
            # Get alert type name for global cooldown tracking
            alert_type = type(alert).__name__
            alert_start = time.time()
            logger.debug(f"[{idx+1}/{alert_count}] Checking alert type {alert_type} for {symbol}")

            # Check the alert
            message = alert.check(df)
            check_time = time.time() - alert_start
            
            if message:
                logger.debug(f"Alert {alert_type} for {symbol} triggered, message: {message[:50]}... (took {check_time:.2f}s)")
                
                # Extract alert subtype from message
                alert_subtype = None
                if "**" in message:
                    parts = message.split("**")
                    if len(parts) >= 3:
                        alert_subtype = parts[
                            1
                        ].strip()  # Get text between first set of **

                # First check global cooldown (across timeframes)
                cooldown_start = time.time()
                cooldown_check = self._is_globally_cooled_down(
                    symbol, alert_type, alert_subtype, interval, message, market_data
                )
                cooldown_time = time.time() - cooldown_start
                
                if not cooldown_check:
                    logger.info(
                        f"Skipping {alert_type} ({alert_subtype}) for {symbol} due to global cooldown (check took {cooldown_time:.2f}s)"
                    )
                    continue
                
                logger.info(f"Alert passed cooldown check in {cooldown_time:.2f}s and will be triggered: {message[:50]}...")
                # Update global cooldown for this alert type and subtype
                update_start = time.time()
                self._update_global_cooldown(
                    symbol, alert_type, alert_subtype, interval, message
                )
                update_time = time.time() - update_start
                logger.debug(f"Updated cooldown for {alert_type} ({alert_subtype}) in {update_time:.2f}s")
                
                triggered.append(message)
            else:
                logger.debug(f"Alert {alert_type} for {symbol} did not trigger (took {check_time:.2f}s)")

        total_time = time.time() - check_start
        logger.info(f"Completed checking {alert_count} alerts for {symbol} in {total_time:.2f}s, triggered: {len(triggered)}")
        return triggered

    def get_symbols(self) -> List[str]:
        """Get list of symbols with alerts"""
        return list(self.alerts.keys())

    def get_alert_count(self, symbol: str = None) -> int:
        """Get number of alerts for a symbol or total"""
        if symbol:
            return len(self.alerts.get(symbol, []))
        return sum(len(alerts) for alerts in self.alerts.values())

    def setup_default_alerts(self, symbol: str):
        """Set up default alerts for a symbol

        DEPRECATED: Use setup_timeframe_specific_alerts instead for better filtering
        """
        logger.warning(f"Using deprecated setup_default_alerts for {symbol}. Consider using setup_timeframe_specific_alerts instead.")
        self.add_alert(RsiAlert(symbol))
        self.add_alert(MacdAlert(symbol))
        self.add_alert(EmaCrossAlert(symbol))
        self.add_alert(BollingerBandAlert(symbol))
        self.add_alert(VolumeSpikeAlert(symbol))
        self.add_alert(AdxAlert(symbol))
        self.add_alert(PatternAlert(symbol))

    def setup_timeframe_specific_alerts(self, symbol: str, interval: str = None):
        """Set up alerts for a symbol based on timeframe appropriateness
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        interval : str, optional
            Timeframe interval (e.g., '5m', '15m', '1h', '4h')
            If not provided, uses the current interval of the AlertManager
        """
        # Use the current interval if none provided
        if interval is None:
            interval = self.current_interval
            
        # If still no interval, use default alerts as fallback
        if interval is None:
            logger.warning(f"No interval specified for {symbol}, using all alerts")
            self.setup_default_alerts(symbol)
            return
            
        logger.info(f"Setting up timeframe-specific alerts for {symbol} on {interval} interval")
        
        # Define interval-appropriate indicators
        short_timeframe_indicators = ["rsi", "volume"]
        medium_timeframe_indicators = ["rsi", "volume", "macd", "ema", "bb", "adx"]
        long_timeframe_indicators = ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"]
        
        # Determine appropriate indicators based on timeframe
        if interval in ["1m", "3m", "5m"]:
            appropriate_indicators = short_timeframe_indicators
            logger.info(f"Using short timeframe indicators for {symbol} ({interval}): {appropriate_indicators}")
        elif interval in ["15m", "30m", "1h"]:
            appropriate_indicators = medium_timeframe_indicators
            logger.info(f"Using medium timeframe indicators for {symbol} ({interval}): {appropriate_indicators}")
        else:  # 4h, 1d, etc.
            appropriate_indicators = long_timeframe_indicators
            logger.info(f"Using long timeframe indicators for {symbol} ({interval}): {appropriate_indicators}")
        
        # Set up the appropriate alerts for this timeframe
        if "rsi" in appropriate_indicators:
            self.add_alert(RsiAlert(symbol))
            
        if "macd" in appropriate_indicators:
            self.add_alert(MacdAlert(symbol))
            
        if "ema" in appropriate_indicators:
            self.add_alert(EmaCrossAlert(symbol))
            
        if "bb" in appropriate_indicators:
            self.add_alert(BollingerBandAlert(symbol))
            
        if "volume" in appropriate_indicators:
            self.add_alert(VolumeSpikeAlert(symbol))
            
        if "adx" in appropriate_indicators:
            self.add_alert(AdxAlert(symbol))
            
        if "pattern" in appropriate_indicators:
            self.add_alert(PatternAlert(symbol))

    def set_user_id(self, user_id: str):
        """Set the user ID for this AlertManager instance

        Parameters:
        -----------
        user_id : str
            User ID to associate with alerts from this manager
        """
        self.current_user_id = user_id
        logger.debug(f"Set user_id to {user_id} for AlertManager")
        return self


# Testing functionality
if __name__ == "__main__":
    import time

    from binance import fetch_market_data

    def test_alerts(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 100):
        print(f"\n===== Testing alerts for {symbol} ({interval}) =====\n")

        # Create alert manager
        manager = AlertManager()

        # Add all alert types
        manager.add_alert(RsiAlert(symbol, oversold=30, overbought=70))
        manager.add_alert(MacdAlert(symbol))
        manager.add_alert(EmaCrossAlert(symbol, short=9, long=21))
        manager.add_alert(BollingerBandAlert(symbol))
        manager.add_alert(VolumeSpikeAlert(symbol, threshold=2.0))
        manager.add_alert(AdxAlert(symbol))
        manager.add_alert(PatternAlert(symbol))

        # Fetch data
        df = fetch_market_data(symbol=symbol, interval=interval, limit=limit)

        # Check alerts - pass the interval parameter
        alerts = manager.check_alerts(symbol, df, interval)

        if alerts:
            print(f"\nTriggered alerts for {symbol}:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print(f"No alerts triggered for {symbol}")

        return alerts

    def test_multiple_pairs():
        pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]
        intervals = ["15m", "1h", "4h"]

        results = {}

        for pair in pairs:
            pair_results = []
            for interval in intervals:
                alerts = test_alerts(pair, interval)
                if alerts:
                    pair_results.extend(alerts)
            results[pair] = pair_results

        # Summary
        print("\n===== ALERT TESTING SUMMARY =====\n")
        total_alerts = sum(len(alerts) for alerts in results.values())
        print(f"Total alerts triggered: {total_alerts}")

        for pair, alerts in results.items():
            print(f"\n{pair}: {len(alerts)} alerts")
            for alert in alerts:
                print(f"  {alert}")

    # Run tests
    # test_multiple_pairs()
