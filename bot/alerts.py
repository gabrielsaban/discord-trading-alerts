import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

# Import indicator functions
from bot.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_ema_cross,
    calculate_bollinger_bands,
    calculate_volume_spikes,
    calculate_adx,
)

# Configure logger
logger = logging.getLogger(__name__)


class AlertCondition:
    """Base class for alert conditions"""

    def __init__(self, symbol: str, cooldown_minutes: int = 240):
        """
        Initialize alert condition

        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        cooldown_minutes : int
            Minimum minutes between repeated alerts
        """
        self.symbol = symbol
        self.last_triggered = None  # Timestamp of last alert
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
        """Check if alert can trigger based on cooldown"""
        now = datetime.now()
        if not self.last_triggered:
            logger.debug(
                f"{type(self).__name__} for {self.symbol} has never triggered before"
            )
            return True

        elapsed = now - self.last_triggered
        cooldown_expired = elapsed > self.cooldown

        if not cooldown_expired:
            minutes_remaining = int(self.cooldown.total_seconds() / 60) - int(
                elapsed.total_seconds() / 60
            )
            logger.debug(
                f"{type(self).__name__} for {self.symbol} in cooldown ({minutes_remaining} minutes remaining)"
            )

        return cooldown_expired

    def mark_triggered(self):
        """Mark alert as triggered now"""
        self.last_triggered = datetime.now()

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


class RsiAlert(AlertCondition):
    """Alert for RSI crossing above/below thresholds"""

    def __init__(
        self,
        symbol: str,
        oversold: float = 30,
        overbought: float = 70,
        cooldown_minutes: int = 240,
    ):
        super().__init__(symbol, cooldown_minutes)
        self.oversold = oversold
        self.overbought = overbought

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None

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
            message = f"🔴 **RSI OVERSOLD**: {self.symbol} RSI at {latest_rsi:.1f}\nPrice: {price_str}"
        # Check for overbought condition (crossing above threshold)
        elif latest_rsi > self.overbought and prev_rsi <= self.overbought:
            logger.info(
                f"RSI overbought alert triggered for {self.symbol}: prev_rsi={prev_rsi:.1f}, latest_rsi={latest_rsi:.1f}, threshold={self.overbought}"
            )
            message = f"🟢 **RSI OVERBOUGHT**: {self.symbol} RSI at {latest_rsi:.1f}\nPrice: {price_str}"

        if message:
            self.mark_triggered()
        return message


class MacdAlert(AlertCondition):
    """Alert for MACD signal line crossovers"""

    def __init__(self, symbol: str, cooldown_minutes: int = 240):
        super().__init__(symbol, cooldown_minutes)

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None
            
        macd_df = calculate_macd(df)
        if macd_df is None or len(macd_df) < 2:
            return None
            
        # Get last two rows to detect crosses
        prev, latest = macd_df.iloc[-2], macd_df.iloc[-1]
        latest_price = df['close'].iloc[-1]
        price_str = self.format_price(latest_price)
        
        message = None
        # Check for bullish crossover (MACD crosses above Signal)
        if latest['MACD'] > latest['Signal'] and prev['MACD'] <= prev['Signal']:
            message = f"🟢 **MACD BULLISH CROSS**: {self.symbol}\nPrice: {price_str}"
        # Check for bearish crossover (MACD crosses below Signal)
        elif latest['MACD'] < latest['Signal'] and prev['MACD'] >= prev['Signal']:
            message = f"🔴 **MACD BEARISH CROSS**: {self.symbol}\nPrice: {price_str}"
        
        if message:
            self.mark_triggered()
        return message


class EmaCrossAlert(AlertCondition):
    """Alert for EMA crossovers"""

    def __init__(
        self, symbol: str, short: int = 9, long: int = 21, cooldown_minutes: int = 240
    ):
        super().__init__(symbol, cooldown_minutes)
        self.short = short
        self.long = long

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None
            
        ema_df = calculate_ema_cross(df, short=self.short, long=self.long)
        if ema_df is None or len(ema_df) < 2:
            return None
            
        latest_price = df['close'].iloc[-1]
        price_str = self.format_price(latest_price)
        
        message = None
        # Check for bullish cross (short EMA crosses above long EMA)
        if ema_df['Cross_Up'].iloc[-1]:
            message = f"🟢 **EMA BULLISH CROSS**: {self.symbol} EMA{self.short} crossed above EMA{self.long}\nPrice: {price_str}"
        # Check for bearish cross (short EMA crosses below long EMA)
        elif ema_df['Cross_Down'].iloc[-1]:
            message = f"🔴 **EMA BEARISH CROSS**: {self.symbol} EMA{self.short} crossed below EMA{self.long}\nPrice: {price_str}"
        
        if message:
            self.mark_triggered()
        return message


class BollingerBandAlert(AlertCondition):
    """Alert for price breaking out of Bollinger Bands"""

    def __init__(
        self, symbol: str, cooldown_minutes: int = 240, squeeze_threshold: float = 0.05
    ):
        super().__init__(symbol, cooldown_minutes)
        self.squeeze_threshold = squeeze_threshold  # For band squeeze detection

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None

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
            message = f"🟢 **UPPER BB BREAKOUT**: {self.symbol} Price broke above upper band\nPrice: {price_str}"

        # Lower band breakout (price crosses below lower band)
        elif latest_price < current["BBL"] and prev_close >= prev["BBL"]:
            message = f"🔴 **LOWER BB BREAKOUT**: {self.symbol} Price broke below lower band\nPrice: {price_str}"

        # Band squeeze (bandwidth narrowing significantly)
        elif (
            current["BandWidth"] < self.squeeze_threshold
            and current["BandWidth"] < prev["BandWidth"]
        ):
            message = f"🟡 **BOLLINGER SQUEEZE**: {self.symbol} Bands narrowing, potential breakout\nPrice: {price_str}"

        if message:
            self.mark_triggered()
        return message


class VolumeSpikeAlert(AlertCondition):
    """Alert for unusual volume spikes"""

    def __init__(
        self,
        symbol: str,
        threshold: float = 2.5,
        cooldown_minutes: int = 240,
        z_score: bool = False,
    ):
        super().__init__(symbol, cooldown_minutes)
        self.threshold = threshold
        self.z_score = z_score

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None

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
            message = f"📊 **VOLUME Z-SCORE SPIKE**: {self.symbol} {direction}\nZ-score: {latest['z_score']:.1f}\nPrice: {price_str}"
        else:
            message = f"📊 **VOLUME SPIKE**: {self.symbol} {direction}\n{latest['volume_ratio']:.1f}x average\nPrice: {price_str}"

        self.mark_triggered()
        return message


class AdxAlert(AlertCondition):
    """Alert for strong trends detected by ADX"""

    def __init__(self, symbol: str, threshold: int = 25, cooldown_minutes: int = 240):
        super().__init__(symbol, cooldown_minutes)
        self.threshold = threshold

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None

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
            message = f"📏 **STRONG {direction} TREND**: {self.symbol} ADX: {latest['ADX']:.1f}\nPrice: {price_str}"

        # Trend direction change during strong trend
        elif (
            latest["Strength"] and prev["Strength"] and latest["Trend"] != prev["Trend"]
        ):
            new_direction = "BULLISH 📈" if latest["Trend"] == "Bullish" else "BEARISH 📉"
            message = f"🔄 **TREND REVERSAL to {new_direction}**: {self.symbol} ADX: {latest['ADX']:.1f}\nPrice: {price_str}"

        if message:
            self.mark_triggered()
        return message


class PatternAlert(AlertCondition):
    """Alert for common price patterns"""

    def __init__(self, symbol: str, cooldown_minutes: int = 240):
        super().__init__(symbol, cooldown_minutes)

    def check(self, df: pd.DataFrame) -> Optional[str]:
        if not self.can_trigger():
            return None

        if len(df) < 5:  # Need at least 5 candles for pattern detection
            return None

        # Get recent candles
        candles = df.iloc[-5:]
        latest_price = df["close"].iloc[-1]
        price_str = self.format_price(latest_price)

        message = None

        # Simple hammer detection (long lower wick, small body, small/no upper wick)
        if self._is_hammer(candles.iloc[-1]):
            message = f"🔨 **HAMMER PATTERN**: {self.symbol} Potential reversal\nPrice: {price_str}"

        # Simple evening star (bullish-neutral-bearish sequence at top)
        elif self._is_evening_star(candles.iloc[-3:]):
            message = f"⭐ **EVENING STAR**: {self.symbol} Potential bearish reversal\nPrice: {price_str}"

        # Simple morning star (bearish-neutral-bullish sequence at bottom)
        elif self._is_morning_star(candles.iloc[-3:]):
            message = f"⭐ **MORNING STAR**: {self.symbol} Potential bullish reversal\nPrice: {price_str}"

        # Engulfing pattern
        elif self._is_engulfing(candles.iloc[-2:]):
            pattern_type = (
                "BULLISH"
                if candles.iloc[-1]["close"] > candles.iloc[-1]["open"]
                else "BEARISH"
            )
            message = (
                f"🔄 **{pattern_type} ENGULFING**: {self.symbol}\nPrice: {price_str}"
            )

        if message:
            self.mark_triggered()
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

    def __init__(self):
        self.alerts: Dict[str, List[AlertCondition]] = {}
        # Track global cooldowns across timeframes - {symbol: {alert_type: last_triggered_time}}
        self.global_cooldowns: Dict[str, Dict[str, datetime]] = {}
        # Global cooldown period in minutes (across all timeframes)
        self.global_cooldown_minutes = 10  # 10 minute global cooldown

    def add_alert(self, alert: AlertCondition):
        """Add alert condition for a symbol"""
        if alert.symbol not in self.alerts:
            self.alerts[alert.symbol] = []
        self.alerts[alert.symbol].append(alert)

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
            # Also clear global cooldowns for this symbol
            if symbol in self.global_cooldowns:
                del self.global_cooldowns[symbol]
        else:
            self.alerts.clear()
            self.global_cooldowns.clear()

    def _is_globally_cooled_down(self, symbol: str, alert_type: str) -> bool:
        """Check if an alert type for a symbol is in global cooldown
        
        Returns True if the alert can trigger (NOT in cooldown)
        Returns False if the alert should not trigger (IS in cooldown)
        """
        now = datetime.now()
        
        # Initialize cooldown tracking for this symbol if needed
        if symbol not in self.global_cooldowns:
            self.global_cooldowns[symbol] = {}
            
        # If this alert type has never been triggered, it's not in cooldown
        if alert_type not in self.global_cooldowns[symbol]:
            return True
            
        # Check if cooldown period has passed
        last_triggered = self.global_cooldowns[symbol][alert_type]
        cooldown_period = timedelta(minutes=self.global_cooldown_minutes)
        
        if now - last_triggered < cooldown_period:
            # Still in cooldown
            minutes_remaining = int((cooldown_period - (now - last_triggered)).total_seconds() / 60)
            logger.debug(f"{alert_type} for {symbol} in GLOBAL cooldown ({minutes_remaining} minutes remaining)")
            return False
        
        # Cooldown period has passed
        return True

    def _update_global_cooldown(self, symbol: str, alert_type: str):
        """Mark an alert type as triggered for global cooldown tracking"""
        if symbol not in self.global_cooldowns:
            self.global_cooldowns[symbol] = {}
            
        self.global_cooldowns[symbol][alert_type] = datetime.now()
        logger.debug(f"Updated global cooldown for {alert_type} on {symbol}")

    def check_alerts(self, symbol: str, df: pd.DataFrame) -> List[str]:
        """Check all alerts for a symbol and return triggered messages"""
        if symbol not in self.alerts:
            logger.debug(f"No alerts registered for {symbol}")
            return []

        logger.debug(f"Checking {len(self.alerts[symbol])} alerts for {symbol}")

        triggered = []
        for alert in self.alerts[symbol]:
            # Get alert type name for global cooldown tracking
            alert_type = type(alert).__name__
            logger.debug(f"Checking alert type {alert_type} for {symbol}")
            
            # First check global cooldown (across timeframes)
            if not self._is_globally_cooled_down(symbol, alert_type):
                logger.debug(f"Skipping {alert_type} for {symbol} due to global cooldown")
                continue
                
            # Check the alert
            message = alert.check(df)
            if message:
                logger.info(f"Alert triggered: {message}")
                # Update global cooldown for this alert type
                self._update_global_cooldown(symbol, alert_type)
                triggered.append(message)

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
        """Set up default alerts for a symbol"""
        self.add_alert(RsiAlert(symbol))
        self.add_alert(MacdAlert(symbol))
        self.add_alert(EmaCrossAlert(symbol))
        self.add_alert(BollingerBandAlert(symbol))
        self.add_alert(VolumeSpikeAlert(symbol))
        self.add_alert(AdxAlert(symbol))
        self.add_alert(PatternAlert(symbol))


# Testing functionality
if __name__ == "__main__":
    from binance import fetch_market_data
    import time

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

        # Check alerts
        alerts = manager.check_alerts(symbol, df)

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
