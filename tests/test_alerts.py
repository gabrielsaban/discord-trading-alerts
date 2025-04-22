import pytest
import pandas as pd

from bot.alerts import (
    AlertCondition,
    RsiAlert,
    MacdAlert,
    EmaCrossAlert,
    BollingerBandAlert,
    VolumeSpikeAlert,
    AdxAlert,
    PatternAlert,
    AlertManager,
)


class TestAlertCondition:
    """Tests for the base AlertCondition class"""

    def test_alert_init(self):
        """Test initialization of AlertCondition"""
        alert = AlertCondition("BTCUSDT", cooldown_minutes=60)
        assert alert.symbol == "BTCUSDT"
        assert alert.last_triggered is None
        assert alert.cooldown.total_seconds() == 60 * 60  # 60 minutes in seconds

    def test_can_trigger_first_time(self):
        """Test can_trigger when alert has not been triggered yet"""
        alert = AlertCondition("BTCUSDT")
        assert alert.can_trigger() is True

    def test_mark_triggered(self):
        """Test mark_triggered updates the last_triggered timestamp"""
        alert = AlertCondition("BTCUSDT")
        assert alert.last_triggered is None
        alert.mark_triggered()
        assert alert.last_triggered is not None

    def test_price_formatting(self):
        """Test price formatting logic for different price ranges"""
        alert = AlertCondition("BTCUSDT")

        # Small prices
        assert alert.format_price(0.00034567) == "0.00034567"
        assert alert.format_price(0.00456789) == "0.004568"

        # Medium prices
        assert alert.format_price(0.123456) == "0.1235"
        assert alert.format_price(12.3456) == "12.35"

        # Large prices
        assert alert.format_price(1234.56) == "1234.6"
        assert alert.format_price(12345.6) == "12345.6"


class TestAlertManager:
    """Tests for the AlertManager class"""

    def test_add_alert(self, alert_manager, mock_alert):
        """Test adding an alert to the manager"""
        alert = mock_alert("BTCUSDT")
        alert_manager.add_alert(alert)

        assert "BTCUSDT" in alert_manager.alerts
        assert len(alert_manager.alerts["BTCUSDT"]) == 1
        assert alert_manager.alerts["BTCUSDT"][0] == alert

    def test_remove_alert(self, alert_manager, mock_alert):
        """Test removing an alert by type"""
        alert1 = mock_alert("BTCUSDT")
        alert2 = RsiAlert("BTCUSDT")

        alert_manager.add_alert(alert1)
        alert_manager.add_alert(alert2)

        assert len(alert_manager.alerts["BTCUSDT"]) == 2

        # Remove by type
        alert_manager.remove_alert("BTCUSDT", RsiAlert)

        assert len(alert_manager.alerts["BTCUSDT"]) == 1
        assert isinstance(alert_manager.alerts["BTCUSDT"][0], type(alert1))

    def test_clear_alerts(self, alert_manager, mock_alert):
        """Test clearing alerts"""
        alert1 = mock_alert("BTCUSDT")
        alert2 = mock_alert("ETHUSDT")

        alert_manager.add_alert(alert1)
        alert_manager.add_alert(alert2)

        assert len(alert_manager.get_symbols()) == 2

        # Clear specific symbol
        alert_manager.clear_alerts("BTCUSDT")

        assert "BTCUSDT" not in alert_manager.alerts
        assert "ETHUSDT" in alert_manager.alerts

        # Clear all
        alert_manager.clear_alerts()

        assert len(alert_manager.alerts) == 0

    def test_check_alerts(self, alert_manager, mock_alert, sample_ohlcv_data):
        """Test checking alerts"""
        # Add a non-triggering alert
        alert1 = mock_alert("BTCUSDT", should_trigger=False)
        # Add a triggering alert
        alert2 = mock_alert("BTCUSDT", should_trigger=True)

        alert_manager.add_alert(alert1)
        alert_manager.add_alert(alert2)

        # Check alerts
        triggered = alert_manager.check_alerts("BTCUSDT", sample_ohlcv_data)

        # Both should have been checked
        assert alert1.was_checked
        assert alert2.was_checked

        # Only the second should have triggered
        assert len(triggered) == 1
        assert triggered[0] == "MOCK ALERT: BTCUSDT"

    def test_get_symbols(self, alert_manager, mock_alert):
        """Test getting all symbols with alerts"""
        alert_manager.add_alert(mock_alert("BTCUSDT"))
        alert_manager.add_alert(mock_alert("ETHUSDT"))
        alert_manager.add_alert(mock_alert("SOLUSDT"))

        symbols = alert_manager.get_symbols()

        assert len(symbols) == 3
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "SOLUSDT" in symbols

    def test_get_alert_count(self, alert_manager, mock_alert):
        """Test getting alert count"""
        alert_manager.add_alert(mock_alert("BTCUSDT"))
        alert_manager.add_alert(mock_alert("BTCUSDT"))  # Second alert for same symbol
        alert_manager.add_alert(mock_alert("ETHUSDT"))

        # Count for specific symbol
        assert alert_manager.get_alert_count("BTCUSDT") == 2
        assert alert_manager.get_alert_count("ETHUSDT") == 1

        # Total count
        assert alert_manager.get_alert_count() == 3


class TestAlertTypes:
    """Tests for specific alert types"""

    def test_rsi_alert(self, sample_ohlcv_data):
        """Test RSI alert"""
        alert = RsiAlert("TESTSYMBOL", oversold=30, overbought=70)
        result = alert.check(sample_ohlcv_data)
        # Basic check that it runs without error
        assert isinstance(result, str) or result is None

    def test_macd_alert(self, sample_ohlcv_data):
        """Test MACD alert"""
        alert = MacdAlert("TESTSYMBOL")
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None

    def test_ema_cross_alert(self, sample_ohlcv_data):
        """Test EMA Cross alert"""
        alert = EmaCrossAlert("TESTSYMBOL", short=9, long=21)
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None

    def test_bollinger_band_alert(self, sample_ohlcv_data):
        """Test Bollinger Band alert"""
        alert = BollingerBandAlert("TESTSYMBOL", squeeze_threshold=0.05)
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None

    def test_volume_spike_alert(self, sample_ohlcv_data):
        """Test Volume Spike alert"""
        alert = VolumeSpikeAlert("TESTSYMBOL", threshold=2.5)
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None

    def test_adx_alert(self, sample_ohlcv_data):
        """Test ADX alert"""
        alert = AdxAlert("TESTSYMBOL", threshold=25)
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None

    def test_pattern_alert(self, sample_ohlcv_data):
        """Test Pattern alert"""
        alert = PatternAlert("TESTSYMBOL")
        result = alert.check(sample_ohlcv_data)
        assert isinstance(result, str) or result is None
