import pytest
import pandas as pd
import numpy as np

from bot.indicators import (
    validate_data,
    calculate_rsi,
    calculate_macd,
    calculate_ema_cross,
    calculate_bollinger_bands,
    calculate_volume_spikes,
    calculate_adx,
)


class TestValidateData:
    """Tests for data validation function"""

    def test_validate_empty_dataframe(self):
        """Test with empty dataframe"""
        df = pd.DataFrame()
        assert validate_data(df, 10) is False

    def test_validate_too_few_rows(self):
        """Test with dataframe that has fewer rows than min_periods"""
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [1000, 2000, 3000],
            }
        )
        assert validate_data(df, 5) is False  # Need at least 5 periods
        assert validate_data(df, 3) is True  # 3 periods is enough

    def test_validate_missing_columns(self):
        """Test with dataframe missing required columns"""
        df = pd.DataFrame(
            {"close": [1, 2, 3, 4, 5], "volume": [1000, 2000, 3000, 4000, 5000]}
        )
        assert validate_data(df, 3) is False  # Missing open, high, low

    def test_validate_nan_values(self):
        """Test with dataframe containing NaN values"""
        df = pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [2, 3, 4, 5, 6],
                "low": [0.5, 1.5, 2.5, 3.5, 4.5],
                "close": [1.5, 2.5, np.nan, 4.5, 5.5],  # NaN in close
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )
        assert validate_data(df, 3) is False

    def test_validate_valid_data(self, sample_ohlcv_data):
        """Test with valid dataframe"""
        assert validate_data(sample_ohlcv_data, 50) is True


class TestRSI:
    """Tests for RSI calculation"""

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test basic RSI calculation"""
        rsi = calculate_rsi(sample_ohlcv_data)
        assert rsi is not None
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_ohlcv_data)
        assert 0 <= rsi.min() <= rsi.max() <= 100  # RSI range is 0-100

    def test_rsi_normalize(self, sample_ohlcv_data):
        """Test RSI normalization option"""
        rsi_normalized = calculate_rsi(sample_ohlcv_data, normalize=True)
        assert rsi_normalized is not None
        assert isinstance(rsi_normalized, pd.Series)
        assert (
            0 <= rsi_normalized.min() <= rsi_normalized.max() <= 1
        )  # Normalized range is 0-1

    def test_rsi_custom_length(self, sample_ohlcv_data):
        """Test RSI with custom length parameter"""
        rsi_default = calculate_rsi(sample_ohlcv_data)  # Default length=14
        rsi_custom = calculate_rsi(sample_ohlcv_data, length=7)  # Custom shorter length

        assert rsi_custom is not None
        assert len(rsi_custom) == len(sample_ohlcv_data)
        # The shorter period RSI should be more responsive (more variance)
        assert rsi_custom.std() > rsi_default.std()


class TestMACD:
    """Tests for MACD calculation"""

    def test_macd_calculation(self, sample_ohlcv_data):
        """Test basic MACD calculation"""
        macd_df = calculate_macd(sample_ohlcv_data)
        assert macd_df is not None
        assert isinstance(macd_df, pd.DataFrame)

        # Check expected columns
        assert "MACD" in macd_df.columns
        assert "Signal" in macd_df.columns
        assert "Histogram" in macd_df.columns

        # Check lengths
        assert len(macd_df) == len(sample_ohlcv_data)

    def test_macd_custom_parameters(self, sample_ohlcv_data):
        """Test MACD with custom parameters"""
        macd_df = calculate_macd(sample_ohlcv_data, fast=8, slow=21, signal=5)
        assert macd_df is not None
        assert "MACD" in macd_df.columns
        assert "Signal" in macd_df.columns
        assert "Histogram" in macd_df.columns


class TestEMACross:
    """Tests for EMA Cross calculation"""

    def test_ema_cross_calculation(self, sample_ohlcv_data):
        """Test basic EMA cross calculation"""
        ema_df = calculate_ema_cross(sample_ohlcv_data)
        assert ema_df is not None
        assert isinstance(ema_df, pd.DataFrame)

        # Check expected columns (default periods are 9 and 21)
        assert "EMA9" in ema_df.columns
        assert "EMA21" in ema_df.columns
        assert "Cross_Up" in ema_df.columns
        assert "Cross_Down" in ema_df.columns

        # Check if crosses are boolean
        assert ema_df["Cross_Up"].dtype == bool
        assert ema_df["Cross_Down"].dtype == bool

        # Make sure we don't have both cross up and down in same candle
        assert not ((ema_df["Cross_Up"] & ema_df["Cross_Down"]).any())

    def test_ema_cross_custom_periods(self, sample_ohlcv_data):
        """Test EMA cross with custom periods"""
        ema_df = calculate_ema_cross(sample_ohlcv_data, short=5, long=15)
        assert ema_df is not None

        # Check custom column names
        assert "EMA5" in ema_df.columns
        assert "EMA15" in ema_df.columns


class TestBollingerBands:
    """Tests for Bollinger Bands calculation"""

    def test_bollinger_bands_calculation(self, sample_ohlcv_data):
        """Test basic Bollinger Bands calculation"""
        bb_df = calculate_bollinger_bands(sample_ohlcv_data)
        assert bb_df is not None
        assert isinstance(bb_df, pd.DataFrame)

        # Check expected columns
        assert "BBL" in bb_df.columns  # Lower band
        assert "BBM" in bb_df.columns  # Middle band (SMA)
        assert "BBU" in bb_df.columns  # Upper band
        assert "BandWidth" in bb_df.columns

        # Verify bands relationship: Lower < Middle < Upper
        assert (bb_df["BBL"] <= bb_df["BBM"]).all()
        assert (bb_df["BBM"] <= bb_df["BBU"]).all()

    def test_bollinger_bands_percentb(self, sample_ohlcv_data):
        """Test Bollinger Bands with percentB normalization"""
        bb_df = calculate_bollinger_bands(sample_ohlcv_data, normalize=True)
        assert bb_df is not None
        assert "PercentB" in bb_df.columns

        # %B should generally be between 0 and 1 (can go outside in extreme cases)
        assert bb_df["PercentB"].min() >= -0.1  # Allow slight margin
        assert bb_df["PercentB"].max() <= 1.1  # Allow slight margin


class TestVolumeSpikes:
    """Tests for Volume Spikes calculation"""

    def test_volume_spikes_calculation(self, sample_ohlcv_data):
        """Test basic Volume Spikes calculation"""
        vol_df = calculate_volume_spikes(sample_ohlcv_data)
        assert vol_df is not None
        assert isinstance(vol_df, pd.DataFrame)

        # Check expected columns
        assert "volume" in vol_df.columns
        assert "avg_volume" in vol_df.columns
        assert "volume_ratio" in vol_df.columns
        assert "spike" in vol_df.columns

        # Verify spike detection
        assert (vol_df["spike"] == (vol_df["volume_ratio"] > 2.0)).all()

    def test_volume_spikes_zscore(self, sample_ohlcv_data):
        """Test Volume Spikes with z-score method"""
        vol_df = calculate_volume_spikes(sample_ohlcv_data, z_score=True, threshold=2.0)
        assert vol_df is not None

        # Check z-score specific columns
        assert "z_score" in vol_df.columns
        assert "spike" in vol_df.columns

        # Verify spike detection with z-score
        assert (vol_df["spike"] == (vol_df["z_score"] > 2.0)).all()


class TestADX:
    """Tests for ADX calculation"""

    def test_adx_calculation(self, sample_ohlcv_data):
        """Test basic ADX calculation"""
        adx_df = calculate_adx(sample_ohlcv_data)
        assert adx_df is not None
        assert isinstance(adx_df, pd.DataFrame)

        # Check expected columns
        assert "ADX" in adx_df.columns
        assert "DI+" in adx_df.columns
        assert "DI-" in adx_df.columns
        assert "Trend" in adx_df.columns
        assert "Strength" in adx_df.columns

        # ADX should be between 0 and 100
        assert adx_df["ADX"].min() >= 0
        assert adx_df["ADX"].max() <= 100

        # Check trend determination logic
        bullish_rows = adx_df["DI+"] > adx_df["DI-"]
        assert (adx_df.loc[bullish_rows, "Trend"] == "Bullish").all()
        assert (adx_df.loc[~bullish_rows, "Trend"] == "Bearish").all()
