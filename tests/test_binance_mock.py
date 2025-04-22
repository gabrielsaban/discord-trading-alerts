import pytest
import pandas as pd
import numpy as np
import json
import logging
from unittest.mock import patch, MagicMock
from requests.exceptions import RequestException, Timeout, ConnectionError

from bot.binance import fetch_market_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_binance_mock")


class TestBinanceAPI:
    """Tests for handling various Binance API responses and error conditions"""

    @patch("bot.binance.requests.get")
    def test_successful_api_response(self, mock_get):
        """Test processing a successful API response with valid data"""
        # Create sample klines data that would be returned by Binance API
        klines_data = [
            [
                1627776000000,
                "40000",
                "41000",
                "39500",
                "40500",
                "100",
                1627779599999,
                "4050000",
                1000,
                "50",
                "2025000",
                "0",
            ],
            [
                1627779600000,
                "40500",
                "42000",
                "40400",
                "41800",
                "120",
                1627783199999,
                "5016000",
                1200,
                "60",
                "2508000",
                "0",
            ],
            [
                1627783200000,
                "41800",
                "42500",
                "41500",
                "42200",
                "90",
                1627786799999,
                "3798000",
                900,
                "45",
                "1899000",
                "0",
            ],
        ]

        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = klines_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Fetch the data
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify the data was properly processed
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(df) == 3, "Should have 3 candles"
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ], "DataFrame should have OHLCV columns"
        assert (
            df["open"].iloc[0] == 40000
        ), "First candle should have open price of 40000"

    @patch("bot.binance.requests.get")
    def test_empty_response(self, mock_get):
        """Test handling an empty but valid API response"""
        # Mock an empty klines array
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Fetch the data
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns an empty DataFrame, not None
        assert isinstance(df, pd.DataFrame), "Should return empty DataFrame, not None"
        assert df.empty, "DataFrame should be empty"

    @patch("bot.binance.requests.get")
    def test_http_error_response(self, mock_get):
        """Test handling an HTTP error response"""
        # Mock a 400 Bad Request response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"code": -1121, "msg": "Invalid symbol"}
        mock_response.raise_for_status.side_effect = RequestException(
            "400 Client Error: Bad Request"
        )
        mock_get.return_value = mock_response

        # Fetch the data - should handle error gracefully
        df = fetch_market_data(symbol="INVALID", interval="15m", limit=3)

        # Verify it returns None on error
        assert df is None, "Should return None on HTTP error"

    @patch("bot.binance.requests.get")
    def test_timeout_error(self, mock_get):
        """Test handling a request timeout"""
        # Mock a timeout error
        mock_get.side_effect = Timeout("Request timed out")

        # Fetch the data - should handle timeout gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None on timeout
        assert df is None, "Should return None on timeout"

    @patch("bot.binance.requests.get")
    def test_connection_error(self, mock_get):
        """Test handling a connection error"""
        # Mock a connection error
        mock_get.side_effect = ConnectionError("Connection failed")

        # Fetch the data - should handle connection error gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None on connection error
        assert df is None, "Should return None on connection error"

    @patch("bot.binance.requests.get")
    def test_invalid_json_response(self, mock_get):
        """Test handling invalid JSON response"""
        # Mock a response with invalid JSON
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        # Fetch the data - should handle JSON decode error gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None on JSON decode error
        assert df is None, "Should return None on JSON decode error"

    @patch("bot.binance.requests.get")
    def test_malformed_response(self, mock_get):
        """Test handling malformed response data"""
        # Mock a response with malformed data
        malformed_data = [
            # Missing fields
            [1627776000000, "40000"],
            # Wrong data types
            [
                1627779600000,
                "not_a_number",
                "41000",
                "39500",
                "40500",
                "100",
                1627779599999,
                "4050000",
                1000,
                "50",
                "2025000",
                "0",
            ],
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = malformed_data
        mock_get.return_value = mock_response

        # Fetch the data - should handle malformed data gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None or empty DataFrame on malformed data
        assert df is None or df.empty, "Should handle malformed data gracefully"

    @patch("bot.binance.requests.get")
    def test_rate_limit_response(self, mock_get):
        """Test handling a rate limit error response"""
        # Mock a 429 Too Many Requests response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "code": -1003,
            "msg": "Too many requests; IP banned",
        }
        mock_response.raise_for_status.side_effect = RequestException(
            "429 Client Error: Too Many Requests"
        )
        mock_get.return_value = mock_response

        # Fetch the data - should handle rate limit error gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None on rate limit error
        assert df is None, "Should return None on rate limit error"

    @patch("bot.binance.requests.get")
    def test_server_error_response(self, mock_get):
        """Test handling a server error response"""
        # Mock a 500 Internal Server Error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = RequestException(
            "500 Server Error: Internal Server Error"
        )
        mock_get.return_value = mock_response

        # Fetch the data - should handle server error gracefully
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=3)

        # Verify it returns None on server error
        assert df is None, "Should return None on server error"

    @patch("bot.binance.requests.get")
    def test_invalid_interval(self, mock_get):
        """Test handling an invalid interval"""
        # Mock a successful response
        klines_data = [
            [
                1627776000000,
                "40000",
                "41000",
                "39500",
                "40500",
                "100",
                1627779599999,
                "4050000",
                1000,
                "50",
                "2025000",
                "0",
            ],
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = klines_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Fetch with invalid interval
        df = fetch_market_data(symbol="BTCUSDT", interval="invalid", limit=3)

        # Verify request still works (function should use a default interval)
        assert df is not None, "Should handle invalid interval gracefully"

    @patch("bot.binance.requests.get")
    def test_negative_limit(self, mock_get):
        """Test handling a negative limit parameter"""
        # Mock a successful response
        klines_data = [
            [
                1627776000000,
                "40000",
                "41000",
                "39500",
                "40500",
                "100",
                1627779599999,
                "4050000",
                1000,
                "50",
                "2025000",
                "0",
            ],
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = klines_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Fetch with negative limit
        df = fetch_market_data(symbol="BTCUSDT", interval="15m", limit=-5)

        # Verify request still works (function should use a positive limit)
        assert df is not None, "Should handle negative limit gracefully"
