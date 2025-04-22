import concurrent.futures
import logging
import random
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from bot.alerts import AlertManager
from bot.db import DatabaseManager
from bot.scheduler import AlertScheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_load")


class TestLoadPerformance:
    """Tests for system performance under load"""

    def test_database_concurrent_access(self):
        """Test database handles multiple concurrent requests"""
        db = DatabaseManager(":memory:")
        db.create_tables()

        # Number of concurrent operations
        n_threads = 10
        n_operations = 50

        # List to collect any errors
        errors = []

        # Create a bunch of test users
        for i in range(n_threads):
            db.create_user(f"user_{i}", f"User {i}")

        # Function to run in threads
        def worker_thread(thread_id):
            try:
                # Perform a mix of read and write operations
                for _ in range(n_operations):
                    operation = random.choice(["add", "remove", "get", "update"])
                    user_id = f"user_{thread_id}"
                    symbol = f"COIN{random.randint(1, 5)}USDT"

                    if operation == "add":
                        db.add_to_watchlist(user_id, symbol)
                    elif operation == "remove":
                        db.remove_from_watchlist(user_id, symbol)
                    elif operation == "get":
                        db.get_user_watchlist(user_id)
                    elif operation == "update":
                        db.update_user_settings(
                            user_id, {"rsi_oversold": random.randint(20, 40)}
                        )
            except Exception as e:
                errors.append(f"Error in thread {thread_id}: {e}")

        # Start timer
        start_time = time.time()

        # Run threads
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # End timer
        end_time = time.time()
        elapsed = end_time - start_time

        # Log performance metrics
        operations_per_second = (n_threads * n_operations) / elapsed
        logger.info(
            f"Database load test: {n_threads} threads * {n_operations} operations = {n_threads * n_operations} total operations"
        )
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Operations per second: {operations_per_second:.2f}")

        # Check for errors
        assert not errors, f"Errors occurred during concurrent access: {errors}"

        # Clean up
        db.close()

    @patch("bot.scheduler.fetch_market_data")
    def test_alert_processing_performance(self, mock_fetch_data, sample_ohlcv_data):
        """Test alert processing performance with many users and symbols"""
        db = DatabaseManager(":memory:")
        db.create_tables()

        # Number of users and symbols to test
        n_users = 20
        n_symbols = 10

        # Create test users and watchlists
        for i in range(n_users):
            user_id = f"user_{i}"
            db.create_user(user_id, f"User {i}")

            # Each user watches a random selection of symbols
            for j in range(5):  # Each user watches 5 symbols
                symbol_idx = random.randint(0, n_symbols - 1)
                symbol = f"COIN{symbol_idx}USDT"
                db.add_to_watchlist(user_id, symbol)

        # Mock the market data fetch to return sample data
        mock_fetch_data.return_value = sample_ohlcv_data

        # Create a callback to count alerts
        alert_count = 0

        def alert_callback(user_id, alerts):
            nonlocal alert_count
            alert_count += len(alerts)

        # Create and initialize the scheduler
        scheduler = AlertScheduler(alert_callback)

        # Start timer
        start_time = time.time()

        # Check alerts for all symbols
        for i in range(n_symbols):
            symbol = f"COIN{i}USDT"
            scheduler.check_symbol_alerts(symbol, "15m")

        # End timer
        end_time = time.time()
        elapsed = end_time - start_time

        # Log performance metrics
        logger.info(f"Alert processing test: {n_users} users, {n_symbols} symbols")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Alerts triggered: {alert_count}")
        logger.info(f"Symbols processed per second: {n_symbols / elapsed:.2f}")

        # Clean up
        db.close()

    @patch("bot.binance.requests.get")
    def test_concurrent_api_requests(self, mock_get):
        """Test handling multiple concurrent API requests"""
        from bot.binance import fetch_market_data

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
        ]

        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = klines_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Number of concurrent requests
        n_requests = 50

        # List of symbols to fetch
        symbols = [f"COIN{i}USDT" for i in range(10)]

        # Start timer
        start_time = time.time()

        # Function to fetch data
        def fetch_symbol_data(symbol):
            try:
                return fetch_market_data(symbol=symbol, interval="15m", limit=2)
            except Exception as e:
                return str(e)

        # Perform concurrent requests using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(fetch_symbol_data, random.choice(symbols)): i
                for i in range(n_requests)
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_symbol):
                request_id = future_to_symbol[future]
                try:
                    data = future.result()
                    results.append((request_id, True))
                except Exception as e:
                    results.append((request_id, False, str(e)))

        # End timer
        end_time = time.time()
        elapsed = end_time - start_time

        # Count successful requests
        successful = sum(1 for _, success, *_ in results if success)

        # Log performance metrics
        logger.info(f"Concurrent API requests test: {n_requests} requests")
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        logger.info(f"Successful requests: {successful}/{n_requests}")
        logger.info(f"Requests per second: {n_requests / elapsed:.2f}")

        # All requests should succeed
        assert (
            successful == n_requests
        ), f"Only {successful}/{n_requests} requests succeeded"
