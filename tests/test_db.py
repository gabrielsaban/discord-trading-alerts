import pytest
import os
import sqlite3
import threading
from datetime import datetime, timedelta

from bot.db import DatabaseManager


class TestDatabaseManager:
    """Tests for the DatabaseManager class"""

    def test_create_tables(self, in_memory_db):
        """Test table creation"""
        # Check that tables exist
        cursor = in_memory_db.connection.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        
        # Check expected tables
        assert 'users' in tables
        assert 'watchlists' in tables
        assert 'triggered_alerts' in tables

    def test_thread_safety(self):
        """Test database thread safety"""
        db = DatabaseManager(":memory:")
        db.create_tables()
        
        results = []
        
        def worker_thread():
            try:
                # Should create a new connection for this thread
                user_id = f"user_{threading.get_ident()}"
                db.create_user(user_id, f"User {user_id}")
                user = db.get_user(user_id)
                results.append((True, user_id))
            except sqlite3.Error as e:
                results.append((False, str(e)))
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker_thread)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check results
        for success, _ in results:
            assert success, "Thread should have completed successfully"
        
        # Clean up
        db.close()

    def test_user_management(self, in_memory_db):
        """Test user management methods"""
        # Create test user
        user_id = "test_user_123"
        result = in_memory_db.create_user(user_id, "Test User", "discord_123")
        assert result is True
        
        # Get user
        user = in_memory_db.get_user(user_id)
        assert user is not None
        assert user['user_id'] == user_id
        assert user['username'] == "Test User"
        assert user['discord_id'] == "discord_123"
        assert 'settings' in user
        
        # Default settings should be applied
        assert 'rsi_oversold' in user['settings']
        assert 'rsi_overbought' in user['settings']
        
        # Update settings
        new_settings = {
            'rsi_oversold': 25,
            'new_setting': 'test'
        }
        result = in_memory_db.update_user_settings(user_id, new_settings)
        assert result is True
        
        # Check updated settings
        user = in_memory_db.get_user(user_id)
        assert user['settings']['rsi_oversold'] == 25
        assert user['settings']['new_setting'] == 'test'
        assert user['settings']['rsi_overbought'] == 70  # Original value preserved

    def test_watchlist_management(self, in_memory_db):
        """Test watchlist management methods"""
        # Create test user
        user_id = "test_user_456"
        in_memory_db.create_user(user_id, "Watchlist Test User")
        
        # Add to watchlist
        result = in_memory_db.add_to_watchlist(user_id, "BTCUSDT", "15m")
        assert result is True
        
        result = in_memory_db.add_to_watchlist(user_id, "ETHUSDT", "1h")
        assert result is True
        
        # Get watchlist
        watchlist = in_memory_db.get_user_watchlist(user_id)
        assert len(watchlist) == 2
        
        symbols = sorted([item['symbol'] for item in watchlist])
        assert symbols == ["BTCUSDT", "ETHUSDT"]
        
        # Remove from watchlist
        result = in_memory_db.remove_from_watchlist(user_id, "BTCUSDT", "15m")
        assert result is True
        
        # Check watchlist after removal
        watchlist = in_memory_db.get_user_watchlist(user_id)
        assert len(watchlist) == 1
        assert watchlist[0]['symbol'] == "ETHUSDT"
        
        # Check inactive items
        watchlist_with_inactive = in_memory_db.get_user_watchlist(user_id, active_only=False)
        assert len(watchlist_with_inactive) == 2

    def test_alert_management(self, in_memory_db):
        """Test alert management methods"""
        # Create test user
        user_id = "test_user_789"
        in_memory_db.create_user(user_id, "Alert Test User")
        
        # Record some alerts
        timestamp = datetime.now()
        
        in_memory_db.record_alert(user_id, "BTCUSDT", "15m", "rsi", "RSI ALERT: BTCUSDT")
        in_memory_db.record_alert(user_id, "ETHUSDT", "1h", "macd", "MACD ALERT: ETHUSDT")
        
        # Get pending alerts
        pending = in_memory_db.get_pending_alerts(user_id)
        assert len(pending) == 2
        
        # Mark as notified
        alert_ids = [alert['id'] for alert in pending]
        result = in_memory_db.mark_alerts_notified(alert_ids)
        assert result is True
        
        # Check pending alerts again
        pending = in_memory_db.get_pending_alerts(user_id)
        assert len(pending) == 0
        
        # Get recent alerts
        recent = in_memory_db.get_recent_alerts(user_id, hours=24)
        assert len(recent) == 2

    def test_symbol_and_user_queries(self, in_memory_db):
        """Test symbol and user query methods"""
        # Create test users and watchlists
        in_memory_db.create_user("user1", "User One")
        in_memory_db.create_user("user2", "User Two")
        
        in_memory_db.add_to_watchlist("user1", "BTCUSDT", "15m")
        in_memory_db.add_to_watchlist("user1", "ETHUSDT", "15m")
        in_memory_db.add_to_watchlist("user2", "BTCUSDT", "15m")
        in_memory_db.add_to_watchlist("user2", "BNBUSDT", "1h")
        
        # Get all active symbols
        symbols = in_memory_db.get_all_active_symbols()
        expected_symbols = [
            ("BTCUSDT", "15m"),
            ("ETHUSDT", "15m"),
            ("BNBUSDT", "1h")
        ]
        
        # Check that all expected symbols are in the results
        for expected in expected_symbols:
            assert expected in symbols
            
        # The total count should be the number of unique symbol-interval pairs
        assert len(symbols) == len(expected_symbols)
        
        # Get users watching a symbol
        users = in_memory_db.get_users_watching_symbol("BTCUSDT", "15m")
        assert len(users) == 2
        assert "user1" in users
        assert "user2" in users
        
        users = in_memory_db.get_users_watching_symbol("ETHUSDT", "15m")
        assert len(users) == 1
        assert "user1" in users 