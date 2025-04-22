import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trading_alerts.db")

# Database constants
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alerts.db")
DEFAULT_ALERT_SETTINGS = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "volume_threshold": 2.0,
    "ema_short": 9,
    "ema_long": 21,
    "bb_squeeze_threshold": 0.05,
    "adx_threshold": 25,
    "cooldown_minutes": 240,
    "enabled_alerts": ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"],
}


class DatabaseManager:
    """Handles all database operations for the trading alerts system"""

    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection and ensure tables exist"""
        # Create data directory if it doesn't exist
        self.db_path = db_path

        # Only create directory if not using in-memory database
        if db_path != ":memory:" and not db_path.startswith("file:memdb"):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Use thread-local storage for connections
        self.local = threading.local()
        self.local.connection = None

        # Initial connection in the main thread
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish connection to the SQLite database in the current thread"""
        try:
            # Close existing connection if any
            if hasattr(self.local, "connection") and self.local.connection:
                self.local.connection.close()

            # Create a new connection for this thread
            self.local.connection = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.local.connection.execute("PRAGMA foreign_keys = ON")
            # Return rows as dictionaries
            self.local.connection.row_factory = sqlite3.Row
            logger.info(
                f"Connected to database at {self.db_path} in thread {threading.get_ident()}"
            )
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    @property
    def connection(self):
        """Get the connection for the current thread"""
        if not hasattr(self.local, "connection") or self.local.connection is None:
            self.connect()
        return self.local.connection

    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        try:
            cursor = self.connection.cursor()

            # Users table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                discord_id TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings TEXT DEFAULT '{}',
                is_active BOOLEAN DEFAULT 1
            )
            """
            )

            # Watchlists table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT DEFAULT '15m',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings TEXT DEFAULT '{}',
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(user_id, symbol, interval),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
            )

            # Triggered alerts table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS triggered_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                was_notified BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
            )

            # Alert channels table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS alert_channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, channel_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
            )

            self.connection.commit()
            logger.info("Database tables created/verified")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {e}")
            self.connection.rollback()
            raise

    def close(self):
        """Close all database connections"""
        try:
            # Close the connection in the current thread
            if hasattr(self.local, "connection") and self.local.connection:
                self.local.connection.close()
                self.local.connection = None
                logger.info(
                    f"Database connection closed in thread {threading.get_ident()}"
                )

            # In a production environment, there might be multiple thread-local connections
            # We can't directly access other thread's connections, but if the app is shutting down,
            # SQLite will close them automatically when the process ends

            # Log the closure attempt
            logger.info("Database connections closure requested")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    # User management methods
    def create_user(self, user_id: str, username: str, discord_id: str = None) -> bool:
        """
        Create a new user

        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        username : str
            Display name of the user
        discord_id : str, optional
            Discord user ID if applicable

        Returns:
        --------
        bool
            True if user was created successfully, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            settings = json.dumps(DEFAULT_ALERT_SETTINGS)

            # Check if user already exists (might need to update discord_id)
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            existing_user = cursor.fetchone()

            if existing_user:
                # Update existing user's discord_id if provided
                if discord_id is not None:
                    cursor.execute(
                        "UPDATE users SET discord_id = ? WHERE user_id = ?",
                        (discord_id, user_id),
                    )
            else:
                # Insert new user
                cursor.execute(
                    "INSERT INTO users (user_id, username, discord_id, settings) VALUES (?, ?, ?, ?)",
                    (user_id, username, discord_id, settings),
                )

            self.connection.commit()
            # Return True since either the user was created or updated successfully
            return True
        except sqlite3.Error as e:
            logger.error(f"Error creating user {username}: {e}")
            self.connection.rollback()
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if row:
                user_dict = dict(row)
                user_dict["settings"] = json.loads(user_dict["settings"])
                return user_dict
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            return None

    def update_user_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """Update user alert settings"""
        try:
            # Get current settings
            current_user = self.get_user(user_id)
            if not current_user:
                return False

            # Merge new settings with existing ones
            current_settings = current_user["settings"]
            merged_settings = {**current_settings, **settings}

            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE users SET settings = ? WHERE user_id = ?",
                (json.dumps(merged_settings), user_id),
            )

            self.connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating settings for user {user_id}: {e}")
            self.connection.rollback()
            return False

    # Watchlist management methods
    def add_to_watchlist(
        self, user_id: str, symbol: str, interval: str = "15m"
    ) -> bool:
        """Add a trading pair to user's watchlist"""
        try:
            cursor = self.connection.cursor()

            # Check if user exists, create if not
            if not self.get_user(user_id):
                self.create_user(user_id, f"User_{user_id[:8]}")

            cursor.execute(
                "INSERT OR IGNORE INTO watchlists (user_id, symbol, interval) VALUES (?, ?, ?)",
                (user_id, symbol.upper(), interval),
            )

            # If already exists but inactive, reactivate it
            if cursor.rowcount == 0:
                cursor.execute(
                    "UPDATE watchlists SET is_active = 1 WHERE user_id = ? AND symbol = ? AND interval = ?",
                    (user_id, symbol.upper(), interval),
                )

            self.connection.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding {symbol} to watchlist for user {user_id}: {e}")
            self.connection.rollback()
            return False

    def remove_from_watchlist(
        self, user_id: str, symbol: str, interval: str = "15m"
    ) -> bool:
        """Remove a trading pair from user's watchlist"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "UPDATE watchlists SET is_active = 0 WHERE user_id = ? AND symbol = ? AND interval = ?",
                (user_id, symbol.upper(), interval),
            )

            self.connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(
                f"Error removing {symbol} from watchlist for user {user_id}: {e}"
            )
            self.connection.rollback()
            return False

    def get_user_watchlist(
        self, user_id: str, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all trading pairs in a user's watchlist"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM watchlists WHERE user_id = ?"
            params = [user_id]

            if active_only:
                query += " AND is_active = 1"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            watchlist = []
            for row in rows:
                item = dict(row)
                item["settings"] = json.loads(item["settings"])
                watchlist.append(item)

            return watchlist
        except sqlite3.Error as e:
            logger.error(f"Error fetching watchlist for user {user_id}: {e}")
            return []

    def update_symbol_settings(
        self, user_id: str, symbol: str, interval: str, settings: Dict[str, Any]
    ) -> bool:
        """Update settings for a specific symbol in a user's watchlist"""
        try:
            cursor = self.connection.cursor()

            # Get current settings
            cursor.execute(
                "SELECT settings FROM watchlists WHERE user_id = ? AND symbol = ? AND interval = ? AND is_active = 1",
                (user_id, symbol.upper(), interval),
            )
            row = cursor.fetchone()

            if not row:
                return False

            # Merge new settings with existing ones
            current_settings = json.loads(row["settings"])
            merged_settings = {**current_settings, **settings}

            cursor.execute(
                "UPDATE watchlists SET settings = ? WHERE user_id = ? AND symbol = ? AND interval = ?",
                (json.dumps(merged_settings), user_id, symbol.upper(), interval),
            )

            self.connection.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(
                f"Error updating settings for {symbol} for user {user_id}: {e}"
            )
            self.connection.rollback()
            return False

    # Alert management methods
    def record_alert(
        self, user_id: str, symbol: str, interval: str, alert_type: str, message: str
    ) -> bool:
        """
        Record a triggered alert in the database

        Parameters:
        -----------
        user_id : str
            User ID the alert is for
        symbol : str
            Trading pair symbol
        interval : str
            Timeframe interval
        alert_type : str
            Type of alert (rsi, macd, etc.)
        message : str
            Alert message to display

        Returns:
        --------
        bool
            True if alert was recorded successfully
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO triggered_alerts (user_id, symbol, interval, alert_type, message) VALUES (?, ?, ?, ?, ?)",
                (user_id, symbol.upper(), interval, alert_type, message),
            )

            self.connection.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error recording alert for {symbol} for user {user_id}: {e}")
            self.connection.rollback()
            return False

    def get_pending_alerts(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get all pending alerts that haven't been notified yet"""
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM triggered_alerts WHERE was_notified = 0"
            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching pending alerts: {e}")
            return []

    def mark_alerts_notified(self, alert_ids: List[int]) -> bool:
        """Mark alerts as having been notified"""
        if not alert_ids:
            return True

        try:
            cursor = self.connection.cursor()
            placeholders = ", ".join(["?" for _ in alert_ids])
            cursor.execute(
                f"UPDATE triggered_alerts SET was_notified = 1 WHERE id IN ({placeholders})",
                alert_ids,
            )

            self.connection.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error marking alerts as notified: {e}")
            self.connection.rollback()
            return False

    def get_recent_alerts(
        self, user_id: str, symbol: str = None, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent alerts for a user within the specified time window"""
        try:
            cursor = self.connection.cursor()
            query = """
            SELECT * FROM triggered_alerts
            WHERE user_id = ? AND triggered_at > datetime('now', ?)
            """
            params = [user_id, f"-{hours} hours"]

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())

            query += " ORDER BY triggered_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching recent alerts for user {user_id}: {e}")
            return []

    # Stats and aggregation methods
    def get_all_active_symbols(self) -> List[Tuple[str, str]]:
        """Get all symbols being watched by any user"""
        try:
            cursor = self.connection.cursor()
            # Using DISTINCT to ensure unique symbol-interval pairs
            cursor.execute(
                "SELECT DISTINCT symbol, interval FROM watchlists WHERE is_active = 1"
            )
            rows = cursor.fetchall()

            # Convert to list of tuples
            symbol_pairs = [(row["symbol"], row["interval"]) for row in rows]

            # Use a set to remove any potential duplicates and convert back to list for expected return type
            unique_pairs = list(set(symbol_pairs))

            # For debugging
            logger.debug(f"All active symbols returned: {unique_pairs}")

            return unique_pairs
        except sqlite3.Error as e:
            logger.error(f"Error fetching active symbols: {e}")
            return []

    def get_users_watching_symbol(self, symbol: str, interval: str) -> List[str]:
        """Get all users watching a specific symbol"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT user_id FROM watchlists WHERE symbol = ? AND interval = ? AND is_active = 1",
                (symbol.upper(), interval),
            )
            rows = cursor.fetchall()

            return [row["user_id"] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching users watching {symbol}: {e}")
            return []

    def get_alert_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get statistics about alerts triggered in the past X days"""
        try:
            cursor = self.connection.cursor()

            # Total alerts by type
            cursor.execute(
                """
                SELECT alert_type, COUNT(*) as count
                FROM triggered_alerts
                WHERE triggered_at > datetime('now', ?)
                GROUP BY alert_type
                ORDER BY count DESC
                """,
                (f"-{days} days",),
            )
            alerts_by_type = {
                row["alert_type"]: row["count"] for row in cursor.fetchall()
            }

            # Most active symbols
            cursor.execute(
                """
                SELECT symbol, COUNT(*) as count
                FROM triggered_alerts
                WHERE triggered_at > datetime('now', ?)
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
                """,
                (f"-{days} days",),
            )
            top_symbols = {row["symbol"]: row["count"] for row in cursor.fetchall()}

            # Total alerts per day
            cursor.execute(
                """
                SELECT date(triggered_at) as day, COUNT(*) as count
                FROM triggered_alerts
                WHERE triggered_at > datetime('now', ?)
                GROUP BY day
                ORDER BY day
                """,
                (f"-{days} days",),
            )
            alerts_by_day = {row["day"]: row["count"] for row in cursor.fetchall()}

            return {
                "alerts_by_type": alerts_by_type,
                "top_symbols": top_symbols,
                "alerts_by_day": alerts_by_day,
                "total_alerts": sum(alerts_by_type.values()),
            }
        except sqlite3.Error as e:
            logger.error(f"Error fetching alert stats: {e}")
            return {
                "alerts_by_type": {},
                "top_symbols": {},
                "alerts_by_day": {},
                "total_alerts": 0,
            }

    def register_alert_channel(
        self, user_id: str, channel_id: str, guild_id: str
    ) -> bool:
        """Register a channel for a user's alerts in the database"""
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                "INSERT OR IGNORE INTO alert_channels (user_id, channel_id, guild_id) VALUES (?, ?, ?)",
                (user_id, channel_id, guild_id),
            )

            self.connection.commit()
            logger.info(
                f"Registered channel {channel_id} for user {user_id} in database"
            )
            return True
        except sqlite3.Error as e:
            logger.error(f"Error registering channel for user {user_id}: {e}")
            self.connection.rollback()
            return False

    def unregister_alert_channel(self, user_id: str, channel_id: str) -> bool:
        """Unregister a channel for a user's alerts"""
        try:
            cursor = self.connection.cursor()

            cursor.execute(
                "DELETE FROM alert_channels WHERE user_id = ? AND channel_id = ?",
                (user_id, channel_id),
            )

            self.connection.commit()
            logger.info(
                f"Unregistered channel {channel_id} for user {user_id} from database"
            )
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error unregistering channel for user {user_id}: {e}")
            self.connection.rollback()
            return False

    def get_user_alert_channels(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all registered alert channels for a user"""
        try:
            cursor = self.connection.cursor()

            cursor.execute("SELECT * FROM alert_channels WHERE user_id = ?", (user_id,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error getting alert channels for user {user_id}: {e}")
            return []


# Singleton instance
_db_instance = None


def get_db() -> DatabaseManager:
    """Get the singleton DatabaseManager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance


# Testing functionality
if __name__ == "__main__":
    # Test the database functionality
    db = get_db()

    def test_user_management():
        print("\n===== Testing User Management =====")
        # Create test user
        test_user_id = "test_user_123"
        success = db.create_user(test_user_id, "Test User", "discord_12345")
        print(f"Create user result: {success}")

        # Get user
        user = db.get_user(test_user_id)
        print(f"User: {user['username']}, Settings: {user['settings']}")

        # Update settings
        new_settings = {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
        }
        success = db.update_user_settings(test_user_id, new_settings)
        print(f"Update settings result: {success}")

        # Verify changes
        user = db.get_user(test_user_id)
        print(f"Updated settings: {user['settings']}")

    def test_watchlist_management():
        print("\n===== Testing Watchlist Management =====")
        test_user_id = "test_user_123"

        # Add symbols to watchlist
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        intervals = ["15m", "1h", "4h"]

        for symbol in symbols:
            for interval in intervals[:2]:  # Only add 15m and 1h for testing
                success = db.add_to_watchlist(test_user_id, symbol, interval)
                print(f"Add {symbol} ({interval}) to watchlist: {success}")

        # Get watchlist
        watchlist = db.get_user_watchlist(test_user_id)
        print(f"User watchlist: {len(watchlist)} items")
        for item in watchlist:
            print(f"  {item['symbol']} ({item['interval']})")

        # Remove a symbol
        success = db.remove_from_watchlist(test_user_id, "ETHUSDT", "15m")
        print(f"Remove ETHUSDT from watchlist: {success}")

        # Verify removal
        watchlist = db.get_user_watchlist(test_user_id)
        print(f"Updated watchlist: {len(watchlist)} items")
        for item in watchlist:
            print(f"  {item['symbol']} ({item['interval']})")

    def test_alert_management():
        print("\n===== Testing Alert Management =====")
        test_user_id = "test_user_123"

        # Record some alerts
        alerts = [
            ("BTCUSDT", "15m", "rsi", "🔴 RSI OVERSOLD: BTCUSDT RSI at 28.5"),
            ("ETHUSDT", "1h", "macd", "🟢 MACD BULLISH CROSS: ETHUSDT"),
            ("BNBUSDT", "15m", "bb", "🟡 BOLLINGER SQUEEZE: BNBUSDT"),
        ]

        for symbol, interval, alert_type, message in alerts:
            success = db.record_alert(
                test_user_id, symbol, interval, alert_type, message
            )
            print(f"Record alert for {symbol}: {success}")

        # Get pending alerts
        pending = db.get_pending_alerts(test_user_id)
        print(f"Pending alerts: {len(pending)}")
        for alert in pending:
            print(f"  {alert['symbol']} - {alert['alert_type']}: {alert['message']}")

        # Mark as notified
        if pending:
            alert_ids = [
                alert["id"] for alert in pending[:1]
            ]  # Mark only the first one
            success = db.mark_alerts_notified(alert_ids)
            print(f"Mark alerts as notified: {success}")

        # Get recent alerts
        recent = db.get_recent_alerts(test_user_id, hours=48)
        print(f"Recent alerts: {len(recent)}")

    def test_stats():
        print("\n===== Testing Stats and Aggregation =====")

        # Get all active symbols
        symbols = db.get_all_active_symbols()
        print(f"Active symbols: {symbols}")

        # Get users watching a symbol
        if symbols:
            symbol, interval = symbols[0]
            users = db.get_users_watching_symbol(symbol, interval)
            print(f"Users watching {symbol} ({interval}): {users}")

        # Get alert stats
        stats = db.get_alert_stats(days=30)
        print(f"Alert stats: {stats}")

    # Run the tests
    try:
        test_user_management()
        test_watchlist_management()
        test_alert_management()
        test_stats()
        print("\n===== All Tests Completed =====")
    finally:
        db.close()
