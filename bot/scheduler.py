import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import pandas as pd
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler

from bot.alerts import AlertManager

# Import our modules
from bot.binance import fetch_market_data
from bot.data_cache import get_cache
from bot.db import get_db

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trading_alerts.scheduler")

# Global variables
# These are now only used for data fetching limits, not scheduling frequency
DEFAULT_INTERVALS = {
    "1m": 60,  # Check every 60 seconds
    "3m": 180,  # Check every 3 minutes
    "5m": 300,  # Check every 5 minutes
    "15m": 900,  # Check every 15 minutes
    "30m": 1800,  # Check every 30 minutes
    "1h": 3600,  # Check every hour
    "2h": 7200,  # Check every 2 hours
    "4h": 14400,  # Check every 4 hours
    "6h": 21600,  # Check every 6 hours
    "8h": 28800,  # Check every 8 hours
    "12h": 43200,  # Check every 12 hours
    "1d": 86400,  # Check every day
}

# Decoupled frequency settings - all timeframes are checked at this interval
# Timeframes are grouped into frequency tiers to balance API usage
CHECK_FREQUENCY = {
    "high": 120,  # 2 minutes for short timeframes (1m, 3m, 5m)
    "medium": 300,  # 5 minutes for medium timeframes (15m, 30m, 1h)
    "low": 600,  # 10 minutes for longer timeframes (2h, 4h+)
}

# Map intervals to frequency tiers
INTERVAL_FREQUENCY_MAP = {
    "1m": "high",
    "3m": "high",
    "5m": "high",
    "15m": "medium",
    "30m": "medium",
    "1h": "medium",
    "2h": "low",
    "4h": "low",
    "6h": "low",
    "8h": "low",
    "12h": "low",
    "1d": "low",
}

# How many candles to fetch for each timeframe
DEFAULT_CANDLE_LIMITS = {
    "1m": 200,
    "3m": 200,
    "5m": 200,
    "15m": 200,
    "30m": 150,
    "1h": 150,
    "2h": 150,
    "4h": 150,
    "6h": 120,
    "8h": 120,
    "12h": 120,
    "1d": 100,
}


class AlertScheduler:
    """Scheduler for periodically checking trading alerts"""

    def __init__(
        self,
        alert_callback: Union[
            Callable[[str, List[str]], None],
            Callable[[str, List[str]], Coroutine[Any, Any, None]],
        ] = None,
    ):
        """
        Initialize the alert scheduler

        Parameters:
        -----------
        alert_callback : callable or coroutine
            Function or coroutine to call when alerts are triggered (user_id, list_of_alerts)
        """
        # Don't store a db reference, we'll get a fresh one in each method
        self.user_alert_managers: Dict[
            str, Dict[str, AlertManager]
        ] = {}  # user_id -> {symbol_interval -> AlertManager}
        self.scheduler = BackgroundScheduler()
        self.alert_callback = alert_callback
        self.running = False
        self.last_run: Dict[
            Tuple[str, str], datetime
        ] = {}  # (symbol, interval) -> last run time
        self.lock = threading.RLock()  # For thread safety
        self.scheduled_symbols = set()  # Track which symbols are scheduled
        self.next_check_time: Dict[
            Tuple[str, str], datetime
        ] = {}  # (symbol, interval) -> next check time
        self.symbols_by_frequency = {
            "high": set(),
            "medium": set(),
            "low": set(),
        }  # Organize symbols by frequency tier

        # Create event loop for running async callbacks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = None

    def initialize(self):
        """Initialize the scheduler and load watched symbols"""
        with self.lock:
            # Start event loop thread if alert_callback is async
            if self.alert_callback and asyncio.iscoroutinefunction(self.alert_callback):
                self._start_event_loop()

            # Get a fresh db connection for this method
            db = get_db()

            # Get all active symbols from the database
            symbols = db.get_all_active_symbols()

            if not symbols:
                logger.info("No symbols found in watchlists")
                return

            # Group symbols by frequency tier
            for symbol, interval in symbols:
                self.add_symbol(symbol, interval)

            # Schedule checks for each frequency tier
            for frequency_tier, check_seconds in CHECK_FREQUENCY.items():
                if self.symbols_by_frequency[frequency_tier]:
                    self.schedule_frequency_check(frequency_tier, check_seconds)

            # Start the scheduler
            if not self.scheduler.running:
                self.scheduler.start()
                self.running = True
                logger.info("Alert scheduler started with decoupled frequency checking")

    def _start_event_loop(self):
        """Start a background thread with an event loop for async callbacks"""
        if self.loop_thread is not None and self.loop_thread.is_alive():
            logger.debug("Event loop thread already running")
            return

        def run_event_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self.loop_thread.start()
        logger.info("Started event loop thread for async callbacks")

    def _run_callback(self, user_id: str, alerts: List[str]):
        """Run the callback, handling both synchronous and asynchronous callbacks"""
        if not self.alert_callback:
            return

        try:
            if asyncio.iscoroutinefunction(self.alert_callback):
                # Import the bot instance to get its event loop
                # This is important for Discord.py's timeouts to work correctly
                from bot.discord_bot import bot

                if bot and hasattr(bot, "loop") and bot.loop:
                    discord_loop = bot.loop
                    # Schedule the coroutine to run on Discord.py's event loop
                    asyncio.run_coroutine_threadsafe(
                        self.alert_callback(user_id, alerts), discord_loop
                    )
                    logger.debug(
                        f"Scheduled alert callback on Discord's event loop for user {user_id}"
                    )
                else:
                    # Fallback to our own loop if bot's loop is not available
                    if self.loop_thread is None or not self.loop_thread.is_alive():
                        logger.warning("Event loop thread not running, starting now")
                        self._start_event_loop()

                    # Schedule it to run in our event loop
                    asyncio.run_coroutine_threadsafe(
                        self.alert_callback(user_id, alerts), self.loop
                    )
                    logger.debug(
                        f"Scheduled alert callback on scheduler's loop for user {user_id}"
                    )
            else:
                # For synchronous callbacks, just call directly
                self.alert_callback(user_id, alerts)
        except Exception as e:
            logger.error(f"Error in alert callback: {e}")
            # Log the full traceback
            import traceback

            logger.error(traceback.format_exc())

    def get_user_alert_manager(
        self, user_id: str, symbol: str, interval: str
    ) -> AlertManager:
        """Get or create an alert manager for a specific user, symbol, and interval"""
        if user_id not in self.user_alert_managers:
            self.user_alert_managers[user_id] = {}

        key = f"{symbol.upper()}_{interval}"
        if key not in self.user_alert_managers[user_id]:
            self.user_alert_managers[user_id][key] = AlertManager()

        return self.user_alert_managers[user_id][key]

    def schedule_frequency_check(self, frequency_tier: str, seconds: int):
        """Schedule periodic checks for a frequency tier"""
        # Skip if no symbols in this tier
        if not self.symbols_by_frequency[frequency_tier]:
            return

        # Add some jitter to avoid all checks happening at once
        jitter = int(seconds * 0.1)  # 10% jitter
        if jitter > 0:
            seconds += int(time.time()) % jitter

        # Create job ID
        job_id = f"check_frequency_{frequency_tier}"

        # Schedule the job
        try:
            # Remove existing job if present
            try:
                self.scheduler.remove_job(job_id)
            except JobLookupError:
                pass

            # Schedule new job
            self.scheduler.add_job(
                self.check_frequency_tier,
                "interval",
                seconds=seconds,
                id=job_id,
                replace_existing=True,
                args=[frequency_tier],
            )

            logger.info(
                f"Scheduled {frequency_tier} frequency tier check every {seconds} seconds for {len(self.symbols_by_frequency[frequency_tier])} symbol/interval pairs"
            )
        except Exception as e:
            logger.error(f"Error scheduling {frequency_tier} frequency tier check: {e}")

    def check_frequency_tier(self, frequency_tier: str):
        """
        Check all symbols in a frequency tier

        This is called periodically by the scheduler for each frequency tier
        """
        logger.info(f"Running {frequency_tier} frequency tier check")

        # Make a copy to avoid modification during iteration
        with self.lock:
            symbols_to_check = list(self.symbols_by_frequency[frequency_tier])

        # Check each symbol/interval in this frequency tier
        for symbol, interval in symbols_to_check:
            try:
                self.check_symbol_alerts(symbol, interval)
            except Exception as e:
                logger.error(f"Error checking {symbol} ({interval}): {e}")
                import traceback

                logger.error(traceback.format_exc())

    def check_symbol_alerts(self, symbol: str, interval: str):
        """
        Check alerts for a specific symbol and interval

        This is called for each symbol/interval pair by the frequency tier checker
        """
        # We need a fresh database connection for each thread
        db = get_db()  # This will get a new connection for this thread

        with self.lock:
            # Update last run time
            self.last_run[(symbol, interval)] = datetime.now()

            # Set the next check time (used for status reporting)
            frequency_tier = INTERVAL_FREQUENCY_MAP.get(interval, "medium")
            seconds = CHECK_FREQUENCY.get(frequency_tier, 300)
            self.next_check_time[(symbol, interval)] = datetime.now() + timedelta(
                seconds=seconds
            )

            logger.info(f"Checking alerts for {symbol} ({interval})")

            try:
                # Fetch market data, using cache when possible
                limit = DEFAULT_CANDLE_LIMITS.get(interval, 200)

                # Determine if we should force refresh based on the interval
                # For shorter timeframes, we need fresher data
                force_refresh = False
                interval_seconds = DEFAULT_INTERVALS.get(interval, 900)
                cache = get_cache()
                cached_df = cache.get(symbol, interval)

                # If we're close to the end of the current candle, force refresh
                if cached_df is not None and not cached_df.empty:
                    latest_time = cached_df.index[-1]
                    now = datetime.now()
                    # Use pandas to handle timezones properly
                    time_diff = (
                        now - pd.Timestamp(latest_time).to_pydatetime()
                    ).total_seconds()

                    # If we're in the last 20% of the candle duration, force refresh
                    if time_diff > interval_seconds * 0.8:
                        force_refresh = True
                        logger.debug(
                            f"Forcing refresh for {symbol} ({interval}) as current candle is nearly complete"
                        )

                df = fetch_market_data(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    force_refresh=force_refresh,
                )

                if df is None or df.empty:
                    logger.warning(f"Failed to fetch data for {symbol} ({interval})")
                    return

                # Get users watching this symbol
                users = db.get_users_watching_symbol(symbol, interval)
                if not users:
                    logger.info(f"No users watching {symbol} ({interval})")
                    return

                # Process alerts for each user independently
                for user_id in users:
                    # Get user settings
                    user = db.get_user(user_id)
                    if not user or not user.get("is_active", False):
                        continue

                    # Get the alert manager for this user/symbol/interval
                    manager = self.get_user_alert_manager(user_id, symbol, interval)

                    # Set up alerts based on user preferences
                    self.setup_user_alerts(
                        manager, user_id, symbol, interval, user["settings"]
                    )

                    # Check for triggered alerts
                    alerts = manager.check_alerts(symbol, df, interval)

                    if alerts:
                        logger.info(
                            f"Found {len(alerts)} alerts for {user_id} on {symbol} ({interval})"
                        )

                        # Modify alerts to include interval information
                        modified_alerts = []
                        for alert in alerts:
                            # Add interval to be used in the title only, not in the body message
                            # Format: original_alert | interval (will be parsed by discord_bot.py)
                            modified_alert = f"{alert} | {interval}"
                            modified_alerts.append(modified_alert)

                        # Record alerts in the database
                        for alert in alerts:
                            alert_type = self._extract_alert_type(alert)
                            db.record_alert(
                                user_id, symbol, interval, alert_type, alert
                            )

                        # Send modified alerts with interval included
                        if modified_alerts:
                            self._run_callback(user_id, modified_alerts)

            except Exception as e:
                logger.error(f"Error checking alerts for {symbol} ({interval}): {e}")
                import traceback

                logger.error(traceback.format_exc())

    def setup_user_alerts(
        self,
        manager: AlertManager,
        user_id: str,
        symbol: str,
        interval: str,
        settings: Dict[str, Any],
    ):
        """Set up alerts based on user preferences"""
        from bot.alerts import (
            AdxAlert,
            BollingerBandAlert,
            EmaCrossAlert,
            MacdAlert,
            PatternAlert,
            RsiAlert,
            VolumeSpikeAlert,
        )

        # Clear existing alerts for this user/symbol
        manager.clear_alerts(symbol)

        # Get enabled alert types
        enabled_alerts = settings.get("enabled_alerts", [])
        cooldown = settings.get("cooldown_minutes", 240)

        # Add alerts based on user settings
        if "rsi" in enabled_alerts:
            manager.add_alert(
                RsiAlert(
                    symbol,
                    oversold=settings.get("rsi_oversold", 30),
                    overbought=settings.get("rsi_overbought", 70),
                    cooldown_minutes=cooldown,
                )
            )

        if "macd" in enabled_alerts:
            manager.add_alert(MacdAlert(symbol, cooldown_minutes=cooldown))

        if "ema" in enabled_alerts:
            manager.add_alert(
                EmaCrossAlert(
                    symbol,
                    short=settings.get("ema_short", 9),
                    long=settings.get("ema_long", 21),
                    cooldown_minutes=cooldown,
                )
            )

        if "bb" in enabled_alerts:
            manager.add_alert(
                BollingerBandAlert(
                    symbol,
                    squeeze_threshold=settings.get("bb_squeeze_threshold", 0.05),
                    cooldown_minutes=cooldown,
                )
            )

        if "volume" in enabled_alerts:
            manager.add_alert(
                VolumeSpikeAlert(
                    symbol,
                    threshold=settings.get("volume_threshold", 2.5),
                    cooldown_minutes=cooldown,
                )
            )

        if "adx" in enabled_alerts:
            manager.add_alert(
                AdxAlert(
                    symbol,
                    threshold=settings.get("adx_threshold", 25),
                    cooldown_minutes=cooldown,
                )
            )

        if "pattern" in enabled_alerts:
            manager.add_alert(PatternAlert(symbol, cooldown_minutes=cooldown))

    def _extract_alert_type(self, alert_message: str) -> str:
        """Extract the alert type from an alert message"""
        if "RSI" in alert_message:
            return "rsi"
        elif "MACD" in alert_message:
            return "macd"
        elif "EMA" in alert_message:
            return "ema"
        elif "BOLLINGER" in alert_message or "BB " in alert_message:
            return "bb"
        elif "VOLUME" in alert_message:
            return "volume"
        elif "ADX" in alert_message or "TREND" in alert_message:
            return "adx"
        elif any(
            pattern in alert_message
            for pattern in ["HAMMER", "STAR", "ENGULFING", "PATTERN"]
        ):
            return "pattern"
        else:
            return "other"

    def add_symbol(self, symbol: str, interval: str):
        """Add a new symbol to be monitored"""
        with self.lock:
            # Add to scheduled symbols set
            symbol_key = f"{symbol.upper()}_{interval}"
            self.scheduled_symbols.add(symbol_key)

            # Determine frequency tier for this interval
            frequency_tier = INTERVAL_FREQUENCY_MAP.get(interval, "medium")

            # Add to appropriate frequency tier
            self.symbols_by_frequency[frequency_tier].add((symbol, interval))

            # Initialize a 'next check time' for status reporting
            seconds = CHECK_FREQUENCY.get(frequency_tier, 300)
            self.next_check_time[(symbol, interval)] = datetime.now() + timedelta(
                seconds=seconds
            )

            logger.info(
                f"Added {symbol} ({interval}) to {frequency_tier} frequency tier"
            )

            # Call schedule_symbol_check to schedule individual checks if needed
            self.schedule_symbol_check(symbol, interval)

    def schedule_symbol_check(self, symbol: str, interval: str):
        """Schedule a check for a specific symbol and interval

        This is a helper method primarily used for testing and can be
        extended for more fine-grained scheduling in the future.
        """
        # Currently, symbols are checked via frequency tiers, not individually
        # This method is here for testing and future extension
        logger.debug(f"Symbol check scheduled for {symbol} ({interval})")
        pass

    def remove_symbol(self, symbol: str, interval: str):
        """Remove a symbol from monitoring"""
        with self.lock:
            symbol_key = f"{symbol.upper()}_{interval}"
            if symbol_key in self.scheduled_symbols:
                self.scheduled_symbols.remove(symbol_key)

                # Remove from frequency tiers
                for tier in self.symbols_by_frequency:
                    if (symbol, interval) in self.symbols_by_frequency[tier]:
                        self.symbols_by_frequency[tier].remove((symbol, interval))
                        logger.info(
                            f"Removed {symbol} ({interval}) from {tier} frequency tier"
                        )

                # Remove next check time
                if (symbol, interval) in self.next_check_time:
                    del self.next_check_time[(symbol, interval)]

                # Remove last run time
                if (symbol, interval) in self.last_run:
                    del self.last_run[(symbol, interval)]

    def check_status(self) -> Dict[str, Any]:
        """Get status information about the scheduler"""
        with self.lock:
            now = datetime.now()

            # Count symbols by frequency tier
            symbols_count = {
                tier: len(symbols)
                for tier, symbols in self.symbols_by_frequency.items()
            }

            # Get next check times
            next_checks = []
            for (symbol, interval), next_time in self.next_check_time.items():
                time_left = (next_time - now).total_seconds()
                if time_left > 0:  # Only include future checks
                    next_checks.append(
                        {
                            "symbol": symbol,
                            "interval": interval,
                            "next_check": next_time.strftime("%H:%M:%S"),
                            "seconds_left": int(time_left),
                        }
                    )

            # Sort by time left
            next_checks.sort(key=lambda x: x["seconds_left"])

            # Get cache stats
            cache = get_cache()
            cache_stats = cache.get_stats()

            return {
                "running": self.running,
                "total_symbols": len(self.scheduled_symbols),
                "symbols_by_frequency": symbols_count,
                "upcoming_checks": next_checks[:10],  # Show only next 10 checks
                "cache_stats": cache_stats,
            }

    def stop(self):
        """Stop the scheduler and clean up resources"""
        with self.lock:
            if self.running:
                try:
                    # Remove all jobs first
                    for job in self.scheduler.get_jobs():
                        try:
                            self.scheduler.remove_job(job.id)
                            logger.info(f"Removed scheduled job: {job.id}")
                        except JobLookupError:
                            continue

                    # Shutdown the scheduler
                    self.scheduler.shutdown(wait=False)

                    # Stop the event loop if it's running
                    if self.loop and self.loop.is_running():
                        logger.info("Stopping event loop...")
                        # Schedule a callback to stop the loop
                        self.loop.call_soon_threadsafe(self.loop.stop)

                        # Wait for the thread to finish if it exists
                        if self.loop_thread and self.loop_thread.is_alive():
                            self.loop_thread.join(timeout=2.0)
                            if self.loop_thread.is_alive():
                                logger.warning(
                                    "Event loop thread did not terminate in time"
                                )

                    # Clear internal data structures
                    self.user_alert_managers.clear()
                    self.scheduled_symbols.clear()
                    self.last_run.clear()
                    self.next_check_time.clear()
                    self.symbols_by_frequency.clear()

                    self.running = False
                    logger.info("Alert scheduler stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping scheduler: {e}")
                    # Still mark as not running
                    self.running = False


# Singleton instance
_scheduler_instance = None


def get_scheduler(alert_callback: Callable = None) -> AlertScheduler:
    """Get the singleton AlertScheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AlertScheduler(alert_callback)
    elif alert_callback and not _scheduler_instance.alert_callback:
        _scheduler_instance.alert_callback = alert_callback
    return _scheduler_instance


# Test functionality
async def test_scheduler():
    """Test the scheduler functionality"""
    from bot.db import DatabaseManager

    # Create a test database in memory
    # Need to pass special parameters to avoid directory creation for in-memory DB
    test_db = DatabaseManager(":memory:")

    # Skip directory creation for in-memory database
    test_db.db_path = ":memory:"
    test_db.connect()
    test_db.create_tables()

    # Set up a user and watchlist
    test_user_id = "test_user_123"
    test_db.create_user(test_user_id, "Test User")
    test_db.add_to_watchlist(test_user_id, "BTCUSDT", "15m")
    test_db.add_to_watchlist(test_user_id, "ETHUSDT", "1h")

    # Define a callback for alerts
    def handle_alert(user_id, alerts):
        print(f"\n===== ALERTS FOR {user_id} =====")
        for alert in alerts:
            print(f"  {alert}")

    # Create the scheduler
    scheduler = AlertScheduler(handle_alert)

    # Initialize
    scheduler.initialize()

    # Force a check
    print("\nRunning immediate check...")
    scheduler.check_symbol_alerts("BTCUSDT", "15m")
    scheduler.check_symbol_alerts("ETHUSDT", "1h")

    # Get status
    status = scheduler.check_status()
    print(f"\nScheduler status: {status}")

    # Run the scheduler for a while
    print("\nRunning scheduler for 10 seconds...")
    await asyncio.sleep(10)

    # Stop
    scheduler.stop()
    test_db.close()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_scheduler())
