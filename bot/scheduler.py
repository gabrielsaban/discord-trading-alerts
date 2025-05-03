import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

import pandas as pd
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.background import BackgroundScheduler

from bot.alerts import AlertManager

# Import our modules
from bot.binance import fetch_market_data
from bot.data_cache import get_cache
from bot.db import get_db
from bot.indicators import calculate_atr

# Setup logging
logger = logging.getLogger("discord_trading_alerts.scheduler")
logger.setLevel(logging.DEBUG)

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
    "3d": 259200,  # Check every 3 days
    "1w": 604800,  # Check every week
}

# Time between each alert check based on frequency tier
CHECK_FREQUENCY = {
    "high": 120,     # High frequency: check every 2 minutes (was 60)
    "medium": 600,   # Medium frequency: check every 10 minutes (was 300)
    "low": 1800,     # Low frequency: check every 30 minutes (was 900)
}

# Map intervals to frequency tiers
INTERVAL_FREQUENCY_MAP = {
    # High frequency
    "1m": "high",
    "3m": "high",
    "5m": "high",
    # Medium frequency
    "15m": "medium",
    "30m": "medium",
    "1h": "medium",
    "2h": "medium",
    # Low frequency
    "4h": "low",     # Moved from medium to low frequency
    "6h": "low",
    "8h": "low",
    "12h": "low",
    "1d": "low",
    "3d": "low",
    "1w": "low",
}

# How many candles to fetch for each timeframe
DEFAULT_CANDLE_LIMITS = {
    "1m": 500,
    "3m": 500,
    "5m": 500,
    "15m": 400,
    "30m": 300,
    "1h": 300,
    "2h": 200,
    "4h": 200,
    "6h": 150,
    "8h": 150,
    "12h": 150,
    "1d": 100,
    "3d": 100,
    "1w": 100,
}

# How often to send batched alerts (in seconds)
BATCH_SEND_INTERVAL = 300  # 5 minutes

from bot.services.batch_aggregator import (
    check_batch_aggregator_callback,
    get_batch_aggregator,
)
from bot.services.cooldown_service import get_cooldown_service

# Import required services
from bot.services.feature_flags import get_flag
from bot.services.override_engine import get_override_engine


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
        
        # Configure scheduler with proper executors and job defaults
        self.scheduler = BackgroundScheduler(
            job_defaults={
                'coalesce': False,  # Run all missed jobs
                'max_instances': 3,  # Allow multiple instances to run concurrently if needed
                'misfire_grace_time': 60  # Allow jobs up to 60 seconds late
            },
            executors={
                'default': {'type': 'threadpool', 'max_workers': 10}  # Increase worker threads
            }
        )
        
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

        # We'll prefer to use Discord's event loop rather than creating our own
        self.discord_loop = None
        self.use_own_loop = False
        self.loop = None
        self.loop_thread = None

    def initialize(self):
        """Initialize the scheduler and load watched symbols"""
        with self.lock:
            # Prefer to use Discord's event loop if available
            if self.alert_callback and asyncio.iscoroutinefunction(self.alert_callback):
                try:
                    # Try to get Discord's event loop
                    from bot.discord_bot import bot

                    if bot and hasattr(bot, "loop") and bot.loop:
                        self.discord_loop = bot.loop
                        logger.info(
                            "Using Discord bot's event loop for async callbacks"
                        )
                    else:
                        # Fall back to creating our own loop if necessary
                        self.use_own_loop = True
                        self._start_event_loop()
                except (ImportError, AttributeError):
                    # Fall back to creating our own loop if necessary
                    self.use_own_loop = True
                    self._start_event_loop()

            # Initialize CooldownService
            try:
                # Import and initialize CooldownService
                cooldown_service = get_cooldown_service()
                logger.info("Initialized CooldownService")

                # Schedule cooldown pruning task
                self.scheduler.add_job(
                    self.prune_expired_cooldowns,
                    "interval",
                    hours=6,  # Run every 6 hours
                    id="cooldown_pruning",
                    replace_existing=True,
                )
                logger.info("Scheduled cooldown pruning every 6 hours")
            except Exception as e:
                logger.error(f"Error initializing CooldownService: {e}")
                raise ImportError(
                    "CooldownService is required but failed to initialize"
                )

            # Initialize OverrideEngine
            try:
                # Import and initialize OverrideEngine
                override_engine = get_override_engine()
                logger.info("Initialized OverrideEngine")
            except Exception as e:
                logger.error(f"Error initializing OverrideEngine: {e}")
                raise ImportError("OverrideEngine is required but failed to initialize")

            # Initialize BatchAggregator if enabled by feature flag
            if get_flag("ENABLE_BATCH_AGGREGATOR", True):  # Default to True
                try:
                    # Import with a local import to avoid circular dependencies
                    from bot.services.batch_aggregator import (
                        check_batch_aggregator_callback,
                        get_batch_aggregator,
                    )

                    # Set the same callback for batch aggregator
                    batch_aggregator = get_batch_aggregator()
                    batch_aggregator.set_callback(self._run_callback)
                    logger.info("Initialized BatchAggregator service")

                    # Verify callback is actually set
                    if check_batch_aggregator_callback():
                        logger.info("BatchAggregator callback verified")
                    else:
                        logger.error(
                            "BatchAggregator callback verification failed - fixing..."
                        )
                        # Try setting callback again with a slight delay to avoid race conditions
                        time.sleep(0.5)
                        batch_aggregator.set_callback(self._run_callback)
                        check_batch_aggregator_callback()
                except Exception as e:
                    logger.error(f"Error initializing BatchAggregator: {e}")
                    logger.warning(
                        "Continuing without BatchAggregator - some functionality may be limited"
                    )

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
                    logger.info(
                        f"Scheduled {frequency_tier} frequency checks every {check_seconds} seconds"
                    )

            # Schedule batch processing task - only if feature flag is enabled
            if get_flag("ENABLE_BATCH_AGGREGATOR", False):
                self.scheduler.add_job(
                    self.process_all_batched_alerts,
                    "interval",
                    seconds=BATCH_SEND_INTERVAL,
                    id="batch_processor",
                    replace_existing=True,
                )
                logger.info(
                    f"Scheduled batch alert processing every {BATCH_SEND_INTERVAL} seconds"
                )
            else:
                logger.info("Batch alert processing not scheduled (batch aggregation is disabled)")

            # Start the scheduler
            if not self.scheduler.running:
                self.scheduler.start()
                self.running = True
                logger.info("Alert scheduler started with decoupled frequency checking")

    def _start_event_loop(self):
        """Start a background thread with an event loop for async callbacks - only used as fallback"""
        if self.loop_thread is not None and self.loop_thread.is_alive():
            logger.debug("Event loop thread already running")
            return

        logger.info("Starting fallback event loop thread for async callbacks")
        self.loop = asyncio.new_event_loop()

        def run_event_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_callback(self, user_id: str, alerts: List[str]):
        """Run the callback, handling both synchronous and asynchronous callbacks"""
        if not self.alert_callback:
            return

        try:
            if asyncio.iscoroutinefunction(self.alert_callback):
                # Use Discord's event loop if available (preferred approach)
                if self.discord_loop:
                    asyncio.run_coroutine_threadsafe(
                        self.alert_callback(user_id, alerts), self.discord_loop
                    )
                    logger.debug(
                        f"Scheduled alert callback on Discord's event loop for user {user_id}"
                    )
                # Fall back to our own loop if needed
                elif self.use_own_loop and self.loop:
                    # Check if our loop is still running
                    if self.loop_thread is None or not self.loop_thread.is_alive():
                        logger.warning("Event loop thread not running, starting now")
                        self._start_event_loop()

                    # Schedule on our own loop
                    asyncio.run_coroutine_threadsafe(
                        self.alert_callback(user_id, alerts), self.loop
                    )
                    logger.debug(
                        f"Scheduled alert callback on scheduler's fallback loop for user {user_id}"
                    )
                else:
                    # Last resort - use create_task, which will only work if we're already in an event loop
                    logger.warning(
                        "No event loop available, attempting create_task (may fail if not in event loop context)"
                    )
                    asyncio.create_task(self.alert_callback(user_id, alerts))
            else:
                # For synchronous callbacks, just call directly
                self.alert_callback(user_id, alerts)
        except Exception as e:
            logger.error(f"Error in alert callback: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def get_user_alert_manager(
        self, user_id: str, symbol: str, interval: str
    ) -> AlertManager:
        """Get or create an AlertManager for a user, symbol, and interval"""
        with self.lock:
            # Create user dict if it doesn't exist
            if user_id not in self.user_alert_managers:
                self.user_alert_managers[user_id] = {}

            # Create symbol-interval key
            symbol_interval = f"{symbol}_{interval}"

            # Create manager if it doesn't exist
            if symbol_interval not in self.user_alert_managers[user_id]:
                manager = AlertManager()
                manager.current_interval = interval
                manager.current_user_id = (
                    user_id  # Set the user ID for batch aggregator integration
                )
                self.user_alert_managers[user_id][symbol_interval] = manager
            else:
                manager = self.user_alert_managers[user_id][symbol_interval]

            return manager

    def schedule_frequency_check(self, frequency_tier: str, seconds: int):
        """Schedule periodic checks for a frequency tier"""
        # Skip if no symbols in this tier
        if not self.symbols_by_frequency[frequency_tier]:
            return

        # Add some jitter to avoid all checks happening at once
        jitter = int(seconds * 0.1)  # 10% jitter
        if jitter > 0:
            import random

            seconds += random.randint(0, jitter)

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
        import time
        import sys
        import traceback
        
        try:
            # Use direct print statements to absolutely ensure output
            print(f"\n\n!!!!! EXECUTING CHECK: {frequency_tier} frequency tier at {datetime.now()} !!!!!\n\n")
            sys.stdout.flush()  # Force flush to ensure it's written immediately
            
            # Also log at various levels
            logger.critical(f"CRITICAL: Starting {frequency_tier} frequency tier check")
            logger.error(f"ERROR: Starting {frequency_tier} frequency tier check")  # Log at ERROR too for visibility
            
            start_time = time.time()
            logger.info(f"Starting {frequency_tier} frequency tier check")

            # Make a copy to avoid modification during iteration
            with self.lock:
                symbols_to_check = list(self.symbols_by_frequency[frequency_tier])
                
            logger.info(f"Checking {len(symbols_to_check)} symbols in {frequency_tier} tier: {symbols_to_check}")
            
            # Print directly to stdout for absolute certainty
            print(f"Symbols to check in {frequency_tier} tier: {symbols_to_check}")
            sys.stdout.flush()

            # Check each symbol/interval in this frequency tier
            for idx, (symbol, interval) in enumerate(symbols_to_check):
                try:
                    symbol_start = time.time()
                    logger.info(f"[{idx+1}/{len(symbols_to_check)}] Checking alerts for {symbol} ({interval})")
                    print(f"Checking symbol {idx+1}/{len(symbols_to_check)}: {symbol} ({interval})")
                    sys.stdout.flush()
                    
                    self.check_symbol_alerts(symbol, interval)
                    symbol_duration = time.time() - symbol_start
                    logger.info(f"Finished check for {symbol} ({interval}) in {symbol_duration:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error checking {symbol} ({interval}): {e}")
                    print(f"ERROR checking {symbol} ({interval}): {e}")
                    sys.stdout.flush()
                    logger.error(traceback.format_exc())
            
            total_duration = time.time() - start_time
            logger.info(f"Completed {frequency_tier} frequency tier check in {total_duration:.2f} seconds")
            print(f"Completed {frequency_tier} frequency tier check in {total_duration:.2f} seconds")
            sys.stdout.flush()
        except Exception as e:
            # Catch any exception that might occur in the check_frequency_tier method
            error_msg = f"CRITICAL ERROR in frequency tier check for {frequency_tier}: {e}"
            logger.critical(error_msg)
            print(f"\n\n!!!!! {error_msg} !!!!!\n\n")
            print(traceback.format_exc())
            sys.stdout.flush()

    def check_symbol_alerts(self, symbol: str, interval: str):
        """
        Check alerts for a specific symbol and interval

        This is called for each symbol/interval pair by the frequency tier checker
        """
        import time
        check_start_time = time.time()
        logger.info(f"Starting alert check for {symbol} ({interval})")
        
        # We need a fresh database connection for each thread
        db = get_db()  # This will get a new connection for this thread
        # Get the cooldown service for ATR reference intervals
        cooldown_service = get_cooldown_service()

        with self.lock:
            # Update last run time
            self.last_run[(symbol, interval)] = datetime.now() + timedelta(hours=1)

            # Set the next check time (used for status reporting)
            frequency_tier = INTERVAL_FREQUENCY_MAP.get(interval, "medium")
            seconds = CHECK_FREQUENCY.get(frequency_tier, 300)
            self.next_check_time[(symbol, interval)] = datetime.now() + timedelta(
                seconds=seconds, hours=1
            )

            logger.debug(f"Checking alerts for {symbol} ({interval})")

            try:
                # Fetch market data, using cache when possible
                fetch_start = time.time()
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
                    now = datetime.utcnow() + timedelta(hours=1)
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

                logger.debug(f"Fetching market data for {symbol} ({interval}), force_refresh={force_refresh}")
                df = fetch_market_data(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    force_refresh=force_refresh,
                )
                fetch_time = time.time() - fetch_start
                logger.info(f"Fetched market data for {symbol} ({interval}) in {fetch_time:.2f} seconds, got {len(df) if df is not None else 0} candles")

                if df is None or df.empty:
                    logger.warning(f"Failed to fetch data for {symbol} ({interval})")
                    return

                # Calculate ATR for dynamic cooldown adjustments if needed
                market_data = None  # Default to None

                # Check if this interval has a reference interval for ATR-based cooldowns
                atr_start = time.time()
                ref_interval = cooldown_service.atr_reference_intervals.get(interval)
                if ref_interval:
                    try:
                        # Fetch data for the reference interval
                        logger.debug(f"Fetching reference interval data for {symbol} ({ref_interval})")
                        ref_df = fetch_market_data(
                            symbol=symbol,
                            interval=ref_interval,
                            limit=DEFAULT_CANDLE_LIMITS.get(ref_interval, 200),
                            force_refresh=force_refresh,
                        )
                        
                        if ref_df is not None and not ref_df.empty:
                            # Calculate ATR directly
                            logger.debug(f"Calculating ATR for {symbol} ({ref_interval})")
                            atr_data = calculate_atr(ref_df, length=14, calculate_percentiles=True)
                            
                            if atr_data is not None:
                                # Extract just the needed values
                                market_data = {
                                    "ATR": atr_data["ATR"].iloc[-1],
                                    "ATR_Percent": atr_data["ATR_Percent"].iloc[-1],
                                    "ATR_Percentile": atr_data["ATR_Percentile"].iloc[-1],
                                }
                                logger.debug(f"ATR data for {symbol}: {market_data}")
                    except Exception as e:
                        logger.error(f"Error calculating ATR for {symbol} ({ref_interval}): {e}")
                atr_time = time.time() - atr_start
                logger.debug(f"ATR calculation for {symbol} took {atr_time:.2f} seconds")

                # Get users watching this symbol
                users_start = time.time()
                users = db.get_users_watching_symbol(symbol, interval)
                if not users:
                    logger.debug(f"No users watching {symbol} ({interval})")
                    return
                users_time = time.time() - users_start
                logger.debug(f"Found {len(users)} users watching {symbol} ({interval}) in {users_time:.2f} seconds")

                # Process alerts for each user independently
                for user_idx, user_id in enumerate(users):
                    user_start = time.time()
                    logger.debug(f"Processing user {user_idx+1}/{len(users)}: {user_id}")
                    
                    # Get user settings
                    user = db.get_user(user_id)
                    if not user or not user.get("is_active", False):
                        continue

                    # Get the alert manager for this user/symbol/interval
                    setup_start = time.time()
                    manager = self.get_user_alert_manager(user_id, symbol, interval)

                    # Set up alerts based on user preferences
                    self.setup_user_alerts(
                        manager, user_id, symbol, interval, user["settings"]
                    )
                    setup_time = time.time() - setup_start
                    logger.debug(f"Set up alerts for {user_id} in {setup_time:.2f} seconds")

                    # Check for triggered alerts with ATR data for cooldown adjustment
                    checking_start = time.time()
                    # Handle both old and new function signatures
                    try:
                        logger.debug(f"Checking alerts for {user_id} on {symbol} ({interval})")
                        alerts = manager.check_alerts(symbol, df, interval, market_data)
                    except TypeError:
                        # Fall back to old function signature (for tests)
                        logger.debug(
                            f"Falling back to legacy check_alerts signature for {symbol}"
                        )
                        alerts = manager.check_alerts(symbol, df)
                    checking_time = time.time() - checking_start
                    logger.debug(f"Alert check for {user_id} on {symbol} ({interval}) took {checking_time:.2f} seconds")

                    if alerts:
                        logger.info(
                            f"Found {len(alerts)} alerts for {user_id} on {symbol} ({interval})"
                        )

                        # Initialize notifications list
                        notifications = []

                        # Format message with alert interval for processing
                        for i, alert_msg in enumerate(alerts):
                            # Format: alert_message | interval | symbol | alert_type
                            formatted_alert = f"{alert_msg} | {interval} | {symbol}"
                            
                            # If the alert object has an alert_type attribute, include it in the message
                            # Use the same index as the alert message to get the correct alert type
                            if i < len(manager.alerts[symbol]) and hasattr(manager.alerts[symbol][i], 'alert_type'):
                                alert_type = manager.alerts[symbol][i].alert_type
                                formatted_alert = f"{alert_msg} | {interval} | {symbol} | {alert_type}"
                                
                            notifications.append(formatted_alert)

                        # Record alerts in the database
                        for alert in alerts:
                            alert_type = self._extract_alert_type(alert)
                            db.record_alert(
                                user_id, symbol, interval, alert_type, alert
                            )

                        # Send modified alerts with interval included
                        if notifications:
                            self._run_callback(user_id, notifications)

                    # Check for batched alerts ready to send
                    # Only process batches if BatchAggregator is enabled
                    if get_flag("ENABLE_BATCH_AGGREGATOR", True):
                        now = datetime.utcnow() + timedelta(hours=1)
                        last_batch_time = manager.last_batch_send.get(symbol, None)

                        if (
                            last_batch_time is None
                            or (now - last_batch_time).total_seconds()
                            >= BATCH_SEND_INTERVAL
                        ):
                            batched_alerts = manager.get_batched_alerts(symbol)
                            manager.last_batch_send[symbol] = now

                            if symbol in batched_alerts and batched_alerts[symbol]:
                                logger.debug(
                                    f"Sending {len(batched_alerts[symbol])} batched alerts for {user_id} on {symbol}"
                                )

                                # Process batched alerts
                                modified_batched = []
                                for batch_alert in batched_alerts[symbol]:
                                    # Add the batched alert's interval if available
                                    batch_interval = batch_alert.get(
                                        "interval", interval
                                    )
                                    batch_message = batch_alert["message"]

                                    # Format: original_alert | interval
                                    modified_batch = (
                                        f"{batch_message} | {batch_interval}"
                                    )
                                    modified_batched.append(modified_batch)

                                    # Record in database
                                    alert_type = self._extract_alert_type(batch_message)
                                    db.record_alert(
                                        user_id,
                                        symbol,
                                        batch_interval,
                                        alert_type,
                                        batch_message,
                                    )

                                # Send batched alerts
                                if modified_batched:
                                    self._run_callback(user_id, modified_batched)
                    
                    user_time = time.time() - user_start
                    logger.debug(f"Processed user {user_id} in {user_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error checking alerts for {symbol} ({interval}): {e}")
                import traceback

                logger.error(traceback.format_exc())
        
        total_check_time = time.time() - check_start_time
        logger.info(f"Completed alert check for {symbol} ({interval}) in {total_check_time:.2f} seconds")

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
        
        # Check if we should use timeframe-specific filtering
        try:
            from bot.services.feature_flags import get_flag
            use_timeframe_filtering = get_flag("ENABLE_TIMEFRAME_FILTERING", True)
        except ImportError:
            # If feature flags not available, default to enabled
            use_timeframe_filtering = True
            
        logger.debug(f"Timeframe filtering {'enabled' if use_timeframe_filtering else 'disabled'} for {symbol} ({interval})")

        # Clear existing alerts for this user/symbol
        manager.clear_alerts(symbol)

        # Get enabled alert types from user settings
        enabled_alerts = settings.get("enabled_alerts", [])
        
        # Get the alerts to set up (either filtered or all enabled)
        if use_timeframe_filtering:
            # Define interval-appropriate indicators
            short_timeframe_indicators = ["rsi", "volume"]
            medium_timeframe_indicators = ["rsi", "volume", "macd", "ema", "bb", "adx"]
            long_timeframe_indicators = ["rsi", "macd", "ema", "bb", "volume", "adx", "pattern"]
            
            # Filter enabled alerts based on timeframe
            if interval in ["1m", "3m", "5m"]:
                appropriate_indicators = short_timeframe_indicators
                logger.info(f"Using short timeframe indicators for {symbol} ({interval})")
            elif interval in ["15m", "30m", "1h"]:
                appropriate_indicators = medium_timeframe_indicators
                logger.info(f"Using medium timeframe indicators for {symbol} ({interval})")
            else:  # 4h, 1d, etc.
                appropriate_indicators = long_timeframe_indicators
                logger.info(f"Using long timeframe indicators for {symbol} ({interval})")
            
            # Only keep enabled alerts that are appropriate for this timeframe
            filtered_alerts = [alert for alert in enabled_alerts if alert in appropriate_indicators]
            
            logger.info(f"Filtered alerts for {symbol} ({interval}): {filtered_alerts} (from user settings: {enabled_alerts})")
            
            alerts_to_setup = filtered_alerts
        else:
            # Use all enabled alerts without filtering
            alerts_to_setup = enabled_alerts
            logger.debug(f"Using all enabled alerts for {symbol} ({interval}): {enabled_alerts}")
        
        # Get cooldown from settings
        cooldown = settings.get("cooldown_minutes", 240)

        # Set up the filtered alerts
        if "rsi" in alerts_to_setup:
            manager.add_alert(
                RsiAlert(
                    symbol,
                    oversold=settings.get("rsi_oversold", 30),
                    overbought=settings.get("rsi_overbought", 70),
                    cooldown_minutes=cooldown,
                )
            )

        if "macd" in alerts_to_setup:
            manager.add_alert(MacdAlert(symbol, cooldown_minutes=cooldown))

        if "ema" in alerts_to_setup:
            manager.add_alert(
                EmaCrossAlert(
                    symbol,
                    short=settings.get("ema_short", 9),
                    long=settings.get("ema_long", 21),
                    cooldown_minutes=cooldown,
                )
            )

        if "bb" in alerts_to_setup:
            manager.add_alert(
                BollingerBandAlert(
                    symbol,
                    squeeze_threshold=settings.get("bb_squeeze_threshold", 0.05),
                    cooldown_minutes=cooldown,
                )
            )

        if "volume" in alerts_to_setup:
            manager.add_alert(
                VolumeSpikeAlert(
                    symbol,
                    threshold=settings.get("volume_threshold", 2.5),
                    cooldown_minutes=cooldown,
                )
            )

        if "adx" in alerts_to_setup:
            manager.add_alert(
                AdxAlert(
                    symbol,
                    threshold=settings.get("adx_threshold", 25),
                    cooldown_minutes=cooldown,
                )
            )

        if "pattern" in alerts_to_setup:
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

                # Remove scheduled job if exists
                job_id = f"check_{symbol.upper()}_{interval}"
                try:
                    self.scheduler.remove_job(job_id)
                    logger.debug(f"Removed scheduled job {job_id}")
                except JobLookupError:
                    # Job might not exist, that's okay
                    pass

                # Remove from each user's alert managers
                for user_id, managers in self.user_alert_managers.items():
                    if symbol_key in managers:
                        # Clean up the alert manager
                        managers[symbol_key].clear_alerts(symbol)
                        # Remove the manager entirely
                        del managers[symbol_key]
                        logger.debug(
                            f"Removed {symbol} ({interval}) alerts for user {user_id}"
                        )

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
            # Stop scheduler first
            if self.scheduler:
                logger.info("Stopping scheduler...")
                try:
                    self.scheduler.shutdown()
                except:
                    logger.exception("Error shutting down scheduler")

            # Stop batch aggregator if enabled
            if get_flag("ENABLE_BATCH_AGGREGATOR", True):
                try:
                    batch_aggregator = get_batch_aggregator()
                    batch_aggregator.stop()
                    logger.info("Stopped BatchAggregator service")
                except Exception as e:
                    logger.error(f"Error stopping BatchAggregator: {e}")

            # Set flags to indicate we're not running
            self.running = False
            self.scheduled_symbols.clear()
            self.symbols_by_frequency = {
                "high": set(),
                "medium": set(),
                "low": set(),
            }

            # Ensure any remaining database operations are completed
            try:
                db = get_db()
                # The DatabaseManager doesn't have a save_all method
                # Instead, any pending transactions should be auto-committed
                # via the standard SQLite transaction mechanisms
                # Just ensure the connection is properly handled
                logger.info("Database connections will be closed during shutdown")
            except Exception as e:
                logger.exception(f"Error handling database during shutdown: {e}")

            # Clear user alert managers
            self.user_alert_managers.clear()

            # If we started our own asyncio event loop, stop it
            if self.use_own_loop and self.loop and self.loop.is_running():
                logger.info("Stopping asyncio event loop...")
                try:
                    # Signal the loop to stop
                    asyncio.run_coroutine_threadsafe(
                        self._stop_loop(), self.loop
                    ).result(timeout=5)
                    # Wait for the thread to finish
                    if self.loop_thread and self.loop_thread.is_alive():
                        self.loop_thread.join(timeout=5)
                except:
                    logger.exception("Error stopping event loop")

            logger.info("Scheduler stopped")

    def process_all_batched_alerts(self):
        """Process all batched alerts across all alert managers"""
        # Early return if batch aggregation is disabled
        if not get_flag("ENABLE_BATCH_AGGREGATOR", False):
            logger.debug("Skipping batch processing (batch aggregation is disabled)")
            return
            
        logger.debug("Processing all batched alerts")

        try:
            # Use BatchAggregator service for processing
            batch_aggregator = get_batch_aggregator()

            # The batch_aggregator will handle the actual processing and callbacks
            # This method is mostly a scheduled trigger for processing batches
            if not get_flag("ENABLE_BATCH_AGGREGATOR", True):
                logger.debug("BatchAggregator service is disabled")
                return

            # Trigger batch processing in the service
            # Note: This is handled internally by the service's background task,
            # but we can manually trigger it here as well
            logger.debug("Triggered batch processing via BatchAggregator service")

            # Explicitly process batches
            batch_aggregator._process_all_batches()

        except Exception as e:
            logger.error(f"Error processing batched alerts: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def prune_expired_cooldowns(self):
        """Prune expired cooldowns to prevent memory leaks"""
        try:
            logger.info("Pruning expired cooldowns...")
            cooldown_service = get_cooldown_service()
            removed = cooldown_service.prune_expired_cooldowns(max_age_hours=24)
            logger.info(f"Pruned {removed} expired cooldowns older than 24 hours")
        except Exception as e:
            logger.error(f"Error pruning cooldowns: {e}")
            import traceback

            logger.error(traceback.format_exc())

    async def _stop_loop(self):
        """Coroutine to stop the event loop gracefully"""
        # Give pending tasks a chance to complete
        await asyncio.sleep(0.1)
        # Stop the loop
        self.loop.stop()


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
