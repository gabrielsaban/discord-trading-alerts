import time
import threading
import logging
import asyncio
from typing import Dict, List, Tuple, Callable, Any, Optional
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

# Import our modules
from bot.binance import fetch_market_data
from bot.alerts import AlertManager
from bot.db import get_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_alerts.scheduler')

# Global variables
DEFAULT_INTERVALS = {
    '1m': 60,       # Check every 60 seconds
    '3m': 180,      # Check every 3 minutes
    '5m': 300,      # Check every 5 minutes
    '15m': 900,     # Check every 15 minutes
    '30m': 1800,    # Check every 30 minutes
    '1h': 3600,     # Check every hour
    '2h': 7200,     # Check every 2 hours
    '4h': 14400,    # Check every 4 hours
    '6h': 21600,    # Check every 6 hours
    '8h': 28800,    # Check every 8 hours
    '12h': 43200,   # Check every 12 hours
    '1d': 86400,    # Check every day
}

# How many candles to fetch for each timeframe
DEFAULT_CANDLE_LIMITS = {
    '1m': 200,
    '3m': 200,
    '5m': 200,
    '15m': 200,
    '30m': 150,
    '1h': 150,
    '2h': 150,
    '4h': 150,
    '6h': 120,
    '8h': 120,
    '12h': 120,
    '1d': 100,
}


class AlertScheduler:
    """Scheduler for periodically checking trading alerts"""
    
    def __init__(self, alert_callback: Callable[[str, List[str]], None] = None):
        """
        Initialize the alert scheduler
        
        Parameters:
        -----------
        alert_callback : callable
            Function to call when alerts are triggered (user_id, list_of_alerts)
        """
        self.db = get_db()
        self.alert_managers: Dict[str, AlertManager] = {}  # One manager per symbol/interval
        self.scheduler = BackgroundScheduler()
        self.alert_callback = alert_callback
        self.running = False
        self.last_run: Dict[Tuple[str, str], datetime] = {}  # (symbol, interval) -> last run time
        self.lock = threading.RLock()  # For thread safety
    
    def initialize(self):
        """Initialize the scheduler and load watched symbols"""
        with self.lock:
            # Get all active symbols from the database
            symbols = self.db.get_all_active_symbols()
            
            if not symbols:
                logger.info("No symbols found in watchlists")
                return
            
            # Initialize alert managers for each symbol/interval
            for symbol, interval in symbols:
                self.ensure_alert_manager(symbol, interval)
            
            # Start the scheduler
            if not self.scheduler.running:
                self.scheduler.start()
                self.running = True
                logger.info("Alert scheduler started")
    
    def ensure_alert_manager(self, symbol: str, interval: str) -> AlertManager:
        """Get or create an alert manager for a symbol/interval pair"""
        key = f"{symbol.upper()}_{interval}"
        
        if key not in self.alert_managers:
            # Create a new alert manager
            manager = AlertManager()
            self.alert_managers[key] = manager
            
            # Schedule periodic checks for this symbol/interval
            self.schedule_symbol_check(symbol, interval)
            
            logger.info(f"Created alert manager for {symbol} ({interval})")
        
        return self.alert_managers[key]
    
    def schedule_symbol_check(self, symbol: str, interval: str):
        """Schedule periodic checks for a symbol/interval pair"""
        seconds = DEFAULT_INTERVALS.get(interval, 900)  # Default to 15m if unknown
        
        # Add some jitter to avoid all checks happening at once
        jitter = int(seconds * 0.1)  # 10% jitter
        if jitter > 0:
            seconds += int(time.time()) % jitter
        
        # Create job ID
        job_id = f"check_{symbol.upper()}_{interval}"
        
        # Schedule the job
        try:
            # Remove existing job if present
            try:
                self.scheduler.remove_job(job_id)
            except JobLookupError:
                pass
            
            # Schedule new job
            self.scheduler.add_job(
                self.check_symbol_alerts,
                'interval', 
                seconds=seconds,
                id=job_id,
                replace_existing=True,
                args=[symbol, interval]
            )
            
            logger.info(f"Scheduled {symbol} ({interval}) check every {seconds} seconds")
        except Exception as e:
            logger.error(f"Error scheduling {symbol} ({interval}) check: {e}")
    
    def check_symbol_alerts(self, symbol: str, interval: str):
        """
        Check alerts for a specific symbol and interval
        
        This is the main method that runs periodically for each symbol/interval
        """
        with self.lock:
            # Update last run time
            self.last_run[(symbol, interval)] = datetime.now()
            
            logger.info(f"Checking alerts for {symbol} ({interval})")
            
            try:
                # Fetch market data
                limit = DEFAULT_CANDLE_LIMITS.get(interval, 200)
                df = fetch_market_data(symbol=symbol, interval=interval, limit=limit)
                
                if df is None or df.empty:
                    logger.warning(f"Failed to fetch data for {symbol} ({interval})")
                    return
                
                # Get users watching this symbol
                users = self.db.get_users_watching_symbol(symbol, interval)
                if not users:
                    logger.info(f"No users watching {symbol} ({interval})")
                    return
                
                # Get the alert manager for this symbol/interval
                manager = self.ensure_alert_manager(symbol, interval)
                
                # Check alerts for each user
                for user_id in users:
                    # Get user settings
                    user = self.db.get_user(user_id)
                    if not user or not user.get('is_active', False):
                        continue
                    
                    # Set up alerts based on user preferences
                    self.setup_user_alerts(manager, user_id, symbol, interval, user['settings'])
                    
                    # Check for triggered alerts
                    alerts = manager.check_alerts(symbol, df)
                    
                    if alerts:
                        logger.info(f"Found {len(alerts)} alerts for {user_id} on {symbol} ({interval})")
                        
                        # Record alerts in the database
                        for alert in alerts:
                            alert_type = self._extract_alert_type(alert)
                            self.db.record_alert(user_id, symbol, interval, alert_type, alert)
                        
                        # Call the callback if provided
                        if self.alert_callback:
                            try:
                                self.alert_callback(user_id, alerts)
                            except Exception as e:
                                logger.error(f"Error in alert callback: {e}")
            
            except Exception as e:
                logger.error(f"Error checking alerts for {symbol} ({interval}): {e}")
    
    def setup_user_alerts(self, manager: AlertManager, user_id: str, symbol: str, interval: str, settings: Dict[str, Any]):
        """Set up alerts based on user preferences"""
        from bot.alerts import (
            RsiAlert, MacdAlert, EmaCrossAlert, BollingerBandAlert,
            VolumeSpikeAlert, AdxAlert, PatternAlert
        )
        
        # Clear existing alerts for this user/symbol
        manager.clear_alerts(symbol)
        
        # Get enabled alert types
        enabled_alerts = settings.get('enabled_alerts', [])
        cooldown = settings.get('cooldown_minutes', 240)
        
        # Add alerts based on user settings
        if 'rsi' in enabled_alerts:
            manager.add_alert(RsiAlert(
                symbol, 
                oversold=settings.get('rsi_oversold', 30),
                overbought=settings.get('rsi_overbought', 70),
                cooldown_minutes=cooldown
            ))
        
        if 'macd' in enabled_alerts:
            manager.add_alert(MacdAlert(
                symbol,
                cooldown_minutes=cooldown
            ))
        
        if 'ema' in enabled_alerts:
            manager.add_alert(EmaCrossAlert(
                symbol,
                short=settings.get('ema_short', 9),
                long=settings.get('ema_long', 21),
                cooldown_minutes=cooldown
            ))
        
        if 'bb' in enabled_alerts:
            manager.add_alert(BollingerBandAlert(
                symbol,
                squeeze_threshold=settings.get('bb_squeeze_threshold', 0.05),
                cooldown_minutes=cooldown
            ))
        
        if 'volume' in enabled_alerts:
            manager.add_alert(VolumeSpikeAlert(
                symbol,
                threshold=settings.get('volume_threshold', 2.5),
                cooldown_minutes=cooldown
            ))
        
        if 'adx' in enabled_alerts:
            manager.add_alert(AdxAlert(
                symbol,
                threshold=settings.get('adx_threshold', 25),
                cooldown_minutes=cooldown
            ))
        
        if 'pattern' in enabled_alerts:
            manager.add_alert(PatternAlert(
                symbol,
                cooldown_minutes=cooldown
            ))
    
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
        elif any(pattern in alert_message for pattern in ["HAMMER", "STAR", "ENGULFING", "PATTERN"]):
            return "pattern"
        else:
            return "other"
    
    def add_symbol(self, symbol: str, interval: str):
        """Add a new symbol to be monitored"""
        with self.lock:
            if self.running:
                # Ensure we have an alert manager for this symbol
                self.ensure_alert_manager(symbol, interval)
                logger.info(f"Added {symbol} ({interval}) to monitoring")
    
    def remove_symbol(self, symbol: str, interval: str):
        """Remove a symbol from monitoring"""
        with self.lock:
            key = f"{symbol.upper()}_{interval}"
            
            if key in self.alert_managers:
                # Remove the alert manager
                del self.alert_managers[key]
                
                # Remove the scheduled job
                job_id = f"check_{symbol.upper()}_{interval}"
                try:
                    self.scheduler.remove_job(job_id)
                except JobLookupError:
                    pass
                
                logger.info(f"Removed {symbol} ({interval}) from monitoring")
    
    def check_status(self) -> Dict[str, Any]:
        """Get the current status of the scheduler"""
        with self.lock:
            active_jobs = []
            for job in self.scheduler.get_jobs():
                next_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S')
                active_jobs.append({
                    'id': job.id,
                    'next_run': next_run
                })
            
            last_run_formatted = {}
            for (symbol, interval), run_time in self.last_run.items():
                last_run_formatted[f"{symbol}_{interval}"] = run_time.strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'running': self.running,
                'active_symbols': len(self.alert_managers),
                'jobs': active_jobs,
                'last_runs': last_run_formatted
            }
    
    def stop(self):
        """Stop the scheduler"""
        with self.lock:
            if self.running:
                self.scheduler.shutdown()
                self.running = False
                logger.info("Alert scheduler stopped")


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
    
    # Override the DB
    scheduler.db = test_db
    
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
