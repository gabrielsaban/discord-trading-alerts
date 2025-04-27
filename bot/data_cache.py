import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Set up logging
logger = logging.getLogger("trading_alerts.data_cache")

# Frequency tiers for intervals
FREQUENCY_TIERS = {
    "high": ["1m", "3m", "5m"],
    "medium": ["15m", "30m", "1h"],
    "low": ["2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
}

# Reverse mapping from interval to tier
INTERVAL_TO_TIER = {}
for tier, intervals in FREQUENCY_TIERS.items():
    for interval in intervals:
        INTERVAL_TO_TIER[interval] = tier


class MarketDataCache:
    """A cache for storing and retrieving market data to reduce API calls"""

    def __init__(self, max_age_seconds: Dict[str, int] = None):
        """
        Initialize the cache

        Parameters:
        -----------
        max_age_seconds : Dict[str, int], optional
            Maximum age in seconds for cached data by interval (e.g., {'1m': 30, '1h': 300})
            If not provided, default values will be used
        """
        # Cache structure: {(symbol, interval): (timestamp, dataframe)}
        self.cache: Dict[Tuple[str, str], Tuple[datetime, pd.DataFrame]] = {}

        # Track last check time for each interval
        self.last_check_times: Dict[str, datetime] = {}

        # Default max age values (how long data is considered fresh)
        self.max_age_seconds = {
            "1m": 30,  # 30 seconds for 1-minute data
            "3m": 60,  # 1 minute for 3-minute data
            "5m": 90,  # 1.5 minutes for 5-minute data
            "15m": 180,  # 3 minutes for 15-minute data
            "30m": 300,  # 5 minutes for 30-minute data
            "1h": 600,  # 10 minutes for 1-hour data
            "2h": 900,  # 15 minutes for 2-hour data
            "4h": 1200,  # 20 minutes for 4-hour data
            "6h": 1500,  # 25 minutes for 6-hour data
            "8h": 1800,  # 30 minutes for 8-hour data
            "12h": 2400,  # 40 minutes for 12-hour data
            "1d": 3600,  # 1 hour for 1-day data
        }

        # Override defaults with provided values if any
        if max_age_seconds:
            self.max_age_seconds.update(max_age_seconds)

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cleanup = datetime.now()

        # Tracking by frequency tier
        self.tier_stats = {
            "high": {"hits": 0, "misses": 0, "last_fetch": None},
            "medium": {"hits": 0, "misses": 0, "last_fetch": None},
            "low": {"hits": 0, "misses": 0, "last_fetch": None},
        }

    def get(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get market data from cache if available and not stale

        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Timeframe interval (e.g., '15m', '1h')

        Returns:
        --------
        pd.DataFrame or None
            Cached dataframe if available and fresh, None otherwise
        """
        key = (symbol.upper(), interval)

        # Update last check time for this interval
        self.last_check_times[interval] = datetime.now()

        # Get the frequency tier for this interval
        tier = INTERVAL_TO_TIER.get(interval, "medium")

        # Check if data exists in cache
        if key in self.cache:
            timestamp, df = self.cache[key]

            # Calculate age of cached data
            age = (datetime.now() - timestamp).total_seconds()
            max_age = self.max_age_seconds.get(
                interval, 300
            )  # Default to 5 minutes if unknown

            # Return cached data if fresh
            if age < max_age:
                self.cache_hits += 1
                self.tier_stats[tier]["hits"] += 1
                logger.debug(f"Cache hit for {symbol} ({interval}), age: {age:.1f}s")
                return df.copy()  # Return a copy to prevent accidental modifications
            else:
                logger.debug(
                    f"Cache expired for {symbol} ({interval}), age: {age:.1f}s > {max_age}s"
                )

        self.cache_misses += 1
        self.tier_stats[tier]["misses"] += 1
        self.tier_stats[tier]["last_fetch"] = datetime.now()
        return None

    def put(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """
        Store market data in cache

        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Timeframe interval (e.g., '15m', '1h')
        df : pd.DataFrame
            Market data to cache
        """
        if df is None or df.empty:
            logger.warning(f"Attempted to cache empty data for {symbol} ({interval})")
            return

        key = (symbol.upper(), interval)
        self.cache[key] = (datetime.now(), df.copy())
        logger.debug(f"Cached data for {symbol} ({interval})")

        # Periodically clean up old entries
        if (datetime.now() - self.last_cleanup).total_seconds() > 3600:  # Once per hour
            self.cleanup()

    def cleanup(self) -> None:
        """Remove all stale entries from the cache"""
        before_count = len(self.cache)

        # Find keys to remove
        keys_to_remove = []
        now = datetime.now()

        for (symbol, interval), (timestamp, _) in self.cache.items():
            age = (now - timestamp).total_seconds()
            max_age = (
                self.max_age_seconds.get(interval, 300) * 5
            )  # Keep for 5x max_age before cleanup

            if age > max_age:
                keys_to_remove.append((symbol, interval))

        # Remove stale entries
        for key in keys_to_remove:
            del self.cache[key]

        self.last_cleanup = now
        after_count = len(self.cache)

        if before_count > after_count:
            logger.info(
                f"Cache cleanup: removed {before_count - after_count} stale entries"
            )

    def get_last_check_time(self, interval: str) -> Optional[datetime]:
        """
        Get the last time an interval was checked

        Parameters:
        -----------
        interval : str
            Timeframe interval (e.g., '15m', '1h')

        Returns:
        --------
        datetime or None
            The last time this interval was checked, or None if never checked
        """
        return self.last_check_times.get(interval)

    def get_symbols_by_tier(self, tier: str) -> List[str]:
        """
        Get all symbols in the cache for a specific frequency tier

        Parameters:
        -----------
        tier : str
            Frequency tier ('high', 'medium', 'low')

        Returns:
        --------
        List[str]
            List of symbols in the cache for the specified tier
        """
        symbols = set()
        intervals = FREQUENCY_TIERS.get(tier, [])

        for symbol, interval in self.cache.keys():
            if interval in intervals:
                symbols.add(symbol)

        return list(symbols)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = 0
        if (self.cache_hits + self.cache_misses) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)

        tier_hit_rates = {}
        for tier, stats in self.tier_stats.items():
            tier_total = stats["hits"] + stats["misses"]
            if tier_total > 0:
                tier_hit_rates[tier] = stats["hits"] / tier_total
            else:
                tier_hit_rates[tier] = 0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "tier_hit_rates": {t: f"{r:.2%}" for t, r in tier_hit_rates.items()},
            "memory_usage_mb": sum(
                df.memory_usage().sum() for _, (_, df) in self.cache.items()
            )
            / (1024 * 1024),
        }


# Singleton instance
_cache_instance = None


def get_cache() -> MarketDataCache:
    """Get the singleton cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MarketDataCache()
    return _cache_instance


def should_force_refresh(interval: str) -> bool:
    """
    Check if we should force refresh data for a given interval based on its frequency tier

    Parameters:
    -----------
    interval : str
        Timeframe interval (e.g., '15m', '1h')

    Returns:
    --------
    bool
        True if a force refresh is recommended
    """
    cache = get_cache()
    last_check_time = cache.get_last_check_time(interval)

    # If we've never checked this interval, we should refresh
    if last_check_time is None:
        return True

    # Get the frequency tier for this interval
    tier = INTERVAL_TO_TIER.get(interval, "medium")

    # Define how often to force refresh by tier (in seconds)
    force_refresh_seconds = {
        "high": 60,  # High frequency: force refresh every minute
        "medium": 300,  # Medium frequency: force refresh every 5 minutes
        "low": 900,  # Low frequency: force refresh every 15 minutes
    }

    # Get refresh period for this tier
    refresh_period = force_refresh_seconds.get(tier, 300)

    # Calculate time elapsed since last check
    elapsed = (datetime.now() - last_check_time).total_seconds()

    # Force refresh if enough time has passed
    return elapsed >= refresh_period
