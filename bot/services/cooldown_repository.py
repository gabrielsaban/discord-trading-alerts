import copy
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("discord_trading_alerts.cooldown_repository")


class CooldownRepository:
    """
    Repository for persisting and retrieving cooldown data.
    Allows cooldown information to be stored to disk and loaded between application restarts.
    """

    # Singleton instance
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern implementation"""
        with cls._lock:
            if cls._instance is None:
                logger.debug("Creating new CooldownRepository instance")
                cls._instance = super(CooldownRepository, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the cooldown repository with default settings"""
        # Skip initialization if already done (part of singleton pattern)
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            # Thread-safe lock for operations
            self.lock = threading.RLock()

            # File path for persisting cooldowns
            self.file_path = os.path.join("data", "cooldowns.json")

            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # In-memory cache of cooldowns
            self.cooldowns = {}

            # Track if cooldowns have been modified since last save
            self.modified_since_save = False

            # Mark as initialized
            self._initialized = True

            # Load existing cooldowns from file if available
            self._load_from_file()

    def get_cooldown(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cooldown information for a specific key

        Parameters:
        -----------
        key : str
            Unique cooldown key

        Returns:
        --------
        dict or None
            Cooldown data if found, None otherwise
        """
        with self.lock:
            cooldown_data = self.cooldowns.get(key)
            if cooldown_data:
                # Return a deep copy to prevent modification of internal state
                return copy.deepcopy(cooldown_data)
            return None

    def set_cooldown(self, key: str, data: Dict[str, Any]) -> None:
        """
        Set or update cooldown information for a specific key

        Parameters:
        -----------
        key : str
            Unique cooldown key
        data : dict
            Cooldown data to store
        """
        with self.lock:
            # Ensure timestamp is a datetime object for internal use
            if "timestamp" in data and isinstance(data["timestamp"], str):
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    logger.error(
                        f"Invalid timestamp format in cooldown data: {data['timestamp']}"
                    )
                    # Use current time as fallback
                    data["timestamp"] = datetime.utcnow() + timedelta(hours=1)

            # Add or update the cooldown with a deep copy to prevent external modification
            self.cooldowns[key] = copy.deepcopy(data)
            self.modified_since_save = True

    def remove_cooldown(self, key: str) -> bool:
        """
        Remove a cooldown by key

        Parameters:
        -----------
        key : str
            Unique cooldown key

        Returns:
        --------
        bool
            True if removed, False if key not found
        """
        with self.lock:
            if key in self.cooldowns:
                del self.cooldowns[key]
                self.modified_since_save = True
                return True
            return False

    def get_all_cooldowns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all stored cooldowns

        Returns:
        --------
        dict
            Dictionary of all cooldowns
        """
        with self.lock:
            # Return a deep copy to prevent modification of internal state
            return {k: copy.deepcopy(v) for k, v in self.cooldowns.items()}

    def clear_all_cooldowns(self) -> None:
        """Clear all cooldowns"""
        with self.lock:
            self.cooldowns.clear()
            self.modified_since_save = True

    def save_to_file(self, force: bool = False) -> bool:
        """
        Save cooldowns to file

        Parameters:
        -----------
        force : bool
            If True, save even if no modifications since last save

        Returns:
        --------
        bool
            True if save was performed, False otherwise
        """
        with self.lock:
            # Skip if no changes and not forced
            if not force and not self.modified_since_save:
                logger.debug("No changes to cooldowns since last save, skipping")
                return False

            try:
                # Prepare data for serialization
                serializable_cooldowns = {}
                for key, data in self.cooldowns.items():
                    # Deep copy to avoid modifying original
                    cooldown_data = copy.deepcopy(data)

                    # Convert datetime to ISO format string for JSON serialization
                    if "timestamp" in cooldown_data and isinstance(
                        cooldown_data["timestamp"], datetime
                    ):
                        cooldown_data["timestamp"] = cooldown_data[
                            "timestamp"
                        ].isoformat()

                    serializable_cooldowns[key] = cooldown_data

                # Write to file with pretty formatting for readability
                with open(self.file_path, "w") as f:
                    json.dump(serializable_cooldowns, f, indent=2)

                logger.info(
                    f"Saved {len(self.cooldowns)} cooldowns to {self.file_path}"
                )
                self.modified_since_save = False
                return True

            except Exception as e:
                logger.error(f"Error saving cooldowns to file: {e}")
                return False

    def _load_from_file(self) -> bool:
        """
        Load cooldowns from file

        Returns:
        --------
        bool
            True if load was successful, False otherwise
        """
        with self.lock:
            if not os.path.exists(self.file_path):
                logger.info(
                    f"Cooldown file {self.file_path} not found, starting with empty state"
                )
                return False

            try:
                with open(self.file_path, "r") as f:
                    loaded_cooldowns = json.load(f)

                # Process loaded data
                for key, data in loaded_cooldowns.items():
                    # Convert timestamp strings to datetime objects
                    if "timestamp" in data and isinstance(data["timestamp"], str):
                        try:
                            data["timestamp"] = datetime.fromisoformat(
                                data["timestamp"]
                            )
                        except ValueError:
                            logger.warning(
                                f"Invalid timestamp format in loaded cooldown: {data['timestamp']}"
                            )
                            # Skip this entry or use current time as fallback
                            data["timestamp"] = datetime.utcnow() + timedelta(hours=1)

                    self.cooldowns[key] = data

                logger.info(
                    f"Loaded {len(self.cooldowns)} cooldowns from {self.file_path}"
                )
                self.modified_since_save = False
                return True

            except Exception as e:
                logger.error(f"Error loading cooldowns from file: {e}")
                return False

    def prune_expired_cooldowns(self, max_age_hours: int = 24) -> int:
        """
        Remove cooldowns older than the specified time

        Parameters:
        -----------
        max_age_hours : int
            Maximum age in hours for cooldown records

        Returns:
        --------
        int
            Number of cooldowns removed
        """
        with self.lock:
            now = datetime.utcnow() + timedelta(hours=1)  # Add 1 hour to match batch_aggregator
            cutoff_time = now - timedelta(hours=max_age_hours)
            keys_to_remove = []

            # Find expired cooldowns
            for key, data in self.cooldowns.items():
                timestamp = data.get("timestamp")
                if timestamp and timestamp < cutoff_time:
                    keys_to_remove.append(key)

            # Remove them
            for key in keys_to_remove:
                del self.cooldowns[key]

            # If any were removed, mark as modified
            if keys_to_remove:
                self.modified_since_save = True

            logger.info(
                f"Pruned {len(keys_to_remove)} expired cooldowns older than {max_age_hours} hours"
            )
            return len(keys_to_remove)

    def get_cooldowns_by_symbol(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all cooldowns for a specific symbol

        Parameters:
        -----------
        symbol : str
            Trading pair symbol

        Returns:
        --------
        dict
            Dictionary of cooldowns for the symbol
        """
        with self.lock:
            return {
                k: copy.deepcopy(v)
                for k, v in self.cooldowns.items()
                if k.startswith(f"{symbol}_")
            }

    def get_symbols_with_cooldowns(self) -> Set[str]:
        """
        Get a set of all symbols that have active cooldowns

        Returns:
        --------
        set
            Set of symbol names
        """
        with self.lock:
            symbols = set()
            for key in self.cooldowns.keys():
                # Extract symbol from key format "SYMBOL_TYPE" or "SYMBOL_SUBTYPE"
                parts = key.split("_", 1)
                if parts:
                    symbols.add(parts[0])
            return symbols


def get_cooldown_repository() -> CooldownRepository:
    """Get or create the singleton CooldownRepository instance"""
    return CooldownRepository()
