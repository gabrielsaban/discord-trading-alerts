import logging
import os
import threading
from typing import Any, Dict, Optional

# Setup logging
logger = logging.getLogger("discord_trading_alerts.services.feature_flags")
logger.setLevel(logging.DEBUG)

# Global feature flags dictionary with thread-safe access
_feature_flags: Dict[str, Any] = {}
_flags_lock = threading.RLock()

# Default flags
DEFAULT_FLAGS = {
    "ENABLE_COOLDOWN_SERVICE": True,  # Enable cooldown service
    "ENABLE_OVERRIDE_ENGINE": True,   # Enable override engine
    "ENABLE_BATCH_AGGREGATOR": False, # Disable batch aggregator
    "DEBUG_LOG_OVERRIDES": True,      # Enable debug logging for overrides
}


def _load_flags_from_env():
    """Load feature flags from environment variables"""
    with _flags_lock:
        for flag_name in DEFAULT_FLAGS.keys():
            env_value = os.getenv(flag_name)
            if env_value is not None:
                # Convert string to appropriate type based on default
                default_type = type(DEFAULT_FLAGS[flag_name])
                if default_type == bool:
                    # Handle boolean conversion from string
                    _feature_flags[flag_name] = env_value.lower() in (
                        "true",
                        "1",
                        "yes",
                        "y",
                        "on",
                    )
                elif default_type == int:
                    try:
                        _feature_flags[flag_name] = int(env_value)
                    except ValueError:
                        logger.error(
                            f"Invalid value for {flag_name}: {env_value}, using default"
                        )
                        _feature_flags[flag_name] = DEFAULT_FLAGS[flag_name]
                elif default_type == float:
                    try:
                        _feature_flags[flag_name] = float(env_value)
                    except ValueError:
                        logger.error(
                            f"Invalid value for {flag_name}: {env_value}, using default"
                        )
                        _feature_flags[flag_name] = DEFAULT_FLAGS[flag_name]
                else:
                    # String or other types
                    _feature_flags[flag_name] = env_value
            else:
                # Use default if not set in environment
                _feature_flags[flag_name] = DEFAULT_FLAGS[flag_name]


def _load_flags_from_db():
    """Load feature flags from database"""
    # This is a placeholder - in a real implementation this would
    # fetch flags from a database table. For this PR, we'll just
    # use environment variables.
    try:
        from bot.db import get_db

        db = get_db()

        # Example implementation (pseudo-code):
        # db_flags = db.get_feature_flags()
        # for flag_name, flag_value in db_flags.items():
        #     _feature_flags[flag_name] = flag_value

        # For now, just log that we attempted this
        logger.debug("No database implementation for feature flags yet")

    except ImportError:
        logger.debug("Could not import database module, skipping DB feature flags")


def _initialize_flags():
    """Initialize feature flags from all sources"""
    with _flags_lock:
        # First set defaults
        for flag_name, default_value in DEFAULT_FLAGS.items():
            _feature_flags[flag_name] = default_value

        # Then load from environment (overrides defaults)
        _load_flags_from_env()

        # Finally load from database (overrides environment)
        _load_flags_from_db()

        logger.info(f"Initialized feature flags: {_feature_flags}")


def get_flag(flag_name: str, default_value: Any = None) -> Any:
    """
    Get the value of a feature flag

    Parameters:
    -----------
    flag_name : str
        Name of the feature flag
    default_value : Any, optional
        Default value if flag is not found

    Returns:
    --------
    Any
        Value of the feature flag, or default if not found
    """
    # Initialize flags if not already done
    if not _feature_flags:
        reload_flags()

    with _flags_lock:
        return _feature_flags.get(
            flag_name, DEFAULT_FLAGS.get(flag_name, default_value)
        )


def set_flag(flag_name: str, value: Any) -> None:
    """
    Set the value of a feature flag

    Parameters:
    -----------
    flag_name : str
        Name of the feature flag
    value : Any
        Value to set
    """
    with _flags_lock:
        _feature_flags[flag_name] = value
        logger.info(f"Set feature flag {flag_name} = {value}")


def reload_flags() -> None:
    """Reload all feature flags from environment and database"""
    with _flags_lock:
        _initialize_flags()


# Initialize flags when module is loaded
_initialize_flags()


# For testing/development
if __name__ == "__main__":
    # Setup console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test getting and setting flags
    print(f"Initial ENABLE_COOLDOWN_SERVICE: {get_flag('ENABLE_COOLDOWN_SERVICE')}")

    # Set a flag
    set_flag("ENABLE_COOLDOWN_SERVICE", True)
    print(
        f"After setting, ENABLE_COOLDOWN_SERVICE: {get_flag('ENABLE_COOLDOWN_SERVICE')}"
    )

    # Reload flags
    reload_flags()
    print(
        f"After reload, ENABLE_COOLDOWN_SERVICE: {get_flag('ENABLE_COOLDOWN_SERVICE')}"
    )

    # Test with non-existent flag
    print(
        f"Non-existent flag with default: {get_flag('NON_EXISTENT', 'default_value')}"
    )
